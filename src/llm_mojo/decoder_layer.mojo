"""One Qwen decoder layer: explicit preflight, workspace, and ordered enqueue.

Both existing sublayers retain their arithmetic. The attention output is the
MLP input; there is no intervening allocation, copy, or synchronization.
"""
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import InlineArray
from llm_mojo.attention_sublayer import (
    AttentionWeights, AttentionCache, AttentionWorkspace,
    _validate_attention_sublayer, enqueue_attention_sublayer,
    enqueue_attention_sublayer_integrated,
)
from llm_mojo.mlp import MLPWeights, MLPWorkspace, _validate_mlp, enqueue_mlp_apple_gpu


def _region[dtype: DType](
    buffer: DeviceBuffer[dtype], required: Int, element_bytes: Int = 2,
) raises -> SIMD[DType.uint64, 2]:
    if required < 1 or len(buffer) < required:
        raise Error("decoder buffer is smaller than its declared dimensions")
    return SIMD[DType.uint64, 2](UInt64(Int(buffer.unsafe_ptr())), UInt64(len(buffer) * element_bytes))


def _decoder_preflight[XL: TensorLayout](
    ctx: DeviceContext, aw: AttentionWeights, cache: AttentionCache,
    mut a: AttentionWorkspace, mw: MLPWeights, m: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    integrated: Bool, gqa_mapping: Int, projection_mapping: Int, mlp_mapping: Int,
) raises:
    comptime assert x.flat_rank == 2
    var r = Int(x.dim[0]())
    var h = aw.hidden
    var i = mw.intermediate
    if (aw.query_heads <= 0 or aw.kv_heads <= 0 or aw.head_dim <= 0
        or aw.head_dim % 2 != 0 or aw.query_heads % aw.kv_heads != 0
        or h != aw.query_heads * aw.head_dim or i <= 0
        or cache.capacity < 1 or cache.capacity > 4096
        or a.capacity < 1 or a.capacity > 4096):
        raise Error("decoder geometry or capacity is invalid")
    if gqa_mapping != 0 or projection_mapping != 0:
        raise Error("decoder baseline requires integrated attention mappings zero")
    if mlp_mapping != 0 and mlp_mapping != 7:
        raise Error("decoder baseline supports MLP mappings zero and seven")
    if mlp_mapping == 7 and r == 1:
        raise Error("decoder single-row baseline requires MLP mapping zero")
    comptime assert x.rank == 2
    if Int(x.layout.stride[0]().product()) != h or Int(x.layout.stride[1]().product()) != 1:
        raise Error("decoder requires contiguous row-major input")
    _ = _validate_attention_sublayer(ctx, aw, cache, a, x,
                                    6 if integrated else 3,
                                    (2 if r >= 16 else 1) if integrated else 0)
    _validate_mlp(ctx, mw, m, TileTensor(a.output, row_major(r, h)), mlp_mapping)
    var n = a.max_rows
    var k = aw.kv_heads * aw.head_dim
    var c = cache.capacity
    # Fixed stack storage. Entries 0..22 are writable; 23..33 are read-only.
    var regions = InlineArray[SIMD[DType.uint64, 2], 34](uninitialized=True)
    regions[0] = _region(cache.key, c * k)
    regions[1] = _region(cache.value, c * k)
    regions[2] = _region(a.normalized, n * h)
    regions[3] = _region(a.raw_query, n * h)
    regions[4] = _region(a.raw_key, n * k)
    regions[5] = _region(a.raw_value, n * k)
    regions[6] = _region(a.query, n * h)
    regions[7] = _region(a.rotated_key, n * k)
    regions[8] = _region(a.attention, n * h)
    regions[9] = _region(a.projected, n * h)
    regions[10] = _region(a.output, n * h)
    regions[11] = _region(a.packed, n * (h + 2 * k))
    regions[12] = _region(a.scratch, n * aw.query_heads * a.capacity if a.materialized else 1)
    regions[13] = _region(a.fp32_scratch, n * aw.query_heads * a.capacity if a.fp32_materialized else 1, 4)
    regions[14] = _region(a.split, 14 * 64 * 66, 4)
    regions[15] = _region(a.prefill_partial, n * aw.query_heads * a.prefill_splits * 66 if a.prefill_splits > 1 else 1, 4)
    regions[16] = _region(m.normalized, m.max_rows * h)
    regions[17] = _region(m.gate, m.max_rows * i)
    regions[18] = _region(m.up, m.max_rows * i)
    regions[19] = _region(m.activated, m.max_rows * i)
    regions[20] = _region(m.gated, m.max_rows * i)
    regions[21] = _region(m.down, m.max_rows * h)
    regions[22] = _region(m.output, m.max_rows * h)
    regions[23] = _region(aw.norm, h)
    regions[24] = _region(aw.qkv, (h + 2 * k) * h)
    regions[25] = _region(aw.bias, h + 2 * k)
    regions[26] = _region(aw.output, h * h)
    regions[27] = _region(a.cosine, a.capacity * aw.head_dim)
    regions[28] = _region(a.sine, a.capacity * aw.head_dim)
    regions[29] = _region(mw.norm, h)
    regions[30] = _region(mw.gate, i * h)
    regions[31] = _region(mw.up, i * h)
    regions[32] = _region(mw.down, h * i)
    regions[33] = SIMD[DType.uint64, 2](UInt64(Int(x.ptr)), UInt64(r * h * 2))
    for left in range(23):
        for right in range(left + 1, 34):
            var l = regions[left]
            var z = regions[right]
            if l[0] < z[0] + z[1] and z[0] < l[0] + l[1]:
                raise Error("decoder writable storage overlaps another live tensor")


def enqueue_decoder_layer[XL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
    mut attention: AttentionWorkspace, mut mw: MLPWeights, mut mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    mlp_mapping: Int = 0, integrated: Bool = True,
    gqa_mapping: Int = 0, projection_mapping: Int = 0,
) raises -> Int:
    """Return the actual attention route; final output is in mlp.output.

    Validate both sublayers before the first dispatch. All inputs and storage
    remain live on one stream until consumers finish. A failure after submission
    starts invalidates this execution; the caller must drain and reset the cache.
    Mapping selection is explicit. Tiny fixtures use integrated=False; the
    optimized attention path requires the Qwen dimensions.
    """
    _decoder_preflight(ctx, aw, cache, attention, mw, mlp, x,
                       integrated, gqa_mapping, projection_mapping, mlp_mapping)
    var actual_route: Int
    if integrated:
        actual_route = enqueue_attention_sublayer_integrated(ctx, aw, cache, attention, x)
    else:
        actual_route = enqueue_attention_sublayer(ctx, aw, cache, attention, x, 3)
    enqueue_mlp_apple_gpu(ctx, mw, mlp,
                         TileTensor(attention.output, row_major(Int(x.dim[0]()), aw.hidden)),
                         mlp_mapping)
    return actual_route
