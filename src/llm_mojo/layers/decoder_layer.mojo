"""One Qwen decoder layer: explicit preflight, workspace, and ordered enqueue.

Both existing sublayers retain their arithmetic. The attention output is the
MLP input; there is no intervening allocation, copy, or synchronization.
"""
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import InlineArray
from llm_mojo.layers.attention_sublayer import (
    AttentionWeights, AttentionCache, AttentionWorkspace,
    _validate_attention_sublayer, enqueue_attention_sublayer,
    enqueue_attention_sublayer_integrated,
)
from llm_mojo.kernels.residual_norm import enqueue_residual_norm
from llm_mojo.layers.mlp import MLPWeights, MLPWorkspace, _validate_mlp, enqueue_mlp_apple_gpu

# Decoder configurations used by the Qwen model. IDs stay numeric because retained
# evidence records them; docs/generation.md#workload-policy describes each route.
comptime DECODER_BASELINE = 0  # integrated FP32 attention; 8x16 MMA projections from 16 rows; MLP mapping 7
comptime DECODER_SPLIT8 = 2  # baseline with split-8 KV prefill attention
comptime DECODER_SPLIT8_TILED = 3  # split-8 attention with 16x16 QKV/Wo projection tiles
comptime DECODER_CONSISTENT = 20  # FP32 G32 attention and rowwise projections at every row count
comptime DECODER_CONSISTENT_MMA = 21  # G32 attention with 8x16 projections and MLP mapping 7 at every row count
comptime DECODER_CONSISTENT_REUSE4 = 22  # G32 attention; each weight reused across four rowwise reductions
comptime DECODER_FUSED_DECODE = 26  # one row: fused QKV/RoPE/cache and SiLU/multiply


def decoder_mappings(configuration: Int, rows: Int) raises -> SIMD[DType.int64, 4]:
    """Configuration ID -> (GQA mapping, projection mapping, MLP mapping, prefill splits).

    A registry of the configurations above, not a performance-based selector.
    """
    if rows < 1:
        raise Error("invalid decoder rows")
    if configuration == DECODER_FUSED_DECODE:
        if rows != 1:
            raise Error("fused QKV configuration requires one decode row")
        return SIMD[DType.int64, 4](0, 0, 0, 1)
    if configuration == DECODER_CONSISTENT:
        return SIMD[DType.int64, 4](5, 0, 0, 1)
    if configuration == DECODER_CONSISTENT_MMA:
        return SIMD[DType.int64, 4](5, 6, 7, 1)
    if configuration == DECODER_CONSISTENT_REUSE4:
        return SIMD[DType.int64, 4](5, 7, 19, 1)
    if (configuration == DECODER_BASELINE or configuration == DECODER_SPLIT8
            or configuration == DECODER_SPLIT8_TILED):
        var split8 = configuration != DECODER_BASELINE
        var tiled = configuration == DECODER_SPLIT8_TILED
        return SIMD[DType.int64, 4](Int64(4 if split8 else 0), Int64(5 if tiled else 0),
                                    Int64(0 if rows == 1 else 7), Int64(8 if split8 else 1))
    raise Error("unknown decoder configuration " + String(configuration))


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
    if (gqa_mapping != 0 and gqa_mapping != 4 and gqa_mapping != 5) or (projection_mapping != 0 and projection_mapping != 5 and projection_mapping != 6 and projection_mapping != 7):
        raise Error("decoder supports declared integrated attention mappings only")
    if projection_mapping >= 6 and gqa_mapping != 5:
        raise Error("policy projection mappings require consistent attention")
    if gqa_mapping == 5 and not ((projection_mapping == 0 and mlp_mapping == 0)
        or (projection_mapping == 6 and mlp_mapping == 7)
        or (projection_mapping >= 7 and mlp_mapping == projection_mapping + 12)):
        raise Error("consistent decoder requires a declared projection and MLP family")
    if not integrated and (gqa_mapping != 0 or projection_mapping != 0):
        raise Error("tiny decoder requires attention mappings zero")
    if mlp_mapping != 0 and mlp_mapping != 7 and mlp_mapping != 19:
        raise Error("decoder supports declared MLP mappings only")
    if mlp_mapping == 7 and r == 1 and projection_mapping < 6:
        raise Error("decoder single-row baseline requires MLP mapping zero")
    comptime assert x.rank == 2
    if Int(x.layout.stride[0]().product()) != h or Int(x.layout.stride[1]().product()) != 1:
        raise Error("decoder requires contiguous row-major input")
    _ = _validate_attention_sublayer(ctx, aw, cache, a, x,
                                    6 + gqa_mapping if integrated else 3,
                                    (2 if projection_mapping == 6 else projection_mapping - 2) if projection_mapping >= 6 else
                                    ((((3 if projection_mapping == 5 else 2) if r >= 16 else 1) if gqa_mapping != 5 else 0) if integrated else 0),
                                    projection_mapping - 4 if projection_mapping >= 7 else (1 if projection_mapping == 5 else 0))
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


def validate_decoder_configuration[XL: TensorLayout](
    ctx: DeviceContext, aw: AttentionWeights, cache: AttentionCache,
    mut attention: AttentionWorkspace, mw: MLPWeights, mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    gqa_mapping: Int, projection_mapping: Int, mlp_mapping: Int,
) raises:
    """Preflight one integrated layer call without enqueueing or changing state."""
    _decoder_preflight(ctx, aw, cache, attention, mw, mlp, x, True,
                       gqa_mapping, projection_mapping, mlp_mapping)


def enqueue_decoder_layer[XL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
    mut attention: AttentionWorkspace, mut mw: MLPWeights, mut mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    mlp_mapping: Int = 0, integrated: Bool = True,
    gqa_mapping: Int = 0, projection_mapping: Int = 0, fuse_qkv: Bool = False,
    fuse_activation: Bool = False,
    fuse_residual_norm: Bool = False, input_normalized: Bool = False,
) raises -> Int:
    """Return the actual attention route; final output is in mlp.output.

    Validate both sublayers before the first dispatch. All inputs and storage
    remain live on one stream until consumers finish. A failure after submission
    starts invalidates this execution; the caller must drain and reset the cache.
    Mapping selection is explicit. Tiny fixtures use integrated=False; the
    optimized attention path requires the Qwen dimensions.
    Residual/norm fusion is an internal composition route: attention.output and
    mlp.normalized are produced together, and the final MLP residual is deferred.
    Its caller must combine attention.output and mlp.down with the next norm
    before consuming mlp.output. input_normalized requires the current row's
    input normalization already stored in attention.normalized.
    """
    if (fuse_residual_norm or input_normalized) and (not fuse_qkv or not fuse_activation or aw.hidden != 896):
        raise Error("residual/norm fusion requires Qwen configuration 26")
    if input_normalized and not fuse_residual_norm:
        raise Error("precomputed normalization requires residual fusion")
    if fuse_qkv and not integrated:
        raise Error("fused QKV requires integrated attention")
    if fuse_activation and (Int(x.dim[0]()) != 1 or mlp_mapping != 0):
        raise Error("fused MLP activation requires one row and mapping zero")
    _decoder_preflight(ctx, aw, cache, attention, mw, mlp, x,
                       integrated, gqa_mapping, projection_mapping, mlp_mapping)
    var actual_route: Int
    if integrated:
        actual_route = enqueue_attention_sublayer_integrated(ctx, aw, cache, attention, x,
                                                           gqa_mapping, projection_mapping, fuse_qkv, input_normalized, fuse_residual_norm)
    else:
        actual_route = enqueue_attention_sublayer(ctx, aw, cache, attention, x, 3)
    if fuse_residual_norm:
        enqueue_residual_norm(ctx,x,TileTensor(attention.projected,row_major(1,896)),
            TileTensor(mw.norm,row_major(896)),TileTensor(attention.output,row_major(1,896)),
            TileTensor(mlp.normalized,row_major(1,896)))
    enqueue_mlp_apple_gpu(ctx, mw, mlp,
                         TileTensor(attention.output, row_major(Int(x.dim[0]()), aw.hidden)),
                         mlp_mapping, fuse_activation, fuse_residual_norm, fuse_residual_norm)
    return actual_route


def enqueue_decoder_layer_configuration[XL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
    mut attention: AttentionWorkspace, mut mw: MLPWeights, mut mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin], configuration: Int,
    fuse_residual_norm: Bool = False, input_normalized: Bool = False,
) raises -> Int:
    var mappings = decoder_mappings(configuration, Int(x.dim[0]()))
    # Configuration 26 fuses QKV/RoPE/cache append and SiLU/multiply.
    var fused = configuration == DECODER_FUSED_DECODE
    return enqueue_decoder_layer(ctx,aw,cache,attention,mw,mlp,x,
        Int(mappings[2]),True,Int(mappings[0]),Int(mappings[1]),fused,fused,
        fuse_residual_norm,input_normalized)
