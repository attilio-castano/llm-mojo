"""One Qwen decoder layer: explicit preflight, workspace, and ordered enqueue.

Both existing sublayers retain their arithmetic. The attention output is the
MLP input; there is no intervening allocation, copy, or synchronization.
The decode composition (configuration 26) runs one layer for S one-token
sequences with a fixed sequence of kernels whose per-row arithmetic does not
depend on S.
"""
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import Array
from llm_mojo.layers.attention_sublayer import (
    AttentionWeights, AttentionCache, AttentionWorkspace,
    _validate_sublayer, enqueue_attention_sublayer,
    enqueue_attention_sublayer_integrated, enqueue_attention_sublayer_integrated_paged,
    enqueue_fused_decode_qkv_paged,
    enqueue_step_qkv_paged,
)
from llm_mojo.kernels.attention_decode import enqueue_paged_attention_g32_apple_gpu
from llm_mojo.kernels.paged_kv import validate_paged_pool
from llm_mojo.kernels.linear import enqueue_linear_decode_rows_apple_gpu
from llm_mojo.kernels.residual_norm import enqueue_residual_norm
from llm_mojo.kernels.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.kernels.swiglu import enqueue_silu_multiply_apple_gpu
from llm_mojo.layers.mlp import MLPWeights, MLPWorkspace, _validate_mlp, enqueue_mlp_apple_gpu

# Decoder configurations used by the Qwen model. IDs stay numeric because retained
# evidence records them; docs/generation.md#workload-policy describes each route.
comptime DECODER_BASELINE = 0  # integrated FP32 attention; 8x16 MMA projections from 16 rows; MLP mapping 7
comptime DECODER_SPLIT8 = 2  # baseline with split-8 KV prefill attention
comptime DECODER_SPLIT8_TILED = 3  # split-8 attention with 16x16 QKV/Wo projection tiles
comptime DECODER_CONSISTENT = 20  # FP32 G32 attention and rowwise projections at every row count
comptime DECODER_CONSISTENT_MMA = 21  # G32 attention with 8x16 projections and MLP mapping 7 at every row count
comptime DECODER_CONSISTENT_REUSE4 = 22  # G32 attention; each weight reused across four rowwise reductions
comptime DECODER_FUSED_DECODE = 26  # decode composition: enqueue_decode_batch_layer, one row per sequence
comptime DECODER_MIXED = 27  # reference step: leading singletons and at most one multi-row tail


def decoder_mappings(configuration: Int, rows: Int) raises -> SIMD[DType.int64, 4]:
    """Configuration ID -> (GQA mapping, projection mapping, MLP mapping, prefill splits).

    A registry of the configurations above, not a performance-based selector.
    """
    if rows < 1:
        raise Error("invalid decoder rows")
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
    var k = aw.kv_heads * aw.head_dim
    _decoder_checks(ctx, aw, a, mw, m, x, integrated, gqa_mapping, projection_mapping, mlp_mapping,
                    cache.length, cache.capacity, cache.kv_heads, cache.head_dim,
                    _region(cache.key, cache.capacity * k), _region(cache.value, cache.capacity * k))


def _decoder_checks[XL: TensorLayout](
    ctx: DeviceContext, aw: AttentionWeights,
    mut a: AttentionWorkspace, mw: MLPWeights, m: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    integrated: Bool, gqa_mapping: Int, projection_mapping: Int, mlp_mapping: Int,
    past: Int, capacity: Int, kv_heads: Int, head_dim: Int,
    key_region: SIMD[DType.uint64, 2], value_region: SIMD[DType.uint64, 2],
) raises:
    """Checks for either KV storage, given its facts and writable regions.

    A pool is one region; its caller passes an empty second region.
    """
    comptime assert x.flat_rank == 2
    var r = Int(x.dim[0]())
    var h = aw.hidden
    var i = mw.intermediate
    if (aw.query_heads <= 0 or aw.kv_heads <= 0 or aw.head_dim <= 0
        or aw.head_dim % 2 != 0 or aw.query_heads % aw.kv_heads != 0
        or h != aw.query_heads * aw.head_dim or i <= 0
        or capacity < 1 or capacity > 4096
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
    _ = _validate_sublayer(ctx, aw, a, x,
                           6 + gqa_mapping if integrated else 3,
                           (2 if projection_mapping == 6 else projection_mapping - 2) if projection_mapping >= 6 else
                           ((((3 if projection_mapping == 5 else 2) if r >= 16 else 1) if gqa_mapping != 5 else 0) if integrated else 0),
                           projection_mapping - 4 if projection_mapping >= 7 else (1 if projection_mapping == 5 else 0),
                           past, capacity, kv_heads, head_dim)
    _validate_mlp(ctx, mw, m, TileTensor(a.output, row_major(r, h)), mlp_mapping)
    var n = a.max_rows
    var k = aw.kv_heads * aw.head_dim
    # Fixed stack storage. Entries 0..22 are writable; 23..33 are read-only.
    var regions = Array[SIMD[DType.uint64, 2], 34](uninitialized=True)
    regions[0] = key_region
    regions[1] = value_region
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
        actual_route = enqueue_attention_sublayer_integrated(ctx, aw, cache, attention, x,
                                                           gqa_mapping, projection_mapping)
    else:
        actual_route = enqueue_attention_sublayer(ctx, aw, cache, attention, x, 3)
    enqueue_mlp_apple_gpu(ctx, mw, mlp,
                         TileTensor(attention.output, row_major(Int(x.dim[0]()), aw.hidden)),
                         mlp_mapping)
    return actual_route


def enqueue_decoder_layer_configuration[XL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
    mut attention: AttentionWorkspace, mut mw: MLPWeights, mut mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin], configuration: Int,
) raises -> Int:
    var mappings = decoder_mappings(configuration, Int(x.dim[0]()))
    return enqueue_decoder_layer(ctx,aw,cache,attention,mw,mlp,x,
        Int(mappings[2]),True,Int(mappings[0]),Int(mappings[1]))


def validate_decoder_configuration_paged[XL: TensorLayout](
    ctx: DeviceContext, aw: AttentionWeights, storage: DeviceBuffer[DType.bfloat16], past: Int,
    mut attention: AttentionWorkspace, mw: MLPWeights, mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin], configuration: Int,
) raises:
    """Preflight one integrated layer call on a paged sequence of `past` cached rows, changing nothing.

    The pool is one writable region. The caller has checked the sequence's
    table against the pool and its written slots.
    """
    var mappings = decoder_mappings(configuration, Int(x.dim[0]()))
    _decoder_checks(ctx, aw, attention, mw, mlp, x, True,
                    Int(mappings[0]), Int(mappings[1]), Int(mappings[2]),
                    past, attention.capacity, aw.kv_heads, aw.head_dim,
                    _region(storage, len(storage)), SIMD[DType.uint64, 2](0, 0))


def enqueue_decoder_layer_configuration_paged[HEAD_MAJOR: Bool, XL: TensorLayout, TL: TensorLayout,
                                              SL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut storage: DeviceBuffer[DType.bfloat16],
    table: TileTensor[DType.int32, TL, MutAnyOrigin],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    past: Int, layer: Int, layers: Int, block_size: Int,
    mut attention: AttentionWorkspace, mut mw: MLPWeights, mut mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin], configuration: Int,
) raises -> Int:
    """enqueue_decoder_layer_configuration for one paged sequence; returns the attention route.

    The rows of x take positions past .. past + R - 1, which `positions` holds
    on the device, and `table` lists the sequence's blocks. The caller advances
    the sequence's length.
    """
    validate_decoder_configuration_paged(ctx, aw, storage, past, attention, mw, mlp, x, configuration)
    var mappings = decoder_mappings(configuration, Int(x.dim[0]()))
    var route = enqueue_attention_sublayer_integrated_paged[HEAD_MAJOR](ctx, aw, storage, table, positions,
        past, layer, layers, block_size, attention, x, Int(mappings[0]), Int(mappings[1]))
    enqueue_mlp_apple_gpu(ctx, mw, mlp,
                         TileTensor(attention.output, row_major(Int(x.dim[0]()), aw.hidden)),
                         Int(mappings[2]))
    return route


def _decode_batch_preflight[QUERY_HEADS: Int, KV_HEADS: Int, HEAD_DIM: Int, SL: TensorLayout, TL: TensorLayout](
    ctx: DeviceContext, aw: AttentionWeights, attention: AttentionWorkspace,
    mw: MLPWeights, mlp: MLPWorkspace, x: DeviceBuffer[DType.bfloat16],
    storage: DeviceBuffer[DType.bfloat16],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    tables: TileTensor[DType.int32, TL, MutAnyOrigin],
    layer: Int, layers: Int, block_size: Int, table_sequences: Int = 0,
) raises:
    comptime assert positions.flat_rank == 1 and tables.flat_rank == 2
    comptime HIDDEN = QUERY_HEADS * HEAD_DIM
    comptime WIDTH = KV_HEADS * HEAD_DIM
    var s = Int(positions.dim[0]())
    var i = mw.intermediate
    if ctx.api() != "metal":
        raise Error("decode composition requires Metal")
    if (aw.query_heads != QUERY_HEADS or aw.kv_heads != KV_HEADS or aw.head_dim != HEAD_DIM
            or aw.hidden != HIDDEN or mw.hidden != HIDDEN or attention.query_heads != QUERY_HEADS
            or attention.kv_heads != KV_HEADS or attention.head_dim != HEAD_DIM
            or mlp.hidden != HIDDEN or mlp.intermediate != i):
        raise Error("decode composition dimensions disagree with the layer")
    var sequences = s if table_sequences == 0 else table_sequences
    if s < 1 or Int(tables.dim[0]()) != sequences or s > attention.max_rows or s > mlp.max_rows:
        raise Error("invalid decode composition rows, layer or pool geometry")
    validate_paged_pool[KV_HEADS, HEAD_DIM](len(storage), layer, layers, block_size, Int(tables.dim[1]()))
    # Fixed stack storage. Entries 0..11 are writable; 12..22 are read-only.
    var regions = Array[SIMD[DType.uint64, 2], 23](uninitialized=True)
    regions[0] = _region(storage, len(storage))
    regions[1] = _region(attention.normalized, s * HIDDEN)
    regions[2] = _region(attention.packed, s * (HIDDEN + 2 * WIDTH))
    regions[3] = _region(attention.query, s * HIDDEN)
    regions[4] = _region(attention.attention, s * HIDDEN)
    regions[5] = _region(attention.projected, s * HIDDEN)
    regions[6] = _region(attention.output, s * HIDDEN)
    regions[7] = _region(mlp.normalized, s * HIDDEN)
    regions[8] = _region(mlp.gate, s * i)
    regions[9] = _region(mlp.up, s * i)
    regions[10] = _region(mlp.gated, s * i)
    regions[11] = _region(mlp.down, s * HIDDEN)
    regions[12] = _region(x, s * HIDDEN)
    regions[13] = _region(aw.norm, HIDDEN)
    regions[14] = _region(aw.qkv, (HIDDEN + 2 * WIDTH) * HIDDEN)
    regions[15] = _region(aw.bias, HIDDEN + 2 * WIDTH)
    regions[16] = _region(aw.output, HIDDEN * HIDDEN)
    regions[17] = _region(attention.cosine, attention.capacity * HEAD_DIM)
    regions[18] = _region(attention.sine, attention.capacity * HEAD_DIM)
    regions[19] = _region(mw.norm, HIDDEN)
    regions[20] = _region(mw.gate, i * HIDDEN)
    regions[21] = _region(mw.up, i * HIDDEN)
    regions[22] = _region(mw.down, HIDDEN * i)
    for left in range(12):
        for right in range(left + 1, 23):
            var l = regions[left]
            var z = regions[right]
            if l[0] < z[0] + z[1] and z[0] < l[0] + l[1]:
                raise Error("decode composition writable storage overlaps another live tensor")


def validate_decode_batch_layer[QUERY_HEADS: Int, KV_HEADS: Int, HEAD_DIM: Int, SL: TensorLayout, TL: TensorLayout](
    ctx: DeviceContext, aw: AttentionWeights, attention: AttentionWorkspace,
    mw: MLPWeights, mlp: MLPWorkspace, x: DeviceBuffer[DType.bfloat16],
    storage: DeviceBuffer[DType.bfloat16],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    tables: TileTensor[DType.int32, TL, MutAnyOrigin],
    layer: Int, layers: Int, block_size: Int,
) raises:
    """Preflight one decode-composition layer without enqueueing or changing state."""
    _decode_batch_preflight[QUERY_HEADS, KV_HEADS, HEAD_DIM](ctx, aw, attention, mw, mlp, x, storage,
                                                             positions, tables, layer, layers, block_size)


def enqueue_decode_batch_layer[QUERY_HEADS: Int, KV_HEADS: Int, HEAD_DIM: Int, PROJECTION: Int, HEAD_MAJOR: Bool,
                               SL: TensorLayout, TL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut attention: AttentionWorkspace,
    mut mw: MLPWeights, mut mlp: MLPWorkspace, mut x: DeviceBuffer[DType.bfloat16],
    mut storage: DeviceBuffer[DType.bfloat16],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    tables: TileTensor[DType.int32, TL, MutAnyOrigin],
    layer: Int, layers: Int, block_size: Int, input_normalized: Bool,
) raises -> Int:
    """One layer of the decode composition for S one-token sequences; returns its launch count.

    x holds the S input rows. Every row runs the single-row Fast decode kernels'
    arithmetic whatever S is: input RMSNorm (skipped when the caller stored it
    in attention.normalized), packed QKV with bias, fused RoPE and K/V append
    into the paged pool, decode attention, Wo, the residual with the MLP norm,
    gate, up, SiLU times up, and down. attention.output then holds the attention
    residual and mlp.down the MLP branch; the caller adds them with the next
    layer's or the final norm. PROJECTION is the projections' batched arrangement,
    and HEAD_MAJOR the pool's order within a block. tables is [S, blocks], one
    row per sequence. The caller checks positions and tables against the pool
    and advances each sequence's length.
    """
    _decode_batch_preflight[QUERY_HEADS, KV_HEADS, HEAD_DIM](ctx, aw, attention, mw, mlp, x, storage,
                                                             positions, tables, layer, layers, block_size)
    comptime HIDDEN = QUERY_HEADS * HEAD_DIM
    comptime WIDTH = KV_HEADS * HEAD_DIM
    comptime PACKED = HIDDEN + 2 * WIDTH
    var s = Int(positions.dim[0]())
    var i = mw.intermediate
    var launches = 9
    var normal = TileTensor(attention.normalized, row_major(s, HIDDEN))
    if not input_normalized:
        enqueue_rms_norm_apple_gpu(ctx, TileTensor(x, row_major(s, HIDDEN)), TileTensor(aw.norm, row_major(HIDDEN)), normal)
        launches += 1
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, normal, TileTensor(aw.qkv, row_major(PACKED, HIDDEN)),
        TileTensor(aw.bias, row_major(PACKED)), TileTensor(attention.packed, row_major(s, PACKED)))
    enqueue_fused_decode_qkv_paged[QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_MAJOR](ctx, attention, storage,
        positions, tables, layer, layers, block_size)
    enqueue_paged_attention_g32_apple_gpu[QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_MAJOR](ctx,
        TileTensor(attention.query, row_major(s, QUERY_HEADS, HEAD_DIM)),
        TileTensor(storage, row_major(len(storage))),
        TileTensor(attention.attention, row_major(s, QUERY_HEADS, HEAD_DIM)),
        positions, tables, 1, layer, layers, block_size)
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, TileTensor(attention.attention, row_major(s, HIDDEN)),
        TileTensor(aw.output, row_major(HIDDEN, HIDDEN)), TileTensor(attention.projected, row_major(s, HIDDEN)))
    enqueue_residual_norm[HIDDEN](ctx, TileTensor(x, row_major(s, HIDDEN)),
        TileTensor(attention.projected, row_major(s, HIDDEN)), TileTensor(mw.norm, row_major(HIDDEN)),
        TileTensor(attention.output, row_major(s, HIDDEN)), TileTensor(mlp.normalized, row_major(s, HIDDEN)))
    var mlp_normal = TileTensor(mlp.normalized, row_major(s, HIDDEN))
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, mlp_normal, TileTensor(mw.gate, row_major(i, HIDDEN)),
        TileTensor(mlp.gate, row_major(s, i)))
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, mlp_normal, TileTensor(mw.up, row_major(i, HIDDEN)),
        TileTensor(mlp.up, row_major(s, i)))
    enqueue_silu_multiply_apple_gpu(ctx, TileTensor(mlp.gate, row_major(s, i)), TileTensor(mlp.up, row_major(s, i)),
        TileTensor(mlp.gated, row_major(s, i)))
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, TileTensor(mlp.gated, row_major(s, i)),
        TileTensor(mw.down, row_major(HIDDEN, i)), TileTensor(mlp.down, row_major(s, HIDDEN)))
    return launches


def validate_mixed_layer[
    QUERY_HEADS: Int, KV_HEADS: Int, HEAD_DIM: Int,
    SL: TensorLayout, WL: TensorLayout, TL: TensorLayout,
](
    ctx: DeviceContext, aw: AttentionWeights, attention: AttentionWorkspace,
    mw: MLPWeights, mlp: MLPWorkspace, x: DeviceBuffer[DType.bfloat16],
    storage: DeviceBuffer[DType.bfloat16],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    slot_mapping: TileTensor[DType.int32, WL, MutAnyOrigin],
    tables: TileTensor[DType.int32, TL, MutAnyOrigin],
    decode_count: Int, layer: Int, layers: Int, block_size: Int,
) raises:
    """Preflight N rows: D leading singletons and at most one multi-row tail.

    D describes row shape, including a one-token prefill remainder. The caller
    validates every position, physical slot and table entry against StepBatch
    and the pool's written counts before this geometry-only check.
    """
    comptime assert positions.flat_rank == 1 and slot_mapping.flat_rank == 1 and tables.flat_rank == 2
    var rows = Int(positions.dim[0]())
    var sequences = Int(tables.dim[0]())
    var tail = rows - decode_count
    if (rows < 1 or decode_count < 0 or decode_count > rows or tail == 1
            or sequences != decode_count + (1 if tail > 0 else 0)
            or Int(slot_mapping.dim[0]()) != rows):
        raise Error("a mixed layer needs leading singleton sequences and at most one multi-row tail")
    var width = Int(tables.dim[1]())
    if (Int(positions.layout.stride[0]().product()) != 1
            or Int(slot_mapping.layout.stride[0]().product()) != 1
            or Int(tables.layout.stride[0]().product()) != width
            or Int(tables.layout.stride[1]().product()) != 1):
        raise Error("mixed step metadata must be contiguous")
    _decode_batch_preflight[QUERY_HEADS, KV_HEADS, HEAD_DIM](ctx, aw, attention, mw, mlp, x, storage,
        positions, tables, layer, layers, block_size, sequences)


def enqueue_mixed_layer[
    QUERY_HEADS: Int, KV_HEADS: Int, HEAD_DIM: Int, PROJECTION: Int, HEAD_MAJOR: Bool,
    SL: TensorLayout, WL: TensorLayout, TL: TensorLayout,
](
    ctx: DeviceContext, mut aw: AttentionWeights, mut attention: AttentionWorkspace,
    mut mw: MLPWeights, mut mlp: MLPWorkspace, mut x: DeviceBuffer[DType.bfloat16],
    mut storage: DeviceBuffer[DType.bfloat16],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    slot_mapping: TileTensor[DType.int32, WL, MutAnyOrigin],
    tables: TileTensor[DType.int32, TL, MutAnyOrigin],
    decode_count: Int, layer: Int, layers: Int, block_size: Int, input_normalized: Bool,
) raises -> Int:
    """Reference mixed step with shared token operations and G32 attention.

    Every N-row projection uses PROJECTION. QKV postprocessing scatters every
    row through slot_mapping. Attention runs once for D singleton rows and once
    for the optional P-row tail, each with the existing G32 arithmetic. The
    caller adds mlp.down to attention.output with the next or final norm, just
    as for the decode composition. No allocation, upload or synchronization.
    """
    validate_mixed_layer[QUERY_HEADS, KV_HEADS, HEAD_DIM](ctx, aw, attention, mw, mlp, x, storage,
        positions, slot_mapping, tables, decode_count, layer, layers, block_size)
    comptime HIDDEN = QUERY_HEADS * HEAD_DIM
    comptime WIDTH = KV_HEADS * HEAD_DIM
    comptime PACKED = HIDDEN + 2 * WIDTH
    var rows = Int(positions.dim[0]())
    var tail = rows - decode_count
    var width = Int(tables.dim[1]())
    var i = mw.intermediate
    var launches = 8
    var normal = TileTensor(attention.normalized, row_major(rows, HIDDEN))
    if not input_normalized:
        enqueue_rms_norm_apple_gpu(ctx, TileTensor(x, row_major(rows, HIDDEN)), TileTensor(aw.norm, row_major(HIDDEN)), normal)
        launches += 1
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, normal, TileTensor(aw.qkv, row_major(PACKED, HIDDEN)),
        TileTensor(aw.bias, row_major(PACKED)), TileTensor(attention.packed, row_major(rows, PACKED)))
    enqueue_step_qkv_paged[QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_MAJOR](ctx, attention, storage,
        positions, slot_mapping, layer, layers, block_size)
    var pool = TileTensor(storage, row_major(len(storage)))
    if decode_count > 0:
        enqueue_paged_attention_g32_apple_gpu[QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_MAJOR](ctx,
            TileTensor(attention.query, row_major(decode_count, QUERY_HEADS, HEAD_DIM)), pool,
            TileTensor(attention.attention, row_major(decode_count, QUERY_HEADS, HEAD_DIM)),
            TileTensor(positions.ptr, row_major(decode_count)), TileTensor(tables.ptr, row_major(decode_count, width)),
            1, layer, layers, block_size)
        launches += 1
    if tail > 0:
        enqueue_paged_attention_g32_apple_gpu[QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_MAJOR](ctx,
            TileTensor(attention.query.unsafe_ptr().unsafe_offset(decode_count * HIDDEN), row_major(tail, QUERY_HEADS, HEAD_DIM)),
            pool,
            TileTensor(attention.attention.unsafe_ptr().unsafe_offset(decode_count * HIDDEN), row_major(tail, QUERY_HEADS, HEAD_DIM)),
            TileTensor(positions.ptr.unsafe_offset(decode_count), row_major(tail)),
            TileTensor(tables.ptr.unsafe_offset(decode_count * width), row_major(1, width)),
            tail, layer, layers, block_size)
        launches += 1
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, TileTensor(attention.attention, row_major(rows, HIDDEN)),
        TileTensor(aw.output, row_major(HIDDEN, HIDDEN)), TileTensor(attention.projected, row_major(rows, HIDDEN)))
    enqueue_residual_norm[HIDDEN](ctx, TileTensor(x, row_major(rows, HIDDEN)),
        TileTensor(attention.projected, row_major(rows, HIDDEN)), TileTensor(mw.norm, row_major(HIDDEN)),
        TileTensor(attention.output, row_major(rows, HIDDEN)), TileTensor(mlp.normalized, row_major(rows, HIDDEN)))
    var mlp_normal = TileTensor(mlp.normalized, row_major(rows, HIDDEN))
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, mlp_normal, TileTensor(mw.gate, row_major(i, HIDDEN)),
        TileTensor(mlp.gate, row_major(rows, i)))
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, mlp_normal, TileTensor(mw.up, row_major(i, HIDDEN)),
        TileTensor(mlp.up, row_major(rows, i)))
    enqueue_silu_multiply_apple_gpu(ctx, TileTensor(mlp.gate, row_major(rows, i)), TileTensor(mlp.up, row_major(rows, i)),
        TileTensor(mlp.gated, row_major(rows, i)))
    enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx, TileTensor(mlp.gated, row_major(rows, i)),
        TileTensor(mw.down, row_major(HIDDEN, i)), TileTensor(mlp.down, row_major(rows, HIDDEN)))
    return launches
