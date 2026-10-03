"""One inspectable Qwen attention sublayer; allocation is separate from enqueue.

See docs/attention-sublayer.md for rounding, lifetime and cache contracts.
The caller initializes weights and rotary tables, retains all objects, and uses
one context/stream through completion. Rotary tables are explicit model inputs.
The K/V writes also have paged forms that address a block pool through a
table (kernels/paged_kv.mojo).
"""
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, global_idx
from std.math import ceildiv
from llm_mojo.kernels.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.kernels.linear import (
    enqueue_linear_apple_gpu,
    enqueue_linear_rowwise_rows_apple_gpu,
    enqueue_linear_prefill_mma_8x16_apple_gpu,
    enqueue_linear_prefill_mma_tile_apple_gpu,
)
from llm_mojo.kernels.rope import enqueue_rope_apple_gpu
from std.builtin.simd import FastMathFlag
from std.sys.info import is_apple_gpu
from llm_mojo.kernels.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.kernels.attention_decode import (
    enqueue_grouped_query_attention_decode_apple_gpu,
    enqueue_grouped_query_attention_consistent_apple_gpu,
    enqueue_paged_attention_g32_apple_gpu,
)
from llm_mojo.kernels.attention_prefill import (
    enqueue_grouped_query_attention_prefill_apple_gpu,
    enqueue_grouped_query_attention_prefill_split_apple_gpu,
    enqueue_paged_attention_prefill_apple_gpu,
    enqueue_paged_attention_prefill_split_apple_gpu,
)
from llm_mojo.kernels.residual import enqueue_residual_apple_gpu
from llm_mojo.kernels.paged_kv import kv_row, validate_paged_pool


struct AttentionWeights(Movable):
    """Q/K/V source-compatible regions in one weight allocation, plus Wo/norm.
    """

    var query_heads: Int
    var kv_heads: Int
    var head_dim: Int
    var hidden: Int
    var qkv: DeviceBuffer[DType.bfloat16]
    var bias: DeviceBuffer[DType.bfloat16]
    var output: DeviceBuffer[DType.bfloat16]
    var norm: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        query_heads: Int = 14,
        kv_heads: Int = 2,
        head_dim: Int = 64,
    ) raises:
        if (
            query_heads <= 0
            or kv_heads <= 0
            or query_heads % kv_heads != 0
            or head_dim <= 0
            or head_dim % 2 != 0
        ):
            raise Error("invalid attention head dimensions")
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.hidden = query_heads * head_dim
        var n = self.hidden + 2 * kv_heads * head_dim
        self.qkv = ctx.enqueue_create_buffer[DType.bfloat16](n * self.hidden)
        self.bias = ctx.enqueue_create_buffer[DType.bfloat16](n)
        self.output = ctx.enqueue_create_buffer[DType.bfloat16](
            self.hidden * self.hidden
        )
        self.norm = ctx.enqueue_create_buffer[DType.bfloat16](self.hidden)


struct AttentionCache(Movable):
    """Fixed storage; length counts the prefix whose writes have been enqueued.
    """

    var capacity: Int
    var length: Int
    var kv_heads: Int
    var head_dim: Int
    var key: DeviceBuffer[DType.bfloat16]
    var value: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        capacity: Int,
        kv_heads: Int = 2,
        head_dim: Int = 64,
    ) raises:
        if capacity < 1 or capacity > 4096 or kv_heads <= 0 or head_dim <= 0:
            raise Error("invalid cache capacity or shape")
        self.capacity = capacity
        self.length = 0
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.key = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * kv_heads * head_dim
        )
        self.value = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * kv_heads * head_dim
        )

    def __init__(
        out self,
        var key: DeviceBuffer[DType.bfloat16],
        var value: DeviceBuffer[DType.bfloat16],
        capacity: Int,
        kv_heads: Int = 2,
        head_dim: Int = 64,
    ) raises:
        """Adopt caller-owned storage, such as views into a shared KV pool."""
        if capacity < 1 or capacity > 4096 or kv_heads <= 0 or head_dim <= 0:
            raise Error("invalid cache capacity or shape")
        if len(key) != capacity * kv_heads * head_dim or len(value) != len(key):
            raise Error("cache storage does not match its shape")
        self.capacity = capacity
        self.length = 0
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.key = key^
        self.value = value^

    def reset(mut self, ctx: DeviceContext) raises:
        """Finish pending use before making all stored entries logically absent.
        """
        ctx.synchronize()
        self.length = 0


struct AttentionWorkspace(Movable):
    """Reusable intermediates and caller-initialized BF16 rotary tables.

    The caller fills cosine/sine[capacity,head_dim] before enqueue, like weights.
    Table generation is outside this operation and its timing boundary. The
    compatibility baseline uses tables from the pinned upstream implementation.
    """

    var max_rows: Int
    var capacity: Int
    var query_heads: Int
    var kv_heads: Int
    var head_dim: Int
    var materialized: Bool
    var normalized: DeviceBuffer[DType.bfloat16]
    var raw_query: DeviceBuffer[DType.bfloat16]
    var raw_key: DeviceBuffer[DType.bfloat16]
    var raw_value: DeviceBuffer[DType.bfloat16]
    var query: DeviceBuffer[DType.bfloat16]
    var rotated_key: DeviceBuffer[DType.bfloat16]
    var attention: DeviceBuffer[DType.bfloat16]
    var projected: DeviceBuffer[DType.bfloat16]
    var output: DeviceBuffer[DType.bfloat16]
    var packed: DeviceBuffer[DType.bfloat16]
    var scratch: DeviceBuffer[DType.bfloat16]
    var fp32_scratch: DeviceBuffer[DType.float32]
    var fp32_materialized: Bool
    var split: DeviceBuffer[DType.float32]
    var prefill_partial: DeviceBuffer[DType.float32]
    var prefill_splits: Int
    var cosine: DeviceBuffer[DType.bfloat16]
    var sine: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        max_rows: Int,
        capacity: Int,
        query_heads: Int = 14,
        kv_heads: Int = 2,
        head_dim: Int = 64,
        materialized: Bool = False,
        fp32_materialized: Bool = True,
        prefill_splits: Int = 1,
    ) raises:
        if (
            max_rows < 1
            or max_rows > capacity
            or capacity > 4096
            or query_heads <= 0
            or kv_heads <= 0
            or query_heads % kv_heads != 0
            or head_dim <= 0
            or head_dim % 2 != 0
            or (prefill_splits != 1 and prefill_splits != 4 and prefill_splits != 8)
        ):
            raise Error("invalid workspace shape")
        self.max_rows = max_rows
        self.capacity = capacity
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.materialized = materialized
        self.fp32_materialized = fp32_materialized
        self.prefill_splits = prefill_splits
        var h = query_heads * head_dim
        var k = kv_heads * head_dim
        self.normalized = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * h
        )
        self.raw_query = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.raw_key = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * k)
        self.raw_value = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * k)
        self.query = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.rotated_key = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * k
        )
        self.attention = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.projected = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.output = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.packed = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * (h + 2 * k)
        )
        self.scratch = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * query_heads * capacity if materialized else 1
        )
        self.fp32_scratch = ctx.enqueue_create_buffer[DType.float32](
            max_rows * query_heads * capacity if fp32_materialized else 1
        )
        self.split = ctx.enqueue_create_buffer[DType.float32](14 * 64 * 66)
        self.prefill_partial = ctx.enqueue_create_buffer[DType.float32](
            max_rows * query_heads * prefill_splits * 66 if prefill_splits > 1 else 1
        )
        self.cosine = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * head_dim
        )
        self.sine = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * head_dim
        )


def _append[
    KL: TensorLayout, VL: TensorLayout, CL: TensorLayout
](
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    cache_key: TileTensor[DType.bfloat16, CL, MutAnyOrigin],
    cache_value: TileTensor[DType.bfloat16, CL, MutAnyOrigin],
    rows: Int32,
    width: Int32,
    past: Int32,
):
    comptime assert (
        key.flat_rank == 2
        and value.flat_rank == 2
        and cache_key.flat_rank == 2
        and cache_value.flat_rank == 2
    )
    var i = global_idx.x
    if i < Int(rows) * Int(width):
        var r = i // Int(width)
        var d = i % Int(width)
        cache_key[Int(past) + r, d] = key[r, d]
        cache_value[Int(past) + r, d] = value[r, d]


def _unpack_qkv[PL: TensorLayout, QL: TensorLayout, KL: TensorLayout](
    packed: TileTensor[DType.bfloat16, PL, MutAnyOrigin],
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    rows: Int32, hidden: Int32, kv_width: Int32,
):
    """Copy BF16 bits from per-token [Q | K | V] to contiguous consumers."""
    comptime assert packed.flat_rank == 2 and query.flat_rank == 2
    comptime assert key.flat_rank == 2 and value.flat_rank == 2
    var h = Int(hidden)
    var k = Int(kv_width)
    var width = h + 2 * k
    var i = global_idx.x
    if i < Int(rows) * width:
        var row = i // width
        var column = i % width
        if column < h:
            query[row, column] = packed[row, column]
        elif column < h + k:
            key[row, column - h] = packed[row, column]
        else:
            value[row, column - h - k] = packed[row, column]


def _fused_decode_qkv_paged[
    QUERY_HEADS: Int, KV_HEADS: Int, HEAD_DIM: Int, HEAD_MAJOR: Bool,
    PL: TensorLayout, TL: TensorLayout, QL: TensorLayout, CL: TensorLayout, SL: TensorLayout,
    BL: TensorLayout,
](
    packed: TileTensor[DType.bfloat16, PL, MutAnyOrigin],
    cosine: TileTensor[DType.bfloat16, TL, MutAnyOrigin],
    sine: TileTensor[DType.bfloat16, TL, MutAnyOrigin],
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    pool: TileTensor[DType.bfloat16, CL, MutAnyOrigin],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    tables: TileTensor[DType.int32, BL, MutAnyOrigin],
    layer: Int32, layers: Int32, block_size: Int32,
):
    """The single-row fused decode for sequence block_idx.y of a paged batch.

    Sequence s reads packed row s and rotates at positions[s]. Its K and V row
    goes to slot p % block_size of block tables[s, p // block_size] in `layer`
    of the flat pool. Each thread owns one rotary pair, with the unfused path's
    BF16 product roundings; threads below the KV width also copy one V element.
    tests/test_paged_kv.mojo checks the unfused path.
    """
    comptime assert is_apple_gpu()
    comptime assert HEAD_DIM % 2 == 0
    comptime assert packed.flat_rank == 2 and cosine.flat_rank == 2 and sine.flat_rank == 2
    comptime assert query.flat_rank == 3 and pool.flat_rank == 1
    comptime assert positions.flat_rank == 1 and tables.flat_rank == 2
    comptime HALF = HEAD_DIM // 2
    comptime WIDTH = KV_HEADS * HEAD_DIM
    comptime QUERY_WIDTH = QUERY_HEADS * HEAD_DIM
    var i = global_idx.x
    var s = block_idx.y
    var p = Int(positions[s])
    var size = Int(block_size)
    var block = Int(tables[s, p // size])
    var slot = p % size
    if i < (QUERY_HEADS + KV_HEADS) * HALF:
        var head = i // HALF
        var pair = i % HALF
        var first_column = head * HEAD_DIM + pair
        var first = rebind[Scalar[DType.bfloat16]](packed[s, first_column])
        var second = rebind[Scalar[DType.bfloat16]](packed[s, first_column + HALF])
        var c0 = rebind[Scalar[DType.bfloat16]](cosine[p, pair])
        var c1 = rebind[Scalar[DType.bfloat16]](cosine[p, pair + HALF])
        var s0 = rebind[Scalar[DType.bfloat16]](sine[p, pair])
        var s1 = rebind[Scalar[DType.bfloat16]](sine[p, pair + HALF])
        var ac: Scalar[DType.bfloat16] = first.fma[FastMathFlag.NONE](c0, 0)
        var bs: Scalar[DType.bfloat16] = second.fma[FastMathFlag.NONE](s0, 0)
        var bc: Scalar[DType.bfloat16] = second.fma[FastMathFlag.NONE](c1, 0)
        var ass: Scalar[DType.bfloat16] = first.fma[FastMathFlag.NONE](s1, 0)
        var low: Scalar[DType.bfloat16] = ac - bs
        var high: Scalar[DType.bfloat16] = bc + ass
        if head < QUERY_HEADS:
            query[s, head, pair] = rebind[query.ElementType](low)
            query[s, head, pair + HALF] = rebind[query.ElementType](high)
        else:
            var key = kv_row[KV_HEADS, HEAD_DIM, HEAD_MAJOR](block, Int(layer), Int(layers), 0, slot,
                                                            Int(head) - QUERY_HEADS, size)
            pool[key + Int(pair)] = rebind[pool.ElementType](low)
            pool[key + Int(pair) + HALF] = rebind[pool.ElementType](high)
    if i < WIDTH:
        var value = kv_row[KV_HEADS, HEAD_DIM, HEAD_MAJOR](block, Int(layer), Int(layers), 1, slot,
                                                          Int(i) // HEAD_DIM, size)
        pool[value + Int(i) % HEAD_DIM] = rebind[pool.ElementType](packed[s, QUERY_WIDTH + WIDTH + i])


def enqueue_fused_decode_qkv_paged[
    QUERY_HEADS: Int, KV_HEADS: Int, HEAD_DIM: Int, HEAD_MAJOR: Bool, SL: TensorLayout, BL: TensorLayout,
](
    ctx: DeviceContext, mut work: AttentionWorkspace, mut storage: DeviceBuffer[DType.bfloat16],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    tables: TileTensor[DType.int32, BL, MutAnyOrigin],
    layer: Int, layers: Int, block_size: Int,
) raises:
    """Unpack, rotate and append S decode rows into a paged pool: work.packed [S, qkv] to work.query and the pool.

    tables is [S, blocks]. The caller has checked every position against its
    table and the rotary tables, and every table entry against the pool.
    """
    comptime WIDTH = KV_HEADS * HEAD_DIM
    comptime THREADS = (QUERY_HEADS + KV_HEADS) * (HEAD_DIM // 2)
    comptime assert positions.flat_rank == 1 and tables.flat_rank == 2
    var sequences = Int(positions.dim[0]())
    if (sequences < 1 or sequences > work.max_rows or Int(tables.dim[0]()) != sequences
            or work.query_heads != QUERY_HEADS or work.kv_heads != KV_HEADS or work.head_dim != HEAD_DIM):
        raise Error("invalid paged fused decode geometry")
    validate_paged_pool[KV_HEADS, HEAD_DIM](len(storage), layer, layers, block_size, Int(tables.dim[1]()))
    var packed = TileTensor(work.packed, row_major(sequences, QUERY_HEADS * HEAD_DIM + 2 * WIDTH))
    var cosine = TileTensor(work.cosine, row_major(work.capacity, HEAD_DIM))
    var sine = TileTensor(work.sine, row_major(work.capacity, HEAD_DIM))
    var query = TileTensor(work.query, row_major(sequences, QUERY_HEADS, HEAD_DIM))
    var pool = TileTensor(storage, row_major(len(storage)))
    comptime kernel = _fused_decode_qkv_paged[QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_MAJOR, type_of(packed.layout),
        type_of(cosine.layout), type_of(query.layout), type_of(pool.layout), SL, BL]
    ctx.enqueue_function[kernel](packed, cosine, sine, query, pool, positions, tables,
        Int32(layer), Int32(layers), Int32(block_size),
        grid_dim=(ceildiv(THREADS, 128), sequences), block_dim=128)


def _append_paged[
    KV_HEADS: Int, HEAD_DIM: Int, HEAD_MAJOR: Bool,
    KL: TensorLayout, VL: TensorLayout, PL: TensorLayout, TL: TensorLayout,
](
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    pool: TileTensor[DType.bfloat16, PL, MutAnyOrigin],
    table: TileTensor[DType.int32, TL, MutAnyOrigin],
    rows: Int32, past: Int32, layer: Int32, layers: Int32, block_size: Int32,
):
    """_append into a paged pool: row r of the chunk goes to position past + r of one sequence."""
    comptime assert key.flat_rank == 2 and value.flat_rank == 2
    comptime assert pool.flat_rank == 1 and table.flat_rank == 1
    comptime WIDTH = KV_HEADS * HEAD_DIM
    var i = global_idx.x
    if i < Int(rows) * WIDTH:
        var r = i // WIDTH
        var column = i % WIDTH
        var position = Int(past) + Int(r)
        var size = Int(block_size)
        var block = Int(table[position // size])
        var head = Int(column) // HEAD_DIM
        var d = Int(column) % HEAD_DIM
        pool[kv_row[KV_HEADS, HEAD_DIM, HEAD_MAJOR](block, Int(layer), Int(layers), 0, position % size, head, size)
             + d] = key[r, column]
        pool[kv_row[KV_HEADS, HEAD_DIM, HEAD_MAJOR](block, Int(layer), Int(layers), 1, position % size, head, size)
             + d] = value[r, column]


def enqueue_append_paged[
    KV_HEADS: Int, HEAD_DIM: Int, HEAD_MAJOR: Bool, KL: TensorLayout, VL: TensorLayout, TL: TensorLayout,
](
    ctx: DeviceContext,
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    mut storage: DeviceBuffer[DType.bfloat16],
    table: TileTensor[DType.int32, TL, MutAnyOrigin],
    past: Int, layer: Int, layers: Int, block_size: Int,
) raises:
    """Write a chunk's K and V rows [R, kv width] at positions past .. past + R - 1 of one paged sequence.

    table lists the sequence's blocks; it must cover every written position.
    The caller has checked its entries against the pool.
    """
    comptime assert key.flat_rank == 2 and value.flat_rank == 2 and table.flat_rank == 1
    comptime WIDTH = KV_HEADS * HEAD_DIM
    var rows = Int(key.dim[0]())
    var width = Int(table.dim[0]())
    if (rows < 1 or Int(key.dim[1]()) != WIDTH or Int(value.dim[0]()) != rows or Int(value.dim[1]()) != WIDTH
            or past < 0 or past + rows > width * block_size):
        raise Error("paged append requires K/V [R, kv width] at positions its table covers")
    validate_paged_pool[KV_HEADS, HEAD_DIM](len(storage), layer, layers, block_size, width)
    var pool = TileTensor(storage, row_major(len(storage)))
    comptime kernel = _append_paged[KV_HEADS, HEAD_DIM, HEAD_MAJOR, KL, VL, type_of(pool.layout), TL]
    ctx.enqueue_function[kernel](key, value, pool, table, Int32(rows), Int32(past),
        Int32(layer), Int32(layers), Int32(block_size),
        grid_dim=ceildiv(rows * WIDTH, 128), block_dim=128)


def _enqueue_attention_qkv(
    ctx: DeviceContext, mut weights: AttentionWeights,
    mut work: AttentionWorkspace, rows: Int, mapping: Int,
) raises:
    """Projection boundary shared by composition and exact-upstream tests.

    The sublayer validates storage before calling. Mapping 0 keeps three
    rowwise launches; 1/2 use packed rowwise/8x16 MMA; 3/4 use 16x16/8x32 MMA.
    Mapping 5 reuses each weight across four rows with the rowwise reduction.
    Packed mappings include an explicit layout copy.
    No allocation, synchronization, new arithmetic, or rounding in the copy.
    """
    if mapping < 0 or mapping > 5:
        raise Error("unknown QKV projection mapping")
    var h = weights.hidden
    var k = weights.kv_heads * weights.head_dim
    var normal = TileTensor(work.normalized, row_major(rows, h))
    var q = TileTensor(work.raw_query, row_major(rows, h))
    var key = TileTensor(work.raw_key, row_major(rows, k))
    var value = TileTensor(work.raw_value, row_major(rows, k))
    if mapping == 0:
        enqueue_linear_apple_gpu(
            ctx, normal, TileTensor(weights.qkv, row_major(h, h)),
            TileTensor(weights.bias, row_major(h)), q,
        )
        enqueue_linear_apple_gpu(
            ctx, normal,
            TileTensor(weights.qkv.unsafe_ptr().unsafe_offset(h * h), row_major(k, h)),
            TileTensor(weights.bias.unsafe_ptr().unsafe_offset(h), row_major(k)), key,
        )
        enqueue_linear_apple_gpu(
            ctx, normal,
            TileTensor(weights.qkv.unsafe_ptr().unsafe_offset((h + k) * h), row_major(k, h)),
            TileTensor(weights.bias.unsafe_ptr().unsafe_offset(h + k), row_major(k)), value,
        )
        return
    var packed = TileTensor(work.packed, row_major(rows, h + 2 * k))
    var weight = TileTensor(weights.qkv, row_major(h + 2 * k, h))
    var bias = TileTensor(weights.bias, row_major(h + 2 * k))
    if mapping == 1:
        enqueue_linear_apple_gpu(ctx, normal, weight, bias, packed)
    elif mapping == 2:
        enqueue_linear_prefill_mma_8x16_apple_gpu(ctx, normal, weight, bias, packed)
    elif mapping == 3:
        enqueue_linear_prefill_mma_tile_apple_gpu[16, 16](ctx, normal, weight, bias, packed)
    elif mapping == 4:
        enqueue_linear_prefill_mma_tile_apple_gpu[8, 32](ctx, normal, weight, bias, packed)
    else:  # 5
        enqueue_linear_rowwise_rows_apple_gpu[4](ctx, normal, weight, bias, packed)
    ctx.enqueue_function[_unpack_qkv[type_of(packed.layout), type_of(q.layout), type_of(key.layout)]](
        packed, q, key, value, Int32(rows), Int32(h), Int32(k),
        grid_dim=ceildiv(rows * (h + 2 * k), 128), block_dim=128,
    )


def _enqueue_attention_wo(
    ctx: DeviceContext, mut weights: AttentionWeights,
    mut work: AttentionWorkspace, rows: Int, use_mma: Bool, tile: Int = 0,
) raises:
    """Shared Wo boundary for composition and isolated timing on identical data."""
    if tile < 0 or tile > 3:
        raise Error("unknown Wo tile mapping")
    var h = weights.hidden
    var a = TileTensor(work.attention, row_major(rows, h))
    var w = TileTensor(weights.output, row_major(h, h))
    var o = TileTensor(work.projected, row_major(rows, h))
    if tile == 3:
        enqueue_linear_rowwise_rows_apple_gpu[4](ctx, a, w, o)
    elif not use_mma:
        enqueue_linear_apple_gpu(ctx, a, w, o)
    elif tile == 0:
        enqueue_linear_prefill_mma_8x16_apple_gpu(ctx, a, w, o)
    elif tile == 1:
        enqueue_linear_prefill_mma_tile_apple_gpu[16, 16](ctx, a, w, o)
    else:
        enqueue_linear_prefill_mma_tile_apple_gpu[8, 32](ctx, a, w, o)


def _validate_attention_sublayer[XL: TensorLayout](
    ctx: DeviceContext, weights: AttentionWeights, cache: AttentionCache,
    work: AttentionWorkspace, x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    route: Int, qkv_mapping: Int = 0, wo_tile: Int = 0,
) raises -> Int:
    """Shared preflight with no enqueue or state mutation."""
    return _validate_sublayer(ctx, weights, work, x, route, qkv_mapping, wo_tile,
                              cache.length, cache.capacity, cache.kv_heads, cache.head_dim)


def _validate_sublayer[XL: TensorLayout](
    ctx: DeviceContext, weights: AttentionWeights,
    work: AttentionWorkspace, x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    route: Int, qkv_mapping: Int, wo_tile: Int,
    past: Int, capacity: Int, kv_heads: Int, head_dim: Int,
) raises -> Int:
    """The preflight of either KV storage: `past` cached rows in a cache of `capacity` rows."""
    comptime assert x.flat_rank == 2
    var r = Int(x.dim[0]())
    var h = weights.hidden
    var d = weights.head_dim
    var nq = weights.query_heads
    var nk = weights.kv_heads
    var k = nk * d
    var p = past
    var t = p + r
    if route < 0 or route > 11:
        raise Error("unknown attention sublayer route")
    if qkv_mapping < 0 or qkv_mapping > 5:
        raise Error("unknown QKV projection mapping")
    if wo_tile < 0 or wo_tile > 3:
        raise Error("unknown Wo tile mapping")
    var launched_route = route
    if route >= 6 and route <= 10 and r == 1:
        launched_route = 4
    elif (route == 4 or route == 5) and r != 1:
        launched_route = 3
    if (
        r <= 0
        or r > work.max_rows
        or Int(x.dim[1]()) != h
        or p < 0
        or t > capacity
        or t > work.capacity
    ):
        raise Error("invalid sublayer rows, hidden size or cache overflow")
    if (
        kv_heads != nk
        or head_dim != d
        or work.query_heads != nq
        or work.kv_heads != nk
        or work.head_dim != d
    ):
        raise Error("sublayer storage dimensions do not agree")
    if route == 0 and not work.materialized:
        raise Error("materialized route requires probability scratch")
    if launched_route == 3 and not work.fp32_materialized:
        raise Error("FP32 route requires FP32 probability scratch")
    if (launched_route == 9 or launched_route == 10) and work.prefill_splits < (4 if launched_route == 9 else 8):
        raise Error("split prefill requires caller-allocated partial storage")
    if (route == 1 or route == 2 or route >= 4) and (nq != 14 or nk != 2 or d != 64):
        raise Error("optimized GQA requires Qwen dimensions")
    if ctx.api() != "metal":
        raise Error("attention sublayer requires Metal")

    return launched_route


def enqueue_attention_sublayer[
    XL: TensorLayout
](
    ctx: DeviceContext,
    mut weights: AttentionWeights,
    mut cache: AttentionCache,
    mut work: AttentionWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    route: Int = 3,
    wo_mma: Bool = False,
    qkv_mapping: Int = 0,
    wo_tile: Int = 0,
) raises -> Int:
    """Enqueue one sublayer and advance cache length; return the launched route.

    Default route 3 materializes FP32 scores/probabilities, with BF16 I/O.
    Route 0 materializes BF16 GQA. Route 1 uses G32 decode / original MMA prefill;
    route 2 uses split64 H4 decode / rolled-QK MMA prefill. Surroundings match.
    Routes 0-2 remain explicit BF16 compatibility comparisons.
    Routes 4/5 use FP32 G32/split64-H4 decode for R=1; for R>1 they use
    materialized FP32 attention and return actual route 3. No length crossover.
    Route 6 uses FP32 rolled-MMA prefill and G32 decode (actual route 4).
    Routes 7/8 use 16/8 query rows per FP32 tile; 9/10 use 4/8 KV splits
    with a separate FP32 merge. All use G32 for R=1 (actual route 4).
    Route 11 uses the same FP32 G32 arithmetic for every query at every R.
    wo_mma selects only the bias-free output projection's 8x16 MMA mapping;
    it preserves BF16 projection output before the separate residual addition.
    It is an explicit experiment, independent of the GQA precision route.
    qkv_mapping 0 keeps separate rowwise projections; 1/2 select the existing
    packed rowwise/MMA projection, followed by a bit-preserving layout copy.
    QKV mappings 3/4 select 16x16/8x32 MMA. When wo_mma is true, wo_tile 1/2
    select those same larger tiles for Wo; zero preserves the 8x16 control.
    X must not overlap any writable workspace/cache region. Read output from
    work.output only after completion and before its next overwrite.
    """
    var launched_route = _validate_attention_sublayer(
        ctx, weights, cache, work, x, route, qkv_mapping, wo_tile
    )
    var r = Int(x.dim[0]())
    var h = weights.hidden
    var d = weights.head_dim
    var nq = weights.query_heads
    var nk = weights.kv_heads
    var k = nk * d
    var p = cache.length
    var t = p + r

    var normal = TileTensor(work.normalized, row_major(r, h))
    enqueue_rms_norm_apple_gpu(
        ctx, x, TileTensor(weights.norm, row_major(h)), normal
    )
    _enqueue_attention_qkv(ctx, weights, work, r, qkv_mapping)
    var q = TileTensor(work.query, row_major(r, nq, d))
    var v2 = TileTensor(work.raw_value, row_major(r, k))
    var c = TileTensor(work.cosine, row_major(work.capacity, d))
    var s = TileTensor(work.sine, row_major(work.capacity, d))
    enqueue_rope_apple_gpu(
        ctx, TileTensor(work.raw_query, row_major(r, nq, d)), c, s, q, p
    )
    enqueue_rope_apple_gpu(
        ctx,
        TileTensor(work.raw_key, row_major(r, nk, d)),
        c,
        s,
        TileTensor(work.rotated_key, row_major(r, nk, d)),
        p,
    )
    var kr = TileTensor(work.rotated_key, row_major(r, k))
    var ck = TileTensor(cache.key, row_major(cache.capacity, k))
    var cv = TileTensor(cache.value, row_major(cache.capacity, k))
    ctx.enqueue_function[
        _append[type_of(kr.layout), type_of(v2.layout), type_of(ck.layout)]
    ](
        kr,
        v2,
        ck,
        cv,
        Int32(r),
        Int32(k),
        Int32(p),
        grid_dim=ceildiv(r * k, 128),
        block_dim=128,
    )
    var keys = TileTensor(cache.key, row_major(t, nk, d))
    var values = TileTensor(cache.value, row_major(t, nk, d))
    var a = TileTensor(work.attention, row_major(r, nq, d))
    if route == 0:
        enqueue_grouped_query_attention_apple_gpu(
            ctx,
            q,
            keys,
            values,
            TileTensor(work.scratch, row_major(r, nq, t)),
            a,
        )
    elif launched_route == 3:
        enqueue_grouped_query_attention_apple_gpu(
            ctx, q, keys, values,
            TileTensor(work.fp32_scratch, row_major(r, nq, t)), a,
        )
    elif launched_route == 11:
        enqueue_grouped_query_attention_consistent_apple_gpu(
            ctx, q, keys, values, a, TileTensor(work.split, row_major(14, 1, 66)))
    elif launched_route == 6:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            32, 32, MMA=True, SCHEDULE=2, FP32=True
        ](ctx, q, keys, values, a)
    elif launched_route == 7:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            16, 32, MMA=True, SCHEDULE=2, FP32=True
        ](ctx, q, keys, values, a)
    elif launched_route == 8:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            8, 32, MMA=True, SCHEDULE=2, FP32=True
        ](ctx, q, keys, values, a)
    elif launched_route == 9:
        enqueue_grouped_query_attention_prefill_split_apple_gpu[4](
            ctx, q, keys, values,
            TileTensor(work.prefill_partial, row_major(r, 14, 4, 66)), a,
        )
    elif launched_route == 10:
        enqueue_grouped_query_attention_prefill_split_apple_gpu[8](
            ctx, q, keys, values,
            TileTensor(work.prefill_partial, row_major(r, 14, 8, 66)), a,
        )
    elif launched_route == 4:
        enqueue_grouped_query_attention_decode_apple_gpu[32, 1, 1, fp32_scores=True](
            ctx, q, keys, values, a,
            TileTensor(work.split, row_major(14, 1, 66)),
        )
    elif launched_route == 5:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 4, 64, fp32_scores=True](
            ctx, q, keys, values, a,
            TileTensor(work.split, row_major(14, 64, 66)),
        )
    elif r == 1:
        if route == 1:
            enqueue_grouped_query_attention_decode_apple_gpu[32, 1, 1](
                ctx,
                q,
                keys,
                values,
                a,
                TileTensor(work.split, row_major(14, 1, 66)),
            )
        else:
            enqueue_grouped_query_attention_decode_apple_gpu[1, 4, 64](
                ctx,
                q,
                keys,
                values,
                a,
                TileTensor(work.split, row_major(14, 64, 66)),
            )
    else:
        if route == 1:
            enqueue_grouped_query_attention_prefill_apple_gpu[32, 32, MMA=True](
                ctx, q, keys, values, a
            )
        else:
            enqueue_grouped_query_attention_prefill_apple_gpu[
                32, 32, MMA=True, SCHEDULE=2
            ](ctx, q, keys, values, a)
    var projected = TileTensor(work.projected, row_major(r, h))
    _enqueue_attention_wo(ctx, weights, work, r, wo_mma, wo_tile)
    enqueue_residual_apple_gpu(
        ctx, x, projected, TileTensor(work.output, row_major(r, h))
    )
    cache.length = t
    return launched_route


def enqueue_attention_sublayer_integrated[XL: TensorLayout](
    ctx: DeviceContext, mut weights: AttentionWeights,
    mut cache: AttentionCache, mut work: AttentionWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    gqa_mapping: Int = 0,
    projection_mapping: Int = 0,
) raises -> Int:
    """Compose the prior Qwen projection and FP32 GQA studies on Metal.

    Packed rowwise QKV and rowwise Wo for R<16; existing 8x16 MMA projections
    for R>=16. FP32 G32 attention for decode, rolled MMA for multi-row calls.
    Sixteen is a conservative study policy, not a universal crossover claim.
    Qwen dimensions only; no materialized probability workspace is required.
    The original enqueue remains the inspectable control. Both APIs preserve
    the same cache, lifetime, BF16 boundary and no-allocation enqueue contract.
    Explicit GQA mappings 1/2 use 16/8 query rows; 3/4 use 4/8 KV splits
    and require preallocated partial storage for multi-row calls. Mapping 0
    preserves the integrated study control; this is not an automatic selector.
    Projection mappings 1/2 change only Wo to 16x16/8x32 MMA; 3/4 change only
    packed QKV to those tiles. They require the control GQA mapping and retain
    rowwise projections below sixteen rows. Mapping 5 combines 16x16 QKV and
    Wo, and also supports GQA mapping 4 (split8) to compose the two studies.
    Other projection/GQA combinations remain outside this bounded study.
    Zero keeps both 8x16 projections.
    GQA mapping 5 is the consistency baseline: route 11 and rowwise projections
    for every row count. Projection mapping 6 forces 8x16 MMA at every row
    count; mapping 7 packs QKV and reuses rowwise weights across four rows.
    """
    comptime assert x.flat_rank == 2
    var call = _integrated_route(gqa_mapping, projection_mapping, Int(x.dim[0]()))
    return enqueue_attention_sublayer(ctx, weights, cache, work, x, Int(call[0]), call[1] != 0,
                                      Int(call[2]), Int(call[3]))


def _integrated_route(gqa_mapping: Int, projection_mapping: Int, rows: Int) raises -> SIMD[DType.int64, 4]:
    """The integrated mappings' sublayer call: (route, Wo MMA, QKV mapping, Wo tile)."""
    if gqa_mapping < 0 or gqa_mapping > 5:
        raise Error("unknown integrated GQA mapping")
    if projection_mapping < 0 or projection_mapping > 7:
        raise Error("unknown integrated projection mapping")
    if projection_mapping >= 6 and gqa_mapping != 5:
        raise Error("policy projection mappings require consistent attention")
    if projection_mapping and gqa_mapping and not ((projection_mapping == 5 and gqa_mapping == 4)
        or (projection_mapping >= 6 and gqa_mapping == 5)):
        raise Error("projection study requires control GQA mapping")
    if gqa_mapping == 5:
        if projection_mapping >= 6:
            # Padding inactive rows changes occupancy, not K reduction order.
            return SIMD[DType.int64, 4](11, Int64(projection_mapping == 6),
                Int64(2 if projection_mapping == 6 else projection_mapping - 2),
                Int64(0 if projection_mapping == 6 else projection_mapping - 4))
        return SIMD[DType.int64, 4](11, 0, 0, 0)
    var use_mma = rows >= 16
    var qkv = 2 if use_mma else 1
    if use_mma and projection_mapping >= 3:
        qkv = 3 if projection_mapping == 5 else projection_mapping
    return SIMD[DType.int64, 4](Int64(6 + gqa_mapping), Int64(use_mma), Int64(qkv),
        Int64(1 if projection_mapping == 5 else (projection_mapping if projection_mapping <= 2 else 0)))


def enqueue_attention_sublayer_paged[
    HEAD_MAJOR: Bool, XL: TensorLayout, TL: TensorLayout, SL: TensorLayout,
](
    ctx: DeviceContext,
    mut weights: AttentionWeights,
    mut storage: DeviceBuffer[DType.bfloat16],
    table: TileTensor[DType.int32, TL, MutAnyOrigin],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    past: Int, layer: Int, layers: Int, block_size: Int,
    mut work: AttentionWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    route: Int,
    wo_mma: Bool = False,
    qkv_mapping: Int = 0,
    wo_tile: Int = 0,
) raises -> Int:
    """enqueue_attention_sublayer for one paged sequence: its K/V in the blocks `table` lists.

    The R rows of x take positions past .. past + R - 1, which `positions` holds
    on the device. Normalization, projections, RoPE, Wo and the residual are the
    contiguous sublayer's; the K/V append and attention use their paged forms,
    for the routes the model's configurations launch: 6, 10 and 11, and 4 for
    one row. The caller advances the sequence's length.
    """
    var launched_route = _validate_sublayer(ctx, weights, work, x, route, qkv_mapping, wo_tile,
                                            past, work.capacity, weights.kv_heads, weights.head_dim)
    if launched_route != 4 and launched_route != 6 and launched_route != 10 and launched_route != 11:
        raise Error("paged attention runs routes 4, 6, 10 and 11")
    comptime assert table.flat_rank == 1 and positions.flat_rank == 1
    var r = Int(x.dim[0]())
    var h = weights.hidden
    var d = weights.head_dim
    var nq = weights.query_heads
    var nk = weights.kv_heads
    var k = nk * d
    var p = past
    var t = p + r
    if Int(positions.dim[0]()) != r:
        raise Error("paged attention needs one position per row")

    var normal = TileTensor(work.normalized, row_major(r, h))
    enqueue_rms_norm_apple_gpu(
        ctx, x, TileTensor(weights.norm, row_major(h)), normal
    )
    _enqueue_attention_qkv(ctx, weights, work, r, qkv_mapping)
    var q = TileTensor(work.query, row_major(r, nq, d))
    var v2 = TileTensor(work.raw_value, row_major(r, k))
    var c = TileTensor(work.cosine, row_major(work.capacity, d))
    var s = TileTensor(work.sine, row_major(work.capacity, d))
    enqueue_rope_apple_gpu(
        ctx, TileTensor(work.raw_query, row_major(r, nq, d)), c, s, q, p
    )
    enqueue_rope_apple_gpu(
        ctx,
        TileTensor(work.raw_key, row_major(r, nk, d)),
        c,
        s,
        TileTensor(work.rotated_key, row_major(r, nk, d)),
        p,
    )
    var kr = TileTensor(work.rotated_key, row_major(r, k))
    enqueue_append_paged[2, 64, HEAD_MAJOR](ctx, kr, v2, storage, table, p, layer, layers, block_size)
    var pool = TileTensor(storage, row_major(len(storage)))
    var a = TileTensor(work.attention, row_major(r, nq, d))
    if launched_route == 6:
        enqueue_paged_attention_prefill_apple_gpu[HEAD_MAJOR](ctx, q, pool, table, a, t, layer, layers, block_size)
    elif launched_route == 10:
        enqueue_paged_attention_prefill_split_apple_gpu[8, HEAD_MAJOR](ctx, q, pool, table,
            TileTensor(work.prefill_partial, row_major(r, 14, 8, 66)), a, t, layer, layers, block_size)
    else:
        # Routes 11 and 4: every query row with route 4's arithmetic.
        enqueue_paged_attention_g32_apple_gpu[14, 2, 64, HEAD_MAJOR](ctx, q, pool, a, positions,
            TileTensor(table.ptr, row_major(1, Int(table.dim[0]()))), r, layer, layers, block_size)
    var projected = TileTensor(work.projected, row_major(r, h))
    _enqueue_attention_wo(ctx, weights, work, r, wo_mma, wo_tile)
    enqueue_residual_apple_gpu(
        ctx, x, projected, TileTensor(work.output, row_major(r, h))
    )
    return launched_route


def enqueue_attention_sublayer_integrated_paged[
    HEAD_MAJOR: Bool, XL: TensorLayout, TL: TensorLayout, SL: TensorLayout,
](
    ctx: DeviceContext, mut weights: AttentionWeights,
    mut storage: DeviceBuffer[DType.bfloat16],
    table: TileTensor[DType.int32, TL, MutAnyOrigin],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    past: Int, layer: Int, layers: Int, block_size: Int,
    mut work: AttentionWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    gqa_mapping: Int = 0,
    projection_mapping: Int = 0,
) raises -> Int:
    """enqueue_attention_sublayer_integrated for one paged sequence, with the same mappings."""
    comptime assert x.flat_rank == 2
    var call = _integrated_route(gqa_mapping, projection_mapping, Int(x.dim[0]()))
    return enqueue_attention_sublayer_paged[HEAD_MAJOR](ctx, weights, storage, table, positions, past, layer,
        layers, block_size, work, x, Int(call[0]), call[1] != 0, Int(call[2]), Int(call[3]))
