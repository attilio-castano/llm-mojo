"""Output-only decode attention: bounded experiments, route 4 and its paged, batched form."""

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.gpu.primitives import warp
from std.math import exp, max, min
from std.sys.info import is_apple_gpu
from llm_mojo.kernels.paged_kv import kv_row, kv_slot_stride, validate_paged_pool


def _decode_kernel[
    groups: Int,
    heads: Int,
    splits: Int,
    conditional_rescale: Bool,
    fp32_scores: Bool,
    QL: TensorLayout,
    KL: TensorLayout,
    VL: TensorLayout,
    OL: TensorLayout,
    WL: TensorLayout,
](
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    workspace: TileTensor[DType.float32, WL, MutAnyOrigin],
    rows: Int32,
):
    comptime assert is_apple_gpu()
    comptime assert query.flat_rank == 3
    comptime assert key.flat_rank == 3
    comptime assert value.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert workspace.flat_rank == 3
    var lane = thread_idx.x % 32
    var group = thread_idx.x // 32
    comptime head_blocks = (7 + heads - 1) // heads
    var kv_head = block_idx.x // head_blocks
    var first_head = kv_head * 7 + (block_idx.x % head_blocks) * heads
    var split = block_idx.y
    # Query rows are independent work. The visible prefix, and therefore every
    # reduction, depends only on this query's absolute position.
    var query_row = block_idx.z
    var visible = Int(rows) - Int(query.dim[0]()) + query_row + 1
    var begin = visible * split // splits
    var end = visible * (split + 1) // splits

    # Comptime indexing scalarizes these small arrays into per-lane registers.
    var q0 = stack_allocation[DType.float32](row_major[heads]()).fill(0)
    var q1 = stack_allocation[DType.float32](row_major[heads]()).fill(0)
    var m = stack_allocation[DType.float32](row_major[heads]()).fill(
        -3.402823466e38
    )
    var z = stack_allocation[DType.float32](row_major[heads]()).fill(0)
    var u0 = stack_allocation[DType.float32](row_major[heads]()).fill(0)
    var u1 = stack_allocation[DType.float32](row_major[heads]()).fill(0)
    comptime assert q0.flat_rank == 1
    comptime assert q1.flat_rank == 1
    comptime assert m.flat_rank == 1
    comptime assert z.flat_rank == 1
    comptime assert u0.flat_rank == 1
    comptime assert u1.flat_rank == 1
    comptime for h in range(heads):
        if first_head + h < (kv_head + 1) * 7:
            q0[h] = query[query_row, first_head + h, lane].cast[DType.float32]()
            q1[h] = query[query_row, first_head + h, lane + 32].cast[DType.float32]()

    for t in range(begin + group, end, groups):
        var k0 = rebind[Float32](key[t, kv_head, lane].cast[DType.float32]())
        var k1 = rebind[Float32](
            key[t, kv_head, lane + 32].cast[DType.float32]()
        )
        var v0 = rebind[Float32](value[t, kv_head, lane].cast[DType.float32]())
        var v1 = rebind[Float32](
            value[t, kv_head, lane + 32].cast[DType.float32]()
        )
        comptime for h in range(heads):
            # All lanes participate, including an unused final head slot.
            var score = warp.sum(q0[h] * k0 + q1[h] * k1) * 0.125
            comptime if not fp32_scores:
                score = score.cast[DType.bfloat16]().cast[DType.float32]()
            comptime if conditional_rescale:
                # score is SIMD-group uniform after warp.sum. Only a new
                # maximum changes the scale of the already accumulated state.
                if score > m[h]:
                    var alpha = exp(m[h] - score)
                    z[h] = alpha * z[h] + 1.0
                    u0[h] = alpha * u0[h] + v0
                    u1[h] = alpha * u1[h] + v1
                    m[h] = score
                else:
                    var beta = exp(score - m[h])
                    z[h] += beta
                    u0[h] += beta * v0
                    u1[h] += beta * v1
            else:
                var new_m = max(m[h], score)
                var alpha = exp(m[h] - new_m)
                var beta = exp(score - new_m)
                z[h] = alpha * z[h] + beta
                u0[h] = alpha * u0[h] + beta * v0
                u1[h] = alpha * u1[h] + beta * v1
                m[h] = new_m

    comptime if groups > 1:
        var partial = stack_allocation[
            DType.float32, address_space=AddressSpace.SHARED
        ](row_major[groups, heads, 66]())
        comptime assert partial.flat_rank == 3
        comptime for h in range(heads):
            partial[group, h, lane] = u0[h]
            partial[group, h, lane + 32] = u1[h]
            if lane == 0:
                partial[group, h, 64] = m[h]
                partial[group, h, 65] = z[h]
        barrier()
        if group == 0:
            comptime for h in range(heads):
                var merged_m: Float32 = -3.402823466e38
                for g in range(groups):
                    merged_m = max(merged_m, partial[g, h, 64])
                var merged_z: Float32 = 0
                var merged0: Float32 = 0
                var merged1: Float32 = 0
                for g in range(groups):
                    var weight = exp(partial[g, h, 64] - merged_m)
                    merged_z += weight * partial[g, h, 65]
                    merged0 += weight * partial[g, h, lane]
                    merged1 += weight * partial[g, h, lane + 32]
                m[h] = merged_m
                z[h] = merged_z
                u0[h] = merged0
                u1[h] = merged1

    if group == 0:
        comptime for h in range(heads):
            var head = first_head + h
            if head < (kv_head + 1) * 7:
                comptime if splits == 1:
                    output[query_row, head, lane] = (u0[h] / z[h]).cast[
                        DType.bfloat16
                    ]()
                    output[query_row, head, lane + 32] = (u1[h] / z[h]).cast[
                        DType.bfloat16
                    ]()
                else:
                    workspace[head, split, lane] = u0[h]
                    workspace[head, split, lane + 32] = u1[h]
                    if lane == 0:
                        workspace[head, split, 64] = m[h]
                        workspace[head, split, 65] = z[h]


def enqueue_grouped_query_attention_consistent_apple_gpu[
    QL: TensorLayout, KL: TensorLayout, VL: TensorLayout,
    OL: TensorLayout, WL: TensorLayout,
](
    context: DeviceContext,
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    workspace: TileTensor[DType.float32, WL, MutAnyOrigin],
) raises:
    """G32 FP32 arithmetic for every causal query, batched in one GPU launch.

    The same 32 groups traverse keys at fixed residue classes modulo 32 and
    merge in group order, independent of query count and cached-prefix length.
    Q/O [R,14,64] are a suffix of K/V [T,2,64]. Workspace is unused.
    """
    comptime assert query.flat_rank == 3 and key.flat_rank == 3
    comptime assert value.flat_rank == 3 and output.flat_rank == 3
    var r = Int(query.dim[0]())
    var t = Int(key.dim[0]())
    if context.api() != "metal" or r < 1 or t < r or t > 4096:
        raise Error("consistent attention requires Metal and a valid causal suffix")
    if (Int(query.dim[1]()) != 14 or Int(query.dim[2]()) != 64
        or Int(key.dim[1]()) != 2 or Int(key.dim[2]()) != 64
        or Int(value.dim[0]()) != t or Int(value.dim[1]()) != 2 or Int(value.dim[2]()) != 64
        or Int(output.dim[0]()) != r or Int(output.dim[1]()) != 14 or Int(output.dim[2]()) != 64):
        raise Error("consistent attention requires Q/O[R,14,64] and K/V[T,2,64]")
    comptime kernel = _decode_kernel[32, 1, 1, False, True, QL, KL, VL, OL, WL]
    context.enqueue_function[kernel](query, key, value, output, workspace, Int32(t),
        grid_dim=(14, 1, r), block_dim=1024)


def _decode_merge_kernel[
    splits: Int,
    OL: TensorLayout,
    WL: TensorLayout,
](
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    workspace: TileTensor[DType.float32, WL, MutAnyOrigin],
):
    comptime assert is_apple_gpu()
    comptime assert output.flat_rank == 3
    comptime assert workspace.flat_rank == 3
    var lane = thread_idx.x
    var head = block_idx.x
    var m: Float32 = -3.402823466e38
    for s in range(splits):
        m = max(m, workspace[head, s, 64])
    var z: Float32 = 0
    var u0: Float32 = 0
    var u1: Float32 = 0
    for s in range(splits):
        var weight = exp(workspace[head, s, 64] - m)
        z += weight * workspace[head, s, 65]
        u0 += weight * workspace[head, s, lane]
        u1 += weight * workspace[head, s, lane + 32]
    output[0, head, lane] = (u0 / z).cast[DType.bfloat16]()
    output[0, head, lane + 32] = (u1 / z).cast[DType.bfloat16]()


def _paged_g32_kernel[
    QUERY_HEADS: Int,
    KV_HEADS: Int,
    HEAD_DIM: Int,
    HEAD_MAJOR: Bool,
    QL: TensorLayout,
    PL: TensorLayout,
    OL: TensorLayout,
    SL: TensorLayout,
    TL: TensorLayout,
](
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    pool: TileTensor[DType.bfloat16, PL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    tables: TileTensor[DType.int32, TL, MutAnyOrigin],
    rows_per_sequence: Int32,
    layer: Int32,
    layers: Int32,
    block_size: Int32,
):
    """Route 4's arithmetic (_decode_kernel[32, 1, 1] with FP32 scores) for one query row of a paged sequence.

    Threadgroup (head, row) attends with query row `row` to the positions[row] + 1
    keys of sequence row // rows_per_sequence, whose blocks its table row lists.
    SIMD group g takes the keys t ≡ g (mod 32) in increasing order, reading the
    table once per block, and the 32 groups merge in group order, as in route 4.
    A block holds a multiple of 32 slots or the whole sequence, so t ≡ g (mod 32)
    exactly when its slot is.
    """
    comptime assert is_apple_gpu()
    comptime assert HEAD_DIM == 64, "each lane owns head dimensions lane and lane + 32"
    comptime assert QUERY_HEADS % KV_HEADS == 0
    comptime assert query.flat_rank == 3 and pool.flat_rank == 1 and output.flat_rank == 3
    comptime assert positions.flat_rank == 1 and tables.flat_rank == 2
    comptime GROUP = QUERY_HEADS // KV_HEADS
    comptime STRIDE = kv_slot_stride[KV_HEADS, HEAD_DIM, HEAD_MAJOR]()
    var lane = thread_idx.x % 32
    var group = thread_idx.x // 32
    var head = block_idx.x
    var kv_head = head // GROUP
    var row = block_idx.y
    var sequence = Int(row) // Int(rows_per_sequence)
    var visible = Int(positions[row]) + 1
    var size = Int(block_size)
    var q0 = rebind[Float32](query[row, head, lane].cast[DType.float32]())
    var q1 = rebind[Float32](query[row, head, lane + 32].cast[DType.float32]())
    var m: Float32 = -3.402823466e38
    var z: Float32 = 0
    var u0: Float32 = 0
    var u1: Float32 = 0
    for first in range(0, visible, size):
        var block = Int(tables[sequence, first // size])
        var key = kv_row[KV_HEADS, HEAD_DIM, HEAD_MAJOR](block, Int(layer), Int(layers), 0, 0, Int(kv_head), size)
        var value = kv_row[KV_HEADS, HEAD_DIM, HEAD_MAJOR](block, Int(layer), Int(layers), 1, 0, Int(kv_head), size)
        for slot in range(Int(group), min(size, visible - first), 32):
            var k0 = rebind[Float32](pool[key + slot * STRIDE + Int(lane)].cast[DType.float32]())
            var k1 = rebind[Float32](pool[key + slot * STRIDE + Int(lane) + 32].cast[DType.float32]())
            var v0 = rebind[Float32](pool[value + slot * STRIDE + Int(lane)].cast[DType.float32]())
            var v1 = rebind[Float32](pool[value + slot * STRIDE + Int(lane) + 32].cast[DType.float32]())
            var score = warp.sum(q0 * k0 + q1 * k1) * 0.125
            var new_m = max(m, score)
            var alpha = exp(m - new_m)
            var beta = exp(score - new_m)
            z = alpha * z + beta
            u0 = alpha * u0 + beta * v0
            u1 = alpha * u1 + beta * v1
            m = new_m
    var partial = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[32, 1, 66]())
    comptime assert partial.flat_rank == 3
    partial[group, 0, lane] = u0
    partial[group, 0, lane + 32] = u1
    if lane == 0:
        partial[group, 0, 64] = m
        partial[group, 0, 65] = z
    barrier()
    if group == 0:
        var merged_m: Float32 = -3.402823466e38
        for g in range(32):
            merged_m = max(merged_m, partial[g, 0, 64])
        var merged_z: Float32 = 0
        var merged0: Float32 = 0
        var merged1: Float32 = 0
        for g in range(32):
            var weight = exp(partial[g, 0, 64] - merged_m)
            merged_z += weight * partial[g, 0, 65]
            merged0 += weight * partial[g, 0, lane]
            merged1 += weight * partial[g, 0, lane + 32]
        output[row, head, lane] = (merged0 / merged_z).cast[DType.bfloat16]()
        output[row, head, lane + 32] = (merged1 / merged_z).cast[DType.bfloat16]()


def enqueue_paged_attention_g32_apple_gpu[
    QUERY_HEADS: Int,
    KV_HEADS: Int,
    HEAD_DIM: Int,
    HEAD_MAJOR: Bool,
    QL: TensorLayout,
    PL: TensorLayout,
    OL: TensorLayout,
    SL: TensorLayout,
    TL: TensorLayout,
](
    context: DeviceContext,
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    pool: TileTensor[DType.bfloat16, PL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    positions: TileTensor[DType.int32, SL, MutAnyOrigin],
    tables: TileTensor[DType.int32, TL, MutAnyOrigin],
    rows_per_sequence: Int,
    layer: Int,
    layers: Int,
    block_size: Int,
) raises:
    """Route 4's arithmetic for R query rows over paged K/V in one launch.

    Q/O are [R, heads, dim], the pool is flat, positions [R] hold each row's
    absolute position and tables [S, blocks] each sequence's blocks, with
    R = S * rows_per_sequence. Batched decode passes one row per sequence; one
    sequence's rows (routes 11 and 4) pass rows_per_sequence = R. The caller has
    checked every position against its table and every table entry against the
    pool, and keeps all storage live through completion.
    """
    comptime assert query.flat_rank == 3 and pool.flat_rank == 1 and output.flat_rank == 3
    comptime assert positions.flat_rank == 1 and tables.flat_rank == 2
    var rows = Int(query.dim[0]())
    var sequences = Int(tables.dim[0]())
    if context.api() != "metal":
        raise Error("paged attention requires the Metal device API")
    if (rows < 1 or Int(query.dim[1]()) != QUERY_HEADS or Int(query.dim[2]()) != HEAD_DIM
            or Int(output.dim[0]()) != rows or Int(output.dim[1]()) != QUERY_HEADS
            or Int(output.dim[2]()) != HEAD_DIM or Int(positions.dim[0]()) != rows
            or rows_per_sequence < 1 or sequences * rows_per_sequence != rows):
        raise Error("paged attention requires Q/O [R, heads, dim], R positions and one table row per sequence")
    validate_paged_pool[KV_HEADS, HEAD_DIM](Int(pool.dim[0]()), layer, layers, block_size, Int(tables.dim[1]()))
    comptime kernel = _paged_g32_kernel[QUERY_HEADS, KV_HEADS, HEAD_DIM, HEAD_MAJOR, QL, PL, OL, SL, TL]
    context.enqueue_function[kernel](
        query, pool, output, positions, tables,
        Int32(rows_per_sequence), Int32(layer), Int32(layers), Int32(block_size),
        grid_dim=(QUERY_HEADS, rows),
        block_dim=32 * 32,
    )


def enqueue_grouped_query_attention_decode_apple_gpu[
    groups: Int,
    heads: Int,
    splits: Int,
    QL: TensorLayout,
    KL: TensorLayout,
    VL: TensorLayout,
    OL: TensorLayout,
    WL: TensorLayout,
    conditional_rescale: Bool = False,
    fp32_scores: Bool = False,
](
    context: DeviceContext,
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    workspace: TileTensor[DType.float32, WL, MutAnyOrigin],
) raises:
    """Enqueue bounded Qwen decode, with explicit experimental ownership.

    Q/O are [1,14,64]; K/V are [T,2,64], 1 <= T <= 4096. All views must
    be contiguous row-major and non-overlapping. Caller owns buffer lifetime.
    Workspace is [14,splits,66] FP32 for split decode. For splits=1 it is
    unused (a [1,1,1] view suffices). The enqueue allocates and synchronizes
    nothing. It issues one dispatch, or two when splits>1. No probabilities
    are exposed; the separate materialized API retains that postcondition.
    fp32_scores=True retains scaled scores in FP32, as well as every online
    softmax/weighted-sum state. The default preserves the BF16-score studies.
    """
    comptime assert groups == 1 or groups == 2 or groups == 8 or groups == 32
    comptime assert heads == 1 or heads == 2 or heads == 4 or heads == 7
    comptime assert splits == 1 or splits == 4 or splits == 16 or splits == 64
    comptime assert query.flat_rank == 3
    comptime assert key.flat_rank == 3
    comptime assert value.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime assert workspace.flat_rank == 3
    if context.api() != "metal":
        raise Error("decode requires the Metal device API")
    var rows = Int(key.dim[0]())
    if (
        Int(query.dim[0]()) != 1
        or Int(query.dim[1]()) != 14
        or Int(query.dim[2]()) != 64
    ):
        raise Error("decode requires Q[1,14,64]")
    if (
        rows < 1
        or rows > 4096
        or Int(key.dim[1]()) != 2
        or Int(key.dim[2]()) != 64
    ):
        raise Error("decode requires K[T,2,64], 1 <= T <= 4096")
    if (
        Int(value.dim[0]()) != rows
        or Int(value.dim[1]()) != 2
        or Int(value.dim[2]()) != 64
    ):
        raise Error("decode value shape must match key")
    if (
        Int(output.dim[0]()) != 1
        or Int(output.dim[1]()) != 14
        or Int(output.dim[2]()) != 64
    ):
        raise Error("decode output shape must match query")
    comptime if splits > 1:
        if (
            Int(workspace.dim[0]()) != 14
            or Int(workspace.dim[1]()) != splits
            or Int(workspace.dim[2]()) != 66
        ):
            raise Error("split decode requires FP32 workspace [14,splits,66]")
    comptime kernel = _decode_kernel[
        groups, heads, splits, conditional_rescale, fp32_scores, QL, KL, VL, OL, WL
    ]
    context.enqueue_function[kernel](
        query,
        key,
        value,
        output,
        workspace,
        Int32(rows),
        grid_dim=(2 * ((7 + heads - 1) // heads), splits),
        block_dim=groups * 32,
    )
    comptime if splits > 1:
        comptime merge = _decode_merge_kernel[splits, OL, WL]
        context.enqueue_function[merge](
            output, workspace, grid_dim=14, block_dim=32
        )
