"""Bounded output-only Qwen decode experiments; no automatic dispatch policy."""

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.gpu.primitives import warp
from std.math import exp, max
from std.sys.info import is_apple_gpu


def _decode_kernel[
    groups: Int,
    heads: Int,
    splits: Int,
    conditional_rescale: Bool,
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
    var begin = Int(rows) * split // splits
    var end = Int(rows) * (split + 1) // splits

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
            q0[h] = query[0, first_head + h, lane].cast[DType.float32]()
            q1[h] = query[0, first_head + h, lane + 32].cast[DType.float32]()

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
                    output[0, head, lane] = (u0[h] / z[h]).cast[
                        DType.bfloat16
                    ]()
                    output[0, head, lane + 32] = (u1[h] / z[h]).cast[
                        DType.bfloat16
                    ]()
                else:
                    workspace[head, split, lane] = u0[h]
                    workspace[head, split, lane + 32] = u1[h]
                    if lane == 0:
                        workspace[head, split, 64] = m[h]
                        workspace[head, split, 65] = z[h]


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
        groups, heads, splits, conditional_rescale, QL, KL, VL, OL, WL
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
