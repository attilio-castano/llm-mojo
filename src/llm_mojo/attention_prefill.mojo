"""Qwen prefill candidates with explicit query-tile ownership.

The materialized control retains BF16 probability rounding. Fused paths keep
FP32 online state and expose only O; the MMA path rounds unnormalized tile
weights to BF16 before PV by default. The explicit FP32 rolled-MMA option
retains FP32 scores and weights through PV. These paths are separately tested.
"""
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from llm_mojo.attention import (
    _validate_grouped_query_attention,
    _grouped_query_attention_qk_apple_gpu_kernel,
    _grouped_query_attention_pv_apple_gpu_kernel,
)
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from std.gpu import block_idx, thread_idx, lane_id
from std.gpu.primitives import warp
from std.math import exp, max, min, ceildiv
from std.sys.info import is_apple_gpu
from std.utils.numerics import neg_inf


def _softmax[
    SL: TensorLayout
](
    scratch: TileTensor[DType.bfloat16, SL, MutAnyOrigin],
    rows: Int32,
    tokens: Int32,
):
    comptime assert scratch.flat_rank == 3
    var lane = Int(lane_id())
    var row_head = block_idx.x * 4 + thread_idx.x // 32
    var r = row_head // 14
    var h = row_head % 14
    if r < Int(rows):
        var visible = Int(tokens) - Int(rows) + r + 1
        var m: Float32 = neg_inf[DType.float32]()
        for t in range(lane, visible, 32):
            m = max(m, rebind[Float32](scratch[r, h, t].cast[DType.float32]()))
        m = warp.max(m)
        var z: Float32 = 0
        for t in range(lane, visible, 32):
            z += exp(
                rebind[Float32](scratch[r, h, t].cast[DType.float32]()) - m
            )
        z = warp.sum(z)
        for t in range(lane, Int(tokens), 32):
            var p: Float32 = 0
            if t < visible:
                p = (
                    exp(
                        rebind[Float32](scratch[r, h, t].cast[DType.float32]())
                        - m
                    )
                    / z
                )
            scratch[r, h, t] = p.cast[DType.bfloat16]()


def enqueue_grouped_query_attention_prefill_materialized_apple_gpu[
    QL: TensorLayout,
    KL: TensorLayout,
    SL: TensorLayout,
](
    ctx: DeviceContext,
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, SL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
) raises:
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    _validate_grouped_query_attention(q, k, v, scratch, output)
    _validate_prefill(ctx, q, k, v, output)
    var r = Int(q.dim[0]())
    var t = Int(k.dim[0]())
    comptime qk = _grouped_query_attention_qk_apple_gpu_kernel[QL, KL, SL]
    comptime sm = _softmax[SL]
    comptime pv = _grouped_query_attention_pv_apple_gpu_kernel[KL, SL, QL]
    ctx.enqueue_function[qk](
        q,
        k,
        scratch,
        Int32(r),
        Int32(t),
        Int32(14),
        Int32(2),
        Int32(64),
        grid_dim=ceildiv(r * 14 * t, 128),
        block_dim=128,
    )
    ctx.enqueue_function[sm](
        scratch, Int32(r), Int32(t), grid_dim=ceildiv(r * 14, 4), block_dim=128
    )
    ctx.enqueue_function[pv](
        v,
        scratch,
        output,
        Int32(r),
        Int32(t),
        Int32(14),
        Int32(2),
        Int32(64),
        grid_dim=ceildiv(r * 896, 128),
        block_dim=128,
    )


def _validate_prefill[
    QL: TensorLayout, KL: TensorLayout
](
    ctx: DeviceContext,
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
) raises:
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    comptime assert v.flat_rank == 3
    comptime assert output.flat_rank == 3
    if ctx.api() != "metal":
        raise Error("prefill requires Metal")
    var r = Int(q.dim[0]())
    var t = Int(k.dim[0]())
    if r < 1 or r > t or t > 4096:
        raise Error("prefill requires 1 <= R <= T <= 4096")
    if Int(q.dim[1]()) != 14 or Int(q.dim[2]()) != 64:
        raise Error("prefill requires Q[R,14,64]")
    if Int(k.dim[1]()) != 2 or Int(k.dim[2]()) != 64:
        raise Error("prefill requires K[T,2,64]")
    if Int(v.dim[0]()) != t or Int(v.dim[1]()) != 2 or Int(v.dim[2]()) != 64:
        raise Error("prefill value shape must match key")
    if (
        Int(output.dim[0]()) != r
        or Int(output.dim[1]()) != 14
        or Int(output.dim[2]()) != 64
    ):
        raise Error("prefill output shape must match query")


def _stream[
    BQ: Int, BK: Int, SHARED: Bool, QL: TensorLayout, KL: TensorLayout
](
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    rows: Int32,
    tokens: Int32,
):
    comptime assert is_apple_gpu()
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    comptime assert v.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime N = BQ // 4
    # Four SIMD groups own disjoint query rows. Each lane retains dimensions
    # d and d+32, with a separate maximum, denominator and numerator per row.
    var lane = Int(lane_id())
    var tid = thread_idx.x
    var r0 = block_idx.y * BQ + tid // 32
    var h = block_idx.x
    var kh = h // 7
    var past = Int(tokens) - Int(rows)
    var end = min(Int(tokens), past + block_idx.y * BQ + BQ)
    var q0 = SIMD[DType.float32, N](0)
    var q1 = SIMD[DType.float32, N](0)
    var u0 = SIMD[DType.float32, N](0)
    var u1 = SIMD[DType.float32, N](0)
    var m = SIMD[DType.float32, N](neg_inf[DType.float32]())
    var z = SIMD[DType.float32, N](0)
    comptime for i in range(N):
        if r0 + i * 4 < Int(rows):
            q0[i] = q[r0 + i * 4, h, lane].cast[DType.float32]()
            q1[i] = q[r0 + i * 4, h, lane + 32].cast[DType.float32]()
    var ks = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[BK, 64]())
    comptime assert ks.flat_rank == 2
    var vs = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[BK, 64]())
    comptime assert vs.flat_rank == 2
    for base in range(0, end, BK):
        comptime if SHARED:
            for index in range(tid, BK * 64, 128):
                var t = base + index // 64
                var d = index % 64
                var kval: Scalar[DType.bfloat16] = 0
                var vval: Scalar[DType.bfloat16] = 0
                if t < end:
                    kval = rebind[Scalar[DType.bfloat16]](k[t, kh, d])
                    vval = rebind[Scalar[DType.bfloat16]](v[t, kh, d])
                ks[index // 64, d] = kval
                vs[index // 64, d] = vval
            barrier()
        for j in range(min(BK, end - base)):
            var t = base + j
            var k0: Float32
            var k1: Float32
            var v0: Float32
            var v1: Float32
            comptime if SHARED:
                k0 = rebind[Float32](ks[j, lane].cast[DType.float32]())
                k1 = rebind[Float32](ks[j, lane + 32].cast[DType.float32]())
                v0 = rebind[Float32](vs[j, lane].cast[DType.float32]())
                v1 = rebind[Float32](vs[j, lane + 32].cast[DType.float32]())
            else:
                k0 = rebind[Float32](k[t, kh, lane].cast[DType.float32]())
                k1 = rebind[Float32](k[t, kh, lane + 32].cast[DType.float32]())
                v0 = rebind[Float32](v[t, kh, lane].cast[DType.float32]())
                v1 = rebind[Float32](v[t, kh, lane + 32].cast[DType.float32]())
            comptime for i in range(N):
                var r = r0 + i * 4
                if r < Int(rows) and t <= past + r:
                    var score = (
                        (warp.sum(q0[i] * k0 + q1[i] * k1) * 0.125)
                        .cast[DType.bfloat16]()
                        .cast[DType.float32]()
                    )
                    var new_m = max(m[i], score)
                    var alpha = exp(m[i] - new_m)
                    var p = exp(score - new_m)
                    z[i] = z[i] * alpha + p
                    u0[i] = u0[i] * alpha + p * v0
                    u1[i] = u1[i] * alpha + p * v1
                    m[i] = new_m
        comptime if SHARED:
            barrier()
    comptime for i in range(N):
        if r0 + i * 4 < Int(rows):
            output[r0 + i * 4, h, lane] = (u0[i] / z[i]).cast[DType.bfloat16]()
            output[r0 + i * 4, h, lane + 32] = (u1[i] / z[i]).cast[
                DType.bfloat16
            ]()


def _mma[
    BQ: Int, BK: Int, HEADS: Int, QL: TensorLayout, KL: TensorLayout
](
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    rows: Int32,
    tokens: Int32,
):
    comptime assert is_apple_gpu()
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    comptime assert v.flat_rank == 3
    comptime assert output.flat_rank == 3
    comptime W = BQ * HEADS // 8
    # Apple's 8x8 fragment assigns two adjacent columns to each lane. Four
    # lanes share one row; XOR 1 and XOR 8 reduce that row without a block sum.
    var tid = thread_idx.x
    var lane = Int(lane_id())
    var fr = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var fc = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var local_r = (tid // 32) * 8 + fr
    var kh = block_idx.x // ceildiv(7, HEADS)
    var head0 = kh * 7 + (block_idx.x % ceildiv(7, HEADS)) * HEADS
    var h = head0 + local_r // BQ
    # Concatenate HEADS query tiles inside one block. Groups never cross the
    # seven query heads belonging to a KV head; the last group can be partial.
    var r = block_idx.y * BQ + local_r % BQ
    var valid = r < Int(rows) and h < kh * 7 + 7
    var past = Int(tokens) - Int(rows)
    var end = min(Int(tokens), past + block_idx.y * BQ + BQ)
    var ks = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[BK, 64]())
    comptime assert ks.flat_rank == 2
    var vs = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[BK, 64]())
    comptime assert vs.flat_rank == 2
    var scores = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[BQ * HEADS, BK]())
    comptime assert scores.flat_rank == 2
    var probs = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[BQ * HEADS, BK]())
    comptime assert probs.flat_rank == 2
    var u = SIMD[DType.float32, 16](0)
    var m: Float32 = neg_inf[DType.float32]()
    var z: Float32 = 0
    for base in range(0, end, BK):
        for index in range(tid, BK * 64, W * 32):
            var t = base + index // 64
            var d = index % 64
            var kval: Scalar[DType.bfloat16] = 0
            var vval: Scalar[DType.bfloat16] = 0
            if t < end:
                kval = rebind[Scalar[DType.bfloat16]](k[t, kh, d])
                vval = rebind[Scalar[DType.bfloat16]](v[t, kh, d])
            ks[index // 64, d] = kval
            vs[index // 64, d] = vval
        barrier()
        comptime for j in range(BK // 8):
            var acc = SIMD[DType.float32, 2](0)
            comptime for ds in range(8):
                var a = SIMD[DType.bfloat16, 2](0)
                if valid:
                    a[0] = rebind[Scalar[DType.bfloat16]](q[r, h, ds * 8 + fc])
                    a[1] = rebind[Scalar[DType.bfloat16]](
                        q[r, h, ds * 8 + fc + 1]
                    )
                var b = SIMD[DType.bfloat16, 2](0)
                b[0] = rebind[Scalar[DType.bfloat16]](
                    ks[j * 8 + fc, ds * 8 + fr]
                )
                b[1] = rebind[Scalar[DType.bfloat16]](
                    ks[j * 8 + fc + 1, ds * 8 + fr]
                )
                var previous = acc
                _mma_apple_8x8(acc, a, b, previous)
            comptime for c in range(2):
                var t = base + j * 8 + fc + c
                var s: Float32 = neg_inf[DType.float32]()
                if valid and t < end and t <= past + r:
                    s = (
                        (acc[c] * 0.125)
                        .cast[DType.bfloat16]()
                        .cast[DType.float32]()
                    )
                scores[local_r, j * 8 + fc + c] = s
        barrier()
        var tile_m: Float32 = neg_inf[DType.float32]()
        comptime for j in range(BK // 8):
            comptime for c in range(2):
                tile_m = max(
                    tile_m, rebind[Float32](scores[local_r, j * 8 + fc + c])
                )
        tile_m = max(tile_m, warp.shuffle_xor(tile_m, UInt32(1)))
        tile_m = max(tile_m, warp.shuffle_xor(tile_m, UInt32(8)))
        var new_m = max(m, tile_m)
        # Rescale the old unnormalized output once per KV tile. Probabilities
        # round to BF16 for matrix multiplication; m and z remain FP32.
        var alpha = exp(m - new_m)
        var tile_z: Float32 = 0
        comptime for j in range(BK // 8):
            comptime for c in range(2):
                var t = base + j * 8 + fc + c
                var p: Float32 = 0
                if valid and t < end and t <= past + r:
                    p = exp(
                        rebind[Float32](scores[local_r, j * 8 + fc + c]) - new_m
                    )
                tile_z += p
                probs[local_r, j * 8 + fc + c] = p.cast[DType.bfloat16]()
        tile_z += warp.shuffle_xor(tile_z, UInt32(1))
        tile_z += warp.shuffle_xor(tile_z, UInt32(8))
        z = z * alpha + tile_z
        m = new_m
        u *= alpha
        barrier()
        comptime for ds in range(8):
            var acc = SIMD[DType.float32, 2](u[ds * 2], u[ds * 2 + 1])
            comptime for j in range(BK // 8):
                var a = SIMD[DType.bfloat16, 2](0)
                var b = SIMD[DType.bfloat16, 2](0)
                a[0] = rebind[Scalar[DType.bfloat16]](
                    probs[local_r, j * 8 + fc]
                )
                a[1] = rebind[Scalar[DType.bfloat16]](
                    probs[local_r, j * 8 + fc + 1]
                )
                b[0] = rebind[Scalar[DType.bfloat16]](
                    vs[j * 8 + fr, ds * 8 + fc]
                )
                b[1] = rebind[Scalar[DType.bfloat16]](
                    vs[j * 8 + fr, ds * 8 + fc + 1]
                )
                var previous = acc
                _mma_apple_8x8(acc, a, b, previous)
            u[ds * 2] = acc[0]
            u[ds * 2 + 1] = acc[1]
        barrier()
    if valid:
        comptime for ds in range(8):
            output[r, h, ds * 8 + fc] = (u[ds * 2] / z).cast[DType.bfloat16]()
            output[r, h, ds * 8 + fc + 1] = (u[ds * 2 + 1] / z).cast[
                DType.bfloat16
            ]()


# One-head ablations. BQ=32/SPLITS=1 preserves the prior tuned control.
def _mma_tuned[
    SCHEDULE: Int, QL: TensorLayout, KL: TensorLayout, OL: TensorLayout,
    FP32: Bool = False, BQ: Int = 32, SPLITS: Int = 1,
](
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    output: TileTensor[DType.float32 if SPLITS > 1 else DType.bfloat16, OL, MutAnyOrigin],
    rows: Int32,
    tokens: Int32,
):
    comptime assert is_apple_gpu()
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    comptime assert v.flat_rank == 3
    comptime assert output.flat_rank == (4 if SPLITS > 1 else 3)
    comptime assert BQ == 8 or BQ == 16 or BQ == 32
    comptime assert SPLITS == 1 or (FP32 and SCHEDULE == 2 and BQ == 32)
    comptime OTYPE = DType.float32 if SPLITS > 1 else DType.bfloat16
    comptime BK = 32
    comptime HEADS = 1
    comptime W = BQ // 8
    # Apple's 8x8 fragment assigns two adjacent columns to each lane. Four
    # lanes share one row; XOR 1 and XOR 8 reduce that row without a block sum.
    var tid = thread_idx.x
    var lane = Int(lane_id())
    var fr = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var fc = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var local_r = (tid // 32) * 8 + fr
    var kh = block_idx.x // ceildiv(7, HEADS)
    var head0 = kh * 7 + (block_idx.x % ceildiv(7, HEADS)) * HEADS
    var h = head0 + local_r // BQ
    # Concatenate HEADS query tiles inside one block. Groups never cross the
    # seven query heads belonging to a KV head; the last group can be partial.
    var r = block_idx.y * BQ + local_r % BQ
    var valid = r < Int(rows) and h < kh * 7 + 7
    var past = Int(tokens) - Int(rows)
    var end = min(Int(tokens), past + block_idx.y * BQ + BQ)
    var begin = 0
    comptime if SPLITS > 1:
        # Partition whole KV tiles: disjoint coverage, unchanged tile order
        # inside a split, and potentially empty causal pieces for early rows.
        var tiles = ceildiv(Int(tokens), BK)
        begin = (tiles * block_idx.z // SPLITS) * BK
        end = min(end, (tiles * (block_idx.z + 1) // SPLITS) * BK)
    var ks = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[BK, 64]())
    comptime assert ks.flat_rank == 2
    var vs = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[BK, 64]())
    comptime assert vs.flat_rank == 2
    var scores = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[BQ * HEADS, BK]())
    comptime assert scores.flat_rank == 2
    comptime PTYPE = DType.float32 if FP32 else DType.bfloat16
    var probs = stack_allocation[
        PTYPE, address_space=AddressSpace.SHARED
    ](row_major[BQ * HEADS, BK]())
    comptime assert probs.flat_rank == 2
    var u = SIMD[DType.float32, 16](0)
    var fragments = (
        SIMD[DType.float32, 2](0),
        SIMD[DType.float32, 2](0),
        SIMD[DType.float32, 2](0),
        SIMD[DType.float32, 2](0),
        SIMD[DType.float32, 2](0),
        SIMD[DType.float32, 2](0),
        SIMD[DType.float32, 2](0),
        SIMD[DType.float32, 2](0),
    )
    var m: Float32 = neg_inf[DType.float32]()
    var z: Float32 = 0
    for base in range(begin, end, BK):
        var lane_scores = SIMD[DType.float32, 8](0)
        for index in range(tid, BK * 64, W * 32):
            var t = base + index // 64
            var d = index % 64
            var kval: Scalar[DType.bfloat16] = 0
            var vval: Scalar[DType.bfloat16] = 0
            if t < end:
                kval = rebind[Scalar[DType.bfloat16]](k[t, kh, d])
                vval = rebind[Scalar[DType.bfloat16]](v[t, kh, d])
            ks[index // 64, d] = kval
            vs[index // 64, d] = vval
        barrier()
        comptime for j in range(BK // 8):
            var acc = SIMD[DType.float32, 2](0)
            comptime if SCHEDULE == 2:
                for ds in range(8):
                    var a = SIMD[DType.bfloat16, 2](0)
                    if valid:
                        a[0] = rebind[Scalar[DType.bfloat16]](
                            q[r, h, ds * 8 + fc]
                        )
                        a[1] = rebind[Scalar[DType.bfloat16]](
                            q[r, h, ds * 8 + fc + 1]
                        )
                    var b = SIMD[DType.bfloat16, 2](0)
                    b[0] = rebind[Scalar[DType.bfloat16]](
                        ks[j * 8 + fc, ds * 8 + fr]
                    )
                    b[1] = rebind[Scalar[DType.bfloat16]](
                        ks[j * 8 + fc + 1, ds * 8 + fr]
                    )
                    var previous = acc
                    _mma_apple_8x8(acc, a, b, previous)
            else:
                comptime for ds in range(8):
                    var a = SIMD[DType.bfloat16, 2](0)
                    if valid:
                        a[0] = rebind[Scalar[DType.bfloat16]](
                            q[r, h, ds * 8 + fc]
                        )
                        a[1] = rebind[Scalar[DType.bfloat16]](
                            q[r, h, ds * 8 + fc + 1]
                        )
                    var b = SIMD[DType.bfloat16, 2](0)
                    b[0] = rebind[Scalar[DType.bfloat16]](
                        ks[j * 8 + fc, ds * 8 + fr]
                    )
                    b[1] = rebind[Scalar[DType.bfloat16]](
                        ks[j * 8 + fc + 1, ds * 8 + fr]
                    )
                    var previous = acc
                    _mma_apple_8x8(acc, a, b, previous)
            comptime for c in range(2):
                var t = base + j * 8 + fc + c
                var s: Float32 = neg_inf[DType.float32]()
                if valid and t < end and t <= past + r:
                    s = acc[c] * 0.125
                    comptime if not FP32:
                        s = s.cast[DType.bfloat16]().cast[DType.float32]()
                comptime if SCHEDULE == 5:
                    lane_scores[j * 2 + c] = s
                else:
                    scores[local_r, j * 8 + fc + c] = s
        # Each lane reads precisely the score addresses it wrote. There is
        # no cross-thread score-memory handoff; row reduction uses shuffles.
        comptime if SCHEDULE != 3:
            barrier()
        var tile_m: Float32 = neg_inf[DType.float32]()
        comptime for j in range(BK // 8):
            comptime for c in range(2):
                comptime if SCHEDULE == 5:
                    tile_m = max(tile_m, lane_scores[j * 2 + c])
                else:
                    tile_m = max(
                        tile_m, rebind[Float32](scores[local_r, j * 8 + fc + c])
                    )
        tile_m = max(tile_m, warp.shuffle_xor(tile_m, UInt32(1)))
        tile_m = max(tile_m, warp.shuffle_xor(tile_m, UInt32(8)))
        var new_m = max(m, tile_m)
        # Rescale the old unnormalized output once per KV tile. Probabilities
        # retain FP32 for the accuracy path; m and z are always FP32.
        var alpha = exp(m - new_m)
        comptime if SPLITS > 1:
            # An all-masked row has m=new_m=-inf. Its neutral state must
            # survive without exp(-inf - -inf) contaminating z or u.
            if z == 0:
                alpha = 0
        var tile_z: Float32 = 0
        comptime for j in range(BK // 8):
            comptime for c in range(2):
                var t = base + j * 8 + fc + c
                var p: Float32 = 0
                if valid and t < end and t <= past + r:
                    comptime if SCHEDULE == 5:
                        p = exp(lane_scores[j * 2 + c] - new_m)
                    else:
                        p = exp(
                            rebind[Float32](scores[local_r, j * 8 + fc + c])
                            - new_m
                        )
                tile_z += p
                probs[local_r, j * 8 + fc + c] = p.cast[PTYPE]()
        tile_z += warp.shuffle_xor(tile_z, UInt32(1))
        tile_z += warp.shuffle_xor(tile_z, UInt32(8))
        z = z * alpha + tile_z
        m = new_m
        comptime if SCHEDULE == 1:
            comptime for ds in range(8):
                fragments[ds] *= alpha
        else:
            u *= alpha
        # PV loads the same lane-owned probability addresses just written.
        # The MMA collective exchanges loaded register fragments. Shared K/V
        # publication and reuse still require the outer block barriers.
        comptime if SCHEDULE != 4:
            barrier()
        comptime for ds in range(8):
            var acc: SIMD[DType.float32, 2]
            comptime if SCHEDULE == 1:
                acc = fragments[ds]
            else:
                acc = SIMD[DType.float32, 2](u[ds * 2], u[ds * 2 + 1])
            comptime for j in range(BK // 8):
                var a = SIMD[PTYPE, 2](0)
                var b = SIMD[PTYPE, 2](0)
                a[0] = rebind[Scalar[PTYPE]](
                    probs[local_r, j * 8 + fc]
                )
                a[1] = rebind[Scalar[PTYPE]](
                    probs[local_r, j * 8 + fc + 1]
                )
                b[0] = rebind[Scalar[PTYPE]](
                    vs[j * 8 + fr, ds * 8 + fc].cast[PTYPE]()
                )
                b[1] = rebind[Scalar[PTYPE]](
                    vs[j * 8 + fr, ds * 8 + fc + 1].cast[PTYPE]()
                )
                var previous = acc
                _mma_apple_8x8(acc, a, b, previous)
            comptime if SCHEDULE == 1:
                fragments[ds] = acc
            else:
                u[ds * 2] = acc[0]
                u[ds * 2 + 1] = acc[1]
        barrier()
    if valid:
        comptime for ds in range(8):
            comptime if SCHEDULE == 1:
                u[ds * 2] = fragments[ds][0]
                u[ds * 2 + 1] = fragments[ds][1]
            comptime if SPLITS > 1:
                comptime assert output.flat_rank == 4
                output[r, h, block_idx.z, ds * 8 + fc] = u[ds * 2].cast[OTYPE]()
                output[r, h, block_idx.z, ds * 8 + fc + 1] = u[ds * 2 + 1].cast[OTYPE]()
            else:
                comptime assert output.flat_rank == 3
                output[r, h, ds * 8 + fc] = (u[ds * 2] / z).cast[OTYPE]()
                output[r, h, ds * 8 + fc + 1] = (u[ds * 2 + 1] / z).cast[OTYPE]()
        comptime if SPLITS > 1:
            comptime assert output.flat_rank == 4
            if fc == 0:
                output[r, h, block_idx.z, 64] = m.cast[OTYPE]()
                output[r, h, block_idx.z, 65] = z.cast[OTYPE]()


def enqueue_grouped_query_attention_prefill_apple_gpu[
    BQ: Int,
    BK: Int,
    QL: TensorLayout,
    KL: TensorLayout,
    MMA: Bool = False,
    HEADS: Int = 1,
    SHARED: Bool = True,
    SCHEDULE: Int = 0,
    FP32: Bool = False,
](
    ctx: DeviceContext,
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
) raises:
    """Borrow non-overlapping contiguous row-major views; enqueue one dispatch.

    Caller owns allocations and synchronization. No global scratch is used.
    Each block retains its output tile while streaming only visible KV tiles.
    """
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    comptime assert BQ == 4 or BQ == 8 or BQ == 16 or BQ == 32
    comptime assert BK == 32 or BK == 64
    comptime assert HEADS == 1 or HEADS == 2 or HEADS == 4
    comptime assert MMA or HEADS == 1
    comptime assert not MMA or BQ >= 8
    comptime assert 0 <= SCHEDULE <= 5
    comptime assert SCHEDULE == 0 or (
        MMA and BK == 32 and HEADS == 1
        and (BQ == 32 or (FP32 and SCHEDULE == 2))
    )
    comptime assert not FP32 or (MMA and SCHEDULE == 2)
    _validate_prefill(ctx, q, k, v, output)
    var r = Int(q.dim[0]())
    var t = Int(k.dim[0]())
    comptime if MMA:
        comptime kernel = (
            _mma[BQ, BK, HEADS, QL, KL] if SCHEDULE
            == 0 else _mma_tuned[SCHEDULE, QL, KL, QL, FP32, BQ]
        )
        ctx.enqueue_function[kernel](
            q,
            k,
            v,
            output,
            Int32(r),
            Int32(t),
            grid_dim=(2 * ceildiv(7, HEADS), ceildiv(r, BQ)),
            block_dim=BQ * HEADS // 8 * 32,
        )
    else:
        comptime kernel = _stream[BQ, BK, SHARED, QL, KL]
        ctx.enqueue_function[kernel](
            q,
            k,
            v,
            output,
            Int32(r),
            Int32(t),
            grid_dim=(14, ceildiv(r, BQ)),
            block_dim=128,
        )


def _merge_prefill_splits[SPLITS: Int, QL: TensorLayout, PL: TensorLayout](
    partial: TileTensor[DType.float32, PL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    rows: Int32,
):
    comptime assert partial.flat_rank == 4
    comptime assert output.flat_rank == 3
    var row_head = block_idx.x * 4 + thread_idx.x // 32
    var r = row_head // 14
    var h = row_head % 14
    var lane = Int(lane_id())
    if r < Int(rows):
        var m: Float32 = neg_inf[DType.float32]()
        for s in range(SPLITS):
            if partial[r, h, s, 65] > 0:
                m = max(m, rebind[Float32](partial[r, h, s, 64]))
        var z: Float32 = 0
        var u0: Float32 = 0
        var u1: Float32 = 0
        for s in range(SPLITS):
            var mass = rebind[Float32](partial[r, h, s, 65])
            if mass > 0:
                var weight = exp(rebind[Float32](partial[r, h, s, 64]) - m)
                z += weight * mass
                u0 += weight * rebind[Float32](partial[r, h, s, lane])
                u1 += weight * rebind[Float32](partial[r, h, s, lane + 32])
        output[r, h, lane] = (u0 / z).cast[DType.bfloat16]()
        output[r, h, lane + 32] = (u1 / z).cast[DType.bfloat16]()


def enqueue_grouped_query_attention_prefill_split_apple_gpu[
    SPLITS: Int, QL: TensorLayout, KL: TensorLayout, PL: TensorLayout,
](
    ctx: DeviceContext,
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    partial: TileTensor[DType.float32, PL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
) raises:
    """FP32 rolled-MMA partial states followed by one stable merge dispatch.

    Borrow disjoint contiguous row-major views, including caller-owned partial
    storage [R,14,SPLITS,66] = [weighted numerator[64], maximum, denominator].
    Empty pieces write (0,-inf,0). No allocation, initialization or sync here.
    """
    comptime assert SPLITS == 4 or SPLITS == 8
    comptime assert q.flat_rank == 3 and k.flat_rank == 3
    comptime assert partial.flat_rank == 4
    _validate_prefill(ctx, q, k, v, output)
    var r = Int(q.dim[0]())
    var t = Int(k.dim[0]())
    if (Int(partial.dim[0]()) != r or Int(partial.dim[1]()) != 14
        or Int(partial.dim[2]()) != SPLITS or Int(partial.dim[3]()) != 66):
        raise Error("split prefill requires FP32 workspace [R,14,SPLITS,66]")
    comptime kernel = _mma_tuned[2, QL, KL, PL, True, 32, SPLITS]
    ctx.enqueue_function[kernel](
        q, k, v, partial, Int32(r), Int32(t),
        grid_dim=(14, ceildiv(r, 32), SPLITS), block_dim=128,
    )
    comptime merge = _merge_prefill_splits[SPLITS, QL, PL]
    ctx.enqueue_function[merge](
        partial, output, Int32(r), grid_dim=ceildiv(r * 14, 4), block_dim=128,
    )
