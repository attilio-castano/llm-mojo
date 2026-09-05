"""Deterministic prefill inputs and explicit benchmark routing."""
from layout import TensorLayout, TileTensor
from llm_mojo.benchmarks.attention_decode_support import decode_input
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.attention_prefill import (
    enqueue_grouped_query_attention_prefill_apple_gpu,
    enqueue_grouped_query_attention_prefill_materialized_apple_gpu,
)
from max.gpu.host import DeviceContext
from std.math import isfinite


def enqueue_variant[
    QL: TensorLayout, KL: TensorLayout, SL: TensorLayout
](
    variant: Int,
    ctx: DeviceContext,
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, SL, MutAnyOrigin],
) raises -> Int:
    if variant == 0:
        enqueue_grouped_query_attention_apple_gpu(ctx, q, k, v, scratch, output)
        return 0
    elif variant == 1:
        enqueue_grouped_query_attention_prefill_materialized_apple_gpu(
            ctx, q, k, v, scratch, output
        )
        return 1
    elif variant == 2:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            4, 32, MMA=False, HEADS=1, SHARED=False
        ](ctx, q, k, v, output)
        return 2
    elif variant == 3:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            8, 32, MMA=False, HEADS=1, SHARED=True
        ](ctx, q, k, v, output)
        return 3
    elif variant == 4:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            16, 32, MMA=False, HEADS=1, SHARED=True
        ](ctx, q, k, v, output)
        return 4
    elif variant == 5:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            32, 32, MMA=False, HEADS=1, SHARED=True
        ](ctx, q, k, v, output)
        return 5
    elif variant == 6:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            16, 64, MMA=False, HEADS=1, SHARED=True
        ](ctx, q, k, v, output)
        return 6
    elif variant == 7:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            16, 32, MMA=True, HEADS=1, SHARED=True
        ](ctx, q, k, v, output)
        return 7
    elif variant == 8:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            32, 32, MMA=True, HEADS=1, SHARED=True
        ](ctx, q, k, v, output)
        return 8
    elif variant == 9:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            16, 32, MMA=True, HEADS=2, SHARED=True
        ](ctx, q, k, v, output)
        return 9
    elif variant == 10:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            8, 32, MMA=True, HEADS=4, SHARED=True
        ](ctx, q, k, v, output)
        return 10
    raise Error("unknown prefill variant")


def fill_prefill[
    QL: TensorLayout, KL: TensorLayout
](
    q: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    k: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    v: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    seed: Int,
    kind: Int = 0,
):
    comptime assert q.flat_rank == 3
    comptime assert k.flat_rank == 3
    comptime assert v.flat_rank == 3
    var past = Int(k.dim[0]()) - Int(q.dim[0]())
    for r in range(Int(q.dim[0]())):
        for h in range(14):
            for d in range(64):
                q[r, h, d] = decode_input(
                    (past + r) * 896 + h * 64 + d, seed, kind, 0
                ).cast[DType.bfloat16]()
                if kind == 4:
                    q[r, h, d] = Float32(1e16).cast[DType.bfloat16]()
    for t in range(Int(k.dim[0]())):
        for h in range(2):
            for d in range(64):
                var i = t * 128 + h * 64 + d
                k[t, h, d] = decode_input(i, seed + 3, kind, 1).cast[
                    DType.bfloat16
                ]()
                if kind == 4:
                    k[t, h, d] = Float32(-1e16).cast[DType.bfloat16]()
                v[t, h, d] = decode_input(i, seed + 7, kind, 2).cast[
                    DType.bfloat16
                ]()


def assert_prefill_close[
    QL: TensorLayout
](
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    expected: List[Float32],
) raises:
    comptime assert output.flat_rank == 3
    if len(expected) != Int(output.dim[0]()) * 896:
        raise Error("prefill oracle length mismatch")
    for r in range(Int(output.dim[0]())):
        for h in range(14):
            for d in range(64):
                var actual = rebind[Float32](
                    output[r, h, d].cast[DType.float32]()
                )
                var target = expected[r * 896 + h * 64 + d]
                if (
                    not isfinite(actual)
                    or not isfinite(target)
                    or abs(actual - target) > 0.015625 + 0.015625 * abs(target)
                ):
                    print("prefill mismatch", r, h, d, actual, target)
                    raise Error("prefill exceeded frozen attention tolerance")
