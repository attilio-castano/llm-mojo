"""Deterministic development inputs shared by decode tests and benchmarks."""

from layout import TensorLayout, TileTensor
from std.math import isfinite


def decode_input(index: Int, seed: Int, kind: Int, operand: Int) -> Float32:
    var x = (
        Float32(((index * 37 + seed * 19 + (index // 17) * 11) % 257) - 128)
        / 64.0
    )
    if kind == 1 and operand < 2:
        x *= 8.0
    if kind == 2 and operand < 2:
        x = 8.0
    if kind == 3 and operand == 2:
        x = Float32(1 if (index // 128) % 2 == 0 else -1) * (
            1.0 + Float32(index % 7) / 64.0
        )
    return x


def fill_decode[
    QL: TensorLayout,
    KL: TensorLayout,
    VL: TensorLayout,
](
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    seed: Int,
    kind: Int = 0,
):
    comptime assert query.flat_rank == 3
    comptime assert key.flat_rank == 3
    comptime assert value.flat_rank == 3
    for h in range(14):
        for d in range(64):
            query[0, h, d] = decode_input(h * 64 + d, seed, kind, 0).cast[
                DType.bfloat16
            ]()
    for t in range(Int(key.dim[0]())):
        for h in range(2):
            for d in range(64):
                var index = (t * 2 + h) * 64 + d
                key[t, h, d] = decode_input(index, seed + 3, kind, 1).cast[
                    DType.bfloat16
                ]()
                value[t, h, d] = decode_input(index, seed + 7, kind, 2).cast[
                    DType.bfloat16
                ]()


def assert_decode_close[
    OL: TensorLayout
](
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    expected: List[Float32],
) raises -> Float32:
    comptime assert output.flat_rank == 3
    if len(expected) != 896:
        raise Error("decode oracle length must be 896")
    var max_error: Float32 = 0
    for h in range(14):
        for d in range(64):
            var actual = rebind[Float32](output[0, h, d].cast[DType.float32]())
            var target = expected[h * 64 + d]
            var error = abs(actual - target)
            if (
                not isfinite(actual)
                or not isfinite(target)
                or error > 0.015625 + 0.015625 * abs(target)
            ):
                print("decode mismatch", h, d, actual, target, error)
                raise Error("decode exceeded frozen tolerance")
            if error > max_error:
                max_error = error
    return max_error
