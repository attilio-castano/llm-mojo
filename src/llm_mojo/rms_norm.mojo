"""Qwen RMSNorm reference implementation."""

from layout import TensorLayout, TileTensor
from std.math import rsqrt


comptime RMS_NORM_EPSILON: Float32 = 1.0e-6


def rms_norm_reference[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Apply the Qwen2 BF16 RMSNorm contract row by row on the host."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 1, "weight must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"

    var rows = Int(input.dim[0]())
    var hidden_size = Int(input.dim[1]())
    if Int(output.dim[0]()) != rows or Int(output.dim[1]()) != hidden_size:
        raise Error("output shape must match input shape")
    if Int(weight.dim[0]()) != hidden_size:
        raise Error("weight length must match the hidden dimension")

    for row in range(rows):
        var sum_of_squares: Scalar[DType.float32] = 0.0
        for hidden in range(hidden_size):
            var value = rebind[Scalar[DType.bfloat16]](input[row, hidden])
            var value_f32 = value.cast[DType.float32]()
            sum_of_squares += value_f32 * value_f32

        var mean_square = sum_of_squares / Float32(hidden_size)
        var inverse_rms = rsqrt(mean_square + RMS_NORM_EPSILON)

        for hidden in range(hidden_size):
            var value = rebind[Scalar[DType.bfloat16]](input[row, hidden])
            var normalized = (value.cast[DType.float32]() * inverse_rms).cast[
                DType.bfloat16
            ]()
            var scale = rebind[Scalar[DType.bfloat16]](weight[hidden])
            output[row, hidden] = rebind[output.ElementType](normalized * scale)
