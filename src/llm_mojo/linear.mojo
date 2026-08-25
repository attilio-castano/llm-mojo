"""BF16 affine linear projection reference and Apple GPU implementations."""

from layout import TensorLayout, TileTensor
from max.gpu.host import DeviceContext
from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.primitives import warp
from std.math import ceildiv
from std.sys.info import is_apple_gpu


comptime LINEAR_APPLE_GPU_BLOCK_SIZE = 128
comptime LINEAR_APPLE_GPU_SIMD_GROUPS = (
    LINEAR_APPLE_GPU_BLOCK_SIZE // WARP_SIZE
)


def _validate_linear[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Validate the shared host and enqueue contract."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"

    var rows = Int(input.dim[0]())
    var input_features = Int(input.dim[1]())
    var output_features = Int(weight.dim[0]())
    if rows <= 0 or input_features <= 0 or output_features <= 0:
        raise Error("linear dimensions must be positive")
    if Int(weight.dim[1]()) != input_features:
        raise Error("weight input dimension must match input features")
    if Int(bias.dim[0]()) != output_features:
        raise Error("bias length must match output features")
    if Int(output.dim[0]()) != rows or Int(output.dim[1]()) != output_features:
        raise Error("output shape must be (rows, output features)")


def linear_reference[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Apply a source-compatible BF16 affine projection on the host."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    _validate_linear(input, weight, bias, output)

    var rows = Int(input.dim[0]())
    var input_features = Int(input.dim[1]())
    var output_features = Int(weight.dim[0]())
    for row in range(rows):
        for output_feature in range(output_features):
            var accumulator: Scalar[DType.float32] = 0.0
            for input_feature in range(input_features):
                var input_value = rebind[Scalar[DType.bfloat16]](
                    input[row, input_feature]
                )
                var weight_value = rebind[Scalar[DType.bfloat16]](
                    weight[output_feature, input_feature]
                )
                accumulator += (
                    input_value.cast[DType.float32]()
                    * weight_value.cast[DType.float32]()
                )

            var bias_value = rebind[Scalar[DType.bfloat16]](
                bias[output_feature]
            )
            var result = (accumulator + bias_value.cast[DType.float32]()).cast[
                DType.bfloat16
            ]()
            output[row, output_feature] = rebind[output.ElementType](result)


def _linear_rowwise_apple_gpu_kernel[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    rows: Int32,
    input_features: Int32,
    output_features: Int32,
):
    """Map one output dot product to one Apple GPU SIMD group."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    comptime assert (
        LINEAR_APPLE_GPU_BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must contain whole SIMD groups"

    var row_count = Int(rows)
    var input_count = Int(input_features)
    var output_count = Int(output_features)
    var lane = lane_id()
    var simd_group = thread_idx.x // WARP_SIZE
    var dot_product = block_idx.x * LINEAR_APPLE_GPU_SIMD_GROUPS + simd_group
    if dot_product < row_count * output_count:
        var row = dot_product // output_count
        var output_feature = dot_product % output_count
        var accumulator: Scalar[DType.float32] = 0.0
        var input_feature = lane
        while input_feature < input_count:
            var input_value = rebind[Scalar[DType.bfloat16]](
                input[row, input_feature]
            )
            var weight_value = rebind[Scalar[DType.bfloat16]](
                weight[output_feature, input_feature]
            )
            accumulator += (
                input_value.cast[DType.float32]()
                * weight_value.cast[DType.float32]()
            )
            input_feature += WARP_SIZE

        var sum = warp.sum(accumulator)
        if lane == 0:
            var bias_value = rebind[Scalar[DType.bfloat16]](
                bias[output_feature]
            )
            var result = (sum + bias_value.cast[DType.float32]()).cast[
                DType.bfloat16
            ]()
            output[row, output_feature] = rebind[output.ElementType](result)


def enqueue_linear_apple_gpu[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Validate and enqueue the rowwise Apple GPU projection baseline."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    _validate_linear(input, weight, bias, output)
    if context.api() != "metal":
        raise Error("Apple GPU linear projection requires the Metal device API")

    var rows = Int(input.dim[0]())
    var input_features = Int(input.dim[1]())
    var output_features = Int(weight.dim[0]())
    var dot_products = rows * output_features
    comptime kernel = _linear_rowwise_apple_gpu_kernel[
        InputLayout, WeightLayout, BiasLayout, OutputLayout
    ]
    context.enqueue_function[kernel](
        input,
        weight,
        bias,
        output,
        Int32(rows),
        Int32(input_features),
        Int32(output_features),
        grid_dim=ceildiv(dot_products, LINEAR_APPLE_GPU_SIMD_GROUPS),
        block_dim=LINEAR_APPLE_GPU_BLOCK_SIZE,
    )
