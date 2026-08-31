"""BF16 affine linear projection reference and Apple GPU implementations."""

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.primitives import warp
from std.math import ceildiv
from std.sys.info import is_apple_gpu


comptime LINEAR_APPLE_GPU_BLOCK_SIZE = 128
comptime LINEAR_APPLE_GPU_SIMD_GROUPS = (
    LINEAR_APPLE_GPU_BLOCK_SIZE // WARP_SIZE
)
comptime LINEAR_APPLE_GPU_TWO_OUTPUTS_PER_SIMD_GROUP = 2
comptime LINEAR_PREFILL_TILE_ROWS = 8
comptime LINEAR_PREFILL_TILE_OUTPUT_FEATURES = 16
comptime LINEAR_PREFILL_DEFAULT_TILE_INPUT_FEATURES = 32
comptime LINEAR_PREFILL_TILE_OUTPUTS = (
    LINEAR_PREFILL_TILE_ROWS * LINEAR_PREFILL_TILE_OUTPUT_FEATURES
)
comptime LINEAR_PREFILL_REGISTER_TILE_ROWS = 2
comptime LINEAR_PREFILL_REGISTER_TILE_OUTPUT_FEATURES = 2
comptime LINEAR_PREFILL_REGISTER_TILE_OUTPUTS_PER_THREAD = (
    LINEAR_PREFILL_REGISTER_TILE_ROWS
    * LINEAR_PREFILL_REGISTER_TILE_OUTPUT_FEATURES
)
comptime LINEAR_PREFILL_REGISTER_TILE_THREADS = (
    LINEAR_PREFILL_TILE_OUTPUTS
    // LINEAR_PREFILL_REGISTER_TILE_OUTPUTS_PER_THREAD
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


def _linear_prefill_direct_apple_gpu_kernel[
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
    """Map one 8x16 output tile to one threadgroup without shared storage."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    comptime assert (
        LINEAR_PREFILL_TILE_OUTPUTS == LINEAR_APPLE_GPU_BLOCK_SIZE
    ), "one thread must own each tile output"

    var local_output = thread_idx.x
    var local_row = local_output // LINEAR_PREFILL_TILE_OUTPUT_FEATURES
    var local_output_feature = (
        local_output % LINEAR_PREFILL_TILE_OUTPUT_FEATURES
    )
    var row = block_idx.y * LINEAR_PREFILL_TILE_ROWS + local_row
    var output_feature = (
        block_idx.x * LINEAR_PREFILL_TILE_OUTPUT_FEATURES + local_output_feature
    )
    var row_count = Int(rows)
    var input_count = Int(input_features)
    var output_count = Int(output_features)
    if row < row_count and output_feature < output_count:
        var accumulator: Scalar[DType.float32] = 0.0
        for input_feature in range(input_count):
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

        var bias_value = rebind[Scalar[DType.bfloat16]](bias[output_feature])
        var result = (accumulator + bias_value.cast[DType.float32]()).cast[
            DType.bfloat16
        ]()
        output[row, output_feature] = rebind[output.ElementType](result)


def _linear_prefill_register_2x2_apple_gpu_kernel[
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
    """Map one direct 2x2 output microtile to each SIMD-group lane."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    comptime assert (
        LINEAR_PREFILL_TILE_ROWS % LINEAR_PREFILL_REGISTER_TILE_ROWS == 0
    ), "BM must contain whole register tiles"
    comptime assert (
        LINEAR_PREFILL_TILE_OUTPUT_FEATURES
        % LINEAR_PREFILL_REGISTER_TILE_OUTPUT_FEATURES
        == 0
    ), "BN must contain whole register tiles"
    comptime assert (
        LINEAR_PREFILL_REGISTER_TILE_THREADS == WARP_SIZE
    ), "one SIMD group must own each 8x16 output tile"

    var lane = thread_idx.x
    var register_tile_columns = (
        LINEAR_PREFILL_TILE_OUTPUT_FEATURES
        // LINEAR_PREFILL_REGISTER_TILE_OUTPUT_FEATURES
    )
    var local_register_row = lane // register_tile_columns
    var local_register_column = lane % register_tile_columns
    var first_row = (
        block_idx.y * LINEAR_PREFILL_TILE_ROWS
        + local_register_row * LINEAR_PREFILL_REGISTER_TILE_ROWS
    )
    var second_row = first_row + 1
    var first_output_feature = (
        block_idx.x * LINEAR_PREFILL_TILE_OUTPUT_FEATURES
        + local_register_column * LINEAR_PREFILL_REGISTER_TILE_OUTPUT_FEATURES
    )
    var second_output_feature = first_output_feature + 1
    var row_count = Int(rows)
    var input_count = Int(input_features)
    var output_count = Int(output_features)

    if first_row < row_count and first_output_feature < output_count:
        var first_first_accumulator: Scalar[DType.float32] = 0.0
        var first_second_accumulator: Scalar[DType.float32] = 0.0
        var second_first_accumulator: Scalar[DType.float32] = 0.0
        var second_second_accumulator: Scalar[DType.float32] = 0.0

        if second_row < row_count and second_output_feature < output_count:
            for input_feature in range(input_count):
                var first_input = rebind[Scalar[DType.bfloat16]](
                    input[first_row, input_feature]
                ).cast[DType.float32]()
                var second_input = rebind[Scalar[DType.bfloat16]](
                    input[second_row, input_feature]
                ).cast[DType.float32]()
                var first_weight = rebind[Scalar[DType.bfloat16]](
                    weight[first_output_feature, input_feature]
                ).cast[DType.float32]()
                var second_weight = rebind[Scalar[DType.bfloat16]](
                    weight[second_output_feature, input_feature]
                ).cast[DType.float32]()
                first_first_accumulator += first_input * first_weight
                first_second_accumulator += first_input * second_weight
                second_first_accumulator += second_input * first_weight
                second_second_accumulator += second_input * second_weight
        else:
            var has_second_row = second_row < row_count
            var has_second_output = second_output_feature < output_count
            for input_feature in range(input_count):
                var first_input = rebind[Scalar[DType.bfloat16]](
                    input[first_row, input_feature]
                ).cast[DType.float32]()
                var first_weight = rebind[Scalar[DType.bfloat16]](
                    weight[first_output_feature, input_feature]
                ).cast[DType.float32]()
                first_first_accumulator += first_input * first_weight

                var second_input: Scalar[DType.float32] = 0.0
                if has_second_row:
                    second_input = rebind[Scalar[DType.bfloat16]](
                        input[second_row, input_feature]
                    ).cast[DType.float32]()
                    second_first_accumulator += second_input * first_weight

                if has_second_output:
                    var second_weight = rebind[Scalar[DType.bfloat16]](
                        weight[second_output_feature, input_feature]
                    ).cast[DType.float32]()
                    first_second_accumulator += first_input * second_weight
                    if has_second_row:
                        second_second_accumulator += (
                            second_input * second_weight
                        )

        var first_bias = rebind[Scalar[DType.bfloat16]](
            bias[first_output_feature]
        ).cast[DType.float32]()
        var first_first_result = (first_first_accumulator + first_bias).cast[
            DType.bfloat16
        ]()
        output[first_row, first_output_feature] = rebind[output.ElementType](
            first_first_result
        )

        if second_row < row_count:
            var second_first_result = (
                second_first_accumulator + first_bias
            ).cast[DType.bfloat16]()
            output[second_row, first_output_feature] = rebind[
                output.ElementType
            ](second_first_result)

        if second_output_feature < output_count:
            var second_bias = rebind[Scalar[DType.bfloat16]](
                bias[second_output_feature]
            ).cast[DType.float32]()
            var first_second_result = (
                first_second_accumulator + second_bias
            ).cast[DType.bfloat16]()
            output[first_row, second_output_feature] = rebind[
                output.ElementType
            ](first_second_result)
            if second_row < row_count:
                var second_second_result = (
                    second_second_accumulator + second_bias
                ).cast[DType.bfloat16]()
                output[second_row, second_output_feature] = rebind[
                    output.ElementType
                ](second_second_result)


def _linear_prefill_tiled_apple_gpu_kernel[
    tile_input_features: Int,
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
    """Stage 8xBK input and 16xBK weight tiles for one 8x16 output tile."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert tile_input_features > 0, "BK must be positive"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    comptime assert (
        LINEAR_PREFILL_TILE_OUTPUTS == LINEAR_APPLE_GPU_BLOCK_SIZE
    ), "one thread must own each tile output"

    var local_output = thread_idx.x
    var local_row = local_output // LINEAR_PREFILL_TILE_OUTPUT_FEATURES
    var local_output_feature = (
        local_output % LINEAR_PREFILL_TILE_OUTPUT_FEATURES
    )
    var row = block_idx.y * LINEAR_PREFILL_TILE_ROWS + local_row
    var output_feature = (
        block_idx.x * LINEAR_PREFILL_TILE_OUTPUT_FEATURES + local_output_feature
    )
    var row_count = Int(rows)
    var input_count = Int(input_features)
    var output_count = Int(output_features)
    var input_tile = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](row_major[LINEAR_PREFILL_TILE_ROWS, tile_input_features]())
    var weight_tile = stack_allocation[
        DType.bfloat16, address_space=AddressSpace.SHARED
    ](
        row_major[
            LINEAR_PREFILL_TILE_OUTPUT_FEATURES,
            tile_input_features,
        ]()
    )
    comptime assert input_tile.flat_rank == 2
    comptime assert weight_tile.flat_rank == 2

    var accumulator: Scalar[DType.float32] = 0.0
    var input_tile_values = LINEAR_PREFILL_TILE_ROWS * tile_input_features
    var weight_tile_values = (
        LINEAR_PREFILL_TILE_OUTPUT_FEATURES * tile_input_features
    )
    var input_tile_start = 0
    while input_tile_start < input_count:
        var load_index = local_output
        while load_index < input_tile_values:
            var load_row = load_index // tile_input_features
            var load_input_feature = load_index % tile_input_features
            var global_row = block_idx.y * LINEAR_PREFILL_TILE_ROWS + load_row
            var global_input_feature = input_tile_start + load_input_feature
            var input_value: Scalar[DType.bfloat16] = 0.0
            if global_row < row_count and global_input_feature < input_count:
                input_value = rebind[Scalar[DType.bfloat16]](
                    input[global_row, global_input_feature]
                )
            input_tile[load_row, load_input_feature] = rebind[
                input_tile.ElementType
            ](input_value)
            load_index += LINEAR_APPLE_GPU_BLOCK_SIZE

        load_index = local_output
        while load_index < weight_tile_values:
            var load_output_feature = load_index // tile_input_features
            var load_input_feature = load_index % tile_input_features
            var global_output_feature = (
                block_idx.x * LINEAR_PREFILL_TILE_OUTPUT_FEATURES
                + load_output_feature
            )
            var global_input_feature = input_tile_start + load_input_feature
            var weight_value: Scalar[DType.bfloat16] = 0.0
            if (
                global_output_feature < output_count
                and global_input_feature < input_count
            ):
                weight_value = rebind[Scalar[DType.bfloat16]](
                    weight[global_output_feature, global_input_feature]
                )
            weight_tile[load_output_feature, load_input_feature] = rebind[
                weight_tile.ElementType
            ](weight_value)
            load_index += LINEAR_APPLE_GPU_BLOCK_SIZE

        barrier()
        if row < row_count and output_feature < output_count:
            for local_input_feature in range(tile_input_features):
                var input_value = rebind[Scalar[DType.bfloat16]](
                    input_tile[local_row, local_input_feature]
                )
                var weight_value = rebind[Scalar[DType.bfloat16]](
                    weight_tile[local_output_feature, local_input_feature]
                )
                accumulator += (
                    input_value.cast[DType.float32]()
                    * weight_value.cast[DType.float32]()
                )
        barrier()
        input_tile_start += tile_input_features

    if row < row_count and output_feature < output_count:
        var bias_value = rebind[Scalar[DType.bfloat16]](bias[output_feature])
        var result = (accumulator + bias_value.cast[DType.float32]()).cast[
            DType.bfloat16
        ]()
        output[row, output_feature] = rebind[output.ElementType](result)


def _linear_two_output_apple_gpu_kernel[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    input_features: Int32,
    output_features: Int32,
):
    """Map two adjacent M=1 output dot products to one SIMD group."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    comptime assert (
        LINEAR_APPLE_GPU_BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must contain whole SIMD groups"

    var input_count = Int(input_features)
    var output_count = Int(output_features)
    var lane = lane_id()
    var simd_group = thread_idx.x // WARP_SIZE
    var output_pair = block_idx.x * LINEAR_APPLE_GPU_SIMD_GROUPS + simd_group
    var first_output_feature = (
        output_pair * LINEAR_APPLE_GPU_TWO_OUTPUTS_PER_SIMD_GROUP
    )
    if first_output_feature < output_count:
        var second_output_feature = first_output_feature + 1
        var first_accumulator: Scalar[DType.float32] = 0.0
        var second_accumulator: Scalar[DType.float32] = 0.0
        var input_feature = lane
        while input_feature < input_count:
            var input_value = rebind[Scalar[DType.bfloat16]](
                input[0, input_feature]
            ).cast[DType.float32]()
            var first_weight = rebind[Scalar[DType.bfloat16]](
                weight[first_output_feature, input_feature]
            ).cast[DType.float32]()
            first_accumulator += input_value * first_weight
            if second_output_feature < output_count:
                var second_weight = rebind[Scalar[DType.bfloat16]](
                    weight[second_output_feature, input_feature]
                ).cast[DType.float32]()
                second_accumulator += input_value * second_weight
            input_feature += WARP_SIZE

        var first_sum = warp.sum(first_accumulator)
        var second_sum = warp.sum(second_accumulator)
        if lane == 0:
            var first_bias = rebind[Scalar[DType.bfloat16]](
                bias[first_output_feature]
            ).cast[DType.float32]()
            var first_result = (first_sum + first_bias).cast[DType.bfloat16]()
            output[0, first_output_feature] = rebind[output.ElementType](
                first_result
            )
            if second_output_feature < output_count:
                var second_bias = rebind[Scalar[DType.bfloat16]](
                    bias[second_output_feature]
                ).cast[DType.float32]()
                var second_result = (second_sum + second_bias).cast[
                    DType.bfloat16
                ]()
                output[0, second_output_feature] = rebind[output.ElementType](
                    second_result
                )


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


def enqueue_linear_prefill_direct_apple_gpu[
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
    """Enqueue the direct 8x16 output-ownership control for prefill."""

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
    comptime kernel = _linear_prefill_direct_apple_gpu_kernel[
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
        grid_dim=(
            ceildiv(output_features, LINEAR_PREFILL_TILE_OUTPUT_FEATURES),
            ceildiv(rows, LINEAR_PREFILL_TILE_ROWS),
        ),
        block_dim=LINEAR_APPLE_GPU_BLOCK_SIZE,
    )


def enqueue_linear_prefill_register_2x2_apple_gpu[
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
    """Enqueue the direct 2x2 register-tiled prefill candidate."""

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
    comptime kernel = _linear_prefill_register_2x2_apple_gpu_kernel[
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
        grid_dim=(
            ceildiv(output_features, LINEAR_PREFILL_TILE_OUTPUT_FEATURES),
            ceildiv(rows, LINEAR_PREFILL_TILE_ROWS),
        ),
        block_dim=LINEAR_PREFILL_REGISTER_TILE_THREADS,
    )


def enqueue_linear_prefill_tiled_apple_gpu_bk[
    tile_input_features: Int,
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
    """Enqueue a shared-memory 8x16xBK prefill experiment."""

    comptime assert (
        tile_input_features == 16
        or tile_input_features == 32
        or tile_input_features == 64
        or tile_input_features == 128
    ), "BK must be one of 16, 32, 64, or 128"
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
    comptime kernel = _linear_prefill_tiled_apple_gpu_kernel[
        tile_input_features, InputLayout, WeightLayout, BiasLayout, OutputLayout
    ]
    context.enqueue_function[kernel](
        input,
        weight,
        bias,
        output,
        Int32(rows),
        Int32(input_features),
        Int32(output_features),
        grid_dim=(
            ceildiv(output_features, LINEAR_PREFILL_TILE_OUTPUT_FEATURES),
            ceildiv(rows, LINEAR_PREFILL_TILE_ROWS),
        ),
        block_dim=LINEAR_APPLE_GPU_BLOCK_SIZE,
    )


def enqueue_linear_prefill_tiled_apple_gpu[
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
    """Enqueue the existing shared-memory 8x16x32 prefill candidate."""

    enqueue_linear_prefill_tiled_apple_gpu_bk[
        LINEAR_PREFILL_DEFAULT_TILE_INPUT_FEATURES
    ](context, input, weight, bias, output)


def enqueue_linear_apple_gpu_two_output[
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
    """Validate and enqueue the explicit M=1 two-output candidate."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    _validate_linear(input, weight, bias, output)
    if context.api() != "metal":
        raise Error("Apple GPU linear projection requires the Metal device API")

    var rows = Int(input.dim[0]())
    if rows != 1:
        raise Error("two-output Apple GPU projection requires M=1")
    var input_features = Int(input.dim[1]())
    var output_features = Int(weight.dim[0]())
    var output_pairs = ceildiv(
        output_features, LINEAR_APPLE_GPU_TWO_OUTPUTS_PER_SIMD_GROUP
    )
    comptime kernel = _linear_two_output_apple_gpu_kernel[
        InputLayout, WeightLayout, BiasLayout, OutputLayout
    ]
    context.enqueue_function[kernel](
        input,
        weight,
        bias,
        output,
        Int32(input_features),
        Int32(output_features),
        grid_dim=ceildiv(output_pairs, LINEAR_APPLE_GPU_SIMD_GROUPS),
        block_dim=LINEAR_APPLE_GPU_BLOCK_SIZE,
    )
