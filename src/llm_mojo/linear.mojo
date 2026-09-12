"""BF16 affine linear projection reference and Apple GPU implementations."""

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
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
comptime LINEAR_PREFILL_MMA_DIM = 8
comptime LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS = 2


def _validate_linear[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
    HAS_BIAS: Bool = True,
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
    if HAS_BIAS and Int(bias.dim[0]()) != output_features:
        raise Error("bias length must match output features")
    if Int(output.dim[0]()) != rows or Int(output.dim[1]()) != output_features:
        raise Error("output shape must be (rows, output features)")


def linear_reference[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
    HAS_BIAS: Bool = True,
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
    _validate_linear[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
    ](input, weight, bias, output)

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

            var bias_value: Scalar[DType.bfloat16] = 0
            comptime if HAS_BIAS:
                bias_value = rebind[Scalar[DType.bfloat16]](
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
    HAS_BIAS: Bool = True,
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
            var bias_value: Scalar[DType.bfloat16] = 0
            comptime if HAS_BIAS:
                bias_value = rebind[Scalar[DType.bfloat16]](
                    bias[output_feature]
                )
            var result = (sum + bias_value.cast[DType.float32]()).cast[
                DType.bfloat16
            ]()
            output[row, output_feature] = rebind[output.ElementType](result)


def _linear_rowwise_rows_apple_gpu_kernel[
    ROW_TILE: Int, IL: TensorLayout, WL: TensorLayout, BL: TensorLayout,
    OL: TensorLayout, HAS_BIAS: Bool,
](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    rows: Int32, inputs: Int32, outputs: Int32,
):
    """One SIMD group reuses a weight across several independent row dots.

    Each row keeps the original lane-strided FP32 accumulator and warp.sum
    order. The row tile changes weight reuse, not the K reduction partition.
    """
    comptime assert is_apple_gpu()
    comptime assert input.flat_rank == 2 and weight.flat_rank == 2
    comptime assert bias.flat_rank == 1 and output.flat_rank == 2
    var r = Int(rows)
    var k = Int(inputs)
    var n = Int(outputs)
    var lane = lane_id()
    var group = thread_idx.x // WARP_SIZE
    var product = block_idx.x * LINEAR_APPLE_GPU_SIMD_GROUPS + group
    var row = (product // n) * ROW_TILE
    var column = product % n
    if row < r:
        var accumulators = SIMD[DType.float32, ROW_TILE](0)
        var feature = lane
        while feature < k:
            var w = rebind[Scalar[DType.bfloat16]](weight[column,feature])
            comptime for j in range(ROW_TILE):
                if row+j < r:
                    var x = rebind[Scalar[DType.bfloat16]](input[row+j,feature])
                    accumulators[j] += x.cast[DType.float32]() * w.cast[DType.float32]()
            feature += WARP_SIZE
        comptime for j in range(ROW_TILE):
            if row+j < r:
                var total = warp.sum(accumulators[j])
                if lane == 0:
                    var b: Scalar[DType.bfloat16] = 0
                    comptime if HAS_BIAS:
                        b = rebind[Scalar[DType.bfloat16]](bias[column])
                    output[row+j,column] = rebind[output.ElementType](
                        (total+b.cast[DType.float32]()).cast[DType.bfloat16]())


def enqueue_linear_rowwise_rows_apple_gpu[
    ROW_TILE: Int, IL: TensorLayout, WL: TensorLayout, BL: TensorLayout,
    OL: TensorLayout, HAS_BIAS: Bool = True,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    comptime assert ROW_TILE == 4 or ROW_TILE == 8 or ROW_TILE == 16
    var rows = Int(input.dim[0]())
    if rows == 1:
        enqueue_linear_apple_gpu[IL,WL,BL,OL,HAS_BIAS](context,input,weight,bias,output)
        return
    _validate_linear[IL,WL,BL,OL,HAS_BIAS](input,weight,bias,output)
    if context.api() != "metal":
        raise Error("Apple GPU linear projection requires the Metal device API")
    var inputs = Int(input.dim[1]())
    var outputs = Int(weight.dim[0]())
    comptime kernel = _linear_rowwise_rows_apple_gpu_kernel[ROW_TILE,IL,WL,BL,OL,HAS_BIAS]
    context.enqueue_function[kernel](input,weight,bias,output,
        Int32(rows),Int32(inputs),Int32(outputs),
        grid_dim=ceildiv(ceildiv(rows,ROW_TILE)*outputs,LINEAR_APPLE_GPU_SIMD_GROUPS),
        block_dim=LINEAR_APPLE_GPU_BLOCK_SIZE)


def enqueue_linear_rowwise_rows_apple_gpu[
    ROW_TILE: Int, IL: TensorLayout, WL: TensorLayout, OL: TensorLayout,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    var bias = TileTensor(weight.ptr,row_major(1))
    enqueue_linear_rowwise_rows_apple_gpu[ROW_TILE,IL,WL,type_of(bias.layout),OL,False](
        context,input,weight,bias,output)


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
    HAS_BIAS: Bool = True,
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

        var first_bias: Float32 = 0
        comptime if HAS_BIAS:
            first_bias = rebind[Scalar[DType.bfloat16]](
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
            var second_bias: Float32 = 0
            comptime if HAS_BIAS:
                second_bias = rebind[Scalar[DType.bfloat16]](
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


def _linear_prefill_mma_8x16_apple_gpu_kernel[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
    HAS_BIAS: Bool = True,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    rows: Int32,
    input_features: Int32,
    output_features: Int32,
):
    """Map one 8x16 output tile to one Apple 8x8 MMA SIMD group."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    comptime assert (
        LINEAR_PREFILL_TILE_ROWS == LINEAR_PREFILL_MMA_DIM
    ), "BM must equal the 8x8 MMA row dimension"
    comptime assert (
        LINEAR_PREFILL_TILE_OUTPUT_FEATURES == 2 * LINEAR_PREFILL_MMA_DIM
    ), "BN must contain two adjacent 8x8 MMA tiles"
    comptime assert (
        WARP_SIZE * LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS
        == LINEAR_PREFILL_MMA_DIM * LINEAR_PREFILL_MMA_DIM
    ), "one SIMD group must collectively own each 8x8 fragment"

    var lane = Int(lane_id())
    # Apple distributes each 8x8 fragment as two adjacent columns in one row
    # per lane. This matches apple_mma_load_8x8/thread_elements().
    var fragment_row = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var fragment_column = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var row = block_idx.y * LINEAR_PREFILL_TILE_ROWS + fragment_row
    var first_output = (
        block_idx.x * LINEAR_PREFILL_TILE_OUTPUT_FEATURES + fragment_column
    )
    var second_output = first_output + LINEAR_PREFILL_MMA_DIM
    var row_count = Int(rows)
    var input_count = Int(input_features)
    var output_count = Int(output_features)
    var first_accumulator = SIMD[
        DType.float32, LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS
    ](0)
    var second_accumulator = SIMD[
        DType.float32, LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS
    ](0)

    var input_tile_start = 0
    while input_tile_start < input_count:
        var input_fragment = SIMD[
            DType.bfloat16, LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS
        ](0)
        var first_weight_fragment = SIMD[
            DType.bfloat16, LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS
        ](0)
        var second_weight_fragment = SIMD[
            DType.bfloat16, LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS
        ](0)

        comptime for element in range(LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS):
            var input_feature = input_tile_start + fragment_column + element
            if row < row_count and input_feature < input_count:
                input_fragment[element] = rebind[Scalar[DType.bfloat16]](
                    input[row, input_feature]
                )

            var weight_input_feature = input_tile_start + fragment_row
            var first_weight_output = first_output + element
            if (
                weight_input_feature < input_count
                and first_weight_output < output_count
            ):
                first_weight_fragment[element] = rebind[Scalar[DType.bfloat16]](
                    weight[first_weight_output, weight_input_feature]
                )

            var second_weight_output = second_output + element
            if (
                weight_input_feature < input_count
                and second_weight_output < output_count
            ):
                second_weight_fragment[element] = rebind[
                    Scalar[DType.bfloat16]
                ](weight[second_weight_output, weight_input_feature])

        var previous_first = first_accumulator
        var previous_second = second_accumulator
        _mma_apple_8x8(
            first_accumulator,
            input_fragment,
            first_weight_fragment,
            previous_first,
        )
        _mma_apple_8x8(
            second_accumulator,
            input_fragment,
            second_weight_fragment,
            previous_second,
        )
        input_tile_start += LINEAR_PREFILL_MMA_DIM

    comptime for element in range(LINEAR_PREFILL_MMA_FRAGMENT_ELEMENTS):
        var first_output_feature = first_output + element
        if row < row_count and first_output_feature < output_count:
            var first_bias: Float32 = 0
            comptime if HAS_BIAS:
                first_bias = rebind[Scalar[DType.bfloat16]](
                    bias[first_output_feature]
                ).cast[DType.float32]()
            var first_result = (first_accumulator[element] + first_bias).cast[
                DType.bfloat16
            ]()
            output[row, first_output_feature] = rebind[output.ElementType](
                first_result
            )

        var second_output_feature = second_output + element
        if row < row_count and second_output_feature < output_count:
            var second_bias: Float32 = 0
            comptime if HAS_BIAS:
                second_bias = rebind[Scalar[DType.bfloat16]](
                    bias[second_output_feature]
                ).cast[DType.float32]()
            var second_result = (
                second_accumulator[element] + second_bias
            ).cast[DType.bfloat16]()
            output[row, second_output_feature] = rebind[output.ElementType](
                second_result
            )


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
    HAS_BIAS: Bool = True,
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
            var first_bias: Float32 = 0
            comptime if HAS_BIAS:
                first_bias = rebind[Scalar[DType.bfloat16]](
                    bias[first_output_feature]
                ).cast[DType.float32]()
            var first_result = (first_sum + first_bias).cast[DType.bfloat16]()
            output[0, first_output_feature] = rebind[output.ElementType](
                first_result
            )
            if second_output_feature < output_count:
                var second_bias: Float32 = 0
                comptime if HAS_BIAS:
                    second_bias = rebind[Scalar[DType.bfloat16]](
                        bias[second_output_feature]
                    ).cast[DType.float32]()
                var second_result = (second_sum + second_bias).cast[
                    DType.bfloat16
                ]()
                output[0, second_output_feature] = rebind[output.ElementType](
                    second_result
                )


def _linear_decode_arranged[
    BLOCK: Int, WIDTH: Int, IL: TensorLayout, WL: TensorLayout,
    BL: TensorLayout, OL: TensorLayout, HAS_BIAS: Bool,
](
    x: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    w: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    rows: Int32, inputs: Int32, outputs: Int32,
):
    comptime assert is_apple_gpu()
    comptime assert x.flat_rank == 2 and w.flat_rank == 2 and y.flat_rank == 2 and bias.flat_rank == 1
    comptime assert BLOCK == 64 or BLOCK == 128 or BLOCK == 256
    comptime assert WIDTH == 0 or WIDTH == 896 or WIDTH == 4864
    var lane = lane_id()
    var dot = block_idx.x * (BLOCK // WARP_SIZE) + thread_idx.x // WARP_SIZE
    if dot < Int(rows) * Int(outputs):
        var row = dot // Int(outputs)
        var column = dot % Int(outputs)
        var acc: Float32 = 0
        comptime if WIDTH:
            # Prefetch four lane-strided operands, then retain the original
            # sequential FP32 updates and SIMD-group reduction order.
            for base in range(0, WIDTH, 4 * WARP_SIZE):
                var xv = SIMD[DType.float32, 4](0)
                var wv = SIMD[DType.float32, 4](0)
                comptime for j in range(4):
                    var k = base + lane + j * WARP_SIZE
                    xv[j] = rebind[Scalar[DType.bfloat16]](x[row,k]).cast[DType.float32]()
                    wv[j] = rebind[Scalar[DType.bfloat16]](w[column,k]).cast[DType.float32]()
                comptime for j in range(4):
                    acc += xv[j] * wv[j]
        else:
            var k = lane
            while k < Int(inputs):
                acc += (rebind[Scalar[DType.bfloat16]](x[row,k]).cast[DType.float32]()
                        * rebind[Scalar[DType.bfloat16]](w[column,k]).cast[DType.float32]())
                k += WARP_SIZE
        var total = warp.sum(acc)
        if lane == 0:
            var b: Scalar[DType.bfloat16] = 0
            comptime if HAS_BIAS:
                b = rebind[Scalar[DType.bfloat16]](bias[column])
            y[row,column] = rebind[y.ElementType]((total + b.cast[DType.float32]()).cast[DType.bfloat16]())


def _enqueue_linear_arranged[
    BLOCK: Int, WIDTH: Int, IL: TensorLayout, WL: TensorLayout,
    BL: TensorLayout, OL: TensorLayout, HAS_BIAS: Bool,
](ctx: DeviceContext, x: TileTensor[DType.bfloat16,IL,MutAnyOrigin],
  w: TileTensor[DType.bfloat16,WL,MutAnyOrigin], b: TileTensor[DType.bfloat16,BL,MutAnyOrigin],
  y: TileTensor[DType.bfloat16,OL,MutAnyOrigin]) raises:
    comptime kernel = _linear_decode_arranged[BLOCK,WIDTH,IL,WL,BL,OL,HAS_BIAS]
    ctx.enqueue_function[kernel](x,w,b,y,Int32(x.dim[0]()),Int32(x.dim[1]()),Int32(w.dim[0]()),
        grid_dim=ceildiv(Int(w.dim[0]()),BLOCK // WARP_SIZE),block_dim=BLOCK)


def enqueue_linear_apple_gpu[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
    HAS_BIAS: Bool = True,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    decode_variant: Int = 0,
) raises:
    """Validate and enqueue the rowwise Apple GPU projection baseline."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    _validate_linear[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
    ](input, weight, bias, output)
    if context.api() != "metal":
        raise Error("Apple GPU linear projection requires the Metal device API")

    if decode_variant < 0 or decode_variant > 5:
        raise Error("unknown projection arrangement")
    if decode_variant:
        if Int(input.dim[0]()) != 1 or (Int(input.dim[1]()) != 896 and Int(input.dim[1]()) != 4864):
            raise Error("projection arrangement requires one row and width 896 or 4864")
        comptime for variant in range(1,6):
            if decode_variant == variant:
                comptime block = 64 if variant == 2 or variant == 4 else (256 if variant == 3 or variant == 5 else 128)
                comptime if variant == 2 or variant == 3:
                    _enqueue_linear_arranged[block,0,InputLayout,WeightLayout,BiasLayout,OutputLayout,HAS_BIAS](context,input,weight,bias,output)
                else:
                    if Int(input.dim[1]()) == 896:
                        _enqueue_linear_arranged[block,896,InputLayout,WeightLayout,BiasLayout,OutputLayout,HAS_BIAS](context,input,weight,bias,output)
                    else:
                        _enqueue_linear_arranged[block,4864,InputLayout,WeightLayout,BiasLayout,OutputLayout,HAS_BIAS](context,input,weight,bias,output)
        return
    var rows = Int(input.dim[0]())
    var input_features = Int(input.dim[1]())
    var output_features = Int(weight.dim[0]())
    var dot_products = rows * output_features
    comptime kernel = _linear_rowwise_apple_gpu_kernel[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
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
    HAS_BIAS: Bool = True,
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
    _validate_linear[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
    ](input, weight, bias, output)
    if context.api() != "metal":
        raise Error("Apple GPU linear projection requires the Metal device API")

    var rows = Int(input.dim[0]())
    var input_features = Int(input.dim[1]())
    var output_features = Int(weight.dim[0]())
    comptime kernel = _linear_prefill_register_2x2_apple_gpu_kernel[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
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


def enqueue_linear_prefill_mma_8x16_apple_gpu[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    OutputLayout: TensorLayout,
    HAS_BIAS: Bool = True,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Enqueue the experimental Apple 8x8-MMA 8x16 linear mapping."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 2, "weight must have rank 2"
    comptime assert bias.flat_rank == 1, "bias must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    _validate_linear[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
    ](input, weight, bias, output)
    if context.api() != "metal":
        raise Error("Apple GPU linear projection requires the Metal device API")

    var rows = Int(input.dim[0]())
    var input_features = Int(input.dim[1]())
    var output_features = Int(weight.dim[0]())
    comptime kernel = _linear_prefill_mma_8x16_apple_gpu_kernel[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
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
        block_dim=WARP_SIZE,
    )


@always_inline
def _linear_mma_fragment[
    TRANSPOSE: Bool, L: TensorLayout
](
    tensor: TileTensor[DType.bfloat16, L, MutAnyOrigin],
    row: Int,
    column: Int,
    rows: Int,
    columns: Int,
) -> SIMD[DType.bfloat16, 2]:
    comptime assert tensor.flat_rank == 2
    var fragment = SIMD[DType.bfloat16, 2](0)
    comptime for e in range(2):
        if row < rows and column + e < columns:
            comptime if TRANSPOSE:
                fragment[e] = rebind[Scalar[DType.bfloat16]](
                    tensor[column + e, row]
                )
            else:
                fragment[e] = rebind[Scalar[DType.bfloat16]](
                    tensor[row, column + e]
                )
    return fragment


@always_inline
def _linear_mma_store[
    HAS_BIAS: Bool, BL: TensorLayout, OL: TensorLayout
](
    bias: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    accumulator: SIMD[DType.float32, 2],
    row: Int,
    column: Int,
    rows: Int,
    columns: Int,
):
    comptime assert bias.flat_rank == 1 and output.flat_rank == 2
    comptime for e in range(2):
        if row < rows and column + e < columns:
            var b: Float32 = 0
            comptime if HAS_BIAS:
                b = rebind[Scalar[DType.bfloat16]](bias[column + e]).cast[
                    DType.float32
                ]()
            var value = (accumulator[e] + b).cast[DType.bfloat16]()
            output[row, column + e] = rebind[output.ElementType](value)


def _linear_mma_tile[
    BM: Int,
    BN: Int,
    IL: TensorLayout,
    WL: TensorLayout,
    BL: TensorLayout,
    OL: TensorLayout,
    HAS_BIAS: Bool,
](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    row_count: Int32,
    input_count: Int32,
    output_count: Int32,
):
    """One SIMD group owns four 8x8 fragments: eight FP32 values per lane.

    16x16 reuses each weight fragment across two input row fragments;
    8x32 reuses one input fragment across four weight fragments. No shared
    storage, barriers, K splitting, or additional intermediate rounding.
    """
    comptime assert is_apple_gpu()
    comptime assert (BM == 16 and BN == 16) or (BM == 8 and BN == 32)
    var rows = Int(row_count)
    var inputs = Int(input_count)
    var outputs = Int(output_count)
    var lane = Int(lane_id())
    var fr = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var fc = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var row = block_idx.y * BM + fr
    var column = block_idx.x * BN + fc
    var c0 = SIMD[DType.float32, 2](0)
    var c1 = SIMD[DType.float32, 2](0)
    var c2 = SIMD[DType.float32, 2](0)
    var c3 = SIMD[DType.float32, 2](0)
    var k = 0
    while k < inputs:
        var a0 = _linear_mma_fragment[False](input, row, k + fc, rows, inputs)
        var b0 = _linear_mma_fragment[True](
            weight, k + fr, column, inputs, outputs
        )
        var b1 = _linear_mma_fragment[True](
            weight, k + fr, column + 8, inputs, outputs
        )
        var p0 = c0
        var p1 = c1
        var p2 = c2
        var p3 = c3
        _mma_apple_8x8(c0, a0, b0, p0)
        _mma_apple_8x8(c1, a0, b1, p1)
        comptime if BM == 16:
            var a1 = _linear_mma_fragment[False](
                input, row + 8, k + fc, rows, inputs
            )
            _mma_apple_8x8(c2, a1, b0, p2)
            _mma_apple_8x8(c3, a1, b1, p3)
        else:
            var b2 = _linear_mma_fragment[True](
                weight, k + fr, column + 16, inputs, outputs
            )
            var b3 = _linear_mma_fragment[True](
                weight, k + fr, column + 24, inputs, outputs
            )
            _mma_apple_8x8(c2, a0, b2, p2)
            _mma_apple_8x8(c3, a0, b3, p3)
        k += 8
    _linear_mma_store[HAS_BIAS](bias, output, c0, row, column, rows, outputs)
    _linear_mma_store[HAS_BIAS](
        bias, output, c1, row, column + 8, rows, outputs
    )
    comptime if BM == 16:
        _linear_mma_store[HAS_BIAS](
            bias, output, c2, row + 8, column, rows, outputs
        )
        _linear_mma_store[HAS_BIAS](
            bias, output, c3, row + 8, column + 8, rows, outputs
        )
    else:
        _linear_mma_store[HAS_BIAS](
            bias, output, c2, row, column + 16, rows, outputs
        )
        _linear_mma_store[HAS_BIAS](
            bias, output, c3, row, column + 24, rows, outputs
        )


def enqueue_linear_prefill_mma_tile_apple_gpu[
    BM: Int,
    BN: Int,
    IL: TensorLayout,
    WL: TensorLayout,
    BL: TensorLayout,
    OL: TensorLayout,
    HAS_BIAS: Bool = True,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    """Contained 16x16/8x32 MMA alternatives to the unchanged 8x16 control."""
    comptime assert (BM == 16 and BN == 16) or (BM == 8 and BN == 32)
    _validate_linear[IL, WL, BL, OL, HAS_BIAS](input, weight, bias, output)
    if context.api() != "metal":
        raise Error("Apple GPU linear projection requires the Metal device API")
    var r = Int(input.dim[0]())
    var k = Int(input.dim[1]())
    var n = Int(weight.dim[0]())
    comptime kernel = _linear_mma_tile[BM, BN, IL, WL, BL, OL, HAS_BIAS]
    context.enqueue_function[kernel](
        input,
        weight,
        bias,
        output,
        Int32(r),
        Int32(k),
        Int32(n),
        grid_dim=(ceildiv(n, BN), ceildiv(r, BM)),
        block_dim=WARP_SIZE,
    )


def enqueue_linear_prefill_mma_tile_apple_gpu[
    BM: Int,
    BN: Int,
    IL: TensorLayout,
    WL: TensorLayout,
    OL: TensorLayout,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    var unused_bias = TileTensor(weight.ptr, row_major(1))
    enqueue_linear_prefill_mma_tile_apple_gpu[
        BM, BN, IL, WL, type_of(unused_bias.layout), OL, False
    ](context, input, weight, unused_bias, output)


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
    HAS_BIAS: Bool = True,
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
    _validate_linear[
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
    ](input, weight, bias, output)
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
        InputLayout, WeightLayout, BiasLayout, OutputLayout, HAS_BIAS
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


def linear_reference[
    IL: TensorLayout, WL: TensorLayout, OL: TensorLayout
](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    """Explicit bias-free overload; no bias allocation or load."""
    # Borrow a metadata-only view to specialize the shared implementation.
    # HAS_BIAS=False removes every access to this argument at compile time.
    var unused_bias = TileTensor(weight.ptr, row_major(1))
    linear_reference[IL, WL, type_of(unused_bias.layout), OL, False](
        input, weight, unused_bias, output
    )


def enqueue_linear_apple_gpu[
    IL: TensorLayout, WL: TensorLayout, OL: TensorLayout
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    decode_variant: Int = 0,
) raises:
    """Explicit bias-free overload; no bias allocation or load."""
    # Borrow a metadata-only view to specialize the shared implementation.
    # HAS_BIAS=False removes every access to this argument at compile time.
    var unused_bias = TileTensor(weight.ptr, row_major(1))
    enqueue_linear_apple_gpu[IL, WL, type_of(unused_bias.layout), OL, False](
        context, input, weight, unused_bias, output, decode_variant
    )


def enqueue_linear_prefill_register_2x2_apple_gpu[
    IL: TensorLayout, WL: TensorLayout, OL: TensorLayout
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    """Explicit bias-free overload; no bias allocation or load."""
    # Borrow a metadata-only view to specialize the shared implementation.
    # HAS_BIAS=False removes every access to this argument at compile time.
    var unused_bias = TileTensor(weight.ptr, row_major(1))
    enqueue_linear_prefill_register_2x2_apple_gpu[
        IL, WL, type_of(unused_bias.layout), OL, False
    ](context, input, weight, unused_bias, output)


def enqueue_linear_prefill_mma_8x16_apple_gpu[
    IL: TensorLayout, WL: TensorLayout, OL: TensorLayout
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    """Explicit bias-free overload; no bias allocation or load."""
    # Borrow a metadata-only view to specialize the shared implementation.
    # HAS_BIAS=False removes every access to this argument at compile time.
    var unused_bias = TileTensor(weight.ptr, row_major(1))
    enqueue_linear_prefill_mma_8x16_apple_gpu[
        IL, WL, type_of(unused_bias.layout), OL, False
    ](context, input, weight, unused_bias, output)


def enqueue_linear_apple_gpu_two_output[
    IL: TensorLayout,
    WL: TensorLayout,
    OL: TensorLayout,
](
    ctx: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    var unused = TileTensor(output.ptr, row_major(1))
    enqueue_linear_apple_gpu_two_output[
        IL, WL, type_of(unused.layout), OL, False
    ](ctx, input, weight, unused, output)


def _linear_pair_decode_kernel[
    TWO: Bool,
    IL: TensorLayout,
    WL: TensorLayout,
    OL: TensorLayout,
](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    gate_weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    up_weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    gate: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    up: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    k: Int32,
    n: Int32,
):
    # A whole block selects one projection. The inner kernel retains its
    # original x-grid ownership and reduction; y only combines launches.
    var weight = gate_weight if block_idx.y == 0 else up_weight
    var output = gate if block_idx.y == 0 else up
    var unused = TileTensor(output.ptr, row_major(1))
    comptime if TWO:
        _linear_two_output_apple_gpu_kernel[
            IL, WL, type_of(unused.layout), OL, False
        ](input, weight, unused, output, k, n)
    else:
        _linear_rowwise_apple_gpu_kernel[
            IL, WL, type_of(unused.layout), OL, False
        ](input, weight, unused, output, 1, k, n)


def enqueue_linear_pair_decode_apple_gpu[
    TWO: Bool,
    IL: TensorLayout,
    WL: TensorLayout,
    OL: TensorLayout,
](
    ctx: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    gate_weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    up_weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    gate: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    up: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    var unused = TileTensor(gate.ptr, row_major(1))
    _validate_linear[IL, WL, type_of(unused.layout), OL, False](
        input, gate_weight, unused, gate
    )
    _validate_linear[IL, WL, type_of(unused.layout), OL, False](
        input, up_weight, unused, up
    )
    if (
        ctx.api() != "metal"
        or Int(input.dim[0]()) != 1
        or Int(gate_weight.dim[0]()) != Int(up_weight.dim[0]())
    ):
        raise Error(
            "paired decode requires Metal, one row and equal output widths"
        )
    var n = Int(gate_weight.dim[0]())
    comptime kernel = _linear_pair_decode_kernel[TWO, IL, WL, OL]
    ctx.enqueue_function[kernel](
        input,
        gate_weight,
        up_weight,
        gate,
        up,
        Int32(input.dim[1]()),
        Int32(n),
        grid_dim=(ceildiv(n, 8 if TWO else 4), 2),
        block_dim=128,
    )


def _linear_cooperative_decode_kernel[
    GROUPS: Int,
    IL: TensorLayout,
    WL: TensorLayout,
    OL: TensorLayout,
](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    k: Int32,
    n: Int32,
):
    comptime assert is_apple_gpu()
    comptime assert GROUPS == 2 or GROUPS == 4
    comptime assert (
        input.flat_rank == 2 and weight.flat_rank == 2 and output.flat_rank == 2
    )
    var lane = lane_id()
    var group = thread_idx.x // WARP_SIZE
    var column = block_idx.x * (4 // GROUPS) + group // GROUPS
    var part = group % GROUPS
    var partials = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[4]())
    var accumulator: Float32 = 0
    var index = part * WARP_SIZE + lane
    if column < Int(n):
        while index < Int(k):
            accumulator += (
                rebind[Scalar[DType.bfloat16]](input[0, index]).cast[
                    DType.float32
                ]()
                * rebind[Scalar[DType.bfloat16]](weight[column, index]).cast[
                    DType.float32
                ]()
            )
            index += GROUPS * WARP_SIZE
    var subtotal = warp.sum(accumulator)
    if lane == 0:
        partials[group] = subtotal
    barrier()
    if part == 0 and lane == 0 and column < Int(n):
        var total: Float32 = 0
        comptime for j in range(GROUPS):
            total += partials[group + j]
        output[0, column] = rebind[output.ElementType](
            total.cast[DType.bfloat16]()
        )


def enqueue_linear_cooperative_decode_apple_gpu[
    GROUPS: Int,
    IL: TensorLayout,
    WL: TensorLayout,
    OL: TensorLayout,
](
    ctx: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises:
    comptime assert GROUPS == 2 or GROUPS == 4
    var unused = TileTensor(output.ptr, row_major(1))
    _validate_linear[IL, WL, type_of(unused.layout), OL, False](
        input, weight, unused, output
    )
    if ctx.api() != "metal" or Int(input.dim[0]()) != 1:
        raise Error("cooperative decode requires Metal and one row")
    comptime kernel = _linear_cooperative_decode_kernel[GROUPS, IL, WL, OL]
    ctx.enqueue_function[kernel](
        input,
        weight,
        output,
        Int32(input.dim[1]()),
        Int32(weight.dim[0]()),
        grid_dim=ceildiv(Int(weight.dim[0]()), 4 // GROUPS),
        block_dim=128,
    )
