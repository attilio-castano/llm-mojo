"""Qwen RMSNorm reference and Apple GPU implementations."""

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.bit import log2_floor
from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.primitives import warp
from std.math import rsqrt
from std.sys.info import is_apple_gpu


comptime RMS_NORM_EPSILON: Float32 = 1.0e-6
comptime RMS_NORM_APPLE_GPU_BLOCK_SIZE = 128
comptime RMS_NORM_APPLE_GPU_SIMD_GROUPS = (
    RMS_NORM_APPLE_GPU_BLOCK_SIZE // WARP_SIZE
)


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
    if rows <= 0 or hidden_size <= 0:
        raise Error("input dimensions must be positive")
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


def _rms_norm_apple_gpu_kernel[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    hidden_size: Int32,
):
    """Map one row to one 128-thread Apple GPU threadgroup."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 1, "weight must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"

    var hidden_count = Int(hidden_size)
    var row = block_idx.x
    var thread = thread_idx.x

    var sum_of_squares: Scalar[DType.float32] = 0.0
    var hidden = thread
    while hidden < hidden_count:
        var value = rebind[Scalar[DType.bfloat16]](input[row, hidden])
        var value_f32 = value.cast[DType.float32]()
        sum_of_squares += value_f32 * value_f32
        hidden += RMS_NORM_APPLE_GPU_BLOCK_SIZE

    var partial_sums = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[RMS_NORM_APPLE_GPU_BLOCK_SIZE]())
    comptime assert partial_sums.flat_rank == 1
    partial_sums[thread] = rebind[partial_sums.ElementType](sum_of_squares)
    barrier()

    var active = RMS_NORM_APPLE_GPU_BLOCK_SIZE
    comptime for _ in range(log2_floor(RMS_NORM_APPLE_GPU_BLOCK_SIZE)):
        active >>= 1
        if thread < active:
            var left = rebind[Scalar[DType.float32]](partial_sums[thread])
            var right = rebind[Scalar[DType.float32]](
                partial_sums[thread + active]
            )
            partial_sums[thread] = rebind[partial_sums.ElementType](
                left + right
            )
        barrier()

    if thread == 0:
        var sum = rebind[Scalar[DType.float32]](partial_sums[0])
        var inverse_rms = rsqrt(sum / Float32(hidden_count) + RMS_NORM_EPSILON)
        partial_sums[0] = rebind[partial_sums.ElementType](inverse_rms)
    barrier()

    var inverse_rms = rebind[Scalar[DType.float32]](partial_sums[0])
    hidden = thread
    while hidden < hidden_count:
        var value = rebind[Scalar[DType.bfloat16]](input[row, hidden])
        var normalized = (value.cast[DType.float32]() * inverse_rms).cast[
            DType.bfloat16
        ]()
        var scale = rebind[Scalar[DType.bfloat16]](weight[hidden])
        output[row, hidden] = rebind[output.ElementType](normalized * scale)
        hidden += RMS_NORM_APPLE_GPU_BLOCK_SIZE


def _rms_norm_apple_gpu_simdgroup_kernel[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    hidden_size: Int32,
):
    """Reduce within SIMD groups and exchange four shared partials."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 1, "weight must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"
    comptime assert (
        RMS_NORM_APPLE_GPU_BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must contain whole SIMD groups"

    var hidden_count = Int(hidden_size)
    var row = block_idx.x
    var thread = thread_idx.x
    var lane = lane_id()
    var simd_group = thread // WARP_SIZE

    var sum_of_squares: Scalar[DType.float32] = 0.0
    var hidden = thread
    while hidden < hidden_count:
        var value = rebind[Scalar[DType.bfloat16]](input[row, hidden])
        var value_f32 = value.cast[DType.float32]()
        sum_of_squares += value_f32 * value_f32
        hidden += RMS_NORM_APPLE_GPU_BLOCK_SIZE

    var simd_sum = warp.sum(sum_of_squares)
    var group_sums = stack_allocation[
        DType.float32, address_space=AddressSpace.SHARED
    ](row_major[RMS_NORM_APPLE_GPU_SIMD_GROUPS]())
    comptime assert group_sums.flat_rank == 1
    if lane == 0:
        group_sums[simd_group] = rebind[group_sums.ElementType](simd_sum)
    barrier()

    var group_value: Scalar[DType.float32] = 0.0
    if thread < RMS_NORM_APPLE_GPU_SIMD_GROUPS:
        group_value = rebind[Scalar[DType.float32]](group_sums[thread])
    var sum = warp.sum(group_value)
    if thread == 0:
        var inverse_rms = rsqrt(sum / Float32(hidden_count) + RMS_NORM_EPSILON)
        group_sums[0] = rebind[group_sums.ElementType](inverse_rms)
    barrier()

    var inverse_rms = rebind[Scalar[DType.float32]](group_sums[0])
    hidden = thread
    while hidden < hidden_count:
        var value = rebind[Scalar[DType.bfloat16]](input[row, hidden])
        var normalized = (value.cast[DType.float32]() * inverse_rms).cast[
            DType.bfloat16
        ]()
        var scale = rebind[Scalar[DType.bfloat16]](weight[hidden])
        output[row, hidden] = rebind[output.ElementType](normalized * scale)
        hidden += RMS_NORM_APPLE_GPU_BLOCK_SIZE


def enqueue_rms_norm_apple_gpu[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Validate and enqueue RMSNorm without synchronizing the context."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 1, "weight must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"

    var rows = Int(input.dim[0]())
    var hidden_size = Int(input.dim[1]())
    if rows <= 0 or hidden_size <= 0:
        raise Error("input dimensions must be positive")
    if Int(output.dim[0]()) != rows or Int(output.dim[1]()) != hidden_size:
        raise Error("output shape must match input shape")
    if Int(weight.dim[0]()) != hidden_size:
        raise Error("weight length must match the hidden dimension")
    if context.api() != "metal":
        raise Error("Apple GPU RMSNorm requires the Metal device API")

    comptime kernel = _rms_norm_apple_gpu_kernel[
        InputLayout, WeightLayout, OutputLayout
    ]
    context.enqueue_function[kernel](
        input,
        weight,
        output,
        Int32(hidden_size),
        grid_dim=rows,
        block_dim=RMS_NORM_APPLE_GPU_BLOCK_SIZE,
    )


def enqueue_rms_norm_apple_gpu_simdgroup[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Validate and enqueue the experimental SIMD-group RMSNorm kernel."""

    comptime assert input.flat_rank == 2, "input must have rank 2"
    comptime assert weight.flat_rank == 1, "weight must have rank 1"
    comptime assert output.flat_rank == 2, "output must have rank 2"

    var rows = Int(input.dim[0]())
    var hidden_size = Int(input.dim[1]())
    if rows <= 0 or hidden_size <= 0:
        raise Error("input dimensions must be positive")
    if Int(output.dim[0]()) != rows or Int(output.dim[1]()) != hidden_size:
        raise Error("output shape must match input shape")
    if Int(weight.dim[0]()) != hidden_size:
        raise Error("weight length must match the hidden dimension")
    if context.api() != "metal":
        raise Error("Apple GPU RMSNorm requires the Metal device API")

    comptime kernel = _rms_norm_apple_gpu_simdgroup_kernel[
        InputLayout, WeightLayout, OutputLayout
    ]
    context.enqueue_function[kernel](
        input,
        weight,
        output,
        Int32(hidden_size),
        grid_dim=rows,
        block_dim=RMS_NORM_APPLE_GPU_BLOCK_SIZE,
    )
