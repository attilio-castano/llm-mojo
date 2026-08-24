"""Qwen RoPE reference and Apple GPU implementations."""

from layout import TensorLayout, TileTensor
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import ceildiv
from std.sys.info import is_apple_gpu


comptime ROPE_APPLE_GPU_BLOCK_SIZE = 128


def _validate_rope[
    InputLayout: TensorLayout,
    CosineLayout: TensorLayout,
    SineLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    cosine: TileTensor[DType.bfloat16, CosineLayout, MutAnyOrigin],
    sine: TileTensor[DType.bfloat16, SineLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    start_position: Int,
) raises:
    """Validate the shared host and enqueue contract."""

    comptime assert input.flat_rank == 3, "input must have rank 3"
    comptime assert cosine.flat_rank == 2, "cosine must have rank 2"
    comptime assert sine.flat_rank == 2, "sine must have rank 2"
    comptime assert output.flat_rank == 3, "output must have rank 3"

    var rows = Int(input.dim[0]())
    var heads = Int(input.dim[1]())
    var head_dim = Int(input.dim[2]())
    if rows <= 0 or heads <= 0 or head_dim <= 0:
        raise Error("input dimensions must be positive")
    if head_dim % 2 != 0:
        raise Error("head dimension must be even")
    if (
        Int(output.dim[0]()) != rows
        or Int(output.dim[1]()) != heads
        or Int(output.dim[2]()) != head_dim
    ):
        raise Error("output shape must match input shape")

    var table_positions = Int(cosine.dim[0]())
    if table_positions <= 0:
        raise Error("rotary table must contain at least one position")
    if Int(cosine.dim[1]()) != head_dim:
        raise Error("cosine head dimension must match input")
    if Int(sine.dim[0]()) != table_positions or Int(sine.dim[1]()) != head_dim:
        raise Error("sine shape must match cosine shape")
    if start_position < 0:
        raise Error("start position must be nonnegative")
    if start_position + rows > table_positions:
        raise Error("rotary table does not cover the requested positions")


def rope_reference[
    InputLayout: TensorLayout,
    CosineLayout: TensorLayout,
    SineLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    cosine: TileTensor[DType.bfloat16, CosineLayout, MutAnyOrigin],
    sine: TileTensor[DType.bfloat16, SineLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    start_position: Int,
) raises:
    """Apply the pinned Qwen2 BF16 rotary contract on the host."""

    comptime assert input.flat_rank == 3, "input must have rank 3"
    comptime assert cosine.flat_rank == 2, "cosine must have rank 2"
    comptime assert sine.flat_rank == 2, "sine must have rank 2"
    comptime assert output.flat_rank == 3, "output must have rank 3"
    _validate_rope(input, cosine, sine, output, start_position)

    var rows = Int(input.dim[0]())
    var heads = Int(input.dim[1]())
    var head_dim = Int(input.dim[2]())
    var half_dim = head_dim // 2
    for row in range(rows):
        var position = start_position + row
        for head in range(heads):
            for pair in range(half_dim):
                var second_dim = pair + half_dim
                var first = rebind[Scalar[DType.bfloat16]](
                    input[row, head, pair]
                )
                var second = rebind[Scalar[DType.bfloat16]](
                    input[row, head, second_dim]
                )
                var cosine_first = rebind[Scalar[DType.bfloat16]](
                    cosine[position, pair]
                )
                var cosine_second = rebind[Scalar[DType.bfloat16]](
                    cosine[position, second_dim]
                )
                var sine_first = rebind[Scalar[DType.bfloat16]](
                    sine[position, pair]
                )
                var sine_second = rebind[Scalar[DType.bfloat16]](
                    sine[position, second_dim]
                )

                var first_cosine: Scalar[DType.bfloat16] = first * cosine_first
                var second_sine: Scalar[DType.bfloat16] = second * sine_first
                var second_cosine: Scalar[DType.bfloat16] = (
                    second * cosine_second
                )
                var first_sine: Scalar[DType.bfloat16] = first * sine_second
                var rotated_first: Scalar[DType.bfloat16] = (
                    first_cosine - second_sine
                )
                var rotated_second: Scalar[DType.bfloat16] = (
                    second_cosine + first_sine
                )
                output[row, head, pair] = rebind[output.ElementType](
                    rotated_first
                )
                output[row, head, second_dim] = rebind[output.ElementType](
                    rotated_second
                )


def _rope_apple_gpu_kernel[
    InputLayout: TensorLayout,
    CosineLayout: TensorLayout,
    SineLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    cosine: TileTensor[DType.bfloat16, CosineLayout, MutAnyOrigin],
    sine: TileTensor[DType.bfloat16, SineLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    start_position: Int32,
    rows: Int32,
    heads: Int32,
    head_dim: Int32,
):
    """Map one half-split rotary pair to one Apple GPU thread."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert input.flat_rank == 3, "input must have rank 3"
    comptime assert cosine.flat_rank == 2, "cosine must have rank 2"
    comptime assert sine.flat_rank == 2, "sine must have rank 2"
    comptime assert output.flat_rank == 3, "output must have rank 3"

    var row_count = Int(rows)
    var head_count = Int(heads)
    var dimension_count = Int(head_dim)
    var half_dim = dimension_count // 2
    var pair_count = row_count * head_count * half_dim
    var flat_pair = global_idx.x
    if flat_pair < pair_count:
        var pair = flat_pair % half_dim
        var row_head = flat_pair // half_dim
        var head = row_head % head_count
        var row = row_head // head_count
        var position = Int(start_position) + row
        var second_dim = pair + half_dim

        var first = rebind[Scalar[DType.bfloat16]](input[row, head, pair])
        var second = rebind[Scalar[DType.bfloat16]](
            input[row, head, second_dim]
        )
        var cosine_first = rebind[Scalar[DType.bfloat16]](
            cosine[position, pair]
        )
        var cosine_second = rebind[Scalar[DType.bfloat16]](
            cosine[position, second_dim]
        )
        var sine_first = rebind[Scalar[DType.bfloat16]](sine[position, pair])
        var sine_second = rebind[Scalar[DType.bfloat16]](
            sine[position, second_dim]
        )

        var first_cosine: Scalar[DType.bfloat16] = first * cosine_first
        var second_sine: Scalar[DType.bfloat16] = second * sine_first
        var second_cosine: Scalar[DType.bfloat16] = second * cosine_second
        var first_sine: Scalar[DType.bfloat16] = first * sine_second
        var rotated_first: Scalar[DType.bfloat16] = first_cosine - second_sine
        var rotated_second: Scalar[DType.bfloat16] = second_cosine + first_sine
        output[row, head, pair] = rebind[output.ElementType](rotated_first)
        output[row, head, second_dim] = rebind[output.ElementType](
            rotated_second
        )


def enqueue_rope_apple_gpu[
    InputLayout: TensorLayout,
    CosineLayout: TensorLayout,
    SineLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    context: DeviceContext,
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    cosine: TileTensor[DType.bfloat16, CosineLayout, MutAnyOrigin],
    sine: TileTensor[DType.bfloat16, SineLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    start_position: Int,
) raises:
    """Validate and enqueue RoPE without synchronizing the context."""

    comptime assert input.flat_rank == 3, "input must have rank 3"
    comptime assert cosine.flat_rank == 2, "cosine must have rank 2"
    comptime assert sine.flat_rank == 2, "sine must have rank 2"
    comptime assert output.flat_rank == 3, "output must have rank 3"
    _validate_rope(input, cosine, sine, output, start_position)
    if context.api() != "metal":
        raise Error("Apple GPU RoPE requires the Metal device API")

    var rows = Int(input.dim[0]())
    var heads = Int(input.dim[1]())
    var head_dim = Int(input.dim[2]())
    var pair_count = rows * heads * (head_dim // 2)
    comptime kernel = _rope_apple_gpu_kernel[
        InputLayout, CosineLayout, SineLayout, OutputLayout
    ]
    context.enqueue_function[kernel](
        input,
        cosine,
        sine,
        output,
        Int32(start_position),
        Int32(rows),
        Int32(heads),
        Int32(head_dim),
        grid_dim=ceildiv(pair_count, ROPE_APPLE_GPU_BLOCK_SIZE),
        block_dim=ROPE_APPLE_GPU_BLOCK_SIZE,
    )
