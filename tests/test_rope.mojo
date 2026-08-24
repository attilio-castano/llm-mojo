from fixtures.rope.reference_data import (
    QWEN_KEY_INCREMENTAL_HEAD_DIM,
    QWEN_KEY_INCREMENTAL_HEADS,
    QWEN_KEY_INCREMENTAL_ROWS,
    QWEN_KEY_INCREMENTAL_START_POSITION,
    QWEN_KEY_INCREMENTAL_TABLE_POSITIONS,
    QWEN_QUERY_DECODE_HEAD_DIM,
    QWEN_QUERY_DECODE_HEADS,
    QWEN_QUERY_DECODE_ROWS,
    QWEN_QUERY_DECODE_START_POSITION,
    QWEN_QUERY_DECODE_TABLE_POSITIONS,
    ROPE_ATOL,
    ROPE_RTOL,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_ROWS,
    TINY_START_POSITION,
    TINY_TABLE_POSITIONS,
    qwen_key_incremental_cosine_rows,
    qwen_key_incremental_expected,
    qwen_key_incremental_input,
    qwen_key_incremental_sine_rows,
    qwen_query_decode_cosine_rows,
    qwen_query_decode_expected,
    qwen_query_decode_input,
    qwen_query_decode_sine_rows,
    tiny_cosine_rows,
    tiny_expected,
    tiny_input,
    tiny_sine_rows,
)
from layout import TensorLayout, TileTensor, row_major
from llm_mojo.rope import enqueue_rope_apple_gpu, rope_reference
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.sys.info import has_apple_gpu_accelerator
from std.testing import TestSuite, assert_raises


def fill_fixture[
    InputLayout: TensorLayout,
    CosineLayout: TensorLayout,
    SineLayout: TensorLayout,
    //,
    rows: Int,
    heads: Int,
    head_dim: Int,
    start_position: Int,
    table_positions: Int,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    cosine: TileTensor[DType.bfloat16, CosineLayout, MutAnyOrigin],
    sine: TileTensor[DType.bfloat16, SineLayout, MutAnyOrigin],
    input_values: List[Float32],
    cosine_rows: List[Float32],
    sine_rows: List[Float32],
) raises:
    comptime assert input.flat_rank == 3
    comptime assert cosine.flat_rank == 2
    comptime assert sine.flat_rank == 2

    if len(input_values) != rows * heads * head_dim:
        raise Error("fixture input length does not match its shape")
    if len(cosine_rows) != rows * head_dim:
        raise Error("fixture cosine length does not match selected rows")
    if len(sine_rows) != rows * head_dim:
        raise Error("fixture sine length does not match selected rows")

    for row in range(rows):
        for head in range(heads):
            for dimension in range(head_dim):
                var index = (row * heads + head) * head_dim + dimension
                var value = input_values[index].cast[DType.bfloat16]()
                input[row, head, dimension] = rebind[input.ElementType](value)

    var zero: Scalar[DType.bfloat16] = 0.0
    for position in range(table_positions):
        for dimension in range(head_dim):
            cosine[position, dimension] = rebind[cosine.ElementType](zero)
            sine[position, dimension] = rebind[sine.ElementType](zero)
    for row in range(rows):
        var position = start_position + row
        for dimension in range(head_dim):
            var index = row * head_dim + dimension
            var cosine_value = cosine_rows[index].cast[DType.bfloat16]()
            var sine_value = sine_rows[index].cast[DType.bfloat16]()
            cosine[position, dimension] = rebind[cosine.ElementType](
                cosine_value
            )
            sine[position, dimension] = rebind[sine.ElementType](sine_value)


def assert_matches_fixture[
    OutputLayout: TensorLayout,
    //,
    rows: Int,
    heads: Int,
    head_dim: Int,
](
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    expected_values: List[Float32],
) raises:
    comptime assert output.flat_rank == 3
    for row in range(rows):
        for head in range(heads):
            for dimension in range(head_dim):
                var index = (row * heads + head) * head_dim + dimension
                var actual_bf16 = rebind[Scalar[DType.bfloat16]](
                    output[row, head, dimension]
                )
                var actual = actual_bf16.cast[DType.float32]()
                var expected = expected_values[index]
                if not isfinite(actual) or not isfinite(expected):
                    raise Error(
                        "RoPE fixture comparison requires finite values"
                    )
                var error = actual - expected
                if error < 0.0:
                    error = -error
                var expected_magnitude = expected
                if expected_magnitude < 0.0:
                    expected_magnitude = -expected_magnitude
                var allowed = ROPE_ATOL + ROPE_RTOL * expected_magnitude
                if (
                    not isfinite(error)
                    or not isfinite(allowed)
                    or error > allowed
                ):
                    print(
                        "RoPE mismatch at row",
                        row,
                        "head",
                        head,
                        "dimension",
                        dimension,
                        ": actual=",
                        actual,
                        " expected=",
                        expected,
                        " error=",
                        error,
                        " allowed=",
                        allowed,
                    )
                    raise Error(
                        "RoPE implementation did not match the oracle fixture"
                    )


def check_reference_fixture[
    rows: Int,
    heads: Int,
    head_dim: Int,
    start_position: Int,
    table_positions: Int,
](
    input_values: List[Float32],
    cosine_rows: List[Float32],
    sine_rows: List[Float32],
    expected_values: List[Float32],
) raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * heads * head_dim
    )
    var cosine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        table_positions * head_dim
    )
    var sine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        table_positions * head_dim
    )
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * heads * head_dim
    )

    var input = TileTensor(input_buffer, row_major[rows, heads, head_dim]())
    var cosine = TileTensor(
        cosine_buffer, row_major[table_positions, head_dim]()
    )
    var sine = TileTensor(sine_buffer, row_major[table_positions, head_dim]())
    var output = TileTensor(output_buffer, row_major[rows, heads, head_dim]())
    fill_fixture[rows, heads, head_dim, start_position, table_positions](
        input, cosine, sine, input_values, cosine_rows, sine_rows
    )
    rope_reference(input, cosine, sine, output, start_position)
    assert_matches_fixture[rows, heads, head_dim](output, expected_values)


def check_apple_gpu_fixture[
    rows: Int,
    heads: Int,
    head_dim: Int,
    start_position: Int,
    table_positions: Int,
](
    input_values: List[Float32],
    cosine_rows: List[Float32],
    sine_rows: List[Float32],
    expected_values: List[Float32],
) raises:
    comptime assert has_apple_gpu_accelerator(), "test requires an Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("test requires the Metal device API")

    var host_input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * heads * head_dim
    )
    var host_cosine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        table_positions * head_dim
    )
    var host_sine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        table_positions * head_dim
    )
    var host_output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * heads * head_dim
    )
    var device_input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * heads * head_dim
    )
    var device_cosine_buffer = context.enqueue_create_buffer[DType.bfloat16](
        table_positions * head_dim
    )
    var device_sine_buffer = context.enqueue_create_buffer[DType.bfloat16](
        table_positions * head_dim
    )
    var device_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * heads * head_dim
    )

    var host_input = TileTensor(
        host_input_buffer, row_major[rows, heads, head_dim]()
    )
    var host_cosine = TileTensor(
        host_cosine_buffer, row_major[table_positions, head_dim]()
    )
    var host_sine = TileTensor(
        host_sine_buffer, row_major[table_positions, head_dim]()
    )
    fill_fixture[rows, heads, head_dim, start_position, table_positions](
        host_input,
        host_cosine,
        host_sine,
        input_values,
        cosine_rows,
        sine_rows,
    )
    context.enqueue_copy(dst_buf=device_input_buffer, src_buf=host_input_buffer)
    context.enqueue_copy(
        dst_buf=device_cosine_buffer, src_buf=host_cosine_buffer
    )
    context.enqueue_copy(dst_buf=device_sine_buffer, src_buf=host_sine_buffer)

    var device_input = TileTensor(
        device_input_buffer, row_major[rows, heads, head_dim]()
    )
    var device_cosine = TileTensor(
        device_cosine_buffer, row_major[table_positions, head_dim]()
    )
    var device_sine = TileTensor(
        device_sine_buffer, row_major[table_positions, head_dim]()
    )
    var device_output = TileTensor(
        device_output_buffer, row_major[rows, heads, head_dim]()
    )
    enqueue_rope_apple_gpu(
        context,
        device_input,
        device_cosine,
        device_sine,
        device_output,
        start_position,
    )
    context.enqueue_copy(
        dst_buf=host_output_buffer, src_buf=device_output_buffer
    )
    context.synchronize()

    var host_output = TileTensor(
        host_output_buffer, row_major[rows, heads, head_dim]()
    )
    assert_matches_fixture[rows, heads, head_dim](host_output, expected_values)


def test_fixture_comparison_rejects_nan() raises:
    var context = DeviceContext()
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](1)
    var output = TileTensor(output_buffer, row_major[1, 1, 1]())
    var nan_value: Scalar[DType.bfloat16] = FloatLiteral.nan
    output[0, 0, 0] = rebind[output.ElementType](nan_value)
    var expected_values: List[Float32] = [0.0]

    with assert_raises(contains="requires finite values"):
        assert_matches_fixture[1, 1, 1](output, expected_values)


def test_reference_rejects_odd_head_dimension() raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](3)
    var cosine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](3)
    var sine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](3)
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](3)
    var input = TileTensor(input_buffer, row_major[1, 1, 3]())
    var cosine = TileTensor(cosine_buffer, row_major[1, 3]())
    var sine = TileTensor(sine_buffer, row_major[1, 3]())
    var output = TileTensor(output_buffer, row_major[1, 1, 3]())

    with assert_raises(contains="head dimension must be even"):
        rope_reference(input, cosine, sine, output, 0)


def test_reference_rejects_uncovered_position() raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var cosine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var sine_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var input = TileTensor(input_buffer, row_major[1, 1, 4]())
    var cosine = TileTensor(cosine_buffer, row_major[1, 4]())
    var sine = TileTensor(sine_buffer, row_major[1, 4]())
    var output = TileTensor(output_buffer, row_major[1, 1, 4]())

    with assert_raises(contains="does not cover"):
        rope_reference(input, cosine, sine, output, 1)


def test_reference_matches_tiny_oracle() raises:
    check_reference_fixture[
        TINY_ROWS,
        TINY_HEADS,
        TINY_HEAD_DIM,
        TINY_START_POSITION,
        TINY_TABLE_POSITIONS,
    ](
        tiny_input(),
        tiny_cosine_rows(),
        tiny_sine_rows(),
        tiny_expected(),
    )


def test_reference_matches_qwen_query_decode_oracle() raises:
    check_reference_fixture[
        QWEN_QUERY_DECODE_ROWS,
        QWEN_QUERY_DECODE_HEADS,
        QWEN_QUERY_DECODE_HEAD_DIM,
        QWEN_QUERY_DECODE_START_POSITION,
        QWEN_QUERY_DECODE_TABLE_POSITIONS,
    ](
        qwen_query_decode_input(),
        qwen_query_decode_cosine_rows(),
        qwen_query_decode_sine_rows(),
        qwen_query_decode_expected(),
    )


def test_reference_matches_qwen_key_incremental_oracle() raises:
    check_reference_fixture[
        QWEN_KEY_INCREMENTAL_ROWS,
        QWEN_KEY_INCREMENTAL_HEADS,
        QWEN_KEY_INCREMENTAL_HEAD_DIM,
        QWEN_KEY_INCREMENTAL_START_POSITION,
        QWEN_KEY_INCREMENTAL_TABLE_POSITIONS,
    ](
        qwen_key_incremental_input(),
        qwen_key_incremental_cosine_rows(),
        qwen_key_incremental_sine_rows(),
        qwen_key_incremental_expected(),
    )


def test_apple_gpu_matches_tiny_oracle() raises:
    check_apple_gpu_fixture[
        TINY_ROWS,
        TINY_HEADS,
        TINY_HEAD_DIM,
        TINY_START_POSITION,
        TINY_TABLE_POSITIONS,
    ](
        tiny_input(),
        tiny_cosine_rows(),
        tiny_sine_rows(),
        tiny_expected(),
    )


def test_apple_gpu_matches_qwen_query_decode_oracle() raises:
    check_apple_gpu_fixture[
        QWEN_QUERY_DECODE_ROWS,
        QWEN_QUERY_DECODE_HEADS,
        QWEN_QUERY_DECODE_HEAD_DIM,
        QWEN_QUERY_DECODE_START_POSITION,
        QWEN_QUERY_DECODE_TABLE_POSITIONS,
    ](
        qwen_query_decode_input(),
        qwen_query_decode_cosine_rows(),
        qwen_query_decode_sine_rows(),
        qwen_query_decode_expected(),
    )


def test_apple_gpu_matches_qwen_key_incremental_oracle() raises:
    check_apple_gpu_fixture[
        QWEN_KEY_INCREMENTAL_ROWS,
        QWEN_KEY_INCREMENTAL_HEADS,
        QWEN_KEY_INCREMENTAL_HEAD_DIM,
        QWEN_KEY_INCREMENTAL_START_POSITION,
        QWEN_KEY_INCREMENTAL_TABLE_POSITIONS,
    ](
        qwen_key_incremental_input(),
        qwen_key_incremental_cosine_rows(),
        qwen_key_incremental_sine_rows(),
        qwen_key_incremental_expected(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
