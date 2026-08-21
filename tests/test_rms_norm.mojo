from fixtures.rms_norm.reference_data import (
    QWEN_HIDDEN_HIDDEN_SIZE,
    QWEN_HIDDEN_ROWS,
    RMS_NORM_ATOL,
    RMS_NORM_RTOL,
    SMALL_HIDDEN_SIZE,
    SMALL_ROWS,
    qwen_hidden_expected,
    qwen_hidden_input,
    qwen_hidden_weight,
    small_expected,
    small_input,
    small_weight,
)
from layout import TensorLayout, TileTensor, row_major
from llm_mojo.rms_norm import (
    enqueue_rms_norm_apple_gpu,
    rms_norm_reference,
)
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.sys.info import has_apple_gpu_accelerator
from std.testing import TestSuite, assert_raises


def fill_fixture[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    //,
    rows: Int,
    hidden_size: Int,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    input_values: List[Float32],
    weight_values: List[Float32],
) raises:
    comptime assert input.flat_rank == 2
    comptime assert weight.flat_rank == 1

    for row in range(rows):
        for hidden in range(hidden_size):
            var index = row * hidden_size + hidden
            var value = input_values[index].cast[DType.bfloat16]()
            input[row, hidden] = rebind[input.ElementType](value)
    for hidden in range(hidden_size):
        var value = weight_values[hidden].cast[DType.bfloat16]()
        weight[hidden] = rebind[weight.ElementType](value)


def assert_matches_fixture[
    OutputLayout: TensorLayout,
    //,
    rows: Int,
    hidden_size: Int,
](
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    expected_values: List[Float32],
) raises:
    comptime assert output.flat_rank == 2
    for row in range(rows):
        for hidden in range(hidden_size):
            var index = row * hidden_size + hidden
            var actual_bf16 = rebind[Scalar[DType.bfloat16]](
                output[row, hidden]
            )
            var actual = actual_bf16.cast[DType.float32]()
            var expected = expected_values[index]
            if not isfinite(actual) or not isfinite(expected):
                raise Error("RMSNorm fixture comparison requires finite values")
            var error = actual - expected
            if error < 0.0:
                error = -error
            var expected_magnitude = expected
            if expected_magnitude < 0.0:
                expected_magnitude = -expected_magnitude
            var allowed = RMS_NORM_ATOL + RMS_NORM_RTOL * expected_magnitude
            if not isfinite(error) or not isfinite(allowed) or error > allowed:
                print(
                    "RMSNorm mismatch at flat index",
                    index,
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
                    "RMSNorm implementation did not match the oracle fixture"
                )


def test_fixture_comparison_rejects_nan() raises:
    var context = DeviceContext()
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](1)
    var output = TileTensor(output_buffer, row_major[1, 1]())
    var nan_value: Scalar[DType.bfloat16] = FloatLiteral.nan
    output[0, 0] = rebind[output.ElementType](nan_value)
    var expected_values: List[Float32] = [0.0]

    with assert_raises(contains="requires finite values"):
        assert_matches_fixture[1, 1](output, expected_values)


def check_reference_fixture[
    rows: Int, hidden_size: Int
](
    input_values: List[Float32],
    weight_values: List[Float32],
    expected_values: List[Float32],
) raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * hidden_size
    )
    var weight_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        hidden_size
    )
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * hidden_size
    )

    var input = TileTensor(input_buffer, row_major[rows, hidden_size]())
    var weight = TileTensor(weight_buffer, row_major[hidden_size]())
    var output = TileTensor(output_buffer, row_major[rows, hidden_size]())
    fill_fixture[rows, hidden_size](input, weight, input_values, weight_values)
    rms_norm_reference(input, weight, output)
    assert_matches_fixture[rows, hidden_size](output, expected_values)


def check_apple_gpu_fixture[
    rows: Int, hidden_size: Int
](
    input_values: List[Float32],
    weight_values: List[Float32],
    expected_values: List[Float32],
) raises:
    comptime assert has_apple_gpu_accelerator(), "test requires an Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("test requires the Metal device API")

    var host_input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * hidden_size
    )
    var host_weight_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        hidden_size
    )
    var host_output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * hidden_size
    )
    var device_input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * hidden_size
    )
    var device_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        hidden_size
    )
    var device_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * hidden_size
    )

    var host_input = TileTensor(
        host_input_buffer, row_major[rows, hidden_size]()
    )
    var host_weight = TileTensor(host_weight_buffer, row_major[hidden_size]())
    fill_fixture[rows, hidden_size](
        host_input, host_weight, input_values, weight_values
    )
    context.enqueue_copy(dst_buf=device_input_buffer, src_buf=host_input_buffer)
    context.enqueue_copy(
        dst_buf=device_weight_buffer, src_buf=host_weight_buffer
    )

    var device_input = TileTensor(
        device_input_buffer, row_major[rows, hidden_size]()
    )
    var device_weight = TileTensor(
        device_weight_buffer, row_major[hidden_size]()
    )
    var device_output = TileTensor(
        device_output_buffer, row_major[rows, hidden_size]()
    )
    enqueue_rms_norm_apple_gpu(
        context, device_input, device_weight, device_output
    )
    context.enqueue_copy(
        dst_buf=host_output_buffer, src_buf=device_output_buffer
    )
    context.synchronize()

    var host_output = TileTensor(
        host_output_buffer, row_major[rows, hidden_size]()
    )
    assert_matches_fixture[rows, hidden_size](host_output, expected_values)


def test_reference_matches_small_oracle() raises:
    var input_values = small_input()
    var weight_values = small_weight()
    var expected_values = small_expected()
    check_reference_fixture[SMALL_ROWS, SMALL_HIDDEN_SIZE](
        input_values, weight_values, expected_values
    )


def test_reference_matches_qwen_hidden_width_oracle() raises:
    var input_values = qwen_hidden_input()
    var weight_values = qwen_hidden_weight()
    var expected_values = qwen_hidden_expected()
    check_reference_fixture[QWEN_HIDDEN_ROWS, QWEN_HIDDEN_HIDDEN_SIZE](
        input_values, weight_values, expected_values
    )


def test_apple_gpu_matches_small_oracle() raises:
    var input_values = small_input()
    var weight_values = small_weight()
    var expected_values = small_expected()
    check_apple_gpu_fixture[SMALL_ROWS, SMALL_HIDDEN_SIZE](
        input_values, weight_values, expected_values
    )


def test_apple_gpu_matches_qwen_hidden_width_oracle() raises:
    var input_values = qwen_hidden_input()
    var weight_values = qwen_hidden_weight()
    var expected_values = qwen_hidden_expected()
    check_apple_gpu_fixture[QWEN_HIDDEN_ROWS, QWEN_HIDDEN_HIDDEN_SIZE](
        input_values, weight_values, expected_values
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
