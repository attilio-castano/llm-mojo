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
from layout import TileTensor, row_major
from llm_mojo.rms_norm import rms_norm_reference
from max.gpu.host import DeviceContext
from std.testing import TestSuite


def check_fixture[rows: Int, hidden_size: Int](
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
    comptime assert input.flat_rank == 2
    comptime assert weight.flat_rank == 1
    comptime assert output.flat_rank == 2

    for row in range(rows):
        for hidden in range(hidden_size):
            var index = row * hidden_size + hidden
            var value = input_values[index].cast[DType.bfloat16]()
            input[row, hidden] = rebind[input.ElementType](value)
    for hidden in range(hidden_size):
        var value = weight_values[hidden].cast[DType.bfloat16]()
        weight[hidden] = rebind[weight.ElementType](value)

    rms_norm_reference(input, weight, output)

    for row in range(rows):
        for hidden in range(hidden_size):
            var index = row * hidden_size + hidden
            var actual_bf16 = rebind[Scalar[DType.bfloat16]](
                output[row, hidden]
            )
            var actual = actual_bf16.cast[DType.float32]()
            var expected = expected_values[index]
            var error = actual - expected
            if error < 0.0:
                error = -error
            var expected_magnitude = expected
            if expected_magnitude < 0.0:
                expected_magnitude = -expected_magnitude
            var allowed = RMS_NORM_ATOL + RMS_NORM_RTOL * expected_magnitude
            if error > allowed:
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
                raise Error("RMSNorm reference did not match the oracle fixture")


def test_reference_matches_small_oracle() raises:
    var input_values = small_input()
    var weight_values = small_weight()
    var expected_values = small_expected()
    check_fixture[SMALL_ROWS, SMALL_HIDDEN_SIZE](
        input_values, weight_values, expected_values
    )


def test_reference_matches_qwen_hidden_width_oracle() raises:
    var input_values = qwen_hidden_input()
    var weight_values = qwen_hidden_weight()
    var expected_values = qwen_hidden_expected()
    check_fixture[QWEN_HIDDEN_ROWS, QWEN_HIDDEN_HIDDEN_SIZE](
        input_values, weight_values, expected_values
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
