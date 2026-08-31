from fixtures.linear.reference_data import (
    LINEAR_ATOL,
    LINEAR_RTOL,
    SHORT_PREFILL_INPUT_FEATURES,
    SHORT_PREFILL_OUTPUT_FEATURES,
    SHORT_PREFILL_ROWS,
    TINY_DECODE_INPUT_FEATURES,
    TINY_DECODE_OUTPUT_FEATURES,
    TINY_DECODE_ROWS,
    short_prefill_bias,
    short_prefill_expected,
    short_prefill_input,
    short_prefill_weight,
    tiny_decode_bias,
    tiny_decode_expected,
    tiny_decode_input,
    tiny_decode_weight,
)
from layout import TensorLayout, TileTensor, row_major
from llm_mojo.linear import (
    enqueue_linear_apple_gpu,
    enqueue_linear_apple_gpu_two_output,
    enqueue_linear_prefill_direct_apple_gpu,
    enqueue_linear_prefill_mma_8x16_apple_gpu,
    enqueue_linear_prefill_register_2x2_apple_gpu,
    enqueue_linear_prefill_tiled_apple_gpu,
    enqueue_linear_prefill_tiled_apple_gpu_bk,
    linear_reference,
)
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.sys.info import has_apple_gpu_accelerator
from std.testing import TestSuite, assert_raises


def fill_fixture[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    //,
    rows: Int,
    input_features: Int,
    output_features: Int,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
    input_values: List[Float32],
    weight_values: List[Float32],
    bias_values: List[Float32],
) raises:
    comptime assert input.flat_rank == 2
    comptime assert weight.flat_rank == 2
    comptime assert bias.flat_rank == 1

    if len(input_values) != rows * input_features:
        raise Error("fixture input length does not match its shape")
    if len(weight_values) != output_features * input_features:
        raise Error("fixture weight length does not match its shape")
    if len(bias_values) != output_features:
        raise Error("fixture bias length does not match its shape")

    for row in range(rows):
        for input_feature in range(input_features):
            var index = row * input_features + input_feature
            var value = input_values[index].cast[DType.bfloat16]()
            input[row, input_feature] = rebind[input.ElementType](value)
    for output_feature in range(output_features):
        for input_feature in range(input_features):
            var index = output_feature * input_features + input_feature
            var value = weight_values[index].cast[DType.bfloat16]()
            weight[output_feature, input_feature] = rebind[weight.ElementType](
                value
            )
        var bias_value = bias_values[output_feature].cast[DType.bfloat16]()
        bias[output_feature] = rebind[bias.ElementType](bias_value)


def assert_matches_fixture[
    OutputLayout: TensorLayout,
    //,
    rows: Int,
    output_features: Int,
](
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    expected_values: List[Float32],
) raises:
    comptime assert output.flat_rank == 2
    for row in range(rows):
        for output_feature in range(output_features):
            var index = row * output_features + output_feature
            var actual_bf16 = rebind[Scalar[DType.bfloat16]](
                output[row, output_feature]
            )
            var actual = actual_bf16.cast[DType.float32]()
            var expected = expected_values[index]
            if not isfinite(actual) or not isfinite(expected):
                raise Error("linear fixture comparison requires finite values")
            var error = actual - expected
            if error < 0.0:
                error = -error
            var expected_magnitude = expected
            if expected_magnitude < 0.0:
                expected_magnitude = -expected_magnitude
            var allowed = LINEAR_ATOL + LINEAR_RTOL * expected_magnitude
            if not isfinite(error) or not isfinite(allowed) or error > allowed:
                print(
                    "linear mismatch at row",
                    row,
                    "output feature",
                    output_feature,
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
                    "linear implementation did not match the oracle fixture"
                )


def check_reference_fixture[
    rows: Int, input_features: Int, output_features: Int
](
    input_values: List[Float32],
    weight_values: List[Float32],
    bias_values: List[Float32],
    expected_values: List[Float32],
) raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * input_features
    )
    var weight_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        output_features * input_features
    )
    var bias_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        output_features
    )
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * output_features
    )

    var input = TileTensor(input_buffer, row_major[rows, input_features]())
    var weight = TileTensor(
        weight_buffer, row_major[output_features, input_features]()
    )
    var bias = TileTensor(bias_buffer, row_major[output_features]())
    var output = TileTensor(output_buffer, row_major[rows, output_features]())
    fill_fixture[rows, input_features, output_features](
        input, weight, bias, input_values, weight_values, bias_values
    )
    linear_reference(input, weight, bias, output)
    assert_matches_fixture[rows, output_features](output, expected_values)


def check_apple_gpu_fixture[
    rows: Int,
    input_features: Int,
    output_features: Int,
    use_direct_prefill: Bool = False,
    use_tiled_prefill: Bool = False,
    use_register_prefill: Bool = False,
    use_mma_prefill: Bool = False,
](
    input_values: List[Float32],
    weight_values: List[Float32],
    bias_values: List[Float32],
    expected_values: List[Float32],
) raises:
    comptime assert has_apple_gpu_accelerator(), "test requires an Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("test requires the Metal device API")

    var host_input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * input_features
    )
    var host_weight_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        output_features * input_features
    )
    var host_bias_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        output_features
    )
    var host_output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * output_features
    )
    var device_input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * input_features
    )
    var device_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features * input_features
    )
    var device_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features
    )
    var device_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * output_features
    )

    var host_input = TileTensor(
        host_input_buffer, row_major[rows, input_features]()
    )
    var host_weight = TileTensor(
        host_weight_buffer, row_major[output_features, input_features]()
    )
    var host_bias = TileTensor(host_bias_buffer, row_major[output_features]())
    fill_fixture[rows, input_features, output_features](
        host_input,
        host_weight,
        host_bias,
        input_values,
        weight_values,
        bias_values,
    )
    context.enqueue_copy(dst_buf=device_input_buffer, src_buf=host_input_buffer)
    context.enqueue_copy(
        dst_buf=device_weight_buffer, src_buf=host_weight_buffer
    )
    context.enqueue_copy(dst_buf=device_bias_buffer, src_buf=host_bias_buffer)

    var device_input = TileTensor(
        device_input_buffer, row_major[rows, input_features]()
    )
    var device_weight = TileTensor(
        device_weight_buffer, row_major[output_features, input_features]()
    )
    var device_bias = TileTensor(
        device_bias_buffer, row_major[output_features]()
    )
    var device_output = TileTensor(
        device_output_buffer, row_major[rows, output_features]()
    )
    comptime assert not (
        (use_direct_prefill and use_tiled_prefill)
        or (use_direct_prefill and use_register_prefill)
        or (use_direct_prefill and use_mma_prefill)
        or (use_tiled_prefill and use_register_prefill)
        or (use_tiled_prefill and use_mma_prefill)
        or (use_register_prefill and use_mma_prefill)
    ), "test implementation modes are mutually exclusive"
    comptime if use_mma_prefill:
        enqueue_linear_prefill_mma_8x16_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    elif use_register_prefill:
        enqueue_linear_prefill_register_2x2_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    elif use_tiled_prefill:
        enqueue_linear_prefill_tiled_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    elif use_direct_prefill:
        enqueue_linear_prefill_direct_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    else:
        enqueue_linear_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    context.enqueue_copy(
        dst_buf=host_output_buffer, src_buf=device_output_buffer
    )
    context.synchronize()

    var host_output = TileTensor(
        host_output_buffer, row_major[rows, output_features]()
    )
    assert_matches_fixture[rows, output_features](host_output, expected_values)


def fill_model_shape[
    InputLayout: TensorLayout,
    WeightLayout: TensorLayout,
    BiasLayout: TensorLayout,
    //,
    rows: Int,
    input_features: Int,
    output_features: Int,
](
    input: TileTensor[DType.bfloat16, InputLayout, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WeightLayout, MutAnyOrigin],
    bias: TileTensor[DType.bfloat16, BiasLayout, MutAnyOrigin],
):
    comptime assert input.flat_rank == 2
    comptime assert weight.flat_rank == 2
    comptime assert bias.flat_rank == 1

    for row in range(rows):
        for input_feature in range(input_features):
            var index = row * input_features + input_feature
            var numerator = (index * 37 + 19) % 257 - 128
            var value = (Float32(numerator) / 128.0).cast[DType.bfloat16]()
            input[row, input_feature] = rebind[input.ElementType](value)
    for output_feature in range(output_features):
        for input_feature in range(input_features):
            var index = output_feature * input_features + input_feature
            var numerator = (index * 13 + 7) % 127 - 63
            var value = (Float32(numerator) / 512.0).cast[DType.bfloat16]()
            weight[output_feature, input_feature] = rebind[weight.ElementType](
                value
            )
        var bias_numerator = (output_feature * 11 + 5) % 31 - 15
        var bias_value = (Float32(bias_numerator) / 64.0).cast[DType.bfloat16]()
        bias[output_feature] = rebind[bias.ElementType](bias_value)


def assert_outputs_match[
    ExpectedLayout: TensorLayout,
    ActualLayout: TensorLayout,
    //,
    rows: Int,
    output_features: Int,
](
    expected: TileTensor[DType.bfloat16, ExpectedLayout, MutAnyOrigin],
    actual: TileTensor[DType.bfloat16, ActualLayout, MutAnyOrigin],
) raises:
    comptime assert expected.flat_rank == 2
    comptime assert actual.flat_rank == 2

    for row in range(rows):
        for output_feature in range(output_features):
            var expected_value = rebind[Scalar[DType.bfloat16]](
                expected[row, output_feature]
            ).cast[DType.float32]()
            var actual_value = rebind[Scalar[DType.bfloat16]](
                actual[row, output_feature]
            ).cast[DType.float32]()
            if not isfinite(actual_value) or not isfinite(expected_value):
                raise Error("model-shape comparison requires finite values")
            var error = actual_value - expected_value
            if error < 0.0:
                error = -error
            var expected_magnitude = expected_value
            if expected_magnitude < 0.0:
                expected_magnitude = -expected_magnitude
            var allowed = LINEAR_ATOL + LINEAR_RTOL * expected_magnitude
            if error > allowed:
                print(
                    "model-shape mismatch at row",
                    row,
                    "output feature",
                    output_feature,
                    ": actual=",
                    actual_value,
                    " expected=",
                    expected_value,
                    " error=",
                    error,
                    " allowed=",
                    allowed,
                )
                raise Error("Apple GPU projection did not match host reference")


def check_model_shape[
    rows: Int,
    input_features: Int,
    output_features: Int,
    use_two_output: Bool = False,
    use_direct_prefill: Bool = False,
    use_tiled_prefill: Bool = False,
    tiled_input_features: Int = 32,
    use_register_prefill: Bool = False,
    use_mma_prefill: Bool = False,
]() raises:
    comptime assert has_apple_gpu_accelerator(), "test requires an Apple GPU"
    comptime assert not (
        (use_two_output and use_direct_prefill)
        or (use_two_output and use_tiled_prefill)
        or (use_two_output and use_register_prefill)
        or (use_two_output and use_mma_prefill)
        or (use_direct_prefill and use_tiled_prefill)
        or (use_direct_prefill and use_register_prefill)
        or (use_direct_prefill and use_mma_prefill)
        or (use_tiled_prefill and use_register_prefill)
        or (use_tiled_prefill and use_mma_prefill)
        or (use_register_prefill and use_mma_prefill)
    ), "test implementation modes are mutually exclusive"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("test requires the Metal device API")

    var host_input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * input_features
    )
    var host_weight_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        output_features * input_features
    )
    var host_bias_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        output_features
    )
    var host_expected_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](rows * output_features)
    var host_actual_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        rows * output_features
    )
    var device_input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * input_features
    )
    var device_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features * input_features
    )
    var device_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features
    )
    var device_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * output_features
    )

    var host_input = TileTensor(
        host_input_buffer, row_major[rows, input_features]()
    )
    var host_weight = TileTensor(
        host_weight_buffer, row_major[output_features, input_features]()
    )
    var host_bias = TileTensor(host_bias_buffer, row_major[output_features]())
    var host_expected = TileTensor(
        host_expected_buffer, row_major[rows, output_features]()
    )
    fill_model_shape[rows, input_features, output_features](
        host_input, host_weight, host_bias
    )
    linear_reference(host_input, host_weight, host_bias, host_expected)

    context.enqueue_copy(dst_buf=device_input_buffer, src_buf=host_input_buffer)
    context.enqueue_copy(
        dst_buf=device_weight_buffer, src_buf=host_weight_buffer
    )
    context.enqueue_copy(dst_buf=device_bias_buffer, src_buf=host_bias_buffer)
    var device_input = TileTensor(
        device_input_buffer, row_major[rows, input_features]()
    )
    var device_weight = TileTensor(
        device_weight_buffer, row_major[output_features, input_features]()
    )
    var device_bias = TileTensor(
        device_bias_buffer, row_major[output_features]()
    )
    var device_output = TileTensor(
        device_output_buffer, row_major[rows, output_features]()
    )
    comptime if use_two_output:
        enqueue_linear_apple_gpu_two_output(
            context, device_input, device_weight, device_bias, device_output
        )
    elif use_mma_prefill:
        enqueue_linear_prefill_mma_8x16_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    elif use_register_prefill:
        enqueue_linear_prefill_register_2x2_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    elif use_tiled_prefill:
        enqueue_linear_prefill_tiled_apple_gpu_bk[tiled_input_features](
            context, device_input, device_weight, device_bias, device_output
        )
    elif use_direct_prefill:
        enqueue_linear_prefill_direct_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    else:
        enqueue_linear_apple_gpu(
            context, device_input, device_weight, device_bias, device_output
        )
    context.enqueue_copy(
        dst_buf=host_actual_buffer, src_buf=device_output_buffer
    )
    context.synchronize()

    var host_actual = TileTensor(
        host_actual_buffer, row_major[rows, output_features]()
    )
    assert_outputs_match[rows, output_features](host_expected, host_actual)


def check_packed_qkv_matches_separate_projections() raises:
    """Prove packed Q|K|V regions match three independent enqueues exactly."""

    comptime rows = 1
    comptime input_features = 896
    comptime query_output_features = 896
    comptime kv_output_features = 128
    comptime qkv_output_features = 1_152
    comptime key_offset = query_output_features
    comptime value_offset = query_output_features + kv_output_features
    comptime assert has_apple_gpu_accelerator(), "test requires an Apple GPU"

    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("test requires the Metal device API")

    var host_input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        input_features
    )
    var host_packed_weight_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](qkv_output_features * input_features)
    var host_packed_bias_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](qkv_output_features)
    var host_expected_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](qkv_output_features)
    var host_fused_output_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](qkv_output_features)
    var host_query_weight_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](query_output_features * input_features)
    var host_key_weight_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](kv_output_features * input_features)
    var host_value_weight_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](kv_output_features * input_features)
    var host_query_bias_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](query_output_features)
    var host_key_bias_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](kv_output_features)
    var host_value_bias_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](kv_output_features)
    var host_query_output_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](query_output_features)
    var host_key_output_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](kv_output_features)
    var host_value_output_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](kv_output_features)

    var host_input = TileTensor(
        host_input_buffer, row_major[rows, input_features]()
    )
    var host_packed_weight = TileTensor(
        host_packed_weight_buffer,
        row_major[qkv_output_features, input_features](),
    )
    var host_packed_bias = TileTensor(
        host_packed_bias_buffer, row_major[qkv_output_features]()
    )
    var host_expected = TileTensor(
        host_expected_buffer, row_major[rows, qkv_output_features]()
    )
    var host_query_weight = TileTensor(
        host_query_weight_buffer,
        row_major[query_output_features, input_features](),
    )
    var host_key_weight = TileTensor(
        host_key_weight_buffer,
        row_major[kv_output_features, input_features](),
    )
    var host_value_weight = TileTensor(
        host_value_weight_buffer,
        row_major[kv_output_features, input_features](),
    )
    var host_query_bias = TileTensor(
        host_query_bias_buffer, row_major[query_output_features]()
    )
    var host_key_bias = TileTensor(
        host_key_bias_buffer, row_major[kv_output_features]()
    )
    var host_value_bias = TileTensor(
        host_value_bias_buffer, row_major[kv_output_features]()
    )
    comptime assert host_packed_weight.flat_rank == 2
    comptime assert host_packed_bias.flat_rank == 1
    comptime assert host_query_weight.flat_rank == 2
    comptime assert host_key_weight.flat_rank == 2
    comptime assert host_value_weight.flat_rank == 2
    comptime assert host_query_bias.flat_rank == 1
    comptime assert host_key_bias.flat_rank == 1
    comptime assert host_value_bias.flat_rank == 1

    fill_model_shape[rows, input_features, qkv_output_features](
        host_input, host_packed_weight, host_packed_bias
    )
    for output_feature in range(query_output_features):
        for input_feature in range(input_features):
            var value = rebind[Scalar[DType.bfloat16]](
                host_packed_weight[output_feature, input_feature]
            )
            host_query_weight[output_feature, input_feature] = rebind[
                host_query_weight.ElementType
            ](value)
        var bias = rebind[Scalar[DType.bfloat16]](
            host_packed_bias[output_feature]
        )
        host_query_bias[output_feature] = rebind[host_query_bias.ElementType](
            bias
        )
    for output_feature in range(kv_output_features):
        for input_feature in range(input_features):
            var key_value = rebind[Scalar[DType.bfloat16]](
                host_packed_weight[key_offset + output_feature, input_feature]
            )
            var value_value = rebind[Scalar[DType.bfloat16]](
                host_packed_weight[value_offset + output_feature, input_feature]
            )
            host_key_weight[output_feature, input_feature] = rebind[
                host_key_weight.ElementType
            ](key_value)
            host_value_weight[output_feature, input_feature] = rebind[
                host_value_weight.ElementType
            ](value_value)
        var key_bias = rebind[Scalar[DType.bfloat16]](
            host_packed_bias[key_offset + output_feature]
        )
        var value_bias = rebind[Scalar[DType.bfloat16]](
            host_packed_bias[value_offset + output_feature]
        )
        host_key_bias[output_feature] = rebind[host_key_bias.ElementType](
            key_bias
        )
        host_value_bias[output_feature] = rebind[host_value_bias.ElementType](
            value_bias
        )
    linear_reference(
        host_input, host_packed_weight, host_packed_bias, host_expected
    )

    var device_input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        input_features
    )
    var device_packed_weight_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](qkv_output_features * input_features)
    var device_packed_bias_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](qkv_output_features)
    var device_fused_output_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](qkv_output_features)
    var device_query_weight_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](query_output_features * input_features)
    var device_key_weight_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](kv_output_features * input_features)
    var device_value_weight_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](kv_output_features * input_features)
    var device_query_bias_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](query_output_features)
    var device_key_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        kv_output_features
    )
    var device_value_bias_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](kv_output_features)
    var device_query_output_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](query_output_features)
    var device_key_output_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](kv_output_features)
    var device_value_output_buffer = context.enqueue_create_buffer[
        DType.bfloat16
    ](kv_output_features)

    context.enqueue_copy(dst_buf=device_input_buffer, src_buf=host_input_buffer)
    context.enqueue_copy(
        dst_buf=device_packed_weight_buffer,
        src_buf=host_packed_weight_buffer,
    )
    context.enqueue_copy(
        dst_buf=device_packed_bias_buffer, src_buf=host_packed_bias_buffer
    )
    context.enqueue_copy(
        dst_buf=device_query_weight_buffer, src_buf=host_query_weight_buffer
    )
    context.enqueue_copy(
        dst_buf=device_key_weight_buffer, src_buf=host_key_weight_buffer
    )
    context.enqueue_copy(
        dst_buf=device_value_weight_buffer, src_buf=host_value_weight_buffer
    )
    context.enqueue_copy(
        dst_buf=device_query_bias_buffer, src_buf=host_query_bias_buffer
    )
    context.enqueue_copy(
        dst_buf=device_key_bias_buffer, src_buf=host_key_bias_buffer
    )
    context.enqueue_copy(
        dst_buf=device_value_bias_buffer, src_buf=host_value_bias_buffer
    )

    var device_input = TileTensor(
        device_input_buffer, row_major[rows, input_features]()
    )
    var device_packed_weight = TileTensor(
        device_packed_weight_buffer,
        row_major[qkv_output_features, input_features](),
    )
    var device_packed_bias = TileTensor(
        device_packed_bias_buffer, row_major[qkv_output_features]()
    )
    var device_fused_output = TileTensor(
        device_fused_output_buffer, row_major[rows, qkv_output_features]()
    )
    var device_query_weight = TileTensor(
        device_query_weight_buffer,
        row_major[query_output_features, input_features](),
    )
    var device_key_weight = TileTensor(
        device_key_weight_buffer,
        row_major[kv_output_features, input_features](),
    )
    var device_value_weight = TileTensor(
        device_value_weight_buffer,
        row_major[kv_output_features, input_features](),
    )
    var device_query_bias = TileTensor(
        device_query_bias_buffer, row_major[query_output_features]()
    )
    var device_key_bias = TileTensor(
        device_key_bias_buffer, row_major[kv_output_features]()
    )
    var device_value_bias = TileTensor(
        device_value_bias_buffer, row_major[kv_output_features]()
    )
    var device_query_output = TileTensor(
        device_query_output_buffer, row_major[rows, query_output_features]()
    )
    var device_key_output = TileTensor(
        device_key_output_buffer, row_major[rows, kv_output_features]()
    )
    var device_value_output = TileTensor(
        device_value_output_buffer, row_major[rows, kv_output_features]()
    )

    enqueue_linear_apple_gpu(
        context,
        device_input,
        device_packed_weight,
        device_packed_bias,
        device_fused_output,
    )
    enqueue_linear_apple_gpu(
        context,
        device_input,
        device_query_weight,
        device_query_bias,
        device_query_output,
    )
    enqueue_linear_apple_gpu(
        context,
        device_input,
        device_key_weight,
        device_key_bias,
        device_key_output,
    )
    enqueue_linear_apple_gpu(
        context,
        device_input,
        device_value_weight,
        device_value_bias,
        device_value_output,
    )
    context.enqueue_copy(
        dst_buf=host_fused_output_buffer, src_buf=device_fused_output_buffer
    )
    context.enqueue_copy(
        dst_buf=host_query_output_buffer, src_buf=device_query_output_buffer
    )
    context.enqueue_copy(
        dst_buf=host_key_output_buffer, src_buf=device_key_output_buffer
    )
    context.enqueue_copy(
        dst_buf=host_value_output_buffer, src_buf=device_value_output_buffer
    )
    context.synchronize()

    var host_fused_output = TileTensor(
        host_fused_output_buffer, row_major[rows, qkv_output_features]()
    )
    var host_query_output = TileTensor(
        host_query_output_buffer, row_major[rows, query_output_features]()
    )
    var host_key_output = TileTensor(
        host_key_output_buffer, row_major[rows, kv_output_features]()
    )
    var host_value_output = TileTensor(
        host_value_output_buffer, row_major[rows, kv_output_features]()
    )
    comptime assert host_fused_output.flat_rank == 2
    comptime assert host_query_output.flat_rank == 2
    comptime assert host_key_output.flat_rank == 2
    comptime assert host_value_output.flat_rank == 2
    assert_outputs_match[rows, qkv_output_features](
        host_expected, host_fused_output
    )
    for output_feature in range(qkv_output_features):
        var fused_value = rebind[Scalar[DType.bfloat16]](
            host_fused_output[0, output_feature]
        )
        var separate_value: Scalar[DType.bfloat16]
        if output_feature < key_offset:
            separate_value = rebind[Scalar[DType.bfloat16]](
                host_query_output[0, output_feature]
            )
        elif output_feature < value_offset:
            separate_value = rebind[Scalar[DType.bfloat16]](
                host_key_output[0, output_feature - key_offset]
            )
        else:
            separate_value = rebind[Scalar[DType.bfloat16]](
                host_value_output[0, output_feature - value_offset]
            )
        if fused_value != separate_value:
            print("packed QKV mismatch at output feature", output_feature)
            raise Error("packed QKV output did not match separate projections")


def test_fixture_comparison_rejects_nan() raises:
    var context = DeviceContext()
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](1)
    var output = TileTensor(output_buffer, row_major[1, 1]())
    var nan_value: Scalar[DType.bfloat16] = FloatLiteral.nan
    output[0, 0] = rebind[output.ElementType](nan_value)
    var expected_values: List[Float32] = [0.0]

    with assert_raises(contains="requires finite values"):
        assert_matches_fixture[1, 1](output, expected_values)


def test_reference_rejects_weight_input_mismatch() raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var weight_buffer = context.enqueue_create_host_buffer[DType.bfloat16](6)
    var bias_buffer = context.enqueue_create_host_buffer[DType.bfloat16](2)
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](2)
    var input = TileTensor(input_buffer, row_major[1, 4]())
    var weight = TileTensor(weight_buffer, row_major[2, 3]())
    var bias = TileTensor(bias_buffer, row_major[2]())
    var output = TileTensor(output_buffer, row_major[1, 2]())

    with assert_raises(contains="weight input dimension"):
        linear_reference(input, weight, bias, output)


def test_reference_rejects_bias_mismatch() raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var weight_buffer = context.enqueue_create_host_buffer[DType.bfloat16](8)
    var bias_buffer = context.enqueue_create_host_buffer[DType.bfloat16](1)
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](2)
    var input = TileTensor(input_buffer, row_major[1, 4]())
    var weight = TileTensor(weight_buffer, row_major[2, 4]())
    var bias = TileTensor(bias_buffer, row_major[1]())
    var output = TileTensor(output_buffer, row_major[1, 2]())

    with assert_raises(contains="bias length"):
        linear_reference(input, weight, bias, output)


def test_reference_matches_tiny_decode_oracle() raises:
    check_reference_fixture[
        TINY_DECODE_ROWS,
        TINY_DECODE_INPUT_FEATURES,
        TINY_DECODE_OUTPUT_FEATURES,
    ](
        tiny_decode_input(),
        tiny_decode_weight(),
        tiny_decode_bias(),
        tiny_decode_expected(),
    )


def test_reference_matches_short_prefill_oracle() raises:
    check_reference_fixture[
        SHORT_PREFILL_ROWS,
        SHORT_PREFILL_INPUT_FEATURES,
        SHORT_PREFILL_OUTPUT_FEATURES,
    ](
        short_prefill_input(),
        short_prefill_weight(),
        short_prefill_bias(),
        short_prefill_expected(),
    )


def test_apple_gpu_matches_tiny_decode_oracle() raises:
    check_apple_gpu_fixture[
        TINY_DECODE_ROWS,
        TINY_DECODE_INPUT_FEATURES,
        TINY_DECODE_OUTPUT_FEATURES,
    ](
        tiny_decode_input(),
        tiny_decode_weight(),
        tiny_decode_bias(),
        tiny_decode_expected(),
    )


def test_apple_gpu_matches_short_prefill_oracle() raises:
    check_apple_gpu_fixture[
        SHORT_PREFILL_ROWS,
        SHORT_PREFILL_INPUT_FEATURES,
        SHORT_PREFILL_OUTPUT_FEATURES,
    ](
        short_prefill_input(),
        short_prefill_weight(),
        short_prefill_bias(),
        short_prefill_expected(),
    )


def test_direct_prefill_apple_gpu_matches_short_prefill_oracle() raises:
    check_apple_gpu_fixture[
        SHORT_PREFILL_ROWS,
        SHORT_PREFILL_INPUT_FEATURES,
        SHORT_PREFILL_OUTPUT_FEATURES,
        True,
    ](
        short_prefill_input(),
        short_prefill_weight(),
        short_prefill_bias(),
        short_prefill_expected(),
    )


def test_direct_prefill_apple_gpu_matches_exact_output_tile() raises:
    check_model_shape[8, 32, 16, False, True]()


def test_direct_prefill_apple_gpu_matches_tile_tails() raises:
    check_model_shape[9, 33, 17, False, True]()


def test_direct_prefill_apple_gpu_matches_qwen_kv_shape() raises:
    check_model_shape[8, 896, 128, False, True]()


def test_register_2x2_prefill_apple_gpu_matches_short_prefill_oracle() raises:
    check_apple_gpu_fixture[
        SHORT_PREFILL_ROWS,
        SHORT_PREFILL_INPUT_FEATURES,
        SHORT_PREFILL_OUTPUT_FEATURES,
        False,
        False,
        True,
    ](
        short_prefill_input(),
        short_prefill_weight(),
        short_prefill_bias(),
        short_prefill_expected(),
    )


def test_register_2x2_prefill_apple_gpu_matches_exact_output_tile() raises:
    check_model_shape[8, 32, 16, False, False, False, 32, True]()


def test_register_2x2_prefill_apple_gpu_matches_tile_tails() raises:
    check_model_shape[9, 129, 17, False, False, False, 32, True]()


def test_register_2x2_prefill_apple_gpu_matches_qwen_kv_shape() raises:
    check_model_shape[8, 896, 128, False, False, False, 32, True]()


def test_mma_8x16_prefill_apple_gpu_matches_short_prefill_oracle() raises:
    check_apple_gpu_fixture[
        SHORT_PREFILL_ROWS,
        SHORT_PREFILL_INPUT_FEATURES,
        SHORT_PREFILL_OUTPUT_FEATURES,
        False,
        False,
        False,
        True,
    ](
        short_prefill_input(),
        short_prefill_weight(),
        short_prefill_bias(),
        short_prefill_expected(),
    )


def test_mma_8x16_prefill_apple_gpu_matches_exact_output_tile() raises:
    check_model_shape[8, 32, 16, False, False, False, 32, False, True]()


def test_mma_8x16_prefill_apple_gpu_matches_tile_tails() raises:
    check_model_shape[9, 129, 17, False, False, False, 32, False, True]()


def test_mma_8x16_prefill_apple_gpu_matches_qwen_kv_shape() raises:
    check_model_shape[8, 896, 128, False, False, False, 32, False, True]()


def test_mma_8x16_prefill_apple_gpu_matches_qwen_kv_decode_shape() raises:
    check_model_shape[1, 896, 128, False, False, False, 32, False, True]()


def test_tiled_prefill_apple_gpu_matches_short_prefill_oracle() raises:
    check_apple_gpu_fixture[
        SHORT_PREFILL_ROWS,
        SHORT_PREFILL_INPUT_FEATURES,
        SHORT_PREFILL_OUTPUT_FEATURES,
        False,
        True,
    ](
        short_prefill_input(),
        short_prefill_weight(),
        short_prefill_bias(),
        short_prefill_expected(),
    )


def test_tiled_prefill_apple_gpu_matches_exact_output_tile() raises:
    check_model_shape[8, 32, 16, False, False, True]()


def test_tiled_prefill_apple_gpu_matches_tile_tails() raises:
    check_model_shape[9, 33, 17, False, False, True]()


def test_tiled_prefill_apple_gpu_matches_qwen_kv_shape() raises:
    check_model_shape[8, 896, 128, False, False, True]()


def test_tiled_prefill_apple_gpu_bk_sweep_matches_tail_shape() raises:
    check_model_shape[9, 129, 17, False, False, True, 16]()
    check_model_shape[9, 129, 17, False, False, True, 32]()
    check_model_shape[9, 129, 17, False, False, True, 64]()
    check_model_shape[9, 129, 17, False, False, True, 128]()


def test_apple_gpu_matches_qwen_query_decode_shape() raises:
    check_model_shape[1, 896, 896]()


def test_apple_gpu_matches_qwen_kv_incremental_shape() raises:
    check_model_shape[3, 896, 128]()


def test_packed_qkv_apple_gpu_matches_separate_projection_regions() raises:
    check_packed_qkv_matches_separate_projections()


def test_two_output_apple_gpu_matches_qwen_query_decode_shape() raises:
    check_model_shape[1, 896, 896, True]()


def test_two_output_apple_gpu_matches_qwen_kv_decode_shape() raises:
    check_model_shape[1, 896, 128, True]()


def test_two_output_apple_gpu_matches_odd_output_tail() raises:
    check_model_shape[1, 37, 5, True]()


def test_two_output_apple_gpu_rejects_prefill_rows() raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](2 * 8)
    var weight_buffer = context.enqueue_create_buffer[DType.bfloat16](3 * 8)
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](3)
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](2 * 3)
    var input = TileTensor(input_buffer, row_major[2, 8]())
    var weight = TileTensor(weight_buffer, row_major[3, 8]())
    var bias = TileTensor(bias_buffer, row_major[3]())
    var output = TileTensor(output_buffer, row_major[2, 3]())

    with assert_raises(contains="requires M=1"):
        enqueue_linear_apple_gpu_two_output(
            context, input, weight, bias, output
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
