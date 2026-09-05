from oracle_data.attention.reference_data import (
    ATTENTION_ATOL,
    ATTENTION_RTOL,
    QWEN_DECODE_HEAD_DIM,
    QWEN_DECODE_KEY_VALUE_HEADS,
    QWEN_DECODE_KEY_VALUE_ROWS,
    QWEN_DECODE_QUERY_HEADS,
    QWEN_DECODE_QUERY_ROWS,
    STABLE_SOFTMAX_DECODE_HEAD_DIM,
    STABLE_SOFTMAX_DECODE_KEY_VALUE_HEADS,
    STABLE_SOFTMAX_DECODE_KEY_VALUE_ROWS,
    STABLE_SOFTMAX_DECODE_QUERY_HEADS,
    STABLE_SOFTMAX_DECODE_QUERY_ROWS,
    TINY_FULL_PREFILL_HEAD_DIM,
    TINY_FULL_PREFILL_KEY_VALUE_HEADS,
    TINY_FULL_PREFILL_KEY_VALUE_ROWS,
    TINY_FULL_PREFILL_QUERY_HEADS,
    TINY_FULL_PREFILL_QUERY_ROWS,
    TINY_INCREMENTAL_PREFILL_HEAD_DIM,
    TINY_INCREMENTAL_PREFILL_KEY_VALUE_HEADS,
    TINY_INCREMENTAL_PREFILL_KEY_VALUE_ROWS,
    TINY_INCREMENTAL_PREFILL_QUERY_HEADS,
    TINY_INCREMENTAL_PREFILL_QUERY_ROWS,
    qwen_decode_expected,
    qwen_decode_key,
    qwen_decode_probabilities,
    qwen_decode_query,
    qwen_decode_value,
    stable_softmax_decode_expected,
    stable_softmax_decode_key,
    stable_softmax_decode_probabilities,
    stable_softmax_decode_query,
    stable_softmax_decode_value,
    tiny_full_prefill_expected,
    tiny_full_prefill_key,
    tiny_full_prefill_probabilities,
    tiny_full_prefill_query,
    tiny_full_prefill_value,
    tiny_incremental_prefill_expected,
    tiny_incremental_prefill_key,
    tiny_incremental_prefill_probabilities,
    tiny_incremental_prefill_query,
    tiny_incremental_prefill_value,
)
from layout import TensorLayout, TileTensor, row_major
from llm_mojo.attention import (
    enqueue_grouped_query_attention_apple_gpu,
    grouped_query_attention_reference,
)
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.sys.info import has_apple_gpu_accelerator
from std.testing import TestSuite, assert_raises


def fill_fixture[
    QueryLayout: TensorLayout,
    KeyLayout: TensorLayout,
    ValueLayout: TensorLayout,
    //,
    query_rows: Int,
    key_value_rows: Int,
    query_heads: Int,
    key_value_heads: Int,
    head_dim: Int,
](
    query: TileTensor[DType.bfloat16, QueryLayout, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KeyLayout, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, ValueLayout, MutAnyOrigin],
    query_values: List[Float32],
    key_values: List[Float32],
    value_values: List[Float32],
) raises:
    comptime assert query.flat_rank == 3
    comptime assert key.flat_rank == 3
    comptime assert value.flat_rank == 3

    if len(query_values) != query_rows * query_heads * head_dim:
        raise Error("fixture query length does not match its shape")
    if len(key_values) != key_value_rows * key_value_heads * head_dim:
        raise Error("fixture key length does not match its shape")
    if len(value_values) != key_value_rows * key_value_heads * head_dim:
        raise Error("fixture value length does not match its shape")

    for row in range(query_rows):
        for head in range(query_heads):
            for dimension in range(head_dim):
                var index = (row * query_heads + head) * head_dim + dimension
                var element = query_values[index].cast[DType.bfloat16]()
                query[row, head, dimension] = rebind[query.ElementType](element)
    for row in range(key_value_rows):
        for head in range(key_value_heads):
            for dimension in range(head_dim):
                var index = (
                    row * key_value_heads + head
                ) * head_dim + dimension
                var key_element = key_values[index].cast[DType.bfloat16]()
                var value_element = value_values[index].cast[DType.bfloat16]()
                key[row, head, dimension] = rebind[key.ElementType](key_element)
                value[row, head, dimension] = rebind[value.ElementType](
                    value_element
                )


def assert_matches_fixture[
    ActualLayout: TensorLayout,
    //,
    first_dim: Int,
    second_dim: Int,
    third_dim: Int,
](
    actual_tensor: TileTensor[DType.bfloat16, ActualLayout, MutAnyOrigin],
    expected_values: List[Float32],
) raises:
    comptime assert actual_tensor.flat_rank == 3
    if len(expected_values) != first_dim * second_dim * third_dim:
        raise Error("fixture expected length does not match its shape")

    for first in range(first_dim):
        for second in range(second_dim):
            for third in range(third_dim):
                var index = (first * second_dim + second) * third_dim + third
                var actual_bf16 = rebind[Scalar[DType.bfloat16]](
                    actual_tensor[first, second, third]
                )
                var actual = actual_bf16.cast[DType.float32]()
                var expected = expected_values[index]
                if not isfinite(actual) or not isfinite(expected):
                    raise Error(
                        "attention fixture comparison requires finite values"
                    )
                var error = actual - expected
                if error < 0.0:
                    error = -error
                var expected_magnitude = expected
                if expected_magnitude < 0.0:
                    expected_magnitude = -expected_magnitude
                var allowed = (
                    ATTENTION_ATOL + ATTENTION_RTOL * expected_magnitude
                )
                if (
                    not isfinite(error)
                    or not isfinite(allowed)
                    or error > allowed
                ):
                    print(
                        "attention mismatch at",
                        first,
                        second,
                        third,
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
                        "attention implementation did not match the oracle"
                    )


def assert_causal_probabilities[
    ScratchLayout: TensorLayout,
    //,
    query_rows: Int,
    key_value_rows: Int,
    query_heads: Int,
](scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],) raises:
    comptime assert scratch.flat_rank == 3
    var past = key_value_rows - query_rows
    for row in range(query_rows):
        var visible_key_count = past + row + 1
        for head in range(query_heads):
            var sum: Scalar[DType.float32] = 0.0
            for key_position in range(key_value_rows):
                var probability_bf16 = rebind[Scalar[DType.bfloat16]](
                    scratch[row, head, key_position]
                )
                var probability = probability_bf16.cast[DType.float32]()
                if not isfinite(probability):
                    raise Error("attention probability must be finite")
                if probability < 0.0 or probability > 1.0:
                    raise Error("attention probability must be in [0, 1]")
                if key_position >= visible_key_count and probability != 0.0:
                    raise Error("causally masked probability must be zero")
                sum += probability
            var normalization_error = sum - 1.0
            if normalization_error < 0.0:
                normalization_error = -normalization_error
            if normalization_error > 0.015625:
                raise Error("attention probabilities must sum to one")


def check_reference_fixture[
    query_rows: Int,
    key_value_rows: Int,
    query_heads: Int,
    key_value_heads: Int,
    head_dim: Int,
](
    query_values: List[Float32],
    key_values: List[Float32],
    value_values: List[Float32],
    expected_values: List[Float32],
    expected_probabilities: List[Float32],
) raises:
    var context = DeviceContext()
    var query_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        query_rows * query_heads * head_dim
    )
    var key_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        key_value_rows * key_value_heads * head_dim
    )
    var value_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        key_value_rows * key_value_heads * head_dim
    )
    var scratch_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        query_rows * query_heads * key_value_rows
    )
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        query_rows * query_heads * head_dim
    )

    var query = TileTensor(
        query_buffer, row_major[query_rows, query_heads, head_dim]()
    )
    var key = TileTensor(
        key_buffer, row_major[key_value_rows, key_value_heads, head_dim]()
    )
    var value = TileTensor(
        value_buffer, row_major[key_value_rows, key_value_heads, head_dim]()
    )
    var scratch = TileTensor(
        scratch_buffer,
        row_major[query_rows, query_heads, key_value_rows](),
    )
    var output = TileTensor(
        output_buffer, row_major[query_rows, query_heads, head_dim]()
    )
    fill_fixture[
        query_rows, key_value_rows, query_heads, key_value_heads, head_dim
    ](query, key, value, query_values, key_values, value_values)

    grouped_query_attention_reference(query, key, value, scratch, output)

    assert_matches_fixture[query_rows, query_heads, head_dim](
        output, expected_values
    )
    assert_matches_fixture[query_rows, query_heads, key_value_rows](
        scratch, expected_probabilities
    )
    assert_causal_probabilities[query_rows, key_value_rows, query_heads](
        scratch
    )


def check_apple_gpu_fixture[
    query_rows: Int,
    key_value_rows: Int,
    query_heads: Int,
    key_value_heads: Int,
    head_dim: Int,
](
    query_values: List[Float32],
    key_values: List[Float32],
    value_values: List[Float32],
    expected_values: List[Float32],
    expected_probabilities: List[Float32],
) raises:
    comptime assert has_apple_gpu_accelerator(), "test requires an Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("test requires the Metal device API")

    var host_query_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        query_rows * query_heads * head_dim
    )
    var host_key_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        key_value_rows * key_value_heads * head_dim
    )
    var host_value_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        key_value_rows * key_value_heads * head_dim
    )
    var host_scratch_buffer = context.enqueue_create_host_buffer[
        DType.bfloat16
    ](query_rows * query_heads * key_value_rows)
    var host_output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](
        query_rows * query_heads * head_dim
    )
    var device_query_buffer = context.enqueue_create_buffer[DType.bfloat16](
        query_rows * query_heads * head_dim
    )
    var device_key_buffer = context.enqueue_create_buffer[DType.bfloat16](
        key_value_rows * key_value_heads * head_dim
    )
    var device_value_buffer = context.enqueue_create_buffer[DType.bfloat16](
        key_value_rows * key_value_heads * head_dim
    )
    # These buffers stay in this caller's scope through copy-back and
    # synchronization; enqueue_grouped_query_attention_apple_gpu borrows them.
    var device_scratch_buffer = context.enqueue_create_buffer[DType.bfloat16](
        query_rows * query_heads * key_value_rows
    )
    var device_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        query_rows * query_heads * head_dim
    )

    var host_query = TileTensor(
        host_query_buffer, row_major[query_rows, query_heads, head_dim]()
    )
    var host_key = TileTensor(
        host_key_buffer,
        row_major[key_value_rows, key_value_heads, head_dim](),
    )
    var host_value = TileTensor(
        host_value_buffer,
        row_major[key_value_rows, key_value_heads, head_dim](),
    )
    fill_fixture[
        query_rows, key_value_rows, query_heads, key_value_heads, head_dim
    ](
        host_query,
        host_key,
        host_value,
        query_values,
        key_values,
        value_values,
    )
    context.enqueue_copy(dst_buf=device_query_buffer, src_buf=host_query_buffer)
    context.enqueue_copy(dst_buf=device_key_buffer, src_buf=host_key_buffer)
    context.enqueue_copy(dst_buf=device_value_buffer, src_buf=host_value_buffer)

    var device_query = TileTensor(
        device_query_buffer, row_major[query_rows, query_heads, head_dim]()
    )
    var device_key = TileTensor(
        device_key_buffer,
        row_major[key_value_rows, key_value_heads, head_dim](),
    )
    var device_value = TileTensor(
        device_value_buffer,
        row_major[key_value_rows, key_value_heads, head_dim](),
    )
    var device_scratch = TileTensor(
        device_scratch_buffer,
        row_major[query_rows, query_heads, key_value_rows](),
    )
    var device_output = TileTensor(
        device_output_buffer, row_major[query_rows, query_heads, head_dim]()
    )
    enqueue_grouped_query_attention_apple_gpu(
        context,
        device_query,
        device_key,
        device_value,
        device_scratch,
        device_output,
    )
    context.enqueue_copy(
        dst_buf=host_scratch_buffer, src_buf=device_scratch_buffer
    )
    context.enqueue_copy(
        dst_buf=host_output_buffer, src_buf=device_output_buffer
    )
    context.synchronize()

    var host_scratch = TileTensor(
        host_scratch_buffer,
        row_major[query_rows, query_heads, key_value_rows](),
    )
    var host_output = TileTensor(
        host_output_buffer, row_major[query_rows, query_heads, head_dim]()
    )
    assert_matches_fixture[query_rows, query_heads, head_dim](
        host_output, expected_values
    )
    assert_matches_fixture[query_rows, query_heads, key_value_rows](
        host_scratch, expected_probabilities
    )
    assert_causal_probabilities[query_rows, key_value_rows, query_heads](
        host_scratch
    )


def test_reference_rejects_query_longer_than_key_value() raises:
    var context = DeviceContext()
    var query_buffer = context.enqueue_create_host_buffer[DType.bfloat16](16)
    var key_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var value_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var scratch_buffer = context.enqueue_create_host_buffer[DType.bfloat16](4)
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](16)
    var query = TileTensor(query_buffer, row_major[2, 2, 4]())
    var key = TileTensor(key_buffer, row_major[1, 1, 4]())
    var value = TileTensor(value_buffer, row_major[1, 1, 4]())
    var scratch = TileTensor(scratch_buffer, row_major[2, 2, 1]())
    var output = TileTensor(output_buffer, row_major[2, 2, 4]())

    with assert_raises(contains="active key/value suffix"):
        grouped_query_attention_reference(query, key, value, scratch, output)


def test_reference_rejects_nondivisible_head_groups() raises:
    var context = DeviceContext()
    var query_buffer = context.enqueue_create_host_buffer[DType.bfloat16](12)
    var key_buffer = context.enqueue_create_host_buffer[DType.bfloat16](8)
    var value_buffer = context.enqueue_create_host_buffer[DType.bfloat16](8)
    var scratch_buffer = context.enqueue_create_host_buffer[DType.bfloat16](3)
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](12)
    var query = TileTensor(query_buffer, row_major[1, 3, 4]())
    var key = TileTensor(key_buffer, row_major[1, 2, 4]())
    var value = TileTensor(value_buffer, row_major[1, 2, 4]())
    var scratch = TileTensor(scratch_buffer, row_major[1, 3, 1]())
    var output = TileTensor(output_buffer, row_major[1, 3, 4]())

    with assert_raises(contains="divide evenly"):
        grouped_query_attention_reference(query, key, value, scratch, output)


def test_reference_rejects_wrong_scratch_shape() raises:
    var context = DeviceContext()
    var query_buffer = context.enqueue_create_host_buffer[DType.bfloat16](8)
    var key_buffer = context.enqueue_create_host_buffer[DType.bfloat16](8)
    var value_buffer = context.enqueue_create_host_buffer[DType.bfloat16](8)
    var scratch_buffer = context.enqueue_create_host_buffer[DType.bfloat16](2)
    var output_buffer = context.enqueue_create_host_buffer[DType.bfloat16](8)
    var query = TileTensor(query_buffer, row_major[1, 2, 4]())
    var key = TileTensor(key_buffer, row_major[2, 1, 4]())
    var value = TileTensor(value_buffer, row_major[2, 1, 4]())
    var scratch = TileTensor(scratch_buffer, row_major[1, 2, 1]())
    var output = TileTensor(output_buffer, row_major[1, 2, 4]())

    with assert_raises(contains="scratch shape"):
        grouped_query_attention_reference(query, key, value, scratch, output)


def test_reference_matches_tiny_full_prefill_oracle() raises:
    check_reference_fixture[
        TINY_FULL_PREFILL_QUERY_ROWS,
        TINY_FULL_PREFILL_KEY_VALUE_ROWS,
        TINY_FULL_PREFILL_QUERY_HEADS,
        TINY_FULL_PREFILL_KEY_VALUE_HEADS,
        TINY_FULL_PREFILL_HEAD_DIM,
    ](
        tiny_full_prefill_query(),
        tiny_full_prefill_key(),
        tiny_full_prefill_value(),
        tiny_full_prefill_expected(),
        tiny_full_prefill_probabilities(),
    )


def test_reference_matches_tiny_incremental_prefill_oracle() raises:
    check_reference_fixture[
        TINY_INCREMENTAL_PREFILL_QUERY_ROWS,
        TINY_INCREMENTAL_PREFILL_KEY_VALUE_ROWS,
        TINY_INCREMENTAL_PREFILL_QUERY_HEADS,
        TINY_INCREMENTAL_PREFILL_KEY_VALUE_HEADS,
        TINY_INCREMENTAL_PREFILL_HEAD_DIM,
    ](
        tiny_incremental_prefill_query(),
        tiny_incremental_prefill_key(),
        tiny_incremental_prefill_value(),
        tiny_incremental_prefill_expected(),
        tiny_incremental_prefill_probabilities(),
    )


def test_reference_matches_stable_softmax_decode_oracle() raises:
    check_reference_fixture[
        STABLE_SOFTMAX_DECODE_QUERY_ROWS,
        STABLE_SOFTMAX_DECODE_KEY_VALUE_ROWS,
        STABLE_SOFTMAX_DECODE_QUERY_HEADS,
        STABLE_SOFTMAX_DECODE_KEY_VALUE_HEADS,
        STABLE_SOFTMAX_DECODE_HEAD_DIM,
    ](
        stable_softmax_decode_query(),
        stable_softmax_decode_key(),
        stable_softmax_decode_value(),
        stable_softmax_decode_expected(),
        stable_softmax_decode_probabilities(),
    )


def test_reference_matches_qwen_decode_oracle() raises:
    check_reference_fixture[
        QWEN_DECODE_QUERY_ROWS,
        QWEN_DECODE_KEY_VALUE_ROWS,
        QWEN_DECODE_QUERY_HEADS,
        QWEN_DECODE_KEY_VALUE_HEADS,
        QWEN_DECODE_HEAD_DIM,
    ](
        qwen_decode_query(),
        qwen_decode_key(),
        qwen_decode_value(),
        qwen_decode_expected(),
        qwen_decode_probabilities(),
    )


def test_apple_gpu_matches_tiny_full_prefill_oracle() raises:
    check_apple_gpu_fixture[
        TINY_FULL_PREFILL_QUERY_ROWS,
        TINY_FULL_PREFILL_KEY_VALUE_ROWS,
        TINY_FULL_PREFILL_QUERY_HEADS,
        TINY_FULL_PREFILL_KEY_VALUE_HEADS,
        TINY_FULL_PREFILL_HEAD_DIM,
    ](
        tiny_full_prefill_query(),
        tiny_full_prefill_key(),
        tiny_full_prefill_value(),
        tiny_full_prefill_expected(),
        tiny_full_prefill_probabilities(),
    )


def test_apple_gpu_matches_tiny_incremental_prefill_oracle() raises:
    check_apple_gpu_fixture[
        TINY_INCREMENTAL_PREFILL_QUERY_ROWS,
        TINY_INCREMENTAL_PREFILL_KEY_VALUE_ROWS,
        TINY_INCREMENTAL_PREFILL_QUERY_HEADS,
        TINY_INCREMENTAL_PREFILL_KEY_VALUE_HEADS,
        TINY_INCREMENTAL_PREFILL_HEAD_DIM,
    ](
        tiny_incremental_prefill_query(),
        tiny_incremental_prefill_key(),
        tiny_incremental_prefill_value(),
        tiny_incremental_prefill_expected(),
        tiny_incremental_prefill_probabilities(),
    )


def test_apple_gpu_matches_stable_softmax_decode_oracle() raises:
    check_apple_gpu_fixture[
        STABLE_SOFTMAX_DECODE_QUERY_ROWS,
        STABLE_SOFTMAX_DECODE_KEY_VALUE_ROWS,
        STABLE_SOFTMAX_DECODE_QUERY_HEADS,
        STABLE_SOFTMAX_DECODE_KEY_VALUE_HEADS,
        STABLE_SOFTMAX_DECODE_HEAD_DIM,
    ](
        stable_softmax_decode_query(),
        stable_softmax_decode_key(),
        stable_softmax_decode_value(),
        stable_softmax_decode_expected(),
        stable_softmax_decode_probabilities(),
    )


def test_apple_gpu_matches_qwen_decode_oracle() raises:
    check_apple_gpu_fixture[
        QWEN_DECODE_QUERY_ROWS,
        QWEN_DECODE_KEY_VALUE_ROWS,
        QWEN_DECODE_QUERY_HEADS,
        QWEN_DECODE_KEY_VALUE_HEADS,
        QWEN_DECODE_HEAD_DIM,
    ](
        qwen_decode_query(),
        qwen_decode_key(),
        qwen_decode_value(),
        qwen_decode_expected(),
        qwen_decode_probabilities(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
