"""Reproducible Apple GPU batch-one linear projection benchmark."""

from layout import TensorLayout, TileTensor, row_major
from llm_mojo.linear import (
    enqueue_linear_apple_gpu,
    enqueue_linear_apple_gpu_two_output,
)
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    Format,
    ThroughputMeasure,
)
from std.sys import get_defined_bool, get_defined_int, is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.time import sleep


comptime ROWS = 1
comptime INPUT_FEATURES = 896
comptime QUERY_OUTPUT_FEATURES = 896
comptime KV_OUTPUT_FEATURES = 128
comptime QKV_OUTPUT_FEATURES = 1_152
comptime RING_LAYERS = 24
comptime BENCHMARK_WARMUP_ITERATIONS = 100
comptime BENCHMARK_MAX_ITERATIONS = 100
comptime BENCHMARK_REPETITIONS = 10
comptime DEFAULT_PROFILE_WARMUP_ITERATIONS = 100
comptime DEFAULT_PROFILE_POST_IDLE_MILLISECONDS = 0

comptime QUERY_WORKLOAD = 0
comptime KV_WORKLOAD = 1
comptime QKV_HOT_WORKLOAD = 2
comptime QKV_RING24_WORKLOAD = 3


@always_inline
def _enqueue_projection[
    use_two_output: Bool,
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
    comptime if use_two_output:
        enqueue_linear_apple_gpu_two_output(
            context, input, weight, bias, output
        )
    else:
        enqueue_linear_apple_gpu(context, input, weight, bias, output)


def _assert_unit_output[
    OutputLayout: TensorLayout
](output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin]) raises:
    comptime assert output.flat_rank == 2, "expected rank-2 output"
    for row in range(Int(output.dim[0]())):
        for output_feature in range(Int(output.dim[1]())):
            var actual = rebind[Scalar[DType.bfloat16]](
                output[row, output_feature]
            )
            if actual != Scalar[DType.bfloat16](1.0):
                raise Error("linear benchmark correctness gate failed")


@always_inline
def bench_linear[
    output_features: Int, use_two_output: Bool
](mut bencher: Bencher) raises capturing:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * INPUT_FEATURES
    )
    var weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features * INPUT_FEATURES
    )
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * output_features
    )
    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime weight_layout = row_major[output_features, INPUT_FEATURES]()
    comptime bias_layout = row_major[output_features]()
    comptime output_layout = row_major[ROWS, output_features]()
    var input = TileTensor(input_buffer, input_layout)
    var weight = TileTensor(weight_buffer, weight_layout)
    var bias = TileTensor(bias_buffer, bias_layout)
    var output = TileTensor(output_buffer, output_layout)
    input_buffer.enqueue_fill(0.5)
    weight_buffer.enqueue_fill(0.001953125)
    bias_buffer.enqueue_fill(0.125)

    _enqueue_projection[use_two_output](context, input, weight, bias, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        _enqueue_projection[use_two_output](
            launch_context, input, weight, bias, output
        )

    bencher_iter_custom(bencher, launch, context)


@always_inline
def bench_qkv_hot[use_two_output: Bool](mut bencher: Bencher) raises capturing:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * INPUT_FEATURES
    )
    var query_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var key_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var value_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var query_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var query_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )

    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime query_weight_layout = row_major[
        QUERY_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime kv_weight_layout = row_major[KV_OUTPUT_FEATURES, INPUT_FEATURES]()
    comptime query_bias_layout = row_major[QUERY_OUTPUT_FEATURES]()
    comptime kv_bias_layout = row_major[KV_OUTPUT_FEATURES]()
    comptime query_output_layout = row_major[ROWS, QUERY_OUTPUT_FEATURES]()
    comptime kv_output_layout = row_major[ROWS, KV_OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var query_weight = TileTensor(query_weight_buffer, query_weight_layout)
    var key_weight = TileTensor(key_weight_buffer, kv_weight_layout)
    var value_weight = TileTensor(value_weight_buffer, kv_weight_layout)
    var query_bias = TileTensor(query_bias_buffer, query_bias_layout)
    var key_bias = TileTensor(key_bias_buffer, kv_bias_layout)
    var value_bias = TileTensor(value_bias_buffer, kv_bias_layout)
    var query_output = TileTensor(query_output_buffer, query_output_layout)
    var key_output = TileTensor(key_output_buffer, kv_output_layout)
    var value_output = TileTensor(value_output_buffer, kv_output_layout)
    input_buffer.enqueue_fill(0.5)
    query_weight_buffer.enqueue_fill(0.001953125)
    key_weight_buffer.enqueue_fill(0.001953125)
    value_weight_buffer.enqueue_fill(0.001953125)
    query_bias_buffer.enqueue_fill(0.125)
    key_bias_buffer.enqueue_fill(0.125)
    value_bias_buffer.enqueue_fill(0.125)

    _enqueue_projection[use_two_output](
        context, input, query_weight, query_bias, query_output
    )
    _enqueue_projection[use_two_output](
        context, input, key_weight, key_bias, key_output
    )
    _enqueue_projection[use_two_output](
        context, input, value_weight, value_bias, value_output
    )
    with query_output_buffer.map_to_host() as mapped_query:
        var host_query = TileTensor(mapped_query, query_output_layout)
        _assert_unit_output(host_query)
    with key_output_buffer.map_to_host() as mapped_key:
        var host_key = TileTensor(mapped_key, kv_output_layout)
        _assert_unit_output(host_key)
    with value_output_buffer.map_to_host() as mapped_value:
        var host_value = TileTensor(mapped_value, kv_output_layout)
        _assert_unit_output(host_value)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        _enqueue_projection[use_two_output](
            launch_context, input, query_weight, query_bias, query_output
        )
        _enqueue_projection[use_two_output](
            launch_context, input, key_weight, key_bias, key_output
        )
        _enqueue_projection[use_two_output](
            launch_context, input, value_weight, value_bias, value_output
        )

    bencher_iter_custom(bencher, launch, context)


@always_inline
def bench_qkv_fused_hot(mut bencher: Bencher) raises capturing:
    """Measure one prepacked QKV projection enqueue."""

    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * INPUT_FEATURES
    )
    var weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QKV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QKV_OUTPUT_FEATURES
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * QKV_OUTPUT_FEATURES
    )

    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime weight_layout = row_major[QKV_OUTPUT_FEATURES, INPUT_FEATURES]()
    comptime bias_layout = row_major[QKV_OUTPUT_FEATURES]()
    comptime output_layout = row_major[ROWS, QKV_OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var weight = TileTensor(weight_buffer, weight_layout)
    var bias = TileTensor(bias_buffer, bias_layout)
    var output = TileTensor(output_buffer, output_layout)
    input_buffer.enqueue_fill(0.5)
    weight_buffer.enqueue_fill(0.001953125)
    bias_buffer.enqueue_fill(0.125)

    enqueue_linear_apple_gpu(context, input, weight, bias, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        enqueue_linear_apple_gpu(launch_context, input, weight, bias, output)

    bencher_iter_custom(bencher, launch, context)


@always_inline
def bench_qkv_ring24[
    use_two_output: Bool
](mut bencher: Bencher) raises capturing:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * INPUT_FEATURES
    )
    var query_weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * QUERY_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var key_weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var value_weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var query_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var query_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )

    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime query_weights_layout = row_major[
        RING_LAYERS * QUERY_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime kv_weights_layout = row_major[
        RING_LAYERS * KV_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime query_bias_layout = row_major[QUERY_OUTPUT_FEATURES]()
    comptime kv_bias_layout = row_major[KV_OUTPUT_FEATURES]()
    comptime query_output_layout = row_major[ROWS, QUERY_OUTPUT_FEATURES]()
    comptime kv_output_layout = row_major[ROWS, KV_OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var query_weights = TileTensor(query_weights_buffer, query_weights_layout)
    var key_weights = TileTensor(key_weights_buffer, kv_weights_layout)
    var value_weights = TileTensor(value_weights_buffer, kv_weights_layout)
    var query_bias = TileTensor(query_bias_buffer, query_bias_layout)
    var key_bias = TileTensor(key_bias_buffer, kv_bias_layout)
    var value_bias = TileTensor(value_bias_buffer, kv_bias_layout)
    var query_output = TileTensor(query_output_buffer, query_output_layout)
    var key_output = TileTensor(key_output_buffer, kv_output_layout)
    var value_output = TileTensor(value_output_buffer, kv_output_layout)
    input_buffer.enqueue_fill(0.5)
    query_weights_buffer.enqueue_fill(0.001953125)
    key_weights_buffer.enqueue_fill(0.001953125)
    value_weights_buffer.enqueue_fill(0.001953125)
    query_bias_buffer.enqueue_fill(0.125)
    key_bias_buffer.enqueue_fill(0.125)
    value_bias_buffer.enqueue_fill(0.125)

    comptime for layer in range(RING_LAYERS):
        var query_weight = query_weights.tile[
            QUERY_OUTPUT_FEATURES, INPUT_FEATURES
        ](layer, 0)
        var key_weight = key_weights.tile[KV_OUTPUT_FEATURES, INPUT_FEATURES](
            layer, 0
        )
        var value_weight = value_weights.tile[
            KV_OUTPUT_FEATURES, INPUT_FEATURES
        ](layer, 0)
        _enqueue_projection[use_two_output](
            context, input, query_weight, query_bias, query_output
        )
        _enqueue_projection[use_two_output](
            context, input, key_weight, key_bias, key_output
        )
        _enqueue_projection[use_two_output](
            context, input, value_weight, value_bias, value_output
        )
    with query_output_buffer.map_to_host() as mapped_query:
        var host_query = TileTensor(mapped_query, query_output_layout)
        _assert_unit_output(host_query)
    with key_output_buffer.map_to_host() as mapped_key:
        var host_key = TileTensor(mapped_key, kv_output_layout)
        _assert_unit_output(host_key)
    with value_output_buffer.map_to_host() as mapped_value:
        var host_value = TileTensor(mapped_value, kv_output_layout)
        _assert_unit_output(host_value)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        comptime for layer in range(RING_LAYERS):
            var query_weight = query_weights.tile[
                QUERY_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            var key_weight = key_weights.tile[
                KV_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            var value_weight = value_weights.tile[
                KV_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            _enqueue_projection[use_two_output](
                launch_context, input, query_weight, query_bias, query_output
            )
            _enqueue_projection[use_two_output](
                launch_context, input, key_weight, key_bias, key_output
            )
            _enqueue_projection[use_two_output](
                launch_context,
                input,
                value_weight,
                value_bias,
                value_output,
            )

    bencher_iter_custom(bencher, launch, context)


@always_inline
def bench_qkv_fused_ring24(mut bencher: Bencher) raises capturing:
    """Measure 24 prepacked QKV projection enqueues with rotating weights."""

    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * INPUT_FEATURES
    )
    var weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * QKV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QKV_OUTPUT_FEATURES
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        ROWS * QKV_OUTPUT_FEATURES
    )

    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime weights_layout = row_major[
        RING_LAYERS * QKV_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime bias_layout = row_major[QKV_OUTPUT_FEATURES]()
    comptime output_layout = row_major[ROWS, QKV_OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var weights = TileTensor(weights_buffer, weights_layout)
    var bias = TileTensor(bias_buffer, bias_layout)
    var output = TileTensor(output_buffer, output_layout)
    input_buffer.enqueue_fill(0.5)
    weights_buffer.enqueue_fill(0.001953125)
    bias_buffer.enqueue_fill(0.125)

    comptime for layer in range(RING_LAYERS):
        var weight = weights.tile[QKV_OUTPUT_FEATURES, INPUT_FEATURES](layer, 0)
        enqueue_linear_apple_gpu(context, input, weight, bias, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        comptime for layer in range(RING_LAYERS):
            var weight = weights.tile[QKV_OUTPUT_FEATURES, INPUT_FEATURES](
                layer, 0
            )
            enqueue_linear_apple_gpu(
                launch_context, input, weight, bias, output
            )

    bencher_iter_custom(bencher, launch, context)


def _add_linear_benchmark[
    output_features: Int, use_two_output: Bool
](mut benchmark: Bench, name: String) raises:
    comptime workload = bench_linear[output_features, use_two_output]
    comptime benchmark_name = (
        "linear_decode_apple_gpu_two_output" if use_two_output else "linear_decode_apple_gpu"
    )
    benchmark.bench_function[workload](
        BenchId(benchmark_name, name),
        [
            ThroughputMeasure(
                BenchMetric.bytes,
                2
                * (
                    INPUT_FEATURES
                    + output_features * INPUT_FEATURES
                    + 2 * output_features
                ),
            )
        ],
    )


def _add_qkv_hot_benchmark[use_two_output: Bool](mut benchmark: Bench) raises:
    comptime workload = bench_qkv_hot[use_two_output]
    comptime benchmark_name = (
        "linear_decode_qkv3_apple_gpu_two_output" if use_two_output else "linear_decode_qkv3_apple_gpu"
    )
    benchmark.bench_function[workload](
        BenchId(benchmark_name, "qkv3-hot-m1-k896-n1152"),
        [
            ThroughputMeasure(
                BenchMetric.bytes,
                2
                * (
                    INPUT_FEATURES
                    + QKV_OUTPUT_FEATURES * INPUT_FEATURES
                    + 2 * QKV_OUTPUT_FEATURES
                ),
            )
        ],
    )


def _add_qkv_ring24_benchmark[
    use_two_output: Bool
](mut benchmark: Bench) raises:
    comptime workload = bench_qkv_ring24[use_two_output]
    comptime benchmark_name = (
        "linear_decode_qkv3_ring24_apple_gpu_two_output" if use_two_output else "linear_decode_qkv3_ring24_apple_gpu"
    )
    benchmark.bench_function[workload](
        BenchId(
            benchmark_name,
            "qkv3-ring24-m1-k896-n1152-layers24",
        ),
        [
            ThroughputMeasure(
                BenchMetric.bytes,
                2
                * (
                    INPUT_FEATURES
                    + RING_LAYERS * QKV_OUTPUT_FEATURES * INPUT_FEATURES
                    + 2 * QKV_OUTPUT_FEATURES
                ),
            )
        ],
    )


def _add_qkv_fused_hot_benchmark(mut benchmark: Bench) raises:
    comptime workload = bench_qkv_fused_hot
    benchmark.bench_function[workload](
        BenchId(
            "linear_decode_qkv3_apple_gpu_fused",
            "qkv3-hot-m1-k896-n1152",
        ),
        [
            ThroughputMeasure(
                BenchMetric.bytes,
                2
                * (
                    INPUT_FEATURES
                    + QKV_OUTPUT_FEATURES * INPUT_FEATURES
                    + 2 * QKV_OUTPUT_FEATURES
                ),
            )
        ],
    )


def _add_qkv_fused_ring24_benchmark(mut benchmark: Bench) raises:
    comptime workload = bench_qkv_fused_ring24
    benchmark.bench_function[workload](
        BenchId(
            "linear_decode_qkv3_ring24_apple_gpu_fused",
            "qkv3-ring24-m1-k896-n1152-layers24",
        ),
        [
            ThroughputMeasure(
                BenchMetric.bytes,
                2
                * (
                    INPUT_FEATURES
                    + RING_LAYERS * QKV_OUTPUT_FEATURES * INPUT_FEATURES
                    + 2 * QKV_OUTPUT_FEATURES
                ),
            )
        ],
    )


def _add_selected_linear_benchmarks[
    output_features: Int,
    variant_comparison: Bool,
    variant_first: Bool,
](mut benchmark: Bench, name: String) raises:
    comptime if variant_comparison:
        comptime if variant_first:
            _add_linear_benchmark[output_features, True](benchmark, name)
            _add_linear_benchmark[output_features, False](benchmark, name)
        else:
            _add_linear_benchmark[output_features, False](benchmark, name)
            _add_linear_benchmark[output_features, True](benchmark, name)
    else:
        _add_linear_benchmark[output_features, False](benchmark, name)


def _add_selected_qkv_hot_benchmarks[
    variant_comparison: Bool, variant_first: Bool
](mut benchmark: Bench) raises:
    comptime if variant_comparison:
        comptime if variant_first:
            _add_qkv_hot_benchmark[True](benchmark)
            _add_qkv_hot_benchmark[False](benchmark)
        else:
            _add_qkv_hot_benchmark[False](benchmark)
            _add_qkv_hot_benchmark[True](benchmark)
    else:
        _add_qkv_hot_benchmark[False](benchmark)


def _add_selected_qkv_ring24_benchmarks[
    variant_comparison: Bool, variant_first: Bool
](mut benchmark: Bench) raises:
    comptime if variant_comparison:
        comptime if variant_first:
            _add_qkv_ring24_benchmark[True](benchmark)
            _add_qkv_ring24_benchmark[False](benchmark)
        else:
            _add_qkv_ring24_benchmark[False](benchmark)
            _add_qkv_ring24_benchmark[True](benchmark)
    else:
        _add_qkv_ring24_benchmark[False](benchmark)


def _add_qkv_hot_fusion_comparison[
    candidate_first: Bool
](mut benchmark: Bench) raises:
    comptime if candidate_first:
        _add_qkv_fused_hot_benchmark(benchmark)
        _add_qkv_hot_benchmark[False](benchmark)
    else:
        _add_qkv_hot_benchmark[False](benchmark)
        _add_qkv_fused_hot_benchmark(benchmark)


def _add_qkv_ring24_fusion_comparison[
    candidate_first: Bool
](mut benchmark: Bench) raises:
    comptime if candidate_first:
        _add_qkv_fused_ring24_benchmark(benchmark)
        _add_qkv_ring24_benchmark[False](benchmark)
    else:
        _add_qkv_ring24_benchmark[False](benchmark)
        _add_qkv_fused_ring24_benchmark(benchmark)


def run_benchmarks() raises:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    comptime variant_comparison = get_defined_bool[
        "LINEAR_BENCH_VARIANT_COMPARISON"
    ]()
    comptime qkv_fusion_comparison = get_defined_bool[
        "LINEAR_BENCH_QKV_FUSION_COMPARISON"
    ]()
    if variant_comparison and qkv_fusion_comparison:
        raise Error("linear benchmark comparison modes are mutually exclusive")
    comptime variant_first = get_defined_bool["LINEAR_BENCH_VARIANT_FIRST"]()
    print("implementation: enqueue_linear_apple_gpu")
    comptime if variant_comparison:
        print("comparison implementation: enqueue_linear_apple_gpu_two_output")
    elif qkv_fusion_comparison:
        print("comparison implementation: enqueue_linear_apple_gpu")
    print("device:", identity.name())
    print("api:", identity.api())
    print("dtype: bfloat16; accumulation: float32")
    print("rows: 1; input features: 896")
    print("timing: enqueue through synchronized device completion")
    print("allocation, initialization, correctness, and mapping excluded")
    comptime if qkv_fusion_comparison:
        print("bytes metric: comparison-normalized logical QKV tensor bytes")
    else:
        print("bytes metric: source-derived unique tensor bytes")
    print("bytes metric is not measured hardware traffic")
    print(
        "benchmark config:",
        BENCHMARK_WARMUP_ITERATIONS,
        "warmup iterations;",
        BENCHMARK_MAX_ITERATIONS,
        "max iterations;",
        BENCHMARK_REPETITIONS,
        "repetitions",
    )
    comptime reverse_order = get_defined_bool["LINEAR_BENCH_REVERSE"]()
    print("workload order:", "descending" if reverse_order else "ascending")
    comptime if variant_comparison:
        print(
            "implementation order:",
            "variant then baseline" if variant_first else (
                "baseline then variant"
            ),
        )

    var benchmark = Bench(
        BenchConfig(
            num_warmup_iters=BENCHMARK_WARMUP_ITERATIONS,
            max_iters=BENCHMARK_MAX_ITERATIONS,
            num_repetitions=BENCHMARK_REPETITIONS,
        )
    )
    comptime if reverse_order:
        comptime if qkv_fusion_comparison:
            _add_qkv_ring24_fusion_comparison[variant_first](benchmark)
            _add_qkv_hot_fusion_comparison[variant_first](benchmark)
        else:
            _add_selected_qkv_ring24_benchmarks[
                variant_comparison, variant_first
            ](benchmark)
            _add_selected_qkv_hot_benchmarks[variant_comparison, variant_first](
                benchmark
            )
            _add_selected_linear_benchmarks[
                QUERY_OUTPUT_FEATURES, variant_comparison, variant_first
            ](benchmark, "q-m1-k896-n896")
            _add_selected_linear_benchmarks[
                KV_OUTPUT_FEATURES, variant_comparison, variant_first
            ](benchmark, "kv-m1-k896-n128")
    else:
        comptime if qkv_fusion_comparison:
            _add_qkv_hot_fusion_comparison[variant_first](benchmark)
            _add_qkv_ring24_fusion_comparison[variant_first](benchmark)
        else:
            _add_selected_linear_benchmarks[
                KV_OUTPUT_FEATURES, variant_comparison, variant_first
            ](benchmark, "kv-m1-k896-n128")
            _add_selected_linear_benchmarks[
                QUERY_OUTPUT_FEATURES, variant_comparison, variant_first
            ](benchmark, "q-m1-k896-n896")
            _add_selected_qkv_hot_benchmarks[variant_comparison, variant_first](
                benchmark
            )
            _add_selected_qkv_ring24_benchmarks[
                variant_comparison, variant_first
            ](benchmark)

    benchmark.config.format = Format.tabular
    print("BENCHMARK_RESULTS_BEGIN")
    print(benchmark)
    print("BENCHMARK_RESULTS_END")


def _run_profile_linear[
    output_features: Int,
    warmup_iterations: Int,
    iterations: Int,
    use_two_output: Bool,
](post_profile_idle_milliseconds: Int) raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        INPUT_FEATURES
    )
    var weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features * INPUT_FEATURES
    )
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features
    )
    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime weight_layout = row_major[output_features, INPUT_FEATURES]()
    comptime bias_layout = row_major[output_features]()
    comptime output_layout = row_major[ROWS, output_features]()
    var input = TileTensor(input_buffer, input_layout)
    var weight = TileTensor(weight_buffer, weight_layout)
    var bias = TileTensor(bias_buffer, bias_layout)
    var output = TileTensor(output_buffer, output_layout)
    input_buffer.enqueue_fill(0.5)
    weight_buffer.enqueue_fill(0.001953125)
    bias_buffer.enqueue_fill(0.125)
    _enqueue_projection[use_two_output](context, input, weight, bias, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    for _ in range(warmup_iterations):
        _enqueue_projection[use_two_output](
            context, input, weight, bias, output
        )
    context.synchronize()
    comptime if use_two_output:
        print("profile implementation: enqueue_linear_apple_gpu_two_output")
    else:
        print("profile implementation: enqueue_linear_apple_gpu")
    print("device:", context.name())
    print("api:", context.api())
    print("rows:", ROWS)
    print("hidden:", INPUT_FEATURES)
    print("output features:", output_features)
    comptime if output_features == QUERY_OUTPUT_FEATURES:
        print("profile workload: q")
    else:
        print("profile workload: kv")
    print("profile dispatches per iteration: 1")
    print("warmup iterations:", warmup_iterations)
    print("profile iterations:", iterations)
    print("post-profile idle milliseconds:", post_profile_idle_milliseconds)
    print("PROFILE_REGION_BEGIN")
    for _ in range(iterations):
        _enqueue_projection[use_two_output](
            context, input, weight, bias, output
        )
    context.synchronize()
    print("PROFILE_REGION_END")
    if post_profile_idle_milliseconds > 0:
        sleep(Float64(post_profile_idle_milliseconds) / 1_000.0)


def _run_profile_qkv_fused[
    layers: Int,
    warmup_iterations: Int,
    iterations: Int,
](post_profile_idle_milliseconds: Int) raises:
    comptime assert (
        layers == 1 or layers == RING_LAYERS
    ), "fused QKV profile supports one or 24 layers"
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        INPUT_FEATURES
    )
    var weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        layers * QKV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QKV_OUTPUT_FEATURES
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QKV_OUTPUT_FEATURES
    )
    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime weights_layout = row_major[
        layers * QKV_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime bias_layout = row_major[QKV_OUTPUT_FEATURES]()
    comptime output_layout = row_major[ROWS, QKV_OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var weights = TileTensor(weights_buffer, weights_layout)
    var bias = TileTensor(bias_buffer, bias_layout)
    var output = TileTensor(output_buffer, output_layout)
    input_buffer.enqueue_fill(0.5)
    weights_buffer.enqueue_fill(0.001953125)
    bias_buffer.enqueue_fill(0.125)

    comptime if layers == 1:
        enqueue_linear_apple_gpu(context, input, weights, bias, output)
    else:
        comptime for layer in range(layers):
            var weight = weights.tile[QKV_OUTPUT_FEATURES, INPUT_FEATURES](
                layer, 0
            )
            enqueue_linear_apple_gpu(context, input, weight, bias, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    for _ in range(warmup_iterations):
        comptime if layers == 1:
            enqueue_linear_apple_gpu(context, input, weights, bias, output)
        else:
            comptime for layer in range(layers):
                var weight = weights.tile[QKV_OUTPUT_FEATURES, INPUT_FEATURES](
                    layer, 0
                )
                enqueue_linear_apple_gpu(context, input, weight, bias, output)
    context.synchronize()
    print("profile implementation: enqueue_linear_apple_gpu")
    print("device:", context.name())
    print("api:", context.api())
    print("rows:", ROWS)
    print("hidden:", INPUT_FEATURES)
    print("output features:", QKV_OUTPUT_FEATURES)
    comptime if layers == 1:
        print("profile workload: qkv-hot")
    else:
        print("profile workload: qkv-ring24")
    print("profile dispatches per iteration:", layers)
    print("warmup iterations:", warmup_iterations)
    print("profile iterations:", iterations)
    print("post-profile idle milliseconds:", post_profile_idle_milliseconds)
    print("PROFILE_REGION_BEGIN")
    for _ in range(iterations):
        comptime if layers == 1:
            enqueue_linear_apple_gpu(context, input, weights, bias, output)
        else:
            comptime for layer in range(layers):
                var weight = weights.tile[QKV_OUTPUT_FEATURES, INPUT_FEATURES](
                    layer, 0
                )
                enqueue_linear_apple_gpu(context, input, weight, bias, output)
    context.synchronize()
    print("PROFILE_REGION_END")
    if post_profile_idle_milliseconds > 0:
        sleep(Float64(post_profile_idle_milliseconds) / 1_000.0)


def _run_profile_qkv_hot[
    warmup_iterations: Int, iterations: Int, use_two_output: Bool
](post_profile_idle_milliseconds: Int) raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        INPUT_FEATURES
    )
    var query_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var key_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var value_weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var query_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var query_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime query_weight_layout = row_major[
        QUERY_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime kv_weight_layout = row_major[KV_OUTPUT_FEATURES, INPUT_FEATURES]()
    comptime query_bias_layout = row_major[QUERY_OUTPUT_FEATURES]()
    comptime kv_bias_layout = row_major[KV_OUTPUT_FEATURES]()
    comptime query_output_layout = row_major[ROWS, QUERY_OUTPUT_FEATURES]()
    comptime kv_output_layout = row_major[ROWS, KV_OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var query_weight = TileTensor(query_weight_buffer, query_weight_layout)
    var key_weight = TileTensor(key_weight_buffer, kv_weight_layout)
    var value_weight = TileTensor(value_weight_buffer, kv_weight_layout)
    var query_bias = TileTensor(query_bias_buffer, query_bias_layout)
    var key_bias = TileTensor(key_bias_buffer, kv_bias_layout)
    var value_bias = TileTensor(value_bias_buffer, kv_bias_layout)
    var query_output = TileTensor(query_output_buffer, query_output_layout)
    var key_output = TileTensor(key_output_buffer, kv_output_layout)
    var value_output = TileTensor(value_output_buffer, kv_output_layout)
    input_buffer.enqueue_fill(0.5)
    query_weight_buffer.enqueue_fill(0.001953125)
    key_weight_buffer.enqueue_fill(0.001953125)
    value_weight_buffer.enqueue_fill(0.001953125)
    query_bias_buffer.enqueue_fill(0.125)
    key_bias_buffer.enqueue_fill(0.125)
    value_bias_buffer.enqueue_fill(0.125)

    _enqueue_projection[use_two_output](
        context, input, query_weight, query_bias, query_output
    )
    _enqueue_projection[use_two_output](
        context, input, key_weight, key_bias, key_output
    )
    _enqueue_projection[use_two_output](
        context, input, value_weight, value_bias, value_output
    )
    with query_output_buffer.map_to_host() as mapped_query:
        var host_query = TileTensor(mapped_query, query_output_layout)
        _assert_unit_output(host_query)
    with key_output_buffer.map_to_host() as mapped_key:
        var host_key = TileTensor(mapped_key, kv_output_layout)
        _assert_unit_output(host_key)
    with value_output_buffer.map_to_host() as mapped_value:
        var host_value = TileTensor(mapped_value, kv_output_layout)
        _assert_unit_output(host_value)

    for _ in range(warmup_iterations):
        _enqueue_projection[use_two_output](
            context, input, query_weight, query_bias, query_output
        )
        _enqueue_projection[use_two_output](
            context, input, key_weight, key_bias, key_output
        )
        _enqueue_projection[use_two_output](
            context, input, value_weight, value_bias, value_output
        )
    context.synchronize()
    comptime if use_two_output:
        print("profile implementation: enqueue_linear_apple_gpu_two_output")
    else:
        print("profile implementation: enqueue_linear_apple_gpu")
    print("device:", context.name())
    print("api:", context.api())
    print("rows:", ROWS)
    print("hidden:", INPUT_FEATURES)
    print("output features:", QKV_OUTPUT_FEATURES)
    print("profile workload: qkv-hot")
    print("profile dispatches per iteration: 3")
    print("warmup iterations:", warmup_iterations)
    print("profile iterations:", iterations)
    print("post-profile idle milliseconds:", post_profile_idle_milliseconds)
    print("PROFILE_REGION_BEGIN")
    for _ in range(iterations):
        _enqueue_projection[use_two_output](
            context, input, query_weight, query_bias, query_output
        )
        _enqueue_projection[use_two_output](
            context, input, key_weight, key_bias, key_output
        )
        _enqueue_projection[use_two_output](
            context, input, value_weight, value_bias, value_output
        )
    context.synchronize()
    print("PROFILE_REGION_END")
    if post_profile_idle_milliseconds > 0:
        sleep(Float64(post_profile_idle_milliseconds) / 1_000.0)


def _run_profile_qkv_ring24[
    warmup_iterations: Int, iterations: Int, use_two_output: Bool
](post_profile_idle_milliseconds: Int) raises:
    var context = DeviceContext()
    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        INPUT_FEATURES
    )
    var query_weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * QUERY_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var key_weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var value_weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * KV_OUTPUT_FEATURES * INPUT_FEATURES
    )
    var query_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var query_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        QUERY_OUTPUT_FEATURES
    )
    var key_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    var value_output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        KV_OUTPUT_FEATURES
    )
    comptime input_layout = row_major[ROWS, INPUT_FEATURES]()
    comptime query_weights_layout = row_major[
        RING_LAYERS * QUERY_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime kv_weights_layout = row_major[
        RING_LAYERS * KV_OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime query_bias_layout = row_major[QUERY_OUTPUT_FEATURES]()
    comptime kv_bias_layout = row_major[KV_OUTPUT_FEATURES]()
    comptime query_output_layout = row_major[ROWS, QUERY_OUTPUT_FEATURES]()
    comptime kv_output_layout = row_major[ROWS, KV_OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var query_weights = TileTensor(query_weights_buffer, query_weights_layout)
    var key_weights = TileTensor(key_weights_buffer, kv_weights_layout)
    var value_weights = TileTensor(value_weights_buffer, kv_weights_layout)
    var query_bias = TileTensor(query_bias_buffer, query_bias_layout)
    var key_bias = TileTensor(key_bias_buffer, kv_bias_layout)
    var value_bias = TileTensor(value_bias_buffer, kv_bias_layout)
    var query_output = TileTensor(query_output_buffer, query_output_layout)
    var key_output = TileTensor(key_output_buffer, kv_output_layout)
    var value_output = TileTensor(value_output_buffer, kv_output_layout)
    input_buffer.enqueue_fill(0.5)
    query_weights_buffer.enqueue_fill(0.001953125)
    key_weights_buffer.enqueue_fill(0.001953125)
    value_weights_buffer.enqueue_fill(0.001953125)
    query_bias_buffer.enqueue_fill(0.125)
    key_bias_buffer.enqueue_fill(0.125)
    value_bias_buffer.enqueue_fill(0.125)

    comptime for layer in range(RING_LAYERS):
        var query_weight = query_weights.tile[
            QUERY_OUTPUT_FEATURES, INPUT_FEATURES
        ](layer, 0)
        var key_weight = key_weights.tile[KV_OUTPUT_FEATURES, INPUT_FEATURES](
            layer, 0
        )
        var value_weight = value_weights.tile[
            KV_OUTPUT_FEATURES, INPUT_FEATURES
        ](layer, 0)
        _enqueue_projection[use_two_output](
            context, input, query_weight, query_bias, query_output
        )
        _enqueue_projection[use_two_output](
            context, input, key_weight, key_bias, key_output
        )
        _enqueue_projection[use_two_output](
            context, input, value_weight, value_bias, value_output
        )
    with query_output_buffer.map_to_host() as mapped_query:
        var host_query = TileTensor(mapped_query, query_output_layout)
        _assert_unit_output(host_query)
    with key_output_buffer.map_to_host() as mapped_key:
        var host_key = TileTensor(mapped_key, kv_output_layout)
        _assert_unit_output(host_key)
    with value_output_buffer.map_to_host() as mapped_value:
        var host_value = TileTensor(mapped_value, kv_output_layout)
        _assert_unit_output(host_value)

    for _ in range(warmup_iterations):
        comptime for layer in range(RING_LAYERS):
            var query_weight = query_weights.tile[
                QUERY_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            var key_weight = key_weights.tile[
                KV_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            var value_weight = value_weights.tile[
                KV_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            _enqueue_projection[use_two_output](
                context, input, query_weight, query_bias, query_output
            )
            _enqueue_projection[use_two_output](
                context, input, key_weight, key_bias, key_output
            )
            _enqueue_projection[use_two_output](
                context, input, value_weight, value_bias, value_output
            )
    context.synchronize()
    comptime if use_two_output:
        print("profile implementation: enqueue_linear_apple_gpu_two_output")
    else:
        print("profile implementation: enqueue_linear_apple_gpu")
    print("device:", context.name())
    print("api:", context.api())
    print("rows:", ROWS)
    print("hidden:", INPUT_FEATURES)
    print("output features:", QKV_OUTPUT_FEATURES)
    print("profile workload: qkv-ring24")
    print("profile dispatches per iteration:", 3 * RING_LAYERS)
    print("warmup iterations:", warmup_iterations)
    print("profile iterations:", iterations)
    print("post-profile idle milliseconds:", post_profile_idle_milliseconds)
    print("PROFILE_REGION_BEGIN")
    for _ in range(iterations):
        comptime for layer in range(RING_LAYERS):
            var query_weight = query_weights.tile[
                QUERY_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            var key_weight = key_weights.tile[
                KV_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            var value_weight = value_weights.tile[
                KV_OUTPUT_FEATURES, INPUT_FEATURES
            ](layer, 0)
            _enqueue_projection[use_two_output](
                context, input, query_weight, query_bias, query_output
            )
            _enqueue_projection[use_two_output](
                context, input, key_weight, key_bias, key_output
            )
            _enqueue_projection[use_two_output](
                context, input, value_weight, value_bias, value_output
            )
    context.synchronize()
    print("PROFILE_REGION_END")
    if post_profile_idle_milliseconds > 0:
        sleep(Float64(post_profile_idle_milliseconds) / 1_000.0)


def main() raises:
    comptime assert has_apple_gpu_accelerator(), "requires an Apple GPU"
    comptime if is_defined["LINEAR_PROFILE_WORKLOAD"]():
        comptime profile_workload = get_defined_int["LINEAR_PROFILE_WORKLOAD"]()
        comptime profile_iterations = get_defined_int[
            "LINEAR_PROFILE_ITERATIONS"
        ]()
        comptime profile_warmup_iterations = get_defined_int[
            "LINEAR_PROFILE_WARMUP_ITERATIONS"
        ]()
        comptime profile_two_output = get_defined_bool[
            "LINEAR_PROFILE_TWO_OUTPUT"
        ]()
        comptime profile_fused_qkv = get_defined_bool[
            "LINEAR_PROFILE_FUSED_QKV"
        ]()
        if profile_two_output and profile_fused_qkv:
            raise Error("linear profile modes are mutually exclusive")
        var post_profile_idle_milliseconds = (
            DEFAULT_PROFILE_POST_IDLE_MILLISECONDS
        )
        comptime if is_defined["LINEAR_PROFILE_POST_IDLE_MILLISECONDS"]():
            post_profile_idle_milliseconds = get_defined_int[
                "LINEAR_PROFILE_POST_IDLE_MILLISECONDS"
            ]()
        comptime if profile_workload == QUERY_WORKLOAD:
            comptime if profile_fused_qkv:
                raise Error("fused QKV profile requires a QKV workload")
            _run_profile_linear[
                QUERY_OUTPUT_FEATURES,
                profile_warmup_iterations,
                profile_iterations,
                profile_two_output,
            ](post_profile_idle_milliseconds)
        elif profile_workload == KV_WORKLOAD:
            comptime if profile_fused_qkv:
                raise Error("fused QKV profile requires a QKV workload")
            _run_profile_linear[
                KV_OUTPUT_FEATURES,
                profile_warmup_iterations,
                profile_iterations,
                profile_two_output,
            ](post_profile_idle_milliseconds)
        elif profile_workload == QKV_HOT_WORKLOAD:
            comptime if profile_fused_qkv:
                _run_profile_qkv_fused[
                    1,
                    profile_warmup_iterations,
                    profile_iterations,
                ](post_profile_idle_milliseconds)
            else:
                _run_profile_qkv_hot[
                    profile_warmup_iterations,
                    profile_iterations,
                    profile_two_output,
                ](post_profile_idle_milliseconds)
        elif profile_workload == QKV_RING24_WORKLOAD:
            comptime if profile_fused_qkv:
                _run_profile_qkv_fused[
                    RING_LAYERS,
                    profile_warmup_iterations,
                    profile_iterations,
                ](post_profile_idle_milliseconds)
            else:
                _run_profile_qkv_ring24[
                    profile_warmup_iterations,
                    profile_iterations,
                    profile_two_output,
                ](post_profile_idle_milliseconds)
        else:
            raise Error("profile workload is not implemented")
    else:
        run_benchmarks()
