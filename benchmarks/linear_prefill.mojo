"""Reproducible Apple GPU prefill linear-projection benchmark."""

from layout import TensorLayout, TileTensor, row_major
from llm_mojo.linear import enqueue_linear_apple_gpu
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
from std.sys import get_defined_bool
from std.sys.info import has_apple_gpu_accelerator


comptime INPUT_FEATURES = 896
comptime KV_OUTPUT_FEATURES = 128
comptime QUERY_OUTPUT_FEATURES = 896
comptime QKV_OUTPUT_FEATURES = 1_152
comptime RING_LAYERS = 24
comptime BENCHMARK_WARMUP_ITERATIONS = 10
comptime BENCHMARK_MAX_ITERATIONS = 20
comptime BENCHMARK_REPETITIONS = 10


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
                raise Error("prefill linear benchmark correctness gate failed")


@always_inline
def bench_prefill_linear[
    rows: Int, output_features: Int, layers: Int
](mut bencher: Bencher) raises capturing:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    comptime assert rows > 0, "benchmark rows must be positive"
    comptime assert layers > 0, "benchmark layers must be positive"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * INPUT_FEATURES
    )
    var weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        layers * output_features * INPUT_FEATURES
    )
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        output_features
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * output_features
    )
    input_buffer.enqueue_fill(0.5)
    weights_buffer.enqueue_fill(0.001953125)
    bias_buffer.enqueue_fill(0.125)

    comptime input_layout = row_major[rows, INPUT_FEATURES]()
    comptime weights_layout = row_major[
        layers * output_features, INPUT_FEATURES
    ]()
    comptime bias_layout = row_major[output_features]()
    comptime output_layout = row_major[rows, output_features]()
    var input = TileTensor(input_buffer, input_layout)
    var weights = TileTensor(weights_buffer, weights_layout)
    var bias = TileTensor(bias_buffer, bias_layout)
    var output = TileTensor(output_buffer, output_layout)

    comptime for layer in range(layers):
        var weight = weights.tile[output_features, INPUT_FEATURES](layer, 0)
        enqueue_linear_apple_gpu(context, input, weight, bias, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        comptime for layer in range(layers):
            var weight = weights.tile[output_features, INPUT_FEATURES](layer, 0)
            enqueue_linear_apple_gpu(
                launch_context, input, weight, bias, output
            )

    bencher_iter_custom(bencher, launch, context)


def _add_benchmark[
    rows: Int, output_features: Int, layers: Int
](mut benchmark: Bench) raises:
    comptime workload = bench_prefill_linear[rows, output_features, layers]
    benchmark.bench_function[workload](
        BenchId(
            "linear_prefill_rowwise_apple_gpu",
            "m"
            + String(rows)
            + "-k"
            + String(INPUT_FEATURES)
            + "-n"
            + String(output_features)
            + "-layers"
            + String(layers),
        ),
        [
            ThroughputMeasure(
                BenchMetric.elements,
                layers * rows * output_features * INPUT_FEATURES,
            ),
            ThroughputMeasure(
                BenchMetric.bytes,
                2
                * (
                    rows * INPUT_FEATURES
                    + layers * output_features * INPUT_FEATURES
                    + output_features
                    + rows * output_features
                ),
            ),
        ],
    )


def _add_row_workloads[rows: Int, reverse: Bool](mut benchmark: Bench) raises:
    comptime if reverse:
        _add_benchmark[rows, QKV_OUTPUT_FEATURES, RING_LAYERS](benchmark)
        _add_benchmark[rows, QKV_OUTPUT_FEATURES, 1](benchmark)
        _add_benchmark[rows, QUERY_OUTPUT_FEATURES, 1](benchmark)
        _add_benchmark[rows, KV_OUTPUT_FEATURES, 1](benchmark)
    else:
        _add_benchmark[rows, KV_OUTPUT_FEATURES, 1](benchmark)
        _add_benchmark[rows, QUERY_OUTPUT_FEATURES, 1](benchmark)
        _add_benchmark[rows, QKV_OUTPUT_FEATURES, 1](benchmark)
        _add_benchmark[rows, QKV_OUTPUT_FEATURES, RING_LAYERS](benchmark)


def run_benchmarks() raises:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    print("implementation: enqueue_linear_apple_gpu")
    print("device:", identity.name())
    print("api:", identity.api())
    print("dtype: bfloat16; accumulation: float32")
    print("input layout: (M, K):(K, 1)")
    print("weight layout: (N, K):(K, 1)")
    print("output layout: (M, N):(N, 1)")
    print("K: 896; N: 128, 896, 1152; layers: 1 or 24")
    print("mapping: one output dot product per SIMD group")
    print(
        "timing: kernel enqueue and synchronized device execution; transfers"
        " excluded"
    )
    print("bytes metric: allocated BF16 tensor footprint")
    print("bytes metric is not a measured DRAM-bandwidth claim")
    print(
        "benchmark config:",
        BENCHMARK_WARMUP_ITERATIONS,
        "warmup iterations;",
        BENCHMARK_MAX_ITERATIONS,
        "max iterations;",
        BENCHMARK_REPETITIONS,
        "repetitions",
    )
    comptime reverse_order = get_defined_bool[
        "LINEAR_PREFILL_BENCH_REVERSE"
    ]()
    print("workload order:", "descending" if reverse_order else "ascending")

    var benchmark = Bench(
        BenchConfig(
            num_warmup_iters=BENCHMARK_WARMUP_ITERATIONS,
            max_iters=BENCHMARK_MAX_ITERATIONS,
            num_repetitions=BENCHMARK_REPETITIONS,
        )
    )
    comptime if reverse_order:
        _add_row_workloads[256, True](benchmark)
        _add_row_workloads[128, True](benchmark)
        _add_row_workloads[64, True](benchmark)
        _add_row_workloads[32, True](benchmark)
        _add_row_workloads[16, True](benchmark)
        _add_row_workloads[8, True](benchmark)
        _add_row_workloads[4, True](benchmark)
        _add_row_workloads[1, True](benchmark)
    else:
        _add_row_workloads[1, False](benchmark)
        _add_row_workloads[4, False](benchmark)
        _add_row_workloads[8, False](benchmark)
        _add_row_workloads[16, False](benchmark)
        _add_row_workloads[32, False](benchmark)
        _add_row_workloads[64, False](benchmark)
        _add_row_workloads[128, False](benchmark)
        _add_row_workloads[256, False](benchmark)

    benchmark.config.format = Format.tabular
    print("BENCHMARK_RESULTS_BEGIN")
    print(benchmark)
    print("BENCHMARK_RESULTS_END")


def main() raises:
    run_benchmarks()
