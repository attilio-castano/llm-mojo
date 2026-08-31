"""Diagnostic Apple 8x8-MMA linear benchmark against phase-best controls."""

from layout import TensorLayout, TileTensor, row_major
from llm_mojo.linear import (
    enqueue_linear_apple_gpu,
    enqueue_linear_prefill_mma_8x16_apple_gpu,
    enqueue_linear_prefill_register_2x2_apple_gpu,
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
from std.sys import get_defined_bool
from std.sys.info import has_apple_gpu_accelerator


comptime INPUT_FEATURES = 896
comptime OUTPUT_FEATURES = 1_152
comptime RING_LAYERS = 24
comptime BENCHMARK_WARMUP_ITERATIONS = 5
comptime BENCHMARK_MAX_ITERATIONS = 10
comptime BENCHMARK_REPETITIONS = 5


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
                raise Error("linear MMA diagnostic correctness gate failed")


@always_inline
def bench_linear_mma[
    rows: Int,
    use_mma: Bool = False,
    use_register_2x2: Bool = False,
](mut bencher: Bencher) raises capturing:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    comptime assert rows > 0, "benchmark rows must be positive"
    comptime assert not (
        use_mma and use_register_2x2
    ), "benchmark implementation modes are mutually exclusive"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * INPUT_FEATURES
    )
    var weights_buffer = context.enqueue_create_buffer[DType.bfloat16](
        RING_LAYERS * OUTPUT_FEATURES * INPUT_FEATURES
    )
    var bias_buffer = context.enqueue_create_buffer[DType.bfloat16](
        OUTPUT_FEATURES
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * OUTPUT_FEATURES
    )
    input_buffer.enqueue_fill(0.5)
    weights_buffer.enqueue_fill(0.001953125)
    bias_buffer.enqueue_fill(0.125)

    comptime input_layout = row_major[rows, INPUT_FEATURES]()
    comptime weights_layout = row_major[
        RING_LAYERS * OUTPUT_FEATURES, INPUT_FEATURES
    ]()
    comptime bias_layout = row_major[OUTPUT_FEATURES]()
    comptime output_layout = row_major[rows, OUTPUT_FEATURES]()
    var input = TileTensor(input_buffer, input_layout)
    var weights = TileTensor(weights_buffer, weights_layout)
    var bias = TileTensor(bias_buffer, bias_layout)
    var output = TileTensor(output_buffer, output_layout)

    comptime for layer in range(RING_LAYERS):
        var weight = weights.tile[OUTPUT_FEATURES, INPUT_FEATURES](layer, 0)
        comptime if use_mma:
            enqueue_linear_prefill_mma_8x16_apple_gpu(
                context, input, weight, bias, output
            )
        elif use_register_2x2:
            enqueue_linear_prefill_register_2x2_apple_gpu(
                context, input, weight, bias, output
            )
        else:
            enqueue_linear_apple_gpu(context, input, weight, bias, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        comptime for layer in range(RING_LAYERS):
            var weight = weights.tile[OUTPUT_FEATURES, INPUT_FEATURES](layer, 0)
            comptime if use_mma:
                enqueue_linear_prefill_mma_8x16_apple_gpu(
                    launch_context, input, weight, bias, output
                )
            elif use_register_2x2:
                enqueue_linear_prefill_register_2x2_apple_gpu(
                    launch_context, input, weight, bias, output
                )
            else:
                enqueue_linear_apple_gpu(
                    launch_context, input, weight, bias, output
                )

    bencher_iter_custom(bencher, launch, context)


def _add_benchmark[
    rows: Int,
    use_mma: Bool = False,
    use_register_2x2: Bool = False,
](mut benchmark: Bench) raises:
    comptime workload = bench_linear_mma[rows, use_mma, use_register_2x2]
    comptime implementation = (
        "mma_8x16" if use_mma else (
            "register_2x2" if use_register_2x2 else "rowwise"
        )
    )
    benchmark.bench_function[workload](
        BenchId(
            "linear_prefill_" + implementation + "_apple_gpu",
            "m"
            + String(rows)
            + "-k"
            + String(INPUT_FEATURES)
            + "-n"
            + String(OUTPUT_FEATURES)
            + "-layers"
            + String(RING_LAYERS),
        ),
        [
            ThroughputMeasure(
                BenchMetric.elements,
                RING_LAYERS * rows * OUTPUT_FEATURES * INPUT_FEATURES,
            ),
            ThroughputMeasure(
                BenchMetric.bytes,
                2
                * (
                    rows * INPUT_FEATURES
                    + RING_LAYERS * OUTPUT_FEATURES * INPUT_FEATURES
                    + OUTPUT_FEATURES
                    + rows * OUTPUT_FEATURES
                ),
            ),
        ],
    )


def _add_pair[
    rows: Int,
    use_register_control: Bool,
    mma_first: Bool,
](mut benchmark: Bench) raises:
    comptime if mma_first:
        _add_benchmark[rows, True](benchmark)
        _add_benchmark[rows, False, use_register_control](benchmark)
    else:
        _add_benchmark[rows, False, use_register_control](benchmark)
        _add_benchmark[rows, True](benchmark)


def run_benchmarks() raises:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    comptime mma_first = get_defined_bool["LINEAR_MMA_DIAGNOSTIC_MMA_FIRST"]()
    comptime reverse_order = get_defined_bool["LINEAR_MMA_DIAGNOSTIC_REVERSE"]()

    print("implementation: enqueue_linear_prefill_mma_8x16_apple_gpu")
    print("M=1,8 control: enqueue_linear_apple_gpu")
    print("M=16,64 control: enqueue_linear_prefill_register_2x2_apple_gpu")
    print("device:", identity.name())
    print("api:", identity.api())
    print("dtype: bfloat16; accumulation: float32")
    print("input layout: (M, K):(K, 1)")
    print("weight layout: (N, K):(K, 1)")
    print("output layout: (M, N):(N, 1)")
    print("workload: K=896; N=1152; layers=24 rotating weights")
    print("MMA mapping: BM=8; BN=16; BK=8; one 32-lane SIMD group")
    print("MMA: two 8x8 operations per K phase; 112 K phases")
    print("MMA: four FP32 accumulators per lane; scratch=0; barriers=0")
    print("rowwise control: lanes partition K; one output per SIMD group")
    print("register control: one 2x2 output microtile per lane")
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
    print("workload order:", "descending" if reverse_order else "ascending")
    print(
        "implementation order:", "mma,control" if mma_first else "control,mma"
    )

    var benchmark = Bench(
        BenchConfig(
            num_warmup_iters=BENCHMARK_WARMUP_ITERATIONS,
            max_iters=BENCHMARK_MAX_ITERATIONS,
            num_repetitions=BENCHMARK_REPETITIONS,
        )
    )
    comptime if reverse_order:
        _add_pair[64, True, mma_first](benchmark)
        _add_pair[16, True, mma_first](benchmark)
        _add_pair[8, False, mma_first](benchmark)
        _add_pair[1, False, mma_first](benchmark)
    else:
        _add_pair[1, False, mma_first](benchmark)
        _add_pair[8, False, mma_first](benchmark)
        _add_pair[16, True, mma_first](benchmark)
        _add_pair[64, True, mma_first](benchmark)

    benchmark.config.format = Format.tabular
    print("BENCHMARK_RESULTS_BEGIN")
    print(benchmark)
    print("BENCHMARK_RESULTS_END")


def main() raises:
    run_benchmarks()
