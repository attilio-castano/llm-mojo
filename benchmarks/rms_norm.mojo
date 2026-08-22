"""Reproducible Apple GPU RMSNorm kernel benchmark."""

from layout import TileTensor, row_major
from llm_mojo.rms_norm import (
    RMS_NORM_APPLE_GPU_BLOCK_SIZE,
    enqueue_rms_norm_apple_gpu,
    enqueue_rms_norm_apple_gpu_shared_tree,
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


comptime HIDDEN_SIZE = 896
comptime BENCHMARK_WARMUP_ITERATIONS = 1_000
comptime BENCHMARK_MAX_ITERATIONS = 1_000
comptime BENCHMARK_REPETITIONS = 10
comptime PROFILE_WARMUP_ITERATIONS = 1_000


@always_inline
def bench_rms_norm[
    rows: Int, use_simdgroup: Bool
](mut bencher: Bencher) raises capturing:
    comptime assert (
        has_apple_gpu_accelerator()
    ), "benchmark requires an Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * HIDDEN_SIZE
    )
    var weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        HIDDEN_SIZE
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * HIDDEN_SIZE
    )
    input_buffer.enqueue_fill(1.0)
    weight_buffer.enqueue_fill(1.0)

    comptime input_layout = row_major[rows, HIDDEN_SIZE]()
    comptime weight_layout = row_major[HIDDEN_SIZE]()
    var input = TileTensor(input_buffer, input_layout)
    var weight = TileTensor(weight_buffer, weight_layout)
    var output = TileTensor(output_buffer, input_layout)

    # Warm the compiled path and retain a correctness gate outside timing.
    comptime if use_simdgroup:
        enqueue_rms_norm_apple_gpu(context, input, weight, output)
    else:
        enqueue_rms_norm_apple_gpu_shared_tree(context, input, weight, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, input_layout)
        comptime assert host_output.flat_rank == 2
        for row in range(rows):
            for hidden in range(HIDDEN_SIZE):
                var actual = rebind[Scalar[DType.bfloat16]](
                    host_output[row, hidden]
                )
                if actual != Scalar[DType.bfloat16](1.0):
                    raise Error("RMSNorm benchmark correctness gate failed")

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        comptime if use_simdgroup:
            enqueue_rms_norm_apple_gpu(launch_context, input, weight, output)
        else:
            enqueue_rms_norm_apple_gpu_shared_tree(
                launch_context, input, weight, output
            )

    bencher_iter_custom(bencher, launch, context)


def add_benchmark[rows: Int, use_simdgroup: Bool](mut benchmark: Bench) raises:
    comptime workload = bench_rms_norm[rows, use_simdgroup]
    comptime if use_simdgroup:
        benchmark.bench_function[workload](
            BenchId(
                "rms_norm_apple_gpu_simdgroup",
                "rows=" + String(rows) + " hidden=" + String(HIDDEN_SIZE),
            ),
            [
                ThroughputMeasure(BenchMetric.elements, rows * HIDDEN_SIZE),
                ThroughputMeasure(BenchMetric.bytes, rows * HIDDEN_SIZE * 6),
            ],
        )
    else:
        benchmark.bench_function[workload](
            BenchId(
                "rms_norm_apple_gpu",
                "rows=" + String(rows) + " hidden=" + String(HIDDEN_SIZE),
            ),
            [
                ThroughputMeasure(BenchMetric.elements, rows * HIDDEN_SIZE),
                ThroughputMeasure(BenchMetric.bytes, rows * HIDDEN_SIZE * 6),
            ],
        )


def add_selected_benchmarks[
    rows: Int, variant_comparison: Bool, variant_first: Bool
](mut benchmark: Bench) raises:
    comptime if variant_comparison:
        comptime if variant_first:
            add_benchmark[rows, True](benchmark)
            add_benchmark[rows, False](benchmark)
        else:
            add_benchmark[rows, False](benchmark)
            add_benchmark[rows, True](benchmark)
    else:
        add_benchmark[rows, True](benchmark)


def run_benchmarks() raises:
    comptime assert (
        has_apple_gpu_accelerator()
    ), "benchmark requires an Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    comptime variant_comparison = get_defined_bool[
        "RMS_NORM_BENCH_VARIANT_COMPARISON"
    ]()
    comptime variant_first = get_defined_bool["RMS_NORM_BENCH_VARIANT_FIRST"]()
    comptime if variant_comparison:
        print("implementation: enqueue_rms_norm_apple_gpu_shared_tree")
        print("comparison implementation: enqueue_rms_norm_apple_gpu")
    else:
        print("implementation: enqueue_rms_norm_apple_gpu")
    print("device:", identity.name())
    print("api:", identity.api())
    print("dtype: bfloat16; accumulation: float32")
    print("layout: (rows, 896):(896, 1); weight: (896):(1)")
    print(
        "work: one row/threadgroup; block_size:",
        RMS_NORM_APPLE_GPU_BLOCK_SIZE,
    )
    print(
        "timing: kernel enqueue and synchronized device execution; transfers"
        " excluded"
    )
    print("bytes metric: logical BF16 input + weight + output traffic per row")
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
    comptime reverse_order = get_defined_bool["RMS_NORM_BENCH_REVERSE"]()
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
        add_selected_benchmarks[4096, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[2048, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[512, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[128, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[16, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[4, variant_comparison, variant_first](benchmark)
        add_selected_benchmarks[1, variant_comparison, variant_first](benchmark)
    else:
        add_selected_benchmarks[1, variant_comparison, variant_first](benchmark)
        add_selected_benchmarks[4, variant_comparison, variant_first](benchmark)
        add_selected_benchmarks[16, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[128, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[512, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[2048, variant_comparison, variant_first](
            benchmark
        )
        add_selected_benchmarks[4096, variant_comparison, variant_first](
            benchmark
        )

    benchmark.config.format = Format.tabular
    print("BENCHMARK_RESULTS_BEGIN")
    print(benchmark)
    print("BENCHMARK_RESULTS_END")


def run_profile_workload[
    rows: Int, iterations: Int, use_simdgroup: Bool
]() raises:
    comptime assert rows > 0, "profile rows must be positive"
    comptime assert iterations > 0, "profile iterations must be positive"
    comptime assert (
        has_apple_gpu_accelerator()
    ), "profile workload requires an Apple GPU"

    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("profile workload requires the Metal device API")

    var input_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * HIDDEN_SIZE
    )
    var weight_buffer = context.enqueue_create_buffer[DType.bfloat16](
        HIDDEN_SIZE
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        rows * HIDDEN_SIZE
    )
    input_buffer.enqueue_fill(1.0)
    weight_buffer.enqueue_fill(1.0)

    comptime input_layout = row_major[rows, HIDDEN_SIZE]()
    comptime weight_layout = row_major[HIDDEN_SIZE]()
    var input = TileTensor(input_buffer, input_layout)
    var weight = TileTensor(weight_buffer, weight_layout)
    var output = TileTensor(output_buffer, input_layout)

    comptime if use_simdgroup:
        enqueue_rms_norm_apple_gpu(context, input, weight, output)
    else:
        enqueue_rms_norm_apple_gpu_shared_tree(context, input, weight, output)
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, input_layout)
        comptime assert host_output.flat_rank == 2
        for row in range(rows):
            for hidden in range(HIDDEN_SIZE):
                var actual = rebind[Scalar[DType.bfloat16]](
                    host_output[row, hidden]
                )
                if actual != Scalar[DType.bfloat16](1.0):
                    raise Error("RMSNorm profile correctness gate failed")

    comptime if use_simdgroup:
        for _ in range(PROFILE_WARMUP_ITERATIONS):
            enqueue_rms_norm_apple_gpu(context, input, weight, output)
    else:
        for _ in range(PROFILE_WARMUP_ITERATIONS):
            enqueue_rms_norm_apple_gpu_shared_tree(
                context, input, weight, output
            )
    context.synchronize()

    comptime if use_simdgroup:
        print("profile implementation: enqueue_rms_norm_apple_gpu")
    else:
        print("profile implementation: enqueue_rms_norm_apple_gpu_shared_tree")
    print("device:", context.name())
    print("api:", context.api())
    print("rows:", rows)
    print("hidden:", HIDDEN_SIZE)
    print("warmup iterations:", PROFILE_WARMUP_ITERATIONS)
    print("profile iterations:", iterations)
    print("PROFILE_REGION_BEGIN")
    comptime if use_simdgroup:
        for _ in range(iterations):
            enqueue_rms_norm_apple_gpu(context, input, weight, output)
    else:
        for _ in range(iterations):
            enqueue_rms_norm_apple_gpu_shared_tree(
                context, input, weight, output
            )
    context.synchronize()
    print("PROFILE_REGION_END")


def main() raises:
    comptime if is_defined["RMS_NORM_PROFILE_ROWS"]():
        comptime profile_rows = get_defined_int["RMS_NORM_PROFILE_ROWS"]()
        comptime profile_iterations = get_defined_int[
            "RMS_NORM_PROFILE_ITERATIONS"
        ]()
        comptime profile_simdgroup = get_defined_bool[
            "RMS_NORM_PROFILE_SIMDGROUP"
        ]()
        run_profile_workload[
            profile_rows, profile_iterations, profile_simdgroup
        ]()
    else:
        run_benchmarks()
