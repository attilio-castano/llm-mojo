"""Apple GPU prefill BK sensitivity benchmark."""

from layout import TensorLayout, TileTensor, row_major
from llm_mojo.linear import (
    enqueue_linear_prefill_direct_apple_gpu,
    enqueue_linear_prefill_register_2x2_apple_gpu,
    enqueue_linear_prefill_tiled_apple_gpu_bk,
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
                raise Error("prefill BK benchmark correctness gate failed")


@always_inline
def bench_prefill_bk[
    rows: Int,
    tile_input_features: Int,
    use_direct: Bool = False,
    use_register_2x2: Bool = False,
](mut bencher: Bencher) raises capturing:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    comptime assert rows > 0, "benchmark rows must be positive"
    comptime assert not (
        use_direct and use_register_2x2
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
        comptime if use_register_2x2:
            enqueue_linear_prefill_register_2x2_apple_gpu(
                context, input, weight, bias, output
            )
        elif use_direct:
            enqueue_linear_prefill_direct_apple_gpu(
                context, input, weight, bias, output
            )
        else:
            enqueue_linear_prefill_tiled_apple_gpu_bk[tile_input_features](
                context, input, weight, bias, output
            )
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, output_layout)
        _assert_unit_output(host_output)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        comptime for layer in range(RING_LAYERS):
            var weight = weights.tile[OUTPUT_FEATURES, INPUT_FEATURES](layer, 0)
            comptime if use_register_2x2:
                enqueue_linear_prefill_register_2x2_apple_gpu(
                    launch_context, input, weight, bias, output
                )
            elif use_direct:
                enqueue_linear_prefill_direct_apple_gpu(
                    launch_context, input, weight, bias, output
                )
            else:
                enqueue_linear_prefill_tiled_apple_gpu_bk[tile_input_features](
                    launch_context, input, weight, bias, output
                )

    bencher_iter_custom(bencher, launch, context)


def _add_benchmark[
    rows: Int,
    tile_input_features: Int,
    use_direct: Bool = False,
    use_register_2x2: Bool = False,
](mut benchmark: Bench) raises:
    comptime workload = bench_prefill_bk[
        rows, tile_input_features, use_direct, use_register_2x2
    ]
    comptime benchmark_name = (
        "linear_prefill_register_2x2_apple_gpu" if use_register_2x2 else (
            "linear_prefill_direct_apple_gpu" if use_direct else (
                "linear_prefill_tiled_bk"
                + String(tile_input_features)
                + "_apple_gpu"
            )
        )
    )
    benchmark.bench_function[workload](
        BenchId(
            benchmark_name,
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


def _add_bk_sequence[
    rows: Int,
    sequence: Int,
](mut benchmark: Bench) raises:
    comptime if sequence == 1:
        _add_benchmark[rows, 16](benchmark)
        _add_benchmark[rows, 32](benchmark)
        _add_benchmark[rows, 128](benchmark)
        _add_benchmark[rows, 64](benchmark)
    elif sequence == 2:
        _add_benchmark[rows, 32](benchmark)
        _add_benchmark[rows, 64](benchmark)
        _add_benchmark[rows, 16](benchmark)
        _add_benchmark[rows, 128](benchmark)
    elif sequence == 3:
        _add_benchmark[rows, 64](benchmark)
        _add_benchmark[rows, 128](benchmark)
        _add_benchmark[rows, 32](benchmark)
        _add_benchmark[rows, 16](benchmark)
    else:
        comptime assert sequence == 4, "BK sequence must be in 1...4"
        _add_benchmark[rows, 128](benchmark)
        _add_benchmark[rows, 16](benchmark)
        _add_benchmark[rows, 64](benchmark)
        _add_benchmark[rows, 32](benchmark)


def _add_direct_comparison[
    rows: Int,
    candidate_first: Bool,
](mut benchmark: Bench) raises:
    comptime if candidate_first:
        _add_benchmark[rows, 16](benchmark)
        _add_benchmark[rows, 16, True](benchmark)
    else:
        _add_benchmark[rows, 16, True](benchmark)
        _add_benchmark[rows, 16](benchmark)


def _add_register_comparison[
    rows: Int,
    candidate_first: Bool,
](mut benchmark: Bench) raises:
    comptime if candidate_first:
        _add_benchmark[rows, 16, False, True](benchmark)
        _add_benchmark[rows, 16, True](benchmark)
    else:
        _add_benchmark[rows, 16, True](benchmark)
        _add_benchmark[rows, 16, False, True](benchmark)


def _add_selected_workload[
    rows: Int,
    sequence: Int,
    direct_comparison: Bool,
    register_comparison: Bool,
    candidate_first: Bool,
](mut benchmark: Bench) raises:
    comptime if register_comparison:
        _add_register_comparison[rows, candidate_first](benchmark)
    elif direct_comparison:
        _add_direct_comparison[rows, candidate_first](benchmark)
    else:
        _add_bk_sequence[rows, sequence](benchmark)


def run_benchmarks() raises:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    comptime sequence_2 = get_defined_bool[
        "LINEAR_PREFILL_BK_SWEEP_SEQUENCE_2"
    ]()
    comptime sequence_3 = get_defined_bool[
        "LINEAR_PREFILL_BK_SWEEP_SEQUENCE_3"
    ]()
    comptime sequence_4 = get_defined_bool[
        "LINEAR_PREFILL_BK_SWEEP_SEQUENCE_4"
    ]()
    comptime direct_comparison = get_defined_bool[
        "LINEAR_PREFILL_BK16_DIRECT_COMPARISON"
    ]()
    comptime candidate_first = get_defined_bool["LINEAR_PREFILL_BK16_FIRST"]()
    comptime register_comparison = get_defined_bool[
        "LINEAR_PREFILL_REGISTER_DIRECT_COMPARISON"
    ]()
    comptime register_first = get_defined_bool[
        "LINEAR_PREFILL_REGISTER_FIRST"
    ]()
    comptime assert not (sequence_2 and sequence_3)
    comptime assert not (sequence_2 and sequence_4)
    comptime assert not (sequence_3 and sequence_4)
    comptime assert not candidate_first or direct_comparison
    comptime assert not register_first or register_comparison
    comptime assert not (direct_comparison and register_comparison)
    comptime assert not direct_comparison or not (
        sequence_2 or sequence_3 or sequence_4
    )
    comptime assert not register_comparison or not (
        sequence_2 or sequence_3 or sequence_4
    )
    comptime selected_candidate_first = (
        register_first if register_comparison else candidate_first
    )
    comptime sequence = 4 if sequence_4 else (
        3 if sequence_3 else (2 if sequence_2 else 1)
    )
    comptime reverse_order = get_defined_bool[
        "LINEAR_PREFILL_BK_SWEEP_REVERSE"
    ]()

    comptime if register_comparison:
        print("implementation: enqueue_linear_prefill_direct_apple_gpu")
        print(
            "comparison implementation: "
            "enqueue_linear_prefill_register_2x2_apple_gpu"
        )
    elif direct_comparison:
        print("implementation: enqueue_linear_prefill_direct_apple_gpu")
        print(
            "comparison implementation: "
            "enqueue_linear_prefill_tiled_apple_gpu_bk"
        )
        print("comparison BK: 16")
    else:
        print("implementation: enqueue_linear_prefill_tiled_apple_gpu_bk")
    print("device:", identity.name())
    print("api:", identity.api())
    print("dtype: bfloat16; accumulation: float32")
    print("input layout: (M, K):(K, 1)")
    print("weight layout: (N, K):(K, 1)")
    print("output layout: (M, N):(N, 1)")
    comptime if register_comparison:
        print(
            "control mapping: BM=8; BN=16; 128 threads; one output per thread"
        )
        print(
            "candidate mapping: BM=8; BN=16; 32 threads; one 2x2 output "
            "microtile per thread"
        )
    else:
        print("mapping: BM=8; BN=16; 128 threads; one thread per output")
    print("workload: K=896; N=1152; layers=24 rotating weights")
    comptime if register_comparison:
        print("direct: four SIMD groups; one FP32 accumulator per thread")
        print("register 2x2: one SIMD group; four FP32 accumulators per thread")
        print("both: full-K streaming; scratch=0 bytes; barriers=0")
    elif direct_comparison:
        print("direct: scratch=0 bytes; shared K phases=0; barriers=0")
        print("BK=16: scratch=768 bytes; K phases=56; barriers=112")
    else:
        print("BK=16: scratch=768 bytes; K phases=56; barriers=112")
        print("BK=32: scratch=1536 bytes; K phases=28; barriers=56")
        print("BK=64: scratch=3072 bytes; K phases=14; barriers=28")
        print("BK=128: scratch=6144 bytes; K phases=7; barriers=14")
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
    comptime if register_comparison:
        print(
            "implementation order:",
            "register_2x2,direct" if register_first else "direct,register_2x2",
        )
    elif direct_comparison:
        print(
            "implementation order:",
            "bk16,direct" if candidate_first else "direct,bk16",
        )
    else:
        comptime if sequence == 1:
            print("BK order: 16,32,128,64")
        elif sequence == 2:
            print("BK order: 32,64,16,128")
        elif sequence == 3:
            print("BK order: 64,128,32,16")
        else:
            print("BK order: 128,16,64,32")

    var benchmark = Bench(
        BenchConfig(
            num_warmup_iters=BENCHMARK_WARMUP_ITERATIONS,
            max_iters=BENCHMARK_MAX_ITERATIONS,
            num_repetitions=BENCHMARK_REPETITIONS,
        )
    )
    comptime if reverse_order:
        _add_selected_workload[
            256,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            128,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            64,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            32,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            16,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            8,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            4,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            1,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
    else:
        _add_selected_workload[
            1,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            4,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            8,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            16,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            32,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            64,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            128,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)
        _add_selected_workload[
            256,
            sequence,
            direct_comparison,
            register_comparison,
            selected_candidate_first,
        ](benchmark)

    benchmark.config.format = Format.tabular
    print("BENCHMARK_RESULTS_BEGIN")
    print(benchmark)
    print("BENCHMARK_RESULTS_END")


def main() raises:
    run_benchmarks()
