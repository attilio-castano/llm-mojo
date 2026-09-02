"""Reproducible materialized Apple GPU GQA baseline benchmark."""

from layout import TensorLayout, TileTensor, row_major
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
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
from std.math import isfinite
from std.sys import get_defined_bool
from std.sys.info import has_apple_gpu_accelerator


comptime QUERY_HEADS = 14
comptime KEY_VALUE_HEADS = 2
comptime HEAD_DIM = 64
comptime BENCHMARK_WARMUP_ITERATIONS = 20
comptime BENCHMARK_MAX_ITERATIONS = 100
comptime BENCHMARK_REPETITIONS = 10
comptime PROBABILITY_SUM_TOLERANCE: Float32 = 0.015625


def _assert_unit_output[
    OutputLayout: TensorLayout,
    //,
    query_rows: Int,
](output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],) raises:
    comptime assert output.flat_rank == 3, "expected rank-3 output"
    for row in range(query_rows):
        for head in range(QUERY_HEADS):
            for dimension in range(HEAD_DIM):
                var actual_bf16 = rebind[Scalar[DType.bfloat16]](
                    output[row, head, dimension]
                )
                var actual = actual_bf16.cast[DType.float32]()
                var error = actual - 1.0
                if error < 0.0:
                    error = -error
                if not isfinite(actual) or error > PROBABILITY_SUM_TOLERANCE:
                    raise Error("attention benchmark output gate failed")


def _assert_causal_probabilities[
    ScratchLayout: TensorLayout,
    //,
    query_rows: Int,
    key_value_rows: Int,
](scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],) raises:
    comptime assert scratch.flat_rank == 3, "expected rank-3 scratch"
    var past = key_value_rows - query_rows
    for row in range(query_rows):
        var visible_key_count = past + row + 1
        for head in range(QUERY_HEADS):
            var total: Scalar[DType.float32] = 0.0
            for key_position in range(key_value_rows):
                var probability_bf16 = rebind[Scalar[DType.bfloat16]](
                    scratch[row, head, key_position]
                )
                var probability = probability_bf16.cast[DType.float32]()
                if (
                    not isfinite(probability)
                    or probability < 0.0
                    or probability > 1.0
                ):
                    raise Error("attention benchmark probability gate failed")
                if key_position >= visible_key_count and probability != 0.0:
                    raise Error("attention benchmark causal gate failed")
                total += probability
            var error = total - 1.0
            if error < 0.0:
                error = -error
            if error > PROBABILITY_SUM_TOLERANCE:
                raise Error("attention benchmark softmax gate failed")


@always_inline
def bench_attention[
    query_rows: Int,
    key_value_rows: Int,
](mut bencher: Bencher) raises capturing:
    comptime assert query_rows > 0, "query rows must be positive"
    comptime assert (
        query_rows <= key_value_rows
    ), "query rows must not exceed key/value rows"
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var context = DeviceContext()
    if context.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    var query_buffer = context.enqueue_create_buffer[DType.bfloat16](
        query_rows * QUERY_HEADS * HEAD_DIM
    )
    var key_buffer = context.enqueue_create_buffer[DType.bfloat16](
        key_value_rows * KEY_VALUE_HEADS * HEAD_DIM
    )
    var value_buffer = context.enqueue_create_buffer[DType.bfloat16](
        key_value_rows * KEY_VALUE_HEADS * HEAD_DIM
    )
    var scratch_buffer = context.enqueue_create_buffer[DType.bfloat16](
        query_rows * QUERY_HEADS * key_value_rows
    )
    var output_buffer = context.enqueue_create_buffer[DType.bfloat16](
        query_rows * QUERY_HEADS * HEAD_DIM
    )
    query_buffer.enqueue_fill(0.25)
    key_buffer.enqueue_fill(0.5)
    value_buffer.enqueue_fill(1.0)

    comptime query_layout = row_major[query_rows, QUERY_HEADS, HEAD_DIM]()
    comptime key_value_layout = row_major[
        key_value_rows, KEY_VALUE_HEADS, HEAD_DIM
    ]()
    comptime scratch_layout = row_major[
        query_rows, QUERY_HEADS, key_value_rows
    ]()
    var query = TileTensor(query_buffer, query_layout)
    var key = TileTensor(key_buffer, key_value_layout)
    var value = TileTensor(value_buffer, key_value_layout)
    var scratch = TileTensor(scratch_buffer, scratch_layout)
    var output = TileTensor(output_buffer, query_layout)

    # Warm the compiled path and retain shape-specific correctness gates outside
    # the timed region. Mapping establishes completion for the initial launch.
    enqueue_grouped_query_attention_apple_gpu(
        context, query, key, value, scratch, output
    )
    with output_buffer.map_to_host() as mapped_output:
        var host_output = TileTensor(mapped_output, query_layout)
        _assert_unit_output[query_rows](host_output)
    with scratch_buffer.map_to_host() as mapped_scratch:
        var host_scratch = TileTensor(mapped_scratch, scratch_layout)
        _assert_causal_probabilities[query_rows, key_value_rows](host_scratch)

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        enqueue_grouped_query_attention_apple_gpu(
            launch_context, query, key, value, scratch, output
        )

    bencher_iter_custom(bencher, launch, context)


def _add_benchmark[
    query_rows: Int,
    key_value_rows: Int,
](mut benchmark: Bench, regime: String) raises:
    comptime workload = bench_attention[query_rows, key_value_rows]
    comptime visible_positions_per_head = (
        query_rows * (2 * key_value_rows - query_rows + 1) // 2
    )
    comptime visible_scores = QUERY_HEADS * visible_positions_per_head
    comptime materialized_scores = query_rows * QUERY_HEADS * key_value_rows
    comptime output_elements = query_rows * QUERY_HEADS * HEAD_DIM
    comptime program_requested_bytes = (
        # QK plus probability-times-V each read two BF16 operands.
        8 * visible_scores * HEAD_DIM
        # Stable softmax reads each visible BF16 score three times.
        + 6 * visible_scores
        # QK and softmax each write every materialized BF16 score slot.
        + 4 * materialized_scores
        # The final stage writes each BF16 output once.
        + 2 * output_elements
    )
    var workload_id = (
        regime
        + "-r"
        + String(query_rows)
        + "-t"
        + String(key_value_rows)
        + "-qh14-kvh2-d64"
    )
    benchmark.bench_function[workload](
        BenchId("grouped_query_attention_apple_gpu_materialized", workload_id),
        [
            ThroughputMeasure(BenchMetric.elements, output_elements),
            ThroughputMeasure(BenchMetric.bytes, program_requested_bytes),
        ],
    )


def _add_workloads[reverse: Bool](mut benchmark: Bench) raises:
    comptime if reverse:
        _add_benchmark[256, 256](benchmark, "full-prefill")
        _add_benchmark[128, 128](benchmark, "full-prefill")
        _add_benchmark[32, 32](benchmark, "full-prefill")
        _add_benchmark[4, 4](benchmark, "full-prefill")
        _add_benchmark[16, 4096](benchmark, "incremental-prefill")
        _add_benchmark[16, 512](benchmark, "incremental-prefill")
        _add_benchmark[4, 128](benchmark, "incremental-prefill")
        _add_benchmark[1, 4096](benchmark, "decode")
        _add_benchmark[1, 1024](benchmark, "decode")
        _add_benchmark[1, 256](benchmark, "decode")
        _add_benchmark[1, 64](benchmark, "decode")
        _add_benchmark[1, 16](benchmark, "decode")
        _add_benchmark[1, 1](benchmark, "decode")
    else:
        _add_benchmark[1, 1](benchmark, "decode")
        _add_benchmark[1, 16](benchmark, "decode")
        _add_benchmark[1, 64](benchmark, "decode")
        _add_benchmark[1, 256](benchmark, "decode")
        _add_benchmark[1, 1024](benchmark, "decode")
        _add_benchmark[1, 4096](benchmark, "decode")
        _add_benchmark[4, 128](benchmark, "incremental-prefill")
        _add_benchmark[16, 512](benchmark, "incremental-prefill")
        _add_benchmark[16, 4096](benchmark, "incremental-prefill")
        _add_benchmark[4, 4](benchmark, "full-prefill")
        _add_benchmark[32, 32](benchmark, "full-prefill")
        _add_benchmark[128, 128](benchmark, "full-prefill")
        _add_benchmark[256, 256](benchmark, "full-prefill")


def main() raises:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    print("implementation: enqueue_grouped_query_attention_apple_gpu")
    print("device:", identity.name())
    print("api:", identity.api())
    print("dtype: BF16 inputs, scores, probabilities, and output")
    print("accumulation: FP32 QK, softmax, and probability-times-V")
    print("shape constants: query_heads=14 key_value_heads=2 head_dim=64")
    print("workload: one hot attention operation; three Metal dispatches")
    print("timing: enqueue through synchronized device completion")
    print("allocation, initialization, correctness, and mapping excluded")
    print("bytes metric: source-derived requests by this materialized baseline")
    print("bytes metric is not measured cache, fabric, or DRAM traffic")
    print(
        "benchmark config:",
        BENCHMARK_WARMUP_ITERATIONS,
        "warmup iterations;",
        BENCHMARK_MAX_ITERATIONS,
        "max iterations;",
        BENCHMARK_REPETITIONS,
        "repetitions",
    )
    comptime reverse = get_defined_bool["ATTENTION_BENCH_REVERSE"]()
    print("workload order:", "descending" if reverse else "ascending")

    var benchmark = Bench(
        BenchConfig(
            num_warmup_iters=BENCHMARK_WARMUP_ITERATIONS,
            max_iters=BENCHMARK_MAX_ITERATIONS,
            num_repetitions=BENCHMARK_REPETITIONS,
        )
    )
    _add_workloads[reverse](benchmark)
    benchmark.config.format = Format.tabular
    print("BENCHMARK_RESULTS_BEGIN")
    print(benchmark)
    print("BENCHMARK_RESULTS_END")
