"""Reproducible materialized Apple GPU GQA baseline benchmark."""

from layout import TensorLayout, TileTensor, row_major
from llm_mojo.attention import (
    ATTENTION_APPLE_GPU_BLOCK_SIZE,
    _grouped_query_attention_pv_apple_gpu_kernel,
    _grouped_query_attention_qk_apple_gpu_kernel,
    _grouped_query_attention_softmax_apple_gpu_kernel,
    enqueue_grouped_query_attention_apple_gpu,
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
from std.math import ceildiv, isfinite
from std.sys import get_defined_bool
from std.sys.info import has_apple_gpu_accelerator


comptime QUERY_HEADS = 14
comptime KEY_VALUE_HEADS = 2
comptime HEAD_DIM = 64
comptime BENCHMARK_WARMUP_ITERATIONS = 20
comptime BENCHMARK_MAX_ITERATIONS = 100
comptime BENCHMARK_REPETITIONS = 10
comptime PROBABILITY_SUM_TOLERANCE: Float32 = 0.015625
comptime STAGE_END_TO_END = 0
comptime STAGE_QK = 1
comptime STAGE_SOFTMAX = 2
comptime STAGE_PV = 3


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


def _assert_unit_qk_scores[
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
            for key_position in range(key_value_rows):
                var score_bf16 = rebind[Scalar[DType.bfloat16]](
                    scratch[row, head, key_position]
                )
                var score = score_bf16.cast[DType.float32]()
                var expected: Scalar[DType.float32] = 0.0
                if key_position < visible_key_count:
                    expected = 1.0
                var error = score - expected
                if error < 0.0:
                    error = -error
                if not isfinite(score) or error > PROBABILITY_SUM_TOLERANCE:
                    raise Error("attention benchmark QK score gate failed")


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


def _enqueue_qk_stage[
    QueryLayout: TensorLayout,
    KeyLayout: TensorLayout,
    ScratchLayout: TensorLayout,
](
    context: DeviceContext,
    query: TileTensor[DType.bfloat16, QueryLayout, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KeyLayout, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
) raises:
    comptime assert query.flat_rank == 3, "query must have rank 3"
    comptime assert key.flat_rank == 3, "key must have rank 3"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"
    var query_rows = Int(query.dim[0]())
    var query_heads = Int(query.dim[1]())
    var head_dim = Int(query.dim[2]())
    var key_value_rows = Int(key.dim[0]())
    var key_value_heads = Int(key.dim[1]())
    var score_count = query_rows * query_heads * key_value_rows
    comptime kernel = _grouped_query_attention_qk_apple_gpu_kernel[
        QueryLayout, KeyLayout, ScratchLayout
    ]
    context.enqueue_function[kernel](
        query,
        key,
        scratch,
        Int32(query_rows),
        Int32(key_value_rows),
        Int32(query_heads),
        Int32(key_value_heads),
        Int32(head_dim),
        grid_dim=ceildiv(score_count, ATTENTION_APPLE_GPU_BLOCK_SIZE),
        block_dim=ATTENTION_APPLE_GPU_BLOCK_SIZE,
    )


def _enqueue_softmax_stage[
    ScratchLayout: TensorLayout,
](
    context: DeviceContext,
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
) raises:
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"
    var query_rows = Int(scratch.dim[0]())
    var query_heads = Int(scratch.dim[1]())
    var key_value_rows = Int(scratch.dim[2]())
    var row_head_count = query_rows * query_heads
    comptime kernel = _grouped_query_attention_softmax_apple_gpu_kernel[
        ScratchLayout
    ]
    context.enqueue_function[kernel](
        scratch,
        Int32(query_rows),
        Int32(key_value_rows),
        Int32(query_heads),
        grid_dim=ceildiv(row_head_count, ATTENTION_APPLE_GPU_BLOCK_SIZE),
        block_dim=ATTENTION_APPLE_GPU_BLOCK_SIZE,
    )


def _enqueue_pv_stage[
    ValueLayout: TensorLayout,
    ScratchLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    context: DeviceContext,
    value: TileTensor[DType.bfloat16, ValueLayout, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    comptime assert value.flat_rank == 3, "value must have rank 3"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"
    comptime assert output.flat_rank == 3, "output must have rank 3"
    var query_rows = Int(output.dim[0]())
    var query_heads = Int(output.dim[1]())
    var head_dim = Int(output.dim[2]())
    var key_value_rows = Int(value.dim[0]())
    var key_value_heads = Int(value.dim[1]())
    var output_count = query_rows * query_heads * head_dim
    comptime kernel = _grouped_query_attention_pv_apple_gpu_kernel[
        ValueLayout, ScratchLayout, OutputLayout
    ]
    context.enqueue_function[kernel](
        value,
        scratch,
        output,
        Int32(query_rows),
        Int32(key_value_rows),
        Int32(query_heads),
        Int32(key_value_heads),
        Int32(head_dim),
        grid_dim=ceildiv(output_count, ATTENTION_APPLE_GPU_BLOCK_SIZE),
        block_dim=ATTENTION_APPLE_GPU_BLOCK_SIZE,
    )


@always_inline
def bench_attention_stage[
    query_rows: Int,
    key_value_rows: Int,
    stage: Int,
](mut bencher: Bencher) raises capturing:
    comptime assert query_rows > 0, "query rows must be positive"
    comptime assert (
        query_rows <= key_value_rows
    ), "query rows must not exceed key/value rows"
    comptime assert (
        stage >= STAGE_END_TO_END and stage <= STAGE_PV
    ), "unknown attention benchmark stage"
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

    # Prepare and validate each exact production kernel outside timing. Uniform
    # visible inputs make softmax a fixed point, so repeated in-place softmax
    # iterations retain valid probabilities without a timed reset operation.
    comptime if stage == STAGE_END_TO_END:
        enqueue_grouped_query_attention_apple_gpu(
            context, query, key, value, scratch, output
        )
        with output_buffer.map_to_host() as mapped_output:
            var host_output = TileTensor(mapped_output, query_layout)
            _assert_unit_output[query_rows](host_output)
        with scratch_buffer.map_to_host() as mapped_scratch:
            var host_scratch = TileTensor(mapped_scratch, scratch_layout)
            _assert_causal_probabilities[query_rows, key_value_rows](
                host_scratch
            )
    elif stage == STAGE_QK:
        _enqueue_qk_stage(context, query, key, scratch)
        with scratch_buffer.map_to_host() as mapped_scratch:
            var host_scratch = TileTensor(mapped_scratch, scratch_layout)
            _assert_unit_qk_scores[query_rows, key_value_rows](host_scratch)
    elif stage == STAGE_SOFTMAX:
        _enqueue_qk_stage(context, query, key, scratch)
        _enqueue_softmax_stage(context, scratch)
        with scratch_buffer.map_to_host() as mapped_scratch:
            var host_scratch = TileTensor(mapped_scratch, scratch_layout)
            _assert_causal_probabilities[query_rows, key_value_rows](
                host_scratch
            )
    else:
        _enqueue_qk_stage(context, query, key, scratch)
        _enqueue_softmax_stage(context, scratch)
        _enqueue_pv_stage(context, value, scratch, output)
        with output_buffer.map_to_host() as mapped_output:
            var host_output = TileTensor(mapped_output, query_layout)
            _assert_unit_output[query_rows](host_output)
        with scratch_buffer.map_to_host() as mapped_scratch:
            var host_scratch = TileTensor(mapped_scratch, scratch_layout)
            _assert_causal_probabilities[query_rows, key_value_rows](
                host_scratch
            )

    @always_inline
    def launch(launch_context: DeviceContext) raises {imm}:
        comptime if stage == STAGE_END_TO_END:
            enqueue_grouped_query_attention_apple_gpu(
                launch_context, query, key, value, scratch, output
            )
        elif stage == STAGE_QK:
            _enqueue_qk_stage(launch_context, query, key, scratch)
        elif stage == STAGE_SOFTMAX:
            _enqueue_softmax_stage(launch_context, scratch)
        else:
            _enqueue_pv_stage(launch_context, value, scratch, output)

    bencher_iter_custom(bencher, launch, context)


def _add_stage_benchmark[
    query_rows: Int,
    key_value_rows: Int,
    stage: Int,
](mut benchmark: Bench, regime: String, benchmark_name: String) raises:
    comptime workload = bench_attention_stage[query_rows, key_value_rows, stage]
    comptime visible_positions_per_head = (
        query_rows * (2 * key_value_rows - query_rows + 1) // 2
    )
    comptime visible_scores = QUERY_HEADS * visible_positions_per_head
    comptime materialized_scores = query_rows * QUERY_HEADS * key_value_rows
    comptime output_elements = query_rows * QUERY_HEADS * HEAD_DIM
    comptime qk_requested_bytes = (
        4 * visible_scores * HEAD_DIM + 2 * materialized_scores
    )
    comptime softmax_requested_bytes = (
        6 * visible_scores + 2 * materialized_scores
    )
    comptime pv_requested_bytes = (
        4 * visible_scores * HEAD_DIM + 2 * output_elements
    )
    var stage_elements: Int
    var program_requested_bytes: Int
    comptime if stage == STAGE_END_TO_END:
        stage_elements = output_elements
        program_requested_bytes = (
            qk_requested_bytes + softmax_requested_bytes + pv_requested_bytes
        )
    elif stage == STAGE_QK:
        stage_elements = materialized_scores
        program_requested_bytes = qk_requested_bytes
    elif stage == STAGE_SOFTMAX:
        stage_elements = materialized_scores
        program_requested_bytes = softmax_requested_bytes
    else:
        stage_elements = output_elements
        program_requested_bytes = pv_requested_bytes
    var workload_id = (
        regime
        + "-r"
        + String(query_rows)
        + "-t"
        + String(key_value_rows)
        + "-qh14-kvh2-d64"
    )
    benchmark.bench_function[workload](
        BenchId(benchmark_name, workload_id),
        [
            ThroughputMeasure(BenchMetric.elements, stage_elements),
            ThroughputMeasure(BenchMetric.bytes, program_requested_bytes),
        ],
    )


def _add_stage_group[
    query_rows: Int,
    key_value_rows: Int,
    reverse_stages: Bool,
](mut benchmark: Bench, regime: String) raises:
    comptime if reverse_stages:
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_PV](
            benchmark,
            regime,
            "grouped_query_attention_pv_apple_gpu_materialized",
        )
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_SOFTMAX](
            benchmark,
            regime,
            "grouped_query_attention_softmax_apple_gpu_materialized",
        )
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_QK](
            benchmark,
            regime,
            "grouped_query_attention_qk_apple_gpu_materialized",
        )
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_END_TO_END](
            benchmark, regime, "grouped_query_attention_apple_gpu_materialized"
        )
    else:
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_END_TO_END](
            benchmark, regime, "grouped_query_attention_apple_gpu_materialized"
        )
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_QK](
            benchmark,
            regime,
            "grouped_query_attention_qk_apple_gpu_materialized",
        )
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_SOFTMAX](
            benchmark,
            regime,
            "grouped_query_attention_softmax_apple_gpu_materialized",
        )
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_PV](
            benchmark,
            regime,
            "grouped_query_attention_pv_apple_gpu_materialized",
        )


def _add_shape[
    query_rows: Int,
    key_value_rows: Int,
    stage_attribution: Bool,
    reverse_stages: Bool,
](mut benchmark: Bench, regime: String) raises:
    comptime if stage_attribution:
        _add_stage_group[query_rows, key_value_rows, reverse_stages](
            benchmark, regime
        )
    else:
        _add_stage_benchmark[query_rows, key_value_rows, STAGE_END_TO_END](
            benchmark, regime, "grouped_query_attention_apple_gpu_materialized"
        )


def _add_workloads[
    reverse: Bool,
    stage_attribution: Bool,
    reverse_stages: Bool,
](mut benchmark: Bench) raises:
    comptime if reverse:
        _add_shape[256, 256, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )
        _add_shape[128, 128, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )
        _add_shape[32, 32, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )
        _add_shape[4, 4, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )
        _add_shape[16, 4096, stage_attribution, reverse_stages](
            benchmark, "incremental-prefill"
        )
        _add_shape[16, 512, stage_attribution, reverse_stages](
            benchmark, "incremental-prefill"
        )
        _add_shape[4, 128, stage_attribution, reverse_stages](
            benchmark, "incremental-prefill"
        )
        _add_shape[1, 4096, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 1024, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 256, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 64, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 16, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 1, stage_attribution, reverse_stages](benchmark, "decode")
    else:
        _add_shape[1, 1, stage_attribution, reverse_stages](benchmark, "decode")
        _add_shape[1, 16, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 64, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 256, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 1024, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[1, 4096, stage_attribution, reverse_stages](
            benchmark, "decode"
        )
        _add_shape[4, 128, stage_attribution, reverse_stages](
            benchmark, "incremental-prefill"
        )
        _add_shape[16, 512, stage_attribution, reverse_stages](
            benchmark, "incremental-prefill"
        )
        _add_shape[16, 4096, stage_attribution, reverse_stages](
            benchmark, "incremental-prefill"
        )
        _add_shape[4, 4, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )
        _add_shape[32, 32, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )
        _add_shape[128, 128, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )
        _add_shape[256, 256, stage_attribution, reverse_stages](
            benchmark, "full-prefill"
        )


def main() raises:
    comptime assert has_apple_gpu_accelerator(), "benchmark requires Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

    print("implementation: enqueue_grouped_query_attention_apple_gpu")
    print("device:", identity.name())
    print("api:", identity.api())
    comptime stage_attribution = get_defined_bool[
        "ATTENTION_BENCH_STAGE_ATTRIBUTION"
    ]()
    comptime reverse_stages = get_defined_bool[
        "ATTENTION_BENCH_REVERSE_STAGES"
    ]()
    comptime assert (
        stage_attribution or not reverse_stages
    ), "reverse stage order requires stage-attribution mode"
    comptime if stage_attribution:
        print("mode: stage-attribution")
        print("workload: one end-to-end call plus three isolated stages")
        print("isolated stage timing: one Metal dispatch through completion")
        print("softmax input: uniform causal probabilities at a fixed point")
        print("stage order:", "reverse" if reverse_stages else "forward")
    else:
        print("mode: end-to-end")
        print("workload: one hot attention operation; three Metal dispatches")
    print("dtype: BF16 inputs, scores, probabilities, and output")
    print("accumulation: FP32 QK, softmax, and probability-times-V")
    print("shape constants: query_heads=14 key_value_heads=2 head_dim=64")
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
    _add_workloads[reverse, stage_attribution, reverse_stages](benchmark)
    benchmark.config.format = Format.tabular
    print("BENCHMARK_RESULTS_BEGIN")
    print(benchmark)
    print("BENCHMARK_RESULTS_END")
