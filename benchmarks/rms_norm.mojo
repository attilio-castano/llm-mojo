"""Reproducible Apple GPU RMSNorm kernel benchmark."""

from layout import TileTensor, row_major
from llm_mojo.rms_norm import (
    RMS_NORM_APPLE_GPU_BLOCK_SIZE,
    enqueue_rms_norm_apple_gpu,
)
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.sys.info import has_apple_gpu_accelerator


comptime HIDDEN_SIZE = 896


@always_inline
def bench_rms_norm[rows: Int](mut bencher: Bencher) raises capturing:
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
    enqueue_rms_norm_apple_gpu(context, input, weight, output)
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
        enqueue_rms_norm_apple_gpu(launch_context, input, weight, output)

    bencher_iter_custom(bencher, launch, context)


def main() raises:
    comptime assert (
        has_apple_gpu_accelerator()
    ), "benchmark requires an Apple GPU"
    var identity = DeviceContext()
    if identity.api() != "metal":
        raise Error("benchmark requires the Metal device API")

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
        "benchmark config: 10 warmup iterations; 1000 max iterations; 5"
        " repetitions"
    )

    var benchmark = Bench(
        BenchConfig(
            num_warmup_iters=10,
            max_iters=1_000,
            num_repetitions=5,
        )
    )
    comptime decode_benchmark = bench_rms_norm[1]
    comptime prefill_benchmark = bench_rms_norm[128]
    benchmark.bench_function[decode_benchmark](
        BenchId("rms_norm_apple_gpu", "rows=1 hidden=896"),
        [
            ThroughputMeasure(BenchMetric.elements, HIDDEN_SIZE),
            ThroughputMeasure(BenchMetric.bytes, HIDDEN_SIZE * 6),
        ],
    )
    benchmark.bench_function[prefill_benchmark](
        BenchId("rms_norm_apple_gpu", "rows=128 hidden=896"),
        [
            ThroughputMeasure(BenchMetric.elements, 128 * HIDDEN_SIZE),
            ThroughputMeasure(BenchMetric.bytes, 128 * HIDDEN_SIZE * 6),
        ],
    )
    print(benchmark)
