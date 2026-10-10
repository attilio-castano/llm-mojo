"""Bounded public-API gate for queued Metal metadata uploads.

Build with the locked toolchain and run on the declared Metal device:
  uv run --locked mojo build -I src src/llm_mojo/benchmarks/metadata_probe.mojo -o /tmp/metadata-probe
  uv run --locked /tmp/metadata-probe [rounds=200000] [pairs=5]

All buffers and kernels are warmed before sampling. A first kernel reads the
current metadata and performs a dependent integer workload; the host then
submits a second metadata record into the same device buffer. Same-context
ordering protects device reuse. Two distinct pinned host sources remain alive
and immutable through final synchronization. This proves only the bounded
upload operation, not model overlap, safe ring reuse, or asynchronous request
completion. No private driver ABI or CPU fallback is used.

Public API contract: max.gpu.host.device_context DeviceContext, HostBuffer,
DeviceBuffer, and DeviceEvent in the official Modular API reference. MAX 26.6
release notes add Apple DeviceEvent support; this probe reports the actual
create_event result on the repository's locked release instead of assuming it.
"""
from layout import TileTensor, TensorLayout, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu import global_idx
from std.sys import argv
from llm_mojo.runtime.clock import now

comptime WORK_ROWS = 4096
comptime METADATA_ROWS = 2048


def _busy[ML: TensorLayout, OL: TensorLayout](
    metadata: TileTensor[DType.uint32, ML, ImmutAnyOrigin],
    result: TileTensor[DType.uint32, OL, MutAnyOrigin], rounds: Int32,
):
    comptime assert metadata.flat_rank == 1 and result.flat_rank == 1
    var i = Int(global_idx.x)
    if i >= WORK_ROWS:
        return
    var tag = metadata[0]
    var x = UInt32(i + 1) ^ tag
    for _ in range(Int(rounds)):
        x = x * UInt32(1664525) + UInt32(1013904223)
        x = x ^ (x >> 13)
    result[i] = x
    # The address depends on the completed recurrence, preventing a read of
    # the old record from being moved ahead of the long workload.
    result[WORK_ROWS + i] = metadata[Int(x & UInt32(METADATA_ROWS - 1))]


def _observe[ML: TensorLayout, OL: TensorLayout](
    metadata: TileTensor[DType.uint32, ML, ImmutAnyOrigin],
    result: TileTensor[DType.uint32, OL, MutAnyOrigin],
):
    comptime assert metadata.flat_rank == 1 and result.flat_rank == 1
    var i = Int(global_idx.x)
    if i < METADATA_ROWS:
        result[i] = metadata[i]


def _enqueue_busy(ctx: DeviceContext, metadata: DeviceBuffer[DType.uint32],
                  mut output: DeviceBuffer[DType.uint32], rounds: Int) raises:
    var meta = TileTensor(metadata, row_major(METADATA_ROWS))
    var out = TileTensor(output, row_major(WORK_ROWS * 2))
    ctx.enqueue_function[_busy[type_of(meta.layout), type_of(out.layout)]](
        meta, out, Int32(rounds), grid_dim=WORK_ROWS // 128, block_dim=128)


def _enqueue_observe(ctx: DeviceContext, metadata: DeviceBuffer[DType.uint32],
                     mut output: DeviceBuffer[DType.uint32]) raises:
    var meta = TileTensor(metadata, row_major(METADATA_ROWS))
    var out = TileTensor(output, row_major(METADATA_ROWS))
    ctx.enqueue_function[_observe[type_of(meta.layout), type_of(out.layout)]](
        meta, out, grid_dim=METADATA_ROWS // 128, block_dim=128)


def _verify(output: DeviceBuffer[DType.uint32], observed: DeviceBuffer[DType.uint32],
            rounds: Int) raises:
    with output.map_to_host() as work:
        for i in range(WORK_ROWS):
            var index = Int(work.unsafe_ptr()[unsafe_offset=i] & UInt32(METADATA_ROWS - 1))
            if work.unsafe_ptr()[unsafe_offset=WORK_ROWS + i] != UInt32(17 + index):
                raise Error("Step1 metadata was overwritten before consumption")
        var expected = UInt32(1) ^ UInt32(17)
        for _ in range(rounds):
            expected = expected * UInt32(1664525) + UInt32(1013904223)
            expected = expected ^ (expected >> 13)
        if work.unsafe_ptr()[unsafe_offset=0] != expected:
            raise Error("GPU workload did not match the independent scalar recurrence")
    with observed.map_to_host() as result:
        for i in range(METADATA_ROWS):
            if result.unsafe_ptr()[unsafe_offset=i] != UInt32(29 + i):
                raise Error("Step2 metadata upload failed at element " + String(i))


def _sample[MAPPED: Bool](ctx: DeviceContext, metadata: DeviceBuffer[DType.uint32],
                         mut output: DeviceBuffer[DType.uint32], mut observed: DeviceBuffer[DType.uint32],
                         first: HostBuffer[DType.uint32], second: HostBuffer[DType.uint32],
                         rounds: Int, pair: Int) raises:
    ctx.enqueue_copy(dst_buf=metadata, src_buf=first)
    ctx.synchronize()
    var start = now()
    _enqueue_busy(ctx, metadata, output, rounds)
    var submitted = now()
    comptime if MAPPED:
        with metadata.map_to_host() as mapped:
            for i in range(METADATA_ROWS):
                mapped.unsafe_ptr()[unsafe_offset=i] = UInt32(29 + i)
    else:
        # This source is idle: only `first` has been submitted in this sample.
        # Populate it while step1 can still be running, then retain it unchanged.
        for i in range(METADATA_ROWS):
            second[i] = UInt32(29 + i)
        ctx.enqueue_copy(dst_buf=metadata, src_buf=second)
    _enqueue_observe(ctx, metadata, observed)
    var staged = now()
    ctx.synchronize()
    var completed = now()
    _verify(output, observed, rounds)
    print("sample", pair, "mapped" if MAPPED else "pinned", Int(submitted - start),
          Int(staged - submitted), Int(completed - staged), Int(completed - start))


def main() raises:
    var args = argv()
    var rounds = Int(args[1]) if len(args) > 1 else 200000
    var pairs = Int(args[2]) if len(args) > 2 else 5
    if rounds < 1 or rounds > 2000000 or pairs < 1 or pairs > 20:
        raise Error("Use 1..2000000 rounds and 1..20 pairs for a bounded probe")
    var ctx = DeviceContext()
    print("device", ctx.name())
    print("api", ctx.api())
    if ctx.api() != "metal":
        raise Error("This probe requires a verified Metal backend")
    print("geometry uint32 metadata_rows", METADATA_ROWS, "work_rows", WORK_ROWS,
          "rounds", rounds, "pairs", pairs)
    try:
        var event = ctx.create_event()
        ctx.stream().record_event(event)
        event.synchronize()
        print("event supported")
    except e:
        print("event unsupported", e)
    var first = ctx.enqueue_create_host_buffer[DType.uint32](METADATA_ROWS)
    var second = ctx.enqueue_create_host_buffer[DType.uint32](METADATA_ROWS)
    var metadata = ctx.enqueue_create_buffer[DType.uint32](METADATA_ROWS)
    var output = ctx.enqueue_create_buffer[DType.uint32](WORK_ROWS * 2)
    var observed = ctx.enqueue_create_buffer[DType.uint32](METADATA_ROWS)
    ctx.synchronize()
    for i in range(METADATA_ROWS):
        first[i] = UInt32(17 + i)
        second[i] = UInt32(29 + i)
    # Warm the exact kernels, both upload arms, and all readback mappings.
    _sample[True](ctx, metadata, output, observed, first, second, 1, -1)
    _sample[False](ctx, metadata, output, observed, first, second, 1, -1)
    var idle_start = now()
    ctx.enqueue_copy(dst_buf=metadata, src_buf=second)
    _enqueue_observe(ctx, metadata, observed)
    var idle_submitted = now()
    ctx.synchronize()
    print("idle pinned", Int(idle_submitted - idle_start), Int(now() - idle_submitted))
    print("columns sample pair arm work_submit_ns metadata_submit_ns final_wait_ns total_ns")
    for pair in range(pairs):
        if pair % 2 == 0:
            _sample[True](ctx, metadata, output, observed, first, second, rounds, pair)
            _sample[False](ctx, metadata, output, observed, first, second, rounds, pair)
        else:
            _sample[False](ctx, metadata, output, observed, first, second, rounds, pair)
            _sample[True](ctx, metadata, output, observed, first, second, rounds, pair)
    # Explicitly retain the immutable pinned sources through the final wait.
    ctx.synchronize()
    if first[0] != 17 or second[0] != 29:
        raise Error("Pinned stage source was modified")
