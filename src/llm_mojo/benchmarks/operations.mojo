"""Inspectable paired timings for RMSNorm, QKV projection, RoPE, and argmax.

All allocations and output checks precede timing. Each sample enqueues a hot
call or a 24-buffer sweep, synchronizes once, and divides by the layer count.
Argmax uses that count for repetitions over the same resident buffers.
"""
from layout import TensorLayout, TileTensor, row_major
from llm_mojo.kernels.rms_norm import enqueue_rms_norm_apple_gpu, enqueue_rms_norm_apple_gpu_shared_tree
from llm_mojo.kernels.linear import (
    enqueue_linear_apple_gpu, enqueue_linear_apple_gpu_two_output,
    enqueue_linear_prefill_direct_apple_gpu, enqueue_linear_prefill_tiled_apple_gpu_bk,
    enqueue_linear_prefill_register_2x2_apple_gpu, enqueue_linear_prefill_mma_8x16_apple_gpu,
)
from llm_mojo.kernels.rope import enqueue_rope_apple_gpu
from llm_mojo.kernels.token_selection import enqueue_argmax
from llm_mojo.kernels.token_selection_contract import bf16_rank
from llm_mojo.kernels.argmax_reduction import balanced_argmax_plan
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns
from std.math import isfinite


def linear_route[IL: TensorLayout, WL: TensorLayout, BL: TensorLayout, OL: TensorLayout](
    variant: Int, ctx: DeviceContext,
    x: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    w: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    b: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
) raises -> Int:
    if variant == 0:
        # Q, K, V are contiguous slices of the same packed weights and output.
        if Int(x.dim[0]()) != 1:
            raise Error("separate QKV comparison is decode only")
        enqueue_linear_apple_gpu(ctx, x, w.tile[896, 896](0, 0), b.tile[896](0), y.tile[1, 896](0, 0))
        enqueue_linear_apple_gpu(ctx, x, w.tile[128, 896](7, 0), b.tile[128](7), y.tile[1, 128](0, 7))
        enqueue_linear_apple_gpu(ctx, x, w.tile[128, 896](8, 0), b.tile[128](8), y.tile[1, 128](0, 8))
        return 0
    elif variant == 1:
        enqueue_linear_apple_gpu(ctx, x, w, b, y)
        return 1
    elif variant == 2:
        enqueue_linear_apple_gpu_two_output(ctx, x, w, b, y)
        return 2
    elif variant == 3:
        enqueue_linear_prefill_direct_apple_gpu(ctx, x, w, b, y)
        return 3
    elif variant == 4:
        enqueue_linear_prefill_tiled_apple_gpu_bk[16](ctx, x, w, b, y)
        return 4
    elif variant == 5:
        enqueue_linear_prefill_register_2x2_apple_gpu(ctx, x, w, b, y)
        return 5
    elif variant == 6:
        enqueue_linear_prefill_mma_8x16_apple_gpu(ctx, x, w, b, y)
        return 6
    raise Error("unknown linear route")




def linear(ctx: DeviceContext, rows: Int, layers: Int, control: Int, candidate: Int, first: Int, seed: Int, repetitions: Int, warmup: Int) raises:
    var xl = row_major(rows, 896)
    var yl = row_major(rows, 1152)
    var wl = row_major(1152, 896)
    var bl = row_major(1152)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
    var yb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 1152)
    var bb = ctx.enqueue_create_buffer[DType.bfloat16](1152)
    xb.enqueue_fill(0.5)
    bb.enqueue_fill(0.125)
    var x = TileTensor(xb, xl)
    var y = TileTensor(yb, yl)
    var b = TileTensor(bb, bl)
    var weights = List[DeviceBuffer[DType.bfloat16]]()
    for layer in range(layers):
        var wb = ctx.enqueue_create_buffer[DType.bfloat16](1152 * 896)
        wb.enqueue_fill((Float32(1 + (seed + layer) % 3) / 512.0).cast[DType.bfloat16]())
        weights.append(wb^)
    def launch(variant: Int, layer: Int) raises {imm, mut weights} -> Int:
        return linear_route(variant, ctx, x, TileTensor(weights[layer], wl), b, y)
    for layer in range(layers):
        for arm in range(2):
            yb.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
            var variant = control if arm == 0 else candidate
            if launch(variant, layer) != variant:
                raise Error("wrong linear route")
            ctx.synchronize()
            var expected = Float32(1 + (seed + layer) % 3) * 0.875 + 0.125
            with yb.map_to_host() as mapped:
                var result = TileTensor(mapped, yl)
                for r in range(rows):
                    for n in range(1152):
                        var actual = rebind[Float32](result[r, n].cast[DType.float32]())
                        if not isfinite(actual) or abs(actual - expected) > 0.015625:
                            raise Error("linear benchmark output check failed")
    print("correctness: passed")
    for arm in range(2):
        var is_candidate = (arm == 0) == (first == 1)
        var variant = candidate if is_candidate else control
        var label = "candidate" if is_candidate else "control"
        for sample in range(warmup + repetitions):
            var started = perf_counter_ns()
            for layer in range(layers):
                _ = launch(variant, layer)
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - started) / Float64(1000 * layers)
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, elapsed)
    print("BENCHMARK_COMPLETE")


def rms_norm(ctx: DeviceContext, rows: Int, layers: Int, control: Int, candidate: Int, first: Int, seed: Int, repetitions: Int, warmup: Int) raises:
    var xl = row_major(rows, 896)
    var wl = row_major(896)
    var yb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
    var wb = ctx.enqueue_create_buffer[DType.bfloat16](896)
    wb.enqueue_fill(1.0)
    var y = TileTensor(yb, xl)
    var w = TileTensor(wb, wl)
    var inputs = List[DeviceBuffer[DType.bfloat16]]()
    for layer in range(layers):
        var xb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
        xb.enqueue_fill(Scalar[DType.bfloat16](1 + (seed + layer) % 3))
        inputs.append(xb^)
    def launch(variant: Int, layer: Int) raises {imm, mut inputs} -> Int:
        var x = TileTensor(inputs[layer], xl)
        if variant == 0:
            enqueue_rms_norm_apple_gpu_shared_tree(ctx, x, w, y)
            return 0
        elif variant == 1:
            enqueue_rms_norm_apple_gpu(ctx, x, w, y)
            return 1
        raise Error("unknown RMSNorm route")
    for layer in range(layers):
        for arm in range(2):
            yb.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
            var variant = control if arm == 0 else candidate
            if launch(variant, layer) != variant:
                raise Error("wrong RMSNorm route")
            ctx.synchronize()
            with yb.map_to_host() as mapped:
                var result = TileTensor(mapped, xl)
                for r in range(rows):
                    for d in range(896):
                        var actual = rebind[Float32](result[r, d].cast[DType.float32]())
                        if not isfinite(actual) or abs(actual - 1.0) > 0.015625:
                            raise Error("RMSNorm benchmark output check failed")
    print("correctness: passed")
    for arm in range(2):
        var is_candidate = (arm == 0) == (first == 1)
        var variant = candidate if is_candidate else control
        var label = "candidate" if is_candidate else "control"
        for sample in range(warmup + repetitions):
            var started = perf_counter_ns()
            for layer in range(layers):
                _ = launch(variant, layer)
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - started) / Float64(1000 * layers)
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, elapsed)
    print("BENCHMARK_COMPLETE")


def rope(ctx: DeviceContext, rows: Int, layers: Int, control: Int, candidate: Int, first: Int, seed: Int, repetitions: Int, warmup: Int) raises:
    var xl = row_major(rows, 14, 64)
    var tl = row_major(rows, 64)
    var yb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
    var cb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 64)
    var sb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 64)
    cb.enqueue_fill(0.5)
    sb.enqueue_fill(0.25)
    var y = TileTensor(yb, xl)
    var c = TileTensor(cb, tl)
    var s = TileTensor(sb, tl)
    var inputs = List[DeviceBuffer[DType.bfloat16]]()
    for layer in range(layers):
        var xb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
        xb.enqueue_fill(Scalar[DType.bfloat16](1 + (seed + layer) % 3))
        inputs.append(xb^)
    def launch(variant: Int, layer: Int) raises {imm, mut inputs} -> Int:
        if variant != 0:
            raise Error("unknown RoPE route")
        enqueue_rope_apple_gpu(ctx, TileTensor(inputs[layer], xl), c, s, y, 0)
        return 0
    for layer in range(layers):
        yb.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
        if launch(0, layer) != 0:
            raise Error("wrong RoPE route")
        ctx.synchronize()
        with yb.map_to_host() as mapped:
            var result = TileTensor(mapped, xl)
            for r in range(rows):
                for h in range(14):
                    for d in range(64):
                        var expected = Float32(1 + (seed + layer) % 3) * Float32(0.25 if d < 32 else 0.75)
                        var actual = rebind[Float32](result[r, h, d].cast[DType.float32]())
                        if not isfinite(actual) or actual != expected:
                            raise Error("RoPE benchmark output check failed")
    print("correctness: passed")
    for arm in range(2):
        var is_candidate = (arm == 0) == (first == 1)
        var variant = candidate if is_candidate else control
        var label = "candidate" if is_candidate else "control"
        for sample in range(warmup + repetitions):
            var started = perf_counter_ns()
            for layer in range(layers):
                _ = launch(variant, layer)
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - started) / Float64(1000 * layers)
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, elapsed)
    print("BENCHMARK_COMPLETE")


def token_selection_route[LL: TensorLayout, PL: TensorLayout, RL: TensorLayout](
    variant: Int, ctx: DeviceContext,
    logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin],
) raises -> Int:
    if variant == 0:
        enqueue_argmax(ctx, logits, partials, result)
    elif variant == 1:
        enqueue_argmax[plan=balanced_argmax_plan[8]()](ctx, logits, partials, result)
    elif variant == 2:
        enqueue_argmax[finish_threads=64, finish_items_per_thread=3,
                      finish_plan=balanced_argmax_plan[3]()](ctx, logits, partials, result)
    else:
        raise Error("unknown token selection route")
    return variant


def token_selection(ctx: DeviceContext, rows: Int, layers: Int, control: Int, candidate: Int, first: Int, seed: Int, repetitions: Int, warmup: Int) raises:
    # A regression diagnostic for Qwen's rounded logits, with allocation and
    # numerical checks outside timing. Both arms reuse identical resident data.
    comptime count = 151936
    comptime groups = (count + 1023) // 1024
    var ll = row_major(rows, count)
    var pl = row_major(rows * groups, 3)
    var rl = row_major(rows, 3)
    var logits_buffer = ctx.enqueue_create_buffer[DType.bfloat16](rows * count)
    var partials_buffer = ctx.enqueue_create_buffer[DType.uint32](rows * groups * 3 + 3)
    var result_buffer = ctx.enqueue_create_buffer[DType.uint32](rows * 3 + 3)
    var logits = TileTensor(logits_buffer, ll)
    var partials = TileTensor(partials_buffer, pl)
    var result = TileTensor(result_buffer, rl)
    var phase = ((seed % 127) + 127) % 127
    def launch(variant: Int) raises {imm} -> Int:
        return token_selection_route(variant, ctx, logits, partials, result)
    # Poison first, then leave finite tied inputs resident for timed calls.
    for pattern in [2, 0, 1]:
        with logits_buffer.map_to_host() as mapped:
            for row in range(rows):
                var early = 1 + (phase + row * 17) % 1023
                var late = count - 1 - row % 127
                for i in range(count):
                    var bits = UInt16(0xBF80 + (i + row + phase) % 127)
                    if i == late or (pattern != 0 and i == early):
                        bits = UInt16(0x3F80)
                    if pattern == 2 and i == late:
                        bits = UInt16(0xFFFF)  # Losing NaN must still poison the result.
                    mapped.unsafe_ptr()[unsafe_offset=row * count + i] = bitcast[DType.bfloat16](bits)
        for arm in range(2):
            var variant = control if arm == 0 else candidate
            partials_buffer.enqueue_fill(0xDEADBEEF)
            result_buffer.enqueue_fill(0xDEADBEEF)
            if launch(variant) != variant:
                raise Error("wrong token selection route")
            ctx.synchronize()
            with result_buffer.map_to_host() as mapped:
                for row in range(rows):
                    var early = 1 + (phase + row * 17) % 1023
                    var late = count - 1 - row % 127
                    var expected_id = late if pattern == 0 else early
                    if (mapped.unsafe_ptr()[unsafe_offset=row * 3] != bf16_rank(UInt16(0x3F80))
                            or mapped.unsafe_ptr()[unsafe_offset=row * 3 + 1] != UInt32(expected_id)
                            or mapped.unsafe_ptr()[unsafe_offset=row * 3 + 2] != UInt32(pattern == 2)):
                        raise Error("token selection benchmark output check failed")
                for i in range(rows * 3, rows * 3 + 3):
                    if mapped.unsafe_ptr()[unsafe_offset=i] != UInt32(0xDEADBEEF):
                        raise Error("token selection result guard changed")
            with partials_buffer.map_to_host() as mapped:
                for i in range(rows * groups * 3, rows * groups * 3 + 3):
                    if mapped.unsafe_ptr()[unsafe_offset=i] != UInt32(0xDEADBEEF):
                        raise Error("token selection scratch guard changed")
    print("correctness: passed")
    print("token-count:", count, "threads:", 128, "items-per-thread:", 8)
    print("finishing: variants 0/1 use 128 threads x 1 item; variant 2 uses 64 x 3 balanced")
    print("logits-dtype: bfloat16 record-dtype: uint32 reuse: same resident buffers")
    for arm in range(2):
        var is_candidate = (arm == 0) == (first == 1)
        var variant = candidate if is_candidate else control
        var label = "candidate" if is_candidate else "control"
        for sample in range(warmup + repetitions):
            var started = perf_counter_ns()
            for _ in range(layers):
                _ = launch(variant)
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - started) / Float64(1000 * layers)
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, elapsed)
    print("BENCHMARK_COMPLETE")


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if len(args) != 11:
        raise Error("expected operation rows layers candidate control candidate-first seed mode repetitions warmup")
    var operation = args[1]
    var rows = Int(args[2])
    var layers = Int(args[3])
    var candidate = Int(args[4])
    var control = Int(args[5])
    var first = Int(args[6])
    var seed = Int(args[7])
    var repetitions = Int(args[9])
    var warmup = Int(args[10])
    if rows < 1 or rows > 4096 or (layers != 1 and layers != 24) or repetitions < 1 or warmup < 0 or (first != 0 and first != 1) or args[8] != "bench":
        raise Error("invalid benchmark arguments")
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("benchmark requires Metal")
    print("device:", ctx.name())
    print("api:", ctx.api())
    print("operation:", operation)
    print("shape:", rows, layers, "seed:", seed)
    print("variants:", control, candidate, "candidate-first:", first)
    if operation == "linear":
        linear(ctx, rows, layers, control, candidate, first, seed, repetitions, warmup)
    elif operation == "rms_norm":
        rms_norm(ctx, rows, layers, control, candidate, first, seed, repetitions, warmup)
    elif operation == "rope":
        rope(ctx, rows, layers, control, candidate, first, seed, repetitions, warmup)
    elif operation == "token_selection":
        token_selection(ctx, rows, layers, control, candidate, first, seed, repetitions, warmup)
    else:
        raise Error("unknown benchmark operation")
