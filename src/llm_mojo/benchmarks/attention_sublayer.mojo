"""Whole attention latency and trace instrument with frozen upstream gates.

Python only reads fixture arrays before timing. All enqueues are engine Mojo.
Ring24 owns distinct weights, inputs and caches; scratch/output are shared.
Odd ring entries negate X, Wqkv and Wo, preserving Q/K/V and negating Y.
"""
from llm_mojo.attention_sublayer import (
    AttentionWeights, AttentionCache, AttentionWorkspace, enqueue_attention_sublayer,
)
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.python import Python, PythonObject
from std.math import isfinite
from std.sys import argv, is_defined, get_defined_int
from std.time import perf_counter_ns, sleep


def _array(name: String) raises -> PythonObject:
    var np = Python.import_module("numpy")
    return np.load("build/oracle_data/attention_sublayer/" + name + ".npy",
                   allow_pickle=False).reshape(-1)


def _load(mut buffer: DeviceBuffer[DType.bfloat16], name: String,
          count: Int, start: Int = 0, sign: Float32 = 1) raises:
    var a = _array(name)
    if start < 0 or start + count > Int(py=a.size):
        raise Error("benchmark fixture slice is outside its frozen array")
    var source = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    with buffer.map_to_host() as mapped:
        var dst = TileTensor(mapped, row_major(count))
        comptime assert dst.flat_rank == 1
        for i in range(count):
            dst[i] = (sign * source[unsafe_offset=start + i]).cast[DType.bfloat16]()


def _poison_suffix(mut buffer: DeviceBuffer[DType.bfloat16], past: Int, rows: Int) raises:
    with buffer.map_to_host() as mapped:
        var dst = TileTensor(mapped, row_major((past + rows) * 128))
        comptime assert dst.flat_rank == 1
        for i in range(past * 128, (past + rows) * 128):
            dst[i] = 123


def _check(mut buffer: DeviceBuffer[DType.bfloat16], name: String,
           count: Int, start: Int, sign: Float32) raises:
    var a = _array(name)
    var source = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    var worst: Float32 = 0
    with buffer.map_to_host() as mapped:
        var result = TileTensor(mapped, row_major(count))
        comptime assert result.flat_rank == 1
        for i in range(count):
            var got = rebind[Float32](result[i].cast[DType.float32]())
            var want = sign * source[unsafe_offset=start + i]
            var error = abs(got - want) / (1 + abs(want))
            if not isfinite(got) or not isfinite(want) or error > 0.03125:
                print("benchmark fixture mismatch", name, i, got, want)
                raise Error("attention benchmark exceeded its FP32 composition gate")
            worst = max(worst, error)
    print("fixture gate:", name, "max scaled:", worst)


def _check_cache(mut cache: DeviceBuffer[DType.bfloat16],
                 mut source: DeviceBuffer[DType.bfloat16], name: String,
                 past: Int, rows: Int) raises:
    var a = _array(name)
    var prefix = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    with cache.map_to_host() as mapped:
        var actual = TileTensor(mapped, row_major((past + rows) * 128))
        comptime assert actual.flat_rank == 1
        for i in range(past * 128):
            # Frozen arrays store exact BF16 values expanded losslessly to FP32.
            var want = UInt16(prefix.unsafe_bitcast[UInt32]()[unsafe_offset=i] >> 16)
            if actual.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i] != want:
                raise Error("benchmark changed a cache prefix bit")
        with source.map_to_host() as sm:
            var original = TileTensor(sm, row_major(rows * 128))
            comptime assert original.flat_rank == 1
            for i in range(rows * 128):
                if actual.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=past * 128 + i] != original.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i]:
                    raise Error("benchmark cache append changed a source bit")


def _enqueue(ctx: DeviceContext, mut weights: AttentionWeights,
             mut cache: AttentionCache, mut work: AttentionWorkspace,
             mut input: DeviceBuffer[DType.bfloat16], r: Int, t: Int,
             variant: Int) raises:
    # Repeat a fixed suffix on the same stream. No prefix upload or reset sync.
    cache.length = t - r
    var route = variant - 1 if variant >= 5 else 3
    var launched = enqueue_attention_sublayer(
        ctx, weights, cache, work, TileTensor(input, row_major(r, 896)),
        route, wo_mma=variant == 4,
    )
    if launched != route or cache.length != t:
        raise Error("attention benchmark route or cache length mismatch")


def main() raises:
    var args = List[String]()
    comptime if is_defined["GQA_PROFILE_ROWS"]():
        args = ["profile", String(get_defined_int["GQA_PROFILE_QUERY_ROWS"]()),
                String(get_defined_int["GQA_PROFILE_ROWS"]()), "1",
                String(get_defined_int["GQA_PROFILE_VARIANT"]()), "3", "1", "53",
                "profile", String(get_defined_int["GQA_PROFILE_ITERATIONS"]()),
                String(get_defined_int["GQA_PROFILE_WARMUP", default=10]())]
    else:
        for arg in argv():
            args.append(String(arg))
    if len(args) != 11:
        raise Error("expected R T layers candidate control candidate-first seed mode repetitions warmup")
    var r = Int(args[1])
    var t = Int(args[2])
    var layers = Int(args[3])
    var candidate = Int(args[4])
    var control = Int(args[5])
    var first = Int(args[6])
    var seed = Int(args[7])
    var mode = args[8]
    var repetitions = Int(args[9])
    var warmup = Int(args[10])
    var dispatches = 10 if candidate == 5 else (11 if candidate == 6 else 12)
    if (r < 1 or r > t or t > 4096 or (layers != 1 and layers != 24)
        or candidate < 3 or candidate > 6 or control != 3 or seed != 53
        or (candidate >= 5 and r != 1)
        or (first != 0 and first != 1) or (mode != "bench" and mode != "profile")
        or (mode == "profile" and (layers != 1 or repetitions * dispatches > 5000))
        or repetitions < 1 or warmup < 0):
        raise Error("invalid attention sublayer benchmark arguments")
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("attention benchmark requires Metal")
    print("device:", ctx.name())
    print("api:", ctx.api())
    print("operation: attention_sublayer")
    print("query rows:", r)
    print("shape:", t, layers, "seed:", seed)
    print("variants:", control, candidate, "candidate-first:", first)
    var work = AttentionWorkspace(ctx, r, t)
    _load(work.cosine, "upstream_7_cosine", t * 64)
    _load(work.sine, "upstream_7_sine", t * 64)
    var weights = List[AttentionWeights]()
    var caches = List[AttentionCache]()
    var inputs = List[DeviceBuffer[DType.bfloat16]]()
    for layer in range(layers):
        var sign: Float32 = -1 if layer % 2 else 1
        var w = AttentionWeights(ctx)
        var cache = AttentionCache(ctx, t)
        var input = ctx.enqueue_create_buffer[DType.bfloat16](r * 896)
        _load(w.qkv, "7_weight", 1152 * 896, sign=sign)
        _load(w.output, "7_output_weight", 896 * 896, sign=sign)
        _load(w.bias, "7_bias", 1152)
        _load(w.norm, "7_norm_weight", 896)
        _load(input, "7_input", r * 896, (t - r) * 896, sign)
        _load(cache.key, "upstream_7_cache_key", t * 128)
        _load(cache.value, "upstream_7_cache_value", t * 128)
        for arm in range(2):
            _poison_suffix(cache.key, t - r, r)
            _poison_suffix(cache.value, t - r, r)
            work.attention.enqueue_fill(123)
            work.projected.enqueue_fill(123)
            work.output.enqueue_fill(123)
            _enqueue(ctx, w, cache, work, input, r, t, control if arm == 0 else candidate)
            _check(work.projected, "fp32_7_projected", r * 896, (t - r) * 896, sign)
            _check(work.output, "fp32_7_output", r * 896, (t - r) * 896, sign)
            _check_cache(cache.key, work.rotated_key, "upstream_7_cache_key", t - r, r)
            _check_cache(cache.value, work.raw_value, "upstream_7_cache_value", t - r, r)
        weights.append(w^)
        caches.append(cache^)
        inputs.append(input^)
    ctx.synchronize()
    print("correctness: passed")

    if mode == "profile":
        for _ in range(warmup):
            _enqueue(ctx, weights[0], caches[0], work, inputs[0], r, t, candidate)
        ctx.synchronize()
        print("profile implementation: enqueue_attention_sublayer")
        print("rows:", r)
        print("hidden: 896")
        print("key value rows:", t)
        print("query heads: 14")
        print("key value heads: 2")
        print("profile workload:", "sublayer-r" + String(r) + "-t" + String(t) + "-v" + String(candidate))
        print("profile dispatches per iteration:", dispatches)
        print("warmup iterations:", warmup)
        print("profile iterations:", repetitions)
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(repetitions):
            _enqueue(ctx, weights[0], caches[0], work, inputs[0], r, t, candidate)
        ctx.synchronize()
        print("PROFILE_REGION_END")
        sleep(0.25)
        return

    for arm in range(2):
        var is_candidate = (arm == 0) == (first == 1)
        var variant = candidate if is_candidate else control
        var label = "candidate" if is_candidate else "control"
        for sample in range(warmup + repetitions):
            var start = perf_counter_ns()
            for layer in range(layers):
                _enqueue(ctx, weights[layer], caches[layer], work, inputs[layer], r, t, variant)
            ctx.synchronize()
            var us = Float64(perf_counter_ns() - start) / Float64(1000 * layers)
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, us)
    print("BENCHMARK_COMPLETE")
