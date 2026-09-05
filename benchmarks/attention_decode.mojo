"""Paired synchronized decode latency; compilation and gates precede timing."""

from attention_decode_support import fill_decode, assert_decode_close
from layout import TensorLayout, TileTensor, row_major
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.attention_decode import (
    enqueue_grouped_query_attention_decode_apple_gpu,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv, is_defined, get_defined_int
from std.time import perf_counter_ns, sleep


def enqueue_variant[
    QL: TensorLayout,
    KL: TensorLayout,
    SL: TensorLayout,
    WL: TensorLayout,
](
    variant: Int,
    ctx: DeviceContext,
    query: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, QL, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, SL, MutAnyOrigin],
    workspace: TileTensor[DType.float32, WL, MutAnyOrigin],
) raises -> Int:
    if variant == 0:
        enqueue_grouped_query_attention_apple_gpu(
            ctx, query, key, value, scratch, output
        )
        return 0
    elif variant == 1:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 1](
            ctx, query, key, value, output, workspace
        )
        return 1
    elif variant == 2:
        enqueue_grouped_query_attention_decode_apple_gpu[2, 1, 1](
            ctx, query, key, value, output, workspace
        )
        return 2
    elif variant == 3:
        enqueue_grouped_query_attention_decode_apple_gpu[8, 1, 1](
            ctx, query, key, value, output, workspace
        )
        return 3
    elif variant == 4:
        enqueue_grouped_query_attention_decode_apple_gpu[32, 1, 1](
            ctx, query, key, value, output, workspace
        )
        return 4
    elif variant == 5:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 4](
            ctx, query, key, value, output, workspace
        )
        return 5
    elif variant == 6:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 16](
            ctx, query, key, value, output, workspace
        )
        return 6
    elif variant == 7:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 64](
            ctx, query, key, value, output, workspace
        )
        return 7
    elif variant == 8:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 2, 64](
            ctx, query, key, value, output, workspace
        )
        return 8
    elif variant == 9:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 4, 64](
            ctx, query, key, value, output, workspace
        )
        return 9
    elif variant == 10:
        enqueue_grouped_query_attention_decode_apple_gpu[1, 7, 64](
            ctx, query, key, value, output, workspace
        )
        return 10
    elif variant == 11:
        enqueue_grouped_query_attention_decode_apple_gpu[
            32, 1, 1, conditional_rescale=True
        ](ctx, query, key, value, output, workspace)
        return 11
    elif variant == 12:
        enqueue_grouped_query_attention_decode_apple_gpu[
            1, 1, 64, conditional_rescale=True
        ](ctx, query, key, value, output, workspace)
        return 12
    else:
        raise Error("unknown decode variant")


def variant_splits(variant: Int) -> Int:
    if variant == 5:
        return 4
    if variant == 6:
        return 16
    if variant >= 7 and variant != 11:
        return 64
    return 1


def main() raises:
    var args = List[String]()
    comptime if is_defined["GQA_PROFILE_ROWS"]():
        args = [
            "profile",
            String(get_defined_int["GQA_PROFILE_ROWS"]()),
            "1",
            String(get_defined_int["GQA_PROFILE_VARIANT"]()),
            "0",
            "1",
            "17",
            "profile",
            String(get_defined_int["GQA_PROFILE_ITERATIONS"]()),
            "100",
        ]
    else:
        for arg in argv():
            args.append(String(arg))
    if len(args) != 10:
        raise Error(
            "expected T layers candidate control candidate-first seed mode"
            " repetitions warmup"
        )
    var rows = Int(args[1])
    var layers = Int(args[2])
    var candidate = Int(args[3])
    var control = Int(args[4])
    var first = Int(args[5])
    var seed = Int(args[6])
    var mode = args[7]
    var repetitions = Int(args[8])
    var warmup = Int(args[9])
    if (
        rows < 1
        or rows > 4096
        or (layers != 1 and layers != 24)
        or candidate < 0
        or candidate > 12
        or control < 0
        or control > 12
        or repetitions < 1
        or warmup < 0
    ):
        raise Error("invalid decode benchmark arguments")
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("decode benchmark requires Metal")
    print("device:", ctx.name())
    print("api:", ctx.api())
    print("shape:", rows, layers, "seed:", seed)
    print("variants:", control, candidate, "candidate-first:", first)
    var ql = row_major(1, 14, 64)
    var kl = row_major(rows, 2, 64)
    var sl = row_major(1, 14, rows)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](896)
    var sb = ctx.enqueue_create_buffer[DType.bfloat16](14 * rows)
    var wb = ctx.enqueue_create_buffer[DType.float32](14 * 64 * 66)
    var output = TileTensor(ob, ql)
    var scratch = TileTensor(sb, sl)
    var queries = List[DeviceBuffer[DType.bfloat16]]()
    var keys = List[DeviceBuffer[DType.bfloat16]]()
    var values = List[DeviceBuffer[DType.bfloat16]]()
    for layer in range(layers):
        var qb = ctx.enqueue_create_buffer[DType.bfloat16](896)
        var kb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 128)
        var vb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 128)
        with qb.map_to_host() as qm:
            with kb.map_to_host() as km:
                with vb.map_to_host() as vm:
                    var q = TileTensor(qm, ql)
                    var k = TileTensor(km, kl)
                    var v = TileTensor(vm, kl)
                    fill_decode(q, k, v, seed + layer * 13)
        var q = TileTensor(qb, ql)
        var k = TileTensor(kb, kl)
        var v = TileTensor(vb, kl)
        enqueue_grouped_query_attention_apple_gpu(ctx, q, k, v, scratch, output)
        var expected = List[Float32]()
        with ob.map_to_host() as mapped:
            var result = TileTensor(mapped, ql)
            comptime assert result.flat_rank == 3
            for h in range(14):
                for d in range(64):
                    expected.append(
                        rebind[Float32](result[0, h, d].cast[DType.float32]())
                    )
        for arm in range(2):
            var variant = control if arm == 0 else candidate
            var workspace = TileTensor(
                wb, row_major(14, variant_splits(variant), 66)
            )
            var launched = enqueue_variant(
                variant, ctx, q, k, v, output, scratch, workspace
            )
            if launched != variant:
                raise Error("benchmark routed to the wrong implementation")
            ctx.synchronize()
            with ob.map_to_host() as mapped:
                _ = assert_decode_close(TileTensor(mapped, ql), expected)
        queries.append(qb^)
        keys.append(kb^)
        values.append(vb^)
    print("correctness: passed")
    ctx.synchronize()

    if mode == "profile":
        var workspace = TileTensor(
            wb, row_major(14, variant_splits(candidate), 66)
        )
        var q = TileTensor(queries[0], ql)
        var k = TileTensor(keys[0], kl)
        var v = TileTensor(values[0], kl)
        for _ in range(warmup):
            _ = enqueue_variant(candidate, ctx, q, k, v, output, scratch, workspace)
        ctx.synchronize()
        print(
            "profile implementation:",
            "enqueue_grouped_query_attention_apple_gpu" if candidate
            == 0 else "enqueue_grouped_query_attention_decode_apple_gpu",
        )
        print("rows: 1")
        print("hidden: 64")
        print("key value rows:", rows)
        print("query heads: 14")
        print("key value heads: 2")
        print(
            "groups:",
            0 if candidate
            == 0 else (
                2 if candidate
                == 2 else (
                    8 if candidate
                    == 3 else (32 if (candidate == 4 or candidate == 11) else 1)
                )
            ),
        )
        print(
            "heads:",
            0 if candidate
            == 0 else (
                2 if candidate
                == 8 else (
                    4 if candidate == 9 else (7 if candidate == 10 else 1)
                )
            ),
        )
        print("splits:", 0 if candidate == 0 else variant_splits(candidate))
        print(
            "profile workload:",
            "decode-t" + String(rows) + "-v" + String(candidate),
        )
        print(
            "profile dispatches per iteration:",
            3 if candidate
            == 0 else (2 if variant_splits(candidate) > 1 else 1),
        )
        print("warmup iterations:", warmup)
        print("profile iterations:", repetitions)
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(repetitions):
            _ = enqueue_variant(candidate, ctx, q, k, v, output, scratch, workspace)
        ctx.synchronize()
        print("PROFILE_REGION_END")
        sleep(0.25)
        return

    for arm in range(2):
        var variant = candidate if ((arm == 0) == (first == 1)) else control
        var label = "candidate" if ((arm == 0) == (first == 1)) else "control"
        var workspace = TileTensor(
            wb, row_major(14, variant_splits(variant), 66)
        )
        for sample in range(warmup + repetitions):
            var started = perf_counter_ns()
            for layer in range(layers):
                var q = TileTensor(queries[layer], ql)
                var k = TileTensor(keys[layer], kl)
                var v = TileTensor(values[layer], kl)
                _ = enqueue_variant(
                    variant, ctx, q, k, v, output, scratch, workspace
                )
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - started) / Float64(
                1000 * layers
            )
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, elapsed)
    print("BENCHMARK_COMPLETE")
