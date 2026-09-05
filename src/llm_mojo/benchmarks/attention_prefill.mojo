"""Paired synchronized prefill latency; compilation and gates precede timing."""

from llm_mojo.benchmarks.attention_prefill_support import (
    fill_prefill,
    assert_prefill_close,
    enqueue_variant,
)
from layout import TensorLayout, TileTensor, row_major
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv, is_defined, get_defined_int
from std.time import perf_counter_ns, sleep


def main() raises:
    var args = List[String]()
    comptime if is_defined["GQA_PROFILE_ROWS"]():
        args = [
            "profile",
            String(get_defined_int["GQA_PROFILE_QUERY_ROWS"]()),
            String(get_defined_int["GQA_PROFILE_ROWS"]()),
            "1",
            String(get_defined_int["GQA_PROFILE_VARIANT"]()),
            "0",
            "1",
            "17",
            "profile",
            String(get_defined_int["GQA_PROFILE_ITERATIONS"]()),
            String(get_defined_int["GQA_PROFILE_WARMUP", default=100]()),
        ]
    else:
        for arg in argv():
            args.append(String(arg))
    if len(args) != 11:
        raise Error(
            "expected R T layers candidate control candidate-first seed mode"
            " repetitions warmup"
        )
    var query_rows = Int(args[1])
    var rows = Int(args[2])
    var layers = Int(args[3])
    var candidate = Int(args[4])
    var control = Int(args[5])
    var first = Int(args[6])
    var seed = Int(args[7])
    var mode = args[8]
    var repetitions = Int(args[9])
    var warmup = Int(args[10])
    if (
        query_rows < 1
        or query_rows > rows
        or rows < 1
        or rows > 4096
        or (layers != 1 and layers != 24)
        or candidate < 0
        or candidate > 10
        or control < 0
        or control > 10
        or (first != 0 and first != 1)
        or (mode != "bench" and mode != "profile")
        or (mode == "profile" and layers != 1)
        or repetitions < 1
        or warmup < 0
    ):
        raise Error("invalid prefill benchmark arguments")
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("prefill benchmark requires Metal")
    print("device:", ctx.name())
    print("api:", ctx.api())
    print("operation: gqa_prefill")
    print("query rows:", query_rows)
    print("shape:", rows, layers, "seed:", seed)
    print("variants:", control, candidate, "candidate-first:", first)
    var ql = row_major(query_rows, 14, 64)
    var kl = row_major(rows, 2, 64)
    var sl = row_major(query_rows, 14, rows)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](query_rows * 896)
    var sb = ctx.enqueue_create_buffer[DType.bfloat16](query_rows * 14 * rows)
    var output = TileTensor(ob, ql)
    var scratch = TileTensor(sb, sl)
    var queries = List[DeviceBuffer[DType.bfloat16]]()
    var keys = List[DeviceBuffer[DType.bfloat16]]()
    var values = List[DeviceBuffer[DType.bfloat16]]()
    for layer in range(layers):
        var qb = ctx.enqueue_create_buffer[DType.bfloat16](query_rows * 896)
        var kb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 128)
        var vb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 128)
        with qb.map_to_host() as qm:
            with kb.map_to_host() as km:
                with vb.map_to_host() as vm:
                    var q = TileTensor(qm, ql)
                    var k = TileTensor(km, kl)
                    var v = TileTensor(vm, kl)
                    fill_prefill(q, k, v, seed + layer * 13)
        var q = TileTensor(qb, ql)
        var k = TileTensor(kb, kl)
        var v = TileTensor(vb, kl)
        enqueue_grouped_query_attention_apple_gpu(ctx, q, k, v, scratch, output)
        var expected = List[Float32]()
        with ob.map_to_host() as mapped:
            var result = TileTensor(mapped, ql)
            comptime assert result.flat_rank == 3
            for r in range(query_rows):
                for h in range(14):
                    for d in range(64):
                        expected.append(
                            rebind[Float32](
                                result[r, h, d].cast[DType.float32]()
                            )
                        )
        for arm in range(2):
            var variant = control if arm == 0 else candidate
            var launched = enqueue_variant(
                variant, ctx, q, k, v, output, scratch
            )
            if launched != variant:
                raise Error("benchmark routed to the wrong implementation")
            ctx.synchronize()
            with ob.map_to_host() as mapped:
                assert_prefill_close(TileTensor(mapped, ql), expected)
        queries.append(qb^)
        keys.append(kb^)
        values.append(vb^)
    print("correctness: passed")
    ctx.synchronize()

    if mode == "profile":
        var q = TileTensor(queries[0], ql)
        var k = TileTensor(keys[0], kl)
        var v = TileTensor(values[0], kl)
        for _ in range(warmup):
            _ = enqueue_variant(candidate, ctx, q, k, v, output, scratch)
        ctx.synchronize()
        print(
            "profile implementation:",
            "enqueue_grouped_query_attention_apple_gpu" if candidate
            == 0 else (
                "enqueue_grouped_query_attention_prefill_materialized_apple_gpu" if candidate
                == 1 else "enqueue_grouped_query_attention_prefill_apple_gpu"
            ),
        )
        print("rows:", query_rows)
        print("hidden: 64")
        print("key value rows:", rows)
        print("query heads: 14")
        print("key value heads: 2")
        print(
            "query tile:",
            0 if candidate
            <= 1 else (
                4 if candidate
                == 2 else (
                    8 if candidate == 3
                    or candidate
                    == 10 else (32 if candidate == 5 or candidate == 8 else 16)
                )
            ),
        )
        print(
            "key tile:", 0 if candidate <= 1 else (64 if candidate == 6 else 32)
        )
        print(
            "heads:",
            0 if candidate
            <= 1 else (2 if candidate == 9 else (4 if candidate == 10 else 1)),
        )
        print(
            "profile workload:",
            "prefill-r"
            + String(query_rows)
            + "-t"
            + String(rows)
            + "-v"
            + String(candidate),
        )
        print("profile dispatches per iteration:", 3 if candidate <= 1 else 1)
        print("warmup iterations:", warmup)
        print("profile iterations:", repetitions)
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(repetitions):
            _ = enqueue_variant(candidate, ctx, q, k, v, output, scratch)
        ctx.synchronize()
        print("PROFILE_REGION_END")
        sleep(0.25)
        return

    for arm in range(2):
        var variant = candidate if ((arm == 0) == (first == 1)) else control
        var label = "candidate" if ((arm == 0) == (first == 1)) else "control"
        for sample in range(warmup + repetitions):
            var started = perf_counter_ns()
            for layer in range(layers):
                var q = TileTensor(queries[layer], ql)
                var k = TileTensor(keys[layer], kl)
                var v = TileTensor(values[layer], kl)
                _ = enqueue_variant(variant, ctx, q, k, v, output, scratch)
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - started) / Float64(
                1000 * layers
            )
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, elapsed)
    print("BENCHMARK_COMPLETE")
