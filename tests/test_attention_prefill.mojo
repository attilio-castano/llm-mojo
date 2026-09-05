from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_raises
from std.sys import get_defined_int
from llm_mojo.benchmarks.attention_prefill_support import (
    enqueue_variant,
    PREFILL_VARIANT_COUNT,
    fill_prefill,
    assert_prefill_close,
)
from oracle_data.attention.prefill_data import (
    PREFILL_CASE_COUNT,
    prefill_case_query_rows,
    prefill_case_key_rows,
    prefill_case_seed,
    prefill_case_kind,
    prefill_case_expected,
)


def test_all_prefill_routes_against_independent_oracles() raises:
    var ctx = DeviceContext()
    print("prefill test device:", ctx.name(), "api:", ctx.api())
    assert_equal(ctx.api(), "metal")
    for case_id in range(PREFILL_CASE_COUNT):
        var r = prefill_case_query_rows(case_id)
        var t = prefill_case_key_rows(case_id)
        var ql = row_major(r, 14, 64)
        var kl = row_major(t, 2, 64)
        var qb = ctx.enqueue_create_buffer[DType.bfloat16](r * 896)
        var kb = ctx.enqueue_create_buffer[DType.bfloat16](t * 128)
        var vb = ctx.enqueue_create_buffer[DType.bfloat16](t * 128)
        var ob = ctx.enqueue_create_buffer[DType.bfloat16](r * 896)
        var sb = ctx.enqueue_create_buffer[DType.bfloat16](r * 14 * t)
        with qb.map_to_host() as qm:
            with kb.map_to_host() as km:
                with vb.map_to_host() as vm:
                    fill_prefill(
                        TileTensor(qm, ql),
                        TileTensor(km, kl),
                        TileTensor(vm, kl),
                        prefill_case_seed(case_id),
                        prefill_case_kind(case_id),
                    )
        var q = TileTensor(qb, ql)
        var k = TileTensor(kb, kl)
        var v = TileTensor(vb, kl)
        var output = TileTensor(ob, ql)
        var scratch = TileTensor(sb, row_major(r, 14, t))
        var eager = prefill_case_expected(case_id)
        var online = prefill_case_expected(case_id, True)
        for variant in range(PREFILL_VARIANT_COUNT):
            # Normal-mode repeated launches exercise the same shared-memory
            # handoffs with fresh poisoned outputs; the oracle is unchanged.
            var repetitions = (
                get_defined_int["PREFILL_REPEAT", default=1]() if variant
                >= 11 else 1
            )
            for _ in range(repetitions):
                ob.enqueue_fill(123.0)
                assert_equal(
                    enqueue_variant(variant, ctx, q, k, v, output, scratch),
                    variant,
                )
                with ob.map_to_host() as mapped:
                    assert_prefill_close(TileTensor(mapped, ql), eager)
                    if variant >= 2:
                        assert_prefill_close(TileTensor(mapped, ql), online)
        with assert_raises(contains="unknown prefill variant"):
            _ = enqueue_variant(99, ctx, q, k, v, output, scratch)
        print(
            "prefill case",
            case_id,
            r,
            t,
            "all",
            PREFILL_VARIANT_COUNT,
            "routes passed",
        )


def _result(
    rows: Int, tokens: Int, variant: Int, perturb_future: Bool = False
) raises -> List[Float32]:
    var ctx = DeviceContext()
    var ql = row_major(rows, 14, 64)
    var kl = row_major(tokens, 2, 64)
    var qb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
    var kb = ctx.enqueue_create_buffer[DType.bfloat16](tokens * 128)
    var vb = ctx.enqueue_create_buffer[DType.bfloat16](tokens * 128)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
    var sb = ctx.enqueue_create_buffer[DType.bfloat16](rows * 14 * tokens)
    with qb.map_to_host() as qm:
        with kb.map_to_host() as km:
            with vb.map_to_host() as vm:
                fill_prefill(
                    TileTensor(qm, ql),
                    TileTensor(km, kl),
                    TileTensor(vm, kl),
                    53,
                )
                if perturb_future:
                    var k = TileTensor(km, kl)
                    var v = TileTensor(vm, kl)
                    comptime assert k.flat_rank == 3
                    comptime assert v.flat_rank == 3
                    for t in range(17, tokens):
                        for h in range(2):
                            for d in range(64):
                                k[t, h, d] = 64
                                v[t, h, d] = -64
    _ = enqueue_variant(
        variant,
        ctx,
        TileTensor(qb, ql),
        TileTensor(kb, kl),
        TileTensor(vb, kl),
        TileTensor(ob, ql),
        TileTensor(sb, row_major(rows, 14, tokens)),
    )
    var result = List[Float32]()
    with ob.map_to_host() as mapped:
        var output = TileTensor(mapped, ql)
        comptime assert output.flat_rank == 3
        for r in range(rows):
            for h in range(14):
                for d in range(64):
                    result.append(
                        rebind[Float32](output[r, h, d].cast[DType.float32]())
                    )
    if variant == 1:
        with sb.map_to_host() as mapped:
            var probabilities = TileTensor(mapped, row_major(rows, 14, tokens))
            comptime assert probabilities.flat_rank == 3
            for r in range(rows):
                for h in range(14):
                    var total: Float32 = 0
                    for t in range(tokens):
                        var p = rebind[Float32](
                            probabilities[r, h, t].cast[DType.float32]()
                        )
                        if t > tokens - rows + r:
                            assert_equal(p, 0)
                        total += p
                    if abs(total - 1) > 0.015625:
                        raise Error("cooperative softmax probability sum")
    return result^


def test_causality_and_full_versus_suffix_prefill() raises:
    for variant in range(PREFILL_VARIANT_COUNT):
        var full = _result(33, 33, variant)
        var suffix = _result(17, 33, variant)
        var perturbed = _result(33, 33, variant, True)
        for i in range(17 * 896):
            # Future values cannot affect any of the first 17 query positions.
            assert_equal(full[i], perturbed[i])
            var expected = full[16 * 896 + i]
            if abs(suffix[i] - expected) > 0.015625 + 0.015625 * abs(expected):
                raise Error("suffix query position differs from full prefill")


def test_prefill_rejects_unsupported_shapes() raises:
    var ctx = DeviceContext()
    var b = ctx.enqueue_create_buffer[DType.bfloat16](8192 * 896)
    var q = TileTensor(b, row_major(16, 14, 64))
    var k = TileTensor(b, row_major(33, 2, 64))
    var s = TileTensor(b, row_major(16, 14, 33))
    with assert_raises(contains="1 <= R <= T"):
        _ = enqueue_variant(
            7, ctx, TileTensor(b, row_major(34, 14, 64)), k, k, q, s
        )
    with assert_raises(contains="1 <= R <= T"):
        _ = enqueue_variant(
            7, ctx, q, TileTensor(b, row_major(4097, 2, 64)), k, q, s
        )
    with assert_raises(contains="Q[R,14,64]"):
        _ = enqueue_variant(
            7, ctx, TileTensor(b, row_major(16, 13, 64)), k, k, q, s
        )
    with assert_raises(contains="K[T,2,64]"):
        _ = enqueue_variant(
            7, ctx, q, TileTensor(b, row_major(33, 2, 32)), k, q, s
        )
    with assert_raises(contains="value shape"):
        _ = enqueue_variant(
            7, ctx, q, k, TileTensor(b, row_major(32, 2, 64)), q, s
        )
    with assert_raises(contains="output shape"):
        _ = enqueue_variant(
            7, ctx, q, k, k, TileTensor(b, row_major(15, 14, 64)), s
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
