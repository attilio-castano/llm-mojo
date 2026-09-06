from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.math import isfinite
from std.gpu import lane_id
from std.sys.info import is_apple_gpu
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.attention_prefill import enqueue_grouped_query_attention_prefill_apple_gpu
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
    rows: Int, tokens: Int, variant: Int, perturb_future: Bool = False,
    fp32: Bool = False
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
    if fp32:
        enqueue_grouped_query_attention_prefill_apple_gpu[
            32, 32, MMA=True, SCHEDULE=2, FP32=True
        ](ctx, TileTensor(qb, ql), TileTensor(kb, kl), TileTensor(vb, kl), TileTensor(ob, ql))
    else:
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


def _strict_fp32(got: Float32, want: Float32) raises:
    if (not isfinite(got) or not isfinite(want)
        or abs(got - want) > 0.0078125 * (1 + abs(want))):
        print("FP32 prefill mismatch", got, want)
        raise Error("FP32 prefill exceeded the unchanged strict gate")


def test_fp32_prefill_edges_against_materialized() raises:
    # Structural coverage supplements the pinned upstream operation suite.
    var ctx = DeviceContext()
    print("FP32 prefill edges:", ctx.name(), ctx.api())
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
        var sb = ctx.enqueue_create_buffer[DType.float32](r * 14 * t)
        with qb.map_to_host() as qm:
            with kb.map_to_host() as km:
                with vb.map_to_host() as vm:
                    fill_prefill(TileTensor(qm, ql), TileTensor(km, kl), TileTensor(vm, kl),
                                 prefill_case_seed(case_id), prefill_case_kind(case_id))
        var q = TileTensor(qb, ql)
        var k = TileTensor(kb, kl)
        var v = TileTensor(vb, kl)
        var output = TileTensor(ob, ql)
        enqueue_grouped_query_attention_apple_gpu(
            ctx, q, k, v, TileTensor(sb, row_major(r, 14, t)), output
        )
        var expected = List[Float32]()
        with ob.map_to_host() as mapped:
            var result = TileTensor(mapped, row_major(r * 896))
            comptime assert result.flat_rank == 1
            for i in range(r * 896):
                expected.append(rebind[Float32](result[i].cast[DType.float32]()))
        for _ in range(get_defined_int["PREFILL_REPEAT", default=2]()):
            ob.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
            enqueue_grouped_query_attention_prefill_apple_gpu[
                32, 32, MMA=True, SCHEDULE=2, FP32=True
            ](ctx, q, k, v, output)
            with ob.map_to_host() as mapped:
                var result = TileTensor(mapped, row_major(r * 896))
                comptime assert result.flat_rank == 1
                for i in range(r * 896):
                    _strict_fp32(rebind[Float32](result[i].cast[DType.float32]()), expected[i])
        print("FP32 prefill edge", case_id, "R", r, "T", t, "passed")


def test_fp32_prefill_causality_and_suffix() raises:
    var full = _result(33, 33, 0, fp32=True)
    var suffix = _result(17, 33, 0, fp32=True)
    var perturbed = _result(33, 33, 0, True, True)
    for i in range(17 * 896):
        assert_equal(full[i], perturbed[i])
        _strict_fp32(suffix[i], full[16 * 896 + i])


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


def _fp32_mma_probe[OL: TensorLayout](output: TileTensor[DType.float32, OL, MutAnyOrigin]):
    comptime assert is_apple_gpu()
    comptime assert output.flat_rank == 2
    var lane = Int(lane_id())
    var row = ((lane & 6) >> 1) + ((lane & 16) >> 2)
    var col = ((lane & 1) << 1) + ((lane & 8) >> 1)
    var a = SIMD[DType.float32, 2](0)
    var b = SIMD[DType.float32, 2](0)
    comptime for element in range(2):
        a[element] = Float32(row + 1) * 0.1 + Float32(col + element) * 0.013671875 + 0.00012345
        b[element] = 1 if row == col + element else 0
    var result = SIMD[DType.float32, 2](0)
    var previous = result
    _mma_apple_8x8(result, a, b, previous)
    comptime for element in range(2):
        output[row, col + element] = result[element]


def test_fp32_mma_operands_preserve_non_bf16_values() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print("device:", ctx.name(), "api:", ctx.api())
    var buffer = ctx.enqueue_create_buffer[DType.float32](64)
    buffer.enqueue_fill(Float32(FloatLiteral.nan))
    var out_tensor = TileTensor(buffer, row_major(8, 8))
    comptime kernel = _fp32_mma_probe[type_of(out_tensor.layout)]
    ctx.enqueue_function[kernel](out_tensor, grid_dim=1, block_dim=32)
    var worst: Float32 = 0
    with buffer.map_to_host() as mapped:
        var result = TileTensor(mapped, row_major(8, 8))
        comptime assert result.flat_rank == 2
        for row in range(8):
            for col in range(8):
                var expected = Float32(row + 1) * 0.1 + Float32(col) * 0.013671875 + 0.00012345
                var error = abs(rebind[Float32](result[row, col]) - expected)
                assert_true(error < 0.000001)
                worst = max(worst, error)
    print("FP32 MMA identity product passed; maximum absolute error:", worst)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
