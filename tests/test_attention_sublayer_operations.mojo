"""Every operation consumes upstream tensors; errors cannot be inherited.

The composition suite separately feeds original X through the whole block.
The FP32 route has a strict accuracy gate. BF16 eager comparisons remain
diagnostic unless SUBLAYER_BF16_COMPATIBILITY=1 explicitly requests their
historical compatibility gate. Standalone operation contracts are unchanged.
"""
from llm_mojo.attention_sublayer import AttentionWeights, AttentionWorkspace
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.linear import (
    enqueue_linear_apple_gpu, enqueue_linear_prefill_mma_8x16_apple_gpu,
)
from llm_mojo.rope import enqueue_rope_apple_gpu
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.attention_decode import (
    enqueue_grouped_query_attention_decode_apple_gpu,
)
from llm_mojo.attention_prefill import (
    enqueue_grouped_query_attention_prefill_apple_gpu,
)
from llm_mojo.residual import enqueue_residual_apple_gpu
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.testing import TestSuite, assert_equal
from std.sys import get_defined_int
from attention_sublayer_support import (
    load_sublayer_fixture,
    assert_sublayer_fixture,
)


def _fp32_decode_prefixes(
    ctx: DeviceContext, mut work: AttentionWorkspace, case_id: Int, t: Int
) raises -> Int:
    """Compare selected causal rows with their frozen upstream FP32 outputs."""
    var prefixes = List[Int]()
    for length in [1, 7, 16, 31, 32, 33, 63, 64, 65, 257, 351, 668, 1024, 4095, 4096]:
        if length <= t:
            prefixes.append(length)
    if prefixes[len(prefixes) - 1] != t:
        prefixes.append(t)
    # A diagnostic compile can prove the strict gate catches the older policy.
    comptime fp32 = get_defined_int["FP32_DECODE_LEGACY_PROBE", default=0]() == 0
    var failures = 0
    for length in prefixes:
        var q = TileTensor(work.query.unsafe_ptr().unsafe_offset((length - 1) * 896),
                           row_major(1, 14, 64))
        var key = TileTensor(work.rotated_key, row_major(length, 2, 64))
        var value = TileTensor(work.raw_value, row_major(length, 2, 64))
        var output = TileTensor(work.attention, row_major(1, 14, 64))
        for route in [4, 5]:
            work.attention.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
            work.split.enqueue_fill(Float32(FloatLiteral.nan))
            if route == 4:
                enqueue_grouped_query_attention_decode_apple_gpu[32, 1, 1, fp32_scores=fp32](
                    ctx, q, key, value, output, TileTensor(work.split, row_major(14, 1, 66))
                )
            else:
                enqueue_grouped_query_attention_decode_apple_gpu[1, 4, 64, fp32_scores=fp32](
                    ctx, q, key, value, output, TileTensor(work.split, row_major(14, 64, 66))
                )
            print("FP32 decode prefix case", case_id, "route", route,
                  "T", length, "FP32 scores", fp32)
            try:
                assert_sublayer_fixture(work.attention, case_id, "attention",
                                        length - 1, 1, 896, 0.0078125, True, "fp32")
            except:
                failures += 1
    return failures


def _operations(case_id: Int, nq: Int, nk: Int, d: Int, t: Int, precision: Bool = False) raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print("local operation gates:", ctx.name(), ctx.api(), "case_id", case_id)
    var h = nq * d
    var k = nk * d
    var weights = AttentionWeights(ctx, nq, nk, d)
    var work = AttentionWorkspace(ctx, t, t, nq, nk, d, True, precision)
    var input = ctx.enqueue_create_buffer[DType.bfloat16](t * h)
    load_sublayer_fixture(input, case_id, "input")
    load_sublayer_fixture(weights.qkv, case_id, "weight")
    load_sublayer_fixture(weights.bias, case_id, "bias")
    load_sublayer_fixture(weights.norm, case_id, "norm_weight")
    load_sublayer_fixture(weights.output, case_id, "output_weight")
    load_sublayer_fixture(work.cosine, case_id, "cosine", True)
    load_sublayer_fixture(work.sine, case_id, "sine", True)
    var x = TileTensor(input, row_major(t, h))
    var normal = TileTensor(work.normalized, row_major(t, h))
    enqueue_rms_norm_apple_gpu(
        ctx, x, TileTensor(weights.norm, row_major(h)), normal
    )
    assert_sublayer_fixture(
        work.normalized, case_id, "normalized", 0, t, h, 0.0078125
    )

    # Replace the preceding result with exactly what upstream consumed.
    load_sublayer_fixture(work.normalized, case_id, "normalized", True)
    enqueue_linear_apple_gpu(
        ctx,
        normal,
        TileTensor(weights.qkv, row_major(h, h)),
        TileTensor(weights.bias, row_major(h)),
        TileTensor(work.raw_query, row_major(t, h)),
    )
    enqueue_linear_apple_gpu(
        ctx,
        normal,
        TileTensor(
            weights.qkv.unsafe_ptr().unsafe_offset(h * h), row_major(k, h)
        ),
        TileTensor(weights.bias.unsafe_ptr().unsafe_offset(h), row_major(k)),
        TileTensor(work.raw_key, row_major(t, k)),
    )
    enqueue_linear_apple_gpu(
        ctx,
        normal,
        TileTensor(
            weights.qkv.unsafe_ptr().unsafe_offset((h + k) * h), row_major(k, h)
        ),
        TileTensor(
            weights.bias.unsafe_ptr().unsafe_offset(h + k), row_major(k)
        ),
        TileTensor(work.raw_value, row_major(t, k)),
    )
    assert_sublayer_fixture(
        work.raw_query, case_id, "raw_query", 0, t, h, 0.0078125
    )
    assert_sublayer_fixture(
        work.raw_key, case_id, "raw_key", 0, t, k, 0.0078125
    )
    assert_sublayer_fixture(
        work.raw_value, case_id, "raw_value", 0, t, k, 0.0078125
    )

    load_sublayer_fixture(work.raw_query, case_id, "raw_query", True)
    load_sublayer_fixture(work.raw_key, case_id, "raw_key", True)
    var cosine = TileTensor(work.cosine, row_major(t, d))
    var sine = TileTensor(work.sine, row_major(t, d))
    enqueue_rope_apple_gpu(
        ctx,
        TileTensor(work.raw_query, row_major(t, nq, d)),
        cosine,
        sine,
        TileTensor(work.query, row_major(t, nq, d)),
        0,
    )
    enqueue_rope_apple_gpu(
        ctx,
        TileTensor(work.raw_key, row_major(t, nk, d)),
        cosine,
        sine,
        TileTensor(work.rotated_key, row_major(t, nk, d)),
        0,
    )
    # With identical BF16 tables and inputs this operation is elementwise exact.
    assert_sublayer_fixture(work.query, case_id, "query", 0, t, h, 0)
    assert_sublayer_fixture(
        work.rotated_key, case_id, "rotated_key", 0, t, k, 0
    )

    load_sublayer_fixture(work.query, case_id, "query", True)
    load_sublayer_fixture(work.rotated_key, case_id, "rotated_key", True)
    load_sublayer_fixture(work.raw_value, case_id, "raw_value", True)
    var keys = TileTensor(work.rotated_key, row_major(t, nk, d))
    var values = TileTensor(work.raw_value, row_major(t, nk, d))
    var query_counts = List[Int]()
    query_counts.append(t)
    var gqa_failures = 0
    if t > 1:
        query_counts.append(1)
    if t > 17:
        query_counts.append(17)
    for r in query_counts:
        var q = TileTensor(
            work.query.unsafe_ptr().unsafe_offset((t - r) * h),
            row_major(r, nq, d),
        )
        var out = TileTensor(work.attention, row_major(r, nq, d))
        for route in range(4 if precision else (3 if nq == 14 else 1)):
            if nq != 14 and route != 0 and route != 3:
                continue
            work.attention.enqueue_fill(123)
            if route == 0:
                enqueue_grouped_query_attention_apple_gpu(
                    ctx,
                    q,
                    keys,
                    values,
                    TileTensor(work.scratch, row_major(r, nq, t)),
                    out,
                )
            elif route == 3:
                enqueue_grouped_query_attention_apple_gpu(
                    ctx, q, keys, values,
                    TileTensor(work.fp32_scratch, row_major(r, nq, t)), out,
                )
            elif r == 1:
                if route == 1:
                    enqueue_grouped_query_attention_decode_apple_gpu[32, 1, 1](
                        ctx,
                        q,
                        keys,
                        values,
                        out,
                        TileTensor(work.split, row_major(14, 1, 66)),
                    )
                else:
                    enqueue_grouped_query_attention_decode_apple_gpu[1, 4, 64](
                        ctx,
                        q,
                        keys,
                        values,
                        out,
                        TileTensor(work.split, row_major(14, 64, 66)),
                    )
            elif route == 1:
                enqueue_grouped_query_attention_prefill_apple_gpu[
                    32, 32, MMA=True
                ](ctx, q, keys, values, out)
            else:
                enqueue_grouped_query_attention_prefill_apple_gpu[
                    32, 32, MMA=True, SCHEDULE=2
                ](ctx, q, keys, values, out)
            print("local GQA route", route, "R", r, "T", t)
            if precision:
                print("operation reference fp32")
                try:
                    assert_sublayer_fixture(
                        work.attention, case_id, "attention", t - r, r, h,
                        0.0078125, route == 3, "fp32"
                    )
                except:
                    gqa_failures += 1
                print("operation reference upstream")
            try:
                assert_sublayer_fixture(
                    work.attention, case_id, "attention", t - r, r, h, 0.03125,
                    not precision and Bool(get_defined_int["SUBLAYER_BF16_COMPATIBILITY", default=0]())
                )
            except:
                gqa_failures += 1

    if precision and nq == 14 and nk == 2 and d == 64:
        gqa_failures += _fp32_decode_prefixes(ctx, work, case_id, t)

    var reference = "fp32" if precision else "upstream"
    load_sublayer_fixture(work.attention, case_id, "attention", True, reference)
    var projected = TileTensor(work.projected, row_major(t, h))
    enqueue_linear_apple_gpu(
        ctx,
        TileTensor(work.attention, row_major(t, h)),
        TileTensor(weights.output, row_major(h, h)),
        projected,
    )
    assert_sublayer_fixture(
        work.projected, case_id, "projected", 0, t, h, 0.03125, True, reference
    )
    if precision:
        # The candidate receives precisely the BF16 attention tensor upstream
        # consumed, so a preceding GQA error cannot contaminate the Wo gate.
        work.projected.enqueue_fill(123)
        enqueue_linear_prefill_mma_8x16_apple_gpu(
            ctx, TileTensor(work.attention, row_major(t, h)),
            TileTensor(weights.output, row_major(h, h)), projected,
        )
        print("local Wo mapping MMA 8x16")
        assert_sublayer_fixture(
            work.projected, case_id, "projected", 0, t, h, 0.03125, True, reference
        )
    load_sublayer_fixture(work.projected, case_id, "projected", True, reference)
    enqueue_residual_apple_gpu(
        ctx, x, projected, TileTensor(work.output, row_major(t, h))
    )
    assert_sublayer_fixture(work.output, case_id, "output", 0, t, h, 0, True, reference)
    if gqa_failures:
        raise Error(
            "upstream GQA compatibility gates failed; all routes reported"
        )


def test_operations_and_bf16_compatibility_diagnostics() raises:
    print("BF16 eager compatibility: strict", get_defined_int["SUBLAYER_BF16_COMPATIBILITY", default=0](),
          "; primary accuracy gate is FP32")
    if get_defined_int["SUBLAYER_HOLDOUT", default=0]():
        _operations(13, 14, 2, 64, 4096)
        _operations(14, 14, 2, 64, 4096)
        return
    if get_defined_int["SUBLAYER_CALIBRATION", default=0]():
        var failed_cases = 0
        for case_id in range(9, 13):
            try:
                _operations(case_id, 14, 2, 64, 4096)
            except:
                failed_cases += 1
        if failed_cases:
            raise Error(
                "calibration found failing cases; current gates unchanged"
            )
        return
    _operations(0, 2, 1, 4, 7)
    _operations(1, 4, 2, 4, 9)
    var lengths = [
        1,
        7,
        33,
        65,
        257,
        4096,
        4096,
        4096,
        4096,
        4096,
        4096,
        4096,
        4096,
    ]
    for i in range(len(lengths)):
        _operations(i + 2, 14, 2, 64, lengths[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
