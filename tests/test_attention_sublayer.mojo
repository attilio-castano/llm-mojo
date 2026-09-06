from llm_mojo.attention_sublayer import (
    AttentionWeights,
    AttentionCache,
    AttentionWorkspace,
    enqueue_attention_sublayer,
)
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.testing import TestSuite, assert_equal, assert_raises
from std.sys import get_defined_int
from attention_sublayer_support import (
    load_sublayer_fixture,
    assert_sublayer_fixture,
    assert_cache_append,
    snapshot,
)


def _case(
    case_id: Int, nq: Int, nk: Int, d: Int, t: Int, route: Int, chunks: Bool,
    reference: String = "upstream", require_close: Bool = True,
) raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print(
        "sublayer device:",
        ctx.name(),
        "api:",
        ctx.api(),
        "case",
        case_id,
        "route",
        route,
        "chunks",
        chunks,
    )
    var h = nq * d
    var k = nk * d
    var capacity = t + 1 if t < 4096 else 4096
    var weights = AttentionWeights(ctx, nq, nk, d)
    load_sublayer_fixture(weights.qkv, case_id, "weight")
    load_sublayer_fixture(weights.bias, case_id, "bias")
    load_sublayer_fixture(weights.norm, case_id, "norm_weight")
    load_sublayer_fixture(weights.output, case_id, "output_weight")
    var cache = AttentionCache(ctx, capacity, nk, d)
    cache.key.enqueue_fill(123)
    cache.value.enqueue_fill(123)
    var work = AttentionWorkspace(ctx, t, capacity, nq, nk, d, route == 0, route == 3)
    print("composition reference", reference, "strict", require_close)
    load_sublayer_fixture(work.cosine, case_id, "cosine", True)
    load_sublayer_fixture(work.sine, case_id, "sine", True)
    var input = ctx.enqueue_create_buffer[DType.bfloat16](t * h)
    load_sublayer_fixture(input, case_id, "input")
    var p = 0
    var step = 0
    var numerical_failures = 0
    while p < t:
        var r = t - p
        if chunks:
            # A long prefix followed by a ragged suffix, ending on one token.
            # Small fixtures include multiple consecutive decode calls.
            if t > 65:
                r = t - 18 if p == 0 else (17 if p == t - 18 else 1)
            else:
                r = 1 if step % 3 != 1 else 3
                if r > t - p:
                    r = t - p
        work.output.enqueue_fill(123)
        work.projected.enqueue_fill(123)
        work.attention.enqueue_fill(123)
        var prefix_k = snapshot(cache.key, p * k)
        var prefix_v = snapshot(cache.value, p * k)
        var view = TileTensor(
            input.unsafe_ptr().unsafe_offset(p * h), row_major(r, h)
        )
        if route == 3:
            # All FP32 accuracy cases also exercise the public default route.
            assert_equal(enqueue_attention_sublayer(ctx, weights, cache, work, view), 3)
        else:
            assert_equal(
                enqueue_attention_sublayer(ctx, weights, cache, work, view, route), route
            )
        assert_equal(cache.length, p + r)
        assert_sublayer_fixture(
            work.normalized, case_id, "normalized", p, r, h, 0.0078125, False
        )
        assert_sublayer_fixture(
            work.raw_query, case_id, "raw_query", p, r, h, 0.0078125, False
        )
        assert_sublayer_fixture(
            work.raw_key, case_id, "raw_key", p, r, k, 0.0078125, False
        )
        assert_sublayer_fixture(
            work.raw_value, case_id, "raw_value", p, r, k, 0.0078125, False
        )
        assert_sublayer_fixture(
            work.query, case_id, "query", p, r, h, 0.0078125, False
        )
        assert_sublayer_fixture(
            work.rotated_key, case_id, "rotated_key", p, r, k, 0.0078125, False
        )
        assert_sublayer_fixture(
            work.attention, case_id, "attention", p, r, h, 0.03125, False, reference
        )
        # Report both gates and finish cache checks even if one numeric gate
        # fails. The case still fails after every boundary has been checked.
        try:
            assert_sublayer_fixture(
                work.projected, case_id, "projected", p, r, h, 0.03125, require_close, reference
            )
        except:
            numerical_failures += 1
        try:
            assert_sublayer_fixture(
                work.output, case_id, "output", p, r, h, 0.03125, require_close, reference
            )
        except:
            numerical_failures += 1
        if reference == "fp32":
            print("composition comparison upstream")
            assert_sublayer_fixture(work.attention, case_id, "attention", p, r, h, 0.03125, False)
            assert_sublayer_fixture(work.projected, case_id, "projected", p, r, h, 0.03125, False)
            assert_sublayer_fixture(work.output, case_id, "output", p, r, h, 0.03125, False)
            print("composition comparison fp32")
        var after_k = snapshot(cache.key, p * k)
        var after_v = snapshot(cache.value, p * k)
        for i in range(p * k):
            assert_equal(after_k[i], prefix_k[i])
            assert_equal(after_v[i], prefix_v[i])
        assert_cache_append(cache.key, work.rotated_key, p, r, k, capacity)
        assert_cache_append(cache.value, work.raw_value, p, r, k, capacity)
        p += r
        step += 1
    with assert_raises(contains="overflow"):
        # Deliberately exceed the remaining capacity, before any enqueue.
        _ = enqueue_attention_sublayer(
            ctx, weights, cache, work, TileTensor(input, row_major(2, h)), route
        )
    assert_equal(cache.length, t)
    with assert_raises(contains="unknown"):
        _ = enqueue_attention_sublayer(
            ctx, weights, cache, work, TileTensor(input, row_major(1, h)), 99
        )
    cache.reset(ctx)
    assert_equal(cache.length, 0)
    _ = enqueue_attention_sublayer(
        ctx, weights, cache, work, TileTensor(input, row_major(1, h)), route
    )
    assert_sublayer_fixture(work.output, case_id, "output", 0, 1, h, 0.03125, require_close, reference)
    if numerical_failures:
        raise Error("sublayer projected/final compatibility gates failed")


def test_tiny_materialized_sublayer_and_cache() raises:
    _case(0, 2, 1, 4, 7, 0, False)
    _case(0, 2, 1, 4, 7, 0, True)
    _case(1, 4, 2, 4, 9, 0, False)
    _case(1, 4, 2, 4, 9, 0, True)


def test_bf16_sublayer_compatibility_diagnostics_and_exact_cache() raises:
    var strict = Bool(get_defined_int["SUBLAYER_BF16_COMPATIBILITY", default=0]())
    print("BF16 eager compatibility: strict", strict, "; primary accuracy gate is FP32")
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
    var start = 11 if get_defined_int["SUBLAYER_HOLDOUT", default=0]() else 0
    var failures = 0
    for i in range(start, len(lengths)):
        for route in range(3):
            try:
                _case(i + 2, 14, 2, 64, lengths[i], route, False, "upstream", strict)
            except:
                failures += 1
            if lengths[i] > 1:
                try:
                    _case(i + 2, 14, 2, 64, lengths[i], route, True, "upstream", strict)
                except:
                    failures += 1
    if failures:
        raise Error("sublayer composition gates failed; all cases reported")


def test_repeated_asynchronous_use() raises:
    var ctx = DeviceContext()
    var weights = AttentionWeights(ctx)
    var cache = AttentionCache(ctx, 65)
    var work = AttentionWorkspace(ctx, 65, 65, materialized=True)
    load_sublayer_fixture(work.cosine, 5, "cosine", True)
    load_sublayer_fixture(work.sine, 5, "sine", True)
    var input = ctx.enqueue_create_buffer[DType.bfloat16](65 * 896)
    load_sublayer_fixture(weights.qkv, 5, "weight")
    load_sublayer_fixture(weights.bias, 5, "bias")
    load_sublayer_fixture(weights.norm, 5, "norm_weight")
    load_sublayer_fixture(weights.output, 5, "output_weight")
    load_sublayer_fixture(input, 5, "input")
    for route in range(4):
        for _ in range(get_defined_int["SUBLAYER_REPEAT", default=3]()):
            cache.reset(ctx)
            work.output.enqueue_fill(123)
            for p in range(65):
                _ = enqueue_attention_sublayer(
                    ctx,
                    weights,
                    cache,
                    work,
                    TileTensor(
                        input.unsafe_ptr().unsafe_offset(p * 896),
                        row_major(1, 896),
                    ),
                    route,
                )
            ctx.synchronize()
            assert_equal(cache.length, 65)
            assert_sublayer_fixture(
                work.output, 5, "output", 64, 1, 896, 0.03125, True,
                "fp32" if route == 3 else "upstream",
            )


def test_cache_gate_checks_bits_including_signed_zero() raises:
    var ctx = DeviceContext()
    var source = ctx.enqueue_create_buffer[DType.bfloat16](1)
    var cache = ctx.enqueue_create_buffer[DType.bfloat16](1)
    cache.enqueue_fill(0)
    with source.map_to_host() as sm:
        var s = TileTensor(sm, row_major(1))
        comptime assert s.flat_rank == 1
        var bits = s.ptr.bitcast[UInt16]()
        bits[unsafe_offset=0] = 0x8000
    var bits = snapshot(source, 1)
    assert_equal(bits[0], UInt16(0x8000))
    with assert_raises(contains="changed a source value"):
        assert_cache_append(cache, source, 0, 1, 1, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
