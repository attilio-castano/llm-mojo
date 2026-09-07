from llm_mojo.attention_sublayer import (
    AttentionWeights,
    AttentionCache,
    AttentionWorkspace,
    enqueue_attention_sublayer,
    enqueue_attention_sublayer_integrated,
    _unpack_qkv,
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
    wo_mma: Bool = False,
    qkv_mapping: Int = 0, integrated: Bool = False, gqa_mapping: Int = 0,
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
        "Wo MMA",
        wo_mma,
        "QKV mapping", qkv_mapping, "integrated", integrated, "GQA mapping", gqa_mapping,
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
    var work = AttentionWorkspace(ctx, t, capacity, nq, nk, d, route == 0,
                                  route >= 3 and route < 6,
                                  8 if gqa_mapping == 4 else (4 if gqa_mapping == 3 else 1))
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
        work.packed.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
        work.raw_query.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
        work.raw_key.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
        work.raw_value.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
        work.prefill_partial.enqueue_fill(Float32(FloatLiteral.nan))
        var prefix_k = snapshot(cache.key, p * k)
        var prefix_v = snapshot(cache.value, p * k)
        var view = TileTensor(
            input.unsafe_ptr().unsafe_offset(p * h), row_major(r, h)
        )
        if integrated:
            assert_equal(enqueue_attention_sublayer_integrated(
                ctx, weights, cache, work, view, gqa_mapping
            ), 4 if r == 1 else 6 + gqa_mapping)
        elif route == 3:
            # All FP32 accuracy cases also exercise the public default route.
            assert_equal(enqueue_attention_sublayer(
                ctx, weights, cache, work, view, wo_mma=wo_mma, qkv_mapping=qkv_mapping
            ), 3)
        else:
            assert_equal(
                enqueue_attention_sublayer(ctx, weights, cache, work, view, route, wo_mma, qkv_mapping),
                (4 if r == 1 else 6) if route == 6 else (3 if route >= 4 and r != 1 else route),
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
    with assert_raises(contains="QKV projection mapping"):
        _ = enqueue_attention_sublayer(
            ctx, weights, cache, work, TileTensor(input, row_major(1, h)), qkv_mapping=3
        )
    assert_equal(cache.length, t)
    cache.reset(ctx)
    assert_equal(cache.length, 0)
    if integrated:
        _ = enqueue_attention_sublayer_integrated(
            ctx, weights, cache, work, TileTensor(input, row_major(1, h))
        )
    else:
        _ = enqueue_attention_sublayer(
            ctx, weights, cache, work, TileTensor(input, row_major(1, h)), route, wo_mma, qkv_mapping
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
    var input = ctx.enqueue_create_buffer[DType.bfloat16](65 * 896)
    load_sublayer_fixture(weights.qkv, 5, "weight")
    load_sublayer_fixture(weights.bias, 5, "bias")
    load_sublayer_fixture(weights.norm, 5, "norm_weight")
    load_sublayer_fixture(weights.output, 5, "output_weight")
    load_sublayer_fixture(input, 5, "input")
    for implementation in range(14):
        var route = implementation - 3 if implementation >= 10 else (6 if implementation >= 8 else (3 if implementation == 4 else (implementation - 1 if implementation >= 5 else implementation)))
        var wo_mma = implementation == 4 or implementation == 7
        # Optimized FP32 decode must also work without probability storage.
        var work = AttentionWorkspace(ctx, 65, 65, materialized=route == 0,
                                      fp32_materialized=route == 3,
                                      prefill_splits=8 if implementation == 13 else (4 if implementation == 12 else 1))
        load_sublayer_fixture(work.cosine, 5, "cosine", True)
        load_sublayer_fixture(work.sine, 5, "sine", True)
        for _ in range(get_defined_int["SUBLAYER_REPEAT", default=3]()):
            cache.reset(ctx)
            work.output.enqueue_fill(123)
            work.prefill_partial.enqueue_fill(Float32(FloatLiteral.nan))
            var p = 0
            while p < 65:
                # Route 6 alternates tiled prefill and final decode, without
                # materialized scratch or a synchronization between enqueues.
                var rows = (33 if p == 0 else (31 if p == 33 else 1)) if route == 6 else 1
                if implementation >= 8:
                    rows = 15 if p == 0 else (16 if p == 15 or p == 48 else (17 if p == 31 else 1))
                var view = TileTensor(input.unsafe_ptr().unsafe_offset(p * 896), row_major(rows, 896))
                var launched: Int
                if implementation >= 9:
                    launched = enqueue_attention_sublayer_integrated(ctx, weights, cache, work, view, implementation - 9)
                else:
                    launched = enqueue_attention_sublayer(
                        ctx, weights, cache, work, view, route,
                        rows >= 16 if implementation == 8 else wo_mma,
                    )
                assert_equal(launched, 4 if route >= 6 and rows == 1 else route)
                p += rows
            ctx.synchronize()
            assert_equal(cache.length, 65)
            assert_sublayer_fixture(
                work.output, 5, "output", 64, 1, 896, 0.03125, True,
                "fp32" if route >= 3 else "upstream",
            )
        if route == 4 or route == 5:
            cache.reset(ctx)
            with assert_raises(contains="FP32 probability scratch"):
                _ = enqueue_attention_sublayer(
                    ctx, weights, cache, work, TileTensor(input, row_major(3, 896)), route
                )
            assert_equal(cache.length, 0)


def test_partitioned_prefill_rejects_missing_storage_before_enqueue() raises:
    var ctx = DeviceContext()
    var weights = AttentionWeights(ctx)
    var cache = AttentionCache(ctx, 65)
    var work = AttentionWorkspace(ctx, 17, 65, fp32_materialized=False)
    var input = ctx.enqueue_create_buffer[DType.bfloat16](17 * 896)
    cache.key.enqueue_fill(123)
    work.output.enqueue_fill(123)
    var before = snapshot(cache.key, 65 * 128)
    var output_before = snapshot(work.output, 17 * 896)
    for mapping in [3, 4]:
        with assert_raises(contains="caller-allocated partial storage"):
            _ = enqueue_attention_sublayer_integrated(
                ctx, weights, cache, work, TileTensor(input, row_major(17, 896)), mapping,
            )
        assert_equal(cache.length, 0)
    with assert_raises(contains="unknown integrated GQA mapping"):
        _ = enqueue_attention_sublayer_integrated(
            ctx, weights, cache, work, TileTensor(input, row_major(17, 896)), 5,
        )
    var after = snapshot(cache.key, 65 * 128)
    for i in range(len(before)):
        assert_equal(before[i], after[i])
    var untouched = snapshot(work.output, 17 * 896)
    for i in range(len(untouched)):
        assert_equal(untouched[i], output_before[i])


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


def test_packed_qkv_handoff_preserves_bits_rows_and_boundaries() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    # 3 rows x (Q=6,K=4,V=4): ragged launch and MMA-independent layout oracle.
    var packed = ctx.enqueue_create_buffer[DType.bfloat16](42)
    var q = ctx.enqueue_create_buffer[DType.bfloat16](19)
    var k = ctx.enqueue_create_buffer[DType.bfloat16](13)
    var v = ctx.enqueue_create_buffer[DType.bfloat16](13)
    q.enqueue_fill(123)
    k.enqueue_fill(123)
    v.enqueue_fill(123)
    var patterns: List[UInt16] = [0x8000, 0, 0x3f80, 0xbf80, 1, 0x7f7f, 0x0080]
    with packed.map_to_host() as mapped:
        var a = TileTensor(mapped, row_major(42))
        for i in range(42):
            a.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i] = patterns[(i + i // 14) % 7]
    var pt = TileTensor(packed, row_major(3, 14))
    var qt = TileTensor(q, row_major(3, 6))
    var kt = TileTensor(k, row_major(3, 4))
    var vt = TileTensor(v, row_major(3, 4))
    ctx.enqueue_function[_unpack_qkv[type_of(pt.layout), type_of(qt.layout), type_of(kt.layout)]](
        pt, qt, kt, vt, Int32(3), Int32(6), Int32(4), grid_dim=1, block_dim=128,
    )
    var qb = snapshot(q, 19)
    var kb = snapshot(k, 13)
    var vb = snapshot(v, 13)
    var before = snapshot(packed, 42)
    for row in range(3):
        for col in range(6):
            assert_equal(qb[row * 6 + col], before[row * 14 + col])
        for col in range(4):
            assert_equal(kb[row * 4 + col], before[row * 14 + 6 + col])
            assert_equal(vb[row * 4 + col], before[row * 14 + 10 + col])
    assert_equal(qb[18], UInt16(0x42f6))  # untouched sentinel 123
    assert_equal(kb[12], UInt16(0x42f6))
    assert_equal(vb[12], UInt16(0x42f6))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
