from attention_decode_support import fill_decode, assert_decode_close
from fixtures.attention.decode_data import (
    DECODE_CASE_COUNT,
    decode_case_rows,
    decode_case_seed,
    decode_case_kind,
    decode_case_expected,
)
from fixtures.attention.reference_data import (
    qwen_decode_query,
    qwen_decode_key,
    qwen_decode_value,
    qwen_decode_expected,
)
from test_attention import fill_fixture
from layout import Idx, TileTensor, row_major
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.attention_decode import (
    enqueue_grouped_query_attention_decode_apple_gpu,
)
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_raises


def check_case[
    groups: Int, heads: Int, splits: Int
](
    ctx: DeviceContext,
    rows: Int,
    seed: Int,
    kind: Int,
    expected: List[Float32],
    diagnostic: List[Float32],
    fixture: Bool = False,
) raises:
    if ctx.api() != "metal":
        raise Error("decode tests require Metal")
    var qb = ctx.enqueue_create_buffer[DType.bfloat16](896)
    # Guard one unused row beyond the logical prefix; fill it with NaNs.
    var kb = ctx.enqueue_create_buffer[DType.bfloat16]((rows + 1) * 128)
    var vb = ctx.enqueue_create_buffer[DType.bfloat16]((rows + 1) * 128)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](896)
    var sb = ctx.enqueue_create_buffer[DType.bfloat16](14 * rows)
    var wb = ctx.enqueue_create_buffer[DType.float32](14 * splits * 66)
    var ql = row_major(1, 14, 64)
    var kl = row_major(rows, 2, 64)
    var wl = row_major(14, splits, 66)
    var query = TileTensor(qb, ql)
    var key = TileTensor(kb, kl)
    var value = TileTensor(vb, kl)
    var output = TileTensor(ob, ql)
    var workspace = TileTensor(wb, wl)
    kb.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
    vb.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
    with qb.map_to_host() as qm:
        with kb.map_to_host() as km:
            with vb.map_to_host() as vm:
                var q = TileTensor(qm, ql)
                var k = TileTensor(km, kl)
                var v = TileTensor(vm, kl)
                if fixture:
                    fill_fixture[1, 7, 14, 2, 64](
                        q,
                        k,
                        v,
                        qwen_decode_query(),
                        qwen_decode_key(),
                        qwen_decode_value(),
                    )
                else:
                    fill_decode(q, k, v, seed, kind)
    # Repeated calls overwrite poisoned output and workspace; empty splits
    # must write neutral states rather than retain prior workspace contents.
    for repeat in range(2):
        ob.enqueue_fill(Float32(FloatLiteral.nan).cast[DType.bfloat16]())
        wb.enqueue_fill(Float32(FloatLiteral.nan))
        comptime if groups == 0:
            var scratch = TileTensor(sb, row_major(1, 14, rows))
            enqueue_grouped_query_attention_apple_gpu(
                ctx, query, key, value, scratch, output
            )
        else:
            enqueue_grouped_query_attention_decode_apple_gpu[
                groups, heads, splits
            ](ctx, query, key, value, output, workspace)
        ctx.synchronize()
        with ob.map_to_host() as mapped:
            var result = TileTensor(mapped, ql)
            _ = assert_decode_close(result, expected)
            _ = assert_decode_close(result, diagnostic)


def check_variant[groups: Int, heads: Int, splits: Int]() raises:
    var ctx = DeviceContext()
    for case_id in range(DECODE_CASE_COUNT):
        check_case[groups, heads, splits](
            ctx,
            decode_case_rows(case_id),
            decode_case_seed(case_id),
            decode_case_kind(case_id),
            decode_case_expected(case_id),
            decode_case_expected(case_id, True),
        )
    check_case[groups, heads, splits](
        ctx, 7, 0, 0, qwen_decode_expected(), qwen_decode_expected(), True
    )
    print(
        "decode variant passed:",
        groups,
        heads,
        splits,
        "cases:",
        DECODE_CASE_COUNT + 1,
    )


def test_decode_fused() raises:
    check_variant[1, 1, 1]()


def test_decode_materialized_oracle() raises:
    check_variant[0, 1, 1]()


def test_decode_rejects_unsupported_shapes() raises:
    var ctx = DeviceContext()
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](4097 * 128)
    var wb = ctx.enqueue_create_buffer[DType.float32](14 * 64 * 66)
    var q = TileTensor(storage, row_major(1, 14, 64))
    var k = TileTensor(storage, row_major(7, 2, 64))
    var w = TileTensor(wb, row_major(14, 4, 66))
    for bad_rows in [0, 4097]:
        var bad = TileTensor(storage, row_major(bad_rows, 2, 64))
        with assert_raises(contains="requires K"):
            enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 4](
                ctx, q, bad, k, q, w
            )
    var prefill = TileTensor(storage, row_major(2, 14, 64))
    with assert_raises(contains="requires Q"):
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 4](
            ctx, prefill, k, k, q, w
        )
    var bad_heads = TileTensor(storage, row_major(1, 12, 64))
    with assert_raises(contains="requires Q"):
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 4](
            ctx, bad_heads, k, k, q, w
        )
    var bad_value = TileTensor(storage, row_major(6, 2, 64))
    with assert_raises(contains="value shape"):
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 4](
            ctx, q, k, bad_value, q, w
        )
    with assert_raises(contains="output shape"):
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 4](
            ctx, q, k, k, bad_heads, w
        )
    var bad_workspace = TileTensor(wb, row_major(14, 3, 66))
    with assert_raises(contains="workspace"):
        enqueue_grouped_query_attention_decode_apple_gpu[1, 1, 4](
            ctx, q, k, k, q, bad_workspace
        )


def test_decode_context_parallel() raises:
    check_variant[2, 1, 1]()
    check_variant[8, 1, 1]()
    check_variant[32, 1, 1]()


def test_decode_split_context() raises:
    check_variant[1, 1, 4]()
    check_variant[1, 1, 16]()
    check_variant[1, 1, 64]()


def test_decode_grouped_head_reuse() raises:
    check_variant[1, 2, 64]()
    check_variant[1, 4, 64]()
    check_variant[1, 7, 64]()


def main() raises:
    print(
        "decode test device:",
        DeviceContext().name(),
        "api:",
        DeviceContext().api(),
    )
    TestSuite.discover_tests[__functions_in_module()]().run()
