from layout import TileTensor, row_major
from max.gpu.host import DeviceContext, DeviceBuffer
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.mlp import (
    MLPWeights,
    MLPWorkspace,
    enqueue_mlp_apple_gpu,
    enqueue_mlp_stage_apple_gpu,
)
from mlp_support import (
    mlp_support,
    load_mlp,
    check_mlp,
    poison_work,
    check_work,
)


def run_case(case_id: String, rows: Int, h: Int, i: Int, mapping: Int) raises:
    var support = mlp_support()
    support.set_mapping(mapping)
    support.verify_case(case_id)
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print("MLP runtime:", ctx.name(), ctx.api(), case_id, "mapping", mapping)
    var weights = MLPWeights(ctx, h, i)
    var cap = rows + 1 if rows < 4096 else rows
    var work = MLPWorkspace(ctx, cap, h, i)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](rows * h)
    load_mlp(xb, case_id + "/X.npy", 0, rows * h)
    load_mlp(weights.norm, case_id + "/norm.npy", 0, h)
    load_mlp(weights.gate, case_id + "/gate.npy", 0, h * i)
    load_mlp(weights.up, case_id + "/up.npy", 0, h * i)
    load_mlp(weights.down, case_id + "/down.npy", 0, h * i)
    var x = TileTensor(xb, row_major(rows, h))
    poison_work(work, rows)
    enqueue_mlp_stage_apple_gpu(ctx, weights, work, x, 0, mapping)
    check_mlp(work.normalized, case_id, "N", "local", 0, rows, h, cap * h)
    load_mlp(work.normalized, case_id + "/N.npy", 0, rows * h)
    enqueue_mlp_stage_apple_gpu(ctx, weights, work, x, 1, mapping)
    enqueue_mlp_stage_apple_gpu(ctx, weights, work, x, 2, mapping)
    check_mlp(work.gate, case_id, "G", "local", 0, rows, i, cap * i)
    check_mlp(work.up, case_id, "U", "local", 0, rows, i, cap * i)
    load_mlp(work.gate, case_id + "/G.npy", 0, rows * i)
    enqueue_mlp_stage_apple_gpu(ctx, weights, work, x, 3, mapping)
    check_mlp(work.activated, case_id, "A", "local", 0, rows, i, cap * i)
    load_mlp(work.activated, case_id + "/A.npy", 0, rows * i)
    load_mlp(work.up, case_id + "/U.npy", 0, rows * i)
    enqueue_mlp_stage_apple_gpu(ctx, weights, work, x, 4, mapping)
    check_mlp(work.gated, case_id, "S", "local", 0, rows, i, cap * i)
    load_mlp(work.gated, case_id + "/S.npy", 0, rows * i)
    enqueue_mlp_stage_apple_gpu(ctx, weights, work, x, 5, mapping)
    check_mlp(work.down, case_id, "D", "local", 0, rows, h, cap * h)
    load_mlp(work.down, case_id + "/D.npy", 0, rows * h)
    enqueue_mlp_stage_apple_gpu(ctx, weights, work, x, 6, mapping)
    check_mlp(work.output, case_id, "Y", "local", 0, rows, h, cap * h)
    poison_work(work, rows)
    enqueue_mlp_apple_gpu(ctx, weights, work, x, mapping)
    check_work(work, case_id, "full", 0, rows)
    var chunks = List[Int]()
    if rows <= 17:
        for _ in range(rows):
            chunks.append(1)
    else:
        chunks = [rows - 18, 17, 1]
    var start = 0
    for chunk in chunks:
        poison_work(work, chunk)
        enqueue_mlp_apple_gpu(
            ctx,
            weights,
            work,
            TileTensor(
                xb.unsafe_ptr().unsafe_offset(start * h), row_major(chunk, h)
            ),
            mapping,
        )
        check_work(work, case_id, "chunk", start, chunk)
        start += chunk
    with xb.map_to_host() as mapped:
        support.unchanged(
            Int(mapped.unsafe_ptr()), case_id + "/X.npy", rows * h
        )


def test_mlp_development() raises:
    var support = mlp_support()
    var cases = support.case_specifications()
    var variants = support.projection_variants()
    for v in range(Int(py=variants.__len__())):
        for j in range(Int(py=cases.__len__())):
            var case_id = cases[j]
            run_case(
                String(py=case_id[0]),
                Int(py=case_id[1]),
                Int(py=case_id[2]),
                Int(py=case_id[3]),
                Int(py=variants[v]),
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def run_invalid_calls_and_async_reuse(mapping: Int) raises:
    var case_id = "h896_i4864_r17_s1601"
    var support = mlp_support()
    support.set_mapping(mapping)
    support.verify_case(case_id)
    var ctx = DeviceContext()
    var weights = MLPWeights(ctx)
    var work = MLPWorkspace(ctx, 17)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](17 * 896)
    load_mlp(xb, case_id + "/X.npy", 0, 17 * 896)
    load_mlp(weights.norm, case_id + "/norm.npy", 0, 896)
    load_mlp(weights.gate, case_id + "/gate.npy", 0, 896 * 4864)
    load_mlp(weights.up, case_id + "/up.npy", 0, 896 * 4864)
    load_mlp(weights.down, case_id + "/down.npy", 0, 896 * 4864)
    poison_work(work, 0)
    for bad_rows in [0, 18]:
        with assert_raises():
            enqueue_mlp_apple_gpu(
                ctx, weights, work, TileTensor(xb, row_major(bad_rows, 896))
            )
    with assert_raises():
        enqueue_mlp_apple_gpu(
            ctx, weights, work, TileTensor(xb, row_major(1, 895))
        )
    with assert_raises():
        enqueue_mlp_stage_apple_gpu(
            ctx, weights, work, TileTensor(xb, row_major(1, 896)), 7
        )
    for bad_mapping in [-1, 8]:
        with assert_raises():
            enqueue_mlp_apple_gpu(
                ctx, weights, work, TileTensor(xb, row_major(1, 896)), bad_mapping
            )
    work.hidden = 895
    with assert_raises():
        enqueue_mlp_apple_gpu(
            ctx, weights, work, TileTensor(xb, row_major(1, 896))
        )
    work.hidden = 896
    with work.output.map_to_host() as mapped:
        support.assert_uniform_bits(Int(mapped.unsafe_ptr()), 17 * 896, 0x42F6)
    with work.normalized.map_to_host() as mapped:
        support.assert_uniform_bits(Int(mapped.unsafe_ptr()), 17 * 896, 0x42F6)
    var saved_d = List[DeviceBuffer[DType.bfloat16]]()
    var saved_y = List[DeviceBuffer[DType.bfloat16]]()
    var lengths = [17, 1, 7, 15, 16, 1, 17, 7, 1, 16, 15, 1]
    for _ in range(len(lengths)):
        saved_d.append(ctx.enqueue_create_buffer[DType.bfloat16](17 * 896))
        saved_y.append(ctx.enqueue_create_buffer[DType.bfloat16](17 * 896))
    for j in range(len(lengths)):
        var r = lengths[j]
        var start = (j * 3) % (18 - r)
        enqueue_mlp_apple_gpu(
            ctx,
            weights,
            work,
            TileTensor(
                xb.unsafe_ptr().unsafe_offset(start * 896), row_major(r, 896)
            ),
            mapping,
        )
        ctx.enqueue_copy(dst_buf=saved_d[j], src_buf=work.down)
        ctx.enqueue_copy(dst_buf=saved_y[j], src_buf=work.output)
    ctx.synchronize()
    for j in range(len(lengths)):
        var r = lengths[j]
        var start = (j * 3) % (18 - r)
        check_mlp(saved_d[j], case_id, "D", "reuse", start, r, 896, r * 896)
        check_mlp(saved_y[j], case_id, "Y", "reuse", start, r, 896, r * 896)


def test_invalid_calls_and_async_reuse() raises:
    var support = mlp_support()
    var variants = support.projection_variants()
    for j in range(Int(py=variants.__len__())):
        run_invalid_calls_and_async_reuse(Int(py=variants[j]))
