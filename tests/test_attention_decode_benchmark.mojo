"""Exercise the actual benchmark routing, independently of kernel parity."""

from llm_mojo.benchmarks.attention_decode import enqueue_variant, variant_splits
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def test_benchmark_launches_every_requested_variant() raises:
    var ctx = DeviceContext()
    var qb = ctx.enqueue_create_buffer[DType.bfloat16](896)
    var kb = ctx.enqueue_create_buffer[DType.bfloat16](7 * 128)
    var vb = ctx.enqueue_create_buffer[DType.bfloat16](7 * 128)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](896)
    var sb = ctx.enqueue_create_buffer[DType.bfloat16](14 * 7)
    var wb = ctx.enqueue_create_buffer[DType.float32](14 * 64 * 66)
    qb.enqueue_fill(0.25)
    kb.enqueue_fill(0.5)
    vb.enqueue_fill(1.0)
    var q = TileTensor(qb, row_major(1, 14, 64))
    var k = TileTensor(kb, row_major(7, 2, 64))
    var v = TileTensor(vb, row_major(7, 2, 64))
    var output = TileTensor(ob, row_major(1, 14, 64))
    var scratch = TileTensor(sb, row_major(1, 14, 7))
    for variant in range(13):
        var workspace = TileTensor(
            wb, row_major(14, variant_splits(variant), 66)
        )
        assert_equal(
            enqueue_variant(variant, ctx, q, k, v, output, scratch, workspace),
            variant,
        )
    ctx.synchronize()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
