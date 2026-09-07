from llm_mojo.linear import (
    linear_reference,
    enqueue_linear_apple_gpu,
    enqueue_linear_prefill_register_2x2_apple_gpu,
    enqueue_linear_prefill_mma_8x16_apple_gpu,
)
from llm_mojo.residual import residual_reference, enqueue_residual_apple_gpu
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_raises


def test_bias_free_projection_has_no_bias_and_handles_ragged_tiles() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](9 * 33)
    var wb = ctx.enqueue_create_buffer[DType.bfloat16](17 * 33)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](9 * 17)
    with xb.map_to_host() as xm:
        var x = TileTensor(xm, row_major(9, 33))
        for r in range(9):
            for k in range(33):
                x[r, k] = (Float32((r * 7 + k * 3) % 17 - 8) / 8).cast[
                    DType.bfloat16
                ]()
    wb.enqueue_fill(0)
    with wb.map_to_host() as wm:
        var w = TileTensor(wm, row_major(17, 33))
        for n in range(17):
            w[n, n] = 1
            w[n, n + 3] = -1
    for route in range(4):
        ob.enqueue_fill(123)
        if route == 0:
            with xb.map_to_host() as xm:
                with wb.map_to_host() as wm:
                    with ob.map_to_host() as om:
                        linear_reference(
                            TileTensor(xm, row_major(9, 33)),
                            TileTensor(wm, row_major(17, 33)),
                            TileTensor(om, row_major(9, 17)),
                        )
        elif route == 1:
            enqueue_linear_apple_gpu(
                ctx,
                TileTensor(xb, row_major(9, 33)),
                TileTensor(wb, row_major(17, 33)),
                TileTensor(ob, row_major(9, 17)),
            )
        elif route == 2:
            enqueue_linear_prefill_register_2x2_apple_gpu(
                ctx,
                TileTensor(xb, row_major(9, 33)),
                TileTensor(wb, row_major(17, 33)),
                TileTensor(ob, row_major(9, 17)),
            )
        else:
            enqueue_linear_prefill_mma_8x16_apple_gpu(
                ctx,
                TileTensor(xb, row_major(9, 33)),
                TileTensor(wb, row_major(17, 33)),
                TileTensor(ob, row_major(9, 17)),
            )
        with ob.map_to_host() as om:
            var output = TileTensor(om, row_major(9, 17))
            for r in range(9):
                for n in range(17):
                    var expected = (
                        Float32(
                            ((r * 7 + n * 3) % 17)
                            - ((r * 7 + (n + 3) * 3) % 17)
                        )
                        / 8
                    )
                    assert_equal(
                        rebind[Float32](output[r, n].cast[DType.float32]()),
                        expected,
                    )
    with assert_raises(contains="weight input"):
        enqueue_linear_apple_gpu(
            ctx,
            TileTensor(xb, row_major(9, 33)),
            TileTensor(wb, row_major(17, 32)),
            TileTensor(ob, row_major(9, 17)),
        )


def test_residual_rounding_and_shape_validation() raises:
    var ctx = DeviceContext()
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](2)
    var bb = ctx.enqueue_create_buffer[DType.bfloat16](2)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](2)
    xb.enqueue_fill(1)
    with bb.map_to_host() as bm:
        var b = TileTensor(bm, row_major(1, 2))
        b[0, 0] = 0.00390625
        b[0, 1] = 0.01171875
    for route in range(2):
        if route == 0:
            with xb.map_to_host() as xm:
                with bb.map_to_host() as bm:
                    with ob.map_to_host() as om:
                        residual_reference(
                            TileTensor(xm, row_major(1, 2)),
                            TileTensor(bm, row_major(1, 2)),
                            TileTensor(om, row_major(1, 2)),
                        )
        else:
            enqueue_residual_apple_gpu(
                ctx,
                TileTensor(xb, row_major(1, 2)),
                TileTensor(bb, row_major(1, 2)),
                TileTensor(ob, row_major(1, 2)),
            )
        with ob.map_to_host() as om:
            var y = TileTensor(om, row_major(1, 2))
            assert_equal(
                rebind[Float32](y[0, 0].cast[DType.float32]()), Float32(1)
            )
            assert_equal(
                rebind[Float32](y[0, 1].cast[DType.float32]()),
                Float32(1.015625),
            )
    with assert_raises(contains="shapes"):
        enqueue_residual_apple_gpu(
            ctx,
            TileTensor(xb, row_major(1, 2)),
            TileTensor(bb, row_major(2, 1)),
            TileTensor(ob, row_major(1, 2)),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
