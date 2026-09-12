from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.python import Python
from std.testing import TestSuite, assert_equal
from llm_mojo.swiglu import enqueue_silu_apple_gpu, enqueue_multiply_apple_gpu


def test_fused_silu_multiply_exact() raises:
    from std.memory import bitcast
    from llm_mojo.swiglu import enqueue_silu_multiply_apple_gpu

    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    # All finite BF16 gate values, plus the actual Qwen width and ragged tails.
    for count in [1, 127, 129, 4864, 65280]:
        var g = ctx.enqueue_create_buffer[DType.bfloat16](count)
        var u = ctx.enqueue_create_buffer[DType.bfloat16](count)
        var a = ctx.enqueue_create_buffer[DType.bfloat16](count)
        var expected = ctx.enqueue_create_buffer[DType.bfloat16](count)
        var actual = ctx.enqueue_create_buffer[DType.bfloat16](count + 17)
        with g.map_to_host() as gm:
            for i in range(count):
                gm.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=i] = UInt16(i if i < 32640 else i + 128)
        for pattern in range(8):
            with u.map_to_host() as um:
                for i in range(count):
                    var bits: UInt16
                    if pattern == 0:
                        bits = 0
                    elif pattern == 1:
                        bits = 0x8000
                    elif pattern == 2:
                        bits = 0x0001
                    elif pattern == 3:
                        bits = 0x807F
                    elif pattern == 4:
                        bits = 0x3FC0
                    elif pattern == 5:
                        bits = 0xBF80
                    else:
                        var value = (i * 4051 + pattern * 37) % 65280
                        bits = UInt16(value if value < 32640 else value + 128)
                    um.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=i] = bits
            enqueue_silu_apple_gpu(ctx, TileTensor(g, row_major(1, count)), TileTensor(a, row_major(1, count)))
            enqueue_multiply_apple_gpu(ctx, TileTensor(a, row_major(1, count)),
                TileTensor(u, row_major(1, count)), TileTensor(expected, row_major(1, count)))
            # The fused route must not depend on an old activation or output.
            a.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
            actual.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
            enqueue_silu_multiply_apple_gpu(ctx, TileTensor(g, row_major(1, count)),
                TileTensor(u, row_major(1, count)), TileTensor(actual, row_major(1, count)))
            with expected.map_to_host() as em:
                with actual.map_to_host() as am:
                    for i in range(count):
                        assert_equal(am.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=i],
                            em.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=i])
                    for i in range(count, count + 17):
                        assert_equal(am.unsafe_ptr().unsafe_bitcast[UInt16]()[unsafe_offset=i], UInt16(0x7FC0))
    print("Fused SiLU multiply: 40 exact sweeps with protected tails")


def test_finite_bf16_sweep() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests")
    var support = Python.import_module("mlp_support")
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print("MLP activation runtime:", ctx.name(), ctx.api())
    var count = 65280
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var ub = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var yb = ctx.enqueue_create_buffer[DType.bfloat16](count)
    with xb.map_to_host() as mapped:
        support.load(Int(mapped.unsafe_ptr()), "activation/sweep_G.npy")
    var x = TileTensor(xb, row_major(1, count))
    var u = TileTensor(ub, row_major(1, count))
    var y = TileTensor(yb, row_major(1, count))
    enqueue_silu_apple_gpu(ctx, x, y)
    with yb.map_to_host() as mapped:
        support.check(
            Int(mapped.unsafe_ptr()), "activation/sweep_A.npy", "A", count
        )
    with xb.map_to_host() as mapped:
        support.load(Int(mapped.unsafe_ptr()), "activation/sweep_A.npy")
    for value in [-16.0, -1.0, -0.015625, 0.0, 0.015625, 1.0, 16.0]:
        ub.enqueue_fill(Float32(value).cast[DType.bfloat16]())
        enqueue_multiply_apple_gpu(ctx, x, u, y)
        with yb.map_to_host() as mapped:
            support.check_product(
                Int(mapped.unsafe_ptr()), count, Float64(value)
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_arbitrary_bf16_products() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests")
    var support = Python.import_module("mlp_support")
    var ctx = DeviceContext()
    var count = 262144
    var a = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](count)
    with a.map_to_host() as am:
        with b.map_to_host() as bm:
            support.multiplication_cases(
                Int(am.unsafe_ptr()), Int(bm.unsafe_ptr()), count
            )
    enqueue_multiply_apple_gpu(
        ctx,
        TileTensor(a, row_major(1, count)),
        TileTensor(b, row_major(1, count)),
        TileTensor(y, row_major(1, count)),
    )
    with y.map_to_host() as mapped:
        support.product_cases_check(Int(mapped.unsafe_ptr()), count)


def test_host_silu_reference() raises:
    from llm_mojo.swiglu import silu_reference

    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests")
    var support = Python.import_module("mlp_support")
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.bfloat16](65280)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](65280)
    with x.map_to_host() as xm:
        support.load(Int(xm.unsafe_ptr()), "activation/sweep_G.npy")
        with y.map_to_host() as ym:
            silu_reference(
                TileTensor(xm, row_major(1, 65280)),
                TileTensor(ym, row_major(1, 65280)),
            )
            support.check(
                Int(ym.unsafe_ptr()), "activation/sweep_A.npy", "A", 65280
            )


def test_rounding_boundaries() raises:
    from llm_mojo.linear import enqueue_linear_apple_gpu
    from llm_mojo.residual import enqueue_residual_apple_gpu

    var ctx = DeviceContext()
    var g = ctx.enqueue_create_buffer[DType.bfloat16](2)
    var u = ctx.enqueue_create_buffer[DType.bfloat16](2)
    var a = ctx.enqueue_create_buffer[DType.bfloat16](2)
    var s = ctx.enqueue_create_buffer[DType.bfloat16](2)
    g.enqueue_fill(0.015625)
    u.enqueue_fill(1.5)
    enqueue_silu_apple_gpu(
        ctx, TileTensor(g, row_major(1, 2)), TileTensor(a, row_major(1, 2))
    )
    enqueue_multiply_apple_gpu(
        ctx,
        TileTensor(a, row_major(1, 2)),
        TileTensor(u, row_major(1, 2)),
        TileTensor(s, row_major(1, 2)),
    )
    with s.map_to_host() as mapped:
        var t = TileTensor(mapped, row_major(2))
        comptime assert t.flat_rank == 1
        assert_equal(t[0].cast[DType.float32](), Float32(0.0118408203125))
    g.enqueue_fill(1)
    with u.map_to_host() as mapped:
        var t = TileTensor(mapped, row_major(2))
        comptime assert t.flat_rank == 1
        t[0] = 1
        t[1] = 0.00390625
    enqueue_linear_apple_gpu(
        ctx,
        TileTensor(g, row_major(1, 2)),
        TileTensor(u, row_major(1, 2)),
        TileTensor(a, row_major(1, 1)),
    )
    s.enqueue_fill(-1)
    enqueue_residual_apple_gpu(
        ctx,
        TileTensor(s, row_major(1, 1)),
        TileTensor(a, row_major(1, 1)),
        TileTensor(g, row_major(1, 1)),
    )
    with g.map_to_host() as mapped:
        var t = TileTensor(mapped, row_major(1))
        comptime assert t.flat_rank == 1
        assert_equal(t[0].cast[DType.float32](), Float32(0))


def test_residual_bf16_boundaries() raises:
    from llm_mojo.residual import enqueue_residual_apple_gpu

    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests")
    var support = Python.import_module("mlp_support")
    var ctx = DeviceContext()
    var count = 195840
    var a = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](count)
    with a.map_to_host() as am:
        with b.map_to_host() as bm:
            support.residual_cases(
                Int(am.unsafe_ptr()), Int(bm.unsafe_ptr()), count
            )
    enqueue_residual_apple_gpu(
        ctx,
        TileTensor(a, row_major(1, count)),
        TileTensor(b, row_major(1, count)),
        TileTensor(y, row_major(1, count)),
    )
    with y.map_to_host() as mapped:
        support.residual_cases_check(Int(mapped.unsafe_ptr()), count)


def test_host_multiply_reference() raises:
    from llm_mojo.swiglu import multiply_reference

    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests")
    var support = Python.import_module("mlp_support")
    var ctx = DeviceContext()
    var count = 65536
    var a = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](count)
    with a.map_to_host() as am:
        with b.map_to_host() as bm:
            support.multiplication_cases(
                Int(am.unsafe_ptr()), Int(bm.unsafe_ptr()), count
            )
            with y.map_to_host() as ym:
                multiply_reference(
                    TileTensor(am, row_major(1, count)),
                    TileTensor(bm, row_major(1, count)),
                    TileTensor(ym, row_major(1, count)),
                )
                support.product_cases_check(Int(ym.unsafe_ptr()), count)


def test_every_finite_bf16_plus_every_subnormal() raises:
    from llm_mojo.residual import enqueue_residual_apple_gpu

    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests")
    var support = Python.import_module("mlp_support")
    var ctx = DeviceContext()
    var count = 65280 * 256
    var a = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](count)
    with a.map_to_host() as am:
        with b.map_to_host() as bm:
            support.residual_subnormal_cases(
                Int(am.unsafe_ptr()), Int(bm.unsafe_ptr())
            )
    enqueue_residual_apple_gpu(
        ctx,
        TileTensor(a, row_major(1, count)),
        TileTensor(b, row_major(1, count)),
        TileTensor(y, row_major(1, count)),
    )
    with y.map_to_host() as mapped:
        support.residual_cases_check(
            Int(mapped.unsafe_ptr()), count, "finite_bf16_by_every_subnormal"
        )
