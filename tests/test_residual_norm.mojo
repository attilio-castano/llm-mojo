from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_equal, assert_raises, TestSuite
from llm_mojo.residual import enqueue_residual_apple_gpu
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.residual_norm import enqueue_residual_norm


def test_exact_composition_and_storage() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    print("Residual norm device:",ctx.name(),ctx.api())
    var x = ctx.enqueue_create_buffer[DType.bfloat16](898)
    var branch = ctx.enqueue_create_buffer[DType.bfloat16](898)
    var weight = ctx.enqueue_create_buffer[DType.bfloat16](898)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](898)
    var norm = ctx.enqueue_create_buffer[DType.bfloat16](898)
    var expected_y = ctx.enqueue_create_buffer[DType.bfloat16](898)
    var expected_norm = ctx.enqueue_create_buffer[DType.bfloat16](898)
    var edge: List[UInt16] = [0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x0480,0x3f80,0xbf80,0x3b80,0x3f81,0xbf81]
    for sweep in range(48):
        var xbits = List[UInt16]()
        var bbits = List[UInt16]()
        var wbits = List[UInt16]()
        for i in range(898):
            var a = bitcast[DType.uint16]((Float32((i*97+sweep*31)%2053-1026)/Float32(211)).cast[DType.bfloat16]())
            var b = bitcast[DType.uint16]((Float32((i*53+sweep*71)%2053-1026)/Float32(317)).cast[DType.bfloat16]())
            if sweep < 14:
                a = edge[(i+sweep)%len(edge)]
                b = edge[(i*3+sweep)%len(edge)]
            elif sweep < 24:
                b = a ^ UInt16(0x8000)
            elif sweep < 32:
                a = UInt16(0x4b00+(i%128))
                b = UInt16(0xcb00+((i+sweep)%128))
            elif sweep == 32:
                a = UInt16(1+i%127)
                b = UInt16(1+(i*3)%127)
            elif sweep == 33:
                a = 0x0480
                b = 0x807f
            elif sweep == 34:
                a = 0x8000
                b = 0x8000
            elif sweep == 35:
                a = UInt16(0x3f80+i%2)
                b = 0x3b80
            var w = bitcast[DType.uint16]((Float32((i*41+sweep*11)%511-255)/Float32(127)).cast[DType.bfloat16]())
            xbits.append(a)
            bbits.append(b)
            wbits.append(w)
        with x.map_to_host() as mapped:
            for i in range(898): mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](xbits[i])
        with branch.map_to_host() as mapped:
            for i in range(898): mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bbits[i])
        with weight.map_to_host() as mapped:
            for i in range(898): mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](wbits[i])
        y.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7fc1)))
        norm.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7fc1)))
        expected_y.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7fc1)))
        expected_norm.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7fc1)))
        var xv = TileTensor(x.unsafe_ptr().unsafe_offset(1),row_major(1,896))
        var bv = TileTensor(branch.unsafe_ptr().unsafe_offset(1),row_major(1,896))
        var wv = TileTensor(weight.unsafe_ptr().unsafe_offset(1),row_major(896))
        var yv = TileTensor(y.unsafe_ptr().unsafe_offset(1),row_major(1,896))
        var nv = TileTensor(norm.unsafe_ptr().unsafe_offset(1),row_major(1,896))
        var eyv = TileTensor(expected_y.unsafe_ptr().unsafe_offset(1),row_major(1,896))
        var env = TileTensor(expected_norm.unsafe_ptr().unsafe_offset(1),row_major(1,896))
        enqueue_residual_apple_gpu(ctx,xv,bv,eyv)
        enqueue_rms_norm_apple_gpu(ctx,eyv,wv,env)
        enqueue_residual_norm(ctx,xv,bv,wv,yv,nv)
        with y.map_to_host() as actual, expected_y.map_to_host() as expected:
            for i in range(898):
                assert_equal(bitcast[DType.uint16](actual.unsafe_ptr()[unsafe_offset=i]),bitcast[DType.uint16](expected.unsafe_ptr()[unsafe_offset=i]))
        with norm.map_to_host() as actual, expected_norm.map_to_host() as expected:
            for i in range(898):
                assert_equal(bitcast[DType.uint16](actual.unsafe_ptr()[unsafe_offset=i]),bitcast[DType.uint16](expected.unsafe_ptr()[unsafe_offset=i]))
        with x.map_to_host() as mapped:
            for i in range(898): assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]),xbits[i])
        with branch.map_to_host() as mapped:
            for i in range(898): assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]),bbits[i])
        with weight.map_to_host() as mapped:
            for i in range(898): assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]),wbits[i])
        with assert_raises(): enqueue_residual_norm(ctx,xv,bv,wv,xv,nv)
        with assert_raises(): enqueue_residual_norm(ctx,xv,bv,wv,yv,yv)
        with assert_raises(): enqueue_residual_norm(ctx,xv,bv,wv,yv,TileTensor(x.unsafe_ptr().unsafe_offset(2),row_major(1,896)))
        with assert_raises(): enqueue_residual_norm(ctx,xv,bv,TileTensor(weight,row_major(895)),yv,nv)
        with assert_raises(): enqueue_residual_norm(ctx,TileTensor(x,row_major(2,896)),bv,wv,yv,nv)
    print("Residual RMSNorm: 48 exact composition sweeps, protected outputs, unchanged inputs and rejected aliases/shapes")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
