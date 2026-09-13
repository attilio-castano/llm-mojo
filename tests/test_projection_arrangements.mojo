from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_equal, assert_raises, TestSuite
from llm_mojo.linear import enqueue_linear_apple_gpu


def test_arrangements_preserve_bits_and_guards() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    print("Projection arrangement device:",ctx.name(),ctx.api())
    var edges: List[UInt16] = [0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x3f80,0xbf80,0x3f81,0xbf81,0x3b80]
    var cases = 0
    for k in [896,4864]:
        for n in [1,7,8,9,33]:
            var x = ctx.enqueue_create_buffer[DType.bfloat16](k+2)
            var w = ctx.enqueue_create_buffer[DType.bfloat16](n*k+2)
            var b = ctx.enqueue_create_buffer[DType.bfloat16](n+2)
            var ref_output = ctx.enqueue_create_buffer[DType.bfloat16](n+2)
            var output = ctx.enqueue_create_buffer[DType.bfloat16](n+2)
            var xt = TileTensor(x.unsafe_ptr().unsafe_offset(1),row_major(1,k))
            var wt = TileTensor(w.unsafe_ptr().unsafe_offset(1),row_major(n,k))
            var bt = TileTensor(b.unsafe_ptr().unsafe_offset(1),row_major(n))
            var expected = TileTensor(ref_output.unsafe_ptr().unsafe_offset(1),row_major(1,n))
            var actual = TileTensor(output.unsafe_ptr().unsafe_offset(1),row_major(1,n))
            for sweep in range(6):
                var xb = List[UInt16]()
                var wb = List[UInt16]()
                var bb = List[UInt16]()
                for i in range(k+2):
                    xb.append(edges[(i+sweep)%len(edges)] if sweep < 2 else bitcast[DType.uint16]((Float32((i*97+sweep*31)%2053-1026)/211).cast[DType.bfloat16]()))
                for i in range(n*k+2):
                    wb.append(edges[(i*3+sweep)%len(edges)] if sweep < 2 else bitcast[DType.uint16]((Float32((i*53+sweep*71)%2053-1026)/317).cast[DType.bfloat16]()))
                for i in range(n+2): bb.append(edges[(i+sweep*3)%len(edges)])
                with x.map_to_host() as m:
                    for i in range(k+2): m.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](xb[i])
                with w.map_to_host() as m:
                    for i in range(n*k+2): m.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](wb[i])
                with b.map_to_host() as m:
                    for i in range(n+2): m.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bb[i])
                for variant in range(6):
                    ref_output.enqueue_fill(-99)
                    output.enqueue_fill(-99)
                    if sweep % 2 == 0:
                        enqueue_linear_apple_gpu(ctx,xt,wt,bt,expected)
                        enqueue_linear_apple_gpu(ctx,xt,wt,bt,actual,variant)
                    else:
                        enqueue_linear_apple_gpu(ctx,xt,wt,expected)
                        enqueue_linear_apple_gpu(ctx,xt,wt,actual,variant)
                    with ref_output.map_to_host() as a:
                        with output.map_to_host() as z:
                            for i in range(n+2):
                                assert_equal(bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i]),bitcast[DType.uint16](z.unsafe_ptr()[unsafe_offset=i]))
                            assert_equal(z.unsafe_ptr()[unsafe_offset=0],Scalar[DType.bfloat16](-99))
                            assert_equal(z.unsafe_ptr()[unsafe_offset=n+1],Scalar[DType.bfloat16](-99))
                with x.map_to_host() as m:
                    for i in range(k+2): assert_equal(bitcast[DType.uint16](m.unsafe_ptr()[unsafe_offset=i]),xb[i])
                with w.map_to_host() as m:
                    for i in range(n*k+2): assert_equal(bitcast[DType.uint16](m.unsafe_ptr()[unsafe_offset=i]),wb[i])
                with b.map_to_host() as m:
                    for i in range(n+2): assert_equal(bitcast[DType.uint16](m.unsafe_ptr()[unsafe_offset=i]),bb[i])
                cases += 1
            with assert_raises(): enqueue_linear_apple_gpu(ctx,xt,wt,actual,6)
            with assert_raises(): enqueue_linear_apple_gpu(ctx,xt,wt,actual,-1)
            var short_x = TileTensor(x,row_major(1,32))
            var short_w = TileTensor(w,row_major(n,32))
            with assert_raises(): enqueue_linear_apple_gpu(ctx,short_x,short_w,actual,1)
            var multi_x = TileTensor(x,row_major(2,k))
            var multi_y = TileTensor(output,row_major(2,n))
            with assert_raises(): enqueue_linear_apple_gpu(ctx,multi_x,wt,multi_y,1)
    print("Exact projection cases:",cases,"variants: 6; guards and inputs unchanged")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
