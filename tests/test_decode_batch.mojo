"""Batched decode: every row of a batched step equals its sequence decoded alone.

The projections use the rows kernel, which reuses each weight across a tile of
rows while keeping every row's lane-strided FP32 sum, warp.sum, bias and single
rounding. At each decode width it must equal one-row launches bit for bit.
"""
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal
from llm_mojo.kernels.linear import enqueue_linear_apple_gpu, enqueue_linear_rowwise_rows_apple_gpu


def _fill(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int) raises:
    # Mixed signs, mantissas and exponents exercise rounded FP32 sums.
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            var raw = UInt32((i*1664525+seed*1013904223) & 0xffffffff)
            var bits = UInt16((raw >> 16) & 0x807f) | UInt16((119+((raw >> 7)%16)) << 7)
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)


def _same(mut expected: DeviceBuffer[DType.bfloat16], mut actual: DeviceBuffer[DType.bfloat16],
          label: String) raises:
    with expected.map_to_host() as a:
        with actual.map_to_host() as b:
            for i in range(len(expected)):
                if (bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i])
                        != bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i])):
                    raise Error(label + " differs at element " + String(i))


def _tile[TILE: Int, HAS_BIAS: Bool](ctx: DeviceContext, mut x: DeviceBuffer[DType.bfloat16],
                                     mut w: DeviceBuffer[DType.bfloat16], mut b: DeviceBuffer[DType.bfloat16],
                                     mut solo: DeviceBuffer[DType.bfloat16], rows: Int, n: Int, k: Int) raises:
    # One guard row on each side stays poisoned.
    var batch = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
    batch.enqueue_fill(-123)
    var input = TileTensor(x,row_major(rows,k))
    var weight = TileTensor(w,row_major(n,k))
    var output = TileTensor(batch.unsafe_ptr().unsafe_offset(n),row_major(rows,n))
    comptime if HAS_BIAS:
        enqueue_linear_rowwise_rows_apple_gpu[TILE](ctx,input,weight,TileTensor(b,row_major(n)),output)
    else:
        enqueue_linear_rowwise_rows_apple_gpu[TILE](ctx,input,weight,output)
    _same(solo,batch,"tile "+String(TILE)+" rows "+String(rows)+" outputs "+String(n)+" inputs "+String(k))


def _widths[HAS_BIAS: Bool](ctx: DeviceContext, n: Int, k: Int) raises:
    var w = ctx.enqueue_create_buffer[DType.bfloat16](n*k)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](n)
    _fill(w,n)
    _fill(b,k)
    for rows in [2, 3, 8, 16, 32, 64]:
        var x = ctx.enqueue_create_buffer[DType.bfloat16](rows*k)
        var solo = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
        _fill(x,rows)
        solo.enqueue_fill(-123)
        var weight = TileTensor(w,row_major(n,k))
        for row in range(rows):
            var input = TileTensor(x.unsafe_ptr().unsafe_offset(row*k),row_major(1,k))
            var output = TileTensor(solo.unsafe_ptr().unsafe_offset((row+1)*n),row_major(1,n))
            comptime if HAS_BIAS:
                enqueue_linear_apple_gpu(ctx,input,weight,TileTensor(b,row_major(n)),output)
            else:
                enqueue_linear_apple_gpu(ctx,input,weight,output)
        _tile[4,HAS_BIAS](ctx,x,w,b,solo,rows,n,k)
        _tile[8,HAS_BIAS](ctx,x,w,b,solo,rows,n,k)
        _tile[16,HAS_BIAS](ctx,x,w,b,solo,rows,n,k)


def test_rows_kernel_equals_one_row_launches_at_decode_widths() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    _widths[True](ctx,1152,896)
    _widths[False](ctx,896,896)
    _widths[False](ctx,4864,896)
    _widths[False](ctx,896,4864)
    _widths[False](ctx,151936,896)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
