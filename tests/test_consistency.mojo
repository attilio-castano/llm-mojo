"""Exact causal schedule checks for the unified G32 attention primitive."""
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.attention_decode import enqueue_grouped_query_attention_consistent_apple_gpu, enqueue_grouped_query_attention_decode_apple_gpu
from llm_mojo.decoder_layer import decoder_mappings
from llm_mojo.linear import enqueue_linear_apple_gpu, enqueue_linear_rowwise_rows_apple_gpu
from decoder_layer_support import decoder_support
from test_decoder_layer import _case


def _fill(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int) raises:
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            mapped.unsafe_ptr()[unsafe_offset=i] = (Float32((i*37+seed*13)%1009-504)/Float32(256)).cast[DType.bfloat16]()


def test_causal_queries_equal_individual_decode() raises:
    var ctx = DeviceContext()
    print("consistency primitive device",ctx.name(),"backend",ctx.api())
    assert_equal(ctx.api(),"metal")
    for t in [1, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 257]:
        var qb = ctx.enqueue_create_buffer[DType.bfloat16](t*896)
        var kb = ctx.enqueue_create_buffer[DType.bfloat16](t*128)
        var vb = ctx.enqueue_create_buffer[DType.bfloat16](t*128)
        var full = ctx.enqueue_create_buffer[DType.bfloat16](t*896)
        var chunks = ctx.enqueue_create_buffer[DType.bfloat16](t*896)
        var wb = ctx.enqueue_create_buffer[DType.float32](1)
        _fill(qb,1)
        _fill(kb,2)
        _fill(vb,3)
        var w = TileTensor(wb,row_major(1,1,1))
        enqueue_grouped_query_attention_consistent_apple_gpu(ctx,
            TileTensor(qb,row_major(t,14,64)),TileTensor(kb,row_major(t,2,64)),
            TileTensor(vb,row_major(t,2,64)),TileTensor(full,row_major(t,14,64)),w)
        for r in [1, 3, 16, 32, 63]:
            chunks.enqueue_fill(-123)
            var start = 0
            while start < t:
                var n = min(r,t-start)
                enqueue_grouped_query_attention_consistent_apple_gpu(ctx,
                    TileTensor(qb.unsafe_ptr().unsafe_offset(start*896),row_major(n,14,64)),
                    TileTensor(kb,row_major(start+n,2,64)),TileTensor(vb,row_major(start+n,2,64)),
                    TileTensor(chunks.unsafe_ptr().unsafe_offset(start*896),row_major(n,14,64)),w)
                start += n
            with full.map_to_host() as a:
                with chunks.map_to_host() as b:
                    for i in range(t*896):
                        assert_equal(bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i]),
                                     bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i]))
        # Bind the generalized kernel to the pre-existing decode arithmetic.
        for p in range(t):
            enqueue_grouped_query_attention_decode_apple_gpu[32,1,1,fp32_scores=True](ctx,
                TileTensor(qb.unsafe_ptr().unsafe_offset(p*896),row_major(1,14,64)),
                TileTensor(kb,row_major(p+1,2,64)),TileTensor(vb,row_major(p+1,2,64)),
                TileTensor(chunks.unsafe_ptr().unsafe_offset(p*896),row_major(1,14,64)),w)
        with full.map_to_host() as a:
            with chunks.map_to_host() as b:
                for i in range(t*896):
                    assert_equal(bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i]),
                                 bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i]))



def _fill_linear(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int) raises:
    # Mixed signs, mantissas and exponents exercise rounded FP32 sums.
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            var raw = UInt32((i*1664525+seed*1013904223) & 0xffffffff)
            var bits = UInt16((raw >> 16) & 0x807f) | UInt16((119+((raw >> 7)%16)) << 7)
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)


def _rowwise_weight_reuse[HAS_BIAS: Bool, ROW_TILE: Int]() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    var n = 19
    for k in [37,896,4864]:
        for rows in [1,2,3,4,5,7,8,15,16,17,33]:
            var xb = ctx.enqueue_create_buffer[DType.bfloat16](rows*k)
            var wb = ctx.enqueue_create_buffer[DType.bfloat16](n*k)
            var bb = ctx.enqueue_create_buffer[DType.bfloat16](n)
            var old = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
            var full = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
            var chunks = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
            _fill_linear(xb,1)
            _fill_linear(wb,2)
            _fill_linear(bb,3)
            old.enqueue_fill(-123)
            full.enqueue_fill(-123)
            var x = TileTensor(xb,row_major(rows,k))
            var w = TileTensor(wb,row_major(n,k))
            var b = TileTensor(bb,row_major(n))
            var a = TileTensor(old.unsafe_ptr().unsafe_offset(n),row_major(rows,n))
            var y = TileTensor(full.unsafe_ptr().unsafe_offset(n),row_major(rows,n))
            comptime if HAS_BIAS:
                enqueue_linear_apple_gpu(ctx,x,w,b,a)
                enqueue_linear_rowwise_rows_apple_gpu[ROW_TILE](ctx,x,w,b,y)
            else:
                enqueue_linear_apple_gpu(ctx,x,w,a)
                enqueue_linear_rowwise_rows_apple_gpu[ROW_TILE](ctx,x,w,y)
            for chunk in [1,3,16]:
                chunks.enqueue_fill(-123)
                var p = 0
                while p < rows:
                    var r = min(chunk,rows-p)
                    var cx = TileTensor(xb.unsafe_ptr().unsafe_offset(p*k),row_major(r,k))
                    var cy = TileTensor(chunks.unsafe_ptr().unsafe_offset((p+1)*n),row_major(r,n))
                    comptime if HAS_BIAS:
                        enqueue_linear_rowwise_rows_apple_gpu[ROW_TILE](ctx,cx,w,b,cy)
                    else:
                        enqueue_linear_rowwise_rows_apple_gpu[ROW_TILE](ctx,cx,w,cy)
                    p += r
                with old.map_to_host() as reference:
                    with full.map_to_host() as actual:
                        with chunks.map_to_host() as split:
                            # Include both guard rows and every active output.
                            for i in range(len(old)):
                                var expected = bitcast[DType.uint16](reference.unsafe_ptr()[unsafe_offset=i])
                                assert_equal(bitcast[DType.uint16](actual.unsafe_ptr()[unsafe_offset=i]),expected)
                                assert_equal(bitcast[DType.uint16](split.unsafe_ptr()[unsafe_offset=i]),expected)


def test_rowwise_weight_reuse_preserves_bits_and_guards() raises:
    _rowwise_weight_reuse[True,4]()
    _rowwise_weight_reuse[False,4]()
    _rowwise_weight_reuse[True,8]()
    _rowwise_weight_reuse[False,8]()
    _rowwise_weight_reuse[True,16]()
    _rowwise_weight_reuse[False,16]()


def test_consistency_mapping_is_independent_of_rows() raises:
    for rows in [1,15,16,17,65,1024,4096]:
        assert_equal(decoder_mappings(20,rows),SIMD[DType.int64,4](5,0,0,1))
        assert_equal(decoder_mappings(21,rows),SIMD[DType.int64,4](5,6,7,1))
        assert_equal(decoder_mappings(22,rows),SIMD[DType.int64,4](5,7,19,1))
        assert_equal(decoder_mappings(23,rows),SIMD[DType.int64,4](5,8,20,1))
        assert_equal(decoder_mappings(24,rows),SIMD[DType.int64,4](5,9,21,1))
    with assert_raises():
        _ = decoder_mappings(20,0)


def test_consistent_decoder_accuracy_and_schedules() raises:
    var support = decoder_support()
    for spec in support.cases():
        if Int(py=spec[3]) == 14 and Int(py=spec[1]) <= 65:
            _case(String(py=spec[0]),Int(py=spec[1]),Int(py=spec[2]),Int(py=spec[3]),
                  Int(py=spec[4]),Int(py=spec[5]),Int(py=spec[6]),20,True)
    support.result_summary()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
