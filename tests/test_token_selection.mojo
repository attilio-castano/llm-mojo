"""BF16 ordering, reduction boundaries and fused projection storage parity."""
from std.memory import bitcast
from std.testing import assert_equal, assert_raises
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.linear import enqueue_linear_apple_gpu
from llm_mojo.token_selection import enqueue_argmax, enqueue_head_argmax, bf16_rank


def reduction(ctx: DeviceContext, count: Int, pattern: Int) raises:
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var groups = (count+1023)//1024
    var partials = ctx.enqueue_create_buffer[DType.uint32]((groups+1)*3)
    var result = ctx.enqueue_create_buffer[DType.uint32](6)
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    var winner = 0
    var best = Float32(-3.402823466e38)
    var invalid: UInt32 = 0
    with logits.map_to_host() as mapped:
        for i in range(count):
            var bits = UInt16(i % 65536)
            if pattern == 1:
                bits = UInt16(0xBF80)  # all negative, last wins
                if i == count-1:
                    bits = UInt16(0x8001)  # negative subnormal
            elif pattern == 2:
                bits = UInt16(0x8000 if i%2 == 0 else 0)
            elif pattern == 3:
                bits = UInt16(0xBF80)
                if i == 1 or i == count-1:
                    bits = UInt16(1)  # tied positive subnormals across groups
            elif pattern == 7:
                bits = UInt16(1 if i == count-1 else 0)
            elif pattern == 8:
                bits = UInt16(0x8000 if i == count-1 else 0x8001)
            elif pattern >= 4:
                bits = UInt16(0x3F80)
                if i == count-1:
                    bits = UInt16(0x7FC0 if pattern == 4 else (0x7F80 if pattern == 5 else 0xFF80))
            # Pattern zero enumerates every finite encoding.
            elif (bits & 0x7F80) == 0x7F80:
                bits = UInt16(0)
            var value = bitcast[DType.bfloat16](bits)
            mapped.unsafe_ptr()[unsafe_offset=i] = value
            var f = value.cast[DType.float32]()
            if f != f or f > Float32(3.402823466e38) or f < Float32(-3.402823466e38):
                invalid = 1
            elif f > best:
                best = f
                winner = i
    enqueue_argmax(ctx,TileTensor(logits,row_major(1,count)),
        TileTensor(partials,row_major(groups,3)),TileTensor(result,row_major(1,3)))
    with result.map_to_host() as mapped:
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=2],invalid)
        if invalid == 0:
            assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=1]),winner)
        for i in range(3,6):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i],UInt32(0xDEADBEEF))
    with partials.map_to_host() as mapped:
        for i in range(groups*3,(groups+1)*3):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i],UInt32(0xDEADBEEF))


def projection(ctx: DeviceContext, count: Int, width: Int) raises:
    var input = ctx.enqueue_create_buffer[DType.bfloat16](width)
    var weight = ctx.enqueue_create_buffer[DType.bfloat16](count*width)
    var reference = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var groups = (count+63)//64
    var partials = ctx.enqueue_create_buffer[DType.uint32]((groups+1)*3)
    var result = ctx.enqueue_create_buffer[DType.uint32](6)
    with input.map_to_host() as mapped:
        for k in range(width):
            mapped.unsafe_ptr()[unsafe_offset=k] = (Float32(1) if width == 1 else Float32((k*13)%31-15)/17).cast[DType.bfloat16]()
    with weight.map_to_host() as mapped:
        for i in range(count*width):
            var value = (Float32((i*17+i//width*7)%97-48)/53).cast[DType.bfloat16]()
            if width == 1:
                var bits = UInt16(i % 65536)
                if (bits & 0x7F80) == 0x7F80:
                    bits = 0
                value = bitcast[DType.bfloat16](bits)
            mapped.unsafe_ptr()[unsafe_offset=i] = value
    enqueue_linear_apple_gpu(ctx,TileTensor(input,row_major(1,width)),
        TileTensor(weight,row_major(count,width)),TileTensor(reference,row_major(1,count)))
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    logits.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
    enqueue_head_argmax[True](ctx,TileTensor(input,row_major(1,width)),TileTensor(weight,row_major(count,width)),
        TileTensor(logits,row_major(1,count)),TileTensor(partials,row_major(groups,3)),TileTensor(result,row_major(1,3)))
    var winner = 0
    var best = Float32(-3.402823466e38)
    with reference.map_to_host() as expected_map:
        with logits.map_to_host() as got:
            for i in range(count):
                assert_equal(bitcast[DType.uint16](expected_map.unsafe_ptr()[unsafe_offset=i]),bitcast[DType.uint16](got.unsafe_ptr()[unsafe_offset=i]))
                var value = expected_map.unsafe_ptr()[unsafe_offset=i].cast[DType.float32]()
                if value > best:
                    best = value
                    winner = i
    with result.map_to_host() as mapped:
        assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=1]),winner)
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=2],UInt32(0))
    logits.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    enqueue_head_argmax[False](ctx,TileTensor(input,row_major(1,width)),TileTensor(weight,row_major(count,width)),
        TileTensor(logits,row_major(1,count)),TileTensor(partials,row_major(groups,3)),TileTensor(result,row_major(1,3)))
    with result.map_to_host() as mapped:
        assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=1]),winner)
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=2],UInt32(0))
        for i in range(3,6):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i],UInt32(0xDEADBEEF))
    with logits.map_to_host() as mapped:
        for i in range(count):
            assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]),UInt16(0x7FC0))
    with partials.map_to_host() as mapped:
        for i in range(groups*3,(groups+1)*3):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i],UInt32(0xDEADBEEF))


def rounded_tie(ctx: DeviceContext) raises:
    # Token 64 is larger in FP32 but ties token 0 after BF16 materialization.
    var input = ctx.enqueue_create_buffer[DType.bfloat16](2)
    var weight = ctx.enqueue_create_buffer[DType.bfloat16](65*2)
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](65)
    var partials = ctx.enqueue_create_buffer[DType.uint32](6)
    var result = ctx.enqueue_create_buffer[DType.uint32](3)
    input.enqueue_fill(1)
    weight.enqueue_fill(-1)
    with weight.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=0] = 1
        mapped.unsafe_ptr()[unsafe_offset=1] = 0
        mapped.unsafe_ptr()[unsafe_offset=128] = 1
        mapped.unsafe_ptr()[unsafe_offset=129] = Float32(1.0/1024).cast[DType.bfloat16]()
    enqueue_head_argmax[True](ctx,TileTensor(input,row_major(1,2)),TileTensor(weight,row_major(65,2)),
        TileTensor(logits,row_major(1,65)),TileTensor(partials,row_major(2,3)),TileTensor(result,row_major(1,3)))
    with result.map_to_host() as mapped:
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=1],UInt32(0))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=2],UInt32(0))
    with logits.map_to_host() as mapped:
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=0].cast[DType.float32](),Float32(1))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=64].cast[DType.float32](),Float32(1))


def main() raises:
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("requires Metal")
    print("device:",ctx.name(),"api:",ctx.api())
    assert_equal(bf16_rank(UInt16(0)),bf16_rank(UInt16(0x8000)))
    for count in [1,31,64,65,1023,1024,1025,65536,151936]:
        for pattern in range(9):
            reduction(ctx,count,pattern)
    for count in [1,63,64,65,1025,151936]:
        projection(ctx,count,896)
    rounded_tie(ctx)
    projection(ctx,65,33)
    projection(ctx,65536,1)
    var input = ctx.enqueue_create_buffer[DType.bfloat16](1)
    var weight = ctx.enqueue_create_buffer[DType.bfloat16](65)
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](65)
    var partials = ctx.enqueue_create_buffer[DType.uint32](6)
    var result = ctx.enqueue_create_buffer[DType.uint32](3)
    input.enqueue_fill(1)
    for bits in [0x7FC0,0x7F80,0xFF80]:
        weight.enqueue_fill(1)
        with weight.map_to_host() as mapped:
            mapped.unsafe_ptr()[unsafe_offset=64] = bitcast[DType.bfloat16](UInt16(bits))
        enqueue_head_argmax[False](ctx,TileTensor(input,row_major(1,1)),TileTensor(weight,row_major(65,1)),
            TileTensor(logits,row_major(1,65)),TileTensor(partials,row_major(2,3)),TileTensor(result,row_major(1,3)))
        with result.map_to_host() as mapped:
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=2],UInt32(1))
    with assert_raises():
        enqueue_argmax(ctx,TileTensor(logits,row_major(1,65)),TileTensor(partials,row_major(0,3)),TileTensor(result,row_major(1,3)))
    with assert_raises():
        enqueue_argmax(ctx,TileTensor(logits,row_major(1,0)),TileTensor(partials,row_major(2,3)),TileTensor(result,row_major(1,3)))
    with assert_raises():
        enqueue_head_argmax[False](ctx,TileTensor(input,row_major(1,1)),TileTensor(weight,row_major(64,1)),
            TileTensor(logits,row_major(1,65)),TileTensor(partials,row_major(2,3)),TileTensor(result,row_major(1,3)))
    print("Token selection ordering, rejection and exact projection tests passed")
