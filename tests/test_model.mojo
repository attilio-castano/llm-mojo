"""Exact primitive checks; these do not establish full-model acceptance."""
from std.testing import TestSuite, assert_equal, assert_raises
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.model import _embedding, _copy_rows, load_bf16, save_bf16
from std.memory import bitcast


def test_embedding_repeated_ids_and_copy_guards() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    print("model primitive device",ctx.name(),"backend",ctx.api())
    var weight = ctx.enqueue_create_buffer[DType.bfloat16](5*896)
    var ids = ctx.enqueue_create_buffer[DType.int32](3)
    var result = ctx.enqueue_create_buffer[DType.bfloat16](5*896)
    var copy = ctx.enqueue_create_buffer[DType.bfloat16](5*896)
    with weight.map_to_host() as mapped:
        for r in range(5):
            for c in range(896):
                mapped.unsafe_ptr()[unsafe_offset=r*896+c] = Float32(r*10+c%7-20).cast[DType.bfloat16]()
    with ids.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=0] = 4
        mapped.unsafe_ptr()[unsafe_offset=1] = 0
        mapped.unsafe_ptr()[unsafe_offset=2] = 4
    result.enqueue_fill(-99)
    copy.enqueue_fill(-98)
    var it = TileTensor(ids,row_major(3))
    var wt = TileTensor(weight,row_major(5,896))
    var rt = TileTensor(result,row_major(3,896))
    var ct = TileTensor(copy,row_major(3,896))
    ctx.enqueue_function[_embedding[type_of(it.layout),type_of(wt.layout),type_of(rt.layout)]](
        it,wt,rt,Int32(3),grid_dim=11,block_dim=256)
    ctx.enqueue_function[_copy_rows[type_of(rt.layout),type_of(ct.layout)]](
        rt,ct,Int32(3),grid_dim=11,block_dim=256)
    with result.map_to_host() as mapped:
        for r in range(5):
            for c in range(896):
                var expected = (0 if r==1 else 4)*10+c%7-20 if r<3 else -99
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=r*896+c].cast[DType.float32](),Float32(expected))
    with copy.map_to_host() as mapped:
        for r in range(5):
            for c in range(896):
                var expected = (0 if r==1 else 4)*10+c%7-20 if r<3 else -98
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=r*896+c].cast[DType.float32](),Float32(expected))


def test_native_binary_io_preserves_bf16_bits() raises:
    var ctx = DeviceContext()
    var values = ctx.enqueue_create_buffer[DType.bfloat16](4)
    var expected: List[UInt16] = [0x3f80,0xbf80,0x8000,0x0001]
    with values.map_to_host() as mapped:
        for i in range(4):
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](expected[i])
    save_bf16(values,"build/model_io_test.bin",4)
    var bytes = open("build/model_io_test.bin","r").read_bytes()
    assert_equal(len(bytes),8)
    var loaded = ctx.enqueue_create_buffer[DType.bfloat16](4)
    load_bf16(loaded,"build/model_io_test.bin")
    with loaded.map_to_host() as mapped:
        for i in range(4):
            assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]),expected[i])
    with assert_raises():
        load_bf16(loaded,"build/model_io_test.bin",8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
