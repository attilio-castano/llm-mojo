"""Exact primitive checks; these do not establish full-model acceptance."""
from std.testing import TestSuite, assert_equal, assert_raises
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.model import _embedding, _copy_rows, load_bf16, save_bf16
from std.memory import bitcast
from llm_mojo.model import select_configuration, select_token_selection, select_copy_free, swap_hidden_buffers
from llm_mojo.generate_cli import generation_budget, is_stop


def test_hidden_buffer_swaps_keep_queued_views_alive() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    var left = ctx.enqueue_create_buffer[DType.bfloat16](896)
    var right = ctx.enqueue_create_buffer[DType.bfloat16](896)
    var original_left = Int(left.unsafe_ptr())
    var original_right = Int(right.unsafe_ptr())
    with left.map_to_host() as mapped:
        for i in range(896):
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](UInt16(i+1))
    right.enqueue_fill(-99)
    for step in range(47):
        var source = TileTensor(left,row_major(1,896))
        var destination = TileTensor(right,row_major(1,896))
        ctx.enqueue_function[_copy_rows[type_of(source.layout),type_of(destination.layout)]](
            source,destination,Int32(1),grid_dim=4,block_dim=256)
        swap_hidden_buffers(left,right)
        assert_equal(Int(left.unsafe_ptr()),original_right if step%2 == 0 else original_left)
        assert_equal(Int(right.unsafe_ptr()),original_left if step%2 == 0 else original_right)
    # No synchronization or host mapping between enqueues and ownership changes.
    with left.map_to_host() as mapped:
        for i in range(896):
            assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]),UInt16(i+1))
    for policy in ["fast","auto","combined","gpu-argmax","fused-head","baseline"]:
        assert_equal(select_copy_free(policy,1,"Apple M4 Pro"),False)
    assert_equal(select_copy_free("buffer-swap",1,"Apple M4 Pro"),True)
    assert_equal(select_copy_free("buffer-swap",2,"Apple M4 Pro"),False)
    assert_equal(select_copy_free("buffer-swap",1,"other"),False)
    assert_equal(select_configuration("buffer-swap",1,64,"Apple M4 Pro"),26)


def test_generation_limits_and_policy() raises:
    for policy in ["fast","auto","combined","fusion","unfused","baseline"]:
        assert_equal(select_token_selection(policy,1,"Apple M4 Pro"),0)
    for policy in ["gpu-argmax","fused-head"]:
        assert_equal(select_configuration(policy,1,64,"Apple M4 Pro"),26)
        assert_equal(select_token_selection(policy,1,"Apple M4 Pro"),1 if policy == "gpu-argmax" else 2)
        assert_equal(select_token_selection(policy,16,"Apple M4 Pro"),0)
        assert_equal(select_token_selection(policy,1,"other"),0)
    assert_equal(generation_budget(4096,32),0)
    assert_equal(generation_budget(4095,32),1)
    assert_equal(generation_budget(1,0),0)
    assert_equal(is_stop(151645),True)
    assert_equal(is_stop(151643),True)
    assert_equal(is_stop(151644),False)
    with assert_raises():
        _ = generation_budget(0,1)
    with assert_raises():
        _ = generation_budget(4097,1)
    with assert_raises():
        _ = generation_budget(1,-1)
    assert_equal(select_configuration("combined",1,1024,"Apple M4 Pro"),26)
    assert_equal(select_configuration("combined",16,256,"Apple M4 Pro"),21)
    assert_equal(select_configuration("combined",1,1024,"other"),0)
    assert_equal(select_configuration("unfused",1,1024,"Apple M4 Pro"),0)
    assert_equal(select_configuration("unfused",16,256,"Apple M4 Pro"),21)
    for total in [1,64,1024,3968,4096]:
        assert_equal(select_configuration("fast",1,total,"Apple M4 Pro"),26)
        assert_equal(select_configuration("auto",1,total,"Apple M4 Pro"),26)
    assert_equal(select_configuration("fusion",1,1024,"Apple M4 Pro"),25)
    assert_equal(select_configuration("fusion",16,256,"Apple M4 Pro"),21)
    assert_equal(select_configuration("fusion",1,1024,"other"),0)
    assert_equal(select_configuration("21",16,256,"Apple M4 Pro"),21)
    assert_equal(select_configuration("fast",1,1024,"other"),0)
    assert_equal(select_configuration("fast",16,256,"Apple M4 Pro"),21)
    assert_equal(select_configuration("fast",16,1024,"Apple M4 Pro"),2)
    assert_equal(select_configuration("auto",64,4096,"Apple M4 Pro"),3)
    assert_equal(select_configuration("fast",16,255,"Apple M4 Pro"),0)
    assert_equal(select_configuration("fast",64,4096,"other"),0)
    assert_equal(select_configuration("auto",17,257,"other"),0)
    with assert_raises():
        _ = select_configuration("fast",0,1,"")
    with assert_raises():
        _ = select_configuration("typo",1,1,"")


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
    save_bf16(values,"build/model_io_slice.bin",2,1)
    var slice = open("build/model_io_slice.bin","r").read_bytes()
    assert_equal(len(slice),4)
    for i in range(2):
        assert_equal(UInt16(slice[2*i]) | (UInt16(slice[2*i+1]) << 8),expected[i+1])
    with assert_raises():
        save_bf16(values,"build/model_io_invalid.bin",2,3)
    with assert_raises():
        load_bf16(loaded,"build/model_io_test.bin",8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
