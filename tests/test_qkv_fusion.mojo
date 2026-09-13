"""Exact comparison with the existing four-kernel BF16 path, including guards."""
from llm_mojo.attention_sublayer import (
    AttentionWorkspace, AttentionCache, AttentionWeights, _enqueue_fused_decode_qkv,
    _unpack_qkv, _append, enqueue_attention_sublayer_integrated,
)
from llm_mojo.rope import enqueue_rope_apple_gpu
from llm_mojo.decoder_layer import decoder_mappings, enqueue_decoder_layer
from llm_mojo.mlp import MLPWeights, MLPWorkspace
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises


def _bits(buffer: DeviceBuffer[DType.bfloat16]) raises -> List[UInt16]:
    var result = List[UInt16](capacity=len(buffer))
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            result.append(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]))
    return result^


def test_fused_decode_exact_boundaries_and_protected_cache() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    print("QKV fusion device:", ctx.name(), "api:", ctx.api())
    var work = AttentionWorkspace(ctx, 1, 4096, 14, 2, 64, False, False)
    var reference = AttentionCache(ctx, 4096)
    var candidate = AttentionCache(ctx, 4096)
    reference.key.enqueue_fill(-19)
    reference.value.enqueue_fill(23)
    candidate.key.enqueue_fill(-19)
    candidate.value.enqueue_fill(23)
    var positions: List[Int] = [0, 1, 63, 64, 1023, 1024, 3968, 4095]
    for position in positions:
        var before_k = _bits(candidate.key)
        var before_v = _bits(candidate.value)
        with work.packed.map_to_host() as mapped:
            for i in range(1152):
                # Diverse exact BF16 bits, signs, subnormals and rounding cases.
                var bits = UInt16((i*1667 + position*61) % 0x4300)
                if i % 2:
                    bits |= 0x8000
                mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)
        with work.cosine.map_to_host() as mapped:
            for i in range(64):
                mapped.unsafe_ptr()[unsafe_offset=position*64+i] = Float32((i*7+position)%63-31).cast[DType.bfloat16]() / 32
        with work.sine.map_to_host() as mapped:
            for i in range(64):
                mapped.unsafe_ptr()[unsafe_offset=position*64+i] = Float32((i*13+position)%61-30).cast[DType.bfloat16]() / 32
        var packed = TileTensor(work.packed, row_major(1,1152))
        var raw_q = TileTensor(work.raw_query, row_major(1,896))
        var raw_k = TileTensor(work.raw_key, row_major(1,128))
        var raw_v = TileTensor(work.raw_value, row_major(1,128))
        ctx.enqueue_function[_unpack_qkv[type_of(packed.layout),type_of(raw_q.layout),type_of(raw_k.layout)]](
            packed,raw_q,raw_k,raw_v,Int32(1),Int32(896),Int32(128),grid_dim=9,block_dim=128)
        var c = TileTensor(work.cosine,row_major(4096,64))
        var s = TileTensor(work.sine,row_major(4096,64))
        enqueue_rope_apple_gpu(ctx,TileTensor(work.raw_query,row_major(1,14,64)),c,s,
                              TileTensor(work.query,row_major(1,14,64)),position)
        enqueue_rope_apple_gpu(ctx,TileTensor(work.raw_key,row_major(1,2,64)),c,s,
                              TileTensor(work.rotated_key,row_major(1,2,64)),position)
        var kr = TileTensor(work.rotated_key,row_major(1,128))
        var ck = TileTensor(reference.key,row_major(4096,128))
        var cv = TileTensor(reference.value,row_major(4096,128))
        ctx.enqueue_function[_append[type_of(kr.layout),type_of(raw_v.layout),type_of(ck.layout)]](
            kr,raw_v,ck,cv,Int32(1),Int32(128),Int32(position),grid_dim=1,block_dim=128)
        var expected_q = _bits(work.query)
        var expected_k = _bits(reference.key)
        var expected_v = _bits(reference.value)
        work.query.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
        # Poison skipped scratch: fused output must come from packed QKV.
        work.raw_query.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
        work.raw_key.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
        work.raw_value.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
        work.rotated_key.enqueue_fill(bitcast[DType.bfloat16](UInt16(0x7FC0)))
        candidate.length = position
        _enqueue_fused_decode_qkv(ctx,work,candidate)
        var actual_q = _bits(work.query)
        var actual_k = _bits(candidate.key)
        var actual_v = _bits(candidate.value)
        for i in range(896):
            assert_equal(actual_q[i],expected_q[i])
        for i in range(4096*128):
            assert_equal(actual_k[i],expected_k[i])
            assert_equal(actual_v[i],expected_v[i])
            if i < position*128 or i >= (position+1)*128:
                assert_equal(actual_k[i],before_k[i])
                assert_equal(actual_v[i],before_v[i])
        print("FUSION_EXACT position",position,"query",896,"cache",4096*128*2)


def test_fusion_rejects_non_decode_before_mutation() raises:
    var ctx = DeviceContext()
    var weights = AttentionWeights(ctx)
    var work = AttentionWorkspace(ctx,2,4,14,2,64,False,False)
    var cache = AttentionCache(ctx,4)
    cache.key.enqueue_fill(-17)
    cache.value.enqueue_fill(29)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](2*896)
    with assert_raises():
        _ = enqueue_attention_sublayer_integrated(ctx,weights,cache,work,
            TileTensor(x,row_major(2,896)),fuse_qkv=True)
    assert_equal(cache.length,0)
    var k = _bits(cache.key)
    for i in range(len(k)):
        assert_equal(k[i],bitcast[DType.uint16](Float32(-17).cast[DType.bfloat16]()))
    with assert_raises():
        _ = decoder_mappings(25,2)
    with assert_raises():
        _ = decoder_mappings(26,2)
    assert_equal(decoder_mappings(26,1),decoder_mappings(0,1))
    assert_equal(decoder_mappings(25,1),decoder_mappings(0,1))


def test_projection_study_rejects_mlp_width_before_mutation() raises:
    var ctx = DeviceContext()
    var aw = AttentionWeights(ctx)
    var mw = MLPWeights(ctx,896,12)
    var work = AttentionWorkspace(ctx,1,4,14,2,64,False,False)
    var mlp = MLPWorkspace(ctx,1,896,12)
    var cache = AttentionCache(ctx,4)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](896)
    x.enqueue_fill(1)
    aw.norm.enqueue_fill(1)
    aw.qkv.enqueue_fill(0)
    aw.bias.enqueue_fill(0)
    aw.output.enqueue_fill(0)
    work.cosine.enqueue_fill(1)
    work.sine.enqueue_fill(0)
    mw.norm.enqueue_fill(1)
    cache.key.enqueue_fill(-17)
    cache.value.enqueue_fill(-17)
    work.normalized.enqueue_fill(-17)
    work.output.enqueue_fill(-17)
    mlp.normalized.enqueue_fill(-17)
    mlp.output.enqueue_fill(-17)
    var before_k = _bits(cache.key)
    var before_v = _bits(cache.value)
    var before_a_norm = _bits(work.normalized)
    var before_a_output = _bits(work.output)
    var before_m_norm = _bits(mlp.normalized)
    var before_m_output = _bits(mlp.output)
    for variant in range(1,6):
        with assert_raises():
            _ = enqueue_decoder_layer(ctx,aw,cache,work,mw,mlp,
                TileTensor(x,row_major(1,896)),fuse_qkv=True,
                fuse_activation=True,fuse_residual_norm=True,decode_variant=variant)
        ctx.synchronize()
        assert_equal(cache.length,0)
        var after_k = _bits(cache.key)
        var after_v = _bits(cache.value)
        var after_a_norm = _bits(work.normalized)
        var after_a_output = _bits(work.output)
        var after_m_norm = _bits(mlp.normalized)
        var after_m_output = _bits(mlp.output)
        for i in range(len(before_k)):
            assert_equal(after_k[i],before_k[i])
            assert_equal(after_v[i],before_v[i])
        for i in range(896):
            assert_equal(after_a_norm[i],before_a_norm[i])
            assert_equal(after_a_output[i],before_a_output[i])
            assert_equal(after_m_norm[i],before_m_norm[i])
            assert_equal(after_m_output[i],before_m_output[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
