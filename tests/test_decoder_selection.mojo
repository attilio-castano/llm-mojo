"""Existing kernel combinations through complete decoder numerical gates."""
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_raises
from std.memory import bitcast
from llm_mojo.decoder_layer import decoder_mappings, enqueue_decoder_layer, enqueue_decoder_layer_configuration, DecoderCache, decoder_policy_configuration, enqueue_decoder_layer_policy
from llm_mojo.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace
from llm_mojo.mlp import MLPWeights, MLPWorkspace
from decoder_layer_support import decoder_support, decoder_snapshot, exact_decoder, load_decoder, poison_decoder, check_decoder_active
from test_decoder_layer import _case, _asynchronous_decoder, _poison, _load_weights, test_discriminating_negative_controls


def _policy_cache[DETERMINISTIC: Bool](reuse_layers: Int) raises:
    var support = decoder_support()
    var name = String("h896_i4864_nq14_nk2_d64_t17_s4001_base")
    support.verify_case(name)
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    with assert_raises(contains="reuse"):
        var invalid_cache = DecoderCache[DETERMINISTIC](ctx,18,2)
    var cache = DecoderCache[DETERMINISTIC](ctx,18,reuse_layers)
    var aw = AttentionWeights(ctx)
    var mw = MLPWeights(ctx)
    var a = AttentionWorkspace(ctx,18,18,14,2,64,False,False,cache.prefill_splits())
    var m = MLPWorkspace(ctx,18)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](17*896)
    load_decoder(x,name,"input_X",0,17*896)
    _load_weights(aw,mw,a,name,17)
    poison_decoder(cache.storage.key,0)
    poison_decoder(cache.storage.value,0)
    var p = 0
    for r in [16,1]:
        var variant = decoder_policy_configuration(DETERMINISTIC,r,p+r,reuse_layers)
        support.configure(name,variant,"policy_api","policy_api")
        _poison(a,m,r)
        var prefix_k = decoder_snapshot(cache.storage.key,p*128)
        var prefix_v = decoder_snapshot(cache.storage.value,p*128)
        var route = enqueue_decoder_layer_policy(ctx,aw,cache,a,mw,m,
            TileTensor(x.unsafe_ptr().unsafe_offset(p*896),row_major(r,896)))
        var gqa = Int(decoder_mappings(variant,r)[0])
        assert_equal(route,11 if gqa == 5 else (4 if r == 1 else 6+gqa))
        assert_equal(cache.storage.length,p+r)
        exact_decoder(cache.storage.key,prefix_k,"policy cache key prefix")
        exact_decoder(cache.storage.value,prefix_v,"policy cache value prefix")
        check_decoder_active(a.projected,name,"B_att","full_slice",p,r,896)
        check_decoder_active(a.output,name,"Z","full_slice",p,r,896)
        check_decoder_active(m.down,name,"B_mlp","full_slice",p,r,896)
        check_decoder_active(m.output,name,"Y","full_slice",p,r,896)
        p += r
        var saved_k = decoder_snapshot(cache.storage.key,len(cache.storage.key))
        var saved_y = decoder_snapshot(m.output,len(m.output))
        for bad_rows in [0,19]:
            with assert_raises(contains="invalid"):
                _ = enqueue_decoder_layer_policy(ctx,aw,cache,a,mw,m,
                    TileTensor(x,row_major(bad_rows,896)))
            assert_equal(cache.storage.length,p)
            exact_decoder(cache.storage.key,saved_k,"policy invalid preserves cache")
            exact_decoder(m.output,saved_y,"policy invalid preserves output")
    with cache.storage.key.map_to_host() as key:
        with cache.storage.value.map_to_host() as value:
            for i in range(17*128,18*128):
                assert_equal(bitcast[DType.uint16](key.unsafe_ptr()[unsafe_offset=i]),UInt16(0x42f6))
                assert_equal(bitcast[DType.uint16](value.unsafe_ptr()[unsafe_offset=i]),UInt16(0x42f6))
    cache.reset(ctx)
    assert_equal(cache.storage.length,0)
    # A different policy creates a different cache type and an empty prefix.
    var other = DecoderCache[not DETERMINISTIC](ctx,18)
    assert_equal(other.storage.length,0)


def test_policy_cache_owner_and_failed_enqueue() raises:
    for reuse_layers in [1,24]:
        _policy_cache[False](reuse_layers)
        _policy_cache[True](reuse_layers)


def test_policy_lookup_exact_configuration_ids() raises:
    var support = decoder_support()
    for spec in support.policy_lookup_cases():
        var deterministic = Int(py=spec[0]) != 0
        assert_equal(decoder_policy_configuration(deterministic,
            Int(py=spec[1]),Int(py=spec[2]),Int(py=spec[3])),Int(py=spec[4]))


def test_selection_cases() raises:
    var support = decoder_support()
    var cases = support.cases()
    var variants = support.selection_variants()
    for spec in cases:
        if Int(py=spec[3]) != 14:
            continue
        for item in variants:
            var variant = Int(py=item)
            _case(String(py=spec[0]),Int(py=spec[1]),Int(py=spec[2]),Int(py=spec[3]),
                  Int(py=spec[4]),Int(py=spec[5]),Int(py=spec[6]),variant,True)
    support.result_summary()


def test_selection_async() raises:
    var support = decoder_support()
    for item in support.selection_variants():
        _asynchronous_decoder(Int(py=item))


def test_selection_preflight() raises:
    var support = decoder_support()
    support.configure("behavior",0,"invalid","preflight")
    var ctx = DeviceContext()
    var aw = AttentionWeights(ctx)
    var mw = MLPWeights(ctx)
    var a = AttentionWorkspace(ctx,17,17,14,2,64,False,False)
    var m = MLPWorkspace(ctx,17)
    var cache = AttentionCache(ctx,17)
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](17*896)
    _poison(a,m,0)
    cache.key.enqueue_fill(0)
    cache.value.enqueue_fill(0)
    var before_k = decoder_snapshot(cache.key,len(cache.key))
    var before_v = decoder_snapshot(cache.value,len(cache.value))
    var before_n = decoder_snapshot(a.normalized,len(a.normalized))
    var before_z = decoder_snapshot(a.output,len(a.output))
    var before_y = decoder_snapshot(m.output,len(m.output))
    with assert_raises(contains="configuration"):
        _ = enqueue_decoder_layer_configuration(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(17,896)),99)
    with assert_raises(contains="split"):
        _ = enqueue_decoder_layer_configuration(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(17,896)),2)
    with assert_raises(contains="decode"):
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(17,896)),8)
    with assert_raises(contains="mappings"):
        _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(17,896)),7,True,4,1)
    a.prefill_splits = 8
    with assert_raises(contains="smaller"):
        _ = enqueue_decoder_layer_configuration(ctx,aw,cache,a,mw,m,TileTensor(xb,row_major(17,896)),3)
    assert_equal(cache.length,0)
    ctx.synchronize()
    exact_decoder(cache.key,before_k,"selection invalid cache key")
    exact_decoder(cache.value,before_v,"selection invalid cache value")
    exact_decoder(a.normalized,before_n,"selection invalid first dispatch")
    exact_decoder(a.output,before_z,"selection invalid attention output")
    exact_decoder(m.output,before_y,"selection invalid MLP output")


def test_selection_negative_controls() raises:
    test_discriminating_negative_controls()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
