"""Existing kernel combinations through complete decoder numerical gates."""
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.decoder_layer import decoder_mappings, enqueue_decoder_layer, enqueue_decoder_layer_configuration
from llm_mojo.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace
from llm_mojo.mlp import MLPWeights, MLPWorkspace
from decoder_layer_support import decoder_support, decoder_snapshot, exact_decoder
from test_decoder_layer import _case, _asynchronous_decoder, _poison, test_discriminating_negative_controls


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
