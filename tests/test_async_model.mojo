"""Exact staged Qwen parity and bounded two-context ownership on actual Metal.

The existing independent decoder fixture supplies three nontrivial layers.
Compare identical reference operations submitted synchronously with staged
prefill, mixed device-token chaining and three-ticket bank reuse. These tests
establish numerical and ownership acceptance, not checkpoint performance.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.layers.decoder_layer import DECODER_MIXED
from llm_mojo.models.qwen2.model import VOCABULARY
from llm_mojo.models.qwen2.plan import configured_plan
from llm_mojo.models.qwen2.runner import QwenAsyncRunner
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool
from decoder_layer_support import decoder_support
from test_mixed_model import CASE, CONTEXT, LAYERS, POISON, _model, _snapshot, _equals, _ids, _table


def _sources(rows: Int) -> List[Int]:
    var result = List[Int](capacity=rows)
    for _ in range(rows):
        result.append(-1)
    return result^


def _mixed(first: Int) -> StepBatch:
    return StepBatch([first,13,14,15,16,17],[31,0,1,2,3,4],[0,1,6],1,[32,5],3,
        [8,4,0,7,3,11],[8*32+31,7*32,7*32+1,7*32+2,7*32+3,7*32+4],[0,5])


def _decode(first: Int, second: Int) -> StepBatch:
    return StepBatch([first,second],[32,5],[0,1,2],2,[33,6],3,
        [8,4,0,7,3,11],[4*32,7*32+5],[0,1])


def _exact[HEAD_MAJOR: Bool](ctx: DeviceContext) raises:
    var asynchronous = QwenAsyncRunner(ctx,_model(ctx,3))
    var reference = _model(ctx,3)
    var actual = KVPool(ctx,12,32,asynchronous.model.kv_geometry(),HEAD_MAJOR)
    var expected = KVPool(ctx,12,32,reference.kv_geometry(),HEAD_MAJOR)
    actual.storage.enqueue_fill(bitcast[DType.bfloat16](POISON))
    expected.storage.enqueue_fill(bitcast[DType.bfloat16](POISON))
    var prefill = StepBatch.sequence(_ids(0,0,31),0,_table(0),32)
    reference.forward(ctx,prefill,expected,configured_plan(DECODER_MIXED,31,31))
    var first = reference.greedy_tokens(ctx)
    var mixed = _mixed(first[0])
    reference.forward(ctx,mixed,expected,configured_plan(DECODER_MIXED,6,32,2))
    var second = reference.greedy_tokens(ctx)
    var decode = _decode(second[0],second[1])
    reference.forward(ctx,decode,expected,configured_plan(DECODER_MIXED,2,33,2))
    var third = reference.greedy_tokens(ctx)
    var logits = _snapshot(reference.logits,2*VOCABULARY)
    var pool = _snapshot(expected.storage,len(expected.storage))
    var ticket0 = asynchronous.submit(prefill,_sources(31),-1,actual)
    var ticket1 = asynchronous.submit(_mixed(0),[0,-1,-1,-1,-1,-1],ticket0,actual)
    assert_equal(asynchronous.pending(),2)
    assert_equal(actual.written[8],32)
    assert_equal(actual.written[7],5)
    var read0 = asynchronous.collect(ticket0)
    assert_equal(read0[0],first[0])
    # Bank 0 is reused while bank 1 can still execute. Its prefix wait captures
    # ticket 1 and cannot acquire a dependency on subsequently submitted work.
    var ticket2 = asynchronous.submit(_decode(0,0),[0,1],ticket1,actual)
    assert_equal(ticket0,0)
    assert_equal(ticket1,1)
    assert_equal(ticket2,2)
    assert_equal(asynchronous.pending(),2)
    var read1 = asynchronous.collect(ticket1)
    var read2 = asynchronous.collect(ticket2)
    for index in range(2):
        assert_equal(read1[index],second[index])
        assert_equal(read2[index],third[index])
    assert_equal(asynchronous.pending(),0)
    _equals(logits,0,asynchronous.model.logits,2*VOCABULARY,"staged final logits")
    _equals(pool,0,actual.storage,len(actual.storage),"staged full guarded KV")
    assert_equal(asynchronous.model.submitted_rows,(31+6+2)*LAYERS)
    assert_equal(actual.written[4],1)
    assert_equal(actual.written[7],6)


def test_async_staged_mixed_tokens_logits_full_kv_and_three_ticket_reuse() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    _exact[False](ctx)
    _exact[True](ctx)


def test_async_unfinished_prefill_and_singleflight_compatibility() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    var runner = QwenAsyncRunner(ctx,_model(ctx,1))
    var reference = _model(ctx,1)
    var actual = KVPool(ctx,3,32,runner.model.kv_geometry())
    var expected = KVPool(ctx,3,32,reference.kv_geometry())
    var first = StepBatch.sequence([1,2,3],0,[0,1,2],32)
    first.logits_rows = List[Int]()
    var second = StepBatch.sequence([4,5],3,[0,1,2],32)
    var ticket0 = runner.submit(first,[-1,-1,-1],-1,actual)
    var ticket1 = runner.submit(second,[-1,-1],-1,actual)
    assert_equal(len(runner.collect(ticket0)),0)
    var selected = runner.collect(ticket1)
    reference.forward(ctx,first,expected,configured_plan(DECODER_MIXED,3,3))
    assert_equal(len(reference.greedy_tokens(ctx)),0)
    reference.forward(ctx,second,expected,configured_plan(DECODER_MIXED,2,5))
    var token = reference.greedy_tokens(ctx)
    assert_equal(selected[0],token[0])
    var third = StepBatch.sequence(selected,5,[0,1,2],32)
    var final = runner.execute(third,actual)
    reference.forward(ctx,third,expected,configured_plan(DECODER_MIXED,1,6))
    var expected_final = reference.greedy_tokens(ctx)
    assert_equal(final[0],expected_final[0])
    runner.drain()
    assert_equal(runner.pending(),0)
    assert_equal(runner.next_ticket,3)
    runner.reset_clock()
    assert_equal(runner.next_ticket,0)
    assert_equal(runner.slots[0].ticket,-1)
    assert_equal(runner.slots[1].ticket,-1)


def test_async_bad_sources_depth_fifo_and_clock_reject_before_state_changes() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    var runner = QwenAsyncRunner(ctx,_model(ctx,1))
    var actual = KVPool(ctx,3,32,runner.model.kv_geometry())
    var first = StepBatch.sequence([1,2],0,[0,1,2],32)
    with assert_raises():
        _ = runner.submit(first,[-1],-1,actual)
    with assert_raises():
        _ = runner.submit(first,[-2,-1],-1,actual)
    with assert_raises():
        _ = runner.submit(first,[-1,-1],0,actual)
    with assert_raises():
        _ = runner.submit(first,[0,-1],0,actual)
    assert_equal(runner.next_ticket,0)
    assert_equal(runner.model.submitted_rows,0)
    assert_equal(actual.written[0],0)
    var ticket0 = runner.submit(first,[-1,-1],-1,actual)
    var second = StepBatch.sequence([0],2,[0,1,2],32)
    with assert_raises():
        _ = runner.submit(second,[1],ticket0,actual)
    with assert_raises():
        _ = runner.submit(second,[0],ticket0+1,actual)
    assert_equal(runner.next_ticket,1)
    assert_equal(actual.written[0],2)
    var ticket1 = runner.submit(second,[0],ticket0,actual)
    var third = StepBatch.sequence([0],3,[0,1,2],32)
    with assert_raises():
        _ = runner.submit(third,[0],ticket1,actual)
    with assert_raises():
        _ = runner.collect(ticket1)
    with assert_raises():
        runner.reset_clock()
    with assert_raises():
        _ = runner.execute(third,actual)
    assert_equal(runner.model.valid,True)
    assert_equal(runner.pending(),2)
    _ = runner.collect(ticket0)
    _ = runner.collect(ticket1)
    assert_equal(runner.pending(),0)
    with assert_raises():
        _ = runner.collect(ticket0)
    assert_equal(runner.model.valid,True)


def test_async_nonfinite_result_drains_dependent_work_and_rejects_future_submission() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    var model = _model(ctx,1)
    # Token 3's vocabulary logit is nonfinite even though the prompt uses
    # healthy embedding rows. The collector must reject every selected ID.
    with model.embedding.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=3*896] = bitcast[DType.bfloat16](UInt16(0x7FC1))
    var runner = QwenAsyncRunner(ctx,model^)
    var actual = KVPool(ctx,3,32,runner.model.kv_geometry())
    var first = StepBatch.sequence([1,2],0,[0,1,2],32)
    var second = StepBatch.sequence([0],2,[0,1,2],32)
    var ticket0 = runner.submit(first,[-1,-1],-1,actual)
    _ = runner.submit(second,[0],ticket0,actual)
    with assert_raises():
        _ = runner.collect(ticket0)
    assert_equal(runner.model.valid,False)
    assert_equal(runner.pending(),0)
    # No physical storage reuse is permitted until both contexts drained.
    actual.truncate_table([0,1,2],0)
    assert_equal(actual.written[0],0)
    with assert_raises():
        _ = runner.submit(first,[-1,-1],-1,actual)
    runner.drain()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
