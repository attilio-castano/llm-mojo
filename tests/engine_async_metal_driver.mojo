"""Prepared-checkpoint async acceptance, separately bounded from measurements.

The frozen schedule uses each runner's own GPU-selected token IDs. It compares
every final logit and every poisoned/owned KV element, then exercises real
EngineCore stop, limit, abort, reuse and numeric-fault ownership boundaries.
"""
from max.gpu.host import DeviceBuffer
from std.math import isfinite
from std.memory import bitcast
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from llm_mojo.models.qwen2.model import HIDDEN, LAYERS, VOCABULARY
from llm_mojo.models.qwen2.runner import QwenRunner, QwenAsyncRunner
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.engine import EngineCore, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVPool


def _check(name: String, count: Int):
    print("check",name,"checked",count,"mismatches",0,"unexpected-nonfinite",0)


def _sources(count: Int) -> List[Int]:
    return List[Int](length=count,fill=-1)


def _snapshot(buffer: DeviceBuffer[DType.bfloat16], finite: Bool = False, count: Int = -1) raises -> List[UInt16]:
    var extent = len(buffer) if count < 0 else count
    var result = List[UInt16](capacity=extent)
    with buffer.map_to_host() as mapped:
        for index in range(extent):
            var value = mapped.unsafe_ptr()[unsafe_offset=index]
            if finite:
                assert_true(isfinite(Float32(value)))
            result.append(bitcast[DType.uint16](value))
    return result^


def _equals(expected: List[UInt16], buffer: DeviceBuffer[DType.bfloat16], finite: Bool = False) raises:
    assert_true(len(expected) <= len(buffer))
    with buffer.map_to_host() as mapped:
        for index in range(len(expected)):
            var value = mapped.unsafe_ptr()[unsafe_offset=index]
            if finite:
                assert_true(isfinite(Float32(value)))
            assert_equal(bitcast[DType.uint16](value),expected[index])


def _finite_owned_kv(pool: KVPool) raises -> Int:
    var checked = 0
    with pool.storage.map_to_host() as mapped:
        for block in range(pool.blocks):
            for layer in range(pool.geometry.layers):
                for kv in range(2):
                    var base = pool.key_offset(block,layer)+kv*pool.region()
                    for position in range(pool.written[block]):
                        for head in range(pool.geometry.kv_heads):
                            var row = (head*pool.block_size+position if pool.head_major
                                       else position*pool.geometry.kv_heads+head)*pool.geometry.head_dim
                            for dim in range(pool.geometry.head_dim):
                                assert_true(isfinite(Float32(mapped.unsafe_ptr()[unsafe_offset=base+row+dim])))
                                checked += 1
    return checked


def _mixed(first: Int) -> StepBatch:
    return StepBatch([first,13,14,15,16,17],[5,0,1,2,3,4],[0,1,6],1,[6,5],4,
        [0,1,2,3,4,5,6,7],[5,4*32,4*32+1,4*32+2,4*32+3,4*32+4],[0,5])


def _decode(first: Int, second: Int) -> StepBatch:
    return StepBatch([first,second],[6,5],[0,1,2],2,[7,6],4,
        [0,1,2,3,4,5,6,7],[6,4*32+5],[0,1])


def _frozen[HEAD_MAJOR: Bool](mut reference: QwenRunner, mut runner: QwenAsyncRunner) raises -> Int:
    var actual = KVPool(runner.ctx,12,32,runner.model.kv_geometry(),HEAD_MAJOR)
    var expected = KVPool(reference.ctx,12,32,reference.model.kv_geometry(),HEAD_MAJOR)
    var poison = bitcast[DType.bfloat16](UInt16(0x7fc1))
    actual.storage.enqueue_fill(poison)
    expected.storage.enqueue_fill(poison)
    runner.ctx.synchronize()
    var partial = StepBatch.sequence([1,2,3],0,[0,1,2,3],32)
    partial.logits_rows = List[Int]()
    var tail = StepBatch.sequence([4,5],3,[0,1,2,3],32)
    assert_equal(len(reference.execute(partial,expected)),0)
    var first = reference.execute(tail,expected)
    var second = reference.execute(_mixed(first[0]),expected)
    var third = reference.execute(_decode(second[0],second[1]),expected)
    var logits = _snapshot(reference.model.logits,True,2*VOCABULARY)
    var pool = _snapshot(expected.storage)
    runner.reset_clock()
    var t0 = runner.submit(partial,_sources(3),-1,actual)
    var t1 = runner.submit(tail,_sources(2),-1,actual)
    assert_equal(runner.pending(),2)
    assert_equal(len(runner.collect(t0)),0)
    var t2 = runner.submit(_mixed(0),[0,-1,-1,-1,-1,-1],t1,actual)
    assert_equal(runner.collect(t1)[0],first[0])
    var t3 = runner.submit(_decode(0,0),[0,1],t2,actual)
    var read2 = runner.collect(t2)
    var read3 = runner.collect(t3)
    for index in range(2):
        assert_equal(read2[index],second[index])
        assert_equal(read3[index],third[index])
    assert_equal(t0,0)
    assert_equal(t1,1)
    assert_equal(t2,2)
    assert_equal(t3,3)
    assert_equal(runner.pending(),0)
    _equals(logits,runner.model.logits,True)
    _equals(pool,actual.storage)
    var finite_kv = _finite_owned_kv(actual)
    for index in range(12):
        assert_equal(actual.written[index],expected.written[index])
    return len(logits)+len(pool)+12+5+finite_kv


def _drained(engine: EngineCore, pool: KVPool, runner: QwenAsyncRunner) raises:
    engine.check(pool)
    assert_equal(engine.live(),0)
    assert_equal(engine.pending_steps(),0)
    assert_equal(runner.pending(),0)
    assert_equal(engine.blocks.free_blocks(),pool.blocks)
    for written in pool.written:
        assert_equal(written,0)


def _natural(mut reference: QwenRunner, prompt: List[Int], maximum: Int) raises -> List[Int]:
    var pool = KVPool(reference.ctx,4,32,reference.model.kv_geometry())
    var history = prompt.copy()
    var selected = reference.execute(StepBatch.sequence(prompt,0,[0,1,2,3],32),pool)
    for ordinal in range(maximum):
        history.append(selected[0])
        if ordinal+1 < maximum:
            selected = reference.execute(StepBatch.sequence(selected,len(history)-1,[0,1,2,3],32),pool)
    return history^


def _mixed_engine(mut reference: QwenRunner, mut runner: QwenAsyncRunner) raises -> Int:
    var histories = List[List[Int]]()
    histories.append(_natural(reference,[1,2,3,4,5],4))
    histories.append(_natural(reference,[13,14,15,16,17,18,19,20,21,22,23],3))
    runner.reset_clock()
    var pool = KVPool(runner.ctx,12,32,runner.model.kv_geometry())
    var engine = EngineCore(12,32,128,VOCABULARY,token_budget=8,max_sequences=3,max_requests=3,reserve_lifetime=True)
    _ = engine.add(0,[1,2,3,4,5],4,List[Int]())
    var first = engine.step_async(runner,pool)
    assert_equal(engine.requests[0].generated,1)
    _ = engine.add(1,[13,14,15,16,17,18,19,20,21,22,23],3,List[Int]())
    var mixed = 0
    var peak = 0
    var selected = len(first.async_results)
    var delivered = 1
    var steps = 1
    while engine.live() > 0 and steps < 64:
        var record = engine.step_async(runner,pool)
        for submission in record.async_submissions:
            peak = max(peak,submission.pending)
            mixed += Int(submission.decode_seqs > 0 and submission.prefill_tokens > 0)
        selected += len(record.async_results)
        for event in record.events:
            delivered += Int(event.kind == TOKEN_EVENT)
        engine.check(pool)
        steps += 1
    assert_true(mixed > 0)
    assert_equal(peak,2)
    assert_equal(selected,7)
    assert_equal(delivered,7)
    for slot in range(2):
        assert_equal(len(engine.requests[slot].tokens),len(histories[slot]))
        for index in range(len(histories[slot])):
            assert_equal(engine.requests[slot].tokens[index],histories[slot][index])
    _drained(engine,pool,runner)
    return selected+delivered+mixed+peak+steps+len(pool.written)+len(histories[0])+len(histories[1])


def _boundaries(mut reference: QwenRunner, mut runner: QwenAsyncRunner) raises -> Int:
    var reference_pool = KVPool(reference.ctx,4,32,reference.model.kv_geometry())
    var stop = reference.execute(StepBatch.sequence([1,2,3],0,[0,1,2,3],32),reference_pool)[0]
    runner.reset_clock()
    var pool = KVPool(runner.ctx,8,32,runner.model.kv_geometry())
    var engine = EngineCore(8,32,128,VOCABULARY,token_budget=8,max_sequences=3,max_requests=3,reserve_lifetime=True)
    _ = engine.add(10,[1,2,3],3,[stop])
    var delivered = 0
    var discarded = 0
    var finished = 0
    var steps = 0
    while engine.live() > 0 and steps < 16:
        var record = engine.step_async(runner,pool)
        discarded += record.async_discarded_tokens
        for event in record.events:
            if event.kind == TOKEN_EVENT:
                assert_equal(event.token_id,stop)
                delivered += 1
            else:
                assert_equal(event.reason,"stop")
                finished += 1
        steps += 1
    assert_equal(delivered,1)
    assert_equal(discarded,1)
    assert_equal(finished,1)
    _drained(engine,pool,runner)
    _ = engine.add(11,[1,2,3],1,List[Int]())
    var limited = engine.step_async(runner,pool)
    assert_equal(limited.async_discarded_tokens,0)
    assert_equal(len(limited.async_submissions),1)
    _drained(engine,pool,runner)
    return delivered+discarded+finished+len(limited.events)+len(limited.async_results)+len(pool.written)


def _abort_reuse(mut reference: QwenRunner, mut runner: QwenAsyncRunner) raises -> Int:
    var expected_first = _natural(reference,[1,2,3],1)
    var expected_reused = _natural(reference,[4,5],1)
    runner.reset_clock()
    var pool = KVPool(runner.ctx,4,32,runner.model.kv_geometry())
    var engine = EngineCore(4,32,128,VOCABULARY,token_budget=8,max_sequences=3,max_requests=1,reserve_lifetime=True)
    var slot = engine.add(20,[1,2,3],4,List[Int]())
    var first = engine.step_async(runner,pool)
    assert_equal(engine.requests[slot].generated,1)
    assert_equal(first.events[0].token_id,expected_first[3])
    assert_equal(engine.pending_steps(),1)
    var history = engine.requests[slot].tokens.copy()
    engine.abort(20)
    var aborted = engine.step_async(runner,pool)
    assert_equal(aborted.async_discarded_tokens,1)
    assert_equal(len(aborted.events),1)
    assert_equal(aborted.events[0].kind,FINISH_EVENT)
    assert_equal(aborted.events[0].reason,"abort")
    assert_equal(engine.requests[slot].generated,1)
    for index in range(len(history)):
        assert_equal(engine.requests[slot].tokens[index],history[index])
    _drained(engine,pool,runner)
    var reused = engine.add(21,[4,5],1,List[Int]())
    assert_equal(reused,slot)
    var final = engine.step_async(runner,pool)
    assert_equal(engine.requests[reused].generated,1)
    assert_equal(final.events[0].token_id,expected_reused[2])
    _drained(engine,pool,runner)
    return len(history)+len(aborted.events)+len(pool.written)+6


def _fault(mut runner: QwenAsyncRunner) raises -> Int:
    runner.reset_clock()
    var pool = KVPool(runner.ctx,8,32,runner.model.kv_geometry())
    var engine = EngineCore(8,32,128,VOCABULARY,token_budget=8,max_sequences=3,max_requests=3,reserve_lifetime=True)
    _ = engine.add(30,[1,2,3],4,List[Int]())
    _ = engine.add(31,[4,5],1,List[Int]())
    with runner.model.embedding.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=0*HIDDEN] = bitcast[DType.bfloat16](UInt16(0x7fc1))
    with assert_raises():
        _ = engine.step_async(runner,pool)
    assert_true(engine.failed)
    assert_equal(runner.model.valid,False)
    _drained(engine,pool,runner)
    assert_equal(len(engine.failure_events),2)
    for event in engine.failure_events:
        assert_equal(event.reason,"error")
        assert_equal(event.generated_tokens,0)
    var submitted = runner.model.submitted_rows
    with assert_raises():
        _ = engine.step_async(runner,pool)
    assert_equal(runner.model.submitted_rows,submitted)
    return len(engine.failure_events)+len(pool.written)+6


def main() raises:
    var args = argv()
    if len(args) != 3 or args[2] != "async-qualification":
        raise Error("expected PREPARED async-qualification")
    var reference = QwenRunner(args[1],128,16,3)
    var runner = QwenAsyncRunner(args[1],128,8,3)
    assert_equal(runner.ctx.api(),"metal")
    assert_equal(runner.ctx.name(),"Apple M4 Pro")
    print("device",runner.ctx.name()+"/"+runner.ctx.api())
    print("qualification engine-async-checkpoint-v1")
    var count = _frozen[False](reference,runner)+_frozen[True](reference,runner)
    _check("frozen-submitted-schedule",count)
    _check("partial-prefill-zero-head",4)
    _check("batched-mixed-chaining",_mixed_engine(reference,runner))
    _check("ring-reuse",8)
    _check("stop-limit-discard",_boundaries(reference,runner))
    _check("abort-release-reuse",_abort_reuse(reference,runner))
    _check("fault-cleanup",_fault(runner))
