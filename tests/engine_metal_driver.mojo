"""Explicit prepared-checkpoint engine acceptance checks.

This changes a private resident tied-head weight after a successful step. It
does not change prepared files or simulate device loss. The real model submits
the next Metal step, GPU argmax detects a nonfinite logit, and QwenRunner must
drain before EngineCore releases logical ownership and rejects future work.
The optional admission check compares natural greedy histories under pressure,
then checks reserved-request cancellation and numeric fault cleanup.
"""
from std.memory import bitcast
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from llm_mojo.layers.decoder_layer import DECODER_MIXED
from llm_mojo.models.qwen2.model import HIDDEN, LAYERS, VOCABULARY
from llm_mojo.models.qwen2.runner import QwenRunner
from llm_mojo.serving.engine import EngineCore, PREFILL, DECODE, WAITING, FINISHED, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVPool


def _assert_list(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def numeric_fault_cleanup(path: String, reserve_lifetime: Bool = False) raises:
    var runner = QwenRunner(path, 128, 8, 3)
    print("engine device", runner.ctx.name(), "backend", runner.ctx.api())
    assert_equal(runner.ctx.api(), "metal")
    var pool = KVPool(runner.ctx, 4, 32, runner.model.kv_geometry())
    var engine = EngineCore(4, 32, 128, VOCABULARY, token_budget=8,
                            max_sequences=3, max_requests=3, reserve_lifetime=reserve_lifetime)
    var a = engine.add(0, [1, 2, 3], 3, List[Int]())
    var first = engine.step(runner, pool)
    assert_equal(first.total_tokens, 3)
    assert_equal(first.prefill_tokens, 3)
    assert_equal(len(first.events), 1)
    assert_equal(first.events[0].kind, TOKEN_EVENT)
    assert_equal(engine.requests[a].state, DECODE)
    assert_equal(engine.requests[a].generated, 1)
    assert_equal(runner.model.valid, True)
    assert_equal(runner.model.submitted_rows, 3 * LAYERS)
    var history = engine.requests[a].tokens.copy()
    var prompt: List[Int] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    if reserve_lifetime:
        # A lifetime reservation includes a second, unwritten physical block.
        prompt = List[Int]()
        for i in range(34):
            prompt.append(100 + i)
    var b = engine.add(1, prompt, 2, List[Int]())
    var c = engine.add(2, [20, 21], 2, List[Int]())
    assert_equal(engine.requests[b].state, WAITING)
    assert_equal(engine.requests[c].state, WAITING)
    # The tied vocabulary weight is used by both embedding and the LM head.
    # Choose a row absent from the pending input so the deliberate NaN reaches
    # the vocabulary projection without corrupting the decoder inputs.
    var poisoned_token = 0 if history[len(history) - 1] != 0 else 15
    with runner.model.embedding.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=poisoned_token * HIDDEN] = bitcast[DType.bfloat16](UInt16(0x7fc0))
    var submitted = runner.model.submitted_rows
    with assert_raises():
        _ = engine.step(runner, pool)
    # The mixed forward enqueued one decode and seven prompt rows through all
    # 24 layers before argmax readback rejected the deliberately nonfinite head.
    assert_equal(runner.model.last_route.configuration, DECODER_MIXED)
    assert_equal(runner.model.submitted_rows - submitted, 8 * LAYERS)
    assert_equal(runner.model.sampled_rows, 1)
    assert_equal(runner.model.valid, False)
    assert_true(engine.failed)
    assert_equal(engine.live(), 0)
    assert_equal(engine.blocks.free_blocks(), 4)
    assert_equal(len(engine.failure_events), 3)
    var finish_counts = List[Int](length=3, fill=0)
    for event in engine.failure_events:
        assert_equal(event.kind, FINISH_EVENT)
        assert_equal(event.reason, "error")
        assert_true(event.request_id >= 0 and event.request_id < 3)
        finish_counts[event.request_id] += 1
        assert_equal(event.generated_tokens, 1 if event.request_id == 0 else 0)
    for count in finish_counts:
        assert_equal(count, 1)
    for slot in [a, b, c]:
        assert_equal(engine.requests[slot].state, FINISHED)
        assert_equal(engine.requests[slot].reason, "error")
        assert_equal(engine.requests[slot].sequence, -1)
    _assert_list(engine.requests[a].tokens, history)
    assert_equal(engine.requests[a].generated, 1)
    assert_equal(engine.requests[b].generated, 0)
    assert_equal(engine.requests[c].generated, 0)
    for written in pool.written:
        assert_equal(written, 0)
    engine.check(pool)
    runner.ctx.synchronize()
    # Invalid engines retain their terminal events and cannot deliver another
    # token or enqueue another forward after ownership has been released.
    var failed_rows = runner.model.submitted_rows
    var failed_step = engine.step_id
    with assert_raises():
        _ = engine.step(runner, pool)
    with assert_raises():
        _ = engine.add(3, [30], 1, List[Int]())
    assert_equal(runner.model.submitted_rows, failed_rows)
    assert_equal(engine.step_id, failed_step)
    assert_equal(len(engine.failure_events), 3)
    engine.check(pool)
    print("engine numeric-fault cleanup passed: submitted rows", failed_rows,
          "terminal requests", len(engine.failure_events), "free blocks", engine.blocks.free_blocks())
    if reserve_lifetime:
        print("engine reserved numeric-fault cleanup passed: unwritten reservation tail released")


def _natural_histories(mut runner: QwenRunner, reserve_lifetime: Bool) raises -> List[List[Int]]:
    runner.model.reset(runner.ctx)
    var pool = KVPool(runner.ctx, 3, 32, runner.model.kv_geometry())
    var engine = EngineCore(3, 32, 128, VOCABULARY, token_budget=8,
                            max_sequences=3, max_requests=3, reserve_lifetime=reserve_lifetime)
    var a = List[Int]()
    var b = List[Int]()
    for i in range(30):
        a.append(i + 1)
    for i in range(38):
        b.append(101 + i)
    _ = engine.add(0, a, 7, List[Int]())
    _ = engine.add(1, b, 3, List[Int]())
    _ = engine.add(2, [301, 302, 303, 304, 305], 3, List[Int]())
    var delivered = List[Int](length=3, fill=0)
    var finishes = List[Int](length=3, fill=0)
    var steps = 0
    var preemptions = 0
    var computed_rows = 0
    while engine.live() > 0 and steps < 1000:
        var record = engine.step(runner, pool)
        preemptions += record.preempted
        computed_rows += record.total_tokens
        for event in record.events:
            var id = event.request_id
            assert_true(id >= 0 and id < 3)
            if event.kind == TOKEN_EVENT:
                assert_equal(finishes[id], 0)
                delivered[id] += 1
                assert_equal(event.generated_tokens, delivered[id])
            else:
                assert_equal(event.kind, FINISH_EVENT)
                assert_equal(event.reason, "length")
                finishes[id] += 1
        engine.check(pool)
        steps += 1
    assert_equal(engine.live(), 0)
    assert_equal(engine.blocks.free_blocks(), 3)
    assert_equal(delivered[0], 7)
    assert_equal(delivered[1], 3)
    assert_equal(delivered[2], 3)
    for finish in finishes:
        assert_equal(finish, 1)
    if reserve_lifetime:
        assert_equal(preemptions, 0)
        assert_equal(computed_rows, 83)
    else:
        assert_true(preemptions > 0)
    var histories = List[List[Int]](capacity=3)
    for i in range(3):
        histories.append(engine.requests[i].tokens.copy())
    for written in pool.written:
        assert_equal(written, 0)
    print("engine admission natural passed:", "reserved" if reserve_lifetime else "incremental",
          "steps", steps, "preemptions", preemptions, "computed rows", computed_rows)
    return histories^


def _reserved_abort_cleanup(mut runner: QwenRunner) raises:
    runner.model.reset(runner.ctx)
    var pool = KVPool(runner.ctx, 3, 32, runner.model.kv_geometry())
    var engine = EngineCore(3, 32, 128, VOCABULARY, token_budget=8,
                            max_sequences=3, max_requests=2, reserve_lifetime=True)
    var a = engine.add(0, List[Int](length=30, fill=42), 7, List[Int]())
    var b = engine.add(1, List[Int](length=38, fill=17), 3, List[Int]())
    var partial = engine.step(runner, pool)
    assert_equal(partial.prefill_tokens, 8)
    assert_equal(len(partial.events), 0)
    assert_equal(engine.requests[a].state, PREFILL)
    assert_equal(engine.blocks.reserved[engine.requests[a].sequence], 36)
    assert_equal(engine.blocks.free_blocks(), 1)
    assert_equal(engine.requests[b].state, WAITING)
    assert_equal(engine.requests[b].sequence, -1)
    engine.abort(0)
    engine.abort(0)
    var aborted = engine.step(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.finished, 1)
    assert_equal(aborted.prefill_tokens, 8)
    assert_equal(len(aborted.events), 1)
    assert_equal(aborted.events[0].kind, FINISH_EVENT)
    assert_equal(aborted.events[0].request_id, 0)
    assert_equal(aborted.events[0].reason, "abort")
    assert_equal(aborted.events[0].generated_tokens, 0)
    assert_equal(engine.requests[a].state, FINISHED)
    assert_equal(engine.requests[a].sequence, -1)
    assert_equal(engine.requests[a].generated, 0)
    assert_equal(engine.blocks.length(engine.requests[b].sequence), 8)
    assert_equal(engine.blocks.reserved[engine.requests[b].sequence], 40)
    engine.check(pool)
    engine.abort(0)
    var steps = 0
    var finishes = 0
    while engine.live() > 0 and steps < 1000:
        var record = engine.step(runner, pool)
        for event in record.events:
            assert_equal(event.request_id, 1)
            if event.kind == FINISH_EVENT:
                finishes += 1
                assert_equal(event.reason, "length")
        engine.check(pool)
        steps += 1
    assert_equal(engine.live(), 0)
    assert_equal(finishes, 1)
    assert_equal(engine.requests[b].generated, 3)
    assert_equal(engine.blocks.free_blocks(), 3)
    for written in pool.written:
        assert_equal(written, 0)
    print("engine reserved abort cleanup passed: cached and unwritten reservation blocks released")


def admission(path: String) raises:
    var runner = QwenRunner(path, 128, 8, 3)
    print("engine device", runner.ctx.name(), "backend", runner.ctx.api())
    assert_equal(runner.ctx.api(), "metal")
    var incremental = _natural_histories(runner, False)
    var reserved = _natural_histories(runner, True)
    assert_equal(len(incremental), 3)
    assert_equal(len(reserved), 3)
    for i in range(3):
        _assert_list(incremental[i], reserved[i])
    print("engine admission equality passed: histories 3 generated tokens 13")
    _reserved_abort_cleanup(runner)
    numeric_fault_cleanup(path, True)


def main() raises:
    var args = argv()
    if len(args) == 2:
        numeric_fault_cleanup(args[1])
    elif len(args) == 3 and args[2] == "admission":
        admission(args[1])
    else:
        raise Error("usage: engine_metal_driver VERIFIED_PREPARED_PATH [admission]")
