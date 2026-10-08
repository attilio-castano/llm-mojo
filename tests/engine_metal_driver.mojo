"""Explicit prepared-checkpoint acceptance driver for numeric fault cleanup.

This changes a private resident tied-head weight after a successful step. It
does not change prepared files or simulate device loss. The real model submits
the next Metal step, GPU argmax detects a nonfinite logit, and QwenRunner must
drain before EngineCore releases logical ownership and rejects future work.
"""
from std.memory import bitcast
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from llm_mojo.layers.decoder_layer import DECODER_MIXED
from llm_mojo.models.qwen2.model import HIDDEN, LAYERS, VOCABULARY
from llm_mojo.models.qwen2.runner import QwenRunner
from llm_mojo.serving.engine import EngineCore, DECODE, WAITING, FINISHED, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVPool


def _assert_list(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def numeric_fault_cleanup(path: String) raises:
    var runner = QwenRunner(path, 128, 8, 3)
    print("engine device", runner.ctx.name(), "backend", runner.ctx.api())
    assert_equal(runner.ctx.api(), "metal")
    var pool = KVPool(runner.ctx, 4, 32, runner.model.kv_geometry())
    var engine = EngineCore(4, 32, 128, VOCABULARY, token_budget=8,
                            max_sequences=3, max_requests=3)
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
    var b = engine.add(1, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 2, List[Int]())
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


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: engine_metal_driver VERIFIED_PREPARED_PATH")
    numeric_fault_cleanup(args[1])
