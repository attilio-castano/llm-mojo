"""Scripted asynchronous ownership and token-delivery lifecycle oracles.

The runner computes tokens immediately but withholds them until FIFO collect.
This proves engine bookkeeping, not GPU concurrency or model numerics. A small
Metal allocation supplies KVPool, whose logical counts are the tested state.
"""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from max.gpu.host import DeviceContext
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.engine import EngineCore, EngineStep, StepCost, WAITING, DRAINING, FINISHED, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVGeometry, KVPool
from llm_mojo.serving.runner import AsyncModelRunner, SimulatedRunner


struct ScriptedAsyncRunner(AsyncModelRunner):
    var reference: SimulatedRunner
    var tickets: List[Int]
    var outputs: List[List[Int]]
    var inputs: List[List[Int]]
    var next_ticket: Int
    var collected: Int
    var max_pending: Int
    var chained: Int
    var fault_ticket: Int
    var fault_kind: Int
    var drains: Int

    def __init__(out self, script: List[Int], vocabulary: Int = 100,
                 fault_ticket: Int = -1, fault_kind: Int = 0) raises:
        self.reference = SimulatedRunner(script, vocabulary)
        self.tickets = List[Int](capacity=2)
        self.outputs = List[List[Int]]()
        self.inputs = List[List[Int]]()
        self.next_ticket = 0
        self.collected = 0
        self.max_pending = 0
        self.chained = 0
        self.fault_ticket = fault_ticket
        self.fault_kind = fault_kind
        self.drains = 0

    def now_ns(self) -> Int:
        return self.reference.now_ns()

    def submit(mut self, batch: StepBatch, source_indices: List[Int], source_ticket: Int,
               mut kv: KVPool) raises -> Int:
        if len(self.tickets) >= 2 or len(source_indices) != batch.rows():
            raise Error("invalid scripted asynchronous submission")
        batch.validate(kv.blocks, kv.block_size, self.reference.vocabulary)
        var ids = batch.token_ids.copy()
        var references = 0
        for i in range(len(source_indices)):
            var source = source_indices[i]
            if source < -1:
                raise Error("invalid symbolic row")
            if source >= 0:
                if (source_ticket != self.next_ticket - 1 or source_ticket < 0
                        or source >= len(self.outputs[source_ticket])):
                    raise Error("invalid scripted predecessor")
                ids[i] = self.outputs[source_ticket][source]
                references += 1
        if references == 0 and source_ticket != -1:
            raise Error("literal batch has a predecessor")
        var resolved = StepBatch(ids.copy(), batch.positions.copy(), batch.query_start.copy(),
            batch.decode_count, batch.seq_lens.copy(), batch.max_blocks, batch.block_table.copy(),
            batch.slot_mapping.copy(), batch.logits_rows.copy())
        var output = self.reference.execute(resolved, kv)
        var ticket = self.next_ticket
        self.next_ticket += 1
        self.chained += references
        self.inputs.append(ids^)
        self.outputs.append(output^)
        self.tickets.append(ticket)
        self.max_pending = max(self.max_pending, len(self.tickets))
        return ticket

    def collect(mut self, ticket: Int) raises -> List[Int]:
        if len(self.tickets) == 0 or self.tickets[0] != ticket:
            raise Error("scripted collection is not FIFO")
        if ticket == self.fault_ticket and self.fault_kind == 1:
            raise Error("injected nonfinite selection")
        var output = self.outputs[ticket].copy()
        if ticket == self.fault_ticket and self.fault_kind == 2 and len(output) > 0:
            output[0] = -1
        if ticket == self.fault_ticket and self.fault_kind == 3:
            output.append(0)
        if len(self.tickets) == 2:
            var successor = self.tickets.pop()
            _ = self.tickets.pop()
            self.tickets.append(successor)
        else:
            _ = self.tickets.pop()
        self.collected += 1
        self.reference.clock_ns += 1
        return output^

    def drain(mut self) raises:
        self.drains += 1
        self.tickets.clear()
        self.reference.clock_ns += 1


def _same(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def _all_free(engine: EngineCore, pool: KVPool) raises:
    engine.check(pool)
    assert_equal(engine.live(), 0)
    assert_equal(engine.pending_steps(), 0)
    assert_equal(engine.blocks.free_blocks(), pool.blocks)
    for count in pool.written:
        assert_equal(count, 0)


def _drain(mut engine: EngineCore, mut runner: ScriptedAsyncRunner, mut pool: KVPool,
           limit: Int = 1000) raises -> Int:
    var calls = 0
    var pressure = 0
    while engine.live() > 0 and calls < limit:
        var record = engine.step_async(runner, pool)
        pressure += record.async_pressure_drain
        assert_true(record.total_tokens <= engine.token_budget)
        assert_true(engine.pending_steps() <= 1)
        engine.check(pool)
        calls += 1
    _all_free(engine, pool)
    assert_true(runner.max_pending <= 2)
    assert_equal(len(runner.tickets), 0)
    return pressure


def test_successor_submitted_before_oldest_collection_and_history_is_real() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=4, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([10, 11, 12])
    var slot = engine.add(1, [1, 2, 3], 5, List[Int]())
    var first = engine.step_async(runner, pool)
    assert_equal(len(first.async_submissions), 2)
    assert_equal(first.async_submissions[0].ticket, 0)
    assert_equal(first.async_submissions[1].ticket, 1)
    assert_equal(first.async_submissions[1].pending, 2)
    assert_equal(first.async_completions[0].ticket, 0)
    assert_equal(first.async_completions[0].pending, 1)
    assert_equal(engine.pending_steps(), 1)
    assert_equal(engine.requests[slot].generated, 1)
    assert_equal(engine.requests[slot].pending_samples, 1)
    assert_equal(len(engine.requests[slot].tokens), 4)
    assert_equal(engine.blocks.length(engine.requests[slot].sequence), 4)
    assert_equal(runner.inputs[1][0], engine.requests[slot].tokens[3])
    var second = engine.step_async(runner, pool)
    assert_equal(len(second.async_submissions), 1)
    assert_equal(second.async_submissions[0].ticket, 2)
    assert_equal(second.async_completions[0].ticket, 1)
    _ = _drain(engine, runner, pool)
    assert_equal(engine.requests[slot].generated, 5)
    assert_true(runner.chained > 0)


def test_stop_defers_block_release_and_discards_one_extra_result() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(2, 32, 64, 100, token_budget=32, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([7])
    var slot = engine.add(1, List[Int](length=32, fill=1), 3, [7])
    var stopped = engine.step_async(runner, pool)
    assert_equal(engine.requests[slot].state, DRAINING)
    assert_equal(engine.requests[slot].generated, 1)
    assert_equal(engine.blocks.free_blocks(), 0)
    assert_equal(stopped.events[0].kind, TOKEN_EVENT)
    assert_equal(stopped.events[1].kind, FINISH_EVENT)
    assert_equal(stopped.events[1].reason, "stop")
    var history = engine.requests[slot].tokens.copy()
    var discarded = engine.step_async(runner, pool)
    assert_equal(len(discarded.events), 0)
    assert_equal(discarded.async_discarded_tokens, 1)
    assert_equal(discarded.async_results[0].disposition, "discarded-stop")
    assert_equal(discarded.async_results[0].generated_tokens, 2)
    _same(engine.requests[slot].tokens, history)
    _all_free(engine, pool)


def test_known_single_token_limit_does_not_extend_the_declared_reservation() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 1, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(1, 32, 33, 100, token_budget=32, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([7])
    var slot = engine.add(1, List[Int](length=32, fill=1), 1, List[Int]())
    var last = engine.step_async(runner, pool)
    assert_equal(len(last.async_submissions), 1)
    assert_equal(runner.next_ticket, 1)
    assert_equal(engine.requests[slot].generated, 1)
    assert_equal(engine.requests[slot].reason, "length")
    _all_free(engine, pool)


def test_zero_generation_exceeding_pool_finishes_once_without_submission() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(1, 1, 1))
    for reserved in [False, True]:
        for aborted in [False, True]:
            var engine = EngineCore(2, 4, 16, 100, reserve_lifetime=reserved, observe_kv=True)
            var runner = ScriptedAsyncRunner([7])
            var prompt = List[Int](length=9, fill=1)
            with assert_raises():
                _ = engine.add(1, prompt, 1, List[Int]())
            assert_equal(len(engine.requests), 0)
            assert_equal(engine.next_ticket, 0)
            var slot = engine.add(1, prompt, 0, List[Int]())
            assert_equal(engine.requests[slot].state, WAITING)
            assert_equal(engine.requests[slot].sequence, -1)
            assert_equal(engine.live(), 1)
            assert_equal(engine.next_ticket, 1)
            assert_equal(len(engine.blocks.active), 0)
            assert_equal(engine.blocks.free_blocks(), 2)
            engine.check(pool)
            if aborted:
                engine.abort(1)
                engine.abort(1)
            var record = engine.step_async(runner, pool)
            var reason = "abort" if aborted else "length"
            assert_equal(record.finished, 1)
            assert_equal(record.aborted, 1 if aborted else 0)
            assert_equal(record.admitted, 0)
            assert_equal(record.total_tokens, 0)
            assert_equal(record.attended_positions, 0)
            assert_equal(len(record.async_submissions), 0)
            assert_equal(len(record.async_completions), 0)
            assert_equal(len(record.events), 1)
            assert_equal(record.events[0].kind, FINISH_EVENT)
            assert_equal(record.events[0].request_id, 1)
            assert_equal(record.events[0].reason, reason)
            assert_equal(record.events[0].prompt_tokens, 9)
            assert_equal(record.events[0].generated_tokens, 0)
            assert_equal(engine.requests[slot].state, FINISHED)
            assert_equal(engine.requests[slot].generated, 0)
            assert_equal(engine.requests[slot].reason, reason)
            _same(engine.requests[slot].tokens, prompt)
            for point in record.kv_observations:
                assert_equal(point.allocated_blocks, 0)
                assert_equal(point.written_tokens, 0)
                assert_equal(point.reserved_tokens, 0)
                assert_equal(point.resident_requests, 0)
            engine.abort(1)
            var idle = engine.step_async(runner, pool)
            assert_equal(idle.finished, 0)
            assert_equal(len(idle.events), 0)
            assert_equal(idle.admitted, 0)
            assert_equal(idle.total_tokens, 0)
            assert_equal(len(idle.async_submissions), 0)
            assert_equal(len(idle.async_completions), 0)
            assert_equal(runner.next_ticket, 0)
            assert_equal(runner.collected, 0)
            assert_equal(runner.reference.steps, 0)
            assert_equal(len(engine.blocks.active), 0)
            _all_free(engine, pool)


def test_abort_before_submission_does_no_gpu_work() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(2, 4, 8, 100)
    var runner = ScriptedAsyncRunner([7])
    _ = engine.add(2, [2], 3, List[Int]())
    engine.abort(2)
    engine.abort(2)
    var boundary = engine.step_async(runner, pool)
    assert_equal(boundary.finished, 1)
    assert_equal(boundary.aborted, 1)
    assert_equal(runner.next_ticket, 0)
    _all_free(engine, pool)


def test_abort_after_delivery_preserves_history_and_discards_pending_result() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=4, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([10, 11, 12])
    var slot = engine.add(1, [1, 2], 8, List[Int]())
    _ = engine.step_async(runner, pool)
    var history = engine.requests[slot].tokens.copy()
    var submitted = runner.next_ticket
    engine.abort(1)
    engine.abort(1)
    var aborted = engine.step_async(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.finished, 1)
    assert_equal(aborted.async_discarded_tokens, 1)
    assert_equal(aborted.async_results[0].disposition, "discarded-abort")
    assert_equal(runner.next_ticket, submitted)
    _same(engine.requests[slot].tokens, history)
    _all_free(engine, pool)


def test_abort_during_partial_prefill_releases_only_after_last_queued_use() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=2, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([7])
    var slot = engine.add(1, [1, 2, 3, 4, 5, 6, 7], 3, List[Int]())
    _ = engine.step_async(runner, pool)
    assert_equal(engine.requests[slot].generated, 0)
    assert_equal(engine.requests[slot].pending_uses, 1)
    engine.abort(1)
    var aborted = engine.step_async(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(engine.requests[slot].generated, 0)
    assert_equal(runner.next_ticket, 2)
    _all_free(engine, pool)


def test_draining_request_slot_cannot_be_reused_until_its_gpu_reference_retires() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(2, 4, 8, 100, max_requests=1, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([7])
    var slot = engine.add(1, [1], 4, [7])
    _ = engine.step_async(runner, pool)
    var old_ticket = engine.requests[slot].ticket
    with assert_raises():
        _ = engine.add(2, [2], 1, List[Int]())
    _ = engine.step_async(runner, pool)
    assert_equal(engine.add(2, [2], 1, List[Int]()), slot)
    assert_true(engine.requests[slot].ticket > old_ticket)
    _ = _drain(engine, runner, pool)
    assert_equal(engine.requests[slot].generated, 1)


def test_arriving_prefill_and_pending_decode_share_a_mixed_successor() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 8, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(8, 4, 32, 100, token_budget=4, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([10, 11, 12])
    var a = engine.add(1, [1, 2, 3], 6, List[Int]())
    _ = engine.step_async(runner, pool)
    var b = engine.add(2, [4, 5, 6, 7, 8, 9, 10], 2, List[Int]())
    var submitting = engine.step_async(runner, pool)
    assert_equal(submitting.async_submissions[0].decode_seqs, 1)
    assert_equal(submitting.async_submissions[0].prefill_seqs, 1)
    assert_equal(submitting.async_submissions[0].prefill_tokens, 3)
    var mixed = engine.step_async(runner, pool)
    assert_equal(mixed.decode_seqs, 1)
    assert_equal(mixed.prefill_tokens, 3)
    _ = _drain(engine, runner, pool)
    assert_equal(engine.requests[a].generated, 6)
    assert_equal(engine.requests[b].generated, 2)


def test_incremental_pressure_drains_before_preemption_and_replay_is_exact() raises:
    var ctx = DeviceContext()
    var async_pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    var sync_pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    var asynchronous = EngineCore(3, 4, 16, 100, token_budget=4)
    var synchronous = EngineCore(3, 4, 16, 100, token_budget=4)
    var async_runner = ScriptedAsyncRunner([10, 11, 12])
    var sync_runner = SimulatedRunner([10, 11, 12], 100)
    for id in range(2):
        _ = asynchronous.add(id, [1, 2, 3, 4], 8, List[Int]())
        _ = synchronous.add(id, [1, 2, 3, 4], 8, List[Int]())
    var pressure = _drain(asynchronous, async_runner, async_pool)
    assert_true(pressure > 0)
    var calls = 0
    while synchronous.live() > 0 and calls < 1000:
        _ = synchronous.step(sync_runner, sync_pool)
        calls += 1
    _all_free(synchronous, sync_pool)
    for i in range(2):
        _same(asynchronous.requests[i].tokens, synchronous.requests[i].tokens)
        assert_equal(asynchronous.requests[i].reason, synchronous.requests[i].reason)


def test_reserved_fifo_waiting_and_fitted_budget_drain_without_replay() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=4, reserve_lifetime=True)
    engine.set_cost_policy(StepCost(100, 100, 1, 0, 0, 250))
    var runner = ScriptedAsyncRunner([10, 11, 12])
    _ = engine.add(1, [1], 12, List[Int]())
    _ = engine.add(2, [2, 3], 4, List[Int]())
    _ = engine.add(3, [3], 1, List[Int]())
    assert_equal(_drain(engine, runner, pool), 0)
    for i in range(3):
        assert_equal(engine.requests[i].preemptions, 0)
        assert_equal(engine.requests[i].reason, "length")


def test_full_context_limit_never_computes_final_emitted_token() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(2, 4, 8, 100, token_budget=4, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([7])
    var slot = engine.add(1, [1, 2, 3], 5, List[Int]())
    _ = _drain(engine, runner, pool)
    assert_equal(len(engine.requests[slot].tokens), 8)
    var rows = 0
    for input in runner.inputs:
        rows += len(input)
    assert_equal(rows, 7)


def test_fault_in_either_ticket_preserves_only_delivered_tokens_and_fails_waiters() raises:
    for fault_ticket in [0, 1]:
        for fault_kind in [1, 2, 3]:
            var ctx = DeviceContext()
            var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
            var engine = EngineCore(4, 4, 16, 100, token_budget=4, reserve_lifetime=True)
            var runner = ScriptedAsyncRunner([10, 11, 12], fault_ticket=fault_ticket, fault_kind=fault_kind)
            _ = engine.add(1, [1], 12, List[Int]())
            _ = engine.add(2, [2], 4, List[Int]())
            if fault_ticket == 1:
                _ = engine.step_async(runner, pool)
            var history = engine.requests[0].tokens.copy()
            _ = engine.add(3, [3, 4], 6, List[Int]())
            with assert_raises():
                _ = engine.step_async(runner, pool)
            assert_true(engine.failed)
            _same(engine.requests[0].tokens, history)
            assert_equal(len(engine.failure_events), 3)
            for i in range(3):
                assert_equal(engine.failure_events[i].request_id, i + 1)
                assert_equal(engine.failure_events[i].reason, "error")
            _all_free(engine, pool)
            assert_equal(runner.drains, 1)
            with assert_raises():
                _ = engine.add(4, [4], 1, List[Int]())
            with assert_raises():
                _ = engine.step_async(runner, pool)


def test_explicit_drain_retires_existing_ticket_without_submitting_more() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, reserve_lifetime=True)
    var runner = ScriptedAsyncRunner([10])
    _ = engine.add(1, [1], 8, List[Int]())
    _ = engine.step_async(runner, pool)
    var count = runner.next_ticket
    engine.abort(1)
    var records = engine.drain_async(runner, pool)
    assert_equal(len(records), 1)
    assert_equal(records[0].async_results[0].disposition, "discarded-abort")
    assert_equal(runner.next_ticket, count)
    _all_free(engine, pool)


def test_boundary_abort_finish_survives_collection_fault_in_step_and_drain() raises:
    for waiting in [True, False]:
        for explicit_drain in [True, False]:
            var ctx = DeviceContext()
            var blocks = 2 if waiting else 4
            var pool = KVPool(ctx, blocks, 4, KVGeometry(1, 1, 1))
            var engine = EngineCore(blocks, 4, 16, 100, token_budget=4,
                                    max_requests=2, reserve_lifetime=True)
            var runner = ScriptedAsyncRunner([10, 11, 12], fault_ticket=1, fault_kind=1)
            _ = engine.add(1, [1], 6 if waiting else 8, List[Int]())
            _ = engine.add(2, [2], 4 if waiting else 6, List[Int]())
            _ = engine.step_async(runner, pool)
            assert_equal(engine.requests[1].pending_uses, 0 if waiting else 1)
            var a_history = engine.requests[0].tokens.copy()
            var b_history = engine.requests[1].tokens.copy()
            engine.abort(2)
            with assert_raises():
                if explicit_drain:
                    _ = engine.drain_async(runner, pool)
                else:
                    _ = engine.step_async(runner, pool)
            assert_true(engine.failed)
            _same(engine.requests[0].tokens, a_history)
            _same(engine.requests[1].tokens, b_history)
            var a_finishes = 0
            var b_finishes = 0
            for event in engine.failure_events:
                assert_equal(event.kind, FINISH_EVENT)
                if event.request_id == 1:
                    assert_equal(event.reason, "error")
                    a_finishes += 1
                elif event.request_id == 2:
                    assert_equal(event.reason, "abort")
                    b_finishes += 1
                else:
                    raise Error("unexpected failure request")
            assert_equal(a_finishes, 1)
            assert_equal(b_finishes, 1)
            assert_equal(len(engine.failure_events), 2)
            _all_free(engine, pool)


def test_synchronous_path_cannot_consume_async_pending_state() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, reserve_lifetime=True)
    var asynchronous = ScriptedAsyncRunner([10])
    var synchronous = SimulatedRunner([10], 100)
    _ = engine.add(1, [1], 8, List[Int]())
    _ = engine.step_async(asynchronous, pool)
    with assert_raises():
        _ = engine.step(synchronous, pool)
    assert_equal(asynchronous.next_ticket, 2)
    _ = _drain(engine, asynchronous, pool)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
