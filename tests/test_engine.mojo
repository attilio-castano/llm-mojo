"""Native scheduler safety, accounting and finite-workload liveness.

The simulator only advances logical KV counts; these are not model numerical
tests. A small Metal allocation supplies the existing KVPool container.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from max.gpu.host import DeviceContext
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.engine import EngineCore, EngineStep, StepCost, WAITING, PREFILL, DECODE, FINISHED, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVGeometry, KVPool
from llm_mojo.serving.runner import ModelRunner, SimulatedRunner


def _assert_list(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def _drain(mut engine: EngineCore, mut runner: SimulatedRunner, mut pool: KVPool, bound: Int = 1000) raises -> Int:
    var steps = 0
    while engine.live() > 0 and steps < bound:
        _ = engine.step(runner, pool)
        engine.check(pool)
        steps += 1
    assert_equal(engine.live(), 0)
    assert_equal(engine.blocks.free_blocks(), pool.blocks)
    return steps


def test_admission_rejections_change_nothing() raises:
    var engine = EngineCore(2, 32, 128, 100, max_requests=1)
    for maximum in [-1, 65, 128, 9223372036854775807]:
        with assert_raises():
            _ = engine.add(1, [1], maximum, List[Int]())
    with assert_raises():
        _ = engine.add(-1, [1], 1, List[Int]())
    with assert_raises():
        _ = engine.add(1, List[Int](), 1, List[Int]())
    with assert_raises():
        _ = engine.add(1, [100], 1, List[Int]())
    with assert_raises():
        _ = engine.add(1, [1], 1, [100])
    assert_equal(len(engine.requests), 0)
    assert_equal(engine.next_ticket, 0)
    assert_equal(engine.blocks.free_blocks(), 2)
    var slot = engine.add(1, [1], 63, List[Int]())
    with assert_raises():
        _ = engine.add(1, [2], 1, List[Int]())
    with assert_raises():
        _ = engine.add(2, [2], 1, List[Int]())
    assert_equal(slot, 0)
    assert_equal(len(engine.requests), 1)
    assert_equal(engine.next_ticket, 1)
    _assert_list(engine.requests[slot].tokens, [1])
    engine.blocks.check()
    var context = EngineCore(2, 32, 64, 100)
    with assert_raises():
        _ = context.add(1, [1], 64, List[Int]())
    assert_equal(len(context.requests), 0)


def test_prefill_chunks_emit_only_at_the_end_and_clocks_account() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=3)
    var runner = SimulatedRunner([11, 12, 13], 100, 1000, 10, 1)
    var slot = engine.add(4, [1, 2, 3, 4, 5, 6, 7], 3, List[Int](), 0)
    var first = engine.step(runner, pool)
    assert_equal(first.total_tokens, 3)
    assert_equal(first.prefill_tokens, 3)
    assert_equal(first.decode_seqs, 0)
    assert_equal(len(first.events), 0)
    assert_equal(first.execute_ns, 1036)
    assert_equal(first.end_ns - first.begin_ns, first.execute_ns)
    assert_equal(engine.requests[slot].state, PREFILL)
    var second = engine.step(runner, pool)
    assert_equal(len(second.events), 0)
    var third = engine.step(runner, pool)
    assert_equal(third.total_tokens, 1)
    assert_equal(len(third.events), 1)
    assert_equal(third.events[0].kind, TOKEN_EVENT)
    assert_equal(third.events[0].generated_tokens, 1)
    assert_equal(third.events[0].emitted_ns, third.end_ns)
    assert_equal(engine.requests[slot].state, DECODE)
    _ = _drain(engine, runner, pool)
    assert_equal(engine.requests[slot].generated, 3)
    assert_equal(engine.requests[slot].reason, "length")


def test_decode_and_one_prompt_share_a_step() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 8, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(8, 4, 32, 100, token_budget=4)
    var runner = SimulatedRunner([10, 11, 12], 100)
    var a = engine.add(1, [1, 2, 3], 6, List[Int]())
    _ = engine.step(runner, pool)
    var b = engine.add(2, [4, 5, 6, 7, 8, 9, 10], 2, List[Int]())
    var mixed = engine.step(runner, pool)
    assert_equal(mixed.decode_seqs, 1)
    assert_equal(mixed.prefill_seqs, 1)
    assert_equal(mixed.prefill_tokens, 3)
    assert_equal(mixed.total_tokens, 4)
    assert_equal(len(mixed.events), 1)
    assert_equal(mixed.events[0].request_id, 1)
    assert_equal(engine.requests[a].generated, 2)
    assert_equal(engine.requests[b].generated, 0)
    _ = _drain(engine, runner, pool)
    assert_equal(engine.requests[a].generated, 6)
    assert_equal(engine.requests[b].generated, 2)


def test_aborts_are_boundary_actions_and_slots_can_be_reused() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(2, 4, 8, 100, max_requests=1)
    var runner = SimulatedRunner([10], 100)
    var slot = engine.add(1, [1, 2], 3, List[Int]())
    engine.abort(1)
    engine.abort(1)
    engine.abort(999)
    assert_equal(engine.requests[slot].state, WAITING)
    var aborted = engine.step(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.total_tokens, 0)
    assert_equal(runner.steps, 0)
    assert_equal(engine.requests[slot].state, FINISHED)
    assert_equal(aborted.events[0].reason, "abort")
    var reused = engine.add(2, [3], 3, [10])
    assert_equal(reused, slot)
    var stopped = engine.step(runner, pool)
    assert_equal(stopped.events[0].kind, TOKEN_EVENT)
    assert_equal(stopped.events[1].kind, FINISH_EVENT)
    assert_equal(stopped.events[1].reason, "stop")
    assert_equal(engine.requests[slot].generated, 1)
    engine.check(pool)
    assert_equal(engine.blocks.free_blocks(), 2)


def test_zero_generation_exceeding_pool_finishes_once_without_execution() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(1, 1, 1))
    for reserved in [False, True]:
        for aborted in [False, True]:
            var engine = EngineCore(2, 4, 16, 100, reserve_lifetime=reserved, observe_kv=True)
            var runner = SimulatedRunner([10], 100)
            var prompt = List[Int](length=9, fill=1)
            # One output token requires prefill and cannot fit this pool.
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
            var record = engine.step(runner, pool)
            var reason = "abort" if aborted else "length"
            assert_equal(record.finished, 1)
            assert_equal(record.aborted, 1 if aborted else 0)
            assert_equal(record.admitted, 0)
            assert_equal(record.total_tokens, 0)
            assert_equal(record.attended_positions, 0)
            assert_equal(record.execute_ns, 0)
            assert_equal(len(record.events), 1)
            assert_equal(record.events[0].kind, FINISH_EVENT)
            assert_equal(record.events[0].request_id, 1)
            assert_equal(record.events[0].reason, reason)
            assert_equal(record.events[0].prompt_tokens, 9)
            assert_equal(record.events[0].generated_tokens, 0)
            assert_equal(engine.requests[slot].state, FINISHED)
            assert_equal(engine.requests[slot].generated, 0)
            assert_equal(engine.requests[slot].reason, reason)
            _assert_list(engine.requests[slot].tokens, prompt)
            for point in record.kv_observations:
                assert_equal(point.allocated_blocks, 0)
                assert_equal(point.written_tokens, 0)
                assert_equal(point.reserved_tokens, 0)
                assert_equal(point.resident_requests, 0)
            engine.abort(1)
            var idle = engine.step(runner, pool)
            assert_equal(idle.finished, 0)
            assert_equal(len(idle.events), 0)
            assert_equal(idle.admitted, 0)
            assert_equal(idle.total_tokens, 0)
            assert_equal(runner.steps, 0)
            assert_equal(engine.live(), 0)
            assert_equal(len(engine.blocks.active), 0)
            assert_equal(engine.blocks.free_blocks(), 2)
            for count in pool.written:
                assert_equal(count, 0)
            engine.check(pool)


def test_zero_generation_keeps_validation_and_queue_bounds() raises:
    for reserved in [False, True]:
        var engine = EngineCore(2, 4, 16, 100, max_requests=1, reserve_lifetime=reserved)
        with assert_raises():
            _ = engine.add(-1, [1], 0, List[Int]())
        with assert_raises():
            _ = engine.add(1, [1], 0, List[Int](), -2)
        with assert_raises():
            _ = engine.add(1, List[Int](), 0, List[Int]())
        with assert_raises():
            _ = engine.add(1, List[Int](length=17, fill=1), 0, List[Int]())
        for invalid in [-1, 100]:
            with assert_raises():
                _ = engine.add(1, [invalid], 0, List[Int]())
            with assert_raises():
                _ = engine.add(1, [1], 0, [invalid])
        assert_equal(len(engine.requests), 0)
        assert_equal(engine.next_ticket, 0)
        var prompt = List[Int](length=9, fill=1)
        var slot = engine.add(1, prompt, 0, List[Int]())
        with assert_raises():
            _ = engine.add(1, [2], 0, List[Int]())
        with assert_raises():
            _ = engine.add(2, [2], 0, List[Int]())
        assert_equal(len(engine.requests), 1)
        assert_equal(engine.next_ticket, 1)
        assert_equal(engine.requests[slot].state, WAITING)
        assert_equal(engine.requests[slot].sequence, -1)
        _assert_list(engine.requests[slot].tokens, prompt)
        assert_equal(len(engine.blocks.active), 0)
        assert_equal(engine.blocks.free_blocks(), 2)
        engine.blocks.check()


def test_pressure_preemption_replays_without_duplicate_delivery() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(3, 4, 16, 100, token_budget=4)
    var runner = SimulatedRunner([10, 11, 12], 100)
    var a = engine.add(1, [1, 2, 3, 4], 8, List[Int]())
    _ = engine.step(runner, pool)
    var b = engine.add(2, [5, 6, 7, 8], 8, List[Int]())
    var delivered = List[Int](length=2, fill=0)
    delivered[0] = 1
    var preemptions = 0
    var steps = 0
    while engine.live() > 0 and steps < 100:
        var record = engine.step(runner, pool)
        preemptions += record.preempted
        for event in record.events:
            if event.kind == TOKEN_EVENT:
                delivered[event.request_id - 1] += 1
                assert_equal(event.generated_tokens, delivered[event.request_id - 1])
        steps += 1
    assert_equal(engine.live(), 0)
    assert_true(preemptions > 0)
    assert_equal(delivered[0], 8)
    assert_equal(delivered[1], 8)
    assert_equal(engine.requests[a].generated, 8)
    assert_equal(engine.requests[b].generated, 8)
    assert_equal(engine.blocks.free_blocks(), 3)
    # Script is schedule-independent: compare the retained histories to solo runs.
    var solo_pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    for slot in [a, b]:
        var solo = EngineCore(3, 4, 16, 100, token_budget=4)
        var reference = SimulatedRunner([10, 11, 12], 100)
        var prompt = List[Int]()
        for i in range(4):
            prompt.append(engine.requests[slot].tokens[i])
        _ = solo.add(1, prompt, 8, List[Int]())
        _ = _drain(solo, reference, solo_pool)
        _assert_list(engine.requests[slot].tokens, solo.requests[0].tokens)


def test_finite_feasible_workloads_drain_across_small_pools() raises:
    var ctx = DeviceContext()
    for blocks in [1, 2, 3, 5]:
        var pool = KVPool(ctx, blocks, 4, KVGeometry(1, 1, 1))
        for budget in [1, 2, 4, 8]:
            for mixed in [False, True]:
                var engine = EngineCore(blocks, 4, blocks * 4, 100, token_budget=budget,
                    max_sequences=3, max_requests=12, watermark_blocks=blocks, mixed_prefill=mixed)
                var runner = SimulatedRunner([10, 11, 12], 100)
                for i in range(12):
                    var length = 1 + i % min(blocks * 4 - 1, 7)
                    var prompt = List[Int](length=length, fill=i % 10)
                    var maximum = min(3, blocks * 4 - length)
                    _ = engine.add(i, prompt, maximum, List[Int]())
                _ = _drain(engine, runner, pool, 500)


def test_standalone_prefill_arm_pauses_decodes_and_uses_its_declared_budget() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 8, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(8, 4, 32, 100, token_budget=32, mixed_prefill=False)
    var runner = SimulatedRunner([10], 100)
    var a = engine.add(1, [1], 3, List[Int]())
    _ = engine.step(runner, pool)
    _ = engine.add(2, [2, 3, 4, 5, 6, 7, 8], 3, List[Int]())
    var prefill = engine.step(runner, pool)
    assert_equal(prefill.prefill_tokens, 7)
    assert_equal(prefill.decode_seqs, 0)
    assert_equal(engine.requests[a].generated, 1)
    var decode = engine.step(runner, pool)
    assert_equal(decode.decode_seqs, 2)
    assert_equal(decode.prefill_seqs, 0)
    _ = _drain(engine, runner, pool)


def test_arrivals_aborts_and_reused_slots_finish_under_pressure() raises:
    var ctx = DeviceContext()
    for mixed in [False, True]:
        var pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
        var engine = EngineCore(3, 4, 12, 100, token_budget=3, max_sequences=3,
                                max_requests=8, mixed_prefill=mixed)
        var runner = SimulatedRunner([10, 11, 12], 100)
        var next_id = 0
        var finished = 0
        var tick = 0
        while (next_id < 40 or engine.live() > 0) and tick < 2000:
            if next_id < 40 and engine.live() < 8:
                var length = 1 + next_id % 8
                _ = engine.add(next_id, List[Int](length=length, fill=next_id % 10),
                               min(4, 12 - length), List[Int](), runner.now_ns())
                if next_id % 7 == 0:
                    engine.abort(next_id)
                    engine.abort(next_id)
                next_id += 1
            var record = engine.step(runner, pool)
            finished += record.finished
            for event in record.events:
                assert_true(event.emitted_ns >= event.arrival_ns)
            tick += 1
        assert_equal(next_id, 40)
        assert_equal(finished, 40)
        assert_equal(engine.live(), 0)
        assert_equal(engine.blocks.free_blocks(), 3)
        assert_true(len(engine.requests) <= 8)


def test_cost_policy_validates_before_mutation_and_saturates() raises:
    var engine = EngineCore(4, 4, 16, 100)
    for policy in [StepCost(-1, 0, 0, 0, 0), StepCost(0, -1, 0, 0, 0),
                   StepCost(0, 0, -1, 0, 0), StepCost(0, 0, 0, -1, 0),
                   StepCost(0, 0, 0, 0, -1), StepCost(0, 0, 0, 0, 0, 0)]:
        with assert_raises():
            engine.set_cost_policy(policy)
        assert_equal(engine.cost_policy_enabled, False)
    engine.set_cost_policy(StepCost(1, 2, 3, 4, 5, 100))
    with assert_raises():
        engine.set_cost_policy(StepCost(9, 9, 9, -1, 9, 900))
    assert_equal(engine.cost_policy.target_ns, 100)
    assert_equal(engine.cost_policy.estimate(2, 3, 1, 1), 23)
    assert_equal(StepCost(Int.MAX, 1, 0, 0, 0).estimate(2, 0, 0, 0), Int.MAX)
    assert_equal(StepCost(0, Int.MAX, 0, 0, 0).estimate(2, 0, 0, 0), Int.MAX)


def test_zero_cost_policy_keeps_the_fixed_scheduler_identical() raises:
    var ctx = DeviceContext()
    var a_pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    var b_pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    var a = EngineCore(3, 4, 12, 100, token_budget=4)
    var b = EngineCore(3, 4, 12, 100, token_budget=4)
    b.set_cost_policy(StepCost(0, 0, 0, 0, 0, 1))
    var a_runner = SimulatedRunner([10, 11, 12], 100)
    var b_runner = SimulatedRunner([10, 11, 12], 100)
    for i in range(4):
        var prompt = List[Int](length=i + 1, fill=i)
        _ = a.add(i, prompt, 6, List[Int]())
        _ = b.add(i, prompt, 6, List[Int]())
    var steps = 0
    while a.live() > 0 and steps < 1000:
        var left = a.step(a_runner, a_pool)
        var right = b.step(b_runner, b_pool)
        assert_equal(left.total_tokens, right.total_tokens)
        assert_equal(left.prefill_tokens, right.prefill_tokens)
        assert_equal(left.decode_seqs, right.decode_seqs)
        assert_equal(left.preempted, right.preempted)
        assert_equal(left.execute_ns, right.execute_ns)
        assert_equal(right.budget_limited, 0)
        assert_equal(len(left.events), len(right.events))
        for i in range(len(left.events)):
            assert_equal(left.events[i].request_id, right.events[i].request_id)
            assert_equal(left.events[i].token_id, right.events[i].token_id)
            assert_equal(left.events[i].reason, right.events[i].reason)
        steps += 1
    assert_equal(a.live(), 0)
    assert_equal(b.live(), 0)
    for i in range(4):
        _assert_list(a.requests[i].tokens, b.requests[i].tokens)


def test_fitted_rows_bound_prefill_and_count_singleton_attention_correctly() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 8, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(8, 4, 32, 100, token_budget=4)
    engine.set_cost_policy(StepCost(0, 10, 0, 0, 0, 25))
    var runner = SimulatedRunner([10], 100)
    _ = engine.add(0, [1, 2, 3, 4, 5, 6, 7], 2, List[Int]())
    var limited = engine.step(runner, pool)
    assert_equal(limited.prefill_tokens, 2)
    assert_equal(limited.predicted_ns, 20)
    assert_equal(limited.budget_limited, 1)
    _ = _drain(engine, runner, pool)
    _ = engine.add(1, [1], 5, List[Int]())
    _ = engine.step(runner, pool)
    _ = engine.add(2, [2, 3, 4, 5, 6], 2, List[Int]())
    engine.set_cost_policy(StepCost(0, 0, 0, 100, 0, 150))
    var singleton = engine.step(runner, pool)
    assert_equal(singleton.decode_seqs, 1)
    assert_equal(singleton.prefill_tokens, 1)
    assert_equal(singleton.predicted_ns, 100)
    assert_equal(singleton.budget_limited, 1)
    _ = _drain(engine, runner, pool)


def test_cost_policy_counts_causal_positions_and_final_logit_rows() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=4)
    var runner = SimulatedRunner([10], 100)
    engine.set_cost_policy(StepCost(0, 0, 1, 0, 0, 6))
    _ = engine.add(0, [1, 2, 3, 4], 1, List[Int]())
    var prefix = engine.step(runner, pool)
    assert_equal(prefix.prefill_tokens, 3)
    assert_equal(prefix.attended_positions, 6)
    assert_equal(prefix.predicted_ns, 6)
    var tail = engine.step(runner, pool)
    assert_equal(tail.prefill_tokens, 1)
    assert_equal(tail.attended_positions, 4)
    assert_equal(tail.predicted_ns, 4)
    assert_equal(engine.live(), 0)
    engine.set_cost_policy(StepCost(0, 0, 0, 0, 100, 25))
    _ = engine.add(1, [1, 2, 3, 4], 2, List[Int]())
    var no_head = engine.step(runner, pool)
    assert_equal(no_head.prefill_tokens, 3)
    assert_equal(no_head.predicted_ns, 0)
    assert_equal(len(no_head.events), 0)
    var mandatory_head = engine.step(runner, pool)
    assert_equal(mandatory_head.prefill_tokens, 1)
    assert_equal(mandatory_head.predicted_ns, 100)
    assert_equal(mandatory_head.budget_limited, 1)
    assert_equal(len(mandatory_head.events), 1)
    _ = _drain(engine, runner, pool)


def test_tiny_target_keeps_decodes_progressing_and_pressure_replays_drain() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(3, 4, 12, 100, token_budget=4)
    var runner = SimulatedRunner([10, 11, 12], 100)
    _ = engine.add(0, [1], 8, List[Int]())
    _ = engine.step(runner, pool)
    _ = engine.add(1, [2, 3, 4, 5], 8, List[Int]())
    engine.set_cost_policy(StepCost(100, 10, 10, 10, 10, 1))
    var over = engine.step(runner, pool)
    assert_equal(over.decode_seqs, 1)
    assert_equal(over.prefill_seqs, 0)
    assert_true(over.predicted_ns > engine.cost_policy.target_ns)
    var steps = _drain(engine, runner, pool, 500)
    assert_true(steps > 0)
    assert_equal(engine.requests[0].generated, 8)
    assert_equal(engine.requests[1].generated, 8)


struct BrokenRunner(ModelRunner):
    def __init__(out self):
        pass

    def now_ns(self) -> Int:
        return 0

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        raise Error("injected synchronous runner failure")


def test_execution_failure_terminalizes_and_invalidates_the_engine() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100)
    _ = engine.add(1, [1], 2, List[Int]())
    _ = engine.add(2, [2], 2, List[Int]())
    var runner = BrokenRunner()
    with assert_raises():
        _ = engine.step(runner, pool)
    assert_true(engine.failed)
    assert_equal(engine.live(), 0)
    assert_equal(engine.blocks.free_blocks(), 4)
    assert_equal(len(engine.failure_events), 2)
    for event in engine.failure_events:
        assert_equal(event.kind, FINISH_EVENT)
        assert_equal(event.reason, "error")
    with assert_raises():
        _ = engine.add(3, [3], 2, List[Int]())
    with assert_raises():
        _ = engine.step(runner, pool)
    engine.check(pool)


def test_abort_after_partial_prefill_releases_cached_blocks() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=2, max_requests=1)
    var runner = SimulatedRunner([10], 100)
    var slot = engine.add(1, [1, 2, 3, 4, 5], 3, List[Int]())
    var partial = engine.step(runner, pool)
    assert_equal(partial.prefill_tokens, 2)
    assert_equal(len(partial.events), 0)
    assert_equal(engine.requests[slot].state, PREFILL)
    assert_equal(engine.requests[slot].generated, 0)
    assert_equal(engine.blocks.length(engine.requests[slot].sequence), 2)
    assert_equal(engine.blocks.free_blocks(), 3)
    var history = engine.requests[slot].tokens.copy()
    var executed = runner.steps
    engine.abort(1)
    engine.abort(1)
    var aborted = engine.step(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.finished, 1)
    assert_equal(aborted.total_tokens, 0)
    assert_equal(runner.steps, executed)
    assert_equal(len(aborted.events), 1)
    assert_equal(aborted.events[0].kind, FINISH_EVENT)
    assert_equal(aborted.events[0].request_id, 1)
    assert_equal(aborted.events[0].reason, "abort")
    assert_equal(aborted.events[0].generated_tokens, 0)
    assert_equal(engine.requests[slot].state, FINISHED)
    assert_equal(engine.requests[slot].sequence, -1)
    _assert_list(engine.requests[slot].tokens, history)
    assert_equal(engine.blocks.free_blocks(), 4)
    for written in pool.written:
        assert_equal(written, 0)
    engine.check(pool)
    engine.abort(1)
    var idle = engine.step(runner, pool)
    assert_equal(len(idle.events), 0)
    assert_equal(runner.steps, executed)
    # Reusing the released request slot and physical block cannot retain the
    # cancelled prompt's cached extent or produce another finish for its ID.
    var reused = engine.add(2, [7, 8], 2, List[Int]())
    assert_equal(reused, slot)
    while engine.live() > 0:
        var next = engine.step(runner, pool)
        for event in next.events:
            assert_equal(event.request_id, 2)
    assert_equal(engine.requests[reused].generated, 2)
    assert_equal(engine.blocks.free_blocks(), 4)
    engine.check(pool)


def test_abort_after_decode_preserves_delivered_history() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=2)
    var runner = SimulatedRunner([10, 11], 100)
    var slot = engine.add(1, [1, 2], 6, List[Int]())
    var prefill = engine.step(runner, pool)
    assert_equal(prefill.prefill_tokens, 2)
    assert_equal(engine.requests[slot].state, DECODE)
    assert_equal(engine.requests[slot].generated, 1)
    var decode = engine.step(runner, pool)
    assert_equal(decode.decode_seqs, 1)
    assert_equal(engine.requests[slot].state, DECODE)
    assert_equal(engine.requests[slot].generated, 2)
    assert_equal(engine.blocks.length(engine.requests[slot].sequence), 3)
    var history = engine.requests[slot].tokens.copy()
    var executed = runner.steps
    engine.abort(1)
    engine.abort(1)
    var aborted = engine.step(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.finished, 1)
    assert_equal(aborted.total_tokens, 0)
    assert_equal(runner.steps, executed)
    assert_equal(len(aborted.events), 1)
    assert_equal(aborted.events[0].kind, FINISH_EVENT)
    assert_equal(aborted.events[0].request_id, 1)
    assert_equal(aborted.events[0].reason, "abort")
    assert_equal(aborted.events[0].generated_tokens, 2)
    assert_equal(engine.requests[slot].state, FINISHED)
    assert_equal(engine.requests[slot].sequence, -1)
    assert_equal(engine.requests[slot].generated, 2)
    _assert_list(engine.requests[slot].tokens, history)
    assert_equal(engine.blocks.free_blocks(), 4)
    for written in pool.written:
        assert_equal(written, 0)
    engine.check(pool)
    engine.abort(1)
    var idle = engine.step(runner, pool)
    assert_equal(len(idle.events), 0)
    assert_equal(idle.total_tokens, 0)
    assert_equal(runner.steps, executed)



def test_reserved_impossible_requests_reject_before_mutation() raises:
    var engine = EngineCore(2, 4, 16, 100, reserve_lifetime=True)
    for maximum in [9, Int.MAX]:
        with assert_raises():
            _ = engine.add(1, [1], maximum, List[Int]())
    # Positive output still requires the entire prompt to fit.
    with assert_raises():
        _ = engine.add(1, List[Int](length=9, fill=1), 1, List[Int]())
    assert_equal(len(engine.requests), 0)
    assert_equal(engine.next_ticket, 0)
    assert_equal(engine.blocks.free_blocks(), 2)
    assert_equal(engine.add(1, [1], 8, List[Int]()), 0)
    assert_equal(engine.blocks.free_blocks(), 2)
    engine.blocks.check()


def test_reserved_peak_excludes_the_last_emitted_token() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(2, 4, 8, 100, token_budget=1, reserve_lifetime=True)
    var runner = SimulatedRunner([10, 11], 100)
    var slot = engine.add(1, [1], 7, List[Int]())
    var first = engine.step(runner, pool)
    var sequence = engine.requests[slot].sequence
    assert_equal(engine.blocks.reserved[sequence], 7)
    assert_equal(engine.blocks.length(sequence), 1)
    assert_equal(len(engine.blocks.tables[sequence]), 2)
    assert_equal(engine.blocks.free_blocks(), 0)
    var rows = first.total_tokens
    while engine.live() > 0:
        var record = engine.step(runner, pool)
        assert_equal(record.preempted, 0)
        rows += record.total_tokens
    assert_equal(rows, 7)
    assert_equal(len(engine.requests[slot].tokens), 8)
    assert_equal(engine.requests[slot].generated, 7)
    assert_equal(engine.blocks.free_blocks(), 2)
    engine.check(pool)


def test_reserved_fifo_waiter_blocks_younger_requests_without_eviction() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=4, reserve_lifetime=True)
    var runner = SimulatedRunner([10], 100)
    var a = engine.add(1, [1], 5, List[Int]())
    _ = engine.step(runner, pool)
    # Two blocks remain. The oldest waiter needs three, the younger one one.
    var b = engine.add(2, List[Int](length=9, fill=2), 4, List[Int]())
    var c = engine.add(3, [3], 1, List[Int]())
    var held_sequence = engine.requests[a].sequence
    while engine.requests[a].state != FINISHED:
        var record = engine.step(runner, pool)
        assert_equal(record.preempted, 0)
        assert_equal(record.admitted, 0)
        assert_equal(engine.requests[b].state, WAITING)
        assert_equal(engine.requests[c].state, WAITING)
        if engine.requests[a].state != FINISHED:
            assert_equal(engine.requests[a].sequence, held_sequence)
    var admitted = engine.step(runner, pool)
    assert_equal(admitted.admitted, 1)
    assert_equal(engine.requests[b].state, PREFILL)
    assert_equal(engine.blocks.reserved[engine.requests[b].sequence], 12)
    assert_equal(engine.requests[c].state, WAITING)
    _ = _drain(engine, runner, pool)


def test_reserved_watermark_counts_unselected_residents_and_yields_to_solo_work() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(3, 4, 12, 100, token_budget=4,
        watermark_blocks=2, mixed_prefill=False, reserve_lifetime=True)
    var runner = SimulatedRunner([10], 100)
    var a = engine.add(1, [1], 4, List[Int]())
    _ = engine.step(runner, pool)
    var b = engine.add(2, [2], 4, List[Int]())
    # Standalone prefill is considered with no rows selected, but A is resident.
    var waiting = engine.step(runner, pool)
    assert_equal(waiting.admitted, 0)
    assert_equal(waiting.decode_seqs, 1)
    assert_equal(engine.requests[b].state, WAITING)
    while engine.requests[a].state != FINISHED:
        _ = engine.step(runner, pool)
    var solo = engine.step(runner, pool)
    assert_equal(solo.admitted, 1)
    assert_equal(engine.requests[b].generated, 1)
    _ = _drain(engine, runner, pool)


def test_reserved_release_paths_return_unwritten_future_blocks() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=2,
        max_requests=1, watermark_blocks=4, reserve_lifetime=True)
    var runner = SimulatedRunner([10], 100)
    var slot = engine.add(1, [1, 2, 3, 4, 5], 11, List[Int]())
    _ = engine.step(runner, pool)
    assert_equal(engine.blocks.free_blocks(), 0)
    assert_equal(engine.blocks.length(engine.requests[slot].sequence), 2)
    assert_equal(engine.blocks.reserved[engine.requests[slot].sequence], 15)
    engine.abort(1)
    engine.abort(1)
    var aborted = engine.step(runner, pool)
    assert_equal(aborted.finished, 1)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.total_tokens, 0)
    assert_equal(engine.blocks.free_blocks(), 4)
    for written in pool.written:
        assert_equal(written, 0)
    # Reuse after prefill abort, then abort in decode after delivered history.
    assert_equal(engine.add(2, [2], 15, List[Int]()), slot)
    _ = engine.step(runner, pool)
    _ = engine.step(runner, pool)
    var history = engine.requests[slot].tokens.copy()
    engine.abort(2)
    var decoded_abort = engine.step(runner, pool)
    assert_equal(decoded_abort.finished, 1)
    _assert_list(engine.requests[slot].tokens, history)
    assert_equal(engine.blocks.free_blocks(), 4)
    # Early stop returns every reserved block, including those never written.
    assert_equal(engine.add(3, [3], 15, [10]), slot)
    var stopped = engine.step(runner, pool)
    assert_equal(stopped.finished, 1)
    assert_equal(engine.requests[slot].reason, "stop")
    assert_equal(engine.requests[slot].generated, 1)
    assert_equal(engine.blocks.free_blocks(), 4)
    # Zero output retains existing feasibility validation and executes no rows.
    assert_equal(engine.add(4, List[Int](length=16, fill=4), 0, List[Int]()), slot)
    var zero = engine.step(runner, pool)
    assert_equal(zero.finished, 1)
    assert_equal(zero.total_tokens, 0)
    assert_equal(engine.blocks.free_blocks(), 4)
    engine.check(pool)


def test_reserved_failure_releases_residents_and_finishes_waiters_once() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=4, reserve_lifetime=True)
    var healthy = SimulatedRunner([10], 100)
    _ = engine.add(1, [1], 12, List[Int]())
    _ = engine.step(healthy, pool)
    var history = engine.requests[0].tokens.copy()
    _ = engine.add(2, [2], 4, List[Int]())
    _ = engine.add(3, [3], 1, List[Int]())
    var broken = BrokenRunner()
    with assert_raises():
        _ = engine.step(broken, pool)
    assert_true(engine.failed)
    assert_equal(engine.live(), 0)
    assert_equal(engine.blocks.free_blocks(), 4)
    assert_equal(len(engine.failure_events), 3)
    for i in range(3):
        assert_equal(engine.failure_events[i].request_id, i + 1)
        assert_equal(engine.failure_events[i].reason, "error")
    _assert_list(engine.requests[0].tokens, history)
    with assert_raises():
        _ = engine.step(broken, pool)
    with assert_raises():
        _ = engine.add(4, [4], 1, List[Int]())
    engine.check(pool)


def test_reserved_capacity_does_not_override_fixed_or_fitted_row_limits() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 8, 4, KVGeometry(1, 1, 1))
    for fitted in [False, True]:
        var engine = EngineCore(8, 4, 32, 100, token_budget=4, reserve_lifetime=True)
        var runner = SimulatedRunner([10], 100)
        if fitted:
            engine.set_cost_policy(StepCost(0, 10, 0, 0, 0, 25))
        _ = engine.add(1, List[Int](length=17, fill=1), 5, List[Int]())
        var first = engine.step(runner, pool)
        assert_equal(engine.blocks.reserved[engine.requests[0].sequence], 21)
        assert_equal(first.prefill_tokens, 2 if fitted else 4)
        while engine.requests[0].generated == 0:
            var record = engine.step(runner, pool)
            assert_true(record.prefill_tokens <= (2 if fitted else 4))
        _ = engine.add(2, [2, 3, 4, 5, 6], 2, List[Int]())
        if fitted:
            engine.set_cost_policy(StepCost(0, 0, 0, 100, 0, 150))
        var mixed = engine.step(runner, pool)
        assert_equal(mixed.decode_seqs, 1)
        assert_equal(mixed.prefill_tokens, 1 if fitted else 3)
        assert_true(mixed.total_tokens <= 4)
        _ = _drain(engine, runner, pool)


def test_reserved_finite_arrivals_drain_across_capacity_and_scheduler_bounds() raises:
    var ctx = DeviceContext()
    for blocks in [1, 2, 3, 5]:
        var pool = KVPool(ctx, blocks, 4, KVGeometry(1, 1, 1))
        for budget in [1, 2, 4, 8]:
            for sequences in [1, 3]:
                for mixed in [False, True]:
                    for margin in [0, blocks // 2, blocks]:
                        var engine = EngineCore(blocks, 4, blocks * 4, 100,
                            token_budget=budget, max_sequences=sequences, max_requests=4,
                            watermark_blocks=margin, mixed_prefill=mixed, reserve_lifetime=True)
                        var runner = SimulatedRunner([10, 11, 12], 100)
                        var next_id = 0
                        var finished = 0
                        var tick = 0
                        while (next_id < 16 or engine.live() > 0) and tick < 500:
                            if next_id < 16 and engine.live() < 4:
                                var length = 1 + next_id % min(blocks * 4 - 1, 7)
                                var maximum = min(3, blocks * 4 - length)
                                _ = engine.add(next_id, List[Int](length=length, fill=next_id % 10),
                                    maximum, [10] if next_id % 3 == 0 else List[Int]())
                                if next_id % 7 == 0:
                                    engine.abort(next_id)
                                next_id += 1
                            var record = engine.step(runner, pool)
                            assert_equal(record.preempted, 0)
                            assert_true(record.total_tokens <= budget)
                            finished += record.finished
                            engine.check(pool)
                            tick += 1
                        assert_equal(next_id, 16)
                        assert_equal(finished, 16)
                        assert_equal(engine.live(), 0)
                        assert_equal(engine.blocks.free_blocks(), blocks)


def test_reserved_pressure_removes_replay_and_preserves_histories() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 40, 32, KVGeometry(1, 1, 1))
    var reference_histories = List[List[Int]]()
    var lengths = [32, 128, 64, 512, 96, 1024, 256, 64]
    for reserved in [False, True]:
        var engine = EngineCore(40, 32, 4096, 100, token_budget=256,
            max_sequences=8, reserve_lifetime=reserved)
        var runner = SimulatedRunner([10, 11, 12], 100)
        for i in range(8):
            _ = engine.add(i, List[Int](length=lengths[i], fill=i), 32, List[Int]())
        var rows = 0
        var preemptions = 0
        var delivered = List[Int](length=8, fill=0)
        var steps = 0
        while engine.live() > 0 and steps < 1000:
            var record = engine.step(runner, pool)
            rows += record.total_tokens
            preemptions += record.preempted
            for event in record.events:
                if event.kind == TOKEN_EVENT:
                    delivered[event.request_id] += 1
                    assert_equal(event.generated_tokens, delivered[event.request_id])
            steps += 1
        assert_equal(engine.live(), 0)
        assert_equal(engine.blocks.free_blocks(), 40)
        assert_equal(rows, 2424 if reserved else 16432)
        assert_equal(preemptions, 0 if reserved else 57)
        for i in range(8):
            assert_equal(delivered[i], 32)
            if reserved:
                _assert_list(engine.requests[i].tokens, reference_histories[i])
            else:
                reference_histories.append(engine.requests[i].tokens.copy())
        engine.check(pool)


def test_kv_observations_capture_same_step_admission_stop_and_future_blocks() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=2,
        reserve_lifetime=True, observe_kv=True)
    var runner = SimulatedRunner([10], 100)
    _ = engine.add(7, [1], 15, [10], 0)
    var record = engine.step(runner, pool)
    assert_equal(record.admitted, 1)
    assert_equal(record.finished, 1)
    assert_equal(record.admitted_request_id, 7)
    assert_equal(record.admitted_ns, record.begin_ns)
    assert_equal(len(record.kv_observations), 4)
    for phase in range(4):
        assert_equal(record.kv_observations[phase].phase, phase)
    var start = record.kv_observations[0]
    var scheduled = record.kv_observations[1]
    var executed = record.kv_observations[2]
    var end = record.kv_observations[3]
    assert_equal(start.allocated_blocks, 0)
    assert_equal(start.waiting_requests, 1)
    assert_equal(scheduled.allocated_blocks, 4)
    assert_equal(scheduled.reserved_tokens, 15)
    assert_equal(scheduled.written_blocks, 0)
    assert_equal(scheduled.written_tokens, 0)
    assert_equal(scheduled.waiting_requests, 0)
    assert_equal(scheduled.resident_requests, 1)
    # The early stop frees three completely unwritten future blocks. Before
    # postprocessing the pool retains one written row in its held capacity.
    assert_equal(executed.allocated_blocks, 4)
    assert_equal(executed.reserved_tokens, 15)
    assert_equal(executed.written_blocks, 1)
    assert_equal(executed.written_tokens, 1)
    assert_equal(end.allocated_blocks, 0)
    assert_equal(end.written_blocks, 0)
    assert_equal(end.written_tokens, 0)
    assert_equal(end.reserved_tokens, 0)
    assert_equal(end.resident_requests, 0)
    assert_equal(record.execute_end_ns - record.execute_begin_ns, record.execute_ns)
    assert_equal(executed.at_ns, record.execute_end_ns)
    assert_equal(end.at_ns, record.end_ns)
    engine.check(pool)


def test_kv_observations_count_waiting_abort_reuse_and_zero_output() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 4, 4, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 4, 16, 100, token_budget=2,
        reserve_lifetime=True, observe_kv=True)
    var runner = SimulatedRunner([10], 100)
    _ = engine.add(1, [1, 2, 3, 4, 5], 11, List[Int]())
    var first = engine.step(runner, pool)
    assert_equal(first.kv_observations[2].written_tokens, 2)
    assert_equal(first.kv_observations[3].allocated_blocks, 4)
    _ = engine.add(2, [2], 1, List[Int]())
    var waiting = engine.step(runner, pool)
    assert_equal(waiting.admitted_request_id, -1)
    assert_equal(waiting.admitted_ns, -1)
    assert_equal(waiting.kv_observations[3].waiting_requests, 1)
    assert_equal(waiting.kv_observations[3].resident_requests, 1)
    assert_equal(waiting.kv_observations[3].written_tokens, 4)
    engine.abort(1)
    var aborted = engine.step(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.finished, 2)
    assert_equal(aborted.admitted_request_id, 2)
    assert_equal(aborted.kv_observations[0].allocated_blocks, 4)
    assert_equal(aborted.kv_observations[0].written_tokens, 4)
    assert_equal(aborted.kv_observations[1].allocated_blocks, 1)
    assert_equal(aborted.kv_observations[1].written_tokens, 0)
    assert_equal(aborted.kv_observations[2].written_tokens, 1)
    assert_equal(aborted.kv_observations[3].allocated_blocks, 0)
    _ = engine.add(3, [3], 0, List[Int]())
    var zero = engine.step(runner, pool)
    assert_equal(zero.finished, 1)
    assert_equal(zero.admitted, 0)
    assert_equal(zero.admitted_request_id, -1)
    assert_equal(zero.admitted_ns, -1)
    assert_equal(zero.execute_begin_ns, zero.execute_end_ns)
    assert_equal(zero.execute_ns, 0)
    for point in zero.kv_observations:
        assert_equal(point.allocated_blocks, 0)
        assert_equal(point.written_tokens, 0)
        assert_equal(point.resident_requests, 0)
    engine.check(pool)


def test_kv_observation_opt_in_preserves_decisions_work_and_histories() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 3, 4, KVGeometry(1, 1, 1))
    for reserved in [False, True]:
        var records = List[EngineStep]()
        var histories = List[List[Int]]()
        for observed in [False, True]:
            var engine = EngineCore(3, 4, 16, 100, token_budget=4,
                reserve_lifetime=reserved, observe_kv=observed)
            var runner = SimulatedRunner([10, 11, 12], 100)
            _ = engine.add(1, [1, 2, 3, 4], 8, List[Int]())
            _ = engine.add(2, [5, 6, 7, 8], 8, List[Int]())
            var step = 0
            var admissions = 0
            while engine.live() > 0 and step < 100:
                var record = engine.step(runner, pool)
                admissions += record.admitted
                if observed:
                    assert_equal(record.total_tokens, records[step].total_tokens)
                    assert_equal(record.prefill_tokens, records[step].prefill_tokens)
                    assert_equal(record.decode_seqs, records[step].decode_seqs)
                    assert_equal(record.admitted, records[step].admitted)
                    assert_equal(record.preempted, records[step].preempted)
                    assert_equal(record.finished, records[step].finished)
                    assert_equal(record.waiting, records[step].waiting)
                    assert_equal(record.blocks_free, records[step].blocks_free)
                    assert_equal(record.begin_ns, records[step].begin_ns)
                    assert_equal(record.end_ns, records[step].end_ns)
                    assert_equal(len(record.events), len(records[step].events))
                    assert_equal(len(record.kv_observations), 4)
                    assert_equal(record.admitted_request_id >= 0, record.admitted == 1)
                    for i in range(len(record.events)):
                        assert_equal(record.events[i].kind, records[step].events[i].kind)
                        assert_equal(record.events[i].request_id, records[step].events[i].request_id)
                        assert_equal(record.events[i].token_id, records[step].events[i].token_id)
                        assert_equal(record.events[i].reason, records[step].events[i].reason)
                        assert_equal(record.events[i].generated_tokens, records[step].events[i].generated_tokens)
                        assert_equal(record.events[i].emitted_ns, records[step].events[i].emitted_ns)
                else:
                    assert_equal(len(record.kv_observations), 0)
                    assert_equal(record.admitted_request_id, -1)
                    records.append(record^)
                step += 1
            assert_equal(engine.live(), 0)
            assert_equal(engine.blocks.free_blocks(), 3)
            assert_true(admissions == 2 if reserved else admissions > 2)
            if observed:
                assert_equal(step, len(records))
                for i in range(2):
                    _assert_list(engine.requests[i].tokens, histories[i])
            else:
                for i in range(2):
                    histories.append(engine.requests[i].tokens.copy())
            engine.check(pool)


def test_fixed_budget_matrix_counts_decode_rows_and_keeps_lifetime_capacity() raises:
    """Seven decodes consume rows while a long prompt owns nine future blocks."""
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 32, 32, KVGeometry(1, 1, 1))
    var reference_histories = List[List[Int]]()
    for budget in [32, 64, 128, 256]:
        var engine = EngineCore(32, 32, 1024, 100, token_budget=budget,
            max_sequences=8, max_requests=8, reserve_lifetime=True)
        var runner = SimulatedRunner([10, 11, 12], 100)
        var rows = 0
        # Introduce seven one-row prompts on successive boundaries. Each
        # becomes a decode, and its sixteen-token budget outlasts all arrivals.
        for i in range(7):
            _ = engine.add(i, [i + 1], 16, List[Int]())
            var seeded = engine.step(runner, pool)
            rows += seeded.total_tokens
            assert_equal(seeded.preempted, 0)
        var long_prompt = engine.add(7, List[Int](length=257, fill=17), 2, List[Int]())
        var mixed = engine.step(runner, pool)
        rows += mixed.total_tokens
        assert_equal(mixed.decode_seqs, 7)
        assert_equal(mixed.prefill_seqs, 1)
        assert_equal(mixed.total_tokens, budget)
        assert_equal(mixed.prefill_tokens, budget - 7)
        assert_equal(mixed.admitted, 1)
        assert_equal(mixed.preempted, 0)
        assert_equal(len(mixed.events), 7)
        for event in mixed.events:
            assert_equal(event.kind, TOKEN_EVENT)
            assert_true(event.request_id < 7)
        var sequence = engine.requests[long_prompt].sequence
        assert_equal(engine.requests[long_prompt].state, PREFILL)
        assert_equal(engine.requests[long_prompt].generated, 0)
        assert_equal(engine.blocks.length(sequence), budget - 7)
        assert_equal(engine.blocks.reserved[sequence], 258)
        assert_equal(len(engine.blocks.tables[sequence]), 9)
        # Seven one-block decodes plus nine prompt blocks are held regardless
        # of the number of prompt rows selected for the current step.
        assert_equal(engine.blocks.free_blocks(), 16)
        engine.check(pool)
        var steps = 0
        while engine.live() > 0 and steps < 1000:
            var record = engine.step(runner, pool)
            assert_true(record.total_tokens <= budget)
            assert_equal(record.preempted, 0)
            rows += record.total_tokens
            steps += 1
        assert_equal(engine.live(), 0)
        assert_equal(rows, 370)  # 7 * (1 + 16 - 1) + (257 + 2 - 1).
        assert_equal(engine.blocks.free_blocks(), 32)
        for i in range(8):
            assert_equal(engine.requests[i].generated, 16 if i < 7 else 2)
            if budget == 32:
                reference_histories.append(engine.requests[i].tokens.copy())
            else:
                _assert_list(engine.requests[i].tokens, reference_histories[i])
        engine.check(pool)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
