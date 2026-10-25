"""Native scheduler safety, accounting and finite-workload liveness.

The simulator only advances logical KV counts; these are not model numerical
tests. A small Metal allocation supplies the existing KVPool container.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from max.gpu.host import DeviceContext
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.blocks import PARTIAL, REGISTERED
from llm_mojo.serving.engine import _frequency_threshold, EngineCore, EngineStep, StepCost, WAITING, PREFILL, DECODE, FINISHED, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVGeometry, KVPool
from llm_mojo.serving.runner import AsyncModelRunner, ModelRunner, SimulatedRunner


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


def _prefix_prompt(length: Int = 65, variant: Int = 0) -> List[Int]:
    var result = List[Int](capacity=length)
    for position in range(length):
        result.append((position + variant) % 100)
    return result^


def _prefix_drain(mut engine: EngineCore, mut runner: SimulatedRunner, mut pool: KVPool,
                  bound: Int = 3000) raises -> Int:
    var rows = 0
    var steps = 0
    while engine.live() > 0 and steps < bound:
        var record = engine.step(runner, pool)
        rows += record.total_tokens
        engine.check(pool)
        steps += 1
    assert_equal(engine.live(), 0)
    assert_equal(engine.cache_references(), 0)
    assert_equal(engine.private_blocks(), 0)
    assert_equal(engine.blocks.free_blocks() + engine.cached_blocks(), pool.blocks)
    return rows


def test_prefix_cache_reuses_exact_history_and_preserves_terminal_accounting() raises:
    var ctx = DeviceContext()
    for reserved in [False, True]:
        var runner = SimulatedRunner([10, 11, 12], 100)
        var pool = KVPool(ctx, 8, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(8, 32, 256, 100, token_budget=32, reserve_lifetime=reserved,
            prefix_caching=True, prefix_namespace=runner.prefix_cache_identity())
        _ = engine.add(0, _prefix_prompt(), 4, List[Int](), cache_salt="one")
        assert_equal(_prefix_drain(engine, runner, pool), 68)
        var expected = engine.requests[0].tokens.copy()
        assert_equal(engine.cached_blocks(), 2)
        _ = engine.add(1, _prefix_prompt(), 4, List[Int](), cache_salt="one")
        var hit = engine.step(runner, pool)
        assert_equal(hit.cache_hit_tokens, 64)
        assert_equal(hit.prefill_tokens, 1)
        assert_equal(hit.cache_hits[0].request_id, 1)
        assert_equal(_prefix_drain(engine, runner, pool), 3)
        _assert_list(engine.requests[0].tokens, expected)
        _ = engine.add(2, _prefix_prompt(), 1, List[Int](), cache_salt="different")
        var miss = engine.step(runner, pool)
        assert_equal(miss.cache_hit_tokens, 0)
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.clear_cache(pool)
        assert_equal(engine.cached_blocks(), 0)
        assert_equal(engine.blocks.free_blocks(), pool.blocks)
        for written in pool.written:
            assert_equal(written, 0)


def test_prefix_cache_shared_reader_abort_keeps_the_other_reader_valid() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10, 11, 12], 100)
    var pool = KVPool(ctx, 5, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(5, 32, 128, 100, token_budget=32, reserve_lifetime=True,
        prefix_caching=True, prefix_namespace=runner.prefix_cache_identity())
    _ = engine.add(0, _prefix_prompt(), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.add(1, _prefix_prompt(), 5, List[Int]())
    _ = engine.step(runner, pool)
    _ = engine.add(2, _prefix_prompt(), 5, List[Int]())
    _ = engine.step(runner, pool)
    var shared = engine.prefix_cache.entries[0].block
    assert_equal(engine.blocks.refs[shared], 2)
    with assert_raises():
        _ = engine.clear_cache(pool)
    engine.abort(1)
    var boundary = engine.step(runner, pool)
    assert_equal(boundary.aborted, 1)
    assert_equal(engine.blocks.refs[shared], 1)
    assert_equal(pool.written[shared], 32)
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.clear_cache(pool)
    engine.check(pool)


def test_prefix_cache_pressure_drains_and_evicts_before_live_preemption() raises:
    var ctx = DeviceContext()
    for reserved in [False, True]:
        var runner = SimulatedRunner([10, 11, 12], 100)
        var pool = KVPool(ctx, 5, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(5, 32, 160, 100, token_budget=32, max_sequences=4,
            reserve_lifetime=reserved, prefix_caching=True,
            prefix_namespace=runner.prefix_cache_identity())
        _ = engine.add(0, _prefix_prompt(), 1, List[Int]())
        _ = _prefix_drain(engine, runner, pool)
        for request in range(1, 9):
            _ = engine.add(request, _prefix_prompt(65, request % 2), 65, List[Int]())
        _ = _prefix_drain(engine, runner, pool, 5000)
        for request in range(len(engine.requests)):
            assert_equal(engine.requests[request].generated, 65)
            assert_equal(engine.requests[request].reason, "length")
        _ = engine.clear_cache(pool)
        engine.check(pool)


def test_prefix_cache_zero_output_does_not_lookup_or_change_retained_blocks() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 3, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(3, 32, 256, 100, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity())
    _ = engine.add(0, _prefix_prompt(), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    var cached = engine.cached_blocks()
    var clock = engine.prefix_cache.clock
    _ = engine.add(1, _prefix_prompt(200), 0, List[Int]())
    var empty = engine.step(runner, pool)
    assert_equal(empty.total_tokens, 0)
    assert_equal(empty.cache_hit_tokens, 0)
    assert_equal(len(empty.cache_hits), 0)
    assert_equal(empty.finished, 1)
    assert_equal(engine.cached_blocks(), cached)
    assert_equal(engine.prefix_cache.clock, clock)
    _ = engine.clear_cache(pool)


struct PrefixFaultRunner(ModelRunner):
    var oracle: SimulatedRunner
    var fail: Bool

    def __init__(out self) raises:
        self.oracle = SimulatedRunner([10], 100)
        self.fail = False

    def prefix_cache_identity(self) -> String:
        return self.oracle.prefix_cache_identity()

    def now_ns(self) -> Int:
        return self.oracle.now_ns()

    def execute(mut self, batch: StepBatch, mut pool: KVPool) raises -> List[Int]:
        var output = self.oracle.execute(batch, pool)
        if self.fail:
            raise Error("injected prefix-cache runner fault after writes")
        return output^


def test_prefix_cache_fault_flushes_earlier_publication_and_all_request_refs() raises:
    var ctx = DeviceContext()
    var runner = PrefixFaultRunner()
    var pool = KVPool(ctx, 4, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 32, 128, 100, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity())
    _ = engine.add(0, _prefix_prompt(), 1, List[Int]())
    _ = engine.step(runner, pool)
    assert_equal(engine.cached_blocks(), 2)
    _ = engine.add(1, _prefix_prompt(), 4, List[Int]())
    runner.fail = True
    with assert_raises():
        _ = engine.step(runner, pool)
    assert_true(engine.failed)
    assert_equal(engine.live(), 0)
    assert_equal(engine.cached_blocks(), 0)
    assert_equal(engine.cache_references(), 0)
    assert_equal(engine.blocks.free_blocks(), pool.blocks)
    for written in pool.written:
        assert_equal(written, 0)
    assert_true(len(engine.failure_cache_events) > 0)
    assert_equal(engine.failure_cache_events[len(engine.failure_cache_events) - 1].kind, "cleared")
    engine.check(pool)


def test_prefix_cache_requires_qualified_namespace_and_rejects_runner_changes_atomically() raises:
    with assert_raises():
        _ = EngineCore(4, 32, 128, 100, prefix_caching=True)
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 4, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 32, 128, 100, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity())
    _ = engine.add(0, _prefix_prompt(), 1, List[Int]())
    var other = SimulatedRunner([11], 100)
    with assert_raises():
        _ = engine.step(other, pool)
    assert_equal(engine.step_id, 0)
    assert_equal(engine.requests[0].sequence, -1)
    assert_equal(engine.cached_blocks(), 0)
    var identity = engine.prefix_namespace.copy()
    engine.prefix_namespace = ""
    with assert_raises():
        _ = engine.step(runner, pool)
    assert_equal(engine.step_id, 0)
    engine.prefix_namespace = identity
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.clear_cache(pool)


def test_prefix_cache_clear_rejects_the_wrong_pool_before_mutation() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 4, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 32, 128, 100, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity())
    _ = engine.add(0, _prefix_prompt(), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    var cached = engine.cached_blocks()
    var event_id = engine.prefix_cache.event_id
    var wrong = KVPool(ctx, 2, 32, KVGeometry(1, 1, 1))
    with assert_raises():
        _ = engine.clear_cache(wrong)
    var empty = KVPool(ctx, 4, 32, KVGeometry(1, 1, 1))
    with assert_raises():
        _ = engine.clear_cache(empty)
    assert_equal(engine.cached_blocks(), cached)
    assert_equal(engine.prefix_cache.event_id, event_id)
    engine.check(pool)
    _ = engine.clear_cache(pool)


def test_active_prefill_cap_validates_and_keeps_the_idle_budget() raises:
    for invalid in [-1, 4097]:
        with assert_raises():
            _ = EngineCore(16, 32, 512, 100, active_prefill_cap=invalid)
    var ctx = DeviceContext()
    for cap in [0, 16, 32, 4096]:
        var pool = KVPool(ctx, 16, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(16, 32, 512, 100, token_budget=256,
            reserve_lifetime=True, active_prefill_cap=cap)
        var runner = SimulatedRunner([10], 100)
        var slot = engine.add(0, List[Int](length=300, fill=1), 2, List[Int]())
        var first = engine.step(runner, pool)
        assert_equal(first.prefill_tokens, 256)
        assert_equal(first.total_tokens, 256)
        assert_equal(len(first.events), 0)
        assert_equal(engine.blocks.reserved[engine.requests[slot].sequence], 301)
        var final = engine.step(runner, pool)
        assert_equal(final.decode_seqs, 0)
        assert_equal(final.prefill_tokens, 44)
        assert_equal(engine.requests[slot].generated, 1)
        _ = _drain(engine, runner, pool)


def test_active_prefill_cap_splits_mixed_prompts_without_changing_histories() raises:
    var ctx = DeviceContext()
    for reserved in [False, True]:
        var reference = List[List[Int]]()
        for cap in [0, 16, 32]:
            var pool = KVPool(ctx, 12, 32, KVGeometry(1, 1, 1))
            var engine = EngineCore(12, 32, 256, 100, token_budget=256,
                reserve_lifetime=reserved, active_prefill_cap=cap)
            var runner = SimulatedRunner([10, 11, 12], 100)
            var incumbent = engine.add(0, [1], 12, List[Int]())
            var first = engine.step(runner, pool)
            var rows = first.total_tokens
            var newcomer = engine.add(1, List[Int](length=65, fill=2), 2, List[Int]())
            var completed = 0
            var mixed_steps = 0
            while completed < 65:
                var record = engine.step(runner, pool)
                var expected = min(65 - completed, 255 if cap == 0 else cap)
                completed += expected
                mixed_steps += 1
                rows += record.total_tokens
                assert_equal(record.decode_seqs, 1)
                assert_equal(record.prefill_tokens, expected)
                assert_equal(record.total_tokens, expected + 1)
                assert_equal(record.budget_limited, 0)
                assert_equal(engine.requests[incumbent].generated, mixed_steps + 1)
                assert_equal(engine.requests[newcomer].generated, 1 if completed == 65 else 0)
                assert_equal(len(record.events), 2 if completed == 65 else 1)
                assert_equal(engine.requests[newcomer].state, DECODE if completed == 65 else PREFILL)
                engine.check(pool)
            assert_equal(mixed_steps, 1 if cap == 0 else 65 // cap + 1)
            rows += _prefix_drain(engine, runner, pool)
            assert_equal(rows, 78)  # (1 + 12 - 1) + (65 + 2 - 1).
            for slot in [incumbent, newcomer]:
                if cap == 0:
                    reference.append(engine.requests[slot].tokens.copy())
                else:
                    _assert_list(engine.requests[slot].tokens, reference[slot])


def test_active_prefill_cap_restores_idle_work_when_the_incumbent_finishes() raises:
    var ctx = DeviceContext()
    for cap in [16, 32]:
        var pool = KVPool(ctx, 8, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(8, 32, 256, 100, token_budget=256,
            reserve_lifetime=True, active_prefill_cap=cap)
        var runner = SimulatedRunner([10], 100)
        _ = engine.add(0, [1], 2, List[Int]())
        _ = engine.step(runner, pool)
        var newcomer = engine.add(1, List[Int](length=100, fill=2), 2, List[Int]())
        var mixed = engine.step(runner, pool)
        assert_equal(mixed.decode_seqs, 1)
        assert_equal(mixed.prefill_tokens, cap)
        assert_equal(mixed.finished, 1)
        assert_equal(engine.requests[newcomer].generated, 0)
        var idle = engine.step(runner, pool)
        assert_equal(idle.decode_seqs, 0)
        assert_equal(idle.prefill_tokens, 100 - cap)
        assert_true(idle.prefill_tokens > cap)
        assert_equal(engine.requests[newcomer].generated, 1)
        _ = _drain(engine, runner, pool)


def test_active_prefill_cap_respects_the_remaining_global_budget() raises:
    var ctx = DeviceContext()
    for cap in [16, 32]:
        var pool = KVPool(ctx, 8, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(8, 32, 256, 100, token_budget=8,
            reserve_lifetime=True, active_prefill_cap=cap)
        var runner = SimulatedRunner([10], 100)
        _ = engine.add(0, [1], 4, List[Int]())
        _ = engine.step(runner, pool)
        _ = engine.add(1, List[Int](length=33, fill=2), 2, List[Int]())
        var mixed = engine.step(runner, pool)
        assert_equal(mixed.decode_seqs, 1)
        assert_equal(mixed.prefill_tokens, 7)
        assert_equal(mixed.total_tokens, 8)
        _ = _drain(engine, runner, pool)


def test_active_prefill_cap_uses_cached_past_and_keeps_partial_blocks_private() raises:
    var ctx = DeviceContext()
    for reserved in [False, True]:
        for cap in [16, 32]:
            for abort_partial in [False, True]:
                var runner = SimulatedRunner([10, 11, 12], 100)
                var pool = KVPool(ctx, 8, 32, KVGeometry(1, 1, 1))
                var engine = EngineCore(8, 32, 256, 100, token_budget=256,
                    reserve_lifetime=reserved, prefix_caching=True,
                    prefix_namespace=runner.prefix_cache_identity(), active_prefill_cap=cap)
                _ = engine.add(0, _prefix_prompt(33), 1, List[Int]())
                _ = _prefix_drain(engine, runner, pool)
                var shared = engine.prefix_cache.entries[0].block
                var incumbent = engine.add(1, _prefix_prompt(33), 10, List[Int]())
                _ = engine.step(runner, pool)
                var newcomer = engine.add(2, _prefix_prompt(97), 2, List[Int]())
                var first = engine.step(runner, pool)
                assert_equal(first.cache_hit_tokens, 32)
                assert_equal(first.cache_hits[0].request_id, 2)
                assert_equal(first.prefill_tokens, cap)
                assert_equal(first.attended_positions, 34 + cap * 32 + cap * (cap + 1) // 2)
                assert_equal(engine.requests[newcomer].generated, 0)
                var sequence = engine.requests[newcomer].sequence
                var tail = engine.blocks.tables[sequence][1]
                assert_equal(engine.blocks.length(sequence), 32 + cap)
                assert_equal(pool.written[tail], cap)
                assert_equal(engine.blocks.states[tail], PARTIAL if cap == 16 else REGISTERED)
                assert_equal(engine.cached_blocks(), 1 if cap == 16 else 2)
                assert_equal(engine.blocks.refs[shared], 2)
                assert_equal(pool.written[shared], 32)
                if reserved:
                    assert_equal(engine.blocks.reserved[sequence], 98)
                if abort_partial:
                    engine.abort(2)
                    var boundary = engine.step(runner, pool)
                    assert_equal(boundary.aborted, 1)
                    assert_equal(boundary.prefill_tokens, 0)
                    assert_equal(engine.requests[newcomer].generated, 0)
                    assert_equal(engine.blocks.refs[shared], 1)
                    assert_equal(pool.written[tail], 0 if cap == 16 else 32)
                elif cap == 16:
                    var second = engine.step(runner, pool)
                    assert_equal(second.prefill_tokens, 16)
                    assert_equal(engine.blocks.length(sequence), 64)
                    assert_equal(pool.written[tail], 32)
                    assert_equal(engine.blocks.states[tail], REGISTERED)
                    assert_equal(engine.cached_blocks(), 2)
                _ = _prefix_drain(engine, runner, pool)
                assert_equal(engine.requests[incumbent].generated, 10)
                assert_equal(engine.requests[newcomer].generated, 0 if abort_partial else 2)
                assert_equal(engine.requests[newcomer].reason, "abort" if abort_partial else "length")
                _ = engine.clear_cache(pool)
                assert_equal(engine.blocks.free_blocks(), pool.blocks)
                for written in pool.written:
                    assert_equal(written, 0)
                engine.check(pool)


def test_active_prefill_cap_and_fitted_cost_keep_decode_and_idle_progress() raises:
    var ctx = DeviceContext()
    for cap in [16, 32]:
        var pool = KVPool(ctx, 8, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(8, 32, 256, 100, token_budget=256,
            reserve_lifetime=True, active_prefill_cap=cap)
        var runner = SimulatedRunner([10], 100)
        var incumbent = engine.add(0, [1], 4, List[Int]())
        _ = engine.step(runner, pool)
        var newcomer = engine.add(1, List[Int](length=100, fill=2), 1, List[Int]())
        engine.set_cost_policy(StepCost(0, 1, 0, 0, 0, 5))
        var fitted = engine.step(runner, pool)
        assert_equal(fitted.decode_seqs, 1)
        assert_equal(fitted.prefill_tokens, 4)
        assert_equal(fitted.budget_limited, 1)
        engine.set_cost_policy(StepCost(1000, 1, 0, 0, 0, 1))
        for generated in [3, 4]:
            var decodes = engine.step(runner, pool)
            assert_equal(decodes.decode_seqs, 1)
            assert_equal(decodes.prefill_tokens, 0)
            assert_equal(engine.requests[incumbent].generated, generated)
        var idle = engine.step(runner, pool)
        assert_equal(idle.decode_seqs, 0)
        assert_equal(idle.prefill_tokens, 1)
        assert_equal(engine.blocks.length(engine.requests[newcomer].sequence), 5)
        engine.set_cost_policy(StepCost(0, 0, 0, 0, 0, 1))
        var rest = engine.step(runner, pool)
        assert_equal(rest.prefill_tokens, 95)
        assert_equal(engine.live(), 0)
        assert_equal(engine.blocks.free_blocks(), pool.blocks)
        engine.check(pool)


def test_active_prefill_cap_fitted_positions_start_after_the_cached_prefix() raises:
    var ctx = DeviceContext()
    for cap in [16, 32]:
        var runner = SimulatedRunner([10], 100)
        var pool = KVPool(ctx, 8, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(8, 32, 256, 100, token_budget=256,
            reserve_lifetime=True, prefix_caching=True,
            prefix_namespace=runner.prefix_cache_identity(), active_prefill_cap=cap)
        _ = engine.add(0, _prefix_prompt(33), 1, List[Int]())
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.add(1, _prefix_prompt(33), 5, List[Int]())
        _ = engine.step(runner, pool)
        _ = engine.add(2, _prefix_prompt(97), 2, List[Int]())
        engine.set_cost_policy(StepCost(0, 0, 1, 0, 0, 100))
        var fitted = engine.step(runner, pool)
        assert_equal(fitted.cache_hit_tokens, 32)
        assert_equal(fitted.decode_seqs, 1)
        assert_equal(fitted.prefill_tokens, 1)
        assert_equal(fitted.attended_positions, 67)  # Existing decode 34, new query 33.
        assert_equal(fitted.predicted_ns, 67)
        assert_equal(fitted.budget_limited, 1)
        engine.set_cost_policy(StepCost(0, 0, 0, 0, 0, 1))
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.clear_cache(pool)
        engine.check(pool)


struct ActiveCapAsyncProbe(AsyncModelRunner):
    var calls: Int

    def __init__(out self):
        self.calls = 0

    def now_ns(self) -> Int:
        return 0

    def submit(mut self, batch: StepBatch, source_indices: List[Int], source_ticket: Int,
               mut pool: KVPool) raises -> Int:
        self.calls += 1
        raise Error("unexpected asynchronous submission")

    def collect(mut self, ticket: Int) raises -> List[Int]:
        self.calls += 1
        raise Error("unexpected asynchronous collection")

    def drain(mut self) raises:
        self.calls += 1


def test_active_prefill_cap_rejects_async_before_boundary_mutation() raises:
    var ctx = DeviceContext()
    for cap in [16, 32]:
        var pool = KVPool(ctx, 4, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(4, 32, 128, 100, active_prefill_cap=cap)
        var runner = ActiveCapAsyncProbe()
        var slot = engine.add(0, [1], 2, List[Int]())
        engine.abort(0)
        with assert_raises(contains="synchronous"):
            _ = engine.step_async(runner, pool)
        assert_equal(runner.calls, 0)
        assert_equal(engine.step_id, 0)
        assert_equal(engine.async_enabled, False)
        assert_equal(len(engine.async_pending), 0)
        assert_equal(engine.requests[slot].state, WAITING)
        assert_equal(engine.requests[slot].sequence, -1)
        assert_equal(engine.blocks.free_blocks(), pool.blocks)
        var healthy = SimulatedRunner([10], 100)
        var boundary = engine.step(healthy, pool)
        assert_equal(boundary.aborted, 1)
        assert_equal(engine.live(), 0)
        engine.check(pool)


def _adaptive_admission_case(mut ctx: DeviceContext, remaining: Int, blocks: Int,
                             expected_reason: String, expected_count: Int,
                             decodes: Int = 1, incumbent_maximum: Int = 16,
                             maximum: Int = 16, max_sequences: Int = 8,
                             watermark: Int = 0, budget: Int = 256,
                             adaptive: Bool = True, fixed_cap: Int = 0) raises:
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, blocks, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(blocks, 32, 256, 100, token_budget=budget,
        max_sequences=max_sequences, watermark_blocks=watermark,
        reserve_lifetime=True, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity(), active_prefill_cap=fixed_cap,
        adaptive_prefill=adaptive, observe_prefill_policy=True)
    _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    for i in range(decodes):
        _ = engine.add(i, _prefix_prompt(33), incumbent_maximum, List[Int]())
        _ = engine.step(runner, pool)
    var newcomer = engine.add(90, _prefix_prompt(32 + remaining), maximum, List[Int]())
    var record = engine.step(runner, pool)
    assert_equal(len(record.prefill_decisions), 1)
    var decision = record.prefill_decisions[0].copy()
    assert_equal(decision.request_id, 90)
    assert_equal(decision.matched_tokens, 32)
    assert_equal(decision.remaining_prompt_tokens, remaining)
    assert_equal(decision.baseline_tokens, min(budget - decodes, remaining))
    assert_equal(decision.available_blocks,
        max(decision.free_blocks + decision.reclaimable_blocks - decision.watermark_blocks, 0))
    assert_equal(decision.required_blocks, (32 + remaining + maximum - 1 + 31) // 32 - 1)
    assert_equal(decision.selected_decodes, decodes)
    assert_equal(decision.reason, expected_reason)
    assert_equal(decision.chosen_tokens, expected_count)
    assert_equal(decision.applied, expected_reason == "adaptive_short_tail")
    assert_equal(record.prefill_tokens, expected_count)
    assert_equal(record.admitted, 0 if expected_count == 0 else 1)
    var row = 0
    assert_equal(len(record.selections), decodes + (1 if expected_count > 0 else 0))
    for selected in record.selections:
        if selected.request_id == 90:
            assert_equal(selected.past, 32)
            assert_equal(selected.query_tokens, expected_count)
            assert_equal(selected.selected_head, row + expected_count - 1 if expected_count == remaining else -1)
        else:
            assert_equal(selected.query_tokens, 1)
            assert_equal(selected.selected_head, row)
        row += selected.query_tokens
    assert_equal(row, record.total_tokens)
    if expected_count > 0:
        assert_equal(engine.blocks.reserved[engine.requests[newcomer].sequence], 32 + remaining + maximum - 1)
    _ = _prefix_drain(engine, runner, pool)
    assert_equal(engine.requests[newcomer].generated, maximum)
    _ = engine.clear_cache(pool)
    assert_equal(engine.blocks.free_blocks(), blocks)
    engine.check(pool)


def test_adaptive_prefill_exact_tail_and_capacity_boundaries() raises:
    var ctx = DeviceContext()
    _adaptive_admission_case(ctx, 32, 4, "tail_length", 32)
    _adaptive_admission_case(ctx, 33, 4, "adaptive_short_tail", 32)
    _adaptive_admission_case(ctx, 64, 5, "adaptive_short_tail", 32)
    _adaptive_admission_case(ctx, 65, 5, "tail_length", 65)
    # K=3; pool occupancy before admission is one shared plus one private.
    _adaptive_admission_case(ctx, 64, 4, "admission_blocked", 0)  # A=K-1.
    _adaptive_admission_case(ctx, 64, 5, "adaptive_short_tail", 32)  # A=K.
    _adaptive_admission_case(ctx, 64, 7, "adaptive_short_tail", 32)  # A=2K-1.
    _adaptive_admission_case(ctx, 64, 8, "capacity_headroom", 64)  # A=2K.
    _adaptive_admission_case(ctx, 64, 6, "adaptive_short_tail", 32, watermark=1)
    _adaptive_admission_case(ctx, 64, 9, "capacity_headroom", 64, watermark=1)


def test_adaptive_prefill_decode_budget_slot_and_baseline_guards() raises:
    var ctx = DeviceContext()
    _adaptive_admission_case(ctx, 64, 4, "decode_count", 64, decodes=0)
    _adaptive_admission_case(ctx, 64, 6, "decode_count", 64, decodes=2)
    _adaptive_admission_case(ctx, 64, 5, "incumbent_terminal", 64, incumbent_maximum=2)
    _adaptive_admission_case(ctx, 64, 5, "adaptive_short_tail", 32, incumbent_maximum=3)
    _adaptive_admission_case(ctx, 64, 5, "last_sequence_slot", 64, max_sequences=2)
    _adaptive_admission_case(ctx, 64, 5, "baseline_at_most_32", 16, budget=17)
    # Positive output budgets one and two both produce exactly their declared histories.
    _adaptive_admission_case(ctx, 64, 4, "adaptive_short_tail", 32, maximum=1)
    _adaptive_admission_case(ctx, 64, 5, "adaptive_short_tail", 32, maximum=2)
    # Observed controls retain the uncapped baseline, even when fixed32 chooses 32.
    _adaptive_admission_case(ctx, 64, 5, "adaptive_disabled", 64, adaptive=False)
    _adaptive_admission_case(ctx, 64, 5, "adaptive_disabled", 32, adaptive=False, fixed_cap=32)


def test_adaptive_prefill_held_tail_publication_abort_and_stop() raises:
    var ctx = DeviceContext()
    for disposition in [0, 1, 2]:
        var runner = SimulatedRunner([10], 100)
        var pool = KVPool(ctx, 5, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(5, 32, 256, 100, max_sequences=8,
            reserve_lifetime=True, prefix_caching=True,
            prefix_namespace=runner.prefix_cache_identity(), adaptive_prefill=True,
            observe_prefill_policy=True)
        _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.add(0, _prefix_prompt(33), 16, List[Int]())
        _ = engine.step(runner, pool)
        var newcomer = engine.add(1, _prefix_prompt(95), 2,
            [10] if disposition == 2 else List[Int]())
        var first = engine.step(runner, pool)
        assert_equal(first.prefill_tokens, 32)
        assert_true(first.prefill_decisions[0].applied)
        assert_equal(engine.requests[newcomer].generated, 0)
        var sequence = engine.requests[newcomer].sequence
        var published = engine.blocks.tables[sequence][1]
        assert_equal(pool.written[published], 32)
        assert_equal(engine.blocks.states[published], REGISTERED)
        assert_equal(first.selections[1].selected_head, -1)
        if disposition == 1:
            engine.abort(1)
        var second = engine.step(runner, pool)
        assert_equal(len(second.prefill_decisions), 0)  # Held path uses current scheduling.
        assert_equal(second.prefill_tokens, 0 if disposition == 1 else 31)
        assert_equal(engine.requests[newcomer].generated, 0 if disposition == 1 else 1)
        if disposition != 1:
            assert_equal(second.selections[1].past, 64)
            assert_equal(second.selections[1].query_tokens, 31)
            assert_equal(second.selections[1].selected_head, 31)
        _ = _prefix_drain(engine, runner, pool)
        assert_equal(engine.requests[newcomer].generated, 0 if disposition == 1 else (1 if disposition == 2 else 2))
        assert_equal(engine.requests[newcomer].reason, "abort" if disposition == 1 else ("stop" if disposition == 2 else "length"))
        _ = engine.clear_cache(pool)
        for written in pool.written:
            assert_equal(written, 0)
        engine.check(pool)


def test_adaptive_prefill_observation_is_optional_and_zero_output_is_noop() raises:
    var ctx = DeviceContext()
    for observe in [False, True]:
        var runner = SimulatedRunner([10], 100)
        var pool = KVPool(ctx, 5, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(5, 32, 256, 100, reserve_lifetime=True,
            prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
            adaptive_prefill=True, observe_prefill_policy=observe)
        _ = engine.add(0, _prefix_prompt(96), 0, List[Int]())
        var noop = engine.step(runner, pool)
        assert_equal(noop.total_tokens, 0)
        assert_equal(noop.finished, 1)
        assert_equal(len(noop.prefill_decisions), 0)
        assert_equal(len(noop.selections), 0)
        _ = engine.add(1, _prefix_prompt(96), 1, List[Int]())
        var idle = engine.step(runner, pool)
        assert_equal(idle.prefill_tokens, 96)
        assert_equal(len(idle.prefill_decisions), 1 if observe else 0)
        assert_equal(len(idle.selections), 1 if observe else 0)
        if observe:
            assert_equal(idle.prefill_decisions[0].reason, "decode_count")
            assert_equal(idle.selections[0].selected_head, 95)
        _ = engine.clear_cache(pool)


def test_adaptive_prefill_protects_hits_and_counts_only_reclaimable_idle_blocks() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 6, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(6, 32, 256, 100, max_sequences=8,
        reserve_lifetime=True, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity(), adaptive_prefill=True,
        observe_prefill_policy=True)
    # An unrelated idle full block can be evicted; the seed match cannot.
    _ = engine.add(100, _prefix_prompt(33, 1), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.add(101, _prefix_prompt(33), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.add(0, _prefix_prompt(33), 16, List[Int]())
    _ = engine.step(runner, pool)
    _ = engine.add(1, _prefix_prompt(96), 2, List[Int]())
    var mixed = engine.step(runner, pool)
    var decision = mixed.prefill_decisions[0].copy()
    assert_equal(decision.matched_tokens, 32)
    assert_equal(decision.free_blocks, 3)
    assert_equal(decision.reclaimable_blocks, 1)
    assert_equal(decision.available_blocks, 4)
    assert_equal(decision.required_blocks, 3)
    assert_true(decision.applied)
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.clear_cache(pool)
    engine.check(pool)


def test_adaptive_prefill_rejects_incompatible_configuration_before_mutation() raises:
    for variant in range(4):
        with assert_raises():
            _ = EngineCore(5, 32, 256, 100, mixed_prefill=variant != 0,
                reserve_lifetime=variant != 1, prefix_caching=variant != 2,
                prefix_namespace="test", active_prefill_cap=32 if variant == 3 else 0,
                adaptive_prefill=True)
    var ctx = DeviceContext()
    for variant in range(5):
        var runner = SimulatedRunner([10], 100)
        var pool = KVPool(ctx, 5, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(5, 32, 256, 100, reserve_lifetime=True,
            prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(), adaptive_prefill=True)
        var slot = engine.add(0, [1], 2, List[Int]())
        engine.abort(0)
        if variant == 0:
            engine.prefix_caching = False
        elif variant == 1:
            engine.mixed_prefill = False
        elif variant == 2:
            engine.reserve_lifetime = False
        elif variant == 3:
            engine.active_prefill_cap = 32
        else:
            engine.cost_policy_enabled = True
        with assert_raises(contains="adaptive prefill"):
            _ = engine.step(runner, pool)
        assert_equal(engine.step_id, 0)
        assert_equal(engine.requests[slot].state, WAITING)
        assert_equal(engine.requests[slot].sequence, -1)
        assert_equal(runner.steps, 0)
        assert_equal(engine.blocks.free_blocks(), 5)
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 5, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(5, 32, 256, 100, reserve_lifetime=True,
        prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(), adaptive_prefill=True)
    with assert_raises(contains="fixed row"):
        engine.set_cost_policy(StepCost(0, 0, 0, 0, 0))
    assert_equal(engine.cost_policy_enabled, False)
    var async_runner = ActiveCapAsyncProbe()
    var slot = engine.add(0, [1], 2, List[Int]())
    engine.abort(0)
    with assert_raises(contains="synchronous"):
        _ = engine.step_async(async_runner, pool)
    assert_equal(engine.step_id, 0)
    assert_equal(engine.requests[slot].state, WAITING)
    assert_equal(async_runner.calls, 0)
    assert_equal(engine.async_enabled, False)
    assert_equal(len(engine.async_pending), 0)
    _ = engine.step(runner, pool)
    engine.check(pool)


def test_adaptive_prefill_fault_flushes_partial_admission_and_retained_cache() raises:
    var ctx = DeviceContext()
    var runner = PrefixFaultRunner()
    var pool = KVPool(ctx, 5, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(5, 32, 256, 100, reserve_lifetime=True,
        prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
        adaptive_prefill=True, observe_prefill_policy=True)
    _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
    _ = engine.step(runner, pool)
    _ = engine.add(0, _prefix_prompt(33), 16, List[Int]())
    _ = engine.step(runner, pool)
    _ = engine.add(1, _prefix_prompt(96), 2, List[Int]())
    var partial = engine.step(runner, pool)
    assert_true(partial.prefill_decisions[0].applied)
    runner.fail = True
    with assert_raises(contains="injected"):
        _ = engine.step(runner, pool)
    assert_true(engine.failed)
    assert_equal(engine.live(), 0)
    assert_equal(engine.cached_blocks(), 0)
    assert_equal(engine.cache_references(), 0)
    assert_equal(engine.blocks.free_blocks(), 5)
    for written in pool.written:
        assert_equal(written, 0)
    _ = engine.clear_cache(pool)
    engine.check(pool)


def test_adaptive_prefill_cold_short_tail_and_defensive_zero_private_guard() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 4, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(4, 32, 256, 100, max_sequences=8,
        reserve_lifetime=True, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity(), adaptive_prefill=True,
        observe_prefill_policy=True)
    var incumbent = engine.add(0, [1], 16, List[Int]())
    _ = engine.step(runner, pool)
    var newcomer = engine.add(1, List[Int](length=64, fill=2), 16, List[Int]())
    # A legal positive-output cache match leaves a query row, so K=0 cannot
    # arise naturally. Check the defensive branch without attaching a table.
    var guard = EngineStep(0, 1)
    var free_before = engine.blocks.free_blocks()
    assert_equal(engine._admission_prefill_count(newcomer, 255, [incumbent],
        64, free_before, 0, 0, free_before, 0, 0, guard), 0)
    assert_equal(guard.prefill_decisions[0].reason, "no_private_blocks")
    assert_equal(guard.prefill_decisions[0].eligible, False)
    assert_equal(engine.blocks.free_blocks(), free_before)
    assert_equal(engine.requests[newcomer].state, WAITING)
    assert_equal(engine.requests[newcomer].sequence, -1)
    var mixed = engine.step(runner, pool)
    assert_equal(mixed.cache_hit_tokens, 0)
    assert_equal(mixed.prefill_decisions[0].matched_tokens, 0)
    assert_equal(mixed.prefill_decisions[0].remaining_prompt_tokens, 64)
    assert_equal(mixed.prefill_decisions[0].required_blocks, 3)
    assert_true(mixed.prefill_decisions[0].applied)
    assert_equal(mixed.prefill_tokens, 32)
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.clear_cache(pool)
    engine.check(pool)


def test_adaptive_prefill_three_arms_preserve_histories_work_and_idle_256() raises:
    var ctx = DeviceContext()
    var reference = List[List[Int]]()
    for arm in range(3):
        var runner = SimulatedRunner([10, 11, 12], 100)
        var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(7, 32, 256, 100, max_sequences=8,
            reserve_lifetime=True, prefix_caching=True,
            prefix_namespace=runner.prefix_cache_identity(), active_prefill_cap=32 if arm == 1 else 0,
            adaptive_prefill=arm == 2, observe_prefill_policy=True)
        _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
        _ = _prefix_drain(engine, runner, pool)
        for i in range(8):
            var prompt = _prefix_prompt(96)
            for p in range(32, 96):
                prompt[p] = (p + (i + 1) * 13) % 100
            _ = engine.add(i, prompt, 16, List[Int]())
        var rows = 0
        var positions = 0
        var hits = 0
        var steps = 0
        var applications = 0
        while engine.live() > 0 and steps < 100:
            var record = engine.step(runner, pool)
            rows += record.total_tokens
            positions += record.attended_positions
            hits += record.cache_hit_tokens
            for decision in record.prefill_decisions:
                applications += 1 if decision.applied else 0
            steps += 1
            engine.check(pool)
        assert_equal(engine.live(), 0)
        assert_equal(steps, 65 if arm == 0 else 69)
        assert_equal(rows, 632)
        assert_equal(positions, 45504)
        assert_equal(hits, 256)
        assert_equal(applications, 7 if arm == 2 else 0)
        for i in range(8):
            assert_equal(engine.requests[i].generated, 16)
            if arm == 0:
                reference.append(engine.requests[i].tokens.copy())
            else:
                _assert_list(engine.requests[i].tokens, reference[i])
        _ = engine.clear_cache(pool)
        engine.check(pool)
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 16, 32, KVGeometry(1, 1, 1))
    var idle_engine = EngineCore(16, 32, 512, 100, max_sequences=8,
        reserve_lifetime=True, prefix_caching=True,
        prefix_namespace=runner.prefix_cache_identity(), adaptive_prefill=True,
        observe_prefill_policy=True)
    _ = idle_engine.add(0, _prefix_prompt(300), 1, List[Int]())
    var idle = idle_engine.step(runner, pool)
    assert_equal(idle.prefill_tokens, 256)
    assert_equal(idle.prefill_decisions[0].baseline_tokens, 256)
    assert_equal(idle.prefill_decisions[0].reason, "decode_count")
    assert_equal(idle.selections[0].selected_head, -1)
    var tail = idle_engine.step(runner, pool)
    assert_equal(tail.prefill_tokens, 44)
    assert_equal(len(tail.prefill_decisions), 0)
    assert_equal(tail.selections[0].past, 256)
    assert_equal(tail.selections[0].selected_head, 43)
    assert_equal(idle_engine.live(), 0)
    _ = idle_engine.clear_cache(pool)
    idle_engine.check(pool)


def _frequency_prompt(request_id: Int, matching: Bool = True, remaining: Int = 64) -> List[Int]:
    var prompt = _prefix_prompt(32 + remaining)
    for p in range(32, len(prompt)):
        prompt[p] = (p + (request_id + 1) * 13) % 100
    if not matching:
        prompt[0] = (prompt[0] + request_id + 1) % 100
    return prompt^


def test_frequency_quantile_is_exact_at_finite_population_boundaries() raises:
    assert_equal(_frequency_threshold(0), 0)
    for gaps in range(1, 401):
        assert_equal(_frequency_threshold(gaps), gaps - 19 * (gaps - 1) // 20)
    assert_equal(_frequency_threshold(120), 7)
    assert_equal(_frequency_threshold(105), 7)
    assert_equal(_frequency_threshold(135), 8)
    assert_equal(_frequency_threshold(16), 2)
    assert_equal(_frequency_threshold(56), 4)
    assert_equal(_frequency_threshold(248), 14)
    var previous = Int.MAX - 1
    assert_equal(_frequency_threshold(Int.MAX), Int.MAX - (previous // 20 * 19 + previous % 20 * 19 // 20))


def _frequency_cohort_case(mut ctx: DeviceContext, matching: Int, queued: Int,
                           maximum: Int, expected_dense: Bool, expected_threshold: Int,
                           blocks: Int = 7) raises:
    var runner = SimulatedRunner([10, 11, 12], 100)
    var pool = KVPool(ctx, blocks, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(blocks, 32, 256, 100, max_sequences=8,
        reserve_lifetime=True, prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
        observe_prefill_policy=True, observe_frequency_policy=True)
    _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    engine.frequency_prefill = True
    for i in range(queued):
        _ = engine.add(i, _frequency_prompt(i, i < matching), maximum, List[Int]())
    var first = engine.step(runner, pool)
    assert_equal(len(first.frequency_snapshots), 1)
    var snapshot = first.frequency_snapshots[0].copy()
    assert_equal(snapshot.cohort_id, 0)
    assert_equal(snapshot.anchor_request_id, 0)
    assert_equal(snapshot.anchor_ticket, 1)
    assert_equal(snapshot.last_ticket, queued)
    assert_equal(snapshot.request_count, queued)
    assert_equal(snapshot.gap_budget, queued * (maximum - 1))
    assert_equal(snapshot.run_count, matching)
    assert_equal(snapshot.predicted_interruptions, matching - 1)
    assert_equal(snapshot.threshold, expected_threshold)
    assert_true(snapshot.valid)
    assert_equal(snapshot.dense, expected_dense)
    assert_equal(snapshot.enabled, expected_dense)
    assert_equal(snapshot.lookup_count, min(matching, queued - 1))
    assert_equal(snapshot.matched_tokens, 32)
    assert_equal(snapshot.remaining_prompt_tokens, 64)
    assert_equal(snapshot.anchor_maximum_outputs, maximum)
    assert_equal(snapshot.available_blocks, blocks - 1)
    assert_equal(snapshot.required_blocks, (64 + maximum - 1 + 31) // 32)
    assert_equal(first.prefill_tokens, 64)
    assert_equal(first.prefill_decisions[0].reason, "decode_count")
    assert_equal(first.frequency_decisions[0].allowed, expected_dense)
    assert_equal(first.frequency_events[0].kind, "frozen")
    for i in range(queued):
        assert_equal(snapshot.request_ids[i], i)
        assert_equal(snapshot.tickets[i], i + 1)
        assert_equal(snapshot.maximum_outputs[i], maximum)
        assert_equal(snapshot.prompt_lengths[i], 96)
    var steps = 1
    var applications = 0
    var closed = False
    while engine.live() > 0 and steps < 400:
        var record = engine.step(runner, pool)
        assert_equal(len(record.frequency_snapshots), 0)
        for decision in record.frequency_decisions:
            assert_equal(decision.gap_budget, snapshot.gap_budget)
            assert_equal(decision.predicted_interruptions, snapshot.predicted_interruptions)
            assert_equal(decision.threshold, snapshot.threshold)
            assert_equal(decision.allowed, expected_dense and decision.in_run)
        for decision in record.prefill_decisions:
            applications += 1 if decision.applied else 0
        for event in record.frequency_events:
            assert_true(event.kind != "invalidated")
            closed = closed or event.kind == "closed"
        steps += 1
        engine.check(pool)
    assert_equal(engine.live(), 0)
    assert_true(closed)
    assert_equal(engine.frequency_cohort.cohort_id, -1)
    assert_equal(applications > 0, expected_dense)
    if matching == 8 and queued == 8 and maximum == 16:
        assert_equal(steps, 69)
        assert_equal(applications, 7)
    _ = engine.clear_cache(pool)
    engine.check(pool)


def test_frequency_snapshot_density_and_frozen_denominator() raises:
    var ctx = DeviceContext()
    _frequency_cohort_case(ctx, 8, 8, 16, True, 7)
    _frequency_cohort_case(ctx, 4, 8, 16, False, 7)
    _frequency_cohort_case(ctx, 7, 7, 16, False, 7)
    _frequency_cohort_case(ctx, 9, 9, 16, True, 8)
    _frequency_cohort_case(ctx, 4, 8, 3, True, 2)
    _frequency_cohort_case(ctx, 8, 8, 8, True, 4)
    _frequency_cohort_case(ctx, 8, 8, 32, False, 14)


def test_frequency_forecast_peeks_preserve_cache_lru_refs_and_kv() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(7, 32, 256, 100, max_sequences=8, reserve_lifetime=True,
        prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(), frequency_prefill=True,
        observe_frequency_policy=True)
    _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    for i in range(8):
        _ = engine.add(i, _frequency_prompt(i), 16, List[Int]())
    var cached = engine.prefix_cache.match(engine.blocks, engine.requests[0].tokens,
        engine.prefix_namespace, "")
    var clock = engine.prefix_cache.clock
    var refs = engine.blocks.refs.copy()
    var states = engine.blocks.states.copy()
    var written = pool.written.copy()
    var free = engine.blocks.free_blocks()
    var entries = engine.cached_blocks()
    var record = EngineStep(0, 32)
    engine._freeze_frequency(0, cached, 6, 3, record)
    assert_equal(engine.prefix_cache.clock, clock)
    assert_equal(engine.cached_blocks(), entries)
    assert_equal(len(engine.prefix_cache.events), 0)
    assert_equal(engine.blocks.free_blocks(), free)
    _assert_list(engine.blocks.refs, refs)
    _assert_list(engine.blocks.states, states)
    _assert_list(pool.written, written)
    assert_equal(record.frequency_snapshots[0].lookup_count, 7)
    assert_equal(record.frequency_snapshots[0].run_count, 8)
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.clear_cache(pool)


def test_frequency_fifo_barrier_salt_duplicate_and_unsupported_geometry() raises:
    var ctx = DeviceContext()
    for scenario in range(7):
        var runner = SimulatedRunner([10], 100)
        var blocks = 10 if scenario == 3 else 7
        var pool = KVPool(ctx, blocks, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(blocks, 32, 256, 100, token_budget=64 if scenario == 4 else 256,
            max_sequences=2 if scenario == 5 else 8, watermark_blocks=1 if scenario == 6 else 0,
            reserve_lifetime=True, prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
            observe_prefill_policy=True, observe_frequency_policy=True)
        _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
        _ = _prefix_drain(engine, runner, pool)
        engine.frequency_prefill = True
        for i in range(8):
            var prompt = _frequency_prompt(0 if scenario == 2 and i == 1 else i, scenario != 0 or i != 1)
            _ = engine.add(i, prompt, 16, List[Int](), cache_salt="isolated" if scenario == 1 and i == 1 else "")
        var first = engine.step(runner, pool)
        var snapshot = first.frequency_snapshots[0].copy()
        assert_equal(snapshot.enabled, False)
        if scenario < 2:
            assert_equal(snapshot.run_count, 1)
            assert_equal(snapshot.reason, "sparse")
            assert_true(snapshot.valid)
        elif scenario == 2:
            assert_equal(snapshot.reason, "duplicate_private_key")
        elif scenario == 3:
            assert_equal(snapshot.reason, "geometry")
        elif scenario == 4:
            assert_equal(snapshot.reason, "row_budget")
        elif scenario == 5:
            assert_equal(snapshot.reason, "sequence_slots")
        else:
            assert_equal(snapshot.reason, "watermark")
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.clear_cache(pool)


def test_frequency_late_arrival_abort_eos_and_ticket_reuse_disable_until_idle() raises:
    var ctx = DeviceContext()
    for scenario in range(3):
        var runner = SimulatedRunner([10], 100)
        var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(7, 32, 256, 100, max_sequences=8, reserve_lifetime=True,
            prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
            observe_prefill_policy=True, observe_frequency_policy=True)
        _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
        _ = _prefix_drain(engine, runner, pool)
        engine.frequency_prefill = True
        for i in range(8):
            _ = engine.add(i, _frequency_prompt(i), 16, [10] if scenario == 2 and i == 0 else List[Int]())
        var first = engine.step(runner, pool)
        assert_true(first.frequency_snapshots[0].enabled)
        if scenario == 0:
            var late = engine.add(90, _frequency_prompt(90), 3, List[Int]())
            assert_equal(engine.requests[late].ticket, 9)
        elif scenario == 1:
            engine.abort(7)
        var invalidated = scenario == 2
        if scenario == 2:
            assert_equal(first.frequency_events[1].kind, "invalidated")
            assert_equal(first.frequency_events[1].reason, "early_termination")
        var steps = 0
        while engine.live() > 0 and steps < 400:
            var record = engine.step(runner, pool)
            for event in record.frequency_events:
                if event.kind == "invalidated":
                    invalidated = True
                    assert_equal(event.reason, "new_arrival" if scenario == 0 else "abort")
            for decision in record.frequency_decisions:
                assert_equal(decision.allowed, False)
                assert_equal(decision.gap_budget, 120)
            for decision in record.prefill_decisions:
                assert_equal(decision.applied, False)
            steps += 1
        assert_true(invalidated)
        assert_equal(engine.live(), 0)
        assert_equal(engine.frequency_cohort.cohort_id, -1)
        var reused = engine.add(99, _frequency_prompt(99), 3, List[Int]())
        assert_true(engine.requests[reused].ticket >= 9)
        var renewed = engine.step(runner, pool)
        assert_equal(renewed.frequency_snapshots[0].cohort_id, 1)
        assert_equal(renewed.frequency_snapshots[0].request_count, 1)
        assert_equal(renewed.frequency_snapshots[0].gap_budget, 2)
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.clear_cache(pool)


def test_frequency_controls_have_sentinels_and_execute_no_forecast() raises:
    var ctx = DeviceContext()
    for adaptive in [False, True]:
        var runner = SimulatedRunner([10], 100)
        var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(7, 32, 256, 100, reserve_lifetime=True,
            prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(), adaptive_prefill=adaptive,
            observe_prefill_policy=True, observe_frequency_policy=True)
        _ = engine.add(0, _frequency_prompt(0), 3, List[Int]())
        var record = engine.step(runner, pool)
        assert_equal(record.frequency_forecast_ns, 0)
        assert_equal(len(record.frequency_snapshots), 0)
        assert_equal(len(record.frequency_events), 0)
        assert_equal(engine.next_frequency_cohort, 0)
        assert_equal(len(record.frequency_decisions), 1)
        var decision = record.frequency_decisions[0].copy()
        assert_equal(decision.cohort_id, -1)
        assert_equal(decision.gap_budget, -1)
        assert_equal(decision.allowed, False)
        assert_equal(decision.reason, "not_executed")
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.clear_cache(pool)


def test_frequency_configuration_rejects_before_mutation_and_fault_drains() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    for scenario in range(4):
        with assert_raises(contains="configuration"):
            _ = EngineCore(7, 32, 256, 100, prefix_caching=scenario != 0,
                prefix_namespace=runner.prefix_cache_identity(), reserve_lifetime=scenario != 1,
                mixed_prefill=scenario != 2, adaptive_prefill=scenario == 3, frequency_prefill=True)
    var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(7, 32, 256, 100, prefix_caching=True, reserve_lifetime=True,
        prefix_namespace=runner.prefix_cache_identity(), frequency_prefill=True, observe_frequency_policy=True)
    _ = engine.add(0, _frequency_prompt(0), 16, List[Int]())
    engine.abort(0)
    with assert_raises(contains="frequency prefill"):
        engine.set_cost_policy(StepCost(0, 0, 0, 0, 0))
    var asynchronous = ActiveCapAsyncProbe()
    with assert_raises(contains="synchronous"):
        _ = engine.step_async(asynchronous, pool)
    assert_equal(asynchronous.calls, 0)
    assert_equal(engine.step_id, 0)
    assert_equal(engine.async_enabled, False)
    engine.active_prefill_cap = 32
    with assert_raises(contains="frequency prefill"):
        _ = engine.step(runner, pool)
    assert_equal(engine.step_id, 0)
    assert_equal(engine.requests[0].state, WAITING)
    engine.active_prefill_cap = 0
    _ = engine.step(runner, pool)
    assert_equal(engine.live(), 0)
    var faulty = PrefixFaultRunner()
    engine.frequency_prefill = False
    _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
    _ = engine.step(faulty, pool)
    engine.frequency_prefill = True
    for i in range(8):
        _ = engine.add(i, _frequency_prompt(i), 16, List[Int]())
    _ = engine.step(faulty, pool)
    faulty.fail = True
    with assert_raises(contains="injected prefix-cache"):
        _ = engine.step(faulty, pool)
    assert_true(engine.failed)
    assert_equal(engine.live(), 0)
    assert_equal(engine.frequency_cohort.cohort_id, -1)
    assert_equal(len(engine.frequency_pending_events), 0)
    assert_equal(len(engine.failure_frequency_events), 2)
    assert_equal(engine.failure_frequency_events[0].kind, "invalidated")
    assert_equal(engine.failure_frequency_events[1].kind, "closed")
    assert_equal(engine.cached_blocks(), 0)
    assert_equal(engine.cache_references(), 0)
    assert_equal(engine.blocks.free_blocks(), 7)
    engine.check(pool)


def test_frequency_first_snapshot_excludes_seed_zero_output_and_processed_abort() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(7, 32, 256, 100, max_sequences=8, reserve_lifetime=True,
        prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
        observe_frequency_policy=True)
    _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    engine.frequency_prefill = True
    _ = engine.add(90, _prefix_prompt(33), 0, List[Int]())
    _ = engine.add(91, _prefix_prompt(33), 5, List[Int]())
    engine.abort(91)
    for i in range(8):
        _ = engine.add(i, _frequency_prompt(i), 16, List[Int]())
    var first = engine.step(runner, pool)
    assert_equal(first.finished, 2)
    assert_equal(first.aborted, 1)
    var snapshot = first.frequency_snapshots[0].copy()
    assert_equal(snapshot.anchor_ticket, 3)
    assert_equal(snapshot.request_count, 8)
    assert_equal(snapshot.gap_budget, 120)
    for i in range(8):
        assert_equal(snapshot.request_ids[i], i)
        assert_equal(snapshot.tickets[i], i + 3)
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.clear_cache(pool)


def test_frequency_held_tail_finishes_after_new_arrival_and_geometry_change_invalidates() raises:
    var ctx = DeviceContext()
    for geometry_change in [False, True]:
        var runner = SimulatedRunner([10], 100)
        var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
        var engine = EngineCore(7, 32, 256, 100, max_sequences=8, reserve_lifetime=True,
            prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
            observe_prefill_policy=True, observe_frequency_policy=True)
        _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
        _ = _prefix_drain(engine, runner, pool)
        engine.frequency_prefill = True
        for i in range(8):
            _ = engine.add(i, _frequency_prompt(i), 16, List[Int]())
        _ = engine.step(runner, pool)
        if geometry_change:
            engine.requests[1].cache_salt = "changed"
            var changed = engine.step(runner, pool)
            assert_equal(changed.frequency_events[0].kind, "invalidated")
            assert_equal(changed.frequency_events[0].reason, "geometry_changed")
            assert_equal(changed.frequency_decisions[0].allowed, False)
        else:
            var mixed = engine.step(runner, pool)
            assert_equal(mixed.prefill_tokens, 32)
            assert_true(mixed.prefill_decisions[0].applied)
            assert_equal(engine.requests[1].generated, 0)
            _ = engine.add(90, _frequency_prompt(90), 3, List[Int]())
            var held = engine.step(runner, pool)
            assert_equal(held.prefill_tokens, 32)
            assert_equal(len(held.frequency_decisions), 0)
            assert_equal(len(held.prefill_decisions), 0)
            assert_equal(held.frequency_forecast_ns, 0)
            assert_equal(engine.requests[1].generated, 1)
            assert_equal(held.frequency_events[0].reason, "new_arrival")
        _ = _prefix_drain(engine, runner, pool)
        _ = engine.clear_cache(pool)
        engine.check(pool)


def test_frequency_checked_denominator_overflow_selects_baseline() raises:
    var ctx = DeviceContext()
    var runner = SimulatedRunner([10], 100)
    var pool = KVPool(ctx, 7, 32, KVGeometry(1, 1, 1))
    var engine = EngineCore(7, 32, 256, 100, max_sequences=8, reserve_lifetime=True,
        prefix_caching=True, prefix_namespace=runner.prefix_cache_identity(),
        observe_frequency_policy=True)
    _ = engine.add(100, _prefix_prompt(33), 1, List[Int]())
    _ = _prefix_drain(engine, runner, pool)
    engine.frequency_prefill = True
    for i in range(2):
        _ = engine.add(i, _frequency_prompt(i), 16, List[Int]())
    # Defensive helper test: public add/context bounds prevent this sum today.
    engine.requests[0].maximum = Int.MAX
    engine.requests[1].maximum = Int.MAX
    var cached = engine.prefix_cache.match(engine.blocks, engine.requests[0].tokens,
        engine.prefix_namespace, "")
    var record = EngineStep(0, 16)
    engine._freeze_frequency(0, cached, 6, 3, record)
    assert_equal(record.frequency_snapshots[0].reason, "gap_overflow")
    assert_equal(record.frequency_snapshots[0].valid, False)
    assert_equal(record.frequency_snapshots[0].enabled, False)
    assert_true(record.frequency_snapshots[0].gap_budget >= 0)
    engine.requests[0].maximum = 16
    engine.requests[1].maximum = 16
    _ = _prefix_drain(engine, runner, pool)
    _ = engine.clear_cache(pool)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
