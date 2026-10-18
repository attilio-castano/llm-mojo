"""Independent execution, flat-result and proof-statement checks.

The flat oracle scans immutable inputs directly. It does not call the merge,
plan interpreter, execution scheduler or certificate producer to decide the
expected winner. Source visit multiplicity is checked separately because the
selection operation is idempotent.
"""
from std.collections import Array
from std.testing import assert_equal
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord
import llm_mojo.kernels.finishing_execution as execution
import llm_mojo.kernels.finishing_execution_replay as replay
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, ReductionNode, INPUT_NODE, MERGE_NODE, input_node, empty_node, merge_node,
    sequential_argmax_plan, balanced_argmax_plan,
)


def execution_record_equal(a: ArgmaxRecord, b: ArgmaxRecord) -> Bool:
    return a.rank == b.rank and a.token == b.token and a.invalid == b.invalid


def execution_flat_oracle[N: Int](
    inputs: Array[ArgmaxRecord, N], count: Int,
) -> ArgmaxRecord:
    """Find max rank, then the least eligible token; OR every invalid word."""
    var top: UInt32 = 0
    var token: UInt32 = 0xFFFFFFFF
    var invalid: UInt32 = 0
    for index in range(count):
        if inputs[index].rank > top:
            top = inputs[index].rank
        invalid |= inputs[index].invalid
    for index in range(count):
        if inputs[index].rank == top and inputs[index].token < token:
            token = inputs[index].token
    return ArgmaxRecord(top, token, invalid)


def execution_input_fixture[N: Int](variant: Int) -> Array[ArgmaxRecord, N]:
    """Asymmetric records, ties, zero ranks, extreme words and losing bad flags."""
    var inputs = Array[ArgmaxRecord, N](fill=ArgmaxRecord(0, 0xFFFFFFFF, 0))
    for index in range(N):
        var rank = UInt32((index * 29 + 11) % 37)
        var token = UInt32(N * 3 + 1 - index)
        var invalid = UInt32(index % 7 == 2)
        if variant == 1:
            rank = UInt32(13)
            token = UInt32((index * 17 + 5) % (N + 1))
        elif variant == 2:
            rank = UInt32(0)
            token = UInt32(0xFFFFFFFF) if index % 2 == 0 else UInt32(N - index)
        elif variant == 3:
            rank = UInt32(0xFFFFFFFF) if index % 3 == 0 else UInt32(index)
            token = UInt32(0xFFFFFFFF - index)
        elif variant == 4:
            invalid = UInt32(1) << UInt32(index % 31)
        inputs[index] = ArgmaxRecord(rank, token, invalid)
    return inputs^


def _append_records[N: Int](mut text: String, records: Array[ArgmaxRecord, N]):
    for record in records:
        text += ":" + String(record.rank) + "," + String(record.token) + "," + String(record.invalid)


def execution_state_key[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    state: execution.FinishingExecutionState[T, I, L, W, S, G, C],
) -> String:
    """Complete independent serialization, including visits and inactive cells."""
    var text = String(state.count) + "/" + String(state.steps) + "/" + String(state.output_ready)
    for base in state.base:
        text += ":b" + String(base)
    for stage in state.stage:
        text += ":s" + String(stage)
    for visits in state.visits:
        text += ":v" + String(visits)
    for ready in state.record_ready:
        text += ":r" + String(ready)
    for ready in state.local_ready:
        text += ":l" + String(ready)
    for ready in state.publish_ready:
        text += ":p" + String(ready)
    for ready in state.subgroup_ready:
        text += ":g" + String(ready)
    for ready in state.final_ready:
        text += ":f" + String(ready)
    _append_records(text, state.best)
    _append_records(text, state.records)
    _append_records(text, state.local_values)
    _append_records(text, state.published)
    _append_records(text, state.subgroup_values)
    _append_records(text, state.final_values)
    text += ":o" + String(state.output.rank) + "," + String(state.output.token) + "," + String(state.output.invalid)
    return text^


def _node_oracle[N: Int, P: Int](plan: ArgmaxPlan[N, P], node: Int,
    inputs: Array[ArgmaxRecord, N]) -> ArgmaxRecord:
    """Compute Boolean support, then scan original leaves without a join."""
    var support = Array[Bool, N * P](fill=False)
    for index in range(node + 1):
        var operation = plan.nodes[index]
        for leaf in range(N):
            if operation.operation == INPUT_NODE:
                support[index * N + leaf] = operation.left == leaf
            elif operation.operation == MERGE_NODE:
                support[index * N + leaf] = (support[operation.left * N + leaf]
                    or support[operation.right * N + leaf])
    var top: UInt32 = 0
    var token: UInt32 = 0xFFFFFFFF
    var invalid: UInt32 = 0
    for leaf in range(N):
        if support[node * N + leaf]:
            if inputs[leaf].rank > top:
                top = inputs[leaf].rank
            invalid |= inputs[leaf].invalid
    for leaf in range(N):
        if support[node * N + leaf] and inputs[leaf].rank == top and inputs[leaf].token < token:
            token = inputs[leaf].token
    return ArgmaxRecord(top, token, invalid)


def _thread_prefix[C: Int](inputs: Array[ArgmaxRecord, C], count: Int,
    thread: Int, threads: Int, items: Int, end: Int) -> ArgmaxRecord:
    var top: UInt32 = 0
    var token: UInt32 = 0xFFFFFFFF
    var invalid: UInt32 = 0
    for source in range(count):
        if source < end and (source % (threads * items)) % threads == thread:
            if inputs[source].rank > top:
                top = inputs[source].rank
            invalid |= inputs[source].invalid
    for source in range(count):
        if (source < end and (source % (threads * items)) % threads == thread
            and inputs[source].rank == top and inputs[source].token < token):
            token = inputs[source].token
    return ArgmaxRecord(top, token, invalid)


def _all_ready[N: Int](ready: Array[Bool, N], begin: Int, end: Int) -> Bool:
    for index in range(begin, end):
        if not ready[index]:
            return False
    return True


def _oracle_enabled[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    state: execution.FinishingExecutionState[T, I, L, W, S, G, C],
    event: execution.FinishingExecutionEvent,
) -> Bool:
    if state.output_ready:
        return False
    var owner = event.owner
    var node = event.index
    if event.kind >= execution.FINISHING_FILL and event.kind <= execution.FINISHING_PUBLISH:
        if owner < 0 or owner >= T or event.generation != state.base[owner]:
            return False
        if event.kind == execution.FINISHING_FILL:
            return state.stage[owner] == 0 and node >= 0 and node < I and not state.record_ready[owner + T * node]
        var records_ready = True
        for item in range(I):
            records_ready = records_ready and state.record_ready[owner + T * item]
        if event.kind == execution.FINISHING_LOCAL_NODE:
            if state.stage[owner] != 0 or node < 0 or node >= L or state.local_ready[owner * L + node] or not records_ready:
                return False
            var operation = descriptor.local.nodes[node]
            return operation.operation != MERGE_NODE or (
                state.local_ready[owner * L + operation.left] and state.local_ready[owner * L + operation.right])
        if node != 0:
            return False
        if event.kind == execution.FINISHING_ACCUMULATE:
            return state.stage[owner] == 0 and records_ready and _all_ready(state.local_ready, owner * L, (owner + 1) * L)
        if event.kind == execution.FINISHING_ADVANCE:
            return state.stage[owner] == 1
        return state.stage[owner] == 2 and not state.publish_ready[owner]
    if event.generation != 0:
        return False
    if event.kind == execution.FINISHING_SUBGROUP_NODE:
        if owner < 0 or owner >= T // W or node < 0 or node >= S or state.subgroup_ready[owner * S + node]:
            return False
        if not _all_ready(state.publish_ready, owner * W, (owner + 1) * W):
            return False
        var operation = descriptor.subgroup.nodes[node]
        return operation.operation != MERGE_NODE or (
            state.subgroup_ready[owner * S + operation.left] and state.subgroup_ready[owner * S + operation.right])
    if owner != 0:
        return False
    if event.kind == execution.FINISHING_FINAL_NODE:
        if node < 0 or node >= G or state.final_ready[node] or not _all_ready(state.subgroup_ready, 0, (T // W) * S):
            return False
        var operation = descriptor.final.nodes[node]
        return operation.operation != MERGE_NODE or (state.final_ready[operation.left] and state.final_ready[operation.right])
    return event.kind == execution.FINISHING_OUTPUT and node == 0 and _all_ready(state.final_ready, 0, G)


def _canonical_events[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    state: execution.FinishingExecutionState[T, I, L, W, S, G, C],
) -> List[execution.FinishingExecutionEvent]:
    var events = List[execution.FinishingExecutionEvent]()
    for thread in range(T):
        for item in range(I):
            events.append(execution.FinishingExecutionEvent(execution.FINISHING_FILL, thread, item, state.base[thread]))
        for node in range(L):
            events.append(execution.FinishingExecutionEvent(execution.FINISHING_LOCAL_NODE, thread, node, state.base[thread]))
        for kind in range(execution.FINISHING_ACCUMULATE, execution.FINISHING_PUBLISH + 1):
            events.append(execution.FinishingExecutionEvent(kind, thread, 0, state.base[thread]))
    for group in range(T // W):
        for node in range(S):
            events.append(execution.FinishingExecutionEvent(execution.FINISHING_SUBGROUP_NODE, group, node, 0))
    for node in range(G):
        events.append(execution.FinishingExecutionEvent(execution.FINISHING_FINAL_NODE, 0, node, 0))
    events.append(execution.FinishingExecutionEvent(execution.FINISHING_OUTPUT, 0, 0, 0))
    return events^


def _oracle_step[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    mut state: execution.FinishingExecutionState[T, I, L, W, S, G, C],
    inputs: Array[ArgmaxRecord, C], event: execution.FinishingExecutionEvent,
):
    var owner = event.owner
    var node = event.index
    var empty = ArgmaxRecord(0, 0xFFFFFFFF, 0)
    if event.kind == execution.FINISHING_FILL:
        var source = state.base[owner] + node * T + owner
        var payload = empty
        if source < state.count:
            payload = inputs[source]
            state.visits[source] += 1
        state.records[node * T + owner] = payload
        state.record_ready[node * T + owner] = True
    elif event.kind == execution.FINISHING_LOCAL_NODE:
        var records = Array[ArgmaxRecord, I](fill=empty)
        for item in range(I):
            records[item] = state.records[item * T + owner]
        state.local_values[owner * L + node] = _node_oracle(descriptor.local, node, records)
        state.local_ready[owner * L + node] = True
    elif event.kind == execution.FINISHING_ACCUMULATE:
        state.best[owner] = _thread_prefix(inputs, state.count, owner, T, I, state.base[owner] + T * I)
        state.stage[owner] = 1
    elif event.kind == execution.FINISHING_ADVANCE:
        state.base[owner] += T * I
        for item in range(I):
            state.records[item * T + owner] = empty
            state.record_ready[item * T + owner] = False
        for local_node in range(L):
            state.local_values[owner * L + local_node] = empty
            state.local_ready[owner * L + local_node] = False
        state.stage[owner] = 0 if state.base[owner] < state.count else 2
    elif event.kind == execution.FINISHING_PUBLISH:
        state.published[owner] = _thread_prefix(inputs, state.count, owner, T, I, state.count)
        state.publish_ready[owner] = True
        state.stage[owner] = 3
    elif event.kind == execution.FINISHING_SUBGROUP_NODE:
        var published = Array[ArgmaxRecord, W](fill=empty)
        for member in range(W):
            published[member] = state.published[owner * W + member]
        state.subgroup_values[owner * S + node] = _node_oracle(descriptor.subgroup, node, published)
        state.subgroup_ready[owner * S + node] = True
    elif event.kind == execution.FINISHING_FINAL_NODE:
        var groups = Array[ArgmaxRecord, T // W](fill=empty)
        for group in range(T // W):
            groups[group] = state.subgroup_values[group * S + descriptor.subgroup.root]
        state.final_values[node] = _node_oracle(descriptor.final, node, groups)
        state.final_ready[node] = True
    else:
        state.output = execution_flat_oracle(inputs, state.count)
        state.output_ready = True
    state.steps += 1


def _assert_state[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    state: execution.FinishingExecutionState[T, I, L, W, S, G, C],
    inputs: Array[ArgmaxRecord, C],
) raises:
    var rounds = 0 if state.count == 0 else 1 + (state.count - 1) // (T * I)
    var total = T * (rounds * (I + L + 2) + 1) + (T // W) * S + G + 1
    assert_equal(execution.finishing_execution_expected_steps(descriptor, state.count), total)
    assert_equal(execution.finishing_execution_remaining_steps(descriptor, state), total - state.steps)
    assert_equal(execution.valid_finishing_execution_state(descriptor, state), True)
    for thread in range(T):
        var end = state.base[thread]
        if state.stage[thread] == 1:
            end += T * I
        assert_equal(execution_record_equal(state.best[thread],
            _thread_prefix(inputs, state.count, thread, T, I, end)), True)
    for source in range(C):
        var expected_visits = 0
        if source < state.count:
            var owner = (source % (T * I)) % T
            var base = state.base[owner]
            if source < base:
                expected_visits = 1
            elif source < base + T * I and state.record_ready[source % (T * I)]:
                expected_visits = 1
        assert_equal(state.visits[source], expected_visits)
    if state.output_ready:
        assert_equal(execution_record_equal(state.output, execution_flat_oracle(inputs, state.count)), True)
        assert_equal(execution.finishing_execution_complete(descriptor, state), True)
        assert_equal(state.steps, total)
    else:
        assert_equal(execution.finishing_execution_complete(descriptor, state), False)


struct _ExecutionRandom(Movable):
    var value: UInt64

    def __init__(out self, seed: Int):
        self.value = UInt64(seed) * 2654435761 + 7

    def below(mut self, bound: Int) -> Int:
        self.value = self.value * 6364136223846793005 + 1442695040888963407
        return Int((self.value >> 33) % UInt64(bound))


def _enabled_events[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    state: execution.FinishingExecutionState[T, I, L, W, S, G, C],
    check_enumeration: Bool,
) raises -> List[execution.FinishingExecutionEvent]:
    var canonical = _canonical_events(state)
    var enabled = List[execution.FinishingExecutionEvent]()
    if check_enumeration:
        assert_equal(execution.finishing_execution_event_capacity(descriptor), len(canonical))
    for index in range(len(canonical)):
        var event = canonical[index]
        var expected = _oracle_enabled(descriptor, state, event)
        if check_enumeration:
            var enumerated = execution.finishing_execution_event_at(descriptor, state, index)
            assert_equal(execution.same_finishing_execution_event(enumerated, event), True)
            assert_equal(execution.finishing_execution_event_enabled(descriptor, state, event), expected)
        if expected:
            enabled.append(event)
    if check_enumeration:
        assert_equal(execution.finishing_execution_enabled_count(descriptor, state), len(enabled))
        for index in range(len(enabled)):
            var selected = execution.finishing_execution_enabled_event(descriptor, state, index)
            assert_equal(execution.same_finishing_execution_event(selected, enabled[index]), True)
        assert_equal(execution.finishing_execution_event_at(descriptor, state, len(canonical)).kind, -1)
        assert_equal(execution.finishing_execution_enabled_event(descriptor, state, len(enabled)).kind, -1)
    return enabled^


def _checked_successor[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    before: execution.FinishingExecutionState[T, I, L, W, S, G, C],
    inputs: Array[ArgmaxRecord, C], event: execution.FinishingExecutionEvent,
) raises -> execution.FinishingExecutionState[T, I, L, W, S, G, C]:
    var actual = before.copy()
    var expected = before.copy()
    var original_inputs = inputs.copy()
    assert_equal(_oracle_enabled(descriptor, before, event), True)
    assert_equal(execution.finishing_execution_event_enabled(descriptor, before, event), True)
    _oracle_step(descriptor, expected, inputs, event)
    assert_equal(execution.step_finishing_execution(descriptor, actual, inputs, event), True)
    assert_equal(execution_state_key(actual), execution_state_key(expected))
    assert_equal(execution.finishing_execution_remaining_steps(descriptor, actual),
        execution.finishing_execution_remaining_steps(descriptor, before) - 1)
    for index in range(C):
        assert_equal(execution_record_equal(inputs[index], original_inputs[index]), True)
    _assert_state(descriptor, actual, inputs)
    return actual^


def _reject_event[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    state: execution.FinishingExecutionState[T, I, L, W, S, G, C],
    inputs: Array[ArgmaxRecord, C], event: execution.FinishingExecutionEvent,
) raises:
    var after = state.copy()
    assert_equal(execution.finishing_execution_event_enabled(descriptor, state, event), False)
    assert_equal(execution.step_finishing_execution(descriptor, after, inputs, event), False)
    assert_equal(execution_state_key(after), execution_state_key(state))


def exhaust_finishing_execution[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    count: Int, variant: Int, mut totals: Array[Int, 4],
) raises:
    """Explore every reachable state and every legal outgoing transition.

    State deduplication includes source multiplicities. The bound is an
    assertion, so reaching it is failure rather than a truncated-exhaustion
    claim. This is finite test evidence, separate from universal proof terms.
    """
    var inputs = execution_input_fixture[C](variant)
    var initial = execution.make_finishing_execution_state[C](descriptor, count)
    var queue = List[execution.FinishingExecutionState[T, I, L, W, S, G, C]]()
    var seen = Dict[String, Bool]()
    seen[execution_state_key(initial)] = True
    queue.append(initial^)
    var cursor = 0
    var edges = 0
    var terminals = 0
    while cursor < len(queue):
        var state = queue[cursor].copy()
        cursor += 1
        _assert_state(descriptor, state, inputs)
        var enabled = _enabled_events(descriptor, state, True)
        if state.output_ready:
            assert_equal(len(enabled), 0)
            terminals += 1
        else:
            assert_equal(len(enabled) > 0, True)
        for event in enabled:
            var after = _checked_successor(descriptor, state, inputs, event)
            edges += 1
            var key = execution_state_key(after)
            if key not in seen:
                seen[key] = True
                queue.append(after^)
                assert_equal(len(queue) < 20000, True)
    assert_equal(terminals > 0, True)
    totals[0] += 1
    totals[1] += len(queue)
    totals[2] += edges
    totals[3] += terminals


def run_seeded_finishing_execution[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    count: Int, variant: Int, seed: Int, mut totals: Array[Int, 3],
) raises:
    var inputs = execution_input_fixture[C](variant)
    var state = execution.make_finishing_execution_state[C](descriptor, count)
    var random = _ExecutionRandom(seed)
    _assert_state(descriptor, state, inputs)
    var budget = execution.finishing_execution_expected_steps(descriptor, count)
    for _ in range(budget):
        var enabled = _enabled_events(descriptor, state, False)
        assert_equal(len(enabled) > 0, True)
        var event = enabled[random.below(len(enabled))]
        if event.kind == execution.FINISHING_FILL:
            var source = state.base[event.owner] + event.owner + T * event.index
            if source >= count:
                totals[2] += 1
        state = _checked_successor(descriptor, state, inputs, event)
    assert_equal(execution.finishing_execution_complete(descriptor, state), True)
    assert_equal(len(_enabled_events(descriptor, state, False)), 0)
    totals[0] += 1
    totals[1] += state.steps


def repeated_finishing_plan() -> ArgmaxPlan[2, 7]:
    # Root 5 duplicates input 0; unused final Empty still has to execute.
    var nodes = Array[ReductionNode, 7](uninitialized=True)
    nodes[0] = input_node(0)
    nodes[1] = input_node(1)
    nodes[2] = merge_node(0, 0)
    nodes[3] = merge_node(2, 1)
    nodes[4] = empty_node()
    nodes[5] = merge_node(3, 4)
    nodes[6] = empty_node()
    return ArgmaxPlan[2, 7](nodes^, 5)


comptime FINISHING_EXECUTION_NATIVE_MUTANTS = 17


def finishing_execution_native_mutant_accepts(mode: Int) -> Bool:
    var pair = sequential_argmax_plan[2]()
    var single = sequential_argmax_plan[1]()
    var descriptor = execution.make_finishing_execution_descriptor[2, 2, 5, 2, 5, 3](pair, pair, single)
    if mode == 0:
        descriptor.spec_version += 1
    elif mode == 1:
        descriptor.operation_profile += 1
    elif mode == 2:
        descriptor.program.stride = 3
    elif mode == 3:
        descriptor.program.thread_scale = 0
    elif mode == 4:
        descriptor.program.item_scale = 1
    elif mode == 5:
        descriptor.program.index_bias = 1
    elif mode == 6:
        descriptor.program.start = 4
    elif mode == 7:
        descriptor.program.loop_guard += 1
    elif mode == 8:
        descriptor.program.load_guard += 1
    elif mode == 9:
        descriptor.program.count_limit = -1
    elif mode == 10:
        descriptor.program.integer_limit = 7
    elif mode == 11:
        descriptor.local.root = 0
    elif mode == 12:
        descriptor.subgroup.root = 0
    elif mode == 13:
        descriptor.final.root = -1
    elif mode == 14:
        descriptor.local.nodes[0] = merge_node(1, 1)
    elif mode == 15:
        descriptor.subgroup.nodes[2] = ReductionNode(0, 1, 0)
    elif mode == 16:
        descriptor.program.threads = 3
    else:
        return True
    return execution.check_finishing_execution_descriptor(descriptor)


def check_finishing_execution_native() raises:
    var small = Array[Int, 4](fill=0)
    var local1 = sequential_argmax_plan[1]()
    var local2 = sequential_argmax_plan[2]()
    var pair = balanced_argmax_plan[2]()
    var repeated = repeated_finishing_plan()
    var one = execution.make_finishing_execution_descriptor[1, 1, 3, 1, 3, 3](local1, local1, local1)
    for count in range(3):
        exhaust_finishing_execution[1, 1, 3, 1, 3, 3, 4](one, count, count, small)
    var items = execution.make_finishing_execution_descriptor[1, 2, 5, 1, 3, 3](local2, local1, local1)
    exhaust_finishing_execution[1, 2, 5, 1, 3, 3, 4](items, 3, 1, small)
    var threads = execution.make_finishing_execution_descriptor[2, 1, 3, 1, 3, 5](local1, local1, pair)
    exhaust_finishing_execution[2, 1, 3, 1, 3, 5, 4](threads, 3, 3, small)
    var subgroup = execution.make_finishing_execution_descriptor[2, 1, 3, 2, 5, 3](local1, pair, local1)
    exhaust_finishing_execution[2, 1, 3, 2, 5, 3, 4](subgroup, 3, 0, small)
    var duplicates = execution.make_finishing_execution_descriptor[1, 2, 7, 1, 3, 3](repeated, local1, local1)
    exhaust_finishing_execution[1, 2, 7, 1, 3, 3, 4](duplicates, 2, 4, small)
    print("Finishing execution exhaustive cases/states/edges/terminal states:", small[0], small[1], small[2], small[3])

    var larger = Array[Int, 3](fill=0)
    var local3 = balanced_argmax_plan[3]()
    var warp32 = balanced_argmax_plan[32]()
    var final2 = sequential_argmax_plan[2]()
    var d32 = execution.make_finishing_execution_descriptor[32, 3, 7, 32, 65, 3](local3, warp32, local1)
    var d64 = execution.make_finishing_execution_descriptor[64, 3, 7, 32, 65, 5](local3, warp32, final2)
    var counts32: Array[Int, 6] = [0, 1, 95, 96, 97, 193]
    for index in range(6):
        run_seeded_finishing_execution[32, 3, 7, 32, 65, 3, 256](d32, counts32[index], index % 5, 71 + index, larger)
    var counts64: Array[Int, 4] = [1, 191, 192, 193]
    for index in range(4):
        run_seeded_finishing_execution[64, 3, 7, 32, 65, 5, 256](d64, counts64[index], (index + 1) % 5, 101 + index, larger)
    var odd = execution.make_finishing_execution_descriptor[3, 5, 11, 1, 3, 7](
        balanced_argmax_plan[5](), local1, local3)
    run_seeded_finishing_execution[3, 5, 11, 1, 3, 7, 64](odd, 31, 4, 53, larger)
    print("Finishing execution seeded cases/events/padded fills:", larger[0], larger[1], larger[2])


def check_finishing_execution_controls() raises:
    for mode in range(FINISHING_EXECUTION_NATIVE_MUTANTS):
        assert_equal(finishing_execution_native_mutant_accepts(mode), False)
    var single = sequential_argmax_plan[1]()
    var pair = sequential_argmax_plan[2]()
    var repeated = repeated_finishing_plan()
    var descriptor = execution.make_finishing_execution_descriptor[2, 2, 5, 2, 5, 3](pair, pair, single)
    var inputs = execution_input_fixture[8](0)
    var state = execution.make_finishing_execution_state[8](descriptor, 5)
    # A thread may advance while its peer has not filled its first record.
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_LOCAL_NODE, 0, 0, 0))
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_ACCUMULATE, 0, 0, 0))
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_ADVANCE, 0, 0, 0))
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_PUBLISH, 0, 0, 0))
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_OUTPUT, 0, 0, 0))
    var fill = execution.FinishingExecutionEvent(execution.FINISHING_FILL, 0, 0, 0)
    state = _checked_successor(descriptor, state, inputs, fill)
    _reject_event(descriptor, state, inputs, fill)
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_LOCAL_NODE, 0, 0, 0))
    state = _checked_successor(descriptor, state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_FILL, 0, 1, 0))
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_LOCAL_NODE, 0, 4, 0))
    for node in range(5):
        state = _checked_successor(descriptor, state, inputs,
            execution.FinishingExecutionEvent(execution.FINISHING_LOCAL_NODE, 0, node, 0))
    state = _checked_successor(descriptor, state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_ACCUMULATE, 0, 0, 0))
    _reject_event(descriptor, state, inputs, execution.FinishingExecutionEvent(execution.FINISHING_ACCUMULATE, 0, 0, 0))
    _reject_event(descriptor, state, inputs, fill)
    var carried_best = state.best[0]
    state = _checked_successor(descriptor, state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_ADVANCE, 0, 0, 0))
    assert_equal(state.base[0], 4)
    assert_equal(state.base[1], 0)
    assert_equal(execution_record_equal(state.best[0], carried_best), True)
    _reject_event(descriptor, state, inputs, fill)
    state = _checked_successor(descriptor, state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_FILL, 0, 1, 4))
    assert_equal(execution_record_equal(state.records[2], ArgmaxRecord(0, 0xFFFFFFFF, 0)), True)
    assert_equal(state.record_ready[2], True)
    for index in range(5, 8):
        assert_equal(state.visits[index], 0)

    # A root before the final node cannot bypass execution of unused nodes.
    var unused_local = execution.make_finishing_execution_descriptor[1, 2, 7, 1, 3, 3](repeated, single, single)
    var local_state = execution.make_finishing_execution_state[8](unused_local, 1)
    for item in range(2):
        local_state = _checked_successor(unused_local, local_state, inputs,
            execution.FinishingExecutionEvent(execution.FINISHING_FILL, 0, item, 0))
    for node in range(6):
        local_state = _checked_successor(unused_local, local_state, inputs,
            execution.FinishingExecutionEvent(execution.FINISHING_LOCAL_NODE, 0, node, 0))
    _reject_event(unused_local, local_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_ACCUMULATE, 0, 0, 0))
    local_state = _checked_successor(unused_local, local_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_LOCAL_NODE, 0, 6, 0))
    assert_equal(execution.finishing_execution_event_enabled(unused_local, local_state,
        execution.FinishingExecutionEvent(execution.FINISHING_ACCUMULATE, 0, 0, 0)), True)

    var unused_group = execution.make_finishing_execution_descriptor[2, 1, 3, 2, 7, 3](single, repeated, single)
    var group_state = execution.make_finishing_execution_state[8](unused_group, 0)
    group_state = _checked_successor(unused_group, group_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_PUBLISH, 0, 0, 0))
    _reject_event(unused_group, group_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_SUBGROUP_NODE, 0, 0, 0))
    group_state = _checked_successor(unused_group, group_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_PUBLISH, 1, 0, 0))
    for node in range(6):
        group_state = _checked_successor(unused_group, group_state, inputs,
            execution.FinishingExecutionEvent(execution.FINISHING_SUBGROUP_NODE, 0, node, 0))
    _reject_event(unused_group, group_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_FINAL_NODE, 0, 0, 0))
    group_state = _checked_successor(unused_group, group_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_SUBGROUP_NODE, 0, 6, 0))
    assert_equal(execution.finishing_execution_event_enabled(unused_group, group_state,
        execution.FinishingExecutionEvent(execution.FINISHING_FINAL_NODE, 0, 0, 0)), True)

    var unused_final = execution.make_finishing_execution_descriptor[2, 1, 3, 1, 3, 7](single, single, repeated)
    var final_state = execution.make_finishing_execution_state[8](unused_final, 0)
    for owner in range(2):
        final_state = _checked_successor(unused_final, final_state, inputs,
            execution.FinishingExecutionEvent(execution.FINISHING_PUBLISH, owner, 0, 0))
        for node in range(3):
            final_state = _checked_successor(unused_final, final_state, inputs,
                execution.FinishingExecutionEvent(execution.FINISHING_SUBGROUP_NODE, owner, node, 0))
    for node in range(6):
        final_state = _checked_successor(unused_final, final_state, inputs,
            execution.FinishingExecutionEvent(execution.FINISHING_FINAL_NODE, 0, node, 0))
    _reject_event(unused_final, final_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_OUTPUT, 0, 0, 0))
    final_state = _checked_successor(unused_final, final_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_FINAL_NODE, 0, 6, 0))
    final_state = _checked_successor(unused_final, final_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_OUTPUT, 0, 0, 0))
    _reject_event(unused_final, final_state, inputs,
        execution.FinishingExecutionEvent(execution.FINISHING_OUTPUT, 0, 0, 0))

    # Invalid extent, event kind/owner/index/generation and native metadata
    # reject before writing. Structural state validity is not trace provenance.
    var initial = execution.make_finishing_execution_state[8](descriptor, 5)
    var malformed: Array[execution.FinishingExecutionEvent, 9] = [
        execution.FinishingExecutionEvent(-1, 0, 0, 0),
        execution.FinishingExecutionEvent(8, 0, 0, 0),
        execution.FinishingExecutionEvent(execution.FINISHING_FILL, -1, 0, 0),
        execution.FinishingExecutionEvent(execution.FINISHING_FILL, 2, 0, 0),
        execution.FinishingExecutionEvent(execution.FINISHING_FILL, 0, -1, 0),
        execution.FinishingExecutionEvent(execution.FINISHING_FILL, 0, 2, 0),
        execution.FinishingExecutionEvent(execution.FINISHING_FILL, 0, 0, 4),
        execution.FinishingExecutionEvent(execution.FINISHING_OUTPUT, 1, 0, 0),
        execution.FinishingExecutionEvent(execution.FINISHING_OUTPUT, 0, 0, 1),
    ]
    for event in malformed:
        _reject_event(descriptor, initial, inputs, event)
    var truncated = execution.make_finishing_execution_state[8](descriptor, 9)
    _reject_event(descriptor, truncated, inputs, fill)
    var negative = execution.make_finishing_execution_state[8](descriptor, -1)
    _reject_event(descriptor, negative, inputs, fill)

    var corrupt_visits = final_state.copy()
    corrupt_visits.visits[0] = 1
    assert_equal(execution.finishing_execution_complete(unused_final, corrupt_visits), False)
    var corrupt_output = final_state.copy()
    corrupt_output.output.token = 0
    assert_equal(execution_record_equal(corrupt_output.output, execution_flat_oracle(inputs, 0)), False)
    var same_rank: Array[ArgmaxRecord, 3] = [ArgmaxRecord(41, 19, 0), ArgmaxRecord(41, 7, 0), ArgmaxRecord(3, 1, 1)]
    assert_equal(execution_record_equal(execution_flat_oracle(same_rank, 3), ArgmaxRecord(41, 7, 1)), True)
    # Independent wrong-result mutations witness sensitivity to tie and badOR.
    assert_equal(execution_record_equal(ArgmaxRecord(41, 19, 1), execution_flat_oracle(same_rank, 3)), False)
    assert_equal(execution_record_equal(ArgmaxRecord(41, 7, 0), execution_flat_oracle(same_rank, 3)), False)


def make_execution_test_trace[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int, E: Int](
    descriptor: execution.FinishingExecutionDescriptor[T, I, L, W, S, G],
    inputs: Array[ArgmaxRecord, C], count: Int, seed: Int,
) raises -> Array[execution.FinishingExecutionEvent, E]:
    var trace = Array[execution.FinishingExecutionEvent, E](
        fill=execution.FinishingExecutionEvent(-1, 0, 0, 0))
    var state = execution.make_finishing_execution_state[C](descriptor, count)
    var random = _ExecutionRandom(seed)
    var steps = execution.finishing_execution_expected_steps(descriptor, count)
    assert_equal(steps <= E, True)
    for index in range(steps):
        var enabled = _enabled_events(descriptor, state, False)
        assert_equal(len(enabled) > 0, True)
        trace[index] = enabled[random.below(len(enabled))]
        state = _checked_successor(descriptor, state, inputs, trace[index])
    assert_equal(execution.finishing_execution_complete(descriptor, state), True)
    return trace^


def _assert_replay_oracle[T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, C: Int, E: Int, N: Int](
    candidate: replay.FinishingExecutionReplay[T, I, L, W, S, G, C, E, N],
    inputs: Array[ArgmaxRecord, C], count: Int,
    trace: Array[execution.FinishingExecutionEvent, E], event_count: Int,
) raises:
    assert_equal(candidate.accepted, True)
    assert_equal(candidate.error, 0)
    assert_equal(candidate.event_count, event_count)
    assert_equal(candidate.empty_id, count)
    assert_equal(candidate.next_value, count + 1 + candidate.merge_count)
    var merge_index = 0
    var support = Array[Bool, N * C](fill=False)
    for source in range(count):
        support[source * C + source] = True
    for event_index in range(event_count):
        assert_equal(execution.same_finishing_execution_event(candidate.trace[event_index], trace[event_index]), True)
        if trace[event_index].kind == execution.FINISHING_ADVANCE:
            continue
        assert_equal(candidate.event_indices[merge_index], event_index)
        assert_equal(candidate.event_kinds[merge_index], trace[event_index].kind)
        var destination = count + 1 + merge_index
        var left = candidate.left[merge_index]
        var right = candidate.right[merge_index]
        assert_equal(candidate.destinations[merge_index], destination)
        assert_equal(left >= 0 and left < destination and right >= 0 and right < destination, True)
        if trace[event_index].kind == execution.FINISHING_FILL:
            var source = trace[event_index].generation + trace[event_index].owner + T * trace[event_index].index
            assert_equal(left, count)
            assert_equal(right, source if source < count else count)
        for source in range(C):
            support[destination * C + source] = support[left * C + source] or support[right * C + source]
        merge_index += 1
    assert_equal(merge_index, candidate.merge_count)
    for value_id in range(N):
        assert_equal(candidate.defined[value_id], value_id < candidate.next_value)
        var top: UInt32 = 0
        var token: UInt32 = 0xFFFFFFFF
        var invalid: UInt32 = 0
        for source in range(C):
            assert_equal(candidate.support[value_id * C + source], support[value_id * C + source])
            if support[value_id * C + source]:
                if inputs[source].rank > top:
                    top = inputs[source].rank
                invalid |= inputs[source].invalid
        for source in range(C):
            if support[value_id * C + source] and inputs[source].rank == top and inputs[source].token < token:
                token = inputs[source].token
        assert_equal(execution_record_equal(candidate.values[value_id], ArgmaxRecord(top, token, invalid)), True)
    assert_equal(candidate.root, candidate.output_id)
    for source in range(C):
        assert_equal(candidate.support[candidate.root * C + source], source < count)
    assert_equal(execution_record_equal(candidate.state.output, execution_flat_oracle(inputs, count)), True)


comptime FINISHING_EXECUTION_REPLAY_MUTANTS = 22


def check_finishing_execution_replay_controls() raises:
    var pair = sequential_argmax_plan[2]()
    var single = sequential_argmax_plan[1]()
    var descriptor = execution.make_finishing_execution_descriptor[2, 2, 5, 2, 5, 3](pair, pair, single)
    var inputs = execution_input_fixture[8](0)
    var count = 5
    var length = execution.finishing_execution_expected_steps(descriptor, count)
    var trace = make_execution_test_trace[2, 2, 5, 2, 5, 3, 8, 128](descriptor, inputs, count, 79)
    var candidate = replay.replay_finishing_execution[128](descriptor, inputs, count, trace, length)
    _assert_replay_oracle(candidate, inputs, count, trace, length)
    assert_equal(replay.check_finishing_execution_replay(descriptor, inputs, count, trace, length, candidate), True)
    for mode in range(FINISHING_EXECUTION_REPLAY_MUTANTS):
        var proposed = candidate.copy()
        if mode == 0:
            proposed.root = proposed.empty_id
        elif mode == 1:
            proposed.output_id = proposed.empty_id
        elif mode == 2:
            proposed.left[0] = 0
        elif mode == 3:
            proposed.right[0] = proposed.empty_id
        elif mode == 4:
            proposed.destinations[0] += 1
        elif mode == 5:
            proposed.event_indices[0] += 1
        elif mode == 6:
            proposed.event_kinds[0] = execution.FINISHING_OUTPUT
        elif mode == 7:
            proposed.values[proposed.root].token += 1
        elif mode == 8:
            proposed.support[proposed.root * 8] = False
        elif mode == 9:
            proposed.support[proposed.root * 8 + 7] = True
        elif mode == 10:
            proposed.state.visits[0] += 1
        elif mode == 11:
            proposed.state.output.invalid ^= 1
        elif mode == 12:
            proposed.best_ids[0] = proposed.empty_id
        elif mode == 13:
            proposed.published_ids[0] = proposed.empty_id
        elif mode == 14:
            proposed.subgroup_ids[0] = proposed.empty_id
        elif mode == 15:
            proposed.final_ids[0] = proposed.empty_id
        elif mode == 16:
            proposed.inputs[0].rank += 1
        elif mode == 17:
            proposed.trace[0].generation += 4
        elif mode == 18:
            proposed.event_count -= 1
        elif mode == 19:
            proposed.next_value -= 1
        elif mode == 20:
            proposed.defined[127] = True
        else:
            proposed.destinations[127] = 0
        assert_equal(replay.check_finishing_execution_replay(descriptor, inputs, count, trace, length, proposed), False)
    var altered_trace = trace.copy()
    altered_trace[1] = altered_trace[0]
    assert_equal(replay.replay_finishing_execution[128](descriptor, inputs, count, altered_trace, length).accepted, False)
    altered_trace = trace.copy()
    altered_trace[0].generation += 4
    assert_equal(replay.replay_finishing_execution[128](descriptor, inputs, count, altered_trace, length).accepted, False)
    assert_equal(replay.replay_finishing_execution[128](descriptor, inputs, count, trace, length - 1).accepted, False)
    assert_equal(replay.replay_finishing_execution[128](descriptor, inputs, count, trace, 129).accepted, False)
    assert_equal(replay.replay_finishing_execution[128](descriptor, inputs, count, trace, -1).accepted, False)
    assert_equal(replay.replay_finishing_execution[6](descriptor, inputs, count, trace, length).accepted, False)
    assert_equal(replay.replay_finishing_execution[5](descriptor, inputs, count, trace, length).accepted, False)
    var altered_inputs = inputs.copy()
    altered_inputs[0].rank += 1
    assert_equal(replay.check_finishing_execution_replay(descriptor, altered_inputs, count, trace, length, candidate), False)
    var altered_descriptor = descriptor.copy()
    altered_descriptor.local.nodes[3] = merge_node(0, 2)
    assert_equal(execution.check_finishing_execution_descriptor(altered_descriptor), True)
    assert_equal(replay.check_finishing_execution_replay(altered_descriptor, inputs, count, trace, length, candidate), False)
    altered_descriptor = descriptor.copy()
    altered_descriptor.program.count_limit -= 1
    assert_equal(execution.check_finishing_execution_descriptor(altered_descriptor), True)
    assert_equal(replay.check_finishing_execution_replay(altered_descriptor, inputs, count, trace, length, candidate), False)
    var altered_candidate = candidate.copy()
    altered_candidate.descriptor.program.count_limit -= 1
    assert_equal(replay.check_finishing_execution_replay(descriptor, inputs, count, trace, length, altered_candidate), False)
    var repeated = repeated_finishing_plan()
    var unused_descriptor = execution.make_finishing_execution_descriptor[2, 2, 7, 2, 5, 3](repeated, pair, single)
    var unused_length = execution.finishing_execution_expected_steps(unused_descriptor, count)
    var unused_trace = make_execution_test_trace[2, 2, 7, 2, 5, 3, 8, 128](unused_descriptor, inputs, count, 89)
    var unused_candidate = replay.replay_finishing_execution[128](unused_descriptor, inputs, count, unused_trace, unused_length)
    assert_equal(unused_candidate.accepted, True)
    unused_descriptor.local.nodes[6] = input_node(0)
    assert_equal(execution.check_finishing_execution_descriptor(unused_descriptor), True)
    assert_equal(replay.check_finishing_execution_replay(unused_descriptor, inputs, count,
        unused_trace, unused_length, unused_candidate), False)
    # Equal payloads still cannot disguise a changed source edge.
    var identical = Array[ArgmaxRecord, 8](fill=ArgmaxRecord(17, 19, 1))
    var same_trace = make_execution_test_trace[2, 2, 5, 2, 5, 3, 8, 128](descriptor, identical, count, 83)
    var same_candidate = replay.replay_finishing_execution[128](descriptor, identical, count, same_trace, length)
    _assert_replay_oracle(same_candidate, identical, count, same_trace, length)
    same_candidate.right[0] = (same_candidate.right[0] + 1) % count
    assert_equal(replay.check_finishing_execution_replay(descriptor, identical, count, same_trace, length, same_candidate), False)
    print("Finishing execution replay malformed candidate controls:", FINISHING_EXECUTION_REPLAY_MUTANTS)
