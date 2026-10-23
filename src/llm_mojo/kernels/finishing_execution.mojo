"""Closed, interleavable execution model for the complete finishing pass.

Each thread owns its loop counter, private records, local DAG and accumulator.
Threads may finish different tiles concurrently: no tile-wide barrier is added.
Exited threads publish to a logical subgroup; after all subgroup DAGs finish,
the final DAG and single output write become available.

The descriptor is shared executable data, not arbitrary Mojo body reflection.
Native integer arithmetic, record algebra and the lowering of logical subgroup
joins to warp/shared-memory instructions remain separate trust obligations.
This CPU interpreter neither supplies nor verifies a GPU memory model.
"""
from std.collections import Array
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, EMPTY_NODE, INPUT_NODE, MERGE_NODE, check_argmax_plan,
)
from llm_mojo.kernels.finishing_proof import (
    FinishingProgram, make_finishing_program, FINISH_LT,
)
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, empty_argmax, merge_argmax,
)


comptime FINISHING_EXECUTION_SPEC_VERSION = 1
comptime FINISHING_EXECUTION_CLOSED_PROFILE = 1
comptime FINISHING_FILL = 0
comptime FINISHING_LOCAL_NODE = 1
comptime FINISHING_ACCUMULATE = 2
comptime FINISHING_ADVANCE = 3
comptime FINISHING_PUBLISH = 4
comptime FINISHING_SUBGROUP_NODE = 5
comptime FINISHING_FINAL_NODE = 6
comptime FINISHING_OUTPUT = 7
comptime FINISHING_INVALID_EVENT = -1
comptime FINISHING_THREAD_ACTIVE = 0
comptime FINISHING_THREAD_ACCUMULATED = 1
comptime FINISHING_THREAD_EXITED = 2
comptime FINISHING_THREAD_PUBLISHED = 3


@fieldwise_init
struct FinishingExecutionDescriptor[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](Copyable, Movable):
    var spec_version: Int
    var operation_profile: Int
    var program: FinishingProgram
    var local: ArgmaxPlan[Self.items, Self.local_nodes]
    var subgroup: ArgmaxPlan[Self.group_width, Self.subgroup_nodes]
    var final: ArgmaxPlan[Self.threads // Self.group_width, Self.final_nodes]


def make_finishing_execution_descriptor[
    final_inputs: Int, //,
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](
    local: ArgmaxPlan[items, local_nodes],
    subgroup: ArgmaxPlan[group_width, subgroup_nodes],
    final: ArgmaxPlan[final_inputs, final_nodes],
    count_limit: Int = 2147483647, integer_limit: Int = Int.MAX,
) -> FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
]:
    comptime assert final_inputs == threads // group_width, "finishing execution final shape rejected"
    # Reconstructing the two fields avoids relying on the compiler to equate
    # the symbolic quotient parameter with its normalized integer spelling.
    var final_shape = ArgmaxPlan[threads // group_width, final_nodes](final.nodes.copy(), final.root)
    return FinishingExecutionDescriptor[
        threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
    ](FINISHING_EXECUTION_SPEC_VERSION, FINISHING_EXECUTION_CLOSED_PROFILE,
      make_finishing_program(threads, items, threads * items, count_limit, integer_limit),
      local.copy(), subgroup.copy(), final_shape^)


def check_finishing_execution_descriptor[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
]) -> Bool:
    if (threads < 1 or items < 1 or local_nodes < 1 or group_width < 1
        or threads % group_width != 0 or subgroup_nodes < 1 or final_nodes < 1):
        return False
    if (descriptor.spec_version != FINISHING_EXECUTION_SPEC_VERSION
        or descriptor.operation_profile != FINISHING_EXECUTION_CLOSED_PROFILE):
        return False
    var limit = UInt64(Int.MAX)
    if UInt64(threads) > limit // UInt64(items):
        return False
    var stride = UInt64(threads) * UInt64(items)
    # Bound every native storage/checker matrix and canonical event-space
    # product before performing its signed arithmetic.
    var groups = UInt64(threads // group_width)
    if (UInt64(threads) > limit // UInt64(local_nodes)
        or groups > limit // UInt64(subgroup_nodes)
        or UInt64(local_nodes) > limit // UInt64(items)
        or UInt64(subgroup_nodes) > limit // UInt64(group_width)
        or UInt64(final_nodes) > limit // groups):
        return False
    if UInt64(items) > limit - 3 or UInt64(local_nodes) > limit - 3 - UInt64(items):
        return False
    var per_thread_events = UInt64(items) + UInt64(local_nodes) + 3
    if UInt64(threads) > limit // per_thread_events:
        return False
    var event_space = UInt64(threads) * per_thread_events
    var subgroup_events = groups * UInt64(subgroup_nodes)
    if event_space > limit - subgroup_events:
        return False
    event_space += subgroup_events
    if event_space > limit - UInt64(final_nodes) or event_space + UInt64(final_nodes) >= limit:
        return False
    var p = descriptor.program
    if (p.threads != threads or p.items != items or UInt64(p.stride) != stride
        or p.start != 0 or p.thread_scale != 1 or p.item_scale != threads
        or p.index_bias != 0 or p.loop_guard != FINISH_LT or p.load_guard != FINISH_LT
        or p.count_limit < 0 or p.integer_limit < 1):
        return False
    # Every active padded index and the first exit base fit this native Int.
    if UInt64(p.count_limit) + stride > UInt64(p.integer_limit):
        return False
    return (check_argmax_plan(descriptor.local)
        and check_argmax_plan(descriptor.subgroup)
        and check_argmax_plan(descriptor.final))


@always_inline
def finishing_execution_index[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], base: Int, thread: Int, item: Int) -> Int:
    """The exact affine index consumed by the descriptor's closed Fill."""
    var p = descriptor.program
    return base + p.thread_scale * thread + p.item_scale * item + p.index_bias


@always_inline
def finishing_execution_loop_active[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], base: Int, count: Int) -> Bool:
    return base < count


@always_inline
def finishing_execution_load_active[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], index: Int, count: Int) -> Bool:
    return index < count


@fieldwise_init
struct FinishingExecutionEvent(ImplicitlyCopyable):
    var kind: Int
    var owner: Int
    var index: Int
    var generation: Int


def make_finishing_execution_event(
    kind: Int, owner: Int = 0, index: Int = 0, generation: Int = 0,
) -> FinishingExecutionEvent:
    return FinishingExecutionEvent(kind, owner, index, generation)


def same_finishing_execution_event(a: FinishingExecutionEvent, b: FinishingExecutionEvent) -> Bool:
    return (a.kind == b.kind and a.owner == b.owner
        and a.index == b.index and a.generation == b.generation)


@fieldwise_init
struct FinishingExecutionState[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](Copyable, Movable):
    var count: Int
    var steps: Int
    var base: Array[Int, Self.threads]
    var stage: Array[Int, Self.threads]
    var best: Array[ArgmaxRecord, Self.threads]
    # Private record storage uses the same striped destination as Fill.
    var records: Array[ArgmaxRecord, Self.threads * Self.items]
    var record_ready: Array[Bool, Self.threads * Self.items]
    var local_values: Array[ArgmaxRecord, Self.threads * Self.local_nodes]
    var local_ready: Array[Bool, Self.threads * Self.local_nodes]
    var published: Array[ArgmaxRecord, Self.threads]
    var publish_ready: Array[Bool, Self.threads]
    var subgroup_values: Array[ArgmaxRecord, (Self.threads // Self.group_width) * Self.subgroup_nodes]
    var subgroup_ready: Array[Bool, (Self.threads // Self.group_width) * Self.subgroup_nodes]
    var final_values: Array[ArgmaxRecord, Self.final_nodes]
    var final_ready: Array[Bool, Self.final_nodes]
    var visits: Array[Int, Self.input_capacity]
    var output: ArgmaxRecord
    var output_ready: Bool


def make_finishing_execution_state[
    threads: Int, items: Int, local_nodes: Int,
    group_width: Int, subgroup_nodes: Int, final_nodes: Int, //, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], count: Int) -> FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
]:
    return FinishingExecutionState[
        threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
    ](count, 0, Array[Int, threads](fill=descriptor.program.start),
      Array[Int, threads](fill=FINISHING_THREAD_ACTIVE if count > 0 else FINISHING_THREAD_EXITED),
      Array[ArgmaxRecord, threads](fill=empty_argmax()),
      Array[ArgmaxRecord, threads * items](fill=empty_argmax()),
      Array[Bool, threads * items](fill=False),
      Array[ArgmaxRecord, threads * local_nodes](fill=empty_argmax()),
      Array[Bool, threads * local_nodes](fill=False),
      Array[ArgmaxRecord, threads](fill=empty_argmax()), Array[Bool, threads](fill=False),
      Array[ArgmaxRecord, (threads // group_width) * subgroup_nodes](fill=empty_argmax()),
      Array[Bool, (threads // group_width) * subgroup_nodes](fill=False),
      Array[ArgmaxRecord, final_nodes](fill=empty_argmax()),
      Array[Bool, final_nodes](fill=False), Array[Int, input_capacity](fill=0), empty_argmax(), False)


def finishing_execution_expected_steps[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], count: Int) -> Int:
    if (not check_finishing_execution_descriptor(descriptor)
        or count < 0 or count > descriptor.program.count_limit):
        return -1
    var rounds = UInt64(0)
    if count > 0:
        rounds = 1 + UInt64(count - 1) // UInt64(descriptor.program.stride)
    var limit = UInt64(Int.MAX)
    var per_round = UInt64(items) + UInt64(local_nodes) + 2
    if per_round > limit or (rounds > 0 and rounds > limit // per_round):
        return -1
    var per_thread = rounds * per_round + 1
    if per_thread > limit or UInt64(threads) > limit // per_thread:
        return -1
    var total = UInt64(threads) * per_thread
    var groups = UInt64(threads // group_width)
    if groups > limit // UInt64(subgroup_nodes):
        return -1
    var group_work = groups * UInt64(subgroup_nodes)
    if total > limit - group_work:
        return -1
    total += group_work
    if total > limit - UInt64(final_nodes) or total + UInt64(final_nodes) >= limit:
        return -1
    return Int(total + UInt64(final_nodes) + 1)


def valid_finishing_execution_state[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
]) -> Bool:
    """Check structural bounds, not reachability from the initial state.

    Trace guarantees require make_state followed by accepted steps. Exposed
    native fields let independent oracles inspect state; callers cannot turn
    arbitrary mutations of these fields into a reachability certificate.
    """
    var expected = finishing_execution_expected_steps(descriptor, state.count)
    if (expected < 0 or state.count > input_capacity or state.steps < 0 or state.steps > expected):
        return False
    for thread in range(threads):
        var base = state.base[thread]
        var stage = state.stage[thread]
        if (base < 0 or base % descriptor.program.stride != 0
            or UInt64(base) > UInt64(state.count) + UInt64(descriptor.program.stride)
            or stage < FINISHING_THREAD_ACTIVE or stage > FINISHING_THREAD_PUBLISHED):
            return False
        if (stage <= FINISHING_THREAD_ACCUMULATED) != (base < state.count):
            return False
        if state.publish_ready[thread] != (stage == FINISHING_THREAD_PUBLISHED):
            return False
    return True


def _finishing_records_ready[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
], thread: Int) -> Bool:
    for item in range(items):
        if not state.record_ready[thread + threads * item]:
            return False
    return True


def finishing_execution_event_enabled[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
], event: FinishingExecutionEvent) -> Bool:
    if (not valid_finishing_execution_state(descriptor, state) or state.output_ready
        or state.steps >= finishing_execution_expected_steps(descriptor, state.count)):
        return False
    var owner = event.owner
    var node = event.index
    if event.kind >= FINISHING_FILL and event.kind <= FINISHING_PUBLISH:
        if owner < 0 or owner >= threads or event.generation != state.base[owner]:
            return False
        var stage = state.stage[owner]
        if event.kind == FINISHING_FILL:
            return (stage == FINISHING_THREAD_ACTIVE and node >= 0 and node < items
                and not state.record_ready[owner + threads * node])
        if event.kind == FINISHING_LOCAL_NODE:
            if (stage != FINISHING_THREAD_ACTIVE or node < 0 or node >= local_nodes
                or state.local_ready[owner * local_nodes + node]
                or not _finishing_records_ready(state, owner)):
                return False
            var operation = descriptor.local.nodes[node]
            if operation.operation == MERGE_NODE:
                return (state.local_ready[owner * local_nodes + operation.left]
                    and state.local_ready[owner * local_nodes + operation.right])
            return True
        if node != 0:
            return False
        if event.kind == FINISHING_ACCUMULATE:
            if stage != FINISHING_THREAD_ACTIVE or not _finishing_records_ready(state, owner):
                return False
            for i in range(local_nodes):
                if not state.local_ready[owner * local_nodes + i]:
                    return False
            return True
        if event.kind == FINISHING_ADVANCE:
            return stage == FINISHING_THREAD_ACCUMULATED
        return stage == FINISHING_THREAD_EXITED and not state.publish_ready[owner]
    if event.generation != 0:
        return False
    if event.kind == FINISHING_SUBGROUP_NODE:
        if (owner < 0 or owner >= threads // group_width or node < 0 or node >= subgroup_nodes
            or state.subgroup_ready[owner * subgroup_nodes + node]):
            return False
        for member in range(group_width):
            if not state.publish_ready[owner * group_width + member]:
                return False
        var operation = descriptor.subgroup.nodes[node]
        if operation.operation == MERGE_NODE:
            return (state.subgroup_ready[owner * subgroup_nodes + operation.left]
                and state.subgroup_ready[owner * subgroup_nodes + operation.right])
        return True
    if owner != 0:
        return False
    if event.kind == FINISHING_FINAL_NODE:
        if node < 0 or node >= final_nodes or state.final_ready[node]:
            return False
        # The final shared barrier follows evaluation/publication by every
        # subgroup. Even well-formed nodes unused by a root are executed.
        for i in range((threads // group_width) * subgroup_nodes):
            if not state.subgroup_ready[i]:
                return False
        var operation = descriptor.final.nodes[node]
        if operation.operation == MERGE_NODE:
            return state.final_ready[operation.left] and state.final_ready[operation.right]
        return True
    if event.kind == FINISHING_OUTPUT and node == 0:
        for i in range(final_nodes):
            if not state.final_ready[i]:
                return False
        return True
    return False


def step_finishing_execution[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], mut state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
], inputs: Array[ArgmaxRecord, input_capacity], event: FinishingExecutionEvent) -> Bool:
    """Execute exactly one canonical event; rejection preserves every cell."""
    if not finishing_execution_event_enabled(descriptor, state, event):
        return False
    var owner = event.owner
    var node = event.index
    if event.kind == FINISHING_FILL:
        var source = finishing_execution_index(descriptor, state.base[owner], owner, node)
        var value = empty_argmax()
        if finishing_execution_load_active(descriptor, source, state.count):
            # A malformed caller-mutated state cannot overflow the visit
            # counter or silently hide a repeated source behind idempotence.
            if source < 0 or source >= input_capacity or state.visits[source] == Int.MAX:
                return False
            value = inputs[source]
            state.visits[source] += 1
        var slot = owner + threads * node
        state.records[slot] = value
        state.record_ready[slot] = True
    elif event.kind == FINISHING_LOCAL_NODE:
        var operation = descriptor.local.nodes[node]
        var value = empty_argmax()
        if operation.operation == INPUT_NODE:
            value = state.records[owner + threads * operation.left]
        elif operation.operation == MERGE_NODE:
            value = merge_argmax(state.local_values[owner * local_nodes + operation.left],
                                state.local_values[owner * local_nodes + operation.right])
        state.local_values[owner * local_nodes + node] = value
        state.local_ready[owner * local_nodes + node] = True
    elif event.kind == FINISHING_ACCUMULATE:
        state.best[owner] = merge_argmax(state.best[owner],
            state.local_values[owner * local_nodes + descriptor.local.root])
        state.stage[owner] = FINISHING_THREAD_ACCUMULATED
    elif event.kind == FINISHING_ADVANCE:
        state.base[owner] += descriptor.program.stride
        for item in range(items):
            state.record_ready[owner + threads * item] = False
            state.records[owner + threads * item] = empty_argmax()
        for i in range(local_nodes):
            state.local_ready[owner * local_nodes + i] = False
            state.local_values[owner * local_nodes + i] = empty_argmax()
        state.stage[owner] = (FINISHING_THREAD_ACTIVE if state.base[owner] < state.count
                              else FINISHING_THREAD_EXITED)
    elif event.kind == FINISHING_PUBLISH:
        state.published[owner] = state.best[owner]
        state.publish_ready[owner] = True
        state.stage[owner] = FINISHING_THREAD_PUBLISHED
    elif event.kind == FINISHING_SUBGROUP_NODE:
        var operation = descriptor.subgroup.nodes[node]
        var value = empty_argmax()
        if operation.operation == INPUT_NODE:
            value = state.published[owner * group_width + operation.left]
        elif operation.operation == MERGE_NODE:
            value = merge_argmax(state.subgroup_values[owner * subgroup_nodes + operation.left],
                                state.subgroup_values[owner * subgroup_nodes + operation.right])
        state.subgroup_values[owner * subgroup_nodes + node] = value
        state.subgroup_ready[owner * subgroup_nodes + node] = True
    elif event.kind == FINISHING_FINAL_NODE:
        var operation = descriptor.final.nodes[node]
        var value = empty_argmax()
        if operation.operation == INPUT_NODE:
            value = state.subgroup_values[operation.left * subgroup_nodes + descriptor.subgroup.root]
        elif operation.operation == MERGE_NODE:
            value = merge_argmax(state.final_values[operation.left], state.final_values[operation.right])
        state.final_values[node] = value
        state.final_ready[node] = True
    else:
        state.output = state.final_values[descriptor.final.root]
        state.output_ready = True
    state.steps += 1
    return True


def finishing_execution_remaining_steps[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
]) -> Int:
    """Native event rank; every accepted transition from a fresh state drops 1.

    This executable rank is not itself a universal proof of its arithmetic.
    Descriptor checking rejects unrepresentable complete event counts.
    """
    if not valid_finishing_execution_state(descriptor, state):
        return -1
    var total = UInt64(0)
    var per_round = UInt64(items) + UInt64(local_nodes) + 2
    for thread in range(threads):
        var stage = state.stage[thread]
        if stage <= FINISHING_THREAD_ACCUMULATED:
            var future = UInt64((state.count - 1 - state.base[thread]) // descriptor.program.stride)
            total += future * per_round + 1  # advance
            if stage == FINISHING_THREAD_ACTIVE:
                total += 1  # accumulate
                for item in range(items):
                    total += UInt64(not state.record_ready[thread + threads * item])
                for node in range(local_nodes):
                    total += UInt64(not state.local_ready[thread * local_nodes + node])
            total += 1  # eventual publish
        elif stage == FINISHING_THREAD_EXITED:
            total += 1
    for i in range((threads // group_width) * subgroup_nodes):
        total += UInt64(not state.subgroup_ready[i])
    for i in range(final_nodes):
        total += UInt64(not state.final_ready[i])
    total += UInt64(not state.output_ready)
    if total > UInt64(Int.MAX):
        return -1
    return Int(total)


def finishing_execution_complete[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
]) -> Bool:
    if (not valid_finishing_execution_state(descriptor, state) or not state.output_ready
        or finishing_execution_remaining_steps(descriptor, state) != 0
        or state.steps != finishing_execution_expected_steps(descriptor, state.count)):
        return False
    for i in range(input_capacity):
        if state.visits[i] != (1 if i < state.count else 0):
            return False
    return True


def finishing_execution_event_at[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
], ordinal: Int) -> FinishingExecutionEvent:
    """Enumerate the whole canonical event space, enabled or disabled.

    The order is stable; the generation is each thread's current base.
    Consumers independently filter with event_enabled or choose an enabled
    ordinal. Invalid/out-of-range ordinals return an invalid event.
    """
    if ordinal < 0 or not check_finishing_execution_descriptor(descriptor):
        return FinishingExecutionEvent(FINISHING_INVALID_EVENT, 0, 0, 0)
    var position = ordinal
    var per_thread = items + local_nodes + 3
    if position < threads * per_thread:
        var owner = position // per_thread
        var part = position % per_thread
        if part < items:
            return FinishingExecutionEvent(FINISHING_FILL, owner, part, state.base[owner])
        part -= items
        if part < local_nodes:
            return FinishingExecutionEvent(FINISHING_LOCAL_NODE, owner, part, state.base[owner])
        return FinishingExecutionEvent(FINISHING_ACCUMULATE + part - local_nodes,
                                      owner, 0, state.base[owner])
    position -= threads * per_thread
    if position < (threads // group_width) * subgroup_nodes:
        return FinishingExecutionEvent(FINISHING_SUBGROUP_NODE, position // subgroup_nodes,
                                      position % subgroup_nodes, 0)
    position -= (threads // group_width) * subgroup_nodes
    if position < final_nodes:
        return FinishingExecutionEvent(FINISHING_FINAL_NODE, 0, position, 0)
    if position == final_nodes:
        return FinishingExecutionEvent(FINISHING_OUTPUT, 0, 0, 0)
    return FinishingExecutionEvent(FINISHING_INVALID_EVENT, 0, 0, 0)


def finishing_execution_event_capacity[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
]) -> Int:
    if not check_finishing_execution_descriptor(descriptor):
        return -1
    return threads * (items + local_nodes + 3) + (threads // group_width) * subgroup_nodes + final_nodes + 1


def finishing_execution_enabled_count[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
]) -> Int:
    var count = 0
    for ordinal in range(finishing_execution_event_capacity(descriptor)):
        if finishing_execution_event_enabled(descriptor, state,
                finishing_execution_event_at(descriptor, state, ordinal)):
            count += 1
    return count


def finishing_execution_enabled_event[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], state: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
], ordinal: Int) -> FinishingExecutionEvent:
    if ordinal >= 0:
        var enabled = 0
        for index in range(finishing_execution_event_capacity(descriptor)):
            var event = finishing_execution_event_at(descriptor, state, index)
            if finishing_execution_event_enabled(descriptor, state, event):
                if enabled == ordinal:
                    return event
                enabled += 1
    return FinishingExecutionEvent(FINISHING_INVALID_EVENT, 0, 0, 0)


def _same_finishing_records[size: Int](a: Array[ArgmaxRecord, size], b: Array[ArgmaxRecord, size]) -> Bool:
    for i in range(size):
        if (a[i].rank != b[i].rank or a[i].token != b[i].token or a[i].invalid != b[i].invalid):
            return False
    return True


def _same_finishing_ints[size: Int](a: Array[Int, size], b: Array[Int, size]) -> Bool:
    for i in range(size):
        if a[i] != b[i]:
            return False
    return True


def _same_finishing_bools[size: Int](a: Array[Bool, size], b: Array[Bool, size]) -> Bool:
    for i in range(size):
        if a[i] != b[i]:
            return False
    return True


def same_finishing_execution_state[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
](a: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
], b: FinishingExecutionState[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes, input_capacity,
]) -> Bool:
    return (a.count == b.count and a.steps == b.steps and a.output_ready == b.output_ready
        and a.output.rank == b.output.rank and a.output.token == b.output.token
        and a.output.invalid == b.output.invalid
        and _same_finishing_ints(a.base, b.base) and _same_finishing_ints(a.stage, b.stage)
        and _same_finishing_ints(a.visits, b.visits)
        and _same_finishing_records(a.best, b.best) and _same_finishing_records(a.records, b.records)
        and _same_finishing_records(a.local_values, b.local_values)
        and _same_finishing_records(a.published, b.published)
        and _same_finishing_records(a.subgroup_values, b.subgroup_values)
        and _same_finishing_records(a.final_values, b.final_values)
        and _same_finishing_bools(a.record_ready, b.record_ready)
        and _same_finishing_bools(a.local_ready, b.local_ready)
        and _same_finishing_bools(a.publish_ready, b.publish_ready)
        and _same_finishing_bools(a.subgroup_ready, b.subgroup_ready)
        and _same_finishing_bools(a.final_ready, b.final_ready))
