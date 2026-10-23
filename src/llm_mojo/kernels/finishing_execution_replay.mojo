"""Fresh finishing traces bound to immutable merge-only SSA provenance.

Edges are reconstructed from the closed descriptor and actual accepted events.
Callers supply neither edges nor payload callbacks. Every value-producing event
gets a fresh destination; native register reuse is tracked by generation and
parallel SSA identifiers. Advance clears private identifiers and creates none.

Copies lower to Join(Empty, value). Their native word equality is checked here;
identifying this encoding with the abstract proof's Join/identity operation is
an explicit semantic refinement obligation. This module is ordinary native
Mojo: Boolean support and concrete replay are not a kernel soundness theorem.
"""
from std.collections import Array
from llm_mojo.kernels.argmax_reduction import EMPTY_NODE, INPUT_NODE, MERGE_NODE
from llm_mojo.kernels.finishing_execution import (
    FinishingExecutionDescriptor, FinishingExecutionEvent, FinishingExecutionState,
    FINISHING_FILL, FINISHING_LOCAL_NODE, FINISHING_ACCUMULATE, FINISHING_ADVANCE,
    FINISHING_PUBLISH, FINISHING_SUBGROUP_NODE, FINISHING_FINAL_NODE, FINISHING_OUTPUT,
    FINISHING_INVALID_EVENT, check_finishing_execution_descriptor,
    make_finishing_execution_state, finishing_execution_index,
    finishing_execution_event_enabled, step_finishing_execution,
    finishing_execution_complete, finishing_execution_expected_steps,
    same_finishing_execution_event, same_finishing_execution_state,
)
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, empty_argmax, merge_argmax,
)


comptime FINISHING_REPLAY_OK = 0
comptime FINISHING_REPLAY_BAD_DESCRIPTOR = 1
comptime FINISHING_REPLAY_BAD_EXTENT = 2
comptime FINISHING_REPLAY_BAD_TRACE_LENGTH = 3
comptime FINISHING_REPLAY_VALUE_CAPACITY = 4
comptime FINISHING_REPLAY_ILLEGAL_EVENT = 5
comptime FINISHING_REPLAY_BAD_PARENT = 6
comptime FINISHING_REPLAY_VALUE_MISMATCH = 7
comptime FINISHING_REPLAY_INCOMPLETE = 8
comptime FINISHING_REPLAY_BAD_SUPPORT = 9


@fieldwise_init
struct FinishingExecutionReplay[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
    event_capacity: Int, value_capacity: Int,
](Copyable, Movable):
    var accepted: Bool
    var error: Int
    var descriptor: FinishingExecutionDescriptor[
        Self.threads, Self.items, Self.local_nodes, Self.group_width,
        Self.subgroup_nodes, Self.final_nodes,
    ]
    var state: FinishingExecutionState[
        Self.threads, Self.items, Self.local_nodes, Self.group_width,
        Self.subgroup_nodes, Self.final_nodes, Self.input_capacity,
    ]
    var inputs: Array[ArgmaxRecord, Self.input_capacity]
    var trace: Array[FinishingExecutionEvent, Self.event_capacity]
    var event_count: Int
    # Initial identifiers [0,count) are input leaves; count is the Empty leaf.
    # destinations/left/right are merge streams, indexed by merge ordinal.
    var empty_id: Int
    var next_value: Int
    var merge_count: Int
    var destinations: Array[Int, Self.value_capacity]
    var left: Array[Int, Self.value_capacity]
    var right: Array[Int, Self.value_capacity]
    var event_indices: Array[Int, Self.value_capacity]
    var event_kinds: Array[Int, Self.value_capacity]
    var values: Array[ArgmaxRecord, Self.value_capacity]
    var defined: Array[Bool, Self.value_capacity]
    var support: Array[Bool, Self.value_capacity * Self.input_capacity]
    var record_ids: Array[Int, Self.threads * Self.items]
    var local_ids: Array[Int, Self.threads * Self.local_nodes]
    var best_ids: Array[Int, Self.threads]
    var published_ids: Array[Int, Self.threads]
    var subgroup_ids: Array[Int, (Self.threads // Self.group_width) * Self.subgroup_nodes]
    var final_ids: Array[Int, Self.final_nodes]
    var output_id: Int
    var root: Int


def _empty_finishing_execution_replay[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
    event_capacity: Int, //, value_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], inputs: Array[ArgmaxRecord, input_capacity], count: Int,
    trace: Array[FinishingExecutionEvent, event_capacity],
) -> FinishingExecutionReplay[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
    input_capacity, event_capacity, value_capacity,
]:
    var state = make_finishing_execution_state[input_capacity](descriptor, count)
    return FinishingExecutionReplay[
        threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
        input_capacity, event_capacity, value_capacity,
    ](False, FINISHING_REPLAY_INCOMPLETE, descriptor.copy(), state^, inputs.copy(),
      Array[FinishingExecutionEvent, event_capacity](
          fill=FinishingExecutionEvent(FINISHING_INVALID_EVENT, 0, 0, 0)),
      0, count, 0, 0,
      Array[Int, value_capacity](fill=-1), Array[Int, value_capacity](fill=-1),
      Array[Int, value_capacity](fill=-1), Array[Int, value_capacity](fill=-1),
      Array[Int, value_capacity](fill=-1),
      Array[ArgmaxRecord, value_capacity](fill=empty_argmax()),
      Array[Bool, value_capacity](fill=False),
      Array[Bool, value_capacity * input_capacity](fill=False),
      Array[Int, threads * items](fill=-1), Array[Int, threads * local_nodes](fill=-1),
      Array[Int, threads](fill=-1), Array[Int, threads](fill=-1),
      Array[Int, (threads // group_width) * subgroup_nodes](fill=-1),
      Array[Int, final_nodes](fill=-1), -1, -1)


def _finishing_replay_same_record(a: ArgmaxRecord, b: ArgmaxRecord) -> Bool:
    return a.rank == b.rank and a.token == b.token and a.invalid == b.invalid


def same_finishing_execution_descriptor[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int,
](a: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], b: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
]) -> Bool:
    """Exact executable data equality, including otherwise unused metadata."""
    var p = a.program
    var q = b.program
    if (a.spec_version != b.spec_version or a.operation_profile != b.operation_profile
        or p.threads != q.threads or p.items != q.items or p.stride != q.stride
        or p.start != q.start or p.thread_scale != q.thread_scale or p.item_scale != q.item_scale
        or p.index_bias != q.index_bias or p.loop_guard != q.loop_guard or p.load_guard != q.load_guard
        or p.count_limit != q.count_limit or p.integer_limit != q.integer_limit
        or a.local.root != b.local.root or a.subgroup.root != b.subgroup.root
        or a.final.root != b.final.root):
        return False
    for i in range(local_nodes):
        var x = a.local.nodes[i]
        var y = b.local.nodes[i]
        if x.operation != y.operation or x.left != y.left or x.right != y.right:
            return False
    for i in range(subgroup_nodes):
        var x = a.subgroup.nodes[i]
        var y = b.subgroup.nodes[i]
        if x.operation != y.operation or x.left != y.left or x.right != y.right:
            return False
    for i in range(final_nodes):
        var x = a.final.nodes[i]
        var y = b.final.nodes[i]
        if x.operation != y.operation or x.left != y.left or x.right != y.right:
            return False
    return True


def replay_finishing_execution[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
    event_capacity: Int, //, value_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], inputs: Array[ArgmaxRecord, input_capacity], count: Int,
    trace: Array[FinishingExecutionEvent, event_capacity], event_count: Int,
) -> FinishingExecutionReplay[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
    input_capacity, event_capacity, value_capacity,
]:
    """Reconstruct a complete fresh trace; insufficient bounds fail closed.

    Native payload comparison is supplementary to exact provenance. Equal
    records never permit different source indices or dependency edges.
    """
    var result = _empty_finishing_execution_replay[value_capacity](
        descriptor, inputs, count, trace)
    if not check_finishing_execution_descriptor(descriptor):
        result.error = FINISHING_REPLAY_BAD_DESCRIPTOR
        return result^
    if count < 0 or count > input_capacity or count > descriptor.program.count_limit:
        result.error = FINISHING_REPLAY_BAD_EXTENT
        return result^
    if event_count < 0 or event_count > event_capacity:
        result.error = FINISHING_REPLAY_BAD_TRACE_LENGTH
        return result^
    if (value_capacity < 1 or count >= value_capacity
        or (input_capacity > 0
            and UInt64(value_capacity) > UInt64(Int.MAX) // UInt64(input_capacity))):
        result.error = FINISHING_REPLAY_VALUE_CAPACITY
        return result^
    if event_count != finishing_execution_expected_steps(descriptor, count):
        result.error = FINISHING_REPLAY_BAD_TRACE_LENGTH
        return result^
    result.next_value = count + 1
    for i in range(count):
        result.values[i] = inputs[i]
        result.defined[i] = True
        result.support[i * input_capacity + i] = True
    result.values[count] = empty_argmax()
    result.defined[count] = True
    for thread in range(threads):
        result.best_ids[thread] = count

    for event_index in range(event_count):
        var event = trace[event_index]
        if not finishing_execution_event_enabled(descriptor, result.state, event):
            result.error = FINISHING_REPLAY_ILLEGAL_EVENT
            return result^
        var owner = event.owner
        var node = event.index
        var left = result.empty_id
        var right = result.empty_id
        if event.kind == FINISHING_FILL:
            var source = finishing_execution_index(descriptor,
                result.state.base[owner], owner, node)
            if source < count:
                right = source
        elif event.kind == FINISHING_LOCAL_NODE:
            var operation = descriptor.local.nodes[node]
            if operation.operation == INPUT_NODE:
                right = result.record_ids[owner + threads * operation.left]
            elif operation.operation == MERGE_NODE:
                left = result.local_ids[owner * local_nodes + operation.left]
                right = result.local_ids[owner * local_nodes + operation.right]
        elif event.kind == FINISHING_ACCUMULATE:
            left = result.best_ids[owner]
            right = result.local_ids[owner * local_nodes + descriptor.local.root]
        elif event.kind == FINISHING_PUBLISH:
            right = result.best_ids[owner]
        elif event.kind == FINISHING_SUBGROUP_NODE:
            var operation = descriptor.subgroup.nodes[node]
            if operation.operation == INPUT_NODE:
                right = result.published_ids[owner * group_width + operation.left]
            elif operation.operation == MERGE_NODE:
                left = result.subgroup_ids[owner * subgroup_nodes + operation.left]
                right = result.subgroup_ids[owner * subgroup_nodes + operation.right]
        elif event.kind == FINISHING_FINAL_NODE:
            var operation = descriptor.final.nodes[node]
            if operation.operation == INPUT_NODE:
                right = result.subgroup_ids[operation.left * subgroup_nodes + descriptor.subgroup.root]
            elif operation.operation == MERGE_NODE:
                left = result.final_ids[operation.left]
                right = result.final_ids[operation.right]
        elif event.kind == FINISHING_OUTPUT:
            right = result.final_ids[descriptor.final.root]

        var produces = event.kind != FINISHING_ADVANCE
        var destination = result.next_value
        if produces:
            if destination >= value_capacity or result.merge_count >= value_capacity:
                result.error = FINISHING_REPLAY_VALUE_CAPACITY
                return result^
            if (left < 0 or left >= destination or right < 0 or right >= destination
                or not result.defined[left] or not result.defined[right]):
                result.error = FINISHING_REPLAY_BAD_PARENT
                return result^
        # Nothing in the result is committed for this event until its native
        # action has passed the same enabledness/index/counter preflight.
        if not step_finishing_execution(descriptor, result.state, inputs, event):
            result.error = FINISHING_REPLAY_ILLEGAL_EVENT
            return result^
        result.trace[result.event_count] = event
        result.event_count += 1
        if not produces:
            for item in range(items):
                result.record_ids[owner + threads * item] = -1
            for i in range(local_nodes):
                result.local_ids[owner * local_nodes + i] = -1
            continue

        var value = merge_argmax(result.values[left], result.values[right])
        var actual = empty_argmax()
        if event.kind == FINISHING_FILL:
            actual = result.state.records[owner + threads * node]
            result.record_ids[owner + threads * node] = destination
        elif event.kind == FINISHING_LOCAL_NODE:
            actual = result.state.local_values[owner * local_nodes + node]
            result.local_ids[owner * local_nodes + node] = destination
        elif event.kind == FINISHING_ACCUMULATE:
            actual = result.state.best[owner]
            result.best_ids[owner] = destination
        elif event.kind == FINISHING_PUBLISH:
            actual = result.state.published[owner]
            result.published_ids[owner] = destination
        elif event.kind == FINISHING_SUBGROUP_NODE:
            actual = result.state.subgroup_values[owner * subgroup_nodes + node]
            result.subgroup_ids[owner * subgroup_nodes + node] = destination
        elif event.kind == FINISHING_FINAL_NODE:
            actual = result.state.final_values[node]
            result.final_ids[node] = destination
        else:
            actual = result.state.output
            result.output_id = destination
            result.root = destination
        if not _finishing_replay_same_record(value, actual):
            result.error = FINISHING_REPLAY_VALUE_MISMATCH
            return result^
        var merge_index = result.merge_count
        result.destinations[merge_index] = destination
        result.left[merge_index] = left
        result.right[merge_index] = right
        result.event_indices[merge_index] = event_index
        result.event_kinds[merge_index] = event.kind
        result.values[destination] = value
        result.defined[destination] = True
        for source in range(input_capacity):
            result.support[destination * input_capacity + source] = (
                result.support[left * input_capacity + source]
                or result.support[right * input_capacity + source])
        result.next_value += 1
        result.merge_count += 1

    if not finishing_execution_complete(descriptor, result.state) or result.root < 0:
        result.error = FINISHING_REPLAY_INCOMPLETE
        return result^
    for value_id in range(result.next_value):
        for source in range(count, input_capacity):
            if result.support[value_id * input_capacity + source]:
                result.error = FINISHING_REPLAY_BAD_SUPPORT
                return result^
    for source in range(input_capacity):
        if result.support[result.root * input_capacity + source] != (source < count):
            result.error = FINISHING_REPLAY_BAD_SUPPORT
            return result^
    result.accepted = True
    result.error = FINISHING_REPLAY_OK
    return result^


def finishing_replay_initially_ready[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
    event_capacity: Int, value_capacity: Int,
](replay: FinishingExecutionReplay[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
    input_capacity, event_capacity, value_capacity,
], value_id: Int) -> Bool:
    return value_id >= 0 and value_id <= replay.empty_id and value_id < value_capacity


def _finishing_replay_same_ints[size: Int](a: Array[Int, size], b: Array[Int, size]) -> Bool:
    for i in range(size):
        if a[i] != b[i]:
            return False
    return True


def same_finishing_execution_replay[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
    event_capacity: Int, value_capacity: Int,
](a: FinishingExecutionReplay[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
    input_capacity, event_capacity, value_capacity,
], b: FinishingExecutionReplay[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
    input_capacity, event_capacity, value_capacity,
]) -> Bool:
    if (a.accepted != b.accepted or a.error != b.error or a.event_count != b.event_count
        or a.empty_id != b.empty_id or a.next_value != b.next_value or a.merge_count != b.merge_count
        or a.output_id != b.output_id or a.root != b.root
        or not same_finishing_execution_descriptor(a.descriptor, b.descriptor)
        or not same_finishing_execution_state(a.state, b.state)):
        return False
    for i in range(input_capacity):
        if not _finishing_replay_same_record(a.inputs[i], b.inputs[i]):
            return False
    for i in range(event_capacity):
        if not same_finishing_execution_event(a.trace[i], b.trace[i]):
            return False
    for i in range(value_capacity):
        if (a.defined[i] != b.defined[i]
            or not _finishing_replay_same_record(a.values[i], b.values[i])):
            return False
    for i in range(value_capacity * input_capacity):
        if a.support[i] != b.support[i]:
            return False
    return (_finishing_replay_same_ints(a.destinations, b.destinations)
        and _finishing_replay_same_ints(a.left, b.left)
        and _finishing_replay_same_ints(a.right, b.right)
        and _finishing_replay_same_ints(a.event_indices, b.event_indices)
        and _finishing_replay_same_ints(a.event_kinds, b.event_kinds)
        and _finishing_replay_same_ints(a.record_ids, b.record_ids)
        and _finishing_replay_same_ints(a.local_ids, b.local_ids)
        and _finishing_replay_same_ints(a.best_ids, b.best_ids)
        and _finishing_replay_same_ints(a.published_ids, b.published_ids)
        and _finishing_replay_same_ints(a.subgroup_ids, b.subgroup_ids)
        and _finishing_replay_same_ints(a.final_ids, b.final_ids))


def check_finishing_execution_replay[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, input_capacity: Int,
    event_capacity: Int, value_capacity: Int,
](descriptor: FinishingExecutionDescriptor[
    threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
], inputs: Array[ArgmaxRecord, input_capacity], count: Int,
    trace: Array[FinishingExecutionEvent, event_capacity], event_count: Int,
    candidate: FinishingExecutionReplay[
        threads, items, local_nodes, group_width, subgroup_nodes, final_nodes,
        input_capacity, event_capacity, value_capacity,
    ],
) -> Bool:
    """Check proposed streams against a new replay, not a digest or payload.

    Neither candidate.accepted nor its metadata influences reconstruction.
    Unused fields are canonical and compared as well as used stream entries.
    """
    if not same_finishing_execution_descriptor(descriptor, candidate.descriptor):
        return False
    var expected = replay_finishing_execution[value_capacity](
        descriptor, inputs, count, trace, event_count)
    return expected.accepted and candidate.accepted and same_finishing_execution_replay(expected, candidate)
