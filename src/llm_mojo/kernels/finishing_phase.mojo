"""One logical tile's private record-population execution model.

Each Fill reads an immutable partial record (or uses Empty for padding), then
writes exactly one private (thread,item) record and its readiness cell. This
ends before the local reduction in token_selection._finish. It introduces no
runtime barrier: readiness for reduction is local to each thread.

This ordinary Mojo interpreter is a model, not reflection of the GPU body.
finishing_phase_check binds requests to checked Nat commutation proofs. Native
address arithmetic, model/source correspondence and device behavior are separate
trust assumptions; proof acceptance does not verify a Metal memory model.
"""
from std.collections import Array
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord, empty_argmax


@fieldwise_init
struct FinishingPhase(ImplicitlyCopyable):
    var threads: Int
    var items: Int
    var base: Int
    var count: Int


@fieldwise_init
struct FinishingFill(ImplicitlyCopyable):
    var thread: Int
    var item: Int
    var destination: Int
    var source: Int
    var loads: Bool


@fieldwise_init
struct FinishingPhaseState[capacity: Int](Copyable, Movable):
    var phase: FinishingPhase
    var records: Array[ArgmaxRecord, Self.capacity]
    var ready: Array[Bool, Self.capacity]


def same_finishing_phase(a: FinishingPhase, b: FinishingPhase) -> Bool:
    return (a.threads == b.threads and a.items == b.items
        and a.base == b.base and a.count == b.count)


def same_finishing_fill(a: FinishingFill, b: FinishingFill) -> Bool:
    return (a.thread == b.thread and a.item == b.item
        and a.destination == b.destination and a.source == b.source
        and a.loads == b.loads)


def valid_finishing_phase(phase: FinishingPhase) -> Bool:
    if (phase.threads <= 0 or phase.items <= 0
        or phase.base < 0 or phase.count < 0):
        return False
    # Unsigned checks avoid overflowing a signed product or padded index.
    # These are descriptor checks, not object-language arithmetic theorems.
    var limit = UInt64(Int.MAX)
    if UInt64(phase.threads) > limit // UInt64(phase.items):
        return False
    var tile = UInt64(phase.threads) * UInt64(phase.items)
    if phase.base < phase.count and UInt64(phase.base) + tile > limit:
        return False
    return True


def finishing_phase_slots(phase: FinishingPhase) -> Int:
    if not valid_finishing_phase(phase):
        return -1
    return Int(UInt64(phase.threads) * UInt64(phase.items))


def finishing_phase_active(phase: FinishingPhase) -> Bool:
    return valid_finishing_phase(phase) and phase.base < phase.count


def make_finishing_fill(phase: FinishingPhase, thread: Int, item: Int) -> FinishingFill:
    if (not finishing_phase_active(phase) or thread < 0 or thread >= phase.threads
        or item < 0 or item >= phase.items):
        return FinishingFill(thread, item, -1, -1, False)
    var destination = Int(UInt64(thread) + UInt64(phase.threads) * UInt64(item))
    var source = Int(UInt64(phase.base) + UInt64(destination))
    return FinishingFill(thread, item, destination, source, source < phase.count)


def check_finishing_fill(phase: FinishingPhase, action: FinishingFill) -> Bool:
    if (not finishing_phase_active(phase) or action.thread < 0
        or action.thread >= phase.threads or action.item < 0 or action.item >= phase.items):
        return False
    return same_finishing_fill(action, make_finishing_fill(phase, action.thread, action.item))


def make_finishing_phase_state[capacity: Int](phase: FinishingPhase) -> FinishingPhaseState[capacity]:
    return FinishingPhaseState[capacity](phase,
        Array[ArgmaxRecord, capacity](fill=empty_argmax()),
        Array[Bool, capacity](fill=False))


def valid_finishing_phase_state[capacity: Int](state: FinishingPhaseState[capacity]) -> Bool:
    var slots = finishing_phase_slots(state.phase)
    return slots > 0 and slots <= capacity


def finishing_fill_enabled[capacity: Int](
    state: FinishingPhaseState[capacity], action: FinishingFill,
) -> Bool:
    return (valid_finishing_phase_state(state) and check_finishing_fill(state.phase, action)
        and not state.ready[action.destination])


def step_finishing_fill[capacity: Int, input_capacity: Int](
    mut state: FinishingPhaseState[capacity], inputs: Array[ArgmaxRecord, input_capacity],
    action: FinishingFill,
) -> Bool:
    """Execute one enabled Fill; failure leaves every modeled cell unchanged."""
    if state.phase.count > input_capacity or not finishing_fill_enabled(state, action):
        return False
    var payload = empty_argmax()
    if action.loads:
        payload = inputs[action.source]
    state.records[action.destination] = payload
    state.ready[action.destination] = True
    return True


def finishing_thread_ready[capacity: Int](state: FinishingPhaseState[capacity], thread: Int) -> Bool:
    """Local reduction prerequisite; this is not an all-thread barrier."""
    if (not valid_finishing_phase_state(state) or not finishing_phase_active(state.phase)
        or thread < 0 or thread >= state.phase.threads):
        return False
    for item in range(state.phase.items):
        var destination = Int(UInt64(thread) + UInt64(state.phase.threads) * UInt64(item))
        if not state.ready[destination]:
            return False
    return True


def same_finishing_phase_state[capacity: Int](
    a: FinishingPhaseState[capacity], b: FinishingPhaseState[capacity],
) -> Bool:
    if not same_finishing_phase(a.phase, b.phase):
        return False
    for slot in range(capacity):
        if (a.ready[slot] != b.ready[slot]
            or a.records[slot].rank != b.records[slot].rank
            or a.records[slot].token != b.records[slot].token
            or a.records[slot].invalid != b.records[slot].invalid):
            return False
    return True
