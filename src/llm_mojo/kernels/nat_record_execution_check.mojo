"""Independent specification of the complete correlated record trace.

No producer or claimed theorem signature is imported. Data bodies and public
statement types are reconstructed independently and checked with ordinary
kernel judgments. Counts, event stores, trace length, initial inputs and token
bound remain universal. All submitted syntax, including unused nodes, is checked.

The heap-held workspace uses the canonical in-place kernel constructor. Its
initialization values and cache seeding are preserved; no typing rule or process
stack limit changes. Resource failures are distinct from invalid submissions.

This Nat/Boolean profile leaves UInt32 interpretation, Mojo source/SSA
correspondence, compiler/GPU/memory behavior, fairness and full-model numerical
determinism as separate obligations.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase_check as phase
import llm_mojo.kernels.finishing_execution_check as execution
import llm_mojo.kernels.finishing_join_check as joined
import llm_mojo.kernels.nat_score_bridge_check as score_profile
from llm_mojo.kernels.finishing_phase_check import (
    _nat, _prop, _var, _pi, _lam, _named, _app, _zero, _succ, _eq,
    _rec, _store_type, _pis, _lams,
)

comptime NAT_RECORD_EXECUTION_DEFINITIONS = 183
comptime RECORD_CHECK_ACCEPTED = 0
comptime RECORD_CHECK_INVALID_PROOF = 1
comptime RECORD_CHECK_RESOURCE_PREFLIGHT = 2
comptime RECORD_CHECK_RESOURCE_EXHAUSTED = 3


@fieldwise_init
struct RecordExecutionCheckResult(ImplicitlyCopyable):
    var accepted: Bool
    var failure_kind: Int
    # Both metrics are -1 when preflight returns before allocating a state.
    var used_nodes: Int
    var fuel: Int


@fieldwise_init
struct RecordExecutionLayout(ImplicitlyCopyable):
    var bridge: score_profile.NatScoreBridgeLayout
    var laws_first: Int
    var trace_first: Int
    var prefix_first: Int
    var basics_first: Int
    var fold_first: Int
    var refinement_first: Int
    var initialization: Int
    var canonical: Int


def _layout_valid(layout: RecordExecutionLayout, count: Int) -> Bool:
    if not score_profile._layout_valid(layout.bridge, count):
        return False
    var next = 148
    if layout.laws_first != next:
        return False
    next += 12
    if layout.trace_first != next:
        return False
    next += 6
    if layout.prefix_first != next:
        return False
    next += 2
    if layout.basics_first != next:
        return False
    next += 5
    if layout.fold_first != next:
        return False
    next += 3
    if layout.refinement_first != next:
        return False
    next += 5
    if layout.initialization != next:
        return False
    next += 1
    if layout.canonical != next:
        return False
    return next + 1 == NAT_RECORD_EXECUTION_DEFINITIONS


def _call[C: Int](mut state: tk._TypedState[C], definition: Int, args: List[Int]) -> Int:
    return execution._apps(state, _named(state, definition), args)


def _number[C: Int](mut state: tk._TypedState[C], value: Int) -> Int:
    var result = _zero(state)
    for _ in range(value):
        result = _succ(state, result)
    return result


def _record_type[C: Int](mut state: tk._TypedState[C]) -> Int:
    return _store_type(state)


def _state_type[C: Int](mut state: tk._TypedState[C]) -> Int:
    var component = _var(state, _nat(state))
    return _pi(state, component, _store_type(state))


def _field[C: Int](mut state: tk._TypedState[C], record: Int, position: Int) -> Int:
    return _app(state, record, _number(state, position))


def _row[C: Int](mut state: tk._TypedState[C], records: Int, slot: Int) -> Int:
    var component = _var(state, _nat(state))
    return _lam(state, component, execution._apps(state, records, [component, slot]))


def _record[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    rank: Int, token: Int, bad: Int) -> Int:
    return _call(state, layout.trace_first, [rank, token, bad])


def _merged[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    left: Int, right: Int) -> Int:
    return _call(state, layout.trace_first + 3, [left, right])


def _encoded[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, record: Int) -> Int:
    return score_profile._encoded(state, layout.bridge, bound,
        _field(state, record, 0), _field(state, record, 1), _field(state, record, 2))


def _valid[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, record: Int) -> Int:
    return _call(state, layout.basics_first, [bound, record])


def _valid_parts[C: Int](mut state: tk._TypedState[C], bound: Int, record: Int) -> List[Int]:
    return [score_profile._le(state, _field(state, record, 1), bound),
        score_profile._bit(state, _field(state, record, 2))]


def _component_merge[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    left: List[Int], right: List[Int]) -> List[Int]:
    var args: List[Int] = [left[0], right[0], left[1], right[1]]
    return [_call(state, layout.laws_first, args),
        _call(state, layout.laws_first + 1, args), score_profile._or(state, left[2], right[2])]


def _component_equal[C: Int](mut state: tk._TypedState[C],
    left: List[Int], right: List[Int]) -> Int:
    return execution._and(state, [_eq(state, left[0], right[0]),
        _eq(state, left[1], right[1]), _eq(state, left[2], right[2])])


def _record_equal[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _component_equal(state,
        [_field(state, left, 0), _field(state, left, 1), _field(state, left, 2)],
        [_field(state, right, 0), _field(state, right, 1), _field(state, right, 2)])


def _record_law_data[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Bool:
    # Both components are selected by exactly the same actual flag139.
    for offset in range(2):
        var ra = _var(state, _nat(state))
        var rb = _var(state, _nat(state))
        var ta = _var(state, _nat(state))
        var tb = _var(state, _nat(state))
        var flag = score_profile._winner(state, layout.bridge, ra, rb, ta, tb)
        var body = score_profile._select(state, flag, ra if offset == 0 else ta,
            rb if offset == 0 else tb)
        var binders: List[Int] = [ra, rb, ta, tb]
        if not phase._data_matches(state, layout.laws_first + offset,
            _pis(state, binders, _nat(state)), _lams(state, binders, body)):
            return False
    return True


def _trace_data[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Bool:
    var rank = _var(state, _nat(state))
    var token = _var(state, _nat(state))
    var bad = _var(state, _nat(state))
    var component = _var(state, _nat(state))
    var predecessor = _call(state, 2, [component])
    var body = score_profile._select(state, component, rank,
        score_profile._select(state, predecessor, token, bad))
    if not phase._data_matches(state, layout.trace_first,
        _pis(state, [rank, token, bad], _record_type(state)),
        _lams(state, [rank, token, bad, component], body)):
        return False

    var ranks = _var(state, _store_type(state))
    var tokens = _var(state, _store_type(state))
    var bads = _var(state, _store_type(state))
    component = _var(state, _nat(state))
    var slot = _var(state, _nat(state))
    var record = _record(state, layout, _app(state, ranks, slot),
        _app(state, tokens, slot), _app(state, bads, slot))
    if not phase._data_matches(state, layout.trace_first + 1,
        _pis(state, [ranks, tokens, bads], _state_type(state)),
        _lams(state, [ranks, tokens, bads, component, slot], _app(state, record, component))):
        return False

    var bound = _var(state, _nat(state))
    var count = _var(state, _nat(state))
    ranks = _var(state, _store_type(state))
    tokens = _var(state, _store_type(state))
    bads = _var(state, _store_type(state))
    component = _var(state, _nat(state))
    var initial_inputs = _call(state, layout.trace_first + 1, [ranks, tokens, bads])
    var empty = _record(state, layout, _zero(state), bound, _zero(state))
    body = _call(state, 5, [_app(state, initial_inputs, component), count,
        _app(state, empty, component)])
    var binders: List[Int] = [bound, count, ranks, tokens, bads]
    var initial_binders = binders.copy()
    initial_binders.append(component)
    if not phase._data_matches(state, layout.trace_first + 2,
        _pis(state, binders, _state_type(state)), _lams(state, initial_binders, body)):
        return False

    var left = _var(state, _record_type(state))
    var right = _var(state, _record_type(state))
    var merged = _component_merge(state, layout,
        [_field(state, left, 0), _field(state, left, 1), _field(state, left, 2)],
        [_field(state, right, 0), _field(state, right, 1), _field(state, right, 2)])
    body = _record(state, layout, merged[0], merged[1], merged[2])
    if not phase._data_matches(state, layout.trace_first + 3,
        _pis(state, [left, right], _record_type(state)), _lams(state, [left, right], body)):
        return False

    var initial = _var(state, _state_type(state))
    var destinations = _var(state, _store_type(state))
    var lefts = _var(state, _store_type(state))
    var rights = _var(state, _store_type(state))
    var steps = _var(state, _nat(state))
    var k = _var(state, _nat(state))
    var old = _var(state, _state_type(state))
    component = _var(state, _nat(state))
    var output = _merged(state, layout, _row(state, old, _app(state, lefts, k)),
        _row(state, old, _app(state, rights, k)))
    var updated = _call(state, 5, [_app(state, old, component),
        _app(state, destinations, k), _app(state, output, component)])
    var step = _lams(state, [k, old, component], updated)
    binders = [initial, destinations, lefts, rights, steps]
    if not phase._data_matches(state, layout.trace_first + 4,
        _pis(state, binders, _state_type(state)),
        _lams(state, binders, _rec(state, steps, initial, step))):
        return False

    bound = _var(state, _nat(state))
    ranks = _var(state, _store_type(state))
    tokens = _var(state, _store_type(state))
    bads = _var(state, _store_type(state))
    count = _var(state, _nat(state))
    k = _var(state, _nat(state))
    var previous = _var(state, _record_type(state))
    var current = _record(state, layout, _app(state, ranks, k),
        _app(state, tokens, k), _app(state, bads, k))
    step = _lams(state, [k, previous], _merged(state, layout, previous, current))
    empty = _record(state, layout, _zero(state), bound, _zero(state))
    binders = [bound, ranks, tokens, bads, count]
    return phase._data_matches(state, layout.trace_first + 5,
        _pis(state, binders, _record_type(state)),
        _lams(state, binders, _rec(state, count, empty, step)))


def _predicate_data[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Bool:
    var bound = _var(state, _nat(state))
    var record = _var(state, _record_type(state))
    if not phase._data_matches(state, layout.basics_first,
        _pis(state, [bound, record], _prop(state)),
        _lams(state, [bound, record], execution._and(state, _valid_parts(state, bound, record)))):
        return False
    bound = _var(state, _nat(state))
    var ranks = _var(state, _store_type(state))
    var tokens = _var(state, _store_type(state))
    var bads = _var(state, _store_type(state))
    var count = _var(state, _nat(state))
    var slot = _var(state, _nat(state))
    var before = _var(state, execution._before(state, slot, count))
    record = _record(state, layout, _app(state, ranks, slot),
        _app(state, tokens, slot), _app(state, bads, slot))
    var body = _pis(state, [slot, before], _valid(state, layout, bound, record))
    var binders: List[Int] = [bound, ranks, tokens, bads, count]
    if not phase._data_matches(state, layout.fold_first,
        _pis(state, binders, _prop(state)), _lams(state, binders, body)):
        return False
    bound = _var(state, _nat(state))
    ranks = _var(state, _store_type(state))
    tokens = _var(state, _store_type(state))
    bads = _var(state, _store_type(state))
    slot = _var(state, _nat(state))
    body = score_profile._encoded(state, layout.bridge, bound,
        _app(state, ranks, slot), _app(state, tokens, slot), _app(state, bads, slot))
    return phase._data_matches(state, layout.fold_first + 1,
        _pis(state, [bound, ranks, tokens, bads], _store_type(state)),
        _lams(state, [bound, ranks, tokens, bads, slot], body))


def _ready_valid[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, records: Int, ready: Int) -> Int:
    return _call(state, layout.refinement_first, [bound, records, ready])


def _encoding_equal[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, records: Int, values: Int) -> Int:
    return _call(state, layout.refinement_first + 1, [bound, records, values])


def _invariant[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, records: Int, ready: Int, values: Int) -> Int:
    return _call(state, layout.refinement_first + 2, [bound, records, ready, values])


def _invariant_data[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Bool:
    var bound = _var(state, _nat(state))
    var records = _var(state, _state_type(state))
    var ready = _var(state, _store_type(state))
    var slot = _var(state, _nat(state))
    var present = _var(state, _eq(state, _app(state, ready, slot), score_profile._one(state)))
    var body = _pis(state, [slot, present], _valid(state, layout, bound, _row(state, records, slot)))
    if not phase._data_matches(state, layout.refinement_first,
        _pis(state, [bound, records, ready], _prop(state)),
        _lams(state, [bound, records, ready], body)):
        return False
    bound = _var(state, _nat(state))
    records = _var(state, _state_type(state))
    var values = _var(state, _store_type(state))
    slot = _var(state, _nat(state))
    body = _pi(state, slot, _eq(state, _encoded(state, layout, bound, _row(state, records, slot)),
        _app(state, values, slot)))
    if not phase._data_matches(state, layout.refinement_first + 1,
        _pis(state, [bound, records, values], _prop(state)),
        _lams(state, [bound, records, values], body)):
        return False
    bound = _var(state, _nat(state))
    records = _var(state, _state_type(state))
    ready = _var(state, _store_type(state))
    values = _var(state, _store_type(state))
    body = execution._and(state, [_ready_valid(state, layout, bound, records, ready),
        _encoding_equal(state, layout, bound, records, values)])
    var binders: List[Int] = [bound, records, ready, values]
    return phase._data_matches(state, layout.refinement_first + 2,
        _pis(state, binders, _prop(state)), _lams(state, binders, body))


def _aci_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    offset: Int) -> Int:
    var bound = _var(state, _nat(state))
    var arity = 1
    if offset == 7:
        arity = 2
    elif offset == 8:
        arity = 3
    var inputs = List[List[Int]]()
    var binders: List[Int] = [bound]
    for _ in range(arity):
        var record: List[Int] = [_var(state, _nat(state)),
            _var(state, _nat(state)), _var(state, _nat(state))]
        for value in record:
            binders.append(value)
        inputs.append(record^)
    for index in range(arity):
        binders.append(_var(state, score_profile._le(state, inputs[index][1], bound)))
        binders.append(_var(state, score_profile._bit(state, inputs[index][2])))
    var left: List[Int]
    var right: List[Int]
    if offset == 7:
        left = _component_merge(state, layout, inputs[0], inputs[1])
        right = _component_merge(state, layout, inputs[1], inputs[0])
    elif offset == 8:
        var ab = _component_merge(state, layout, inputs[0], inputs[1])
        var bc = _component_merge(state, layout, inputs[1], inputs[2])
        left = _component_merge(state, layout, ab, inputs[2])
        right = _component_merge(state, layout, inputs[0], bc)
    elif offset == 9:
        left = _component_merge(state, layout, inputs[0], inputs[0])
        right = inputs[0].copy()
    else:
        var empty: List[Int] = [_zero(state), bound, _zero(state)]
        left = (_component_merge(state, layout, empty, inputs[0]) if offset == 10
            else _component_merge(state, layout, inputs[0], empty))
        right = inputs[0].copy()
    return _pis(state, binders, _component_equal(state, left, right))


def _record_law_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    offset: Int) -> Int:
    if offset >= 7:
        return _aci_type(state, layout, offset)
    if offset == 3:
        var left = _var(state, _nat(state))
        var right = _var(state, _nat(state))
        var left_bit = _var(state, score_profile._bit(state, left))
        var right_bit = _var(state, score_profile._bit(state, right))
        return _pis(state, [left, right, left_bit, right_bit],
            score_profile._bit(state, score_profile._or(state, left, right)))
    if offset == 6:
        var value = _var(state, _nat(state))
        return _pi(state, value, score_profile._le(state, value, value))
    var bound = _var(state, _nat(state))
    var ra = _var(state, _nat(state))
    var rb = _var(state, _nat(state))
    var ta = _var(state, _nat(state))
    var tb = _var(state, _nat(state))
    if offset == 2:
        var bounded_a = _var(state, score_profile._le(state, ta, bound))
        var bounded_b = _var(state, score_profile._le(state, tb, bound))
        return _pis(state, [bound, ra, rb, ta, tb, bounded_a, bounded_b],
            score_profile._le(state, _call(state, layout.laws_first + 1, [ra, rb, ta, tb]), bound))
    var ba = _var(state, _nat(state))
    var bb = _var(state, _nat(state))
    var merged = _component_merge(state, layout, [ra, ta, ba], [rb, tb, bb])
    var encoded = score_profile._encoded(state, layout.bridge, bound, merged[0], merged[1], merged[2])
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb]
    var result: Int
    if offset == 4:
        result = _eq(state, encoded, _call(state, layout.bridge.encoding_first + 1,
            [bound, ra, rb, ta, tb, ba, bb]))
    else:
        binders.append(_var(state, score_profile._le(state, ta, bound)))
        binders.append(_var(state, score_profile._le(state, tb, bound)))
        binders.append(_var(state, score_profile._bit(state, ba)))
        binders.append(_var(state, score_profile._bit(state, bb)))
        result = _eq(state, encoded, _call(state, 86,
            [score_profile._encoded(state, layout.bridge, bound, ra, ta, ba),
                score_profile._encoded(state, layout.bridge, bound, rb, tb, bb)]))
    return _pis(state, binders, result)


def _prefix_type[C: Int](mut state: tk._TypedState[C], offset: Int) -> Int:
    var source = _var(state, _nat(state))
    if offset == 1:
        return _pi(state, source, execution._before(state, source, _succ(state, source)))
    var count = _var(state, _nat(state))
    var before = _var(state, execution._before(state, source, count))
    return _pis(state, [source, count, before], execution._before(state, source, _succ(state, count)))


def _basics_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    offset: Int) -> Int:
    var bound = _var(state, _nat(state))
    var left = _var(state, _record_type(state))
    var right = _var(state, _record_type(state))
    var binders: List[Int] = [bound, left, right]
    if offset == 1:
        for position in range(3):
            binders.append(_var(state, _eq(state, _field(state, left, position),
                _field(state, right, position))))
        return _pis(state, binders, _eq(state,
            _encoded(state, layout, bound, left), _encoded(state, layout, bound, right)))
    binders.append(_var(state, _valid(state, layout, bound, left)))
    if offset == 2:
        for position in range(1, 3):
            binders.append(_var(state, _eq(state, _field(state, left, position),
                _field(state, right, position))))
        return _pis(state, binders, _valid(state, layout, bound, right))
    binders.append(_var(state, _valid(state, layout, bound, right)))
    var merged = _merged(state, layout, left, right)
    var result: Int
    if offset == 3:
        result = _valid(state, layout, bound, merged)
    else:
        result = _eq(state, _encoded(state, layout, bound, merged),
            _call(state, 86, [_encoded(state, layout, bound, left), _encoded(state, layout, bound, right)]))
    return _pis(state, binders, result)


def _encoded_inputs[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, ranks: Int, tokens: Int, bads: Int) -> Int:
    return _call(state, layout.fold_first + 1, [bound, ranks, tokens, bads])


def _folded[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, ranks: Int, tokens: Int, bads: Int, count: Int) -> Int:
    return _call(state, layout.trace_first + 5, [bound, ranks, tokens, bads, count])


def _flat[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    bound: Int, ranks: Int, tokens: Int, bads: Int, count: Int) -> Int:
    return execution._flat(state, _named(state, 86), _zero(state),
        _encoded_inputs(state, layout, bound, ranks, tokens, bads), count)


def _fold_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Int:
    var bound = _var(state, _nat(state))
    var ranks = _var(state, _store_type(state))
    var tokens = _var(state, _store_type(state))
    var bads = _var(state, _store_type(state))
    var count = _var(state, _nat(state))
    var valid_inputs = _var(state, _call(state, layout.fold_first, [bound, ranks, tokens, bads, count]))
    var record = _folded(state, layout, bound, ranks, tokens, bads, count)
    var result = execution._and(state, [_valid(state, layout, bound, record),
        _eq(state, _encoded(state, layout, bound, record),
            _flat(state, layout, bound, ranks, tokens, bads, count))])
    return _pis(state, [bound, ranks, tokens, bads, count, valid_inputs], result)


def _step_state[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    records: Int, destination: Int, left: Int, right: Int) -> Int:
    var merged = _merged(state, layout, _row(state, records, left), _row(state, records, right))
    var component = _var(state, _nat(state))
    return _lam(state, component, _call(state, 5,
        [_app(state, records, component), destination, _app(state, merged, component)]))


def _step_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Int:
    var bound = _var(state, _nat(state))
    var records = _var(state, _state_type(state))
    var ready = _var(state, _store_type(state))
    var values = _var(state, _store_type(state))
    var destination = _var(state, _nat(state))
    var left = _var(state, _nat(state))
    var right = _var(state, _nat(state))
    var invariant = _var(state, _invariant(state, layout, bound, records, ready, values))
    var left_ready = _var(state, _eq(state, _app(state, ready, left), score_profile._one(state)))
    var right_ready = _var(state, _eq(state, _app(state, ready, right), score_profile._one(state)))
    var result = _invariant(state, layout, bound,
        _step_state(state, layout, records, destination, left, right),
        _call(state, 5, [ready, destination, score_profile._one(state)]),
        _call(state, 5, [values, destination,
            _call(state, 86, [_app(state, values, left), _app(state, values, right)])]))
    return _pis(state, [bound, records, ready, values, destination, left, right,
        invariant, left_ready, right_ready], result)


def _run[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout,
    initial: Int, destinations: Int, lefts: Int, rights: Int, steps: Int) -> Int:
    return _call(state, layout.trace_first + 4, [initial, destinations, lefts, rights, steps])


def _prefix_refinement_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Int:
    var bound = _var(state, _nat(state))
    var initial = _var(state, _state_type(state))
    var ready = _var(state, _store_type(state))
    var values = _var(state, _store_type(state))
    var support = _var(state, execution._support_type(state))
    var destinations = _var(state, _store_type(state))
    var lefts = _var(state, _store_type(state))
    var rights = _var(state, _store_type(state))
    var steps = _var(state, _nat(state))
    var invariant = _var(state, _invariant(state, layout, bound, initial, ready, values))
    var structural = _var(state, execution._support_prefix(state,
        ready, support, destinations, lefts, rights, steps))
    var result = _invariant(state, layout, bound,
        _run(state, layout, initial, destinations, lefts, rights, steps),
        execution._ready(state, ready, destinations, steps),
        execution._values(state, _named(state, 86), values, destinations, lefts, rights, steps))
    return _pis(state, [bound, initial, ready, values, support, destinations,
        lefts, rights, steps, invariant, structural], result)


def _initial_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Int:
    var bound = _var(state, _nat(state))
    var ranks = _var(state, _store_type(state))
    var tokens = _var(state, _store_type(state))
    var bads = _var(state, _store_type(state))
    var count = _var(state, _nat(state))
    var valid_inputs = _var(state, _call(state, layout.fold_first, [bound, ranks, tokens, bads, count]))
    var result = _invariant(state, layout, bound,
        _call(state, layout.trace_first + 2, [bound, count, ranks, tokens, bads]),
        execution._initial_ready(state, count),
        execution._initial_values(state, _zero(state),
            _encoded_inputs(state, layout, bound, ranks, tokens, bads), count))
    return _pis(state, [bound, ranks, tokens, bads, count, valid_inputs], result)


def _canonical_type[C: Int](mut state: tk._TypedState[C], layout: RecordExecutionLayout) -> Int:
    var bound = _var(state, _nat(state))
    var ranks = _var(state, _store_type(state))
    var tokens = _var(state, _store_type(state))
    var bads = _var(state, _store_type(state))
    var count = _var(state, _nat(state))
    var support = _var(state, execution._support_type(state))
    var destinations = _var(state, _store_type(state))
    var lefts = _var(state, _store_type(state))
    var rights = _var(state, _store_type(state))
    var valid_inputs = _var(state, _call(state, layout.fold_first, [bound, ranks, tokens, bads, count]))
    var observation = _var(state, _nat(state))
    var empty_support = _var(state, _pi(state, observation, _eq(state,
        _app(state, _app(state, support, count), observation), _zero(state))))
    var source = _var(state, _nat(state))
    var before = _var(state, execution._before(state, source, count))
    observation = _var(state, _nat(state))
    var input_support = _var(state, _pis(state, [source, before, observation], _eq(state,
        _app(state, _app(state, support, source), observation),
        _app(state, execution._unit(state, source), observation))))
    var steps = _var(state, _nat(state))
    var initial_ready = execution._initial_ready(state, count)
    var structural = _var(state, execution._support_prefix(state,
        initial_ready, support, destinations, lefts, rights, steps))
    var root = _var(state, _nat(state))
    var ready = _var(state, _eq(state,
        _app(state, execution._ready(state, initial_ready, destinations, steps), root), score_profile._one(state)))
    var complete = _var(state, execution._covered(state, _app(state, support, root), count))
    var initial = _call(state, layout.trace_first + 2, [bound, count, ranks, tokens, bads])
    var actual = _row(state, _run(state, layout, initial, destinations, lefts, rights, steps), root)
    var canonical = _folded(state, layout, bound, ranks, tokens, bads, count)
    var result = _record_equal(state, actual, canonical)
    return _pis(state, [bound, ranks, tokens, bads, count, support, destinations, lefts, rights,
        valid_inputs, empty_support, input_support, steps, structural, root, ready, complete], result)


def _new_specification[C: Int](mut state: tk._TypedState[C],
    layout: RecordExecutionLayout, root: Int) -> Bool:
    if (not _record_law_data(state, layout) or not _trace_data(state, layout)
        or not _predicate_data(state, layout) or not _invariant_data(state, layout)):
        return False
    for offset in range(2, 12):
        if not joined._closed(state, _named(state, layout.laws_first + offset),
            _record_law_type(state, layout, offset)):
            return False
    for offset in range(2):
        if not joined._closed(state, _named(state, layout.prefix_first + offset), _prefix_type(state, offset)):
            return False
    for offset in range(1, 5):
        if not joined._closed(state, _named(state, layout.basics_first + offset),
            _basics_type(state, layout, offset)):
            return False
    if not joined._closed(state, _named(state, layout.fold_first + 2), _fold_type(state, layout)):
        return False
    if not joined._closed(state, _named(state, layout.refinement_first + 3), _step_type(state, layout)):
        return False
    if not joined._closed(state, _named(state, layout.refinement_first + 4),
        _prefix_refinement_type(state, layout)):
        return False
    if not joined._closed(state, _named(state, layout.initialization), _initial_type(state, layout)):
        return False
    var expected = _canonical_type(state, layout)
    return (joined._closed(state, _named(state, layout.canonical), expected)
        and joined._closed(state, root, expected))



def _check_in_state[C: Int, S: Int](mut state: tk._TypedState[S],
    bundle: tk.ProofTermBundle[C], layout: RecordExecutionLayout) -> Bool:
    state.fuel = S * 2048
    tk._enable_native_numerals(state)
    if not tk._check_inductive_environment(state):
        return False
    for index in range(bundle.count):
        var actual = tk._infer(state, index)
        if actual == -1 or actual == -4:
            return False
    # Root147 is explicitly checked at its old mathematical binary-merge
    # theorem type; the supplied root is instead checked at the new trace type.
    if (not score_profile._qualified_prefix(state)
        or not score_profile._new_specification(state, layout.bridge,
            _named(state, layout.bridge.encoding_first + 5))
        or not _new_specification(state, layout, bundle.root)):
        return False
    return (state.fuel > 0 and state.syntax.count < S
        and not state.arithmetic_verified and not state.fast_arithmetic)


def _preflight_kind[C: Int, scratch: Int](
    bundle: tk.ProofTermBundle[C], layout: RecordExecutionLayout,
) -> Int:
    # Quotas take priority even when the same input has other invalid metadata.
    # No state is allocated and no syntax entry is read on this branch.
    if (scratch <= 0 or bundle.count > scratch or bundle.definition_count > scratch
        or bundle.nat_constructor_count > scratch or bundle.constructor_count > scratch):
        return RECORD_CHECK_RESOURCE_PREFLIGHT
    if (bundle.count <= 0 or bundle.count > C
        or bundle.definition_count != NAT_RECORD_EXECUTION_DEFINITIONS
        or bundle.definition_count > C
        or bundle.nat_constructor_count < 0 or bundle.nat_constructor_count > C
        or bundle.constructor_count < 0 or bundle.constructor_count > C
        or not _layout_valid(layout, bundle.count) or not phase._phase_syntax(bundle)
        or bundle.root < 0 or bundle.root >= bundle.count):
        return RECORD_CHECK_INVALID_PROOF
    return RECORD_CHECK_ACCEPTED


def check_nat_record_execution_result[C: Int, scratch: Int = 65536](
    bundle: tk.ProofTermBundle[C], layout: RecordExecutionLayout,
) -> RecordExecutionCheckResult:
    """Check once and report proof rejection separately from resource failure.

    The fixed default workspace is separate from computation shape. Every
    metadata, quota, layout and strict syntax check precedes allocation. After
    allocation the canonical constructor and unchanged admission logic run once.
    A false result at either resource bound is conservatively exhaustion.
    """
    var preflight = _preflight_kind[C, scratch](bundle, layout)
    if preflight != RECORD_CHECK_ACCEPTED:
        return RecordExecutionCheckResult(False, preflight, -1, -1)
    var states = score_profile._state_holder[C, scratch](bundle)
    var accepted = _check_in_state(states[0], bundle, layout)
    var kind = RECORD_CHECK_ACCEPTED
    if not accepted:
        if states[0].fuel <= 0 or states[0].syntax.count >= scratch:
            kind = RECORD_CHECK_RESOURCE_EXHAUSTED
        else:
            kind = RECORD_CHECK_INVALID_PROOF
    return RecordExecutionCheckResult(accepted, kind, states[0].syntax.count, states[0].fuel)


def check_nat_record_execution_proof[C: Int, scratch: Int = 65536](
    bundle: tk.ProofTermBundle[C], layout: RecordExecutionLayout,
) -> Bool:
    """Preserve the Boolean judgment; resource failure also returns false.

    Definitions and exports are closed. Unused raw binder nodes may be open
    and must still infer. Use the result API when reporting semantic rejection
    controls, so a resource failure cannot be mistaken for an invalid proof.
    """
    return check_nat_record_execution_result[C, scratch](bundle, layout).accepted
