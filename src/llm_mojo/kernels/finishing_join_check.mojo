"""Independent admission of the total Nat join and its finishing theorem.

Expected raw bodies and public theorem types are constructed in TypedState,
without importing any record producer helper. The final statement specializes
the old canonical trace theorem to the checked total Nat join and Empty zero;
no join-law premise remains. Native rank/token/word interpretation is separate.
Resource failure is a failed positive gate, never a semantic rejection result.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase_check as phase
import llm_mojo.kernels.finishing_execution_check as old
from llm_mojo.kernels.finishing_phase_check import (
    _nat, _prop, _var, _pi, _lam, _named, _app, _zero, _succ, _eq,
    _rec, _store_type, _pis, _lams,
)


def _call[C: Int](mut state: tk._TypedState[C], definition: Int,
    arguments: List[Int]) -> Int:
    return old._apps(state, _named(state, definition), arguments)


def _one[C: Int](mut state: tk._TypedState[C]) -> Int:
    return _succ(state, _zero(state))


def _select[C: Int](mut state: tk._TypedState[C], flag: Int,
    yes: Int, no: Int) -> Int:
    return _call(state, 3, [flag, yes, no])


def _max[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 55, [left, right])


def _half[C: Int](mut state: tk._TypedState[C], value: Int) -> Int:
    return _call(state, 64, [value])


def _parity[C: Int](mut state: tk._TypedState[C], value: Int) -> Int:
    return _call(state, 65, [value])


def _is_bit[C: Int](mut state: tk._TypedState[C], value: Int) -> Int:
    return _call(state, 66, [value])


def _pack[C: Int](mut state: tk._TypedState[C], score: Int, flag: Int) -> Int:
    return _call(state, 67, [score, flag])


def _or[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 68, [left, right])


def _join[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 86, [left, right])


def _matches[C: Int](mut state: tk._TypedState[C], definition: Int,
    expected_type: Int, expected_body: Int) -> Bool:
    return phase._data_matches(state, definition, expected_type, expected_body)


def _maximum_data[C: Int](mut state: tk._TypedState[C]) -> Bool:
    var left = _var(state, _nat(state))
    var right = _var(state, _nat(state))
    var zero_case = _lam(state, right, right)
    var predecessor = _var(state, _nat(state))
    var smaller_max = _var(state, _store_type(state))
    right = _var(state, _nat(state))
    var right_predecessor = _var(state, _nat(state))
    var ignored = _var(state, _nat(state))
    var both_successors = _lams(state, [right_predecessor, ignored],
        _succ(state, _app(state, smaller_max, right_predecessor)))
    var successor_case = _lams(state, [predecessor, smaller_max, right],
        _rec(state, right, _succ(state, predecessor), both_successors))
    return _matches(state, 55, _pi(state, left, _store_type(state)),
        _lam(state, left, _rec(state, left, zero_case, successor_case)))


def _parity_data[C: Int](mut state: tk._TypedState[C]) -> Bool:
    var value = _var(state, _nat(state))
    var k = _var(state, _nat(state))
    var accumulated = _var(state, _nat(state))
    var step = _lams(state, [k, accumulated], _succ(state, _succ(state, accumulated)))
    if not _matches(state, 61, _pi(state, value, _nat(state)),
        _lam(state, value, _rec(state, value, _zero(state), step))):
        return False

    var score = _var(state, _nat(state))
    var flag = _var(state, _nat(state))
    var position = _var(state, _nat(state))
    if not _matches(state, 62, _pis(state, [score, flag], _store_type(state)),
        _lams(state, [score, flag, position], _select(state, position, score, flag))):
        return False

    value = _var(state, _nat(state))
    k = _var(state, _nat(state))
    var previous = _var(state, _store_type(state))
    var old_score = _app(state, previous, _zero(state))
    var old_flag = _app(state, previous, _one(state))
    var next_score = _select(state, old_flag, old_score, _succ(state, old_score))
    var next_flag = _select(state, old_flag, _one(state), _zero(state))
    step = _lams(state, [k, previous], _call(state, 62, [next_score, next_flag]))
    var initial_pair = _call(state, 62, [_zero(state), _zero(state)])
    if not _matches(state, 63, _pi(state, value, _store_type(state)),
        _lam(state, value, _rec(state, value, initial_pair, step))):
        return False

    value = _var(state, _nat(state))
    if not _matches(state, 64, _pi(state, value, _nat(state)),
        _lam(state, value, _app(state, _call(state, 63, [value]), _zero(state)))):
        return False
    value = _var(state, _nat(state))
    if not _matches(state, 65, _pi(state, value, _nat(state)),
        _lam(state, value, _app(state, _call(state, 63, [value]), _one(state)))):
        return False

    flag = _var(state, _nat(state))
    var proposition = _var(state, _prop(state))
    var flag_zero = _var(state, _eq(state, flag, _zero(state)))
    var zero_case = _var(state, _pi(state, flag_zero, proposition))
    var flag_one = _var(state, _eq(state, flag, _one(state)))
    var one_case = _var(state, _pi(state, flag_one, proposition))
    var body = _pis(state, [proposition, zero_case, one_case], proposition)
    if not _matches(state, 66, _pi(state, flag, _prop(state)), _lam(state, flag, body)):
        return False

    score = _var(state, _nat(state))
    flag = _var(state, _nat(state))
    var doubled = _call(state, 61, [score])
    if not _matches(state, 67, _pis(state, [score, flag], _nat(state)),
        _lams(state, [score, flag], _select(state, flag, doubled, _succ(state, doubled)))):
        return False
    var left = _var(state, _nat(state))
    var right = _var(state, _nat(state))
    return _matches(state, 68, _pis(state, [left, right], _nat(state)),
        _lams(state, [left, right], _select(state, left, right, _one(state))))


def _join_data[C: Int](mut state: tk._TypedState[C]) -> Bool:
    var left = _var(state, _nat(state))
    var right = _var(state, _nat(state))
    var score = _max(state, _half(state, left), _half(state, right))
    var flag = _or(state, _parity(state, left), _parity(state, right))
    return _matches(state, 86, _pis(state, [left, right], _nat(state)),
        _lams(state, [left, right], _pack(state, score, flag)))


def _maximum_law_type[C: Int](mut state: tk._TypedState[C], law: Int) -> Int:
    var left = _var(state, _nat(state))
    if law == 56:
        return _pi(state, left, _eq(state, _max(state, left, _zero(state)), left))
    if law == 57:
        return _pi(state, left, _eq(state, _max(state, _zero(state), left), left))
    if law == 60:
        return _pi(state, left, _eq(state, _max(state, left, left), left))
    var right = _var(state, _nat(state))
    if law == 58:
        return _pis(state, [left, right], _eq(state,
            _max(state, left, right), _max(state, right, left)))
    var last = _var(state, _nat(state))
    return _pis(state, [left, right, last], _eq(state,
        _max(state, _max(state, left, right), last),
        _max(state, left, _max(state, right, last))))


def _parity_law_type[C: Int](mut state: tk._TypedState[C], law: Int) -> Int:
    if law == 70 or law == 71:
        var value = _var(state, _nat(state))
        var result = _is_bit(state, _parity(state, value))
        if law == 71:
            result = _eq(state, _pack(state, _half(state, value), _parity(state, value)), value)
        return _pi(state, value, result)
    var score = _var(state, _nat(state))
    var flag = _var(state, _nat(state))
    var bit = _var(state, _is_bit(state, flag))
    var packed = _pack(state, score, flag)
    var result = _eq(state, _half(state, packed), score)
    if law == 78:
        result = _eq(state, _parity(state, packed), flag)
    return _pis(state, [score, flag, bit], result)


def _boolean_law_type[C: Int](mut state: tk._TypedState[C], law: Int) -> Int:
    var arity = 1
    if law == 80 or law == 81:
        arity = 2
    elif law == 82:
        arity = 3
    var values = List[Int]()
    var binders = List[Int]()
    for _ in range(arity):
        var value = _var(state, _nat(state))
        values.append(value)
        binders.append(value)
    for value in values:
        binders.append(_var(state, _is_bit(state, value)))
    var left: Int
    var right: Int
    if law == 80:
        return _pis(state, binders, _is_bit(state, _or(state, values[0], values[1])))
    if law == 81:
        left = _or(state, values[0], values[1])
        right = _or(state, values[1], values[0])
    elif law == 82:
        left = _or(state, _or(state, values[0], values[1]), values[2])
        right = _or(state, values[0], _or(state, values[1], values[2]))
    elif law == 83:
        left = _or(state, _zero(state), values[0])
        right = values[0]
    elif law == 84:
        left = _or(state, values[0], _zero(state))
        right = values[0]
    else:
        left = _or(state, values[0], values[0])
        right = values[0]
    return _pis(state, binders, _eq(state, left, right))


def _join_law_type[C: Int](mut state: tk._TypedState[C], law: Int) -> Int:
    var left = _var(state, _nat(state))
    if law == 89:
        return _pi(state, left, _eq(state, _join(state, _zero(state), left), left))
    if law == 90:
        return _pi(state, left, _eq(state, _join(state, left, _zero(state)), left))
    if law == 93:
        return _pi(state, left, _eq(state, _join(state, left, left), left))
    var right = _var(state, _nat(state))
    if law == 87:
        return _pis(state, [left, right], _eq(state,
            _half(state, _join(state, left, right)),
            _max(state, _half(state, left), _half(state, right))))
    if law == 88:
        return _pis(state, [left, right], _eq(state,
            _parity(state, _join(state, left, right)),
            _or(state, _parity(state, left), _parity(state, right))))
    if law == 91:
        return _pis(state, [left, right], _eq(state,
            _join(state, left, right), _join(state, right, left)))
    var last = _var(state, _nat(state))
    return _pis(state, [left, right, last], _eq(state,
        _join(state, _join(state, left, right), last),
        _join(state, left, _join(state, right, last))))


def _canonical_record_type[C: Int](mut state: tk._TypedState[C]) -> Int:
    """Exact old result specialized to the concrete Join and Empty zero."""
    var inputs = _var(state, _store_type(state))
    var count = _var(state, _nat(state))
    var support = _var(state, old._support_type(state))
    var destinations = _var(state, _store_type(state))
    var lefts = _var(state, _store_type(state))
    var rights = _var(state, _store_type(state))
    var observation = _var(state, _nat(state))
    var empty_support = _var(state, _pi(state, observation, _eq(state,
        _app(state, _app(state, support, count), observation), _zero(state))))
    var source = _var(state, _nat(state))
    var before = _var(state, old._before(state, source, count))
    observation = _var(state, _nat(state))
    var input_support = _var(state, _pis(state, [source, before, observation], _eq(state,
        _app(state, _app(state, support, source), observation),
        _app(state, old._unit(state, source), observation))))
    var steps = _var(state, _nat(state))
    var initial_ready = old._initial_ready(state, count)
    var structural = _var(state, old._support_prefix(state,
        initial_ready, support, destinations, lefts, rights, steps))
    var root = _var(state, _nat(state))
    var ready = _var(state, _eq(state,
        _app(state, old._ready(state, initial_ready, destinations, steps), root), _one(state)))
    var complete = _var(state, old._covered(state, _app(state, support, root), count))
    var join = _named(state, 86)
    var initial_values = old._initial_values(state, _zero(state), inputs, count)
    var actual = _app(state, old._values(state,
        join, initial_values, destinations, lefts, rights, steps), root)
    var expected = old._flat(state, join, _zero(state), inputs, count)
    return _pis(state, [inputs, count, support, destinations, lefts, rights,
        empty_support, input_support, steps, structural, root, ready, complete],
        _eq(state, actual, expected))


def _closed[C: Int](mut state: tk._TypedState[C], root: Int, expected: Int) -> Bool:
    var actual = tk._infer(state, root)
    return (expected >= 0 and actual >= 0 and tk._infer(state, expected) == _prop(state)
        and state.context_high[root] == 0 and state.context_high[expected] == 0
        and tk._convert(state, actual, expected))


def check_finishing_join_proof[C: Int, scratch: Int = 24576](
    bundle: tk.ProofTermBundle[C],
) -> Bool:
    """Admit every input node/definition before checking the closed exports.

    A positive resource gate is required before interpreting any negative
    result as a semantic rejection. There are no native arithmetic or record
    theorem opcodes in this profile. Ordinary equivalent proof terms remain
    valid; the final root is checked by type, rather than byte identity.
    """
    if (scratch <= 0 or bundle.count > scratch or bundle.definition_count > scratch
        or bundle.nat_constructor_count > scratch or bundle.constructor_count > scratch
        or bundle.definition_count < 95 or not phase._phase_syntax(bundle)
        or bundle.root < 0 or bundle.root >= bundle.count):
        return False
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[C, scratch](program, bundle)
    state.fuel = scratch * 1024
    tk._enable_native_numerals(state)
    if (not tk._check_inductive_environment(state)
        or not phase._canonical_phase_data(state) or not old._core_data(state)
        or not old._algebra_data(state) or not old._initial_data(state)
        or not old._trace_data(state) or not _maximum_data(state)
        or not _parity_data(state) or not _join_data(state)):
        return False
    for index in range(bundle.count):
        var actual = tk._infer(state, index)
        if actual == -1 or actual == -4:
            return False

    var old_roots: List[Int] = [54, 53, 49, 39, 41, 44]
    var old_types: List[Int] = [old._canonical_type(state),
        old._prefix_refinement_type(state), old._initialization_type(state),
        old._union_type(state), old._full_type(state), old._leaf_type(state)]
    for index in range(len(old_roots)):
        if not _closed(state, _named(state, old_roots[index]), old_types[index]):
            return False
    for law in range(56, 61):
        if not _closed(state, _named(state, law), _maximum_law_type(state, law)):
            return False
    var parity_laws: List[Int] = [70, 71, 77, 78]
    for law in parity_laws:
        if not _closed(state, _named(state, law), _parity_law_type(state, law)):
            return False
    for law in range(80, 86):
        if not _closed(state, _named(state, law), _boolean_law_type(state, law)):
            return False
    for law in range(87, 94):
        if not _closed(state, _named(state, law), _join_law_type(state, law)):
            return False
    var expected = _canonical_record_type(state)
    if (not _closed(state, _named(state, 94), expected)
        or not _closed(state, bundle.root, expected)):
        return False
    return (state.fuel > 0 and state.syntax.count < scratch
        and not state.arithmetic_verified and not state.fast_arithmetic)
