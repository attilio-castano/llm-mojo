"""Independent specification of the symbolic bounded-record encoding bridge.

No producer is imported. Every data body and public theorem type after the
checked Nat Join prefix is built independently. Ordinary equivalent proof
bodies are accepted; export indices and alleged signatures grant no authority.
The phase, execution and join specifications are reused. This profile addresses
arbitrary-bound Nat records with Boolean flags; native interpretation and
source/GPU correspondence remain separate.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase_check as phase
import llm_mojo.kernels.finishing_execution_check as execution
import llm_mojo.kernels.finishing_join_check as joined
from llm_mojo.kernels.finishing_phase_check import (
    _nat, _prop, _var, _pi, _lam, _named, _app, _zero, _succ, _eq,
    _rec, _store_type, _pis, _lams,
)

comptime NAT_SCORE_BRIDGE_DEFINITIONS = 148


@fieldwise_init
struct NatScoreBridgeLayout(ImplicitlyCopyable):
    var complement_first: Int
    var order_first: Int
    var comparison_first: Int
    var embedding_first: Int
    var selector_first: Int
    var encoding_first: Int
    var predecessor_conversion_root: Int


def _layout_valid(layout: NatScoreBridgeLayout, count: Int) -> Bool:
    var next = 111
    if layout.complement_first != next:
        return False
    next += 6
    if layout.order_first != next:
        return False
    next += 8
    if layout.comparison_first != next:
        return False
    next += 8
    if layout.embedding_first != next:
        return False
    next += 6
    if layout.selector_first != next:
        return False
    next += 3
    if layout.encoding_first != next:
        return False
    next += 6
    return (next == NAT_SCORE_BRIDGE_DEFINITIONS
        and layout.predecessor_conversion_root >= 0
        and layout.predecessor_conversion_root < count)


def _call[C: Int](mut state: tk._TypedState[C], definition: Int,
    arguments: List[Int]) -> Int:
    return execution._apps(state, _named(state, definition), arguments)


def _one[C: Int](mut state: tk._TypedState[C]) -> Int:
    return _succ(state, _zero(state))


def _add[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 95, [left, right])


def _mul[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 96, [left, right])


def _le[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 97, [left, right])


def _lt[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _le(state, _succ(state, left), right)


def _true[C: Int](mut state: tk._TypedState[C]) -> Int:
    var proposition = _var(state, _prop(state))
    var evidence = _var(state, proposition)
    return _pis(state, [proposition, evidence], proposition)


def _false[C: Int](mut state: tk._TypedState[C]) -> Int:
    var proposition = _var(state, _prop(state))
    return _pi(state, proposition, proposition)


def _comp[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    bound: Int, value: Int) -> Int:
    return _call(state, layout.complement_first, [bound, value])


def _score[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    bound: Int, rank: Int, token: Int) -> Int:
    return _call(state, layout.embedding_first, [bound, rank, token])


def _winner[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    ra: Int, rb: Int, ta: Int, tb: Int) -> Int:
    return _call(state, layout.selector_first, [ra, rb, ta, tb])


def _encoded[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    bound: Int, rank: Int, token: Int, bad: Int) -> Int:
    return _call(state, layout.encoding_first, [bound, rank, token, bad])


def _select[C: Int](mut state: tk._TypedState[C], flag: Int,
    left: Int, right: Int) -> Int:
    return _call(state, 3, [flag, left, right])


def _max[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 55, [left, right])


def _pack[C: Int](mut state: tk._TypedState[C], score: Int, bad: Int) -> Int:
    return _call(state, 67, [score, bad])


def _bit[C: Int](mut state: tk._TypedState[C], bad: Int) -> Int:
    return _call(state, 66, [bad])


def _or[C: Int](mut state: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(state, 68, [left, right])


def _arithmetic_data[C: Int](mut state: tk._TypedState[C]) -> Bool:
    # Add(a,b) recurses on b, returning a at zero and successor at each step.
    var left = _var(state, _nat(state))
    var right = _var(state, _nat(state))
    var predecessor = _var(state, _nat(state))
    var accumulated = _var(state, _nat(state))
    var step = _lams(state, [predecessor, accumulated], _succ(state, accumulated))
    if not phase._data_matches(state, 95, _pis(state, [left, right], _nat(state)),
        _lams(state, [left, right], _rec(state, right, left, step))):
        return False

    # Mul(k,x) recurses on x, adding k to the previous product.
    var factor = _var(state, _nat(state))
    var extent = _var(state, _nat(state))
    predecessor = _var(state, _nat(state))
    accumulated = _var(state, _nat(state))
    step = _lams(state, [predecessor, accumulated], _add(state, factor, accumulated))
    if not phase._data_matches(state, 96, _pis(state, [factor, extent], _nat(state)),
        _lams(state, [factor, extent], _rec(state, extent, _zero(state), step))):
        return False

    # Le(0,b)=Church True; Le(Sa,0)=Church False; Le(Sa,Sb)=Le(a,b).
    left = _var(state, _nat(state))
    right = _var(state, _nat(state))
    var argument = _var(state, _nat(state))
    var base = _lam(state, argument, _true(state))
    predecessor = _var(state, _nat(state))
    argument = _var(state, _nat(state))
    var previous = _var(state, _pi(state, argument, _prop(state)))
    var observation = _var(state, _nat(state))
    var observation_predecessor = _var(state, _nat(state))
    var ignored = _var(state, _prop(state))
    var observation_step = _lams(state, [observation_predecessor, ignored],
        _app(state, previous, observation_predecessor))
    step = _lams(state, [predecessor, previous, observation],
        _rec(state, observation, _false(state), observation_step))
    var body = _app(state, _rec(state, left, base, step), right)
    return phase._data_matches(state, 97,
        _pis(state, [left, right], _prop(state)), _lams(state, [left, right], body))


def _complement_data[C: Int](mut state: tk._TypedState[C],
    layout: NatScoreBridgeLayout) -> Bool:
    var bound = _var(state, _nat(state))
    var value = _var(state, _nat(state))
    var zero_case = _lam(state, value, _zero(state))
    var predecessor = _var(state, _nat(state))
    var previous = _var(state, _store_type(state))
    value = _var(state, _nat(state))
    var value_predecessor = _var(state, _nat(state))
    var ignored = _var(state, _nat(state))
    var next_value = _lams(state, [value_predecessor, ignored],
        _app(state, previous, value_predecessor))
    var successor_case = _lams(state, [predecessor, previous, value],
        _rec(state, value, _succ(state, predecessor), next_value))
    return phase._data_matches(state, layout.complement_first,
        _pi(state, bound, _store_type(state)),
        _lam(state, bound, _rec(state, bound, zero_case, successor_case)))


def _score_data[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout) -> Bool:
    var bound = _var(state, _nat(state))
    var rank = _var(state, _nat(state))
    var token = _var(state, _nat(state))
    var body = _add(state, _mul(state, rank, _succ(state, bound)),
        _comp(state, layout, bound, token))
    return phase._data_matches(state, layout.embedding_first,
        _pis(state, [bound, rank, token], _nat(state)),
        _lams(state, [bound, rank, token], body))


def _selector_data[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout) -> Bool:
    var ra = _var(state, _nat(state))
    var rb = _var(state, _nat(state))
    var ta = _var(state, _nat(state))
    var tb = _var(state, _nat(state))
    var tie = _select(state, _call(state, 4, [ra, rb]),
        execution._before_flag(state, tb, ta), _zero(state))
    var flag = _or(state, execution._before_flag(state, ra, rb), tie)
    var binders: List[Int] = [ra, rb, ta, tb]
    return phase._data_matches(state, layout.selector_first,
        _pis(state, binders, _nat(state)), _lams(state, binders, flag))


def _encoding_data[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout) -> Bool:
    var bound = _var(state, _nat(state))
    var rank = _var(state, _nat(state))
    var token = _var(state, _nat(state))
    var bad = _var(state, _nat(state))
    var binders: List[Int] = [bound, rank, token, bad]
    if not phase._data_matches(state, layout.encoding_first,
        _pis(state, binders, _nat(state)), _lams(state, binders,
            _pack(state, _score(state, layout, bound, rank, token), bad))):
        return False
    bound = _var(state, _nat(state))
    var ra = _var(state, _nat(state))
    var rb = _var(state, _nat(state))
    var ta = _var(state, _nat(state))
    var tb = _var(state, _nat(state))
    var ba = _var(state, _nat(state))
    var bb = _var(state, _nat(state))
    var union = _or(state, ba, bb)
    var body = _select(state, _winner(state, layout, ra, rb, ta, tb),
        _encoded(state, layout, bound, ra, ta, union),
        _encoded(state, layout, bound, rb, tb, union))
    binders = [bound, ra, rb, ta, tb, ba, bb]
    return phase._data_matches(state, layout.encoding_first + 1,
        _pis(state, binders, _nat(state)), _lams(state, binders, body))


def _arithmetic_type[C: Int](mut state: tk._TypedState[C], definition: Int) -> Int:
    var left = _var(state, _nat(state))
    if definition == 98:
        return _pi(state, left, _eq(state, _add(state, _zero(state), left), left))
    if definition == 104:
        return _pi(state, left, _eq(state, _mul(state, _zero(state), left), _zero(state)))
    if definition == 105:
        return _pi(state, left, _le(state, _zero(state), left))
    if definition == 108:
        var impossible = _var(state, _lt(state, left, left))
        return _pis(state, [left, impossible], _false(state))
    var right = _var(state, _nat(state))
    if definition == 99:
        return _pis(state, [left, right], _eq(state,
            _add(state, _succ(state, left), right), _succ(state, _add(state, left, right))))
    if definition == 101:
        return _pis(state, [left, right], _eq(state,
            _add(state, left, right), _add(state, right, left)))
    if definition == 103:
        return _pis(state, [left, right], _eq(state,
            _mul(state, _succ(state, left), right), _add(state, _mul(state, left, right), right)))
    if definition == 109:
        var result = _var(state, _prop(state))
        var strict = _var(state, _lt(state, left, right))
        var reverse = _var(state, _le(state, right, left))
        var strict_case = _var(state, _pi(state, strict, result))
        var reverse_case = _var(state, _pi(state, reverse, result))
        return _pis(state, [left, right, result, strict_case, reverse_case], result)
    if definition == 110:
        var ordered = _var(state, _le(state, left, right))
        var reverse = _var(state, _le(state, right, left))
        return _pis(state, [left, right, ordered, reverse], _eq(state, left, right))
    var last = _var(state, _nat(state))
    if definition == 100:
        return _pis(state, [left, right, last], _eq(state,
            _add(state, _add(state, left, right), last),
            _add(state, left, _add(state, right, last))))
    if definition == 102:
        # Binder order is suffix, left, right, equality.
        var equality = _var(state, _eq(state, _add(state, right, left), _add(state, last, left)))
        return _pis(state, [left, right, last, equality], _eq(state, right, last))
    var ordered = _var(state, _le(state, left, right))
    if definition == 106:
        var next_ordered = _var(state, _le(state, right, last))
        return _pis(state, [left, right, last, ordered, next_ordered], _le(state, left, last))
    # 107: suffix monotonicity, with the suffix before its hypothesis.
    return _pis(state, [left, right, last, ordered], _le(state,
        _add(state, left, last), _add(state, right, last)))


def _complement_type[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    offset: Int) -> Int:
    var bound = _var(state, _nat(state))
    if offset == 2:
        return _pi(state, bound, _eq(state, _comp(state, layout, bound, bound), _zero(state)))
    var left = _var(state, _nat(state))
    var left_comp = _comp(state, layout, bound, left)
    if offset == 1 or offset == 3:
        var bounded = _var(state, _le(state, left, bound))
        var result = (_eq(state, _add(state, left_comp, left), bound) if offset == 1
            else _le(state, left_comp, bound))
        return _pis(state, [bound, left, bounded], result)
    var right = _var(state, _nat(state))
    var right_comp = _comp(state, layout, bound, right)
    if offset == 4:
        var left_bound = _var(state, _le(state, left, bound))
        var right_bound = _var(state, _le(state, right, bound))
        var equal = _var(state, _eq(state, left_comp, right_comp))
        return _pis(state, [bound, left, right, left_bound, right_bound, equal],
            _eq(state, left, right))
    var ordered = _var(state, _le(state, left, right))
    var bounded = _var(state, _le(state, right, bound))
    return _pis(state, [bound, left, right, ordered, bounded], _le(state, right_comp, left_comp))


def _order_type[C: Int](mut state: tk._TypedState[C], offset: Int) -> Int:
    if offset == 0 or offset == 1:
        var prefix = _var(state, _nat(state))
        var left = _var(state, _nat(state))
        var right = _var(state, _nat(state))
        var premise = _var(state, _le(state, left, right) if offset == 0 else _lt(state, left, right))
        var lhs = _add(state, prefix, left)
        var rhs = _add(state, prefix, right)
        return _pis(state, [prefix, left, right, premise],
            _le(state, lhs, rhs) if offset == 0 else _lt(state, lhs, rhs))
    var left = _var(state, _nat(state))
    var right = _var(state, _nat(state))
    if offset == 2:
        var factor = _var(state, _nat(state))
        var premise = _var(state, _le(state, left, right))
        return _pis(state, [left, right, factor, premise], _le(state,
            _mul(state, left, factor), _mul(state, right, factor)))
    if offset == 3:
        var premise = _var(state, _lt(state, left, right))
        return _pis(state, [left, right, premise], _le(state, left, right))
    var selects_right = offset == 4 or offset == 6
    var smaller = left if selects_right else right
    var larger = right if selects_right else left
    var premise = _var(state, _le(state, smaller, larger) if offset < 6 else _lt(state, smaller, larger))
    return _pis(state, [left, right, premise], _eq(state, _max(state, left, right), larger))


def _comparison_type[C: Int](mut state: tk._TypedState[C], offset: Int) -> Int:
    var left = _var(state, _nat(state))
    var right = _var(state, _nat(state))
    var flag = execution._before_flag(state, left, right)
    if offset == 2:
        return _pis(state, [left, right], _bit(state, flag))
    if offset == 4:
        return _pis(state, [left, right], _bit(state, _call(state, 4, [left, right])))
    var premise: Int
    var result: Int
    if offset == 0:
        premise = _var(state, execution._before(state, left, right))
        result = _lt(state, left, right)
    elif offset == 1:
        premise = _var(state, _lt(state, left, right))
        result = execution._before(state, left, right)
    elif offset == 3:
        premise = _var(state, execution._before(state, left, right))
        result = _eq(state, flag, _one(state))
    elif offset == 5:
        premise = _var(state, _eq(state, _call(state, 4, [left, right]), _zero(state)))
        result = _eq(state, left, right)
    elif offset == 6:
        premise = _var(state, _lt(state, left, right))
        result = _eq(state, flag, _one(state))
    else:
        premise = _var(state, _le(state, right, left))
        result = _eq(state, flag, _zero(state))
    return _pis(state, [left, right, premise], result)


def _embedding_type[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    offset: Int) -> Int:
    var bound = _var(state, _nat(state))
    if offset == 1:
        var left = _var(state, _nat(state))
        var right = _var(state, _nat(state))
        var strict = _var(state, _lt(state, left, right))
        var bounded = _var(state, _le(state, right, bound))
        return _pis(state, [bound, left, right, strict, bounded], _lt(state,
            _comp(state, layout, bound, right), _comp(state, layout, bound, left)))
    var ra = _var(state, _nat(state))
    var rb = _var(state, _nat(state)) if offset == 2 or offset == 5 else ra
    var ta = _var(state, _nat(state))
    var tb = _var(state, _nat(state))
    var ba = _var(state, _le(state, ta, bound))
    var bb = _var(state, _le(state, tb, bound))
    var sa = _score(state, layout, bound, ra, ta)
    var sb = _score(state, layout, bound, rb, tb)
    var binders: List[Int] = [bound, ra]
    if offset == 2 or offset == 5:
        binders.append(rb)
    binders.append(ta)
    binders.append(tb)
    binders.append(ba)
    binders.append(bb)
    var premise: Int
    var result: Int
    if offset == 2:
        premise = _var(state, _lt(state, ra, rb))
        result = _lt(state, sa, sb)
    elif offset == 3:
        premise = _var(state, _lt(state, tb, ta))
        result = _lt(state, sa, sb)
    elif offset == 4:
        premise = _var(state, _le(state, ta, tb))
        result = _le(state, sb, sa)
    else:
        premise = _var(state, _eq(state, sa, sb))
        result = execution._and(state, [_eq(state, ra, rb), _eq(state, ta, tb)])
    binders.append(premise)
    return _pis(state, binders, result)


def _selector_type[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    offset: Int) -> Int:
    var bound = _var(state, _nat(state)) if offset == 2 else -1
    var ra = _var(state, _nat(state))
    var rb = _var(state, _nat(state))
    var ta = _var(state, _nat(state))
    var tb = _var(state, _nat(state))
    var flag = _winner(state, layout, ra, rb, ta, tb)
    if offset == 1:
        return _pis(state, [ra, rb, ta, tb], _bit(state, flag))
    var ba = _var(state, _le(state, ta, bound))
    var bb = _var(state, _le(state, tb, bound))
    var sa = _score(state, layout, bound, ra, ta)
    var sb = _score(state, layout, bound, rb, tb)
    return _pis(state, [bound, ra, rb, ta, tb, ba, bb],
        _eq(state, _max(state, sa, sb), _select(state, flag, sa, sb)))


def _encoding_type[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    offset: Int) -> Int:
    if offset == 2:
        var left = _var(state, _nat(state))
        var right = _var(state, _nat(state))
        var bad = _var(state, _nat(state))
        var flag = _var(state, _nat(state))
        return _pis(state, [left, right, bad, flag], _eq(state,
            _pack(state, _select(state, flag, left, right), bad),
            _select(state, flag, _pack(state, left, bad), _pack(state, right, bad))))
    var bound = _var(state, _nat(state))
    if offset == 3:
        return _pi(state, bound, _eq(state,
            _encoded(state, layout, bound, _zero(state), bound, _zero(state)), _zero(state)))
    var ra = _var(state, _nat(state))
    var rb = _var(state, _nat(state))
    var ta = _var(state, _nat(state))
    var tb = _var(state, _nat(state))
    var ba = _var(state, _nat(state))
    var bb = _var(state, _nat(state))
    var ta_bound = _var(state, _le(state, ta, bound))
    var tb_bound = _var(state, _le(state, tb, bound))
    var ba_bit = _var(state, _bit(state, ba))
    var bb_bit = _var(state, _bit(state, bb))
    var ea = _encoded(state, layout, bound, ra, ta, ba)
    var eb = _encoded(state, layout, bound, rb, tb, bb)
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb,
        ta_bound, tb_bound, ba_bit, bb_bit]
    var result: Int
    if offset == 4:
        binders.append(_var(state, _eq(state, ea, eb)))
        result = execution._and(state,
            [_eq(state, ra, rb), _eq(state, ta, tb), _eq(state, ba, bb)])
    else:
        var merged = _call(state, layout.encoding_first + 1, [bound, ra, rb, ta, tb, ba, bb])
        result = _eq(state, _call(state, 86, [ea, eb]), merged)
    return _pis(state, binders, result)


def _predecessor_type[C: Int](mut state: tk._TypedState[C]) -> Int:
    var target = _var(state, _nat(state))
    var predecessor = _var(state, _nat(state))
    var ignored = _var(state, _nat(state))
    var body = _rec(state, target, _zero(state),
        _lams(state, [predecessor, ignored], predecessor))
    return _pi(state, target, _eq(state, _call(state, 2, [target]), body))


def _qualified_prefix[C: Int](mut state: tk._TypedState[C]) -> Bool:
    if (not phase._canonical_phase_data(state) or not execution._core_data(state)
        or not execution._algebra_data(state) or not execution._initial_data(state)
        or not execution._trace_data(state) or not joined._maximum_data(state)
        or not joined._parity_data(state) or not joined._join_data(state)):
        return False
    var old_roots: List[Int] = [54, 53, 49, 39, 41, 44]
    var old_types: List[Int] = [execution._canonical_type(state),
        execution._prefix_refinement_type(state), execution._initialization_type(state),
        execution._union_type(state), execution._full_type(state), execution._leaf_type(state)]
    for index in range(len(old_roots)):
        if not joined._closed(state, _named(state, old_roots[index]), old_types[index]):
            return False
    for law in range(56, 61):
        if not joined._closed(state, _named(state, law), joined._maximum_law_type(state, law)):
            return False
    for law in [70, 71, 77, 78]:
        if not joined._closed(state, _named(state, law), joined._parity_law_type(state, law)):
            return False
    for law in range(80, 86):
        if not joined._closed(state, _named(state, law), joined._boolean_law_type(state, law)):
            return False
    for law in range(87, 94):
        if not joined._closed(state, _named(state, law), joined._join_law_type(state, law)):
            return False
    return joined._closed(state, _named(state, 94), joined._canonical_record_type(state))


def _new_specification[C: Int](mut state: tk._TypedState[C], layout: NatScoreBridgeLayout,
    root: Int) -> Bool:
    if (not _arithmetic_data(state) or not _complement_data(state, layout)
        or not _score_data(state, layout) or not _selector_data(state, layout)
        or not _encoding_data(state, layout)):
        return False
    for law in range(98, 111):
        if not joined._closed(state, _named(state, law), _arithmetic_type(state, law)):
            return False
    for offset in range(1, 6):
        if not joined._closed(state, _named(state, layout.complement_first + offset),
            _complement_type(state, layout, offset)):
            return False
    for offset in range(8):
        if not joined._closed(state, _named(state, layout.order_first + offset),
            _order_type(state, offset)):
            return False
    for offset in range(8):
        if not joined._closed(state, _named(state, layout.comparison_first + offset),
            _comparison_type(state, offset)):
            return False
    for offset in range(1, 6):
        if not joined._closed(state, _named(state, layout.embedding_first + offset),
            _embedding_type(state, layout, offset)):
            return False
    for offset in range(1, 3):
        if not joined._closed(state, _named(state, layout.selector_first + offset),
            _selector_type(state, layout, offset)):
            return False
    for offset in range(2, 6):
        if not joined._closed(state, _named(state, layout.encoding_first + offset),
            _encoding_type(state, layout, offset)):
            return False
    return (joined._closed(state, layout.predecessor_conversion_root, _predecessor_type(state))
        and joined._closed(state, root, _encoding_type(state, layout, 5)))



@no_inline
def _state_holder[C: Int, S: Int](bundle: tk.ProofTermBundle[C]) -> List[tk._TypedState[S]]:
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    # Construct the canonical kernel state directly into its sole heap slot. The
    # callback is nonthrowing and runs exactly once, immediately after allocation.
    # No return/exception/read/destruction can observe the uninitialized slot.
    def initialize() {imm program, imm bundle} -> tk._TypedState[S]:
        return tk._make_state[C, S](program, bundle)
    var states = List[tk._TypedState[S]](unsafe_uninit_length=1)
    states.unsafe_ptr().unsafe_write(init_with=initialize)
    return states^


def _check_in_state[C: Int, scratch: Int](mut state: tk._TypedState[scratch],
    bundle: tk.ProofTermBundle[C], layout: NatScoreBridgeLayout) -> Bool:
    state.fuel = scratch * 2048
    tk._enable_native_numerals(state)
    if not tk._check_inductive_environment(state):
        return False
    for index in range(bundle.count):
        var inferred = tk._infer(state, index)
        if inferred == -1 or inferred == -4:
            return False
    if (not _qualified_prefix(state) or not _new_specification(state, layout, bundle.root)):
        return False
    return (state.fuel > 0 and state.syntax.count < scratch
        and not state.arithmetic_verified and not state.fast_arithmetic)

def check_nat_score_bridge_proof[C: Int, scratch: Int = 40960](
    bundle: tk.ProofTermBundle[C], layout: NatScoreBridgeLayout,
) -> Bool:
    """Provisional resource-bounded profile, requiring its own positive gate.

    Metadata limits and the strict syntax policy precede state allocation.
    All definitions and all submitted syntax, including unused entries, are
    admitted. Definitions and exported proofs must be closed. Raw binder nodes
    may correctly have open contexts. Resource exhaustion returns false and
    must never be presented as a semantic counterexample or proof rejection.
    """
    if (scratch <= 0 or bundle.count <= 0 or bundle.count > C or bundle.count > scratch
        or bundle.definition_count > scratch or bundle.nat_constructor_count > scratch
        or bundle.constructor_count > scratch or bundle.nat_constructor_count < 0
        or bundle.constructor_count < 0
        or bundle.definition_count != NAT_SCORE_BRIDGE_DEFINITIONS
        or not _layout_valid(layout, bundle.count) or not phase._phase_syntax(bundle)
        or bundle.root < 0 or bundle.root >= bundle.count):
        return False
    var states = _state_holder[C, scratch](bundle)
    return _check_in_state(states[0], bundle, layout)
