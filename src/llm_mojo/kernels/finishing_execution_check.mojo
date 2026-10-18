"""Independent admission of the abstract finishing execution theorem.

Canonical definitions and six exported theorem types are reconstructed here,
without importing producer specification helpers. Every submitted node and
ordinary definition is checked, including unused syntax. Only the previously
certified numeral abbreviations are enabled; native arithmetic, reflection,
private inductive families and theorem-export opcodes are forbidden.

Descriptor binding is exact executable-data comparison. It does not prove the
compiler, native UInt32 record interpretation or GPU/warp refinement correct.
The universal result explicitly quantifies its join-algebra evidence.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase_check as phase
import llm_mojo.kernels.finishing_execution_trace as trace
import llm_mojo.kernels.finishing_execution as model

from llm_mojo.kernels.finishing_phase_check import (
    _node, _nat, _prop, _var, _pi, _lam, _named, _app, _a2, _a3,
    _zero, _succ, _eq, _rec, _store_type, _update, _pis, _lams,
)


@fieldwise_init
struct FinishingExecutionCertificate[
    threads: Int, items: Int, local_nodes: Int, group_width: Int,
    subgroup_nodes: Int, final_nodes: Int, capacity: Int,
](Copyable, Movable):
    var descriptor: model.FinishingExecutionDescriptor[
        Self.threads, Self.items, Self.local_nodes, Self.group_width,
        Self.subgroup_nodes, Self.final_nodes,
    ]
    var proof: trace.FinishingExecutionProof[Self.capacity]


def _apps[C: Int](mut s: tk._TypedState[C], var head: Int,
    arguments: List[Int]) -> Int:
    for argument in arguments:
        head = _app(s, head, argument)
    return head


def _call[C: Int](mut s: tk._TypedState[C], definition: Int,
    arguments: List[Int]) -> Int:
    return _apps(s, _named(s, definition), arguments)


def _join_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var a = _var(s, _nat(s))
    var b = _var(s, _nat(s))
    return _pis(s, [a, b], _nat(s))


def _support_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var slot = _var(s, _nat(s))
    return _pi(s, slot, _store_type(s))


def _and[C: Int](mut s: tk._TypedState[C], propositions: List[Int]) -> Int:
    var result = _var(s, _prop(s))
    var premises = List[Int]()
    for proposition in propositions:
        premises.append(_var(s, proposition))
    var consume = _var(s, _pis(s, premises, result))
    return _pis(s, [result, consume], result)


def _join[C: Int](mut s: tk._TypedState[C], j: Int, a: Int, b: Int) -> Int:
    return _a2(s, j, a, b)


def _select[C: Int](mut s: tk._TypedState[C], flag: Int,
    yes: Int, no: Int) -> Int:
    return _call(s, 3, [flag, yes, no])


def _values[C: Int](mut s: tk._TypedState[C], j: Int, initial: Int,
    destinations: Int, lefts: Int, rights: Int, count: Int) -> Int:
    return _call(s, 23, [j, initial, destinations, lefts, rights, count])


def _ready[C: Int](mut s: tk._TypedState[C], initial: Int,
    destinations: Int, count: Int) -> Int:
    return _call(s, 24, [initial, destinations, count])


def _legal_step[C: Int](mut s: tk._TypedState[C], j: Int, reference: Int,
    initial: Int, destinations: Int, lefts: Int, rights: Int, step: Int) -> Int:
    return _call(s, 25, [j, reference, initial, destinations, lefts, rights, step])


def _legal_prefix[C: Int](mut s: tk._TypedState[C], j: Int, reference: Int,
    initial: Int, destinations: Int, lefts: Int, rights: Int, count: Int) -> Int:
    return _call(s, 26, [j, reference, initial, destinations, lefts, rights, count])


def _flat[C: Int](mut s: tk._TypedState[C], j: Int, empty: Int,
    inputs: Int, count: Int) -> Int:
    return _call(s, 27, [j, empty, inputs, count])


def _take[C: Int](mut s: tk._TypedState[C], flag: Int,
    empty: Int, value: Int) -> Int:
    return _call(s, 30, [flag, empty, value])


def _union[C: Int](mut s: tk._TypedState[C], left: Int, right: Int) -> Int:
    return _call(s, 31, [left, right])


def _masked[C: Int](mut s: tk._TypedState[C], j: Int, empty: Int,
    inputs: Int, mask: Int, count: Int) -> Int:
    return _call(s, 32, [j, empty, inputs, mask, count])


def _unit[C: Int](mut s: tk._TypedState[C], source: Int) -> Int:
    return _call(s, 33, [source])


def _before[C: Int](mut s: tk._TypedState[C], source: Int, count: Int) -> Int:
    return _call(s, 34, [source, count])


def _covered[C: Int](mut s: tk._TypedState[C], mask: Int, count: Int) -> Int:
    return _call(s, 35, [mask, count])


def _before_flag[C: Int](mut s: tk._TypedState[C], source: Int, count: Int) -> Int:
    return _call(s, 45, [source, count])


def _initial_values[C: Int](mut s: tk._TypedState[C], empty: Int,
    inputs: Int, count: Int) -> Int:
    return _call(s, 46, [empty, inputs, count])


def _initial_ready[C: Int](mut s: tk._TypedState[C], count: Int) -> Int:
    return _call(s, 47, [count])


def _reference[C: Int](mut s: tk._TypedState[C], j: Int, empty: Int,
    inputs: Int, support: Int, count: Int) -> Int:
    var slot = _var(s, _nat(s))
    return _lam(s, slot, _masked(s, j, empty, inputs, _app(s, support, slot), count))


def _support_step[C: Int](mut s: tk._TypedState[C], initial: Int,
    support: Int, destinations: Int, lefts: Int, rights: Int, step: Int) -> Int:
    return _call(s, 50, [initial, support, destinations, lefts, rights, step])


def _support_prefix[C: Int](mut s: tk._TypedState[C], initial: Int,
    support: Int, destinations: Int, lefts: Int, rights: Int, count: Int) -> Int:
    return _call(s, 51, [initial, support, destinations, lefts, rights, count])


def _matches[C: Int](mut s: tk._TypedState[C], index: Int,
    expected: Int, body: Int) -> Bool:
    return phase._data_matches(s, index, expected, body)


def _core_data[C: Int](mut s: tk._TypedState[C]) -> Bool:
    var j = _var(s, _join_type(s))
    var initial = _var(s, _store_type(s))
    var destinations = _var(s, _store_type(s))
    var lefts = _var(s, _store_type(s))
    var rights = _var(s, _store_type(s))
    var n = _var(s, _nat(s))
    var k = _var(s, _nat(s))
    var state = _var(s, _store_type(s))
    var next_value = _join(s, j, _app(s, state, _app(s, lefts, k)),
        _app(s, state, _app(s, rights, k)))
    var step = _lams(s, [k, state], _update(s, state,
        _app(s, destinations, k), next_value))
    var binders: List[Int] = [j, initial, destinations, lefts, rights, n]
    if not _matches(s, 23, _pis(s, binders, _store_type(s)),
        _lams(s, binders, _rec(s, n, initial, step))):
        return False

    initial = _var(s, _store_type(s))
    destinations = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    k = _var(s, _nat(s))
    state = _var(s, _store_type(s))
    step = _lams(s, [k, state], _update(s, state,
        _app(s, destinations, k), _succ(s, _zero(s))))
    binders = [initial, destinations, n]
    if not _matches(s, 24, _pis(s, binders, _store_type(s)),
        _lams(s, binders, _rec(s, n, initial, step))):
        return False

    j = _var(s, _join_type(s))
    var reference = _var(s, _store_type(s))
    initial = _var(s, _store_type(s))
    destinations = _var(s, _store_type(s))
    lefts = _var(s, _store_type(s))
    rights = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    var ready = _ready(s, initial, destinations, n)
    var left = _app(s, lefts, n)
    var right = _app(s, rights, n)
    var destination = _app(s, destinations, n)
    var propositions: List[Int] = [
        _eq(s, _app(s, ready, left), _succ(s, _zero(s))),
        _eq(s, _app(s, ready, right), _succ(s, _zero(s))),
        _eq(s, _app(s, ready, destination), _zero(s)),
        _eq(s, _app(s, reference, destination),
            _join(s, j, _app(s, reference, left), _app(s, reference, right))),
    ]
    binders = [j, reference, initial, destinations, lefts, rights, n]
    if not _matches(s, 25, _pis(s, binders, _prop(s)),
        _lams(s, binders, _and(s, propositions))):
        return False

    j = _var(s, _join_type(s))
    reference = _var(s, _store_type(s))
    initial = _var(s, _store_type(s))
    destinations = _var(s, _store_type(s))
    lefts = _var(s, _store_type(s))
    rights = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    k = _var(s, _nat(s))
    var previous = _var(s, _prop(s))
    step = _lams(s, [k, previous], _and(s,
        [previous, _legal_step(s, j, reference, initial, destinations, lefts, rights, k)]))
    binders = [j, reference, initial, destinations, lefts, rights, n]
    if not _matches(s, 26, _pis(s, binders, _prop(s)),
        _lams(s, binders, _rec(s, n, _eq(s, _zero(s), _zero(s)), step))):
        return False

    j = _var(s, _join_type(s))
    var empty = _var(s, _nat(s))
    var inputs = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    k = _var(s, _nat(s))
    var accumulated = _var(s, _nat(s))
    step = _lams(s, [k, accumulated], _join(s, j, accumulated, _app(s, inputs, k)))
    binders = [j, empty, inputs, n]
    return _matches(s, 27, _pis(s, binders, _nat(s)),
        _lams(s, binders, _rec(s, n, empty, step)))


def _algebra_data[C: Int](mut s: tk._TypedState[C]) -> Bool:
    var flag = _var(s, _nat(s))
    var empty = _var(s, _nat(s))
    var value = _var(s, _nat(s))
    if not _matches(s, 30, _pis(s, [flag, empty, value], _nat(s)),
        _lams(s, [flag, empty, value], _select(s, flag, empty, value))):
        return False
    var left = _var(s, _store_type(s))
    var right = _var(s, _store_type(s))
    var at = _var(s, _nat(s))
    if not _matches(s, 31, _pis(s, [left, right, at], _nat(s)),
        _lams(s, [left, right, at], _select(s, _app(s, left, at),
            _app(s, right, at), _succ(s, _zero(s))))):
        return False
    var j = _var(s, _join_type(s))
    empty = _var(s, _nat(s))
    var inputs = _var(s, _store_type(s))
    var mask = _var(s, _store_type(s))
    var n = _var(s, _nat(s))
    var k = _var(s, _nat(s))
    var accumulated = _var(s, _nat(s))
    var step = _lams(s, [k, accumulated], _join(s, j, accumulated,
        _take(s, _app(s, mask, k), empty, _app(s, inputs, k))))
    var binders: List[Int] = [j, empty, inputs, mask, n]
    if not _matches(s, 32, _pis(s, binders, _nat(s)),
        _lams(s, binders, _rec(s, n, empty, step))):
        return False
    var source = _var(s, _nat(s))
    at = _var(s, _nat(s))
    if not _matches(s, 33, _pis(s, [source, at], _nat(s)),
        _lams(s, [source, at], _select(s,
            _call(s, 4, [source, at]), _succ(s, _zero(s)), _zero(s)))):
        return False
    source = _var(s, _nat(s))
    n = _var(s, _nat(s))
    k = _var(s, _nat(s))
    var previous = _var(s, _prop(s))
    var base = _lam(s, n, _rec(s, n, _named(s, 0),
        _lams(s, [k, previous], _eq(s, _zero(s), _zero(s)))))
    var predecessor = _var(s, _nat(s))
    var bound = _var(s, _nat(s))
    var recursive = _var(s, _pi(s, bound, _prop(s)))
    n = _var(s, _nat(s))
    k = _var(s, _nat(s))
    previous = _var(s, _prop(s))
    step = _lams(s, [predecessor, recursive, n], _rec(s, n, _named(s, 0),
        _lams(s, [k, previous], _app(s, recursive, k))))
    bound = _var(s, _nat(s))
    if not _matches(s, 34, _pi(s, source, _pi(s, bound, _prop(s))),
        _lam(s, source, _rec(s, source, base, step))):
        return False
    mask = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    k = _var(s, _nat(s))
    previous = _var(s, _prop(s))
    step = _lams(s, [k, previous], _and(s,
        [previous, _eq(s, _app(s, mask, k), _succ(s, _zero(s)))]))
    return _matches(s, 35, _pis(s, [mask, n], _prop(s)),
        _lams(s, [mask, n], _rec(s, n, _eq(s, _zero(s), _zero(s)), step)))


def _initial_data[C: Int](mut s: tk._TypedState[C]) -> Bool:
    var source = _var(s, _nat(s))
    var n = _var(s, _nat(s))
    var k = _var(s, _nat(s))
    var previous = _var(s, _nat(s))
    var base = _lam(s, n, _rec(s, n, _zero(s),
        _lams(s, [k, previous], _succ(s, _zero(s)))))
    var predecessor = _var(s, _nat(s))
    var recursive = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    k = _var(s, _nat(s))
    previous = _var(s, _nat(s))
    var step = _lams(s, [predecessor, recursive, n], _rec(s, n, _zero(s),
        _lams(s, [k, previous], _app(s, recursive, k))))
    if not _matches(s, 45, _pi(s, source, _store_type(s)),
        _lam(s, source, _rec(s, source, base, step))):
        return False
    var empty = _var(s, _nat(s))
    var inputs = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    if not _matches(s, 46, _pis(s, [empty, inputs, n], _store_type(s)),
        _lams(s, [empty, inputs, n], _update(s, inputs, n, empty))):
        return False
    n = _var(s, _nat(s))
    var slot = _var(s, _nat(s))
    var flags = _lam(s, slot, _before_flag(s, slot, n))
    return _matches(s, 47, _pi(s, n, _store_type(s)),
        _lam(s, n, _update(s, flags, n, _succ(s, _zero(s)))))


def _trace_data[C: Int](mut s: tk._TypedState[C]) -> Bool:
    var initial = _var(s, _store_type(s))
    var support = _var(s, _support_type(s))
    var destinations = _var(s, _store_type(s))
    var lefts = _var(s, _store_type(s))
    var rights = _var(s, _store_type(s))
    var n = _var(s, _nat(s))
    var ready = _ready(s, initial, destinations, n)
    var left = _app(s, lefts, n)
    var right = _app(s, rights, n)
    var destination = _app(s, destinations, n)
    var at = _var(s, _nat(s))
    var mask = _app(s, support, destination)
    var union = _union(s, _app(s, support, left), _app(s, support, right))
    var propositions: List[Int] = [
        _eq(s, _app(s, ready, left), _succ(s, _zero(s))),
        _eq(s, _app(s, ready, right), _succ(s, _zero(s))),
        _eq(s, _app(s, ready, destination), _zero(s)),
        _pi(s, at, _eq(s, _app(s, mask, at), _app(s, union, at))),
    ]
    var binders: List[Int] = [initial, support, destinations, lefts, rights, n]
    if not _matches(s, 50, _pis(s, binders, _prop(s)),
        _lams(s, binders, _and(s, propositions))):
        return False
    initial = _var(s, _store_type(s))
    support = _var(s, _support_type(s))
    destinations = _var(s, _store_type(s))
    lefts = _var(s, _store_type(s))
    rights = _var(s, _store_type(s))
    n = _var(s, _nat(s))
    var k = _var(s, _nat(s))
    var previous = _var(s, _prop(s))
    var step = _lams(s, [k, previous], _and(s,
        [previous, _support_step(s, initial, support, destinations, lefts, rights, k)]))
    binders = [initial, support, destinations, lefts, rights, n]
    return _matches(s, 51, _pis(s, binders, _prop(s)),
        _lams(s, binders, _rec(s, n, _eq(s, _zero(s), _zero(s)), step)))


def _algebra[C: Int](mut s: tk._TypedState[C]) -> List[Int]:
    var j = _var(s, _join_type(s))
    var empty = _var(s, _nat(s))
    var a = _var(s, _nat(s))
    var b = _var(s, _nat(s))
    var c = _var(s, _nat(s))
    var associative = _var(s, _pis(s, [a, b, c], _eq(s,
        _join(s, j, _join(s, j, a, b), c), _join(s, j, a, _join(s, j, b, c)))))
    a = _var(s, _nat(s))
    b = _var(s, _nat(s))
    var commutative = _var(s, _pis(s, [a, b],
        _eq(s, _join(s, j, a, b), _join(s, j, b, a))))
    a = _var(s, _nat(s))
    var left_unit = _var(s, _pi(s, a, _eq(s, _join(s, j, empty, a), a)))
    a = _var(s, _nat(s))
    var right_unit = _var(s, _pi(s, a, _eq(s, _join(s, j, a, empty), a)))
    a = _var(s, _nat(s))
    var idempotent = _var(s, _pi(s, a, _eq(s, _join(s, j, a, a), a)))
    return [j, empty, associative, commutative, left_unit, right_unit, idempotent]


def _support_premises[C: Int](mut s: tk._TypedState[C], support: Int,
    count: Int) -> List[Int]:
    var at = _var(s, _nat(s))
    var empty_support = _var(s, _pi(s, at, _eq(s,
        _app(s, _app(s, support, count), at), _zero(s))))
    var source = _var(s, _nat(s))
    var before = _var(s, _before(s, source, count))
    at = _var(s, _nat(s))
    var input_support = _var(s, _pis(s, [source, before, at], _eq(s,
        _app(s, _app(s, support, source), at), _app(s, _unit(s, source), at))))
    return [empty_support, input_support]


def _canonical_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var binders = _algebra(s)
    var j = binders[0]
    var empty = binders[1]
    var inputs = _var(s, _store_type(s))
    var count = _var(s, _nat(s))
    var support = _var(s, _support_type(s))
    var destinations = _var(s, _store_type(s))
    var lefts = _var(s, _store_type(s))
    var rights = _var(s, _store_type(s))
    var support_premises = _support_premises(s, support, count)
    var n = _var(s, _nat(s))
    var initial_ready = _initial_ready(s, count)
    var structural = _var(s, _support_prefix(s,
        initial_ready, support, destinations, lefts, rights, n))
    var root = _var(s, _nat(s))
    var ready = _var(s, _eq(s,
        _app(s, _ready(s, initial_ready, destinations, n), root), _succ(s, _zero(s))))
    var complete = _var(s, _covered(s, _app(s, support, root), count))
    for binder in [inputs, count, support, destinations, lefts, rights,
        support_premises[0], support_premises[1], n, structural, root, ready, complete]:
        binders.append(binder)
    return _pis(s, binders, _eq(s,
        _app(s, _values(s, j, _initial_values(s, empty, inputs, count),
            destinations, lefts, rights, n), root), _flat(s, j, empty, inputs, count)))


def _prefix_refinement_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var binders = _algebra(s)
    var j = binders[0]
    var empty = binders[1]
    var inputs = _var(s, _store_type(s))
    var count = _var(s, _nat(s))
    var support = _var(s, _support_type(s))
    var initial = _var(s, _store_type(s))
    var destinations = _var(s, _store_type(s))
    var lefts = _var(s, _store_type(s))
    var rights = _var(s, _store_type(s))
    var n = _var(s, _nat(s))
    var structural = _var(s, _support_prefix(s,
        initial, support, destinations, lefts, rights, n))
    for binder in [inputs, count, support, initial, destinations, lefts, rights, n, structural]:
        binders.append(binder)
    return _pis(s, binders, _legal_prefix(s, j,
        _reference(s, j, empty, inputs, support, count), initial, destinations, lefts, rights, n))


def _initialization_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var binders = _algebra(s)
    var j = binders[0]
    var empty = binders[1]
    var inputs = _var(s, _store_type(s))
    var count = _var(s, _nat(s))
    var support = _var(s, _support_type(s))
    var premises = _support_premises(s, support, count)
    for binder in [inputs, count, support, premises[0], premises[1]]:
        binders.append(binder)
    var slot = _var(s, _nat(s))
    var ready = _var(s, _eq(s,
        _app(s, _initial_ready(s, count), slot), _succ(s, _zero(s))))
    binders.append(slot)
    binders.append(ready)
    return _pis(s, binders, _eq(s,
        _app(s, _initial_values(s, empty, inputs, count), slot),
        _app(s, _reference(s, j, empty, inputs, support, count), slot)))


def _union_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var binders = _algebra(s)
    var j = binders[0]
    var empty = binders[1]
    var inputs = _var(s, _store_type(s))
    var left = _var(s, _store_type(s))
    var right = _var(s, _store_type(s))
    var n = _var(s, _nat(s))
    for binder in [inputs, left, right, n]:
        binders.append(binder)
    return _pis(s, binders, _eq(s,
        _masked(s, j, empty, inputs, _union(s, left, right), n),
        _join(s, j, _masked(s, j, empty, inputs, left, n),
            _masked(s, j, empty, inputs, right, n))))


def _full_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var binders = _algebra(s)
    var j = binders[0]
    var empty = binders[1]
    var inputs = _var(s, _store_type(s))
    var mask = _var(s, _store_type(s))
    var n = _var(s, _nat(s))
    var complete = _var(s, _covered(s, mask, n))
    for binder in [inputs, mask, n, complete]:
        binders.append(binder)
    return _pis(s, binders, _eq(s,
        _masked(s, j, empty, inputs, mask, n), _flat(s, j, empty, inputs, n)))


def _leaf_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var binders = _algebra(s)
    var j = binders[0]
    var empty = binders[1]
    var source = _var(s, _nat(s))
    var inputs = _var(s, _store_type(s))
    var n = _var(s, _nat(s))
    var before = _var(s, _before(s, source, n))
    for binder in [source, inputs, n, before]:
        binders.append(binder)
    return _pis(s, binders, _eq(s,
        _masked(s, j, empty, inputs, _unit(s, source), n), _app(s, inputs, source)))


def check_finishing_execution_proof[C: Int, scratch: Int = 16384](
    proof: trace.FinishingExecutionProof[C],
) -> Bool:
    """Admit canonical complete-trace semantics under explicit join laws."""
    if (scratch <= 0 or proof.bundle.count > scratch or proof.bundle.definition_count > scratch
        or proof.bundle.nat_constructor_count > scratch or proof.bundle.constructor_count > scratch
        or proof.bundle.definition_count < 55 or not phase._phase_syntax(proof.bundle)
        or proof.bundle.root != proof.canonical_root):
        return False
    var roots = [proof.canonical_root, proof.prefix_refinement_root,
        proof.initialization_root, proof.fold_union_root, proof.full_fold_root, proof.leaf_root]
    for root in roots:
        if root < 0 or root >= proof.bundle.count:
            return False
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[C, scratch](program, proof.bundle)
    state.fuel = scratch * 512
    tk._enable_native_numerals(state)
    if (not tk._check_inductive_environment(state)
        or not phase._canonical_phase_data(state) or not _core_data(state)
        or not _algebra_data(state) or not _initial_data(state) or not _trace_data(state)):
        return False
    for j in range(proof.bundle.count):
        var inferred = tk._infer(state, j)
        if inferred == -1 or inferred == -4:
            return False
    var expected = [_canonical_type(state), _prefix_refinement_type(state),
        _initialization_type(state), _union_type(state), _full_type(state), _leaf_type(state)]
    var prop = _prop(state)
    for j in range(len(roots)):
        var root = roots[j]
        var type = expected[j]
        if (type < 0 or tk._infer(state, type) != prop or state.context_high[type] != 0
            or tk._infer(state, root) < 0 or state.context_high[root] != 0
            or not tk._convert(state, tk._infer(state, root), type)):
            return False
    return not state.arithmetic_verified and not state.fast_arithmetic


def same_finishing_execution_descriptor[
    T: Int, I: Int, L: Int, W: Int, S: Int, G: Int,
](a: model.FinishingExecutionDescriptor[T, I, L, W, S, G],
    b: model.FinishingExecutionDescriptor[T, I, L, W, S, G]) -> Bool:
    if (a.spec_version != b.spec_version or a.operation_profile != b.operation_profile
        or a.local.root != b.local.root or a.subgroup.root != b.subgroup.root
        or a.final.root != b.final.root):
        return False
    var p = a.program
    var q = b.program
    if (p.threads != q.threads or p.items != q.items or p.stride != q.stride
        or p.start != q.start or p.thread_scale != q.thread_scale or p.item_scale != q.item_scale
        or p.index_bias != q.index_bias or p.loop_guard != q.loop_guard
        or p.load_guard != q.load_guard or p.count_limit != q.count_limit
        or p.integer_limit != q.integer_limit):
        return False
    for n in range(L):
        var x = a.local.nodes[n]
        var y = b.local.nodes[n]
        if x.operation != y.operation or x.left != y.left or x.right != y.right:
            return False
    for n in range(S):
        var x = a.subgroup.nodes[n]
        var y = b.subgroup.nodes[n]
        if x.operation != y.operation or x.left != y.left or x.right != y.right:
            return False
    for n in range(G):
        var x = a.final.nodes[n]
        var y = b.final.nodes[n]
        if x.operation != y.operation or x.left != y.left or x.right != y.right:
            return False
    return True


def make_finishing_execution_certificate[
    T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, //,
    capacity: Int = 8192,
](descriptor: model.FinishingExecutionDescriptor[T, I, L, W, S, G],
) -> FinishingExecutionCertificate[T, I, L, W, S, G, capacity]:
    return FinishingExecutionCertificate[T, I, L, W, S, G, capacity](
        descriptor.copy(), trace.make_finishing_execution_proof[capacity]())


def check_finishing_execution_certificate_request[
    T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, //,
    C: Int, scratch: Int = 16384,
](certificate: FinishingExecutionCertificate[T, I, L, W, S, G, C],
    expected: model.FinishingExecutionDescriptor[T, I, L, W, S, G],
) -> Bool:
    return (model.check_finishing_execution_descriptor(expected)
        and same_finishing_execution_descriptor(certificate.descriptor, expected)
        and check_finishing_execution_proof[C, scratch](certificate.proof))


def check_finishing_execution_certificate[
    T: Int, I: Int, L: Int, W: Int, S: Int, G: Int, //,
    C: Int, scratch: Int = 16384,
](certificate: FinishingExecutionCertificate[T, I, L, W, S, G, C]) -> Bool:
    return check_finishing_execution_certificate_request[C, scratch](
        certificate, certificate.descriptor)


def check_finishing_execution_rules[
    capacity: Int = 8192, scratch: Int = 16384,
]() -> Bool:
    var proof = trace.make_finishing_execution_proof[capacity]()
    return check_finishing_execution_proof[capacity, scratch](proof)
