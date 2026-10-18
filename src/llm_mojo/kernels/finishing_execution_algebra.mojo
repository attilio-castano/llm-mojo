"""Ordinary Nat proofs relating provenance masks to an abstract join fold.

Join algebra is explicit quantified evidence, not a kernel rule. Masks describe
input contribution independently of values; arbitrary runtime extents remain
Nat binders. The native UInt32-record interpretation is a separate refinement.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_terms as ex

comptime EXEC_TAKE = 30
comptime EXEC_MASK_UNION = 31
comptime EXEC_MASKED_FOLD = 32
comptime EXEC_UNIT_MASK = 33
comptime EXEC_BEFORE = 34
comptime EXEC_ALL_PREFIX = 35
comptime EXEC_INTERCHANGE = 36
comptime EXEC_BIT_UNION = 37
comptime EXEC_FOLD_EXTENSIONAL = 38
comptime EXEC_FOLD_UNION = 39
comptime EXEC_ZERO_FOLD = 40
comptime EXEC_FULL_FOLD = 41
comptime EXEC_ALGEBRA_DEFINITIONS = 42


def _join[C: Int](mut p: ph._Builder[C], j: Int, a: Int, b: Int) -> Int:
    return ph._apps(p, j, [a, b])


def _rec[C: Int](mut p: ph._Builder[C], n: Int, base: Int, step: Int) -> Int:
    return ph._node(p, tk.PT_NAT_REC, n, base, step)


def _take[C: Int](mut p: ph._Builder[C], flag: Int, empty: Int, value: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_TAKE), [flag, empty, value])


def _union[C: Int](mut p: ph._Builder[C], a: Int, b: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_MASK_UNION), [a, b])


def _fold[C: Int](mut p: ph._Builder[C], j: Int, empty: Int,
    inputs: Int, mask: Int, n: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_MASKED_FOLD), [j, empty, inputs, mask, n])


def _unit[C: Int](mut p: ph._Builder[C], source: Int) -> Int:
    return ph._app(p, ph._named(p, EXEC_UNIT_MASK), source)


def _before[C: Int](mut p: ph._Builder[C], source: Int, count: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_BEFORE), [source, count])


def _prefix[C: Int](mut p: ph._Builder[C], mask: Int, count: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_ALL_PREFIX), [mask, count])


def _and_type[C: Int](mut p: ph._Builder[C], propositions: List[Int]) -> Int:
    var result = ph._var(p, ph._prop(p))
    var arguments = List[Int]()
    for proposition in propositions:
        arguments.append(ph._var(p, proposition))
    var consume = ph._var(p, ph._pis(p, arguments, result))
    return ph._pis(p, [result, consume], result)


def _and_intro[C: Int](mut p: ph._Builder[C], propositions: List[Int],
    proofs: List[Int]) -> Int:
    var result = ph._var(p, ph._prop(p))
    var arguments = List[Int]()
    for proposition in propositions:
        arguments.append(ph._var(p, proposition))
    var consume = ph._var(p, ph._pis(p, arguments, result))
    return ph._lams(p, [result, consume], ph._apps(p, consume, proofs))


def _project[C: Int](mut p: ph._Builder[C], conjunction: Int,
    propositions: List[Int], index: Int) -> Int:
    var arguments = List[Int]()
    for proposition in propositions:
        arguments.append(ph._var(p, proposition))
    return ph._apps(p, conjunction,
        [propositions[index], ph._lams(p, arguments, arguments[index])])


def _left[C: Int](mut p: ph._Builder[C], j: Int, a: Int, b: Int,
    equal: Int, fixed: Int) -> Int:
    var value = ph._var(p, ph._nat(p))
    return ph._cong(p, ph._lam(p, value, _join(p, j, value, fixed)), a, b, equal)


def _right[C: Int](mut p: ph._Builder[C], j: Int, fixed: Int,
    a: Int, b: Int, equal: Int) -> Int:
    var value = ph._var(p, ph._nat(p))
    return ph._cong(p, ph._lam(p, value, _join(p, j, fixed, value)), a, b, equal)


def _algebra[C: Int](mut p: ph._Builder[C]) -> List[Int]:
    """Binder order: Join, Empty, associativity, commutativity, left/right unit, idempotence."""
    var j = ph._var(p, ex.execution_join_type(p))
    var empty = ph._var(p, ph._nat(p))
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var c = ph._var(p, ph._nat(p))
    var associative = ph._var(p, ph._pis(p, [a, b, c], ph._eq(p,
        _join(p, j, _join(p, j, a, b), c), _join(p, j, a, _join(p, j, b, c)))))
    a = ph._var(p, ph._nat(p))
    b = ph._var(p, ph._nat(p))
    var commutative = ph._var(p, ph._pis(p, [a, b],
        ph._eq(p, _join(p, j, a, b), _join(p, j, b, a))))
    a = ph._var(p, ph._nat(p))
    var left_unit = ph._var(p, ph._pi(p, a, ph._eq(p, _join(p, j, empty, a), a)))
    a = ph._var(p, ph._nat(p))
    var right_unit = ph._var(p, ph._pi(p, a, ph._eq(p, _join(p, j, a, empty), a)))
    a = ph._var(p, ph._nat(p))
    var idempotent = ph._var(p, ph._pi(p, a, ph._eq(p, _join(p, j, a, a), a)))
    return [j, empty, associative, commutative, left_unit, right_unit, idempotent]


def _call[C: Int](mut p: ph._Builder[C], definition: Int,
    algebra: List[Int], more: List[Int]) -> Int:
    var arguments = algebra.copy()
    for argument in more:
        arguments.append(argument)
    return ph._apps(p, ph._named(p, definition), arguments)


def _data[C: Int](mut p: ph._Builder[C]):
    var flag = ph._var(p, ph._nat(p))
    var empty = ph._var(p, ph._nat(p))
    var value = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pis(p, [flag, empty, value], ph._nat(p)),
        ph._lams(p, [flag, empty, value], ph._select(p, flag, empty, value)), EXEC_TAKE)
    var left = ph._var(p, ph._store(p))
    var right = ph._var(p, ph._store(p))
    var at = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pis(p, [left, right, at], ph._nat(p)),
        ph._lams(p, [left, right, at],
            ph._select(p, ph._app(p, left, at), ph._app(p, right, at), ph._one(p))), EXEC_MASK_UNION)
    var j = ph._var(p, ex.execution_join_type(p))
    empty = ph._var(p, ph._nat(p))
    var inputs = ph._var(p, ph._store(p))
    var mask = ph._var(p, ph._store(p))
    var n = ph._var(p, ph._nat(p))
    var k = ph._var(p, ph._nat(p))
    var accumulated = ph._var(p, ph._nat(p))
    var step = ph._lams(p, [k, accumulated], _join(p, j, accumulated,
        _take(p, ph._app(p, mask, k), empty, ph._app(p, inputs, k))))
    ph._declare(p, ph._pis(p, [j, empty, inputs, mask, n], ph._nat(p)),
        ph._lams(p, [j, empty, inputs, mask, n], _rec(p, n, empty, step)), EXEC_MASKED_FOLD)
    var source = ph._var(p, ph._nat(p))
    at = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pis(p, [source, at], ph._nat(p)),
        ph._lams(p, [source, at], ph._select(p, ph._eqb(p, source, at), ph._one(p), ph._zero(p))),
        EXEC_UNIT_MASK)
    # Before(0,S n)=True, Before(S s,S n)=Before(s,n), Before(s,0)=False.
    source = ph._var(p, ph._nat(p))
    n = ph._var(p, ph._nat(p))
    k = ph._var(p, ph._nat(p))
    var previous = ph._var(p, ph._prop(p))
    var base = ph._lam(p, n, _rec(p, n, ph._false(p),
        ph._lams(p, [k, previous], ph._eq(p, ph._zero(p), ph._zero(p)))))
    var predecessor = ph._var(p, ph._nat(p))
    var bound_index = ph._var(p, ph._nat(p))
    var recursive = ph._var(p, ph._pi(p, bound_index, ph._prop(p)))
    n = ph._var(p, ph._nat(p))
    k = ph._var(p, ph._nat(p))
    previous = ph._var(p, ph._prop(p))
    step = ph._lams(p, [predecessor, recursive, n], _rec(p, n, ph._false(p),
        ph._lams(p, [k, previous], ph._app(p, recursive, k))))
    var result_index = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, source, ph._pi(p, result_index, ph._prop(p))),
        ph._lam(p, source, _rec(p, source, base, step)), EXEC_BEFORE)
    mask = ph._var(p, ph._store(p))
    n = ph._var(p, ph._nat(p))
    k = ph._var(p, ph._nat(p))
    previous = ph._var(p, ph._prop(p))
    step = ph._lams(p, [k, previous], _and_type(p,
        [previous, ph._eq(p, ph._app(p, mask, k), ph._one(p))]))
    ph._declare(p, ph._pis(p, [mask, n], ph._prop(p)),
        ph._lams(p, [mask, n], _rec(p, n, ph._eq(p, ph._zero(p), ph._zero(p)), step)),
        EXEC_ALL_PREFIX)


def _interchange[C: Int](mut p: ph._Builder[C]):
    var algebra = _algebra(p)
    var j = algebra[0]
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var c = ph._var(p, ph._nat(p))
    var d = ph._var(p, ph._nat(p))
    var t0 = _join(p, j, _join(p, j, a, b), _join(p, j, c, d))
    var t1 = _join(p, j, a, _join(p, j, b, _join(p, j, c, d)))
    var t2 = _join(p, j, a, _join(p, j, _join(p, j, b, c), d))
    var t3 = _join(p, j, a, _join(p, j, _join(p, j, c, b), d))
    var t4 = _join(p, j, a, _join(p, j, c, _join(p, j, b, d)))
    var t5 = _join(p, j, _join(p, j, a, c), _join(p, j, b, d))
    var proof = ph._apps(p, algebra[2], [a, b, _join(p, j, c, d)])
    var inner = ph._apps(p, algebra[2], [b, c, d])
    inner = ph._sym(p, _join(p, j, _join(p, j, b, c), d),
        _join(p, j, b, _join(p, j, c, d)), inner)
    var next = _right(p, j, a, _join(p, j, b, _join(p, j, c, d)),
        _join(p, j, _join(p, j, b, c), d), inner)
    proof = ph._trans(p, t0, t1, t2, proof, next)
    inner = _left(p, j, _join(p, j, b, c), _join(p, j, c, b),
        ph._apps(p, algebra[3], [b, c]), d)
    next = _right(p, j, a, _join(p, j, _join(p, j, b, c), d),
        _join(p, j, _join(p, j, c, b), d), inner)
    proof = ph._trans(p, t0, t2, t3, proof, next)
    next = _right(p, j, a, _join(p, j, _join(p, j, c, b), d),
        _join(p, j, c, _join(p, j, b, d)), ph._apps(p, algebra[2], [c, b, d]))
    proof = ph._trans(p, t0, t3, t4, proof, next)
    next = ph._sym(p, t5, t4, ph._apps(p, algebra[2], [a, c, _join(p, j, b, d)]))
    proof = ph._trans(p, t0, t4, t5, proof, next)
    var binders = algebra.copy()
    for binder in [a, b, c, d]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, ph._eq(p, t0, t5)),
        ph._lams(p, binders, proof), EXEC_INTERCHANGE)


def _bit_union[C: Int](mut p: ph._Builder[C]):
    var algebra = _algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var value = ph._var(p, ph._nat(p))
    var at_a = _take(p, a, empty, value)
    var at_b = _take(p, b, empty, value)
    var result = ph._eq(p, _take(p, ph._select(p, a, b, ph._one(p)), empty, value),
        _join(p, j, at_a, at_b))
    var motive = ph._lam(p, a, ph._pi(p, b, result))
    var base = ph._lam(p, b, ph._sym(p, _join(p, j, empty, at_b), at_b,
        ph._app(p, algebra[4], at_b)))
    var ap = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, ap))
    var bn = ph._var(p, ph._nat(p))
    var inner_motive = ph._lam(p, bn, ph._eq(p, value,
        _join(p, j, value, _take(p, bn, empty, value))))
    var inner_base = ph._sym(p, _join(p, j, value, empty), value,
        ph._app(p, algebra[5], value))
    var bp = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, bp))
    var inner_step = ph._lams(p, [bp, inner_ih],
        ph._sym(p, _join(p, j, value, value), value, ph._app(p, algebra[6], value)))
    var step = ph._lams(p, [ap, ih, b],
        ph._induct(p, inner_motive, inner_base, inner_step, b))
    var binders = algebra.copy()
    binders.append(value)
    binders.append(a)
    ph._declare(p, ph._pis(p, binders, ph._pi(p, b, result)),
        ph._lams(p, binders, ph._induct(p, motive, base, step, a)), EXEC_BIT_UNION)


def _fold_extensional[C: Int](mut p: ph._Builder[C]):
    var algebra = _algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var left = ph._var(p, ph._store(p))
    var right = ph._var(p, ph._store(p))
    var at = ph._var(p, ph._nat(p))
    var equal = ph._var(p, ph._pi(p, at,
        ph._eq(p, ph._app(p, left, at), ph._app(p, right, at))))
    var n = ph._var(p, ph._nat(p))
    var result = ph._eq(p, _fold(p, j, empty, inputs, left, n),
        _fold(p, j, empty, inputs, right, n))
    var motive = ph._lam(p, n, result)
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var old_left = _fold(p, j, empty, inputs, left, k)
    var old_right = _fold(p, j, empty, inputs, right, k)
    var value = ph._app(p, inputs, k)
    var take_left = _take(p, ph._app(p, left, k), empty, value)
    var take_right = _take(p, ph._app(p, right, k), empty, value)
    var first = _left(p, j, old_left, old_right, ih, take_left)
    var flag = ph._var(p, ph._nat(p))
    var take_equal = ph._cong(p, ph._lam(p, flag, _take(p, flag, empty, value)),
        ph._app(p, left, k), ph._app(p, right, k), ph._app(p, equal, k))
    var second = _right(p, j, old_right, take_left, take_right, take_equal)
    var step = ph._lams(p, [k, ih], ph._trans(p,
        _join(p, j, old_left, take_left), _join(p, j, old_right, take_left),
        _join(p, j, old_right, take_right), first, second))
    var binders = algebra.copy()
    for binder in [inputs, left, right, equal, n]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, result),
        ph._lams(p, binders, ph._induct(p, motive, ph._refl(p, empty), step, n)),
        EXEC_FOLD_EXTENSIONAL)


def _fold_union[C: Int](mut p: ph._Builder[C]):
    var algebra = _algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var left = ph._var(p, ph._store(p))
    var right = ph._var(p, ph._store(p))
    var n = ph._var(p, ph._nat(p))
    var union = _union(p, left, right)
    var result = ph._eq(p, _fold(p, j, empty, inputs, union, n),
        _join(p, j, _fold(p, j, empty, inputs, left, n), _fold(p, j, empty, inputs, right, n)))
    var motive = ph._lam(p, n, result)
    var base = ph._sym(p, _join(p, j, empty, empty), empty, ph._app(p, algebra[4], empty))
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var a = _fold(p, j, empty, inputs, left, k)
    var b = _fold(p, j, empty, inputs, right, k)
    var c = _take(p, ph._app(p, left, k), empty, ph._app(p, inputs, k))
    var d = _take(p, ph._app(p, right, k), empty, ph._app(p, inputs, k))
    var taken_union = _take(p, ph._app(p, union, k), empty, ph._app(p, inputs, k))
    var t0 = _join(p, j, _fold(p, j, empty, inputs, union, k), taken_union)
    var t1 = _join(p, j, _join(p, j, a, b), taken_union)
    var t2 = _join(p, j, _join(p, j, a, b), _join(p, j, c, d))
    var t3 = _join(p, j, _join(p, j, a, c), _join(p, j, b, d))
    var first = _left(p, j, _fold(p, j, empty, inputs, union, k), _join(p, j, a, b), ih, taken_union)
    var bit = _call(p, EXEC_BIT_UNION, algebra,
        [ph._app(p, inputs, k), ph._app(p, left, k), ph._app(p, right, k)])
    var second = _right(p, j, _join(p, j, a, b), taken_union, _join(p, j, c, d), bit)
    var third = _call(p, EXEC_INTERCHANGE, algebra, [a, b, c, d])
    var step = ph._lams(p, [k, ih], ph._trans(p, t0, t2, t3,
        ph._trans(p, t0, t1, t2, first, second), third))
    var binders = algebra.copy()
    for binder in [inputs, left, right, n]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, result),
        ph._lams(p, binders, ph._induct(p, motive, base, step, n)), EXEC_FOLD_UNION)


def _zero_fold[C: Int](mut p: ph._Builder[C]):
    var algebra = _algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var at = ph._var(p, ph._nat(p))
    var zero_mask = ph._lam(p, at, ph._zero(p))
    var n = ph._var(p, ph._nat(p))
    var result = ph._eq(p, _fold(p, j, empty, inputs, zero_mask, n), empty)
    var motive = ph._lam(p, n, result)
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var old = _fold(p, j, empty, inputs, zero_mask, k)
    var first = _left(p, j, old, empty, ih, empty)
    var second = ph._app(p, algebra[5], empty)
    var step = ph._lams(p, [k, ih], ph._trans(p,
        _join(p, j, old, empty), _join(p, j, empty, empty), empty, first, second))
    var binders = algebra.copy()
    for binder in [inputs, n]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, result),
        ph._lams(p, binders, ph._induct(p, motive, ph._refl(p, empty), step, n)), EXEC_ZERO_FOLD)


def _full_fold[C: Int](mut p: ph._Builder[C]):
    var algebra = _algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var mask = ph._var(p, ph._store(p))
    var n = ph._var(p, ph._nat(p))
    var covered = ph._var(p, _prefix(p, mask, n))
    var result = ph._eq(p, _fold(p, j, empty, inputs, mask, n),
        ex.execution_flat_fold(p, j, empty, inputs, n))
    var motive = ph._lam(p, n, ph._pi(p, covered, result))
    var base_covered = ph._var(p, _prefix(p, mask, ph._zero(p)))
    var base = ph._lam(p, base_covered, ph._refl(p, empty))
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var next_covered = ph._var(p, _prefix(p, mask, ph._succ(p, k)))
    var components: List[Int] = [_prefix(p, mask, k), ph._eq(p, ph._app(p, mask, k), ph._one(p))]
    var old_covered = _project(p, next_covered, components, 0)
    var mask_equal = _project(p, next_covered, components, 1)
    var old = _fold(p, j, empty, inputs, mask, k)
    var canonical = ex.execution_flat_fold(p, j, empty, inputs, k)
    var value = ph._app(p, inputs, k)
    var selected = _take(p, ph._app(p, mask, k), empty, value)
    var first = _left(p, j, old, canonical, ph._app(p, ih, old_covered), selected)
    var bit = ph._var(p, ph._nat(p))
    var selected_equal = ph._cong(p, ph._lam(p, bit, _take(p, bit, empty, value)),
        ph._app(p, mask, k), ph._one(p), mask_equal)
    var second = _right(p, j, canonical, selected, value, selected_equal)
    var step = ph._lams(p, [k, ih, next_covered], ph._trans(p,
        _join(p, j, old, selected), _join(p, j, canonical, selected),
        _join(p, j, canonical, value), first, second))
    var binders = algebra.copy()
    for binder in [inputs, mask, n]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, ph._pi(p, covered, result)),
        ph._lams(p, binders, ph._induct(p, motive, base, step, n)), EXEC_FULL_FOLD)


def append_finishing_execution_algebra[C: Int](mut p: ph._Builder[C]):
    _data(p)
    _interchange(p)
    _bit_union(p)
    _fold_extensional(p)
    _fold_union(p)
    _zero_fold(p)
    _full_fold(p)


def make_finishing_execution_algebra_builder[capacity: Int = 8192]() -> ph._Builder[capacity]:
    var p = ex.make_finishing_execution_builder[capacity]()
    append_finishing_execution_algebra(p)
    p.bundle.root = ph._named(p, EXEC_FULL_FOLD)
    return p^
