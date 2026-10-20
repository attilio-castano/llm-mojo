"""Universal checked singleton-support fold certificates.

These ordinary Nat-induction terms relate the independent masked fold to one
input at an in-range source. The input function stays inside the outer source
induction motive, so successor sources may instantiate the IH with shifted
inputs. No finite source/count unrolling, literal oracle or leaf axiom is used.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.finishing_execution_terms as ex

comptime EXEC_LEAF_SHIFT = 42
comptime EXEC_ZERO_LEAF = 43
comptime EXEC_LEAF_CORRECT = 44
comptime EXEC_BEFORE_FLAG = 45
comptime EXEC_INITIAL_VALUES = 46
comptime EXEC_INITIAL_READY = 47
comptime EXEC_FLAG_BOUND_SOUND = 48
comptime EXEC_INITIAL_INVARIANT = 49
comptime EXEC_LEAF_DEFINITIONS = 50


def execution_shifted_inputs[C: Int](mut p: ph._Builder[C], inputs: Int) -> Int:
    var at = ph._var(p, ph._nat(p))
    return ph._lam(p, at, ph._app(p, inputs, ph._succ(p, at)))


def _leaf_shift[C: Int](mut p: ph._Builder[C]):
    var algebra = a._algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var source = ph._var(p, ph._nat(p))
    var n = ph._var(p, ph._nat(p))
    var shifted = execution_shifted_inputs(p, inputs)
    var source_next = ph._succ(p, source)
    var result = ph._eq(p,
        a._fold(p, j, empty, inputs, a._unit(p, source_next), ph._succ(p, n)),
        a._fold(p, j, empty, shifted, a._unit(p, source), n))
    var motive = ph._lam(p, n, result)
    var base = ph._app(p, algebra[5], empty)
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var old_left = a._fold(p, j, empty, inputs,
        a._unit(p, source_next), ph._succ(p, k))
    var old_right = a._fold(p, j, empty, shifted, a._unit(p, source), k)
    var value = ph._app(p, inputs, ph._succ(p, k))
    var taken = a._take(p, ph._app(p, a._unit(p, source), k), empty, value)
    var step = ph._lams(p, [k, ih],
        a._left(p, j, old_left, old_right, ih, taken))
    var binders = algebra.copy()
    for binder in [inputs, source, n]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, result),
        ph._lams(p, binders, ph._induct(p, motive, base, step, n)), EXEC_LEAF_SHIFT)


def _zero_leaf[C: Int](mut p: ph._Builder[C]):
    var algebra = a._algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var n = ph._var(p, ph._nat(p))
    var first = ph._app(p, inputs, ph._zero(p))
    var result = ph._eq(p,
        a._fold(p, j, empty, inputs, a._unit(p, ph._zero(p)), ph._succ(p, n)), first)
    var motive = ph._lam(p, n, result)
    var base = ph._app(p, algebra[4], first)
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var old = a._fold(p, j, empty, inputs,
        a._unit(p, ph._zero(p)), ph._succ(p, k))
    var carried = a._left(p, j, old, first, ih, empty)
    var reduced = ph._app(p, algebra[5], first)
    var step = ph._lams(p, [k, ih], ph._trans(p,
        a._join(p, j, old, empty), a._join(p, j, first, empty), first,
        carried, reduced))
    var binders = algebra.copy()
    for binder in [inputs, n]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, result),
        ph._lams(p, binders, ph._induct(p, motive, base, step, n)), EXEC_ZERO_LEAF)


def _leaf_type[C: Int](mut p: ph._Builder[C], j: Int,
    empty: Int, inputs: Int, source: Int, count: Int) -> Int:
    var before = ph._var(p, a._before(p, source, count))
    return ph._pi(p, before, ph._eq(p,
        a._fold(p, j, empty, inputs, a._unit(p, source), count),
        ph._app(p, inputs, source)))


def _leaf_correct[C: Int](mut p: ph._Builder[C]):
    var algebra = a._algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var source = ph._var(p, ph._nat(p))
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var outer_motive = ph._lam(p, source, ph._pis(p, [inputs, count],
        _leaf_type(p, j, empty, inputs, source, count)))

    # Source zero: no leaf fits extent zero; each successor extent is ZeroLeaf.
    inputs = ph._var(p, ph._store(p))
    count = ph._var(p, ph._nat(p))
    var inner_motive = ph._lam(p, count,
        _leaf_type(p, j, empty, inputs, ph._zero(p), count))
    var impossible = ph._var(p, a._before(p, ph._zero(p), ph._zero(p)))
    var zero_result = ph._eq(p,
        a._fold(p, j, empty, inputs, a._unit(p, ph._zero(p)), ph._zero(p)),
        ph._app(p, inputs, ph._zero(p)))
    var inner_base = ph._lam(p, impossible, ph._app(p, impossible, zero_result))
    var k = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, k))
    var before = ph._var(p, a._before(p, ph._zero(p), ph._succ(p, k)))
    var inner_step = ph._lams(p, [k, inner_ih, before],
        a._call(p, EXEC_ZERO_LEAF, algebra, [inputs, k]))
    var outer_base = ph._lams(p, [inputs, count],
        ph._induct(p, inner_motive, inner_base, inner_step, count))

    # Successor source: under a successor bound, shift inputs and apply the
    # source IH. Quantifying inputs inside that IH is essential here.
    var predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, predecessor))
    inputs = ph._var(p, ph._store(p))
    count = ph._var(p, ph._nat(p))
    var successor = ph._succ(p, predecessor)
    inner_motive = ph._lam(p, count,
        _leaf_type(p, j, empty, inputs, successor, count))
    impossible = ph._var(p, a._before(p, successor, ph._zero(p)))
    zero_result = ph._eq(p,
        a._fold(p, j, empty, inputs, a._unit(p, successor), ph._zero(p)),
        ph._app(p, inputs, successor))
    inner_base = ph._lam(p, impossible, ph._app(p, impossible, zero_result))
    k = ph._var(p, ph._nat(p))
    inner_ih = ph._var(p, ph._app(p, inner_motive, k))
    before = ph._var(p, a._before(p, successor, ph._succ(p, k)))
    var shifted = execution_shifted_inputs(p, inputs)
    var left = a._fold(p, j, empty, inputs, a._unit(p, successor), ph._succ(p, k))
    var middle = a._fold(p, j, empty, shifted, a._unit(p, predecessor), k)
    var right = ph._app(p, inputs, successor)
    var shift = a._call(p, EXEC_LEAF_SHIFT, algebra, [inputs, predecessor, k])
    var previous = ph._apps(p, outer_ih, [shifted, k, before])
    inner_step = ph._lams(p, [k, inner_ih, before],
        ph._trans(p, left, middle, right, shift, previous))
    var outer_step = ph._lams(p, [predecessor, outer_ih, inputs, count],
        ph._induct(p, inner_motive, inner_base, inner_step, count))

    source = ph._var(p, ph._nat(p))
    var binders = algebra.copy()
    binders.append(source)
    ph._declare(p, ph._pis(p, binders, ph._app(p, outer_motive, source)),
        ph._lams(p, binders,
            ph._induct(p, outer_motive, outer_base, outer_step, source)), EXEC_LEAF_CORRECT)


def execution_before_flag[C: Int](mut p: ph._Builder[C],
    source: Int, count: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_BEFORE_FLAG), [source, count])


def execution_initial_values[C: Int](mut p: ph._Builder[C],
    empty: Int, inputs: Int, count: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_INITIAL_VALUES), [empty, inputs, count])


def execution_initial_ready[C: Int](mut p: ph._Builder[C], count: Int) -> Int:
    return ph._app(p, ph._named(p, EXEC_INITIAL_READY), count)


def execution_support_type[C: Int](mut p: ph._Builder[C]) -> Int:
    var slot = ph._var(p, ph._nat(p))
    return ph._pi(p, slot, ph._store(p))


def execution_support_reference[C: Int](mut p: ph._Builder[C], j: Int,
    empty: Int, inputs: Int, support: Int, count: Int) -> Int:
    var slot = ph._var(p, ph._nat(p))
    return ph._lam(p, slot,
        a._fold(p, j, empty, inputs, ph._app(p, support, slot), count))


def _initial_data[C: Int](mut p: ph._Builder[C]):
    # The boolean reflection profile is a checked NatRec definition, independent
    # of native comparison. Its soundness theorem is definition 48 below.
    var source = ph._var(p, ph._nat(p))
    var count = ph._var(p, ph._nat(p))
    var k = ph._var(p, ph._nat(p))
    var previous = ph._var(p, ph._nat(p))
    var base = ph._lam(p, count, a._rec(p, count, ph._zero(p),
        ph._lams(p, [k, previous], ph._one(p))))
    var predecessor = ph._var(p, ph._nat(p))
    var recursive = ph._var(p, ph._store(p))
    count = ph._var(p, ph._nat(p))
    k = ph._var(p, ph._nat(p))
    previous = ph._var(p, ph._nat(p))
    var step = ph._lams(p, [predecessor, recursive, count],
        a._rec(p, count, ph._zero(p),
            ph._lams(p, [k, previous], ph._app(p, recursive, k))))
    ph._declare(p, ph._pi(p, source, ph._store(p)),
        ph._lam(p, source, a._rec(p, source, base, step)), EXEC_BEFORE_FLAG)

    var empty = ph._var(p, ph._nat(p))
    var inputs = ph._var(p, ph._store(p))
    count = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pis(p, [empty, inputs, count], ph._store(p)),
        ph._lams(p, [empty, inputs, count], ph._update(p, inputs, count, empty)),
        EXEC_INITIAL_VALUES)

    count = ph._var(p, ph._nat(p))
    var slot = ph._var(p, ph._nat(p))
    var flags = ph._lam(p, slot, execution_before_flag(p, slot, count))
    ph._declare(p, ph._pi(p, count, ph._store(p)),
        ph._lam(p, count, ph._update(p, flags, count, ph._one(p))), EXEC_INITIAL_READY)


def _flag_bound_type[C: Int](mut p: ph._Builder[C],
    source: Int, count: Int) -> Int:
    var flag = ph._var(p, ph._eq(p,
        execution_before_flag(p, source, count), ph._one(p)))
    return ph._pi(p, flag, a._before(p, source, count))


def _flag_bound_sound[C: Int](mut p: ph._Builder[C]):
    var source = ph._var(p, ph._nat(p))
    var count = ph._var(p, ph._nat(p))
    var outer_motive = ph._lam(p, source,
        ph._pi(p, count, _flag_bound_type(p, source, count)))

    count = ph._var(p, ph._nat(p))
    var inner_motive = ph._lam(p, count,
        _flag_bound_type(p, ph._zero(p), count))
    var flag = ph._var(p, ph._eq(p,
        execution_before_flag(p, ph._zero(p), ph._zero(p)), ph._one(p)))
    var inner_base = ph._lam(p, flag,
        ph._apps(p, ph._named(p, ph.PHASE_ZERO_SUCC), [ph._zero(p), flag]))
    var k = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, k))
    flag = ph._var(p, ph._eq(p,
        execution_before_flag(p, ph._zero(p), ph._succ(p, k)), ph._one(p)))
    var inner_step = ph._lams(p, [k, inner_ih, flag], ph._refl(p, ph._zero(p)))
    var outer_base = ph._lam(p, count,
        ph._induct(p, inner_motive, inner_base, inner_step, count))

    var predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, predecessor))
    var successor = ph._succ(p, predecessor)
    count = ph._var(p, ph._nat(p))
    inner_motive = ph._lam(p, count, _flag_bound_type(p, successor, count))
    flag = ph._var(p, ph._eq(p,
        execution_before_flag(p, successor, ph._zero(p)), ph._one(p)))
    inner_base = ph._lam(p, flag,
        ph._apps(p, ph._named(p, ph.PHASE_ZERO_SUCC), [ph._zero(p), flag]))
    k = ph._var(p, ph._nat(p))
    inner_ih = ph._var(p, ph._app(p, inner_motive, k))
    flag = ph._var(p, ph._eq(p,
        execution_before_flag(p, successor, ph._succ(p, k)), ph._one(p)))
    inner_step = ph._lams(p, [k, inner_ih, flag], ph._apps(p, outer_ih, [k, flag]))
    var outer_step = ph._lams(p, [predecessor, outer_ih, count],
        ph._induct(p, inner_motive, inner_base, inner_step, count))
    source = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, source, ph._app(p, outer_motive, source)),
        ph._lam(p, source,
            ph._induct(p, outer_motive, outer_base, outer_step, source)), EXEC_FLAG_BOUND_SOUND)


def _initial_invariant[C: Int](mut p: ph._Builder[C]):
    var algebra = a._algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var support = ph._var(p, execution_support_type(p))
    var at = ph._var(p, ph._nat(p))
    var empty_support = ph._var(p, ph._pi(p, at, ph._eq(p,
        ph._app(p, ph._app(p, support, count), at), ph._zero(p))))
    var source = ph._var(p, ph._nat(p))
    var before = ph._var(p, a._before(p, source, count))
    at = ph._var(p, ph._nat(p))
    var input_support = ph._var(p, ph._pis(p, [source, before, at], ph._eq(p,
        ph._app(p, ph._app(p, support, source), at), ph._app(p, a._unit(p, source), at))))
    var values = execution_initial_values(p, empty, inputs, count)
    var ready = execution_initial_ready(p, count)
    var reference = execution_support_reference(p, j, empty, inputs, support, count)
    var slot = ph._var(p, ph._nat(p))
    var observed_ready = ph._var(p, ph._eq(p, ph._app(p, ready, slot), ph._one(p)))
    var actual = ph._app(p, values, slot)
    var expected = ph._app(p, reference, slot)
    var result = ph._eq(p, actual, expected)

    # The sentinel at exactly N contains Empty and has the empty support.
    var equal = ph._var(p, ph._eq(p, slot, count))
    var actual_at_count = ph._app(p, values, count)
    var reference_at_count = ph._app(p, reference, count)
    var hit = ph._apps(p, ph._named(p, ph.PHASE_UPDATE_HIT), [inputs, count, empty])
    at = ph._var(p, ph._nat(p))
    var zero_mask = ph._lam(p, at, ph._zero(p))
    var empty_fold = a._fold(p, j, empty, inputs, zero_mask, count)
    var extensional = a._call(p, a.EXEC_FOLD_EXTENSIONAL, algebra,
        [inputs, ph._app(p, support, count), zero_mask, empty_support, count])
    var zero = a._call(p, a.EXEC_ZERO_FOLD, algebra, [inputs, count])
    var reference_is_empty = ph._trans(p, reference_at_count,
        empty_fold, empty, extensional, zero)
    var count_result = ph._trans(p, actual_at_count, empty, reference_at_count,
        hit, ph._sym(p, reference_at_count, empty, reference_is_empty))
    var observation = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, observation, ph._eq(p,
        ph._app(p, values, observation), ph._app(p, reference, observation)))
    var yes = ph._lam(p, equal,
        ph._subst(p, motive, ph._sym(p, slot, count, equal), count_result))

    # Away from the sentinel, readiness is BeforeFlag. Its checked soundness
    # gives the bound needed by the singleton fold theorem.
    var apart = ph._var(p, ph._apart(p, slot, count))
    var reverse_apart = ph._apart_sym(p, slot, count, apart)
    var value_miss = ph._apps(p, ph._named(p, ph.PHASE_UPDATE_MISS),
        [inputs, count, empty, slot, reverse_apart])
    observation = ph._var(p, ph._nat(p))
    var flags = ph._lam(p, observation, execution_before_flag(p, observation, count))
    var ready_miss = ph._apps(p, ph._named(p, ph.PHASE_UPDATE_MISS),
        [flags, count, ph._one(p), slot, reverse_apart])
    var flag_value = execution_before_flag(p, slot, count)
    var ready_at_slot = ph._app(p, ready, slot)
    var flag_equal = ph._trans(p, flag_value, ready_at_slot, ph._one(p),
        ph._sym(p, ready_at_slot, flag_value, ready_miss), observed_ready)
    var bound = ph._apps(p, ph._named(p, EXEC_FLAG_BOUND_SOUND), [slot, count, flag_equal])
    var pointwise_support = ph._apps(p, input_support, [slot, bound])
    var unit_fold = a._fold(p, j, empty, inputs, a._unit(p, slot), count)
    var extensional_input = a._call(p, a.EXEC_FOLD_EXTENSIONAL, algebra,
        [inputs, ph._app(p, support, slot), a._unit(p, slot), pointwise_support, count])
    var leaf = a._call(p, EXEC_LEAF_CORRECT, algebra, [slot, inputs, count, bound])
    var input_value = ph._app(p, inputs, slot)
    var reference_is_input = ph._trans(p, expected, unit_fold, input_value,
        extensional_input, leaf)
    var no = ph._lam(p, apart, ph._trans(p, actual, input_value, expected,
        value_miss, ph._sym(p, expected, input_value, reference_is_input)))
    var body = ph._apps(p, ph._named(p, ph.PHASE_DECIDE), [slot, count, result, yes, no])
    body = ph._lams(p, [slot, observed_ready], body)
    var binders = algebra.copy()
    for binder in [inputs, count, support, empty_support, input_support]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders,
        ex.execution_initial_invariant(p, values, ready, reference)),
        ph._lams(p, binders, body), EXEC_INITIAL_INVARIANT)


def append_finishing_execution_leaves[C: Int](mut p: ph._Builder[C]):
    if p.bundle.definition_count != a.EXEC_ALGEBRA_DEFINITIONS:
        p.bundle.count = -1
        return
    _leaf_shift(p)
    _zero_leaf(p)
    _leaf_correct(p)
    _initial_data(p)
    _flag_bound_sound(p)
    _initial_invariant(p)


def make_finishing_execution_leaves_builder[
    capacity: Int = 8192,
]() -> ph._Builder[capacity]:
    var p = a.make_finishing_execution_algebra_builder[capacity]()
    append_finishing_execution_leaves(p)
    p.bundle.root = ph._named(p, EXEC_INITIAL_INVARIANT)
    return p^
