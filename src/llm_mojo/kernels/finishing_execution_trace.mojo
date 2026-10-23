"""Checked structural provenance turns complete traces into a canonical join.

Legality is readiness, fresh writes and pointwise support union, independent of
payload values. Initialization is derived from input/Empty support equations;
the root equation follows from complete input support. Neither reference-value
preservation nor the final result is smuggled into the definition of legality.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_terms as ex
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.finishing_execution_leaves as leaf

comptime EXEC_SUPPORT_STEP = 50
comptime EXEC_SUPPORT_PREFIX = 51
comptime EXEC_STEP_REFINEMENT = 52
comptime EXEC_PREFIX_REFINEMENT = 53
comptime EXEC_CANONICAL_TRACE = 54
comptime EXEC_TRACE_DEFINITIONS = 55
comptime FINISHING_EXECUTION_TRACE_CAPACITY = 8192


@fieldwise_init
struct FinishingExecutionProof[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var canonical_root: Int
    var prefix_refinement_root: Int
    var initialization_root: Int
    var fold_union_root: Int
    var full_fold_root: Int
    var leaf_root: Int


def support_step_propositions[C: Int](mut p: ph._Builder[C], initial_ready: Int,
    support: Int, destinations: Int, lefts: Int, rights: Int, step: Int) -> List[Int]:
    var ready = ex.execution_readiness(p, initial_ready, destinations, step)
    var left = ph._app(p, lefts, step)
    var right = ph._app(p, rights, step)
    var destination = ph._app(p, destinations, step)
    var at = ph._var(p, ph._nat(p))
    var mask = ph._app(p, support, destination)
    var union = a._union(p, ph._app(p, support, left), ph._app(p, support, right))
    return [ph._eq(p, ph._app(p, ready, left), ph._one(p)),
        ph._eq(p, ph._app(p, ready, right), ph._one(p)),
        ph._eq(p, ph._app(p, ready, destination), ph._zero(p)),
        ph._pi(p, at, ph._eq(p, ph._app(p, mask, at), ph._app(p, union, at)))]


def support_step[C: Int](mut p: ph._Builder[C], initial_ready: Int,
    support: Int, destinations: Int, lefts: Int, rights: Int, step: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_SUPPORT_STEP),
        [initial_ready, support, destinations, lefts, rights, step])


def support_prefix[C: Int](mut p: ph._Builder[C], initial_ready: Int,
    support: Int, destinations: Int, lefts: Int, rights: Int, count: Int) -> Int:
    return ph._apps(p, ph._named(p, EXEC_SUPPORT_PREFIX),
        [initial_ready, support, destinations, lefts, rights, count])


def _data[C: Int](mut p: ph._Builder[C]):
    var initial = ph._var(p, ph._store(p))
    var support = ph._var(p, leaf.execution_support_type(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var n = ph._var(p, ph._nat(p))
    var binders: List[Int] = [initial, support, destinations, lefts, rights, n]
    ph._declare(p, ph._pis(p, binders, ph._prop(p)),
        ph._lams(p, binders, a._and_type(p,
            support_step_propositions(p, initial, support, destinations, lefts, rights, n))),
        EXEC_SUPPORT_STEP)
    initial = ph._var(p, ph._store(p))
    support = ph._var(p, leaf.execution_support_type(p))
    destinations = ph._var(p, ph._store(p))
    lefts = ph._var(p, ph._store(p))
    rights = ph._var(p, ph._store(p))
    n = ph._var(p, ph._nat(p))
    var k = ph._var(p, ph._nat(p))
    var previous = ph._var(p, ph._prop(p))
    var step = ph._lams(p, [k, previous], a._and_type(p,
        [previous, support_step(p, initial, support, destinations, lefts, rights, k)]))
    binders = [initial, support, destinations, lefts, rights, n]
    ph._declare(p, ph._pis(p, binders, ph._prop(p)),
        ph._lams(p, binders, a._rec(p, n,
            ph._eq(p, ph._zero(p), ph._zero(p)), step)), EXEC_SUPPORT_PREFIX)


def _step_refinement[C: Int](mut p: ph._Builder[C]):
    var algebra = a._algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var support = ph._var(p, leaf.execution_support_type(p))
    var initial = ph._var(p, ph._store(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var k = ph._var(p, ph._nat(p))
    var structural = ph._var(p, support_step(p, initial, support, destinations, lefts, rights, k))
    var reference = leaf.execution_support_reference(p, j, empty, inputs, support, count)
    var before = support_step_propositions(p, initial, support, destinations, lefts, rights, k)
    var after = ex.execution_step_propositions(p, j, reference, initial, destinations, lefts, rights, k)
    var destination_mask = ph._app(p, support, ph._app(p, destinations, k))
    var left_mask = ph._app(p, support, ph._app(p, lefts, k))
    var right_mask = ph._app(p, support, ph._app(p, rights, k))
    var union = a._union(p, left_mask, right_mask)
    var support_equation = a._project(p, structural, before, 3)
    var first = a._call(p, a.EXEC_FOLD_EXTENSIONAL, algebra,
        [inputs, destination_mask, union, support_equation, count])
    var second = a._call(p, a.EXEC_FOLD_UNION, algebra, [inputs, left_mask, right_mask, count])
    var lhs = a._fold(p, j, empty, inputs, destination_mask, count)
    var middle = a._fold(p, j, empty, inputs, union, count)
    var rhs = a._join(p, j, a._fold(p, j, empty, inputs, left_mask, count),
        a._fold(p, j, empty, inputs, right_mask, count))
    var local_equation = ph._trans(p, lhs, middle, rhs, first, second)
    var evidence: List[Int] = [a._project(p, structural, before, 0),
        a._project(p, structural, before, 1), a._project(p, structural, before, 2), local_equation]
    var body = a._and_intro(p, after, evidence)
    var result = ex.execution_step_legal(p, j, reference, initial, destinations, lefts, rights, k)
    var binders = algebra.copy()
    for binder in [inputs, count, support, initial, destinations, lefts, rights, k, structural]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body), EXEC_STEP_REFINEMENT)


def _prefix_refinement[C: Int](mut p: ph._Builder[C]):
    var algebra = a._algebra(p)
    var j = algebra[0]
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var support = ph._var(p, leaf.execution_support_type(p))
    var initial = ph._var(p, ph._store(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var reference = leaf.execution_support_reference(p, j, algebra[1], inputs, support, count)
    var n = ph._var(p, ph._nat(p))
    var structural = ph._var(p, support_prefix(p, initial, support, destinations, lefts, rights, n))
    var result = ex.execution_legal_prefix(p, j, reference, initial, destinations, lefts, rights, n)
    var motive = ph._lam(p, n, ph._pi(p, structural, result))
    var base_guard = ph._var(p, support_prefix(p, initial, support, destinations, lefts, rights, ph._zero(p)))
    var base = ph._lam(p, base_guard, ph._refl(p, ph._zero(p)))
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var next_guard = ph._var(p, support_prefix(p, initial, support, destinations, lefts, rights, ph._succ(p, k)))
    var before: List[Int] = [support_prefix(p, initial, support, destinations, lefts, rights, k),
        support_step(p, initial, support, destinations, lefts, rights, k)]
    var prior_guard = a._project(p, next_guard, before, 0)
    var step_guard = a._project(p, next_guard, before, 1)
    var prior = ph._app(p, ih, prior_guard)
    var current = a._call(p, EXEC_STEP_REFINEMENT, algebra,
        [inputs, count, support, initial, destinations, lefts, rights, k, step_guard])
    var after: List[Int] = [ex.execution_legal_prefix(p, j, reference, initial, destinations, lefts, rights, k),
        ex.execution_step_legal(p, j, reference, initial, destinations, lefts, rights, k)]
    var step = ph._lams(p, [k, ih, next_guard], a._and_intro(p, after, [prior, current]))
    var binders = algebra.copy()
    for binder in [inputs, count, support, initial, destinations, lefts, rights, n]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, ph._pi(p, structural, result)),
        ph._lams(p, binders, ph._induct(p, motive, base, step, n)), EXEC_PREFIX_REFINEMENT)


def _canonical[C: Int](mut p: ph._Builder[C]):
    var algebra = a._algebra(p)
    var j = algebra[0]
    var empty = algebra[1]
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var support = ph._var(p, leaf.execution_support_type(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var at = ph._var(p, ph._nat(p))
    var empty_support = ph._var(p, ph._pi(p, at, ph._eq(p,
        ph._app(p, ph._app(p, support, count), at), ph._zero(p))))
    var source = ph._var(p, ph._nat(p))
    var before = ph._var(p, a._before(p, source, count))
    at = ph._var(p, ph._nat(p))
    var input_support = ph._var(p, ph._pis(p, [source, before, at], ph._eq(p,
        ph._app(p, ph._app(p, support, source), at), ph._app(p, a._unit(p, source), at))))
    var initial_values = leaf.execution_initial_values(p, empty, inputs, count)
    var initial_ready = leaf.execution_initial_ready(p, count)
    var reference = leaf.execution_support_reference(p, j, empty, inputs, support, count)
    var n = ph._var(p, ph._nat(p))
    var structural = ph._var(p, support_prefix(p, initial_ready, support, destinations, lefts, rights, n))
    var root = ph._var(p, ph._nat(p))
    var ready = ph._var(p, ph._eq(p, ph._app(p,
        ex.execution_readiness(p, initial_ready, destinations, n), root), ph._one(p)))
    var complete = ph._var(p, a._prefix(p, ph._app(p, support, root), count))
    var initial = a._call(p, leaf.EXEC_INITIAL_INVARIANT, algebra,
        [inputs, count, support, empty_support, input_support])
    var legal = a._call(p, EXEC_PREFIX_REFINEMENT, algebra,
        [inputs, count, support, initial_ready, destinations, lefts, rights, n, structural])
    var root_equation = a._call(p, a.EXEC_FULL_FOLD, algebra,
        [inputs, ph._app(p, support, root), count, complete])
    var actual = ph._app(p, ex.execution_values(p, j, initial_values, destinations, lefts, rights, n), root)
    var expected = ex.execution_flat_fold(p, j, empty, inputs, count)
    var body = ph._apps(p, ph._named(p, ex.EXECUTION_COMPLETION),
        [j, initial_values, initial_ready, reference, destinations, lefts, rights,
            initial, n, legal, root, ready, empty, inputs, count, root_equation])
    var binders = algebra.copy()
    for binder in [inputs, count, support, destinations, lefts, rights,
        empty_support, input_support, n, structural, root, ready, complete]:
        binders.append(binder)
    ph._declare(p, ph._pis(p, binders, ph._eq(p, actual, expected)),
        ph._lams(p, binders, body), EXEC_CANONICAL_TRACE)


def append_finishing_execution_trace[C: Int](mut p: ph._Builder[C]):
    _data(p)
    _step_refinement(p)
    _prefix_refinement(p)
    _canonical(p)


def make_finishing_execution_trace_builder[
    capacity: Int = FINISHING_EXECUTION_TRACE_CAPACITY,
]() -> ph._Builder[capacity]:
    # Construct once and append in place. Nesting the large fixed-capacity
    # builder return values can exhaust a normal O0 native process stack before
    # admission starts; this changes neither the rules nor the theorem.
    var bundle = tk.empty_proof_terms[capacity]()
    var nodes = Dict[Tuple[Int, Int, Int, Int, Int], Int]()
    for index in range(bundle.count):
        var node = bundle.nodes[index]
        if node.kind != tk.PT_VAR:
            nodes[(node.kind, node.left, node.right, node.extra, node.value)] = index
    var p = ph._Builder[capacity](bundle^, nodes^)
    ph._data(p)
    ph._no_confusion(p)
    ph._eqb_refl(p)
    ph._decide(p)
    ph._apart_to_eqb(p)
    ph._eqb_to_apart(p)
    ph._updates(p)
    ph._swap(p)
    ph._pending_and_ready(p)
    ph._phase_roots(p)
    ex.append_finishing_execution_core(p)
    a.append_finishing_execution_algebra(p)
    leaf.append_finishing_execution_leaves(p)
    append_finishing_execution_trace(p)
    p.bundle.root = ph._named(p, EXEC_CANONICAL_TRACE)
    return p^


def make_finishing_execution_proof[
    capacity: Int = FINISHING_EXECUTION_TRACE_CAPACITY,
]() -> FinishingExecutionProof[capacity]:
    var p = make_finishing_execution_trace_builder[capacity]()
    var prefix = ph._named(p, EXEC_PREFIX_REFINEMENT)
    var initialization = ph._named(p, leaf.EXEC_INITIAL_INVARIANT)
    var union = ph._named(p, a.EXEC_FOLD_UNION)
    var full = ph._named(p, a.EXEC_FULL_FOLD)
    var singleton = ph._named(p, leaf.EXEC_LEAF_CORRECT)
    return FinishingExecutionProof[capacity](p.bundle.copy(), p.bundle.root,
        prefix, initialization, union, full, singleton)
