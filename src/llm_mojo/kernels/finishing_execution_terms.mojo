"""Untrusted, shape-independent merge-machine execution proof terms.

The machine stores abstract Nat-valued records. A step reads two ready cells,
joins their values, and writes one fresh destination. Values and readiness are
actual NatRec programs; LegalPrefix records the operational premises, rather
than defining legality to mean the desired result. Checked Nat induction proves
the reference invariant for every trace length, store and schedule.

The final theorem requires a proof that the designated reference equals the
independently defined flat input fold. This module does not grant that premise.
The algebra/coverage layer must derive it before a complete executor export is
admitted. Native records, address decoding and source execution refinement are
separate interpretation obligations.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph

comptime EXECUTION_PROOF_CAPACITY = 8192
comptime EXECUTION_VALUES = 23
comptime EXECUTION_READINESS = 24
comptime EXECUTION_STEP_LEGAL = 25
comptime EXECUTION_LEGAL_PREFIX = 26
comptime EXECUTION_FLAT_FOLD = 27
comptime EXECUTION_INVARIANT = 28
comptime EXECUTION_COMPLETION = 29
comptime EXECUTION_DEFINITIONS = 30


@fieldwise_init
struct ExecutionProof[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var values_root: Int
    var readiness_root: Int
    var step_legal_root: Int
    var legal_prefix_root: Int
    var fold_root: Int
    var invariant_root: Int
    var completion_root: Int


def execution_join_type[C: Int](mut p: ph._Builder[C]) -> Int:
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    return ph._pis(p, [a, b], ph._nat(p))


def execution_values[C: Int](mut p: ph._Builder[C], join: Int, initial: Int,
    destinations: Int, lefts: Int, rights: Int, steps: Int) -> Int:
    return ph._apps(p, ph._named(p, EXECUTION_VALUES),
        [join, initial, destinations, lefts, rights, steps])


def execution_readiness[C: Int](mut p: ph._Builder[C], initial: Int,
    destinations: Int, steps: Int) -> Int:
    return ph._apps(p, ph._named(p, EXECUTION_READINESS),
        [initial, destinations, steps])


def execution_step_legal[C: Int](mut p: ph._Builder[C], join: Int,
    reference: Int, initial_ready: Int, destinations: Int, lefts: Int,
    rights: Int, step: Int) -> Int:
    return ph._apps(p, ph._named(p, EXECUTION_STEP_LEGAL),
        [join, reference, initial_ready, destinations, lefts, rights, step])


def execution_legal_prefix[C: Int](mut p: ph._Builder[C], join: Int,
    reference: Int, initial_ready: Int, destinations: Int, lefts: Int,
    rights: Int, steps: Int) -> Int:
    return ph._apps(p, ph._named(p, EXECUTION_LEGAL_PREFIX),
        [join, reference, initial_ready, destinations, lefts, rights, steps])


def execution_flat_fold[C: Int](mut p: ph._Builder[C], join: Int,
    empty: Int, inputs: Int, count: Int) -> Int:
    return ph._apps(p, ph._named(p, EXECUTION_FLAT_FOLD),
        [join, empty, inputs, count])


def execution_initial_invariant[C: Int](mut p: ph._Builder[C],
    initial_values: Int, initial_ready: Int, reference: Int) -> Int:
    var slot = ph._var(p, ph._nat(p))
    var ready = ph._var(p, ph._eq(p,
        ph._app(p, initial_ready, slot), ph._one(p)))
    return ph._pis(p, [slot, ready], ph._eq(p,
        ph._app(p, initial_values, slot), ph._app(p, reference, slot)))


def execution_step_propositions[C: Int](mut p: ph._Builder[C], join: Int,
    reference: Int, initial_ready: Int, destinations: Int, lefts: Int,
    rights: Int, step: Int) -> List[Int]:
    var ready = execution_readiness(p, initial_ready, destinations, step)
    var left = ph._app(p, lefts, step)
    var right = ph._app(p, rights, step)
    var destination = ph._app(p, destinations, step)
    return [
        ph._eq(p, ph._app(p, ready, left), ph._one(p)),
        ph._eq(p, ph._app(p, ready, right), ph._one(p)),
        ph._eq(p, ph._app(p, ready, destination), ph._zero(p)),
        ph._eq(p, ph._app(p, reference, destination),
            ph._apps(p, join, [ph._app(p, reference, left),
                ph._app(p, reference, right)])),
    ]


def execution_project[C: Int](mut p: ph._Builder[C],
    propositions: List[Int], evidence: Int, selected: Int) -> Int:
    """Ordinary elimination of the Church conjunction, not a kernel rule."""
    if selected < 0 or selected >= len(propositions):
        p.bundle.count = -1
        return -1
    var binders = List[Int]()
    for proposition in propositions:
        binders.append(ph._var(p, proposition))
    var choose = ph._lams(p, binders, binders[selected])
    return ph._apps(p, evidence, [propositions[selected], choose])


def _execution_data[C: Int](mut p: ph._Builder[C]):
    # Values(J,S0,D,L,R,n) is the actual sequential merge execution.
    var join = ph._var(p, execution_join_type(p))
    var initial = ph._var(p, ph._store(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var steps = ph._var(p, ph._nat(p))
    var k = ph._var(p, ph._nat(p))
    var state = ph._var(p, ph._store(p))
    var value = ph._apps(p, join, [
        ph._app(p, state, ph._app(p, lefts, k)),
        ph._app(p, state, ph._app(p, rights, k))])
    var step = ph._lams(p, [k, state],
        ph._update(p, state, ph._app(p, destinations, k), value))
    var body = ph._node(p, tk.PT_NAT_REC, steps, initial, step)
    var binders: List[Int] = [join, initial, destinations, lefts, rights, steps]
    ph._declare(p, ph._pis(p, binders, ph._store(p)),
        ph._lams(p, binders, body), EXECUTION_VALUES)

    # Readiness(P0,D,n) is a separate concrete write-one execution.
    initial = ph._var(p, ph._store(p))
    destinations = ph._var(p, ph._store(p))
    steps = ph._var(p, ph._nat(p))
    k = ph._var(p, ph._nat(p))
    state = ph._var(p, ph._store(p))
    step = ph._lams(p, [k, state],
        ph._update(p, state, ph._app(p, destinations, k), ph._one(p)))
    body = ph._node(p, tk.PT_NAT_REC, steps, initial, step)
    binders = [initial, destinations, steps]
    ph._declare(p, ph._pis(p, binders, ph._store(p)),
        ph._lams(p, binders, body), EXECUTION_READINESS)

    # A step requires ready operands, a fresh destination, and its local DAG
    # reference equation. It asserts neither a prefix invariant nor final output.
    join = ph._var(p, execution_join_type(p))
    var reference = ph._var(p, ph._store(p))
    initial = ph._var(p, ph._store(p))
    destinations = ph._var(p, ph._store(p))
    lefts = ph._var(p, ph._store(p))
    rights = ph._var(p, ph._store(p))
    k = ph._var(p, ph._nat(p))
    binders = [join, reference, initial, destinations, lefts, rights, k]
    body = ph._and4(p, execution_step_propositions(p,
        join, reference, initial, destinations, lefts, rights, k))
    ph._declare(p, ph._pis(p, binders, ph._prop(p)),
        ph._lams(p, binders, body), EXECUTION_STEP_LEGAL)

    # LegalPrefix(0)=True; LegalPrefix(S k)=LegalPrefix(k) and StepLegal(k).
    join = ph._var(p, execution_join_type(p))
    reference = ph._var(p, ph._store(p))
    initial = ph._var(p, ph._store(p))
    destinations = ph._var(p, ph._store(p))
    lefts = ph._var(p, ph._store(p))
    rights = ph._var(p, ph._store(p))
    steps = ph._var(p, ph._nat(p))
    k = ph._var(p, ph._nat(p))
    var previous = ph._var(p, ph._prop(p))
    step = ph._lams(p, [k, previous], ph._and4(p,
        [previous, execution_step_legal(p,
            join, reference, initial, destinations, lefts, rights, k)]))
    body = ph._node(p, tk.PT_NAT_REC, steps,
        ph._eq(p, ph._zero(p), ph._zero(p)), step)
    binders = [join, reference, initial, destinations, lefts, rights, steps]
    ph._declare(p, ph._pis(p, binders, ph._prop(p)),
        ph._lams(p, binders, body), EXECUTION_LEGAL_PREFIX)

    # An independent sequential specification: Fold(0)=Empty,
    # Fold(S k)=Join(Fold(k), Input(k)). No reduction-tree reference occurs here.
    join = ph._var(p, execution_join_type(p))
    var empty = ph._var(p, ph._nat(p))
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    k = ph._var(p, ph._nat(p))
    var result = ph._var(p, ph._nat(p))
    step = ph._lams(p, [k, result],
        ph._apps(p, join, [result, ph._app(p, inputs, k)]))
    body = ph._node(p, tk.PT_NAT_REC, count, empty, step)
    binders = [join, empty, inputs, count]
    ph._declare(p, ph._pis(p, binders, ph._nat(p)),
        ph._lams(p, binders, body), EXECUTION_FLAT_FOLD)


def _invariant_at[C: Int](mut p: ph._Builder[C], join: Int,
    initial_values: Int, initial_ready: Int, reference: Int,
    destinations: Int, lefts: Int, rights: Int, steps: Int) -> Int:
    var legal = ph._var(p, execution_legal_prefix(p, join, reference,
        initial_ready, destinations, lefts, rights, steps))
    var slot = ph._var(p, ph._nat(p))
    var ready = ph._var(p, ph._eq(p, ph._app(p,
        execution_readiness(p, initial_ready, destinations, steps), slot), ph._one(p)))
    var actual = ph._app(p, execution_values(p, join, initial_values,
        destinations, lefts, rights, steps), slot)
    return ph._pis(p, [legal, slot, ready],
        ph._eq(p, actual, ph._app(p, reference, slot)))


def _execution_invariant[C: Int](mut p: ph._Builder[C]):
    var join = ph._var(p, execution_join_type(p))
    var initial_values = ph._var(p, ph._store(p))
    var initial_ready = ph._var(p, ph._store(p))
    var reference = ph._var(p, ph._store(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var initial_invariant = ph._var(p, execution_initial_invariant(p,
        initial_values, initial_ready, reference))
    var n = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, n, _invariant_at(p, join, initial_values,
        initial_ready, reference, destinations, lefts, rights, n))

    var legal_zero = ph._var(p, execution_legal_prefix(p, join, reference,
        initial_ready, destinations, lefts, rights, ph._zero(p)))
    var x = ph._var(p, ph._nat(p))
    var ready_zero = ph._var(p, ph._eq(p,
        ph._app(p, initial_ready, x), ph._one(p)))
    var base = ph._lams(p, [legal_zero, x, ready_zero],
        ph._apps(p, initial_invariant, [x, ready_zero]))

    n = ph._var(p, ph._nat(p))
    var ih = ph._var(p, _invariant_at(p, join, initial_values,
        initial_ready, reference, destinations, lefts, rights, n))
    var legal_next = ph._var(p, execution_legal_prefix(p, join, reference,
        initial_ready, destinations, lefts, rights, ph._succ(p, n)))
    x = ph._var(p, ph._nat(p))
    var previous_ready = execution_readiness(p, initial_ready, destinations, n)
    var destination = ph._app(p, destinations, n)
    var next_ready = ph._update(p, previous_ready, destination, ph._one(p))
    var ready_next = ph._var(p, ph._eq(p,
        ph._app(p, next_ready, x), ph._one(p)))
    var prefix_propositions: List[Int] = [
        execution_legal_prefix(p, join, reference,
            initial_ready, destinations, lefts, rights, n),
        execution_step_legal(p, join, reference,
            initial_ready, destinations, lefts, rights, n),
    ]
    var legal_previous = execution_project(p, prefix_propositions, legal_next, 0)
    var local_legal = execution_project(p, prefix_propositions, legal_next, 1)
    var local_propositions = execution_step_propositions(p,
        join, reference, initial_ready, destinations, lefts, rights, n)
    var ready_left = execution_project(p, local_propositions, local_legal, 0)
    var ready_right = execution_project(p, local_propositions, local_legal, 1)
    var reference_equation = execution_project(p, local_propositions, local_legal, 3)
    var left = ph._app(p, lefts, n)
    var right = ph._app(p, rights, n)
    var previous_values = execution_values(p, join, initial_values,
        destinations, lefts, rights, n)
    var actual_left = ph._app(p, previous_values, left)
    var actual_right = ph._app(p, previous_values, right)
    var reference_left = ph._app(p, reference, left)
    var reference_right = ph._app(p, reference, right)
    var left_equation = ph._apps(p, ih, [legal_previous, left, ready_left])
    var right_equation = ph._apps(p, ih, [legal_previous, right, ready_right])
    var joined = ph._apps(p, join, [actual_left, actual_right])
    var middle = ph._apps(p, join, [reference_left, actual_right])
    var reference_joined = ph._apps(p, join, [reference_left, reference_right])
    var v = ph._var(p, ph._nat(p))
    var first = ph._cong(p, ph._lam(p, v,
        ph._apps(p, join, [v, actual_right])),
        actual_left, reference_left, left_equation)
    v = ph._var(p, ph._nat(p))
    var second = ph._cong(p, ph._lam(p, v,
        ph._apps(p, join, [reference_left, v])),
        actual_right, reference_right, right_equation)
    var joined_reference = ph._trans(p, joined, middle, reference_joined, first, second)
    var reference_destination = ph._app(p, reference, destination)
    var value_reference = ph._trans(p, joined, reference_joined,
        reference_destination, joined_reference,
        ph._sym(p, reference_destination, reference_joined, reference_equation))
    var next_values = ph._update(p, previous_values, destination, joined)
    var actual = ph._app(p, next_values, x)
    var result_type = ph._eq(p, actual, ph._app(p, reference, x))

    # Case destination=x: the checked update-hit equation writes the joined
    # value, and six ordinary equality transports establish its reference.
    var equal = ph._var(p, ph._eq(p, destination, x))
    var at_destination = ph._app(p, next_values, destination)
    var hit = ph._apps(p, ph._named(p, ph.PHASE_UPDATE_HIT),
        [previous_values, destination, joined])
    var destination_result = ph._trans(p, at_destination, joined,
        reference_destination, hit, value_reference)
    var observation = ph._var(p, ph._nat(p))
    var observation_motive = ph._lam(p, observation,
        ph._eq(p, ph._app(p, next_values, observation),
            ph._app(p, reference, observation)))
    var yes = ph._lam(p, equal,
        ph._subst(p, observation_motive, equal, destination_result))

    # Case destination!=x: both value and ready stores retain their old cells.
    # Transport the observed new readiness back before applying the IH.
    var apart = ph._var(p, ph._apart(p, destination, x))
    var ready_miss = ph._apps(p, ph._named(p, ph.PHASE_UPDATE_MISS),
        [previous_ready, destination, ph._one(p), x, apart])
    var old_ready_at_x = ph._app(p, previous_ready, x)
    var new_ready_at_x = ph._app(p, next_ready, x)
    var old_ready = ph._trans(p, old_ready_at_x, new_ready_at_x, ph._one(p),
        ph._sym(p, new_ready_at_x, old_ready_at_x, ready_miss), ready_next)
    var old_reference = ph._apps(p, ih, [legal_previous, x, old_ready])
    var miss = ph._apps(p, ph._named(p, ph.PHASE_UPDATE_MISS),
        [previous_values, destination, joined, x, apart])
    var no = ph._lam(p, apart, ph._trans(p, actual,
        ph._app(p, previous_values, x), ph._app(p, reference, x),
        miss, old_reference))
    var body = ph._apps(p, ph._named(p, ph.PHASE_DECIDE),
        [destination, x, result_type, yes, no])
    var step = ph._lams(p, [n, ih, legal_next, x, ready_next], body)
    var steps = ph._var(p, ph._nat(p))
    body = ph._induct(p, motive, base, step, steps)
    var binders: List[Int] = [join, initial_values, initial_ready, reference,
        destinations, lefts, rights, initial_invariant, steps]
    ph._declare(p, ph._pis(p, binders, _invariant_at(p, join, initial_values,
        initial_ready, reference, destinations, lefts, rights, steps)),
        ph._lams(p, binders, body), EXECUTION_INVARIANT)


def _execution_completion[C: Int](mut p: ph._Builder[C]):
    var join = ph._var(p, execution_join_type(p))
    var initial_values = ph._var(p, ph._store(p))
    var initial_ready = ph._var(p, ph._store(p))
    var reference = ph._var(p, ph._store(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var initial_invariant = ph._var(p, execution_initial_invariant(p,
        initial_values, initial_ready, reference))
    var steps = ph._var(p, ph._nat(p))
    var legal = ph._var(p, execution_legal_prefix(p, join, reference,
        initial_ready, destinations, lefts, rights, steps))
    var root = ph._var(p, ph._nat(p))
    var ready = ph._var(p, ph._eq(p, ph._app(p,
        execution_readiness(p, initial_ready, destinations, steps), root), ph._one(p)))
    var empty = ph._var(p, ph._nat(p))
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var expected = execution_flat_fold(p, join, empty, inputs, count)
    var reference_value = ph._app(p, reference, root)
    var root_equation = ph._var(p, ph._eq(p, reference_value, expected))
    var actual = ph._app(p, execution_values(p, join, initial_values,
        destinations, lefts, rights, steps), root)
    var invariant = ph._apps(p, ph._named(p, EXECUTION_INVARIANT),
        [join, initial_values, initial_ready, reference, destinations, lefts,
            rights, initial_invariant, steps, legal, root, ready])
    var body = ph._trans(p, actual, reference_value, expected, invariant, root_equation)
    var binders: List[Int] = [join, initial_values, initial_ready, reference,
        destinations, lefts, rights, initial_invariant, steps, legal, root,
        ready, empty, inputs, count, root_equation]
    ph._declare(p, ph._pis(p, binders, ph._eq(p, actual, expected)),
        ph._lams(p, binders, body), EXECUTION_COMPLETION)


def append_finishing_execution_core[C: Int](mut p: ph._Builder[C]):
    """Append checked-data proposals and proofs to the unchanged phase prefix."""
    if p.bundle.definition_count != ph.PHASE_DEFINITIONS:
        p.bundle.count = -1
        return
    _execution_data(p)
    _execution_invariant(p)
    _execution_completion(p)


def make_finishing_execution_builder[
    capacity: Int = EXECUTION_PROOF_CAPACITY,
]() -> ph._Builder[capacity]:
    var phase = ph.make_finishing_phase_proof[capacity](0, 1)
    var nodes = Dict[Tuple[Int, Int, Int, Int, Int], Int]()
    for j in range(phase.bundle.count):
        var n = phase.bundle.nodes[j]
        if n.kind != tk.PT_VAR:
            nodes[(n.kind, n.left, n.right, n.extra, n.value)] = j
    var builder = ph._Builder[capacity](phase.bundle.copy(), nodes^)
    append_finishing_execution_core(builder)
    return builder^


def make_finishing_execution_proof[
    capacity: Int = EXECUTION_PROOF_CAPACITY,
]() -> ExecutionProof[capacity]:
    var builder = make_finishing_execution_builder[capacity]()
    var values = ph._named(builder, EXECUTION_VALUES)
    var readiness = ph._named(builder, EXECUTION_READINESS)
    var step_legal = ph._named(builder, EXECUTION_STEP_LEGAL)
    var legal_prefix = ph._named(builder, EXECUTION_LEGAL_PREFIX)
    var fold = ph._named(builder, EXECUTION_FLAT_FOLD)
    var invariant = ph._named(builder, EXECUTION_INVARIANT)
    var completion = ph._named(builder, EXECUTION_COMPLETION)
    builder.bundle.root = completion
    return ExecutionProof[capacity](builder.bundle.copy(), values,
        readiness, step_legal, legal_prefix, fold, invariant, completion)
