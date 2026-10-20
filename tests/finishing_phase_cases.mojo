"""Independent private-Fill execution and universal proof contracts."""
from std.collections import Array
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase as phase
import llm_mojo.kernels.finishing_phase_check as checked
from llm_mojo.kernels.finishing_phase_terms import PHASE_PROOF_CAPACITY, make_finishing_phase_proof
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord

comptime PHASE_ALIAS = 0
comptime PHASE_WRONG_SOURCE = 1
comptime PHASE_STALE_FOOTPRINT = 2
comptime PHASE_REQUEST_BINDING = 3
comptime PHASE_STORED_PHASE = 4
comptime PHASE_LEFT_OWNER = 5
comptime PHASE_RIGHT_OWNER = 6
comptime PHASE_WRONG_PADDING = 7
comptime PHASE_WRONG_THEOREM = 8
comptime PHASE_CERTIFICATE_SOURCE = 9
comptime PHASE_FALSE_APART = 10
comptime PHASE_FALSE_PAIR_TEST = 11
comptime PHASE_MISSING_ENABLED = 12
comptime PHASE_WRONG_READY = 13
comptime PHASE_MISSING_FRAME = 14
comptime PHASE_ROOT_BINDING = 15
comptime PHASE_UNUSED_MALFORMED = 16
comptime PHASE_REFLECTION = 17
comptime PHASE_EXPORT = 18
comptime PHASE_WRONG_UPDATE = 19
comptime PHASE_WRONG_EQB = 20
comptime PHASE_WRONG_NAT = 21
comptime PHASE_CYCLIC = 22
comptime PHASE_OPEN = 23
comptime PHASE_NEGATIVE_COUNT = 24
comptime PHASE_ROOT_BOUNDARY = 25
comptime PHASE_EXPECTED_EVENT = 26
comptime PHASE_EXPECTED_KEYS = 27
comptime PHASE_NEGATIVE_KEY = 28
comptime PHASE_SCRATCH_NODES = 29
comptime PHASE_SCRATCH_DEFINITIONS = 30
comptime PHASE_SCRATCH_CONSTRUCTORS = 31
comptime PHASE_FORGED_DESTINATION = 32
comptime FINISHING_PHASE_MUTANTS = 33


def phase_pair_accepts(threads: Int, items: Int, base: Int, count: Int,
    left_thread: Int, left_item: Int, right_thread: Int, right_item: Int) -> Bool:
    var p = phase.FinishingPhase(threads, items, base, count)
    var left = phase.make_finishing_fill(p, left_thread, left_item)
    var right = phase.make_finishing_fill(p, right_thread, right_item)
    var certificate = checked.make_finishing_phase_commutation(p, left, right)
    return checked.check_finishing_phase_commutation_request(certificate, p, left, right)


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _app[C: Int](mut p: tk.ProofTermBundle[C], f: Int, x: Int) -> Int:
    return _n(p, tk.PT_APP, f, x)


def _apps[C: Int](mut p: tk.ProofTermBundle[C], var f: Int, xs: List[Int]) -> Int:
    for x in xs:
        f = _app(p, f, x)
    return f


def _pis[C: Int](mut p: tk.ProofTermBundle[C], xs: List[Int], var body: Int) -> Int:
    for index in range(len(xs)):
        body = _n(p, tk.PT_PI, xs[len(xs) - index - 1], body)
    return body


def independent_phase_type[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    """All four pointwise store equations, with independent binder assembly.

    Every initial store, both payload triplets and observation are arbitrary.
    This statement is not built from a finite set of concrete record values.
    """
    var nat = _n(p, tk.PT_NAT)
    var a = _n(p, tk.PT_VAR, nat)
    var b = _n(p, tk.PT_VAR, nat)
    var apart = _n(p, tk.PT_VAR, _apps(p,
        _n(p, tk.PT_DEFINED_REF, value=6), [a, b]))
    var binders: List[Int] = [a, b, apart]
    var stores = List[Int]()
    for _ in range(4):
        var key = _n(p, tk.PT_VAR, nat)
        var store = _n(p, tk.PT_VAR, _n(p, tk.PT_PI, key, nat))
        stores.append(store)
        binders.append(store)
    var av = List[Int]()
    var bv = List[Int]()
    for _ in range(3):
        var value = _n(p, tk.PT_VAR, nat)
        av.append(value)
        binders.append(value)
    for _ in range(3):
        var value = _n(p, tk.PT_VAR, nat)
        bv.append(value)
        binders.append(value)
    var at = _n(p, tk.PT_VAR, nat)
    binders.append(at)
    var zero = _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var one = _app(p, _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), zero)
    var update = _n(p, tk.PT_DEFINED_REF, value=5)
    var components = List[Int]()
    for component in range(4):
        var x = av[component] if component < 3 else one
        var y = bv[component] if component < 3 else one
        var ab = _apps(p, update, [_apps(p, update, [stores[component], a, x]), b, y, at])
        var ba = _apps(p, update, [_apps(p, update, [stores[component], b, y]), a, x, at])
        components.append(_n(p, tk.PT_VAR, _n(p, tk.PT_EQ, ab, ba)))
    var result = _n(p, tk.PT_VAR, _n(p, tk.PT_PROP))
    var consumer = _n(p, tk.PT_VAR, _pis(p, components, result))
    return _pis(p, binders, _pis(p, [result, consumer], result))


def independent_phase_contract_accepts() -> Bool:
    var proof = make_finishing_phase_proof(0, 8)
    var expected = independent_phase_type(proof.bundle)
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[PHASE_PROOF_CAPACITY, 16384](program, proof.bundle)
    tk._enable_native_numerals(state)
    if not tk._check_inductive_environment(state):
        return False
    var actual = tk._infer(state, proof.commutation_root)
    return (actual >= 0 and tk._infer(state, expected) == tk._put(state, tk.PT_PROP)
        and state.context_high[expected] == 0 and state.context_high[proof.commutation_root] == 0
        and tk._convert(state, actual, expected)
        and not state.arithmetic_verified and not state.fast_arithmetic)


def phase_alternate_proof_accepts() -> Bool:
    var p = phase.FinishingPhase(3, 5, 0, 7)
    var left = phase.make_finishing_fill(p, 0, 0)
    var right = phase.make_finishing_fill(p, 2, 2)
    var certificate = checked.make_finishing_phase_commutation(p, left, right)
    var nat = _n(certificate.proof.bundle, tk.PT_NAT)
    var dummy = _n(certificate.proof.bundle, tk.PT_VAR, nat)
    var zero = _n(certificate.proof.bundle, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var wrapped = _app(certificate.proof.bundle,
        _n(certificate.proof.bundle, tk.PT_LAM, dummy, certificate.proof.selected_commutation_root), zero)
    certificate.proof.selected_commutation_root = wrapped
    certificate.proof.bundle.root = wrapped
    return wrapped >= 0 and checked.check_finishing_phase_commutation(certificate)


def phase_mutant_accepts(mode: Int) -> Bool:
    var p = phase.FinishingPhase(3, 5, 0, 7)
    var left = phase.make_finishing_fill(p, 0, 0)
    var right = phase.make_finishing_fill(p, 2, 2)
    var certificate = checked.make_finishing_phase_commutation(p, left, right)
    if certificate.proof.bundle.count <= 0:
        return True
    if mode == PHASE_ALIAS:
        # Both events are individually canonical. Admission must check that
        # EqB(0,0)=1 and Apart(0,0) cannot be manufactured by the proposal.
        var aliased = checked.make_finishing_phase_commutation(p, left, left)
        return checked.check_finishing_phase_commutation_request(aliased, p, left, left)
    if mode == PHASE_REQUEST_BINDING:
        var other = p
        other.base = 1
        return checked.check_finishing_phase_commutation_request(certificate, other, left, right)
    if mode == PHASE_EXPECTED_EVENT:
        var other = left
        other.source = 1
        return checked.check_finishing_phase_commutation_request(certificate, p, other, right)
    if mode == PHASE_EXPECTED_KEYS:
        return checked.check_finishing_phase_proof(certificate.proof, 1, 8)
    if mode == PHASE_SCRATCH_NODES:
        return checked.check_finishing_phase_proof[PHASE_PROOF_CAPACITY, 64](certificate.proof, 0, 8)
    if mode == PHASE_SCRATCH_DEFINITIONS or mode == PHASE_SCRATCH_CONSTRUCTORS:
        comptime Scratch = 2048
        if certificate.proof.bundle.count > Scratch:
            return True
        if mode == PHASE_SCRATCH_DEFINITIONS:
            for index in range(certificate.proof.bundle.definition_count, Scratch + 1):
                certificate.proof.bundle.definition_types[index] = certificate.proof.bundle.definition_types[0]
                certificate.proof.bundle.definition_bodies[index] = certificate.proof.bundle.definition_bodies[0]
            certificate.proof.bundle.definition_count = Scratch + 1
        else:
            for index in range(Scratch + 1):
                certificate.proof.bundle.nat_constructor_types[index] = certificate.proof.bundle.nat_constructor_types[0]
            certificate.proof.bundle.nat_constructor_count = Scratch + 1
        return checked.check_finishing_phase_proof[PHASE_PROOF_CAPACITY, Scratch](certificate.proof, 0, 8)
    var zero = _n(certificate.proof.bundle, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    if mode == PHASE_FORGED_DESTINATION:
        certificate.right.destination = 0
    elif mode == PHASE_WRONG_SOURCE:
        certificate.left.source = 1
    elif mode == PHASE_STALE_FOOTPRINT:
        certificate.proof.left_slot = 1
    elif mode == PHASE_STORED_PHASE:
        certificate.phase.count = 8
        return checked.check_finishing_phase_commutation_request(certificate, p, left, right)
    elif mode == PHASE_LEFT_OWNER:
        certificate.left.thread = 1
    elif mode == PHASE_RIGHT_OWNER:
        certificate.right.item = 1
    elif mode == PHASE_WRONG_PADDING:
        certificate.left.loads = False
    elif mode == PHASE_WRONG_THEOREM:
        certificate.proof.commutation_root = certificate.proof.ready_root
    elif mode == PHASE_CERTIFICATE_SOURCE:
        certificate.proof = make_finishing_phase_proof(0, 9)
    elif mode == PHASE_FALSE_APART:
        certificate.proof.pair_apart_root = _n(certificate.proof.bundle, tk.PT_REFL, zero)
    elif mode == PHASE_FALSE_PAIR_TEST:
        certificate.proof.pair_test_root = _n(certificate.proof.bundle, tk.PT_REFL, zero)
    elif mode == PHASE_MISSING_ENABLED:
        certificate.proof.enabled_ab_root = -1
    elif mode == PHASE_WRONG_READY:
        certificate.proof.ready_root = certificate.proof.frame_root
    elif mode == PHASE_MISSING_FRAME:
        certificate.proof.frame_root = -1
    elif mode == PHASE_ROOT_BINDING:
        certificate.proof.bundle.root = certificate.proof.commutation_root
    elif mode == PHASE_UNUSED_MALFORMED:
        _ = _app(certificate.proof.bundle, _n(certificate.proof.bundle, tk.PT_NAT), zero)
    elif mode == PHASE_REFLECTION:
        _ = _n(certificate.proof.bundle, tk.PT_AFFINE_EQ, zero, zero)
    elif mode == PHASE_EXPORT:
        _ = _n(certificate.proof.bundle, tk.PT_ARITHMETIC, value=tk.DEC_LE_TRANS)
    elif mode == PHASE_WRONG_UPDATE or mode == PHASE_WRONG_EQB:
        var definition = 5 if mode == PHASE_WRONG_UPDATE else 4
        var declared = certificate.proof.bundle.definition_types[definition]
        var binders = List[Int]()
        var current = declared
        while certificate.proof.bundle.nodes[current].kind == tk.PT_PI:
            binders.append(certificate.proof.bundle.nodes[current].left)
            current = certificate.proof.bundle.nodes[current].right
        var body = zero
        if mode == PHASE_WRONG_UPDATE:
            if len(binders) != 4:
                return True
            body = _app(certificate.proof.bundle, binders[0], binders[3])
        for index in range(len(binders)):
            body = _n(certificate.proof.bundle, tk.PT_LAM, binders[len(binders) - index - 1], body)
        certificate.proof.bundle.definition_bodies[definition] = body
    elif mode == PHASE_WRONG_NAT:
        certificate.proof.bundle.nat_constructor_types[1] = certificate.proof.bundle.nat_constructor_types[0]
    elif mode == PHASE_CYCLIC:
        certificate.proof.bundle.definition_bodies[0] = _n(certificate.proof.bundle, tk.PT_DEFINED_REF, value=0)
    elif mode == PHASE_OPEN:
        certificate.proof.bundle.definition_bodies[0] = _n(certificate.proof.bundle, tk.PT_VAR,
            _n(certificate.proof.bundle, tk.PT_PROP))
    elif mode == PHASE_NEGATIVE_COUNT:
        certificate.proof.bundle.count = -1
    elif mode == PHASE_ROOT_BOUNDARY:
        certificate.proof.selected_commutation_root = certificate.proof.bundle.count
        certificate.proof.bundle.root = certificate.proof.bundle.count
    elif mode == PHASE_NEGATIVE_KEY:
        certificate.proof.left_slot = -1
    else:
        return True
    return checked.check_finishing_phase_commutation(certificate)


def _record_equal(a: ArgmaxRecord, b: ArgmaxRecord) -> Bool:
    return a.rank == b.rank and a.token == b.token and a.invalid == b.invalid


def _oracle_step[C: Int, N: Int](mut state: phase.FinishingPhaseState[C],
    inputs: Array[ArgmaxRecord, N], thread: Int, item: Int) -> Bool:
    var p = state.phase
    if (p.threads <= 0 or p.items <= 0 or p.base < 0 or p.count < 0
        or p.base >= p.count or p.count > N or p.threads * p.items > C
        or thread < 0 or thread >= p.threads or item < 0 or item >= p.items):
        return False
    var destination = item * p.threads + thread
    if state.ready[destination]:
        return False
    var source = p.base + destination
    var payload = ArgmaxRecord(0, 0xFFFFFFFF, 0)
    if source < p.count:
        payload = inputs[source]
    state.records[destination] = payload
    state.ready[destination] = True
    return True


def _oracle_ready[C: Int](state: phase.FinishingPhaseState[C], thread: Int) -> Bool:
    if state.phase.base >= state.phase.count or thread < 0 or thread >= state.phase.threads:
        return False
    for item in range(state.phase.items):
        if not state.ready[item * state.phase.threads + thread]:
            return False
    return True


def _assert_states[C: Int](got: phase.FinishingPhaseState[C], expected: phase.FinishingPhaseState[C]) raises:
    assert_equal(got.phase.threads, expected.phase.threads)
    assert_equal(got.phase.items, expected.phase.items)
    assert_equal(got.phase.base, expected.phase.base)
    assert_equal(got.phase.count, expected.phase.count)
    for index in range(C):
        assert_equal(_record_equal(got.records[index], expected.records[index]), True)
        assert_equal(got.ready[index], expected.ready[index])


def check_phase_execution[threads: Int, items: Int](base: Int, count: Int) raises:
    comptime C = threads * items + 2
    var p = phase.FinishingPhase(threads, items, base, count)
    var inputs = Array[ArgmaxRecord, 64](fill=ArgmaxRecord(0, 0, 0))
    for index in range(64):
        inputs[index] = ArgmaxRecord(UInt32(11 + index * 13), UInt32(63 - index), UInt32(index % 2))
    var original_inputs = inputs.copy()
    var initial = phase.make_finishing_phase_state[C](p)
    for index in range(C):
        initial.records[index] = ArgmaxRecord(UInt32(400 + index), UInt32(900 - index), 1)
        initial.ready[index] = index >= threads * items
    var left = phase.make_finishing_fill(p, 0, 0)
    var right = phase.make_finishing_fill(p, threads - 1, items - 1)
    var ab = initial.copy()
    var ba = initial.copy()
    var expected_ab = initial.copy()
    var expected_ba = initial.copy()
    assert_equal(phase.step_finishing_fill(ab, inputs, left), _oracle_step(expected_ab, inputs, 0, 0))
    assert_equal(phase.step_finishing_fill(ab, inputs, right), _oracle_step(expected_ab, inputs, threads - 1, items - 1))
    assert_equal(phase.step_finishing_fill(ba, inputs, right), _oracle_step(expected_ba, inputs, threads - 1, items - 1))
    assert_equal(phase.step_finishing_fill(ba, inputs, left), _oracle_step(expected_ba, inputs, 0, 0))
    _assert_states(ab, expected_ab)
    _assert_states(ba, expected_ba)
    _assert_states(ab, ba)
    for thread in range(threads):
        assert_equal(phase.finishing_thread_ready(ab, thread), _oracle_ready(expected_ab, thread))
    for index in range(64):
        assert_equal(_record_equal(inputs[index], original_inputs[index]), True)


def check_phase_native_controls() raises:
    check_phase_execution[1, 1](0, 1)
    check_phase_execution[3, 5](0, 7)
    check_phase_execution[4, 3](2, 13)
    check_phase_execution[1, 5](0, 1)
    check_phase_execution[64, 3](0, 1)
    check_phase_execution[1, 65](0, 1)
    check_phase_execution[5, 1](0, 3)
    check_phase_execution[3, 1](3, 3)
    check_phase_execution[3, 5](0, 0)
    check_phase_execution[3, 5](8, 7)
    var counts: Array[Int, 5] = [0, 1, 14, 15, 16]
    for count in counts:
        check_phase_execution[3, 5](0, count)
        check_phase_execution[3, 5](1, count)
        check_phase_execution[3, 5](count, count)
        check_phase_execution[3, 5](count + 1, count)
    var p = phase.FinishingPhase(3, 2, 0, 2)
    var state = phase.make_finishing_phase_state[8](p)
    var inputs = Array[ArgmaxRecord, 4](fill=ArgmaxRecord(71, 29, 1))
    var first = phase.make_finishing_fill(p, 0, 0)
    var padding = phase.make_finishing_fill(p, 0, 1)
    assert_equal(phase.step_finishing_fill(state, inputs, first), True)
    assert_equal(phase.finishing_thread_ready(state, 0), False)
    assert_equal(phase.step_finishing_fill(state, inputs, padding), True)
    assert_equal(_record_equal(state.records[3], ArgmaxRecord(0, 0xFFFFFFFF, 0)), True)
    assert_equal(phase.finishing_thread_ready(state, 0), True)
    assert_equal(phase.finishing_thread_ready(state, 1), False)
    var before = state.copy()
    assert_equal(phase.finishing_fill_enabled(state, padding), False)
    assert_equal(phase.step_finishing_fill(state, inputs, padding), False)
    _assert_states(state, before)
    p.count = 5
    state = phase.make_finishing_phase_state[8](p)
    first = phase.make_finishing_fill(p, 0, 0)
    before = state.copy()
    assert_equal(phase.finishing_fill_enabled(state, first), True)
    assert_equal(phase.step_finishing_fill(state, inputs, first), False)
    _assert_states(state, before)
    # Invalid native event metadata must preserve state even when input extent
    # is sufficient and the corresponding canonical target is still pending.
    p = phase.FinishingPhase(3, 2, 0, 2)
    for mode in range(3):
        state = phase.make_finishing_phase_state[8](p)
        first = phase.make_finishing_fill(p, 0, 0)
        before = state.copy()
        if mode == 0:
            first.source = 1
        elif mode == 1:
            first.destination = 1
        else:
            first.loads = False
        assert_equal(phase.finishing_fill_enabled(state, first), False)
        assert_equal(phase.step_finishing_fill(state, inputs, first), False)
        _assert_states(state, before)
    assert_equal(phase.valid_finishing_phase(phase.FinishingPhase(Int.MAX, 2, 0, 1)), False)
    assert_equal(phase.valid_finishing_phase(phase.FinishingPhase(2, 1,
        Int(9223372036854775806), Int.MAX)), False)


def check_phase_counterexamples() raises:
    var a = ArgmaxRecord(17, 23, 0)
    var b = ArgmaxRecord(91, 4, 1)
    # Bad aliased destinations make immutable-source copies order dependent.
    var ab = Array[ArgmaxRecord, 2](fill=ArgmaxRecord(0, 0, 0))
    var ba = ab.copy()
    ab[0] = a
    ab[0] = b
    ba[0] = b
    ba[0] = a
    assert_equal(_record_equal(ab[0], ba[0]), False)
    # Two wrong-source copies can commute while violating intended Fill data.
    ab[0] = b
    ab[1] = a
    ba[1] = a
    ba[0] = b
    assert_equal(_record_equal(ab[0], ba[0]) and _record_equal(ab[1], ba[1]), True)
    assert_equal(_record_equal(ab[0], a), False)
    assert_equal(phase_mutant_accepts(PHASE_WRONG_SOURCE), False)
