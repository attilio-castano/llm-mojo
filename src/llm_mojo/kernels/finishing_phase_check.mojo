"""Admission of private-fill requests and ordinary Nat commutation terms.

The operational model is in finishing_phase. The untrusted proof producer is
finishing_phase_terms. This module independently specifies the data definitions
and theorem types; no commutation, frame or enabledness opcode is introduced.
"""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_phase import (
    FinishingPhase, FinishingFill, check_finishing_fill,
    same_finishing_phase, same_finishing_fill,
)
from llm_mojo.kernels.finishing_phase_terms import (
    PhaseProof, PHASE_PROOF_CAPACITY, make_finishing_phase_proof,
)


@fieldwise_init
struct FinishingPhaseCommutation[capacity: Int](Copyable, Movable):
    var phase: FinishingPhase
    var left: FinishingFill
    var right: FinishingFill
    var proof: PhaseProof[Self.capacity]


def make_finishing_phase_commutation[capacity: Int = PHASE_PROOF_CAPACITY](
    phase: FinishingPhase, left: FinishingFill, right: FinishingFill,
) -> FinishingPhaseCommutation[capacity]:
    return FinishingPhaseCommutation[capacity](phase, left, right,
        make_finishing_phase_proof[capacity](left.destination, right.destination))


def _node[C: Int](mut s: tk._TypedState[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk._put(s, kind, left, right, extra, value)


def _nat[C: Int](mut s: tk._TypedState[C]) -> Int:
    return _node(s, tk.PT_NAT)


def _prop[C: Int](mut s: tk._TypedState[C]) -> Int:
    return _node(s, tk.PT_PROP)


def _var[C: Int](mut s: tk._TypedState[C], type: Int) -> Int:
    return _node(s, tk.PT_VAR, type)


def _pi[C: Int](mut s: tk._TypedState[C], binder: Int, body: Int) -> Int:
    return _node(s, tk.PT_PI, binder, body)


def _lam[C: Int](mut s: tk._TypedState[C], binder: Int, body: Int) -> Int:
    return _node(s, tk.PT_LAM, binder, body)


def _named[C: Int](mut s: tk._TypedState[C], definition: Int) -> Int:
    return _node(s, tk.PT_DEFINED_REF, value=definition)


def _app[C: Int](mut s: tk._TypedState[C], f: Int, x: Int) -> Int:
    return _node(s, tk.PT_APP, f, x)


def _a2[C: Int](mut s: tk._TypedState[C], f: Int, x: Int, y: Int) -> Int:
    return _app(s, _app(s, f, x), y)


def _a3[C: Int](mut s: tk._TypedState[C], f: Int, x: Int, y: Int, z: Int) -> Int:
    return _app(s, _a2(s, f, x, y), z)


def _zero[C: Int](mut s: tk._TypedState[C]) -> Int:
    return _node(s, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _succ[C: Int](mut s: tk._TypedState[C], x: Int) -> Int:
    return _app(s, _node(s, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), x)


def _eq[C: Int](mut s: tk._TypedState[C], a: Int, b: Int) -> Int:
    return _node(s, tk.PT_EQ, a, b)


def _rec[C: Int](mut s: tk._TypedState[C], n: Int, zero: Int, step: Int) -> Int:
    return _node(s, tk.PT_NAT_REC, n, zero, step)


def _store_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var x = _var(s, _nat(s))
    return _pi(s, x, _nat(s))


def _update[C: Int](mut s: tk._TypedState[C], store: Int, key: Int, value: Int) -> Int:
    return _a3(s, _named(s, 5), store, key, value)


def _apart[C: Int](mut s: tk._TypedState[C], a: Int, b: Int) -> Int:
    return _a2(s, _named(s, 6), a, b)


def _pis[C: Int](mut s: tk._TypedState[C], binders: List[Int], body: Int) -> Int:
    var result = body
    for j in range(len(binders)):
        result = _pi(s, binders[len(binders) - j - 1], result)
    return result


def _lams[C: Int](mut s: tk._TypedState[C], binders: List[Int], body: Int) -> Int:
    var result = body
    for j in range(len(binders)):
        result = _lam(s, binders[len(binders) - j - 1], result)
    return result


def _data_matches[C: Int](mut s: tk._TypedState[C], index: Int,
    expected_type: Int, expected_body: Int) -> Bool:
    return (expected_type >= 0 and expected_body >= 0
        and tk._infer(s, expected_type) >= 0 and tk._infer(s, expected_body) >= 0
        and tk._convert(s, s.syntax.definition_types[index], expected_type)
        and tk._convert(s, s.syntax.definition_bodies[index], expected_body))


def _canonical_phase_data[C: Int](mut s: tk._TypedState[C]) -> Bool:
    """Independent specification of all eight operational/logical data bodies."""
    var proposition = _var(s, _prop(s))
    if not _data_matches(s, 0, _prop(s), _pi(s, proposition, proposition)):
        return False
    var n = _var(s, _nat(s))
    var predecessor = _var(s, _nat(s))
    var previous = _var(s, _prop(s))
    var body = _rec(s, n, _eq(s, _zero(s), _zero(s)),
        _lams(s, [predecessor, previous], _named(s, 0)))
    if not _data_matches(s, 1, _pi(s, n, _prop(s)), _lam(s, n, body)):
        return False
    n = _var(s, _nat(s))
    predecessor = _var(s, _nat(s))
    previous = _var(s, _nat(s))
    body = _rec(s, n, _zero(s), _lams(s, [predecessor, previous], predecessor))
    if not _data_matches(s, 2, _pi(s, n, _nat(s)), _lam(s, n, body)):
        return False
    n = _var(s, _nat(s))
    var chosen = _var(s, _nat(s))
    var fallback = _var(s, _nat(s))
    predecessor = _var(s, _nat(s))
    previous = _var(s, _nat(s))
    body = _rec(s, n, chosen, _lams(s, [predecessor, previous], fallback))
    if not _data_matches(s, 3, _pis(s, [n, chosen, fallback], _nat(s)),
        _lams(s, [n, chosen, fallback], body)):
        return False
    # EqB is structural equality testing, not a native comparison oracle.
    var a = _var(s, _nat(s))
    var b = _var(s, _nat(s))
    predecessor = _var(s, _nat(s))
    previous = _var(s, _nat(s))
    var base = _lam(s, b, _rec(s, b, _zero(s),
        _lams(s, [predecessor, previous], _succ(s, _zero(s)))))
    predecessor = _var(s, _nat(s))
    var recursive = _var(s, _store_type(s))
    b = _var(s, _nat(s))
    var bp = _var(s, _nat(s))
    previous = _var(s, _nat(s))
    var step = _lams(s, [predecessor, recursive, b],
        _rec(s, b, _succ(s, _zero(s)),
            _lams(s, [bp, previous], _app(s, recursive, bp))))
    body = _rec(s, a, base, step)
    if not _data_matches(s, 4, _pi(s, a, _store_type(s)), _lam(s, a, body)):
        return False
    var store = _var(s, _store_type(s))
    var key = _var(s, _nat(s))
    var value = _var(s, _nat(s))
    var observation = _var(s, _nat(s))
    body = _a3(s, _named(s, 3), _a2(s, _named(s, 4), key, observation),
        value, _app(s, store, observation))
    if not _data_matches(s, 5, _pis(s, [store, key, value, observation], _nat(s)),
        _lams(s, [store, key, value, observation], body)):
        return False
    a = _var(s, _nat(s))
    b = _var(s, _nat(s))
    var equality = _var(s, _eq(s, a, b))
    if not _data_matches(s, 6, _pis(s, [a, b], _prop(s)),
        _lams(s, [a, b], _pi(s, equality, _named(s, 0)))):
        return False
    a = _var(s, _nat(s))
    b = _var(s, _nat(s))
    proposition = _var(s, _prop(s))
    equality = _var(s, _eq(s, a, b))
    var separation = _var(s, _apart(s, a, b))
    var on_equal = _var(s, _pi(s, equality, proposition))
    var on_apart = _var(s, _pi(s, separation, proposition))
    return _data_matches(s, 7, _pis(s, [a, b], _prop(s)),
        _lams(s, [a, b], _pis(s, [proposition, on_equal, on_apart], proposition)))


def _swap_value[C: Int](mut s: tk._TypedState[C], store: Int, a: Int, b: Int,
    av: Int, bv: Int, observation: Int) -> Int:
    return _app(s, _update(s, _update(s, store, a, av), b, bv), observation)


def _commutation_type[C: Int](mut s: tk._TypedState[C], left: Int = -1,
    right: Int = -1) -> Int:
    var binders = List[Int]()
    var a = left
    var b = right
    if left < 0 or right < 0:
        a = _var(s, _nat(s))
        b = _var(s, _nat(s))
        binders.append(a)
        binders.append(b)
        binders.append(_var(s, _apart(s, a, b)))
    var stores = List[Int]()
    for _ in range(4):
        var store = _var(s, _store_type(s))
        stores.append(store)
        binders.append(store)
    var av = List[Int]()
    var bv = List[Int]()
    for _ in range(3):
        var value = _var(s, _nat(s))
        av.append(value)
        binders.append(value)
    for _ in range(3):
        var value = _var(s, _nat(s))
        bv.append(value)
        binders.append(value)
    var observation = _var(s, _nat(s))
    binders.append(observation)
    var result = _var(s, _prop(s))
    var component_proofs = List[Int]()
    for component in range(4):
        var va = av[component] if component < 3 else _succ(s, _zero(s))
        var vb = bv[component] if component < 3 else _succ(s, _zero(s))
        var equation = _eq(s,
            _swap_value(s, stores[component], a, b, va, vb, observation),
            _swap_value(s, stores[component], b, a, vb, va, observation))
        component_proofs.append(_var(s, equation))
    var consumer = _var(s, _pis(s, component_proofs, result))
    return _pis(s, binders, _pi(s, result, _pi(s, consumer, result)))


def _enabled_type[C: Int](mut s: tk._TypedState[C], direction: Int,
    left: Int = -1, right: Int = -1) -> Int:
    var binders = List[Int]()
    var a = left
    var b = right
    if left < 0 or right < 0:
        a = _var(s, _nat(s))
        b = _var(s, _nat(s))
        binders.append(a)
        binders.append(b)
        binders.append(_var(s, _apart(s, a, b)))
    var store = _var(s, _store_type(s))
    var pa = _var(s, _eq(s, _app(s, store, a), _zero(s)))
    var pb = _var(s, _eq(s, _app(s, store, b), _zero(s)))
    binders.append(store)
    binders.append(pa)
    binders.append(pb)
    var changed = a if direction == 0 else b
    var observed = b if direction == 0 else a
    var after = _app(s, _update(s, store, changed, _succ(s, _zero(s))), observed)
    return _pis(s, binders, _eq(s, after, _zero(s)))


def _ready_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var store = _var(s, _store_type(s))
    var key = _var(s, _nat(s))
    var one = _succ(s, _zero(s))
    return _pis(s, [store, key], _eq(s, _app(s, _update(s, store, key, one), key), one))


def _frame_type[C: Int](mut s: tk._TypedState[C]) -> Int:
    var store = _var(s, _store_type(s))
    var key = _var(s, _nat(s))
    var value = _var(s, _nat(s))
    var observation = _var(s, _nat(s))
    var apart = _var(s, _apart(s, key, observation))
    return _pis(s, [store, key, value, observation, apart],
        _eq(s, _app(s, _update(s, store, key, value), observation),
            _app(s, store, observation)))


def _phase_syntax[C: Int](p: tk.ProofTermBundle[C]) -> Bool:
    if (not tk._valid_syntax(p) or p.family_arity != -1 or p.constructor_count != 0
        or p.definition_count < 8):
        return False
    for j in range(p.count):
        var n = p.nodes[j]
        if (n.kind != tk.PT_TYPE and n.kind != tk.PT_PROP and n.kind != tk.PT_NAT
            and n.kind != tk.PT_VAR and n.kind != tk.PT_PI and n.kind != tk.PT_LAM
            and n.kind != tk.PT_APP and n.kind != tk.PT_LITERAL
            and n.kind != tk.PT_NAT_REC and n.kind != tk.PT_DEFINED_REF
            and n.kind != tk.PT_CONSTRUCTOR and n.kind != tk.PT_ELIMINATOR
            and n.kind != tk.PT_EQ and n.kind != tk.PT_REFL and n.kind != tk.PT_SUBST):
            return False
        if n.kind == tk.PT_CONSTRUCTOR and n.extra != tk.NAT_FAMILY_ID:
            return False
        if n.kind == tk.PT_ELIMINATOR and n.value != tk.NAT_FAMILY_ID:
            return False
    return True


def check_finishing_phase_proof[C: Int, scratch: Int = 16384](
    proof: PhaseProof[C], left_slot: Int, right_slot: Int,
) -> Bool:
    """Check universal four-store proofs and actual selected-key separation.

    All submitted syntax and definitions, including unused entries, are
    checked. Only previously certified numeral abbreviations are enabled;
    arithmetic exports and affine/order reflection are forbidden. Pointwise
    equations avoid assuming function extensionality or a packed record code.
    """
    if (scratch <= 0 or left_slot < 0 or right_slot < 0
        or proof.left_slot != left_slot or proof.right_slot != right_slot
        or proof.bundle.count > scratch or proof.bundle.definition_count > scratch
        or proof.bundle.nat_constructor_count > scratch
        or proof.bundle.constructor_count > scratch or not _phase_syntax(proof.bundle)
        or proof.bundle.root != proof.selected_commutation_root):
        return False
    var roots = [proof.commutation_root, proof.enabled_ab_root, proof.enabled_ba_root,
        proof.ready_root, proof.frame_root, proof.pair_test_root, proof.pair_apart_root,
        proof.selected_commutation_root, proof.selected_enabled_ab_root,
        proof.selected_enabled_ba_root]
    for root in roots:
        if root < 0 or root >= proof.bundle.count:
            return False
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[C, scratch](program, proof.bundle)
    state.fuel = scratch * 256
    tk._enable_native_numerals(state)
    if not tk._check_inductive_environment(state) or not _canonical_phase_data(state):
        return False
    for j in range(proof.bundle.count):
        var inferred = tk._infer(state, j)
        if inferred == -1 or inferred == -4:
            return False
    var left = _node(state, tk.PT_LITERAL, value=left_slot)
    var right = _node(state, tk.PT_LITERAL, value=right_slot)
    var pair_test = _eq(state, _a2(state, _named(state, 4), left, right),
        _succ(state, _zero(state)))
    var expected = [_commutation_type(state), _enabled_type(state, 0),
        _enabled_type(state, 1), _ready_type(state), _frame_type(state),
        pair_test, _apart(state, left, right), _commutation_type(state, left, right),
        _enabled_type(state, 0, left, right), _enabled_type(state, 1, left, right)]
    var prop = _prop(state)
    for j in range(len(roots)):
        var root = roots[j]
        var type = expected[j]
        if (tk._infer(state, type) != prop or state.context_high[type] != 0
            or tk._infer(state, root) < 0 or state.context_high[root] != 0
            or not tk._convert(state, tk._infer(state, root), type)):
            return False
    return (not state.arithmetic_verified and not state.fast_arithmetic)


def check_finishing_phase_commutation_request[C: Int, scratch: Int = 16384](
    certificate: FinishingPhaseCommutation[C], phase: FinishingPhase,
    left: FinishingFill, right: FinishingFill,
) -> Bool:
    """Bind both operational events and the exact requested phase before admission.

    A native inequality is never a separation proof: the Nat proof must supply
    Apart for these independently reconstructed destination keys. Payloads are
    universally quantified; input immutability and valid input extent are the
    native interpreter preconditions. No source-index or padding claim follows
    merely from the abstract store-update commutation theorem.
    """
    if (not same_finishing_phase(certificate.phase, phase)
        or not same_finishing_fill(certificate.left, left)
        or not same_finishing_fill(certificate.right, right)
        or not check_finishing_fill(phase, left) or not check_finishing_fill(phase, right)):
        return False
    return check_finishing_phase_proof[C, scratch](certificate.proof,
        left.destination, right.destination)


def check_finishing_phase_commutation[C: Int, scratch: Int = 16384](
    certificate: FinishingPhaseCommutation[C],
) -> Bool:
    return check_finishing_phase_commutation_request[C, scratch](certificate,
        certificate.phase, certificate.left, certificate.right)
