"""Candidate maintained controls for the independent148 Nat record bridge.

These are candidate tests until a measured positive compiler/native gate
qualifies the chosen scratch budget. A false result from resource exhaustion
must never be counted as semantic rejection. This module excludes the future
record-law definitions148+ and does not claim native UInt32/GPU refinement.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_phase_check as phase
import llm_mojo.kernels.finishing_execution_trace as trace
import llm_mojo.kernels.finishing_execution_algebra as algebra
import llm_mojo.kernels.nat_join_max as maximum
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_join_bool as boolean
import llm_mojo.kernels.finishing_join_terms as join
import llm_mojo.kernels.nat_score_link as arithmetic
import llm_mojo.kernels.nat_score_complement as complement
import llm_mojo.kernels.nat_score_order as order
import llm_mojo.kernels.nat_comparison_bridge as comparison
import llm_mojo.kernels.nat_score_embedding as embedding
import llm_mojo.kernels.nat_record_selector as selector
import llm_mojo.kernels.nat_record_encoding as encoding
import llm_mojo.kernels.nat_score_bridge_check as checked

comptime NATIVE_SCORE_CAPACITY = 16384
comptime NATIVE_SCORE_SEMANTIC_CONTROLS = 30
comptime NATIVE_SCORE_RESOURCE_CONTROLS = 3
comptime NATIVE_SCORE_ASSUMED_BRIDGE = 8


@no_inline
def _builder_holder() -> List[ph._Builder[NATIVE_SCORE_CAPACITY]]:
    var builders = List[ph._Builder[NATIVE_SCORE_CAPACITY]](capacity=1)
    builders.append(trace.make_finishing_execution_trace_builder[NATIVE_SCORE_CAPACITY]())
    return builders^


def _append_proof(mut p: ph._Builder[NATIVE_SCORE_CAPACITY],
    mut layouts: List[checked.NatScoreBridgeLayout]) -> List[tk.ProofTermBundle[NATIVE_SCORE_CAPACITY]]:
    """Borrow the one heap builder; preserve the existing148 proof terms."""
    maximum.append_record_max_terms(p)
    bits.append_record_parity_terms(p)
    var booleans = boolean.append_record_bool_terms(p)
    join.append_record_join_terms(p, booleans)
    var report = arithmetic.append_nat_score_arithmetic(p)
    var complement_first = p.bundle.definition_count
    var complements = complement.append_nat_score_complement(p)
    var order_first = p.bundle.definition_count
    var orders = order.append_nat_score_order_terms(p)
    var comparison_first = p.bundle.definition_count
    var comparisons = comparison.append_nat_comparison_terms(p)
    var embedding_first = p.bundle.definition_count
    var embeddings = embedding.append_nat_score_embedding(p, complements, orders)
    var selector_first = p.bundle.definition_count
    var selectors = selector.append_nat_record_selector(p, comparisons, orders, embeddings)
    var encoding_first = p.bundle.definition_count
    var encodings = encoding.append_nat_record_encoding(p, complements, embeddings, selectors)
    p.bundle.root = ph._named(p, encodings.merge_preserves)
    if (not report.valid or p.bundle.definition_count != 148
        or complement_first != 111 or order_first != 117 or comparison_first != 125
        or embedding_first != 133 or selector_first != 139 or encoding_first != 142
        or encodings.merge_preserves != 147):
        p.bundle.count = -1
    layouts.append(checked.NatScoreBridgeLayout(complement_first, order_first,
        comparison_first, embedding_first, selector_first, encoding_first,
        report.predecessor_conversion_root))
    var proofs = List[tk.ProofTermBundle[NATIVE_SCORE_CAPACITY]](capacity=1)
    proofs.append(p.bundle.copy())
    return proofs^


@no_inline
def make_native_score_proof_holder(mut layouts: List[checked.NatScoreBridgeLayout]) -> List[tk.ProofTermBundle[NATIVE_SCORE_CAPACITY]]:
    var builders = _builder_holder()
    return _append_proof(builders[0], layouts)


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _named[C: Int](mut p: tk.ProofTermBundle[C], definition: Int) -> Int:
    return _n(p, tk.PT_DEFINED_REF, value=definition)


def _call[C: Int](mut p: tk.ProofTermBundle[C], definition: Int,
    arguments: List[Int]) -> Int:
    var result = _named(p, definition)
    for argument in arguments:
        result = _n(p, tk.PT_APP, result, argument)
    return result


def _lams[C: Int](mut p: tk.ProofTermBundle[C], binders: List[Int], var body: Int) -> Int:
    for index in range(len(binders)):
        body = _n(p, tk.PT_LAM, binders[len(binders) - index - 1], body)
    return body


def _pis[C: Int](mut p: tk.ProofTermBundle[C], binders: List[Int], var body: Int) -> Int:
    for index in range(len(binders)):
        body = _n(p, tk.PT_PI, binders[len(binders) - index - 1], body)
    return body


def _binders[C: Int](p: tk.ProofTermBundle[C], definition: Int) -> List[Int]:
    var binders = List[Int]()
    var current = p.definition_types[definition]
    while p.nodes[current].kind == tk.PT_PI:
        binders.append(p.nodes[current].left)
        current = p.nodes[current].right
    return binders^


def _zero[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _replace_with_binder[C: Int](mut p: tk.ProofTermBundle[C],
    definition: Int, selected: Int):
    var binders = _binders(p, definition)
    p.definition_bodies[definition] = _lams(p, binders, binders[selected])


def _change_merge[C: Int](mut p: tk.ProofTermBundle[C], lose_flag: Bool):
    var binders = _binders(p, 143)
    var bound = binders[0]
    var ra = binders[1]
    var rb = binders[2]
    var ta = binders[3]
    var tb = binders[4]
    var ba = binders[5]
    var bb = binders[6]
    var union = _call(p, 68, [ba, bb])
    var body: Int
    if lose_flag:
        var flag = _call(p, 139, [ra, rb, ta, tb])
        body = _call(p, 3, [flag,
            _call(p, 142, [bound, ra, ta, ba]),
            _call(p, 142, [bound, rb, tb, ba])])
    else:
        body = _call(p, 142, [bound, ra, ta, union])
    p.definition_bodies[143] = _lams(p, binders, body)


def _assume_bridge[C: Int](mut p: tk.ProofTermBundle[C]):
    """A welltyped pointwise tautology, adding the very equation being proved.

    It retains the complete original record-domain telescope, then accepts
    h:Eq(Join(EncodeA,EncodeB),Merge(...)) and returns h. The independent
    profile must reject its extra premise even though ordinary typing admits
    the entire mutated environment. No malformed proof body is involved.
    """
    var binders = _binders(p, 147)
    var equation = p.definition_types[147]
    while p.nodes[equation].kind == tk.PT_PI:
        equation = p.nodes[equation].right
    var assumption = _n(p, tk.PT_VAR, equation)
    binders.append(assumption)
    p.definition_types[147] = _pis(p, binders, equation)
    p.definition_bodies[147] = _lams(p, binders, assumption)


@no_inline
def native_score_accepts[scratch: Int = 40960]() -> Bool:
    var layouts = List[checked.NatScoreBridgeLayout](capacity=1)
    var proofs = make_native_score_proof_holder(layouts)
    return checked.check_nat_score_bridge_proof[NATIVE_SCORE_CAPACITY, scratch](proofs[0], layouts[0])


@no_inline
def native_score_alternate_accepts[scratch: Int = 40960]() -> Bool:
    var layouts = List[checked.NatScoreBridgeLayout](capacity=1)
    var proofs = make_native_score_proof_holder(layouts)
    if proofs[0].count < 1:
        return False
    var nat = _n(proofs[0], tk.PT_NAT)
    var dummy = _n(proofs[0], tk.PT_VAR, nat)
    var zero = _zero(proofs[0])
    var wrapper = _n(proofs[0], tk.PT_LAM, dummy, proofs[0].root)
    proofs[0].root = _n(proofs[0], tk.PT_APP, wrapper, zero)
    return checked.check_nat_score_bridge_proof[NATIVE_SCORE_CAPACITY, scratch](proofs[0], layouts[0])


@no_inline
def native_score_mutant_accepts[scratch: Int = 40960](mode: Int) -> Bool:
    var layouts = List[checked.NatScoreBridgeLayout](capacity=1)
    var proofs = make_native_score_proof_holder(layouts)
    if proofs[0].count < 1:
        return True
    var zero = _zero(proofs[0])
    if mode == 0:
        _replace_with_binder(proofs[0], 95, 0)  # Add drops the right input.
    elif mode == 1:
        _replace_with_binder(proofs[0], 96, 0)  # Mul returns the left factor.
    elif mode == 2:
        var binders = _binders(proofs[0], 97)
        var proposition = _n(proofs[0], tk.PT_VAR, _n(proofs[0], tk.PT_PROP))
        var evidence = _n(proofs[0], tk.PT_VAR, proposition)
        var truth = _pis(proofs[0], [proposition, evidence], proposition)
        proofs[0].definition_bodies[97] = _lams(proofs[0], binders, truth)
    elif mode == 3:
        _replace_with_binder(proofs[0], 111, 0)  # Complement ignores the token.
    elif mode == 4:
        _replace_with_binder(proofs[0], 133, 1)  # Score ignores radix/token.
    elif mode == 5:
        var binders = _binders(proofs[0], 139)
        proofs[0].definition_bodies[139] = _lams(proofs[0], binders, zero)
    elif mode == 6:
        var binders = _binders(proofs[0], 142)
        var score = _call(proofs[0], 133, [binders[0], binders[1], binders[2]])
        var body = _call(proofs[0], 67, [score, zero])
        proofs[0].definition_bodies[142] = _lams(proofs[0], binders, body)
    elif mode == 7:
        _change_merge(proofs[0], False)
    elif mode == 8:
        _assume_bridge(proofs[0])
    elif mode == 9:
        proofs[0].definition_bodies[112] = proofs[0].definition_bodies[114]
    elif mode == 10:
        proofs[0].definition_bodies[125] = proofs[0].definition_bodies[126]
    elif mode == 11:
        proofs[0].definition_bodies[135] = proofs[0].definition_bodies[136]
    elif mode == 12:
        proofs[0].definition_bodies[140] = proofs[0].definition_bodies[141]
    elif mode == 13:
        proofs[0].definition_bodies[146] = proofs[0].definition_bodies[147]
    elif mode == 14:
        var binders = _binders(proofs[0], 144)
        proofs[0].definition_bodies[144] = _lams(proofs[0], binders,
            _n(proofs[0], tk.PT_REFL, zero))
    elif mode == 15:
        _ = _n(proofs[0], tk.PT_APP, _n(proofs[0], tk.PT_NAT), zero)
    elif mode == 16:
        proofs[0].root = _n(proofs[0], tk.PT_VAR, _n(proofs[0], tk.PT_NAT))
    elif mode == 17:
        proofs[0].definition_bodies[147] = _named(proofs[0], 147)
    elif mode == 18:
        # Two unused declarations form an acyclic but forward edge145→146.
        # They are both Nat-valued, so a forward-reference rejection cannot
        # be confused with a type mismatch at that edge.
        var nat = _n(proofs[0], tk.PT_NAT)
        proofs[0].definition_types[145] = nat
        proofs[0].definition_bodies[145] = _named(proofs[0], 146)
        proofs[0].definition_types[146] = nat
        proofs[0].definition_bodies[146] = zero
    elif mode == 19:
        layouts[0].predecessor_conversion_root = proofs[0].definition_bodies[145]
    elif mode == 20:
        _ = _n(proofs[0], tk.PT_ARITHMETIC, value=tk.DEC_LE_TRANS)
    elif mode == 21:
        _ = _n(proofs[0], tk.PT_AFFINE_EQ, zero, zero)
    elif mode == 22:
        layouts[0].complement_first += 1
    elif mode == 23:
        layouts[0].order_first += 1
    elif mode == 24:
        layouts[0].comparison_first += 1
    elif mode == 25:
        layouts[0].embedding_first += 1
    elif mode == 26:
        layouts[0].selector_first += 1
    elif mode == 27:
        layouts[0].encoding_first += 1
    elif mode == 28:
        _ = _n(proofs[0], tk.PT_LITERAL_LE, zero, zero)
    elif mode == 29:
        _change_merge(proofs[0], True)
    else:
        return True
    return checked.check_nat_score_bridge_proof[NATIVE_SCORE_CAPACITY, scratch](proofs[0], layouts[0])


@no_inline
def native_score_resource_mutant_accepts(mode: Int) -> Bool:
    """Allocation preflights only, explicitly outside semantic controls."""
    var layouts = List[checked.NatScoreBridgeLayout](capacity=1)
    var proofs = make_native_score_proof_holder(layouts)
    comptime Scratch = NATIVE_SCORE_CAPACITY - 1
    if proofs[0].count <= 64 or proofs[0].count > Scratch:
        return True  # A bad source-size setup must fail the test itself.
    if mode == 0:
        for index in range(proofs[0].definition_count, Scratch + 1):
            proofs[0].definition_types[index] = proofs[0].definition_types[0]
            proofs[0].definition_bodies[index] = proofs[0].definition_bodies[0]
        proofs[0].definition_count = Scratch + 1
    elif mode == 1:
        for index in range(Scratch + 1):
            proofs[0].nat_constructor_types[index] = proofs[0].nat_constructor_types[0]
        proofs[0].nat_constructor_count = Scratch + 1
    elif mode == 2:
        return checked.check_nat_score_bridge_proof[NATIVE_SCORE_CAPACITY, 64](proofs[0], layouts[0])
    else:
        return True
    return checked.check_nat_score_bridge_proof[NATIVE_SCORE_CAPACITY, Scratch](proofs[0], layouts[0])



@no_inline
def native_score_assumed_bridge_is_welltyped[scratch: Int = 40960]() -> Bool:
    """Raw typing positive for control8; does not grant profile admission."""
    var layouts = List[checked.NatScoreBridgeLayout](capacity=1)
    var proofs = make_native_score_proof_holder(layouts)
    if proofs[0].count <= 0:
        return False
    _assume_bridge(proofs[0])
    if proofs[0].count > scratch or not phase._phase_syntax(proofs[0]):
        return False
    var states = checked._state_holder[NATIVE_SCORE_CAPACITY, scratch](proofs[0])
    states[0].fuel = scratch * 2048
    tk._enable_native_numerals(states[0])
    if not tk._check_inductive_environment(states[0]):
        return False
    for index in range(proofs[0].count):
        var actual = tk._infer(states[0], index)
        if actual == -1 or actual == -4:
            return False
    var root = proofs[0].root
    var actual = tk._infer(states[0], root)
    return (actual >= 0 and tk._infer(states[0], actual) == tk._put(states[0], tk.PT_PROP)
        and states[0].context_high[root] == 0 and states[0].context_high[actual] == 0
        and states[0].fuel > 0 and states[0].syntax.count < scratch
        and not states[0].arithmetic_verified and not states[0].fast_arithmetic)
