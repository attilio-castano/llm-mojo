"""Ordinary terms for total Nat complement and its bounded laws.

Comp(M,x) is truncated subtraction by constructor recursion. Conservation,
boundedness, injectivity and order reversal expose their domain hypotheses.
No native subtraction, order reflection or additional primitive participates.
Export IDs propose ordinary definitions and require independent admission.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.nat_score_link as link

comptime SCORE_COMPLEMENT_DEFINITIONS = 6


@fieldwise_init
struct ComplementExports(ImplicitlyCopyable):
    var comp: Int
    var conservation: Int
    var diagonal: Int
    var bounded: Int
    var injective: Int
    var weak_reversal: Int


def score_comp[C: Int](mut p: ph._Builder[C], exports: ComplementExports,
    maximum: Int, value: Int) -> Int:
    return ph._apps(p, ph._named(p, exports.comp), [maximum, value])


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var index = p.bundle.definition_count
    ph._declare(p, type, body, index)
    return index


def _call[C: Int](mut p: ph._Builder[C], definition: Int,
    arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _truth_proof[C: Int](mut p: ph._Builder[C]) -> Int:
    # Le(0,n) is the arithmetic module's Church True, not Eq(0,0).
    var proposition = ph._var(p, ph._prop(p))
    var evidence = ph._var(p, proposition)
    return ph._lams(p, [proposition, evidence], evidence)


def _transport_left[C: Int](mut p: ph._Builder[C], right: Int,
    equality: Int, evidence: Int) -> Int:
    var left = ph._var(p, ph._nat(p))
    return ph._subst(p, ph._lam(p, left, link.score_le(p, left, right)),
        equality, evidence)


def _transport_right[C: Int](mut p: ph._Builder[C], left: Int,
    equality: Int, evidence: Int) -> Int:
    var right = ph._var(p, ph._nat(p))
    return ph._subst(p, ph._lam(p, right, link.score_le(p, left, right)),
        equality, evidence)


def _le_self_sum[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    # 0 <= right, transported through suffix addition and commutativity:
    # Add(0,left) <= Add(right,left), hence left <= Add(left,right).
    var nonnegative = _call(p, link.SCORE_NONNEG, [right])
    var shifted = _call(p, link.SCORE_LE_ADD,
        [ph._zero(p), right, left, nonnegative])
    var zero_left = _call(p, link.SCORE_ADD_ZERO_LEFT, [left])
    var normalized_left = _transport_left(p,
        link.score_add(p, right, left), zero_left, shifted)
    var commutative = _call(p, link.SCORE_ADD_COMM, [right, left])
    return _transport_right(p, left, commutative, normalized_left)


def _complement_data[C: Int](mut p: ph._Builder[C]) -> Int:
    var maximum = ph._var(p, ph._nat(p))
    var value = ph._var(p, ph._nat(p))
    var base = ph._lam(p, value, ph._zero(p))
    var predecessor = ph._var(p, ph._nat(p))
    var previous = ph._var(p, ph._store(p))
    value = ph._var(p, ph._nat(p))
    var value_predecessor = ph._var(p, ph._nat(p))
    var ignored = ph._var(p, ph._nat(p))
    var value_step = ph._lams(p, [value_predecessor, ignored],
        ph._app(p, previous, value_predecessor))
    var step = ph._lams(p, [predecessor, previous, value],
        ph._node(p, tk.PT_NAT_REC, value, ph._succ(p, predecessor), value_step))
    return _declare(p, ph._pi(p, maximum, ph._store(p)),
        ph._lam(p, maximum, ph._node(p, tk.PT_NAT_REC, maximum, base, step)))


def _conservation_at[C: Int](mut p: ph._Builder[C], exports: ComplementExports,
    maximum: Int, value: Int) -> Int:
    var bound = ph._var(p, link.score_le(p, value, maximum))
    return ph._pi(p, bound, ph._eq(p,
        link.score_add(p, score_comp(p, exports, maximum, value), value), maximum))


def _conservation[C: Int](mut p: ph._Builder[C], exports: ComplementExports) -> Int:
    # Induct on the maximum, keeping the value and its bound universally
    # quantified inside the motive. The successor/successor case consumes
    # the predecessor's conservation proof, rather than assuming subtraction.
    var maximum = ph._var(p, ph._nat(p))
    var value = ph._var(p, ph._nat(p))
    var outer_motive = ph._lam(p, maximum,
        ph._pi(p, value, _conservation_at(p, exports, maximum, value)))

    value = ph._var(p, ph._nat(p))
    var zero_motive = ph._lam(p, value,
        _conservation_at(p, exports, ph._zero(p), value))
    var zero_bound = ph._var(p, link.score_le(p, ph._zero(p), ph._zero(p)))
    var zero_base = ph._lam(p, zero_bound, ph._refl(p, ph._zero(p)))
    var value_predecessor = ph._var(p, ph._nat(p))
    var zero_ih = ph._var(p, ph._app(p, zero_motive, value_predecessor))
    var value_successor = ph._succ(p, value_predecessor)
    var impossible = ph._var(p, link.score_le(p, value_successor, ph._zero(p)))
    var zero_equation = ph._eq(p, link.score_add(p,
        score_comp(p, exports, ph._zero(p), value_successor), value_successor),
        ph._zero(p))
    var zero_step = ph._lams(p, [value_predecessor, zero_ih, impossible],
        ph._app(p, impossible, zero_equation))
    value = ph._var(p, ph._nat(p))
    var outer_base = ph._lam(p, value,
        ph._induct(p, zero_motive, zero_base, zero_step, value))

    var maximum_predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, maximum_predecessor))
    var maximum_successor = ph._succ(p, maximum_predecessor)
    value = ph._var(p, ph._nat(p))
    var inner_motive = ph._lam(p, value,
        _conservation_at(p, exports, maximum_successor, value))
    zero_bound = ph._var(p, link.score_le(p, ph._zero(p), maximum_successor))
    var inner_base = ph._lam(p, zero_bound, ph._refl(p, maximum_successor))
    value_predecessor = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, value_predecessor))
    var next_bound = ph._var(p, link.score_le(p,
        ph._succ(p, value_predecessor), maximum_successor))
    var smaller_sum = link.score_add(p,
        score_comp(p, exports, maximum_predecessor, value_predecessor), value_predecessor)
    var smaller_equal = ph._apps(p, outer_ih, [value_predecessor, next_bound])
    var inner_step = ph._lams(p, [value_predecessor, inner_ih, next_bound],
        ph._succ_cong(p, smaller_sum, maximum_predecessor, smaller_equal))
    value = ph._var(p, ph._nat(p))
    var outer_step = ph._lams(p, [maximum_predecessor, outer_ih, value],
        ph._induct(p, inner_motive, inner_base, inner_step, value))
    maximum = ph._var(p, ph._nat(p))
    return _declare(p, ph._pi(p, maximum, ph._app(p, outer_motive, maximum)),
        ph._lam(p, maximum,
            ph._induct(p, outer_motive, outer_base, outer_step, maximum)))


def _diagonal[C: Int](mut p: ph._Builder[C], exports: ComplementExports) -> Int:
    var maximum = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, maximum,
        ph._eq(p, score_comp(p, exports, maximum, maximum), ph._zero(p)))
    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    var step = ph._lams(p, [predecessor, ih], ih)
    maximum = ph._var(p, ph._nat(p))
    return _declare(p, ph._pi(p, maximum, ph._app(p, motive, maximum)),
        ph._lam(p, maximum,
            ph._induct(p, motive, ph._refl(p, ph._zero(p)), step, maximum)))


def _bounded[C: Int](mut p: ph._Builder[C], exports: ComplementExports) -> Int:
    var maximum = ph._var(p, ph._nat(p))
    var value = ph._var(p, ph._nat(p))
    var bound = ph._var(p, link.score_le(p, value, maximum))
    var complement = score_comp(p, exports, maximum, value)
    var conservation = _call(p, exports.conservation, [maximum, value, bound])
    var result = _transport_right(p, complement, conservation,
        _le_self_sum(p, complement, value))
    return _declare(p, ph._pis(p, [maximum, value, bound],
        link.score_le(p, complement, maximum)),
        ph._lams(p, [maximum, value, bound], result))


def _injective[C: Int](mut p: ph._Builder[C], exports: ComplementExports) -> Int:
    var maximum = ph._var(p, ph._nat(p))
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var left_bound = ph._var(p, link.score_le(p, left, maximum))
    var right_bound = ph._var(p, link.score_le(p, right, maximum))
    var left_comp = score_comp(p, exports, maximum, left)
    var right_comp = score_comp(p, exports, maximum, right)
    var equal = ph._var(p, ph._eq(p, left_comp, right_comp))
    var left_sum = link.score_add(p, left_comp, left)
    var aligned_sum = link.score_add(p, right_comp, left)
    var right_sum = link.score_add(p, right_comp, right)
    var point = ph._var(p, ph._nat(p))
    var aligned = ph._cong(p, ph._lam(p, point, link.score_add(p, point, left)),
        left_comp, right_comp, equal)
    var left_conservation = _call(p, exports.conservation, [maximum, left, left_bound])
    var right_conservation = _call(p, exports.conservation, [maximum, right, right_bound])
    var aligned_to_maximum = ph._trans(p, aligned_sum, left_sum, maximum,
        ph._sym(p, left_sum, aligned_sum, aligned), left_conservation)
    var aligned_to_right = ph._trans(p, aligned_sum, maximum, right_sum,
        aligned_to_maximum, ph._sym(p, right_sum, maximum, right_conservation))
    # The relocated cancellation theorem removes a shared right suffix.
    # Commute the common complement into that position explicitly.
    var left_suffix_sum = link.score_add(p, left, right_comp)
    var right_suffix_sum = link.score_add(p, right, right_comp)
    var first = ph._trans(p, left_suffix_sum, aligned_sum, right_sum,
        _call(p, link.SCORE_ADD_COMM, [left, right_comp]), aligned_to_right)
    var suffix_equal = ph._trans(p, left_suffix_sum, right_sum, right_suffix_sum,
        first, _call(p, link.SCORE_ADD_COMM, [right_comp, right]))
    var result = _call(p, link.SCORE_ADD_CANCEL, [right_comp, left, right, suffix_equal])
    var binders: List[Int] = [maximum, left, right, left_bound, right_bound, equal]
    return _declare(p, ph._pis(p, binders, ph._eq(p, left, right)),
        ph._lams(p, binders, result))


def _reversal_at[C: Int](mut p: ph._Builder[C], exports: ComplementExports,
    maximum: Int, left: Int, right: Int) -> Int:
    var ordered = ph._var(p, link.score_le(p, left, right))
    var bound = ph._var(p, link.score_le(p, right, maximum))
    return ph._pis(p, [ordered, bound], link.score_le(p,
        score_comp(p, exports, maximum, right), score_comp(p, exports, maximum, left)))


def _weak_reversal[C: Int](mut p: ph._Builder[C], exports: ComplementExports) -> Int:
    var maximum = ph._var(p, ph._nat(p))
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var outer_motive = ph._lam(p, maximum, ph._pis(p, [left, right],
        _reversal_at(p, exports, maximum, left, right)))
    left = ph._var(p, ph._nat(p))
    right = ph._var(p, ph._nat(p))
    var ordered = ph._var(p, link.score_le(p, left, right))
    var bound = ph._var(p, link.score_le(p, right, ph._zero(p)))
    # Both complements reduce to zero for a zero maximum, independently
    # of the two hypotheses. Their order is Church True.
    var outer_base = ph._lams(p, [left, right, ordered, bound], _truth_proof(p))

    var maximum_predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, maximum_predecessor))
    var maximum_successor = ph._succ(p, maximum_predecessor)
    left = ph._var(p, ph._nat(p))
    right = ph._var(p, ph._nat(p))
    var middle_motive = ph._lam(p, left, ph._pi(p, right,
        _reversal_at(p, exports, maximum_successor, left, right)))
    right = ph._var(p, ph._nat(p))
    ordered = ph._var(p, link.score_le(p, ph._zero(p), right))
    bound = ph._var(p, link.score_le(p, right, maximum_successor))
    var middle_base = ph._lams(p, [right, ordered, bound],
        _call(p, exports.bounded, [maximum_successor, right, bound]))

    var left_predecessor = ph._var(p, ph._nat(p))
    var middle_ih = ph._var(p, ph._app(p, middle_motive, left_predecessor))
    var left_successor = ph._succ(p, left_predecessor)
    right = ph._var(p, ph._nat(p))
    var inner_motive = ph._lam(p, right,
        _reversal_at(p, exports, maximum_successor, left_successor, right))
    var impossible = ph._var(p, link.score_le(p, left_successor, ph._zero(p)))
    bound = ph._var(p, link.score_le(p, ph._zero(p), maximum_successor))
    var zero_result = link.score_le(p,
        score_comp(p, exports, maximum_successor, ph._zero(p)),
        score_comp(p, exports, maximum_successor, left_successor))
    var inner_base = ph._lams(p, [impossible, bound], ph._app(p, impossible, zero_result))
    var right_predecessor = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, right_predecessor))
    ordered = ph._var(p, link.score_le(p, left_successor, ph._succ(p, right_predecessor)))
    bound = ph._var(p, link.score_le(p, ph._succ(p, right_predecessor), maximum_successor))
    var inner_step = ph._lams(p, [right_predecessor, inner_ih, ordered, bound],
        ph._apps(p, outer_ih, [left_predecessor, right_predecessor, ordered, bound]))
    right = ph._var(p, ph._nat(p))
    var middle_step = ph._lams(p, [left_predecessor, middle_ih, right],
        ph._induct(p, inner_motive, inner_base, inner_step, right))
    left = ph._var(p, ph._nat(p))
    var outer_step = ph._lams(p, [maximum_predecessor, outer_ih, left],
        ph._induct(p, middle_motive, middle_base, middle_step, left))
    maximum = ph._var(p, ph._nat(p))
    return _declare(p, ph._pi(p, maximum, ph._app(p, outer_motive, maximum)),
        ph._lam(p, maximum,
            ph._induct(p, outer_motive, outer_base, outer_step, maximum)))


def append_nat_score_complement[C: Int](mut p: ph._Builder[C]) -> ComplementExports:
    if p.bundle.count <= 0 or p.bundle.definition_count < link.NAT_SCORE_DEFINITIONS:
        p.bundle.count = -1
        return ComplementExports(-1, -1, -1, -1, -1, -1)
    # Dynamic IDs permit subsequent modules to append without reserved gaps.
    var first = p.bundle.definition_count
    var exports = ComplementExports(_complement_data(p), -1, -1, -1, -1, -1)
    exports.conservation = _conservation(p, exports)
    exports.diagonal = _diagonal(p, exports)
    exports.bounded = _bounded(p, exports)
    exports.injective = _injective(p, exports)
    exports.weak_reversal = _weak_reversal(p, exports)
    if p.bundle.count < 0 or p.bundle.definition_count != first + SCORE_COMPLEMENT_DEFINITIONS:
        p.bundle.count = -1
        return ComplementExports(-1, -1, -1, -1, -1, -1)
    return exports
