"""Untrusted ordinary comparison proofs for the symbolic record bridge.

The Church-valued arithmetic order and the existing Before proposition are
related by checked proof terms. They are not treated as definitionally equal.
Native comparisons and machine words remain separate refinement obligations.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.finishing_execution_leaves as leaf
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_join_bool as bo
import llm_mojo.kernels.nat_score_link as score


@fieldwise_init
struct ComparisonExports(ImplicitlyCopyable):
    var before_to_lt: Int
    var lt_to_before: Int
    var before_flag_bit: Int
    var before_flag_complete: Int
    var equality_flag_bit: Int
    var equality_flag_zero_sound: Int
    var lt_flag_one: Int
    var reverse_le_flag_zero: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int,
    args: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), args)


def _goal[C: Int](mut p: ph._Builder[C], mode: Int,
    left: Int, right: Int) -> Int:
    var before = a._before(p, left, right)
    var strict = score.score_le(p, ph._succ(p, left), right)
    if mode < 2:
        var evidence = ph._var(p, before if mode == 0 else strict)
        return ph._pi(p, evidence, strict if mode == 0 else before)
    var flag = leaf.execution_before_flag(p, left, right)
    if mode == 2:
        return bits.record_is_bit(p, flag)
    var evidence = ph._var(p, before)
    return ph._pi(p, evidence, ph._eq(p, flag, ph._one(p)))


def _base_case[C: Int](mut p: ph._Builder[C], mode: Int,
    left: Int, right: Int, right_is_zero: Bool) -> Int:
    if mode == 2:
        return bo._intro(p, not right_is_zero)
    var before = a._before(p, left, right)
    var strict = score.score_le(p, ph._succ(p, left), right)
    var premise = before if mode != 1 else strict
    var evidence = ph._var(p, premise)
    var result: Int
    if right_is_zero:
        if mode < 2:
            result = evidence
        else:
            result = ph._app(p, evidence, ph._eq(p,
                leaf.execution_before_flag(p, left, right), ph._one(p)))
    elif mode == 0:
        # This branch is used only at left=0,right=succ(k). Pred is ordinary
        # Nat recursion and conversion reduces Le(1,succ(k)) to Le(0,k).
        var predecessor = ph._app(p, ph._named(p, ph.PHASE_PREDECESSOR), right)
        result = _call(p, score.SCORE_NONNEG, [predecessor])
    elif mode == 1:
        result = ph._refl(p, ph._zero(p))
    else:
        result = ph._refl(p, ph._one(p))
    return ph._lam(p, evidence, result)


def _nested_comparison[C: Int](mut p: ph._Builder[C], mode: Int) -> Int:
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, left, ph._pi(p, right, _goal(p, mode, left, right)))

    right = ph._var(p, ph._nat(p))
    var inner = ph._lam(p, right, _goal(p, mode, ph._zero(p), right))
    var base = _base_case(p, mode, ph._zero(p), ph._zero(p), True)
    var k = ph._var(p, ph._nat(p))
    var ignored = ph._var(p, ph._app(p, inner, k))
    var step = ph._lams(p, [k, ignored],
        _base_case(p, mode, ph._zero(p), ph._succ(p, k), False))
    var outer_base = ph._lam(p, right, ph._induct(p, inner, base, step, right))

    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    right = ph._var(p, ph._nat(p))
    var successor = ph._succ(p, predecessor)
    inner = ph._lam(p, right, _goal(p, mode, successor, right))
    base = _base_case(p, mode, successor, ph._zero(p), True)
    k = ph._var(p, ph._nat(p))
    ignored = ph._var(p, ph._app(p, inner, k))
    step = ph._lams(p, [k, ignored], ph._app(p, ih, k))
    var outer_step = ph._lams(p, [predecessor, ih, right],
        ph._induct(p, inner, base, step, right))
    left = ph._var(p, ph._nat(p))
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pi(p, left, ph._app(p, motive, left)),
        ph._lam(p, left, ph._induct(p, motive, outer_base, outer_step, left)), definition)
    return definition


def _bit_from_equal[C: Int](mut p: ph._Builder[C], value: Int,
    one: Bool, evidence: Int) -> Int:
    var endpoint = ph._one(p) if one else ph._zero(p)
    var slot = ph._var(p, ph._nat(p))
    return ph._subst(p, ph._lam(p, slot, bits.record_is_bit(p, slot)),
        ph._sym(p, value, endpoint, evidence), bo._intro(p, one))


def _equality_flag[C: Int](mut p: ph._Builder[C], zero_sound: Bool) -> Int:
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var flag = ph._eqb(p, left, right)
    var zero = ph._var(p, ph._eq(p, flag, ph._zero(p))) if zero_sound else -1
    var goal = ph._eq(p, left, right) if zero_sound else bits.record_is_bit(p, flag)
    var equal = ph._var(p, ph._eq(p, left, right))
    var equal_body = equal
    if not zero_sound:
        var slot = ph._var(p, ph._nat(p))
        var flag_zero = ph._subst(p,
            ph._lam(p, slot, ph._eq(p, ph._eqb(p, left, slot), ph._zero(p))),
            equal, _call(p, ph.PHASE_EQB_REFL, [left]))
        equal_body = _bit_from_equal(p, flag, False, flag_zero)
    var yes = ph._lam(p, equal, equal_body)
    var apart = ph._var(p, ph._apart(p, left, right))
    var flag_one = _call(p, ph.PHASE_APART_TO_EQB, [left, right, apart])
    var apart_body: Int
    if zero_sound:
        var zero_one = ph._trans(p, ph._zero(p), flag, ph._one(p),
            ph._sym(p, flag, ph._zero(p), zero), flag_one)
        var impossible = _call(p, ph.PHASE_ZERO_SUCC, [ph._zero(p), zero_one])
        apart_body = ph._app(p, impossible, goal)
    else:
        apart_body = _bit_from_equal(p, flag, True, flag_one)
    var no = ph._lam(p, apart, apart_body)
    var body = _call(p, ph.PHASE_DECIDE, [left, right, goal, yes, no])
    var binders: List[Int] = [left, right]
    if zero_sound:
        binders.append(zero)
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, goal), ph._lams(p, binders, body), definition)
    return definition


def _strict_flag[C: Int](mut p: ph._Builder[C], lt_to_before: Int,
    complete: Int) -> Int:
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var strict = ph._var(p, score.score_le(p, ph._succ(p, left), right))
    var before = _call(p, lt_to_before, [left, right, strict])
    var result = ph._eq(p, leaf.execution_before_flag(p, left, right), ph._one(p))
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, [left, right, strict], result),
        ph._lams(p, [left, right, strict], _call(p, complete, [left, right, before])), definition)
    return definition


def _reverse_flag[C: Int](mut p: ph._Builder[C], before_to_lt: Int,
    flag_bit: Int) -> Int:
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var weak = ph._var(p, score.score_le(p, right, left))
    var flag = leaf.execution_before_flag(p, left, right)
    var result = ph._eq(p, flag, ph._zero(p))
    var zero = ph._var(p, ph._eq(p, flag, ph._zero(p)))
    var yes = ph._lam(p, zero, zero)
    var one = ph._var(p, ph._eq(p, flag, ph._one(p)))
    var before = _call(p, leaf.EXEC_FLAG_BOUND_SOUND, [left, right, one])
    var strict = _call(p, before_to_lt, [left, right, before])
    var loop = _call(p, score.SCORE_LE_TRANS,
        [ph._succ(p, left), right, left, strict, weak])
    var impossible = _call(p, score.SCORE_LT_IRREFL, [left, loop])
    var no = ph._lam(p, one, ph._app(p, impossible, result))
    var body = ph._apps(p, _call(p, flag_bit, [left, right]), [result, yes, no])
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, [left, right, weak], result),
        ph._lams(p, [left, right, weak], body), definition)
    return definition


def append_nat_comparison_terms[C: Int](mut p: ph._Builder[C]) -> ComparisonExports:
    var forward = _nested_comparison(p, 0)
    var backward = _nested_comparison(p, 1)
    var flag_bit = _nested_comparison(p, 2)
    var complete = _nested_comparison(p, 3)
    var equality_bit = _equality_flag(p, False)
    var equality_zero = _equality_flag(p, True)
    var strict = _strict_flag(p, backward, complete)
    var reverse = _reverse_flag(p, forward, flag_bit)
    return ComparisonExports(forward, backward, flag_bit, complete,
        equality_bit, equality_zero, strict, reverse)
