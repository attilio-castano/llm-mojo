"""Nat-valued record selection and its derived score-maximum equation.

B wins when its rank is greater, or ranks agree and its token is smaller.
The equation follows from checked Nat comparison and order proofs. Rank and
token components share this exact selector and operand tuple. Native UInt32
comparisons, bitwise OR, Mojo source and GPU correspondence remain separate.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_leaves as leaf
import llm_mojo.kernels.nat_join_max as maximum
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_join_bool as boolean
import llm_mojo.kernels.nat_score_link as arithmetic
import llm_mojo.kernels.nat_score_order as order
import llm_mojo.kernels.nat_comparison_bridge as comparison
import llm_mojo.kernels.nat_score_embedding as embedding

comptime RECORD_SELECTOR_DEFINITIONS = 3


@fieldwise_init
struct RecordSelectorExports(ImplicitlyCopyable):
    var flag: Int
    var flag_bit: Int
    var max_selected: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _flag[C: Int](mut p: ph._Builder[C], definition: Int,
    ra: Int, rb: Int, ta: Int, tb: Int) -> Int:
    return _call(p, definition, [ra, rb, ta, tb])


def record_b_wins[C: Int](mut p: ph._Builder[C], exports: RecordSelectorExports,
    ra: Int, rb: Int, ta: Int, tb: Int) -> Int:
    return _flag(p, exports.flag, ra, rb, ta, tb)


def _tie_flag[C: Int](mut p: ph._Builder[C], ra: Int, rb: Int,
    ta: Int, tb: Int) -> Int:
    # EqB is zero on equality; Select takes its first branch at zero.
    return ph._select(p, ph._eqb(p, ra, rb),
        leaf.execution_before_flag(p, tb, ta), ph._zero(p))


def _data[C: Int](mut p: ph._Builder[C]) -> Int:
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var binders: List[Int] = [ra, rb, ta, tb]
    var body = bits.record_bool_or(p, leaf.execution_before_flag(p, ra, rb),
        _tie_flag(p, ra, rb, ta, tb))
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, ph._nat(p)), ph._lams(p, binders, body), definition)
    return definition


def _bit[C: Int](mut p: ph._Builder[C], flag: Int,
    comparisons: comparison.ComparisonExports) -> Int:
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var rank_flag = leaf.execution_before_flag(p, ra, rb)
    var token_flag = leaf.execution_before_flag(p, tb, ta)
    var equality_flag = ph._eqb(p, ra, rb)
    var equality_bit = _call(p, comparisons.equality_flag_bit, [ra, rb])
    var rank_bit = _call(p, comparisons.before_flag_bit, [ra, rb])
    var token_bit = _call(p, comparisons.before_flag_bit, [tb, ta])
    var slot = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, slot, bits.record_is_bit(p,
        ph._select(p, slot, token_flag, ph._zero(p))))
    var tie_bit = bits._bit_transport(p, equality_flag, equality_bit,
        motive, token_bit, boolean._intro(p, False))
    var tie = _tie_flag(p, ra, rb, ta, tb)
    # Expand the existing ordinary Boolean closure proof producer. Its leaves
    # are constructor cases and reflexivity; no new Boolean rule is installed.
    var body = boolean._cases(p, 0, [rank_flag, tie], [rank_bit, tie_bit], 0)
    var binders: List[Int] = [ra, rb, ta, tb]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, bits.record_is_bit(p,
        _flag(p, flag, ra, rb, ta, tb))), ph._lams(p, binders, body), definition)
    return definition


def _rank_flag_one[C: Int](mut p: ph._Builder[C], ra: Int,
    rb: Int, ta: Int, tb: Int, evidence: Int) -> Int:
    var rank_flag = leaf.execution_before_flag(p, ra, rb)
    var tie = _tie_flag(p, ra, rb, ta, tb)
    var slot = ph._var(p, ph._nat(p))
    return ph._cong(p, ph._lam(p, slot, bits.record_bool_or(p, slot, tie)),
        rank_flag, ph._one(p), evidence)


def _rank_flag_zero[C: Int](mut p: ph._Builder[C], flag: Int,
    ra: Int, rb: Int, ta: Int, tb: Int, rank_zero: Int,
    equality_endpoint: Int, equality_proof: Int) -> Int:
    # Once the rank flag is zero, BoolOr reduces to its tie argument. The EqB
    # equation then reduces that argument to token_flag (0) or zero (1).
    var rank_flag = leaf.execution_before_flag(p, ra, rb)
    var token_flag = leaf.execution_before_flag(p, tb, ta)
    var equality_flag = ph._eqb(p, ra, rb)
    var tie = _tie_flag(p, ra, rb, ta, tb)
    var slot = ph._var(p, ph._nat(p))
    var first = ph._cong(p, ph._lam(p, slot, bits.record_bool_or(p, slot, tie)),
        rank_flag, ph._zero(p), rank_zero)
    slot = ph._var(p, ph._nat(p))
    var select = ph._lam(p, slot, ph._select(p, slot, token_flag, ph._zero(p)))
    var last = ph._cong(p, select, equality_flag, equality_endpoint, equality_proof)
    var target = ph._select(p, equality_endpoint, token_flag, ph._zero(p))
    return ph._trans(p, _flag(p, flag, ra, rb, ta, tb), tie, target, first, last)


def _reverse_strict_equality_flag[C: Int](mut p: ph._Builder[C],
    ra: Int, rb: Int, strict: Int) -> Int:
    # Le(S rb,ra) and ra=rb would yield rb<rb. Apart is an ordinary
    # Eq(ra,rb)->False proof, consumed by the checked phase EqB theorem.
    var equal = ph._var(p, ph._eq(p, ra, rb))
    var slot = ph._var(p, ph._nat(p))
    var loop = ph._subst(p, ph._lam(p, slot,
        arithmetic.score_le(p, ph._succ(p, rb), slot)), equal, strict)
    var impossible = _call(p, arithmetic.SCORE_LT_IRREFL, [rb, loop])
    var apart = ph._lam(p, equal, impossible)
    return _call(p, ph.PHASE_APART_TO_EQB, [ra, rb, apart])


def _equal_rank_equality_flag[C: Int](mut p: ph._Builder[C],
    ra: Int, rb: Int, equal: Int) -> Int:
    var slot = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, slot,
        ph._eq(p, ph._eqb(p, ra, slot), ph._zero(p)))
    return ph._subst(p, motive, equal, _call(p, ph.PHASE_EQB_REFL, [ra]))


def _selection_result[C: Int](mut p: ph._Builder[C], flag: Int,
    sa: Int, sb: Int, right: Bool, flag_equal: Int, maximum_equal: Int) -> Int:
    var endpoint = ph._one(p) if right else ph._zero(p)
    var selected = sb if right else sa
    var slot = ph._var(p, ph._nat(p))
    var select = ph._lam(p, slot, ph._select(p, slot, sa, sb))
    var select_equal = ph._cong(p, select, flag, endpoint, flag_equal)
    var output = ph._select(p, flag, sa, sb)
    return ph._trans(p, maximum.record_max(p, sa, sb), selected, output,
        maximum_equal, ph._sym(p, output, selected, select_equal))


def _max_selected[C: Int](mut p: ph._Builder[C], flag: Int,
    comparisons: comparison.ComparisonExports, helpers: order.ScoreOrderExports,
    scores: embedding.ScoreEmbeddingExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ta_bound = ph._var(p, arithmetic.score_le(p, ta, bound))
    var tb_bound = ph._var(p, arithmetic.score_le(p, tb, bound))
    var sa = embedding.score_value(p, scores, bound, ra, ta)
    var sb = embedding.score_value(p, scores, bound, rb, tb)
    var actual_flag = _flag(p, flag, ra, rb, ta, tb)
    var goal = ph._eq(p, maximum.record_max(p, sa, sb), ph._select(p, actual_flag, sa, sb))

    # First branch: ra<rb. Rank dominance proves sa<sb; rank flag=1 makes
    # B win independently of the EqB and token branches.
    var ranks_ab = ph._var(p, arithmetic.score_le(p, ph._succ(p, ra), rb))
    var score_ab = _call(p, scores.rank_dominance,
        [bound, ra, rb, ta, tb, ta_bound, tb_bound, ranks_ab])
    var rank_one = _call(p, comparisons.lt_flag_one, [ra, rb, ranks_ab])
    var flag_one = _rank_flag_one(p, ra, rb, ta, tb, rank_one)
    var max_b = _call(p, helpers.max_right_lt, [sa, sb, score_ab])
    var yes = ph._lam(p, ranks_ab,
        _selection_result(p, actual_flag, sa, sb, True, flag_one, max_b))

    # Otherwise rb<=ra. A second checked order elimination distinguishes the
    # strict reverse rank order from equality; no native comparison branches.
    var ranks_ba_weak = ph._var(p, arithmetic.score_le(p, rb, ra))
    var ranks_ba = ph._var(p, arithmetic.score_le(p, ph._succ(p, rb), ra))
    var score_ba = _call(p, scores.rank_dominance,
        [bound, rb, ra, tb, ta, tb_bound, ta_bound, ranks_ba])
    var rank_zero = _call(p, comparisons.reverse_le_flag_zero, [ra, rb, ranks_ba_weak])
    var equality_one = _reverse_strict_equality_flag(p, ra, rb, ranks_ba)
    var flag_zero = _rank_flag_zero(p, flag, ra, rb, ta, tb, rank_zero,
        ph._one(p), equality_one)
    var max_a = _call(p, helpers.max_left_lt, [sa, sb, score_ba])
    var reverse = ph._lam(p, ranks_ba,
        _selection_result(p, actual_flag, sa, sb, False, flag_zero, max_a))

    # Equal ranks: derive Eq(ra,rb) from both weak-order premises, then reduce
    # the actual flag to BeforeFlag(tb,ta). Score transports use this equality.
    var ranks_ab_weak = ph._var(p, arithmetic.score_le(p, ra, rb))
    var ranks_equal = _call(p, arithmetic.SCORE_LE_ANTISYM,
        [ra, rb, ranks_ab_weak, ranks_ba_weak])
    var equality_zero = _equal_rank_equality_flag(p, ra, rb, ranks_equal)
    var flag_token = _rank_flag_zero(p, flag, ra, rb, ta, tb,
        rank_zero, ph._zero(p), equality_zero)
    var token_flag = leaf.execution_before_flag(p, tb, ta)

    var token_ba = ph._var(p, arithmetic.score_le(p, ph._succ(p, tb), ta))
    var token_scores = _call(p, scores.token_dominance,
        [bound, ra, ta, tb, ta_bound, tb_bound, token_ba])
    var slot = ph._var(p, ph._nat(p))
    var strict_motive = ph._lam(p, slot, arithmetic.score_le(p, ph._succ(p, sa),
        embedding.score_value(p, scores, bound, slot, tb)))
    var strict_scores = ph._subst(p, strict_motive, ranks_equal, token_scores)
    var token_one = _call(p, comparisons.lt_flag_one, [tb, ta, token_ba])
    flag_one = ph._trans(p, actual_flag, token_flag, ph._one(p), flag_token, token_one)
    max_b = _call(p, helpers.max_right_lt, [sa, sb, strict_scores])
    var token_yes = ph._lam(p, token_ba,
        _selection_result(p, actual_flag, sa, sb, True, flag_one, max_b))

    var token_ab_weak = ph._var(p, arithmetic.score_le(p, ta, tb))
    token_scores = _call(p, scores.token_weak,
        [bound, ra, ta, tb, ta_bound, tb_bound, token_ab_weak])
    slot = ph._var(p, ph._nat(p))
    var weak_motive = ph._lam(p, slot, arithmetic.score_le(p,
        embedding.score_value(p, scores, bound, slot, tb), sa))
    var weak_scores = ph._subst(p, weak_motive, ranks_equal, token_scores)
    var token_zero = _call(p, comparisons.reverse_le_flag_zero, [tb, ta, token_ab_weak])
    flag_zero = ph._trans(p, actual_flag, token_flag, ph._zero(p), flag_token, token_zero)
    max_a = _call(p, helpers.max_left_le, [sa, sb, weak_scores])
    var token_no = ph._lam(p, token_ab_weak,
        _selection_result(p, actual_flag, sa, sb, False, flag_zero, max_a))
    var tokens = _call(p, arithmetic.SCORE_ORDER_CASES, [tb, ta, goal, token_yes, token_no])
    var equal = ph._lam(p, ranks_ab_weak, tokens)
    var reversed_or_equal = _call(p, arithmetic.SCORE_ORDER_CASES, [rb, ra, goal, reverse, equal])
    var no = ph._lam(p, ranks_ba_weak, reversed_or_equal)
    var body = _call(p, arithmetic.SCORE_ORDER_CASES, [ra, rb, goal, yes, no])
    var binders: List[Int] = [bound, ra, rb, ta, tb, ta_bound, tb_bound]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, goal), ph._lams(p, binders, body), definition)
    return definition


def append_nat_record_selector[C: Int](mut p: ph._Builder[C],
    comparisons: comparison.ComparisonExports, helpers: order.ScoreOrderExports,
    scores: embedding.ScoreEmbeddingExports) -> RecordSelectorExports:
    var first = p.bundle.definition_count
    var flag = _data(p)
    var flag_bit = _bit(p, flag, comparisons)
    var max_selected = _max_selected(p, flag, comparisons, helpers, scores)
    if p.bundle.count < 0 or p.bundle.definition_count != first + RECORD_SELECTOR_DEFINITIONS:
        p.bundle.count = -1
        return RecordSelectorExports(-1, -1, -1)
    return RecordSelectorExports(flag, flag_bit, max_selected)
