"""Ordinary symbolic bounded-token score proofs; no native arithmetic oracle.

The bound M is arbitrary Nat. Scores may exceed native machine integers.
Admission and independent specification checking are required before use.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.nat_score_link as ar
import llm_mojo.kernels.nat_score_complement as co
import llm_mojo.kernels.nat_score_order as order


@fieldwise_init
struct ScoreEmbeddingExports(ImplicitlyCopyable):
    var score: Int
    var strict_complement: Int
    var rank_dominance: Int
    var token_dominance: Int
    var token_weak: Int
    var score_injective: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, args: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), args)


def score_value[C: Int](mut p: ph._Builder[C], exports: ScoreEmbeddingExports,
    bound: Int, rank: Int, token: Int) -> Int:
    return _call(p, exports.score, [bound, rank, token])


def _value[C: Int](mut p: ph._Builder[C], definition: Int,
    bound: Int, rank: Int, token: Int) -> Int:
    return _call(p, definition, [bound, rank, token])


def _lt[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return ar.score_le(p, ph._succ(p, left), right)


def _transport_left[C: Int](mut p: ph._Builder[C], right: Int,
    equality: Int, evidence: Int) -> Int:
    var slot = ph._var(p, ph._nat(p))
    return ph._subst(p, ph._lam(p, slot, ar.score_le(p, slot, right)), equality, evidence)


def _transport_right[C: Int](mut p: ph._Builder[C], left: Int,
    equality: Int, evidence: Int) -> Int:
    var slot = ph._var(p, ph._nat(p))
    return ph._subst(p, ph._lam(p, slot, ar.score_le(p, left, slot)), equality, evidence)


def _data[C: Int](mut p: ph._Builder[C], complement: co.ComplementExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var rank = ph._var(p, ph._nat(p))
    var token = ph._var(p, ph._nat(p))
    var body = ar.score_add(p, ar.score_mul(p, rank, ph._succ(p, bound)),
        co.score_comp(p, complement, bound, token))
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, [bound, rank, token], ph._nat(p)),
        ph._lams(p, [bound, rank, token], body), definition)
    return definition


def _strict_complement[C: Int](mut p: ph._Builder[C], complement: co.ComplementExports,
    helpers: order.ScoreOrderExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var x = ph._var(p, ph._nat(p))
    var y = ph._var(p, ph._nat(p))
    var strict = ph._var(p, _lt(p, x, y))
    var bounded = ph._var(p, ar.score_le(p, y, bound))
    var cx = co.score_comp(p, complement, bound, x)
    var cy = co.score_comp(p, complement, bound, y)
    var result = _lt(p, cy, cx)
    var weak = _call(p, helpers.lt_to_le, [x, y, strict])
    var x_bound = _call(p, ar.SCORE_LE_TRANS, [x, y, bound, weak, bounded])
    var reversed = _call(p, complement.weak_reversal, [bound, x, y, weak, bounded])
    var success = ph._var(p, result)
    var yes = ph._lam(p, success, success)
    var opposite = ph._var(p, ar.score_le(p, cx, cy))
    var comp_equal = _call(p, ar.SCORE_LE_ANTISYM, [cx, cy, opposite, reversed])
    var equal = _call(p, complement.injective, [bound, x, y, x_bound, bounded, comp_equal])
    var loop = _transport_right(p, ph._succ(p, x), ph._sym(p, x, y, equal), strict)
    var impossible = _call(p, ar.SCORE_LT_IRREFL, [x, loop])
    var no = ph._lam(p, opposite, ph._app(p, impossible, result))
    var body = _call(p, ar.SCORE_ORDER_CASES, [cy, cx, result, yes, no])
    var binders: List[Int] = [bound, x, y, strict, bounded]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body), definition)
    return definition


def _rank_dominance[C: Int](mut p: ph._Builder[C], complement: co.ComplementExports,
    helpers: order.ScoreOrderExports, score: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ta_bound = ph._var(p, ar.score_le(p, ta, bound))
    var tb_bound = ph._var(p, ar.score_le(p, tb, bound))
    var ranks = ph._var(p, _lt(p, ra, rb))
    var radix = ph._succ(p, bound)
    var pa = ar.score_mul(p, ra, radix)
    var pb = ar.score_mul(p, rb, radix)
    var ca = co.score_comp(p, complement, bound, ta)
    var cb = co.score_comp(p, complement, bound, tb)
    var sa = _value(p, score, bound, ra, ta)
    var sb = _value(p, score, bound, rb, tb)
    var next_product = ar.score_mul(p, ph._succ(p, ra), radix)
    var boundary = ar.score_add(p, pa, radix)
    var ca_bound = _call(p, complement.bounded, [bound, ta, ta_bound])
    var first = _call(p, helpers.prefix_lt, [pa, ca, radix, ca_bound])
    var product_eq = _call(p, ar.SCORE_MUL_SUCC_LEFT, [ra, radix])
    first = _transport_right(p, ph._succ(p, sa),
        ph._sym(p, next_product, boundary, product_eq), first)
    var second = _call(p, helpers.mul_left_le, [ph._succ(p, ra), rb, radix, ranks])
    var third = _call(p, helpers.prefix_le, [pb, ph._zero(p), cb,
        _call(p, ar.SCORE_NONNEG, [cb])])
    var joined = _call(p, ar.SCORE_LE_TRANS,
        [ph._succ(p, sa), next_product, pb, first, second])
    var body = _call(p, ar.SCORE_LE_TRANS, [ph._succ(p, sa), pb, sb, joined, third])
    var binders: List[Int] = [bound, ra, rb, ta, tb, ta_bound, tb_bound, ranks]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, _lt(p, sa, sb)), ph._lams(p, binders, body), definition)
    return definition


def _token_order[C: Int](mut p: ph._Builder[C], complement: co.ComplementExports,
    helpers: order.ScoreOrderExports, score: Int, strict_complement: Int, strict: Bool) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var rank = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ba = ph._var(p, ar.score_le(p, ta, bound))
    var bb = ph._var(p, ar.score_le(p, tb, bound))
    var token_order = ph._var(p, _lt(p, tb, ta) if strict else ar.score_le(p, ta, tb))
    var ca = co.score_comp(p, complement, bound, ta)
    var cb = co.score_comp(p, complement, bound, tb)
    var prefix = ar.score_mul(p, rank, ph._succ(p, bound))
    var sa = _value(p, score, bound, rank, ta)
    var sb = _value(p, score, bound, rank, tb)
    var result: Int
    var body: Int
    if strict:
        var comp_order = _call(p, strict_complement, [bound, tb, ta, token_order, ba])
        body = _call(p, helpers.prefix_lt, [prefix, ca, cb, comp_order])
        result = _lt(p, sa, sb)
    else:
        var comp_order = _call(p, complement.weak_reversal, [bound, ta, tb, token_order, bb])
        body = _call(p, helpers.prefix_le, [prefix, cb, ca, comp_order])
        result = ar.score_le(p, sb, sa)
    var binders: List[Int] = [bound, rank, ta, tb, ba, bb, token_order]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body), definition)
    return definition


def _score_injective[C: Int](mut p: ph._Builder[C], complement: co.ComplementExports,
    score: Int, rank_dominance: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ba = ph._var(p, ar.score_le(p, ta, bound))
    var bb = ph._var(p, ar.score_le(p, tb, bound))
    var sa = _value(p, score, bound, ra, ta)
    var sb = _value(p, score, bound, rb, tb)
    var equal = ph._var(p, ph._eq(p, sa, sb))
    var rank_goal = ph._eq(p, ra, rb)
    var strict_ab = ph._var(p, _lt(p, ra, rb))
    var strict_scores = _call(p, rank_dominance, [bound, ra, rb, ta, tb, ba, bb, strict_ab])
    var loop = _transport_right(p, ph._succ(p, sa), ph._sym(p, sa, sb, equal), strict_scores)
    var impossible = _call(p, ar.SCORE_LT_IRREFL, [sa, loop])
    var first_case = ph._lam(p, strict_ab, ph._app(p, impossible, rank_goal))

    var weak_ba = ph._var(p, ar.score_le(p, rb, ra))
    var strict_ba = ph._var(p, _lt(p, rb, ra))
    strict_scores = _call(p, rank_dominance, [bound, rb, ra, tb, ta, bb, ba, strict_ba])
    loop = _transport_right(p, ph._succ(p, sb), equal, strict_scores)
    impossible = _call(p, ar.SCORE_LT_IRREFL, [sb, loop])
    var reverse_case = ph._lam(p, strict_ba, ph._app(p, impossible, rank_goal))
    var weak_ab = ph._var(p, ar.score_le(p, ra, rb))
    var same = ph._lam(p, weak_ab, _call(p, ar.SCORE_LE_ANTISYM, [ra, rb, weak_ab, weak_ba]))
    var second_case = ph._lam(p, weak_ba,
        _call(p, ar.SCORE_ORDER_CASES, [rb, ra, rank_goal, reverse_case, same]))
    var rank_equal = _call(p, ar.SCORE_ORDER_CASES, [ra, rb, rank_goal, first_case, second_case])

    var slot = ph._var(p, ph._nat(p))
    var sb_at_ra = _value(p, score, bound, ra, tb)
    var rank_score = ph._cong(p, ph._lam(p, slot, _value(p, score, bound, slot, tb)),
        ra, rb, rank_equal)
    var fixed_equal = ph._trans(p, sa, sb, sb_at_ra, equal,
        ph._sym(p, sb_at_ra, sb, rank_score))
    var product = ar.score_mul(p, ra, ph._succ(p, bound))
    var ca = co.score_comp(p, complement, bound, ta)
    var cb = co.score_comp(p, complement, bound, tb)
    var ca_suffix = ar.score_add(p, ca, product)
    var cb_suffix = ar.score_add(p, cb, product)
    var first = _call(p, ar.SCORE_ADD_COMM, [ca, product])
    var last = _call(p, ar.SCORE_ADD_COMM, [product, cb])
    var suffix_equal = ph._trans(p, ca_suffix, sa, sb_at_ra, first, fixed_equal)
    suffix_equal = ph._trans(p, ca_suffix, sb_at_ra, cb_suffix, suffix_equal, last)
    var comp_equal = _call(p, ar.SCORE_ADD_CANCEL, [product, ca, cb, suffix_equal])
    var token_equal = _call(p, complement.injective, [bound, ta, tb, ba, bb, comp_equal])
    var props: List[Int] = [rank_goal, ph._eq(p, ta, tb)]
    var result = a._and_type(p, props)
    var body = a._and_intro(p, props, [rank_equal, token_equal])
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb, equal]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body), definition)
    return definition


def append_nat_score_embedding[C: Int](mut p: ph._Builder[C],
    complement: co.ComplementExports, helpers: order.ScoreOrderExports) -> ScoreEmbeddingExports:
    var score = _data(p, complement)
    var comp_strict = _strict_complement(p, complement, helpers)
    var rank = _rank_dominance(p, complement, helpers, score)
    var token = _token_order(p, complement, helpers, score, comp_strict, True)
    var token_weak = _token_order(p, complement, helpers, score, comp_strict, False)
    var injective = _score_injective(p, complement, score, rank)
    return ScoreEmbeddingExports(score, comp_strict, rank, token, token_weak, injective)
