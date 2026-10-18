"""Untrusted Boolean-record encoding and exact Nat merge preservation proofs.

This is an arbitrary-bound mathematical record model. It does not identify
Nat values with UInt32 or assert compiler/GPU source correspondence.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_join_bool as bo
import llm_mojo.kernels.finishing_join_terms as join
import llm_mojo.kernels.nat_score_link as ar
import llm_mojo.kernels.nat_score_complement as co
import llm_mojo.kernels.nat_score_embedding as em
import llm_mojo.kernels.nat_record_selector as sel


@fieldwise_init
struct EncodedRecordExports(ImplicitlyCopyable):
    var encode: Int
    var merge: Int
    var pack_select: Int
    var identity: Int
    var injective: Int
    var merge_preserves: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, args: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), args)


def encoded_record[C: Int](mut p: ph._Builder[C], exports: EncodedRecordExports,
    bound: Int, rank: Int, token: Int, bad: Int) -> Int:
    return _call(p, exports.encode, [bound, rank, token, bad])


def _encode[C: Int](mut p: ph._Builder[C], definition: Int,
    bound: Int, rank: Int, token: Int, bad: Int) -> Int:
    return _call(p, definition, [bound, rank, token, bad])


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var definition = p.bundle.definition_count
    ph._declare(p, type, body, definition)
    return definition


def _data[C: Int](mut p: ph._Builder[C], embedding: em.ScoreEmbeddingExports,
    selector: sel.RecordSelectorExports) -> List[Int]:
    var bound = ph._var(p, ph._nat(p))
    var rank = ph._var(p, ph._nat(p))
    var token = ph._var(p, ph._nat(p))
    var bad = ph._var(p, ph._nat(p))
    var encode = _declare(p, ph._pis(p, [bound, rank, token, bad], ph._nat(p)),
        ph._lams(p, [bound, rank, token, bad], bits.record_pack(p,
            em.score_value(p, embedding, bound, rank, token), bad)))
    bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ba = ph._var(p, ph._nat(p))
    var bb = ph._var(p, ph._nat(p))
    var flag = sel.record_b_wins(p, selector, ra, rb, ta, tb)
    var bad_union = bits.record_bool_or(p, ba, bb)
    var body = ph._select(p, flag,
        _encode(p, encode, bound, ra, ta, bad_union),
        _encode(p, encode, bound, rb, tb, bad_union))
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb]
    var merge = _declare(p, ph._pis(p, binders, ph._nat(p)), ph._lams(p, binders, body))
    return [encode, merge]


def _pack_select[C: Int](mut p: ph._Builder[C]) -> Int:
    var flag = ph._var(p, ph._nat(p))
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var bad = ph._var(p, ph._nat(p))
    var lhs = bits.record_pack(p, ph._select(p, flag, left, right), bad)
    var rhs = ph._select(p, flag, bits.record_pack(p, left, bad), bits.record_pack(p, right, bad))
    var motive = ph._lam(p, flag, ph._eq(p, lhs, rhs))
    var base = ph._refl(p, bits.record_pack(p, left, bad))
    var k = ph._var(p, ph._nat(p))
    var ignored = ph._var(p, ph._app(p, motive, k))
    var step = ph._lams(p, [k, ignored], ph._refl(p, bits.record_pack(p, right, bad)))
    return _declare(p, ph._pis(p, [left, right, bad, flag], ph._eq(p, lhs, rhs)),
        ph._lams(p, [left, right, bad, flag], ph._induct(p, motive, base, step, flag)))


def _identity[C: Int](mut p: ph._Builder[C], complement: co.ComplementExports,
    embedding: em.ScoreEmbeddingExports, encode: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var radix = ph._succ(p, bound)
    var product = ar.score_mul(p, ph._zero(p), radix)
    var comp = co.score_comp(p, complement, bound, bound)
    var score = em.score_value(p, embedding, bound, ph._zero(p), bound)
    var slot = ph._var(p, ph._nat(p))
    var product_zero = _call(p, ar.SCORE_MUL_ZERO_LEFT, [radix])
    var first = ph._cong(p, ph._lam(p, slot, ar.score_add(p, slot, comp)),
        product, ph._zero(p), product_zero)
    var middle = ar.score_add(p, ph._zero(p), comp)
    var second = _call(p, ar.SCORE_ADD_ZERO_LEFT, [comp])
    var score_comp = ph._trans(p, score, middle, comp, first, second)
    var score_zero = ph._trans(p, score, comp, ph._zero(p), score_comp,
        _call(p, complement.diagonal, [bound]))
    slot = ph._var(p, ph._nat(p))
    var body = ph._cong(p, ph._lam(p, slot, bits.record_pack(p, slot, ph._zero(p))),
        score, ph._zero(p), score_zero)
    return _declare(p, ph._pi(p, bound, ph._eq(p,
        _encode(p, encode, bound, ph._zero(p), bound, ph._zero(p)), ph._zero(p))),
        ph._lam(p, bound, body))


def _injective[C: Int](mut p: ph._Builder[C], embedding: em.ScoreEmbeddingExports,
    encode: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ba = ph._var(p, ph._nat(p))
    var bb = ph._var(p, ph._nat(p))
    var ta_bound = ph._var(p, ar.score_le(p, ta, bound))
    var tb_bound = ph._var(p, ar.score_le(p, tb, bound))
    var bit_a = ph._var(p, bits.record_is_bit(p, ba))
    var bit_b = ph._var(p, bits.record_is_bit(p, bb))
    var ea = _encode(p, encode, bound, ra, ta, ba)
    var eb = _encode(p, encode, bound, rb, tb, bb)
    var equality = ph._var(p, ph._eq(p, ea, eb))
    var sa = em.score_value(p, embedding, bound, ra, ta)
    var sb = em.score_value(p, embedding, bound, rb, tb)
    var scores = ph._cong(p, ph._named(p, bits.RECORD_HALF), ea, eb, equality)
    var half_a = _call(p, bits.RECORD_PACK_HALF, [sa, ba, bit_a])
    var half_b = _call(p, bits.RECORD_PACK_HALF, [sb, bb, bit_b])
    scores = ph._trans(p, sa, bits.record_half(p, ea), bits.record_half(p, eb),
        ph._sym(p, bits.record_half(p, ea), sa, half_a), scores)
    scores = ph._trans(p, sa, bits.record_half(p, eb), sb, scores, half_b)
    var rank_token = _call(p, embedding.score_injective, [bound, ra, rb, ta, tb,
        ta_bound, tb_bound, scores])
    var pair_props: List[Int] = [ph._eq(p, ra, rb), ph._eq(p, ta, tb)]
    var rank_eq = a._project(p, rank_token, pair_props, 0)
    var token_eq = a._project(p, rank_token, pair_props, 1)
    var flags = ph._cong(p, ph._named(p, bits.RECORD_PARITY), ea, eb, equality)
    var parity_a = _call(p, bits.RECORD_PACK_PARITY, [sa, ba, bit_a])
    var parity_b = _call(p, bits.RECORD_PACK_PARITY, [sb, bb, bit_b])
    flags = ph._trans(p, ba, bits.record_parity(p, ea), bits.record_parity(p, eb),
        ph._sym(p, bits.record_parity(p, ea), ba, parity_a), flags)
    flags = ph._trans(p, ba, bits.record_parity(p, eb), bb, flags, parity_b)
    var props: List[Int] = [pair_props[0], pair_props[1], ph._eq(p, ba, bb)]
    var result = a._and_type(p, props)
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb, ta_bound, tb_bound,
        bit_a, bit_b, equality]
    return _declare(p, ph._pis(p, binders, result),
        ph._lams(p, binders, a._and_intro(p, props, [rank_eq, token_eq, flags])))


def _preservation[C: Int](mut p: ph._Builder[C], embedding: em.ScoreEmbeddingExports,
    selector: sel.RecordSelectorExports, encode: Int, merge: Int, pack_select: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ba = ph._var(p, ph._nat(p))
    var bb = ph._var(p, ph._nat(p))
    var ta_bound = ph._var(p, ar.score_le(p, ta, bound))
    var tb_bound = ph._var(p, ar.score_le(p, tb, bound))
    var bit_a = ph._var(p, bits.record_is_bit(p, ba))
    var bit_b = ph._var(p, bits.record_is_bit(p, bb))
    var ea = _encode(p, encode, bound, ra, ta, ba)
    var eb = _encode(p, encode, bound, rb, tb, bb)
    var sa = em.score_value(p, embedding, bound, ra, ta)
    var sb = em.score_value(p, embedding, bound, rb, tb)
    var half_a = _call(p, bits.RECORD_PACK_HALF, [sa, ba, bit_a])
    var half_b = _call(p, bits.RECORD_PACK_HALF, [sb, bb, bit_b])
    var parity_a = _call(p, bits.RECORD_PACK_PARITY, [sa, ba, bit_a])
    var parity_b = _call(p, bits.RECORD_PACK_PARITY, [sb, bb, bit_b])
    var raw_score = _max(p, bits.record_half(p, ea), bits.record_half(p, eb))
    var max_score = _max(p, sa, sb)
    var raw_flag = bits.record_bool_or(p, bits.record_parity(p, ea), bits.record_parity(p, eb))
    var union = bits.record_bool_or(p, ba, bb)
    var slot = ph._var(p, ph._nat(p))
    var score_first = ph._cong(p, ph._lam(p, slot, _max(p, slot, bits.record_half(p, eb))),
        bits.record_half(p, ea), sa, half_a)
    slot = ph._var(p, ph._nat(p))
    var score_last = ph._cong(p, ph._lam(p, slot, _max(p, sa, slot)),
        bits.record_half(p, eb), sb, half_b)
    var score_eq = ph._trans(p, raw_score, _max(p, sa, bits.record_half(p, eb)),
        max_score, score_first, score_last)
    slot = ph._var(p, ph._nat(p))
    var flag_first = ph._cong(p, ph._lam(p, slot, bits.record_bool_or(p, slot, bits.record_parity(p, eb))),
        bits.record_parity(p, ea), ba, parity_a)
    slot = ph._var(p, ph._nat(p))
    var flag_last = ph._cong(p, ph._lam(p, slot, bits.record_bool_or(p, ba, slot)),
        bits.record_parity(p, eb), bb, parity_b)
    var flag_eq = ph._trans(p, raw_flag,
        bits.record_bool_or(p, ba, bits.record_parity(p, eb)), union, flag_first, flag_last)
    var actual = join.record_join(p, ea, eb)
    var middle = bits.record_pack(p, max_score, union)
    var first = join._pack_congruence(p, raw_score, max_score, raw_flag, union, score_eq, flag_eq)
    var winner = sel.record_b_wins(p, selector, ra, rb, ta, tb)
    var selected = ph._select(p, winner, sa, sb)
    var selection = _call(p, selector.max_selected, [bound, ra, rb, ta, tb, ta_bound, tb_bound])
    slot = ph._var(p, ph._nat(p))
    var second = ph._cong(p, ph._lam(p, slot, bits.record_pack(p, slot, union)),
        max_score, selected, selection)
    var selected_pack = bits.record_pack(p, selected, union)
    var first_two = ph._trans(p, actual, middle, selected_pack, first, second)
    var expected = _call(p, merge, [bound, ra, rb, ta, tb, ba, bb])
    var last = _call(p, pack_select, [sa, sb, union, winner])
    var body = ph._trans(p, actual, selected_pack, expected, first_two, last)
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb, ta_bound, tb_bound, bit_a, bit_b]
    return _declare(p, ph._pis(p, binders, ph._eq(p, actual, expected)), ph._lams(p, binders, body))


def _max[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return _call(p, 55, [left, right])


def append_nat_record_encoding[C: Int](mut p: ph._Builder[C],
    complement: co.ComplementExports, embedding: em.ScoreEmbeddingExports,
    selector: sel.RecordSelectorExports) -> EncodedRecordExports:
    var data = _data(p, embedding, selector)
    var pack_select = _pack_select(p)
    var identity = _identity(p, complement, embedding, data[0])
    var injective = _injective(p, embedding, data[0])
    var preservation = _preservation(p, embedding, selector, data[0], data[1], pack_select)
    return EncodedRecordExports(data[0], data[1], pack_select, identity, injective, preservation)
