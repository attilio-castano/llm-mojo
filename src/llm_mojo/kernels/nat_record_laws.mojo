"""Componentwise laws for the mathematical Nat record merge.

Twelve ordinary declarations derive bounded-token and Boolean closure,
encoding preservation, associativity, commutativity, idempotence and both units.
Every rank is arbitrary Nat and token bound M remains symbolic. Intermediate
validity is derived before applying encoding injectivity. Producer export IDs
grant no judgment; native words/source/GPU refinement remains separate.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_join_bool as boolean
import llm_mojo.kernels.finishing_join_terms as join
import llm_mojo.kernels.nat_score_link as arithmetic
import llm_mojo.kernels.nat_record_selector as selector
import llm_mojo.kernels.nat_record_encoding as encoding

comptime RECORD_LAWS_DEFINITIONS = 12


@fieldwise_init
struct RecordLawsExports(ImplicitlyCopyable):
    # rank/token: (ra,rb,ta,tb) -> Nat.
    var rank: Int
    var token: Int
    # token_bounded: (M,ra,rb,ta,tb,ta<=M,tb<=M) -> Tmu<=M.
    var token_bounded: Int
    # bit_closed: (ba,bb,IsBit ba,IsBit bb) -> IsBit(OR ba bb).
    var bit_closed: Int
    # component_encoding: (M,ra,rb,ta,tb,ba,bb) ->
    #   Eq(Enc(M,Rmu,Tmu,OR ba bb), merge143(M,ra,rb,ta,tb,ba,bb)).
    var component_encoding: Int
    # homomorphism: same seven values, then (ta<=M,tb<=M,IsBit ba,IsBit bb)
    #   -> Eq(Enc mu(A,B), Join(Enc A,Enc B)).
    var homomorphism: Int
    # le_refl: (n) -> Le(n,n), by ordinary Nat induction.
    var le_refl: Int
    # Law binder order: M; input triples grouped (ra,ta,ba),(rb,tb,bb),...
    # then proof pairs (ta<=M,IsBit ba),(tb<=M,IsBit bb),... .
    # Each conclusion is ChurchAnd(rankEq,tokenEq,badEq), in that order.
    # commutative has two input triples; associative has three; the last
    # three laws have one. E=(0,M,0), with its validity derived internally.
    var commutative: Int
    var associative: Int
    var idempotent: Int
    var left_unit: Int
    var right_unit: Int


@fieldwise_init
struct _ValidRecordTerms(ImplicitlyCopyable):
    # Untrusted producer metadata: only ordinary syntax nodes are submitted.
    var rank: Int
    var token: Int
    var bad: Int
    var token_bound: Int
    var bad_bit: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var definition = p.bundle.definition_count
    ph._declare(p, type, body, definition)
    return definition


def record_merge_rank[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    ra: Int, rb: Int, ta: Int, tb: Int) -> Int:
    return _call(p, exports.rank, [ra, rb, ta, tb])


def record_merge_token[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    ra: Int, rb: Int, ta: Int, tb: Int) -> Int:
    return _call(p, exports.token, [ra, rb, ta, tb])


def _data[C: Int](mut p: ph._Builder[C], choice: selector.RecordSelectorExports,
    rank: Bool) -> Int:
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var flag = selector.record_b_wins(p, choice, ra, rb, ta, tb)
    var body = ph._select(p, flag, ra if rank else ta, rb if rank else tb)
    var binders: List[Int] = [ra, rb, ta, tb]
    return _declare(p, ph._pis(p, binders, ph._nat(p)), ph._lams(p, binders, body))


def _token_bounded[C: Int](mut p: ph._Builder[C], token: Int,
    choice: selector.RecordSelectorExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ha = ph._var(p, arithmetic.score_le(p, ta, bound))
    var hb = ph._var(p, arithmetic.score_le(p, tb, bound))
    var flag = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, flag,
        arithmetic.score_le(p, ph._select(p, flag, ta, tb), bound))
    var predecessor = ph._var(p, ph._nat(p))
    var ignored = ph._var(p, ph._app(p, motive, predecessor))
    var step = ph._lams(p, [predecessor, ignored], hb)
    var actual = selector.record_b_wins(p, choice, ra, rb, ta, tb)
    var body = ph._induct(p, motive, ha, step, actual)
    var binders: List[Int] = [bound, ra, rb, ta, tb, ha, hb]
    return _declare(p, ph._pis(p, binders, arithmetic.score_le(p,
        _call(p, token, [ra, rb, ta, tb]), bound)), ph._lams(p, binders, body))


def _bit_closed[C: Int](mut p: ph._Builder[C]) -> Int:
    var ba = ph._var(p, ph._nat(p))
    var bb = ph._var(p, ph._nat(p))
    var ha = ph._var(p, bits.record_is_bit(p, ba))
    var hb = ph._var(p, bits.record_is_bit(p, bb))
    var body = boolean._cases(p, 0, [ba, bb], [ha, hb], 0)
    var binders: List[Int] = [ba, bb, ha, hb]
    return _declare(p, ph._pis(p, binders,
        bits.record_is_bit(p, bits.record_bool_or(p, ba, bb))), ph._lams(p, binders, body))


def _component_encoding[C: Int](mut p: ph._Builder[C],
    enc: encoding.EncodedRecordExports, choice: selector.RecordSelectorExports,
    rank: Int, token: Int) -> Int:
    # A single flag controls both rank and token. Constructor induction checks
    # that correlated component selection equals the actual merge143 branch.
    # This equation needs no token/bit premises and decodes no arbitrary Nat.
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ba = ph._var(p, ph._nat(p))
    var bb = ph._var(p, ph._nat(p))
    var union = bits.record_bool_or(p, ba, bb)
    var ea = encoding.encoded_record(p, enc, bound, ra, ta, union)
    var eb = encoding.encoded_record(p, enc, bound, rb, tb, union)
    var flag = ph._var(p, ph._nat(p))
    var generic_left = encoding.encoded_record(p, enc, bound,
        ph._select(p, flag, ra, rb), ph._select(p, flag, ta, tb), union)
    var generic_right = ph._select(p, flag, ea, eb)
    var motive = ph._lam(p, flag, ph._eq(p, generic_left, generic_right))
    var predecessor = ph._var(p, ph._nat(p))
    var ignored = ph._var(p, ph._app(p, motive, predecessor))
    var step = ph._lams(p, [predecessor, ignored], ph._refl(p, eb))
    var actual = selector.record_b_wins(p, choice, ra, rb, ta, tb)
    var body = ph._induct(p, motive, ph._refl(p, ea), step, actual)
    var left = encoding.encoded_record(p, enc, bound,
        _call(p, rank, [ra, rb, ta, tb]), _call(p, token, [ra, rb, ta, tb]), union)
    var right = _call(p, enc.merge, [bound, ra, rb, ta, tb, ba, bb])
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb]
    return _declare(p, ph._pis(p, binders, ph._eq(p, left, right)),
        ph._lams(p, binders, body))


def _homomorphism[C: Int](mut p: ph._Builder[C], enc: encoding.EncodedRecordExports,
    rank: Int, token: Int, component_encoding: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ra = ph._var(p, ph._nat(p))
    var rb = ph._var(p, ph._nat(p))
    var ta = ph._var(p, ph._nat(p))
    var tb = ph._var(p, ph._nat(p))
    var ba = ph._var(p, ph._nat(p))
    var bb = ph._var(p, ph._nat(p))
    var ha = ph._var(p, arithmetic.score_le(p, ta, bound))
    var hb = ph._var(p, arithmetic.score_le(p, tb, bound))
    var ia = ph._var(p, bits.record_is_bit(p, ba))
    var ib = ph._var(p, bits.record_is_bit(p, bb))
    var encoded = encoding.encoded_record(p, enc, bound,
        _call(p, rank, [ra, rb, ta, tb]), _call(p, token, [ra, rb, ta, tb]),
        bits.record_bool_or(p, ba, bb))
    var merge = _call(p, enc.merge, [bound, ra, rb, ta, tb, ba, bb])
    var joined = join.record_join(p, encoding.encoded_record(p, enc, bound, ra, ta, ba),
        encoding.encoded_record(p, enc, bound, rb, tb, bb))
    var components = _call(p, component_encoding, [bound, ra, rb, ta, tb, ba, bb])
    var preserved = _call(p, enc.merge_preserves,
        [bound, ra, rb, ta, tb, ba, bb, ha, hb, ia, ib])
    var body = ph._trans(p, encoded, merge, joined,
        components, ph._sym(p, joined, merge, preserved))
    var binders: List[Int] = [bound, ra, rb, ta, tb, ba, bb, ha, hb, ia, ib]
    return _declare(p, ph._pis(p, binders, ph._eq(p, encoded, joined)),
        ph._lams(p, binders, body))


def _le_refl[C: Int](mut p: ph._Builder[C]) -> Int:
    var n = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, n, arithmetic.score_le(p, n, n))
    # Le(0,0) is Church True, rather than Eq(0,0).
    var proposition = ph._var(p, ph._prop(p))
    var evidence = ph._var(p, proposition)
    var base = ph._lams(p, [proposition, evidence], evidence)
    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    var step = ph._lams(p, [predecessor, ih], ih)
    n = ph._var(p, ph._nat(p))
    return _declare(p, ph._pi(p, n, ph._app(p, motive, n)),
        ph._lam(p, n, ph._induct(p, motive, base, step, n)))


def _input_records[C: Int](mut p: ph._Builder[C], bound: Int,
    count: Int) -> List[_ValidRecordTerms]:
    var result = List[_ValidRecordTerms](capacity=count)
    for _ in range(count):
        var rank = ph._var(p, ph._nat(p))
        var token = ph._var(p, ph._nat(p))
        var bad = ph._var(p, ph._nat(p))
        var bounded = ph._var(p, arithmetic.score_le(p, token, bound))
        var bit = ph._var(p, bits.record_is_bit(p, bad))
        result.append(_ValidRecordTerms(rank, token, bad, bounded, bit))
    return result^


def _merge[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    bound: Int, left: _ValidRecordTerms, right: _ValidRecordTerms) -> _ValidRecordTerms:
    var rank = record_merge_rank(p, exports, left.rank, right.rank, left.token, right.token)
    var token = record_merge_token(p, exports, left.rank, right.rank, left.token, right.token)
    var bad = bits.record_bool_or(p, left.bad, right.bad)
    var bounded = _call(p, exports.token_bounded,
        [bound, left.rank, right.rank, left.token, right.token, left.token_bound, right.token_bound])
    var bit = _call(p, exports.bit_closed, [left.bad, right.bad, left.bad_bit, right.bad_bit])
    return _ValidRecordTerms(rank, token, bad, bounded, bit)


def _encode[C: Int](mut p: ph._Builder[C], enc: encoding.EncodedRecordExports,
    bound: Int, record: _ValidRecordTerms) -> Int:
    return encoding.encoded_record(p, enc, bound, record.rank, record.token, record.bad)


def _homo[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    bound: Int, left: _ValidRecordTerms, right: _ValidRecordTerms) -> Int:
    return _call(p, exports.homomorphism,
        [bound, left.rank, right.rank, left.token, right.token, left.bad, right.bad,
            left.token_bound, right.token_bound, left.bad_bit, right.bad_bit])


def _inject[C: Int](mut p: ph._Builder[C], enc: encoding.EncodedRecordExports,
    bound: Int, left: _ValidRecordTerms, right: _ValidRecordTerms, equal: Int) -> Int:
    return _call(p, enc.injective,
        [bound, left.rank, right.rank, left.token, right.token, left.bad, right.bad,
            left.token_bound, right.token_bound, left.bad_bit, right.bad_bit, equal])


def _record_equal[C: Int](mut p: ph._Builder[C],
    left: _ValidRecordTerms, right: _ValidRecordTerms) -> Int:
    return a._and_type(p, [ph._eq(p, left.rank, right.rank),
        ph._eq(p, left.token, right.token), ph._eq(p, left.bad, right.bad)])


def _declare_law[C: Int](mut p: ph._Builder[C], bound: Int,
    inputs: List[_ValidRecordTerms], left: _ValidRecordTerms,
    right: _ValidRecordTerms, body: Int) -> Int:
    var binders: List[Int] = [bound]
    for record in inputs:
        binders.append(record.rank)
        binders.append(record.token)
        binders.append(record.bad)
    for record in inputs:
        binders.append(record.token_bound)
        binders.append(record.bad_bit)
    return _declare(p, ph._pis(p, binders, _record_equal(p, left, right)),
        ph._lams(p, binders, body))


def _commutative[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    enc: encoding.EncodedRecordExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var inputs = _input_records(p, bound, 2)
    var left = _merge(p, exports, bound, inputs[0], inputs[1])
    var right = _merge(p, exports, bound, inputs[1], inputs[0])
    var el = _encode(p, enc, bound, left)
    var er = _encode(p, enc, bound, right)
    var ea = _encode(p, enc, bound, inputs[0])
    var eb = _encode(p, enc, bound, inputs[1])
    var ab = join.record_join(p, ea, eb)
    var ba = join.record_join(p, eb, ea)
    var first = _homo(p, exports, bound, inputs[0], inputs[1])
    var commute = _call(p, join.RECORD_JOIN_COMMUTATIVE, [ea, eb])
    var last = _homo(p, exports, bound, inputs[1], inputs[0])
    var equal = ph._trans(p, el, ab, ba, first, commute)
    equal = ph._trans(p, el, ba, er, equal, ph._sym(p, er, ba, last))
    return _declare_law(p, bound, inputs, left, right, _inject(p, enc, bound, left, right, equal))


def _associative[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    enc: encoding.EncodedRecordExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var inputs = _input_records(p, bound, 3)
    # Both inner merges receive derived closure, which is then consumed by
    # the two outer homomorphism proofs and both final injectivity premises.
    var ab_record = _merge(p, exports, bound, inputs[0], inputs[1])
    var bc_record = _merge(p, exports, bound, inputs[1], inputs[2])
    var left = _merge(p, exports, bound, ab_record, inputs[2])
    var right = _merge(p, exports, bound, inputs[0], bc_record)
    var el = _encode(p, enc, bound, left)
    var er = _encode(p, enc, bound, right)
    var ea = _encode(p, enc, bound, inputs[0])
    var eb = _encode(p, enc, bound, inputs[1])
    var ec = _encode(p, enc, bound, inputs[2])
    var eab = _encode(p, enc, bound, ab_record)
    var ebc = _encode(p, enc, bound, bc_record)
    var join_ab = join.record_join(p, ea, eb)
    var join_bc = join.record_join(p, eb, ec)
    var outer_left = join.record_join(p, eab, ec)
    var flat_left = join.record_join(p, join_ab, ec)
    var flat_right = join.record_join(p, ea, join_bc)
    var outer_right = join.record_join(p, ea, ebc)
    var outer_left_eq = _homo(p, exports, bound, ab_record, inputs[2])
    var inner_left_eq = _homo(p, exports, bound, inputs[0], inputs[1])
    var inner_right_eq = _homo(p, exports, bound, inputs[1], inputs[2])
    var outer_right_eq = _homo(p, exports, bound, inputs[0], bc_record)
    var slot = ph._var(p, ph._nat(p))
    var expand_left = ph._cong(p, ph._lam(p, slot, join.record_join(p, slot, ec)),
        eab, join_ab, inner_left_eq)
    slot = ph._var(p, ph._nat(p))
    var expand_right = ph._cong(p, ph._lam(p, slot, join.record_join(p, ea, slot)),
        ebc, join_bc, inner_right_eq)
    var reassociate = _call(p, join.RECORD_JOIN_ASSOCIATIVE, [ea, eb, ec])
    var equal = ph._trans(p, el, outer_left, flat_left, outer_left_eq, expand_left)
    equal = ph._trans(p, el, flat_left, flat_right, equal, reassociate)
    equal = ph._trans(p, el, flat_right, outer_right, equal,
        ph._sym(p, outer_right, flat_right, expand_right))
    equal = ph._trans(p, el, outer_right, er, equal,
        ph._sym(p, er, outer_right, outer_right_eq))
    return _declare_law(p, bound, inputs, left, right, _inject(p, enc, bound, left, right, equal))


def _idempotent[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    enc: encoding.EncodedRecordExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var inputs = _input_records(p, bound, 1)
    var output = _merge(p, exports, bound, inputs[0], inputs[0])
    var eo = _encode(p, enc, bound, output)
    var ea = _encode(p, enc, bound, inputs[0])
    var joined = join.record_join(p, ea, ea)
    var homomorphism = _homo(p, exports, bound, inputs[0], inputs[0])
    var collapse = _call(p, join.RECORD_JOIN_IDEMPOTENT, [ea])
    var equal = ph._trans(p, eo, joined, ea, homomorphism, collapse)
    return _declare_law(p, bound, inputs, output, inputs[0],
        _inject(p, enc, bound, output, inputs[0], equal))


def _unit[C: Int](mut p: ph._Builder[C], exports: RecordLawsExports,
    enc: encoding.EncodedRecordExports, left: Bool) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var inputs = _input_records(p, bound, 1)
    var empty = _ValidRecordTerms(ph._zero(p), bound, ph._zero(p),
        _call(p, exports.le_refl, [bound]), boolean._intro(p, False))
    var first = empty if left else inputs[0]
    var second = inputs[0] if left else empty
    var output = _merge(p, exports, bound, first, second)
    var eo = _encode(p, enc, bound, output)
    var ea = _encode(p, enc, bound, inputs[0])
    var ee = _encode(p, enc, bound, empty)
    var joined = join.record_join(p, ee, ea) if left else join.record_join(p, ea, ee)
    var zero_join = (join.record_join(p, ph._zero(p), ea) if left
        else join.record_join(p, ea, ph._zero(p)))
    var homomorphism = _homo(p, exports, bound, first, second)
    var empty_eq = _call(p, enc.identity, [bound])
    var slot = ph._var(p, ph._nat(p))
    var context = (ph._lam(p, slot, join.record_join(p, slot, ea)) if left
        else ph._lam(p, slot, join.record_join(p, ea, slot)))
    var replace_empty = ph._cong(p, context, ee, ph._zero(p), empty_eq)
    var unit_eq = _call(p,
        join.RECORD_JOIN_LEFT_UNIT if left else join.RECORD_JOIN_RIGHT_UNIT, [ea])
    var equal = ph._trans(p, eo, joined, zero_join, homomorphism, replace_empty)
    equal = ph._trans(p, eo, zero_join, ea, equal, unit_eq)
    return _declare_law(p, bound, inputs, output, inputs[0],
        _inject(p, enc, bound, output, inputs[0], equal))


def append_nat_record_laws[C: Int](mut p: ph._Builder[C],
    enc: encoding.EncodedRecordExports, choice: selector.RecordSelectorExports) -> RecordLawsExports:
    """Append ordinary candidate laws; returned IDs grant no admission."""
    var first = p.bundle.definition_count
    var rank = _data(p, choice, True)
    var token = _data(p, choice, False)
    var bounded = _token_bounded(p, token, choice)
    var bit = _bit_closed(p)
    var component = _component_encoding(p, enc, choice, rank, token)
    var homo = _homomorphism(p, enc, rank, token, component)
    var refl = _le_refl(p)
    var exports = RecordLawsExports(rank, token, bounded, bit, component, homo, refl,
        -1, -1, -1, -1, -1)
    exports.commutative = _commutative(p, exports, enc)
    exports.associative = _associative(p, exports, enc)
    exports.idempotent = _idempotent(p, exports, enc)
    exports.left_unit = _unit(p, exports, enc, True)
    exports.right_unit = _unit(p, exports, enc, False)
    if p.bundle.count < 0 or p.bundle.definition_count != first + RECORD_LAWS_DEFINITIONS:
        p.bundle.count = -1
        return RecordLawsExports(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
    return exports
