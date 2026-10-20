"""Checked base-two decomposition of every Nat by a recursive clock.

Append after the maximum definitions 55..60. Ordinary Nat recursion, its
checked induction eliminator, Pi/lambda/application, equality and substitution
establish the score/Boolean-flag representation. No division, literal
arithmetic or native parity oracle is used. This is not a native UInt32 record
or GPU execution refinement.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a

comptime RECORD_DOUBLE = 61
comptime RECORD_PAIR = 62
comptime RECORD_CLOCK = 63
comptime RECORD_HALF = 64
comptime RECORD_PARITY = 65
comptime RECORD_IS_BIT = 66
comptime RECORD_PACK = 67
comptime RECORD_BOOL_OR = 68
comptime RECORD_CLOCK_INVARIANT = 69
comptime RECORD_PARITY_BIT = 70
comptime RECORD_REPACK = 71
comptime RECORD_EVEN_ODD = 72
comptime RECORD_EVEN_HALF = 73
comptime RECORD_EVEN_PARITY = 74
comptime RECORD_ODD_HALF = 75
comptime RECORD_ODD_PARITY = 76
comptime RECORD_PACK_HALF = 77
comptime RECORD_PACK_PARITY = 78
comptime RECORD_PACK_ROUNDTRIP = 79
comptime RECORD_PARITY_DEFINITIONS = 80


def record_double[C: Int](mut p: ph._Builder[C], n: Int) -> Int:
    return ph._app(p, ph._named(p, RECORD_DOUBLE), n)


def record_pair[C: Int](mut p: ph._Builder[C], quotient: Int, bit: Int) -> Int:
    return ph._apps(p, ph._named(p, RECORD_PAIR), [quotient, bit])


def record_clock[C: Int](mut p: ph._Builder[C], n: Int) -> Int:
    return ph._app(p, ph._named(p, RECORD_CLOCK), n)


def record_half[C: Int](mut p: ph._Builder[C], n: Int) -> Int:
    return ph._app(p, ph._named(p, RECORD_HALF), n)


def record_parity[C: Int](mut p: ph._Builder[C], n: Int) -> Int:
    return ph._app(p, ph._named(p, RECORD_PARITY), n)


def record_is_bit[C: Int](mut p: ph._Builder[C], bit: Int) -> Int:
    return ph._app(p, ph._named(p, RECORD_IS_BIT), bit)


def record_pack[C: Int](mut p: ph._Builder[C], quotient: Int, bit: Int) -> Int:
    return ph._apps(p, ph._named(p, RECORD_PACK), [quotient, bit])


def record_bool_or[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return ph._apps(p, ph._named(p, RECORD_BOOL_OR), [left, right])


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _rec[C: Int](mut p: ph._Builder[C], n: Int, base: Int, step: Int) -> Int:
    return ph._node(p, tk.PT_NAT_REC, n, base, step)


def _bit_intro[C: Int](mut p: ph._Builder[C], bit: Int,
    equality: Int, one: Bool) -> Int:
    var proposition = ph._var(p, ph._prop(p))
    var zero_equal = ph._var(p, ph._eq(p, bit, ph._zero(p)))
    var zero_case = ph._var(p, ph._pi(p, zero_equal, proposition))
    var one_equal = ph._var(p, ph._eq(p, bit, ph._one(p)))
    var one_case = ph._var(p, ph._pi(p, one_equal, proposition))
    return ph._lams(p, [proposition, zero_case, one_case],
        ph._app(p, one_case if one else zero_case, equality))


def _bit_transport[C: Int](mut p: ph._Builder[C], bit: Int, evidence: Int,
    motive: Int, at_zero: Int, at_one: Int) -> Int:
    """ChurchOr elimination, transporting each constructor case by Eq."""
    var zero_equal = ph._var(p, ph._eq(p, bit, ph._zero(p)))
    var one_equal = ph._var(p, ph._eq(p, bit, ph._one(p)))
    var zero_case = ph._lam(p, zero_equal, ph._subst(p, motive,
        ph._sym(p, bit, ph._zero(p), zero_equal), at_zero))
    var one_case = ph._lam(p, one_equal, ph._subst(p, motive,
        ph._sym(p, bit, ph._one(p), one_equal), at_one))
    return ph._apps(p, evidence, [ph._app(p, motive, bit), zero_case, one_case])


def _data[C: Int](mut p: ph._Builder[C]):
    var n = ph._var(p, ph._nat(p))
    var predecessor = ph._var(p, ph._nat(p))
    var previous = ph._var(p, ph._nat(p))
    var step = ph._lams(p, [predecessor, previous], ph._succ(p, ph._succ(p, previous)))
    ph._declare(p, ph._pi(p, n, ph._nat(p)),
        ph._lam(p, n, _rec(p, n, ph._zero(p), step)), RECORD_DOUBLE)

    var quotient = ph._var(p, ph._nat(p))
    var bit = ph._var(p, ph._nat(p))
    var index = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pis(p, [quotient, bit], ph._store(p)),
        ph._lams(p, [quotient, bit, index], ph._select(p, index, quotient, bit)), RECORD_PAIR)

    n = ph._var(p, ph._nat(p))
    predecessor = ph._var(p, ph._nat(p))
    var clock_previous = ph._var(p, ph._store(p))
    var old_quotient = ph._app(p, clock_previous, ph._zero(p))
    var old_bit = ph._app(p, clock_previous, ph._one(p))
    var next_quotient = ph._select(p, old_bit, old_quotient, ph._succ(p, old_quotient))
    var next_bit = ph._select(p, old_bit, ph._one(p), ph._zero(p))
    step = ph._lams(p, [predecessor, clock_previous], record_pair(p, next_quotient, next_bit))
    ph._declare(p, ph._pi(p, n, ph._store(p)),
        ph._lam(p, n, _rec(p, n, record_pair(p, ph._zero(p), ph._zero(p)), step)), RECORD_CLOCK)

    n = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, n, ph._nat(p)),
        ph._lam(p, n, ph._app(p, record_clock(p, n), ph._zero(p))), RECORD_HALF)
    n = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, n, ph._nat(p)),
        ph._lam(p, n, ph._app(p, record_clock(p, n), ph._one(p))), RECORD_PARITY)

    bit = ph._var(p, ph._nat(p))
    var proposition = ph._var(p, ph._prop(p))
    var zero_equal = ph._var(p, ph._eq(p, bit, ph._zero(p)))
    var zero_case = ph._var(p, ph._pi(p, zero_equal, proposition))
    var one_equal = ph._var(p, ph._eq(p, bit, ph._one(p)))
    var one_case = ph._var(p, ph._pi(p, one_equal, proposition))
    ph._declare(p, ph._pi(p, bit, ph._prop(p)),
        ph._lam(p, bit, ph._pis(p, [proposition, zero_case, one_case], proposition)), RECORD_IS_BIT)

    quotient = ph._var(p, ph._nat(p))
    bit = ph._var(p, ph._nat(p))
    var doubled = record_double(p, quotient)
    ph._declare(p, ph._pis(p, [quotient, bit], ph._nat(p)),
        ph._lams(p, [quotient, bit], ph._select(p, bit, doubled, ph._succ(p, doubled))), RECORD_PACK)
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pis(p, [left, right], ph._nat(p)),
        ph._lams(p, [left, right], ph._select(p, left, right, ph._one(p))), RECORD_BOOL_OR)


def _next_bit_proof[C: Int](mut p: ph._Builder[C], bit: Int, evidence: Int) -> Int:
    var selected = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, selected,
        record_is_bit(p, ph._select(p, selected, ph._one(p), ph._zero(p))))
    return _bit_transport(p, bit, evidence, motive,
        _bit_intro(p, ph._one(p), ph._refl(p, ph._one(p)), True),
        _bit_intro(p, ph._zero(p), ph._refl(p, ph._zero(p)), False))


def _next_pack_proof[C: Int](mut p: ph._Builder[C], quotient: Int,
    bit: Int, evidence: Int) -> Int:
    var selected = ph._var(p, ph._nat(p))
    var next_quotient = ph._select(p, selected, quotient, ph._succ(p, quotient))
    var next_bit = ph._select(p, selected, ph._one(p), ph._zero(p))
    var motive = ph._lam(p, selected, ph._eq(p,
        record_pack(p, next_quotient, next_bit), ph._succ(p, record_pack(p, quotient, selected))))
    var doubled = record_double(p, quotient)
    return _bit_transport(p, bit, evidence, motive,
        ph._refl(p, ph._succ(p, doubled)), ph._refl(p, ph._succ(p, ph._succ(p, doubled))))


def _clock_propositions[C: Int](mut p: ph._Builder[C], n: Int) -> List[Int]:
    return [record_is_bit(p, record_parity(p, n)),
        ph._eq(p, record_pack(p, record_half(p, n), record_parity(p, n)), n)]


def _clock_invariant[C: Int](mut p: ph._Builder[C]):
    var n = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, n, a._and_type(p, _clock_propositions(p, n)))
    var zero = ph._zero(p)
    var base = a._and_intro(p, _clock_propositions(p, zero),
        [_bit_intro(p, record_parity(p, zero), ph._refl(p, zero), False), ph._refl(p, zero)])
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var old = _clock_propositions(p, k)
    var bit_evidence = a._project(p, ih, old, 0)
    var repack_evidence = a._project(p, ih, old, 1)
    var half = record_half(p, k)
    var parity = record_parity(p, k)
    var next = ph._succ(p, k)
    var first = _next_pack_proof(p, half, parity, bit_evidence)
    var second = ph._succ_cong(p,
        record_pack(p, half, parity), k, repack_evidence)
    var packed_next = record_pack(p, record_half(p, next), record_parity(p, next))
    var repack_next = ph._trans(p, packed_next, ph._succ(p, record_pack(p, half, parity)),
        next, first, second)
    var step = ph._lams(p, [k, ih], a._and_intro(p, _clock_propositions(p, next),
        [_next_bit_proof(p, parity, bit_evidence), repack_next]))
    ph._declare(p, ph._pi(p, n, a._and_type(p, _clock_propositions(p, n))),
        ph._lam(p, n, ph._induct(p, motive, base, step, n)), RECORD_CLOCK_INVARIANT)
    n = ph._var(p, ph._nat(p))
    var invariant = _call(p, RECORD_CLOCK_INVARIANT, [n])
    ph._declare(p, ph._pi(p, n, record_is_bit(p, record_parity(p, n))),
        ph._lam(p, n, a._project(p, invariant, _clock_propositions(p, n), 0)), RECORD_PARITY_BIT)
    n = ph._var(p, ph._nat(p))
    invariant = _call(p, RECORD_CLOCK_INVARIANT, [n])
    ph._declare(p, ph._pi(p, n,
        ph._eq(p, record_pack(p, record_half(p, n), record_parity(p, n)), n)),
        ph._lam(p, n, a._project(p, invariant, _clock_propositions(p, n), 1)), RECORD_REPACK)


def _even_odd_propositions[C: Int](mut p: ph._Builder[C], quotient: Int) -> List[Int]:
    var even = record_double(p, quotient)
    var odd = ph._succ(p, even)
    return [ph._eq(p, record_half(p, even), quotient),
        ph._eq(p, record_parity(p, even), ph._zero(p)),
        ph._eq(p, record_half(p, odd), quotient),
        ph._eq(p, record_parity(p, odd), ph._one(p))]


def _select_congruence[C: Int](mut p: ph._Builder[C], old: Int, new: Int,
    equality: Int, yes: Int, no: Int) -> Int:
    var flag = ph._var(p, ph._nat(p))
    return ph._cong(p, ph._lam(p, flag, ph._select(p, flag, yes, no)), old, new, equality)


def _even_odd[C: Int](mut p: ph._Builder[C]):
    var quotient = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, quotient, a._and_type(p, _even_odd_propositions(p, quotient)))
    var zero = ph._zero(p)
    var base = a._and_intro(p, _even_odd_propositions(p, zero),
        [ph._refl(p, zero), ph._refl(p, zero), ph._refl(p, zero), ph._refl(p, ph._one(p))])
    var k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var propositions = _even_odd_propositions(p, k)
    var odd_half = a._project(p, ih, propositions, 2)
    var odd_parity = a._project(p, ih, propositions, 3)
    var old_odd = ph._succ(p, record_double(p, k))
    var old_half = record_half(p, old_odd)
    var old_parity = record_parity(p, old_odd)
    var next = ph._succ(p, k)
    var even_next = record_double(p, next)
    var half_even_next = record_half(p, even_next)
    var parity_even_next = record_parity(p, even_next)
    var first = _select_congruence(p, old_parity, ph._one(p), odd_parity,
        old_half, ph._succ(p, old_half))
    var second = ph._succ_cong(p, old_half, k, odd_half)
    var even_half = ph._trans(p, half_even_next, ph._succ(p, old_half), next, first, second)
    var even_parity = _select_congruence(p, old_parity, ph._one(p), odd_parity, ph._one(p), zero)
    first = _select_congruence(p, parity_even_next, zero, even_parity,
        half_even_next, ph._succ(p, half_even_next))
    var new_odd_half = ph._trans(p, record_half(p, ph._succ(p, even_next)),
        half_even_next, next, first, even_half)
    var new_odd_parity = _select_congruence(p, parity_even_next, zero,
        even_parity, ph._one(p), zero)
    var step = ph._lams(p, [k, ih], a._and_intro(p, _even_odd_propositions(p, next),
        [even_half, even_parity, new_odd_half, new_odd_parity]))
    ph._declare(p, ph._pi(p, quotient, a._and_type(p, _even_odd_propositions(p, quotient))),
        ph._lam(p, quotient, ph._induct(p, motive, base, step, quotient)), RECORD_EVEN_ODD)
    for projection in range(4):
        quotient = ph._var(p, ph._nat(p))
        propositions = _even_odd_propositions(p, quotient)
        var invariant = _call(p, RECORD_EVEN_ODD, [quotient])
        ph._declare(p, ph._pi(p, quotient, propositions[projection]),
            ph._lam(p, quotient, a._project(p, invariant, propositions, projection)), RECORD_EVEN_HALF + projection)


def _pack_projections[C: Int](mut p: ph._Builder[C]):
    var quotient = ph._var(p, ph._nat(p))
    var bit = ph._var(p, ph._nat(p))
    var evidence = ph._var(p, record_is_bit(p, bit))
    var selected = ph._var(p, ph._nat(p))
    var half_motive = ph._lam(p, selected,
        ph._eq(p, record_half(p, record_pack(p, quotient, selected)), quotient))
    var half_body = _bit_transport(p, bit, evidence, half_motive,
        _call(p, RECORD_EVEN_HALF, [quotient]), _call(p, RECORD_ODD_HALF, [quotient]))
    ph._declare(p, ph._pis(p, [quotient, bit, evidence],
        ph._eq(p, record_half(p, record_pack(p, quotient, bit)), quotient)),
        ph._lams(p, [quotient, bit, evidence], half_body), RECORD_PACK_HALF)

    quotient = ph._var(p, ph._nat(p))
    bit = ph._var(p, ph._nat(p))
    evidence = ph._var(p, record_is_bit(p, bit))
    selected = ph._var(p, ph._nat(p))
    var parity_motive = ph._lam(p, selected,
        ph._eq(p, record_parity(p, record_pack(p, quotient, selected)), selected))
    var parity_body = _bit_transport(p, bit, evidence, parity_motive,
        _call(p, RECORD_EVEN_PARITY, [quotient]), _call(p, RECORD_ODD_PARITY, [quotient]))
    ph._declare(p, ph._pis(p, [quotient, bit, evidence],
        ph._eq(p, record_parity(p, record_pack(p, quotient, bit)), bit)),
        ph._lams(p, [quotient, bit, evidence], parity_body), RECORD_PACK_PARITY)

    quotient = ph._var(p, ph._nat(p))
    bit = ph._var(p, ph._nat(p))
    evidence = ph._var(p, record_is_bit(p, bit))
    var propositions: List[Int] = [
        ph._eq(p, record_half(p, record_pack(p, quotient, bit)), quotient),
        ph._eq(p, record_parity(p, record_pack(p, quotient, bit)), bit),
    ]
    ph._declare(p, ph._pis(p, [quotient, bit, evidence], a._and_type(p, propositions)),
        ph._lams(p, [quotient, bit, evidence], a._and_intro(p, propositions,
            [_call(p, RECORD_PACK_HALF, [quotient, bit, evidence]),
             _call(p, RECORD_PACK_PARITY, [quotient, bit, evidence])])), RECORD_PACK_ROUNDTRIP)


def append_record_parity_terms[C: Int](mut p: ph._Builder[C]):
    """Append raw definitions/proofs in place, without nested large builders."""
    if p.bundle.definition_count != RECORD_DOUBLE:
        p.bundle.count = -1
        return
    _data(p)
    _clock_invariant(p)
    _even_odd(p)
    _pack_projections(p)
