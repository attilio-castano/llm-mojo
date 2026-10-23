"""Untrusted Nat arithmetic definitions and ordinary dependent proof terms.

Every theorem below is a lambda/application/induction/substitution term. There
is no arithmetic oracle in this producer. The generic kernel independently
checks the acyclic definition table and the required exported signatures.
"""
import llm_mojo.kernels.proof_term_syntax as tk
from llm_mojo.kernels.nat_arithmetic_terms import (
    _ArithmeticBuilder, _declare_definition, _empty_builder,
    NatArithmeticCore, NatArithmeticLibrary,
    _node, _nat, _prop, _var, _number, _succ, _named, _app, _app2, _app3,
    _pi, _lam, _plus, _mul, _base, _eq, _le, _lt, _refl, _subst, _sym, _trans,
    _cong, _cong_left, _cong_right, _nat_induction, _rec,
    _truth, _unit, _false, _absurd,
    NAT_REP_ADD, NAT_REP_MUL, NAT_REP_LE, NAT_REP_LT, NAT_REP_SCALE,
    NAT_REP_BASE, NAT_REP_DIV, NAT_REP_MOD, NAT_REPRESENTATIONS,
)
from llm_mojo.kernels.nat_arithmetic_division import (
    declare_nat_arithmetic_division, make_nat_arithmetic_division,
)
from std.collections import Array

comptime NAT_ARITHMETIC_CAPACITY = 4096


def _p2[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int, result: Int) -> Int:
    return _pi(p, a, _pi(p, b, result))


def _p3[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int, c: Int, result: Int) -> Int:
    return _pi(p, a, _p2(p, b, c, result))


def _l2[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int, body: Int) -> Int:
    return _lam(p, a, _lam(p, b, body))


def _l3[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int, c: Int, body: Int) -> Int:
    return _lam(p, a, _l2(p, b, c, body))


def _d[C: Int](mut p: _ArithmeticBuilder[C], type: Int, body: Int) -> Int:
    return _declare_definition(p, type, body)


def declare_nat_arithmetic_definitions[C: Int](mut p: _ArithmeticBuilder[C]) -> Bool:
    """Raw canonical Add/Mul/Le bodies; no arithmetic shorthand is used."""
    if p.bundle.definition_count != 0:
        return False
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var ih = _var(p, _nat(p))
    var add_body = _l2(p, a, b, _rec(p, b, a, _l2(p, n, ih, _succ(p, ih))))
    if _d(p, _p2(p, a, b, _nat(p)), add_body) != 0:
        return False
    var k = _var(p, _nat(p))
    var x = _var(p, _nat(p))
    var j = _var(p, _nat(p))
    var acc = _var(p, _nat(p))
    var mul_body = _l2(p, k, x, _rec(p, x, _number(p, 0),
        _l2(p, j, acc, _plus(p, k, acc))))
    if _d(p, _p2(p, k, x, _nat(p)), mul_body) != 1:
        return False
    var lhs = _var(p, _nat(p))
    var rhs = _var(p, _nat(p))
    var left_pred = _var(p, _nat(p))
    var argument = _var(p, _nat(p))
    var previous = _var(p, _pi(p, argument, _prop(p)))
    var right = _var(p, _nat(p))
    var right_pred = _var(p, _nat(p))
    var ignored = _var(p, _prop(p))
    var right_step = _l2(p, right_pred, ignored, _app(p, previous, right_pred))
    var left_step = _l2(p, left_pred, previous,
        _lam(p, right, _rec(p, right, _false(p), right_step)))
    var le_body = _l2(p, lhs, rhs, _app(p,
        _rec(p, lhs, _lam(p, argument, _truth(p)), left_step), rhs))
    return _d(p, _p2(p, lhs, rhs, _prop(p)), le_body) == 2


def declare_nat_arithmetic_pred[C: Int](mut p: _ArithmeticBuilder[C]) -> Bool:
    if p.bundle.definition_count != 4:
        return False
    var n = _var(p, _nat(p))
    var predecessor = _var(p, _nat(p))
    var ignored = _var(p, _nat(p))
    return _d(p, _pi(p, n, _nat(p)), _lam(p, n,
        _rec(p, n, _number(p, 0), _l2(p, predecessor, ignored, predecessor)))) == 4


def declare_nat_arithmetic_base[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    """Base(stride, round) is data recursion, with no driver declarations."""
    if p.bundle.definition_count != 5:
        return -1
    var stride = _var(p, _nat(p))
    var round = _var(p, _nat(p))
    var predecessor = _var(p, _nat(p))
    var previous = _var(p, _nat(p))
    var step = _l2(p, predecessor, previous, _plus(p, previous, stride))
    var body = _l2(p, stride, round, _rec(p, round, _number(p, 0), step))
    return _d(p, _p2(p, stride, round, _nat(p)), body)


def _base_zero[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var stride = _var(p, _nat(p))
    var zero = _number(p, 0)
    var equation = _eq(p, _base(p, stride, zero), zero)
    return _d(p, _pi(p, stride, equation), _lam(p, stride, _refl(p, zero)))


def _base_step[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var stride = _var(p, _nat(p))
    var round = _var(p, _nat(p))
    var next = _plus(p, _base(p, stride, round), stride)
    var equation = _eq(p, _base(p, stride, _succ(p, round)), next)
    return _d(p, _p2(p, stride, round, equation),
        _l2(p, stride, round, _refl(p, next)))


def _succ_cong[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int, h: Int) -> Int:
    var x = _var(p, _nat(p))
    return _cong(p, _lam(p, x, _succ(p, x)), a, b, h)


def _succ_cancel[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int, h: Int) -> Int:
    return _cong(p, _named(p, 4), _succ(p, a), _succ(p, b), h)


def _le_transport_left[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _le(p, x, b)), equality, evidence)


def _le_transport_right[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _le(p, a, x)), equality, evidence)


def _add_zero_left[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _eq(p, _plus(p, _number(p, 0), q), q))
    var n = _var(p, _nat(p))
    var ih = _var(p, _eq(p, _plus(p, _number(p, 0), n), n))
    var step = _l2(p, n, ih, _succ_cong(p, _plus(p, _number(p, 0), n), n, ih))
    var body = _lam(p, target, _nat_induction(p, motive, _refl(p, _number(p, 0)), step, target))
    return _d(p, _pi(p, target, _eq(p, _plus(p, _number(p, 0), target), target)), body)


def _add_succ_left[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var a = _var(p, _nat(p))
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var conclusion = _eq(p, _plus(p, _succ(p, a), q), _succ(p, _plus(p, a, q)))
    var motive = _lam(p, q, conclusion)
    var n = _var(p, _nat(p))
    var ih = _var(p, _eq(p, _plus(p, _succ(p, a), n), _succ(p, _plus(p, a, n))))
    var step = _l2(p, n, ih, _succ_cong(p,
        _plus(p, _succ(p, a), n), _succ(p, _plus(p, a, n)), ih))
    var body = _l2(p, a, target, _nat_induction(p, motive, _refl(p, _succ(p, a)), step, target))
    return _d(p, _p2(p, a, target,
        _eq(p, _plus(p, _succ(p, a), target), _succ(p, _plus(p, a, target)))), body)


def _add_assoc[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var lhs = _plus(p, _plus(p, a, b), q)
    var rhs = _plus(p, a, _plus(p, b, q))
    var motive = _lam(p, q, _eq(p, lhs, rhs))
    var n = _var(p, _nat(p))
    var x = _plus(p, _plus(p, a, b), n)
    var y = _plus(p, a, _plus(p, b, n))
    var ih = _var(p, _eq(p, x, y))
    var step = _l2(p, n, ih, _succ_cong(p, x, y, ih))
    var body = _l3(p, a, b, target, _nat_induction(p, motive,
        _refl(p, _plus(p, a, b)), step, target))
    return _d(p, _p3(p, a, b, target, _eq(p,
        _plus(p, _plus(p, a, b), target), _plus(p, a, _plus(p, b, target)))), body)


def _add_comm[C: Int](mut p: _ArithmeticBuilder[C], zero_left: Int, succ_left: Int) -> Int:
    var a = _var(p, _nat(p))
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _eq(p, _plus(p, a, q), _plus(p, q, a)))
    var n = _var(p, _nat(p))
    var x = _plus(p, a, n)
    var y = _plus(p, n, a)
    var ih = _var(p, _eq(p, x, y))
    var edge = _app2(p, _named(p, succ_left), n, a)
    var step_eq = _trans(p, _succ(p, x), _succ(p, y), _plus(p, _succ(p, n), a),
        _succ_cong(p, x, y, ih), _sym(p, _plus(p, _succ(p, n), a), _succ(p, y), edge))
    var step = _l2(p, n, ih, step_eq)
    var base_eq = _app(p, _named(p, zero_left), a)
    var base = _sym(p, _plus(p, _number(p, 0), a), a, base_eq)
    var body = _l2(p, a, target, _nat_induction(p, motive, base, step, target))
    return _d(p, _p2(p, a, target, _eq(p, _plus(p, a, target), _plus(p, target, a))), body)


def _add_cancel[C: Int](mut p: _ArithmeticBuilder[C], comm: Int) -> Int:
    # First cancel a shared right suffix by structural induction.
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var h = _var(p, _eq(p, _plus(p, a, q), _plus(p, b, q)))
    var motive = _lam(p, q, _p3(p, a, b, h, _eq(p, a, b)))
    var aa = _var(p, _nat(p))
    var bb = _var(p, _nat(p))
    var hh = _var(p, _eq(p, aa, bb))
    var base = _l3(p, aa, bb, hh, hh)
    var n = _var(p, _nat(p))
    var ia = _var(p, _nat(p))
    var ib = _var(p, _nat(p))
    var eh = _var(p, _eq(p, _plus(p, ia, n), _plus(p, ib, n)))
    var ih = _var(p, _p3(p, ia, ib, eh, _eq(p, ia, ib)))
    var sa = _var(p, _nat(p))
    var sb = _var(p, _nat(p))
    var sh = _var(p, _eq(p, _plus(p, sa, _succ(p, n)), _plus(p, sb, _succ(p, n))))
    var step = _l2(p, n, ih, _l3(p, sa, sb, sh,
        _app3(p, ih, sa, sb, _succ_cancel(p, _plus(p, sa, n), _plus(p, sb, n), sh))))
    var cancel_right = _d(p, _pi(p, target,
        _p3(p, a, b, _var(p, _eq(p, _plus(p, a, target), _plus(p, b, target))), _eq(p, a, b))),
        _lam(p, target, _nat_induction(p, motive, base, step, target)))
    var prefix = _var(p, _nat(p))
    var left = _var(p, _nat(p))
    var right = _var(p, _nat(p))
    var equality = _var(p, _eq(p, _plus(p, prefix, left), _plus(p, prefix, right)))
    var first = _app2(p, _named(p, comm), left, prefix)
    var last = _app2(p, _named(p, comm), prefix, right)
    var middle = _trans(p, _plus(p, left, prefix), _plus(p, prefix, left),
        _plus(p, prefix, right), first, equality)
    var converted = _trans(p, _plus(p, left, prefix), _plus(p, prefix, right),
        _plus(p, right, prefix), middle, last)
    var body = _l3(p, prefix, left, right, _lam(p, equality,
        _app3(p, _app(p, _named(p, cancel_right), prefix), left, right, converted)))
    return _d(p, _p3(p, prefix, left, right, _pi(p, equality, _eq(p, left, right))), body)


def _nonneg[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var n = _var(p, _nat(p))
    return _d(p, _pi(p, n, _le(p, _number(p, 0), n)), _lam(p, n, _unit(p)))


def _le_refl[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _le(p, q, q))
    var n = _var(p, _nat(p))
    var ih = _var(p, _le(p, n, n))
    return _d(p, _pi(p, target, _le(p, target, target)), _lam(p, target,
        _nat_induction(p, motive, _unit(p), _l2(p, n, ih, ih), target)))


def _le_step[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var target = _var(p, _nat(p))
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var h = _var(p, _le(p, a, b))
    var motive = _lam(p, a, _p2(p, b, h, _le(p, a, _succ(p, b))))
    var zb = _var(p, _nat(p))
    var zh = _var(p, _le(p, _number(p, 0), zb))
    var base = _l2(p, zb, zh, _unit(p))
    var n = _var(p, _nat(p))
    var ib = _var(p, _nat(p))
    var ihh = _var(p, _le(p, n, ib))
    var ih = _var(p, _p2(p, ib, ihh, _le(p, n, _succ(p, ib))))
    var bt = _var(p, _nat(p))
    var mb = _var(p, _nat(p))
    var mh = _var(p, _le(p, _succ(p, n), mb))
    var bmot = _lam(p, mb, _pi(p, mh, _le(p, _succ(p, n), _succ(p, mb))))
    var bad = _var(p, _le(p, _succ(p, n), _number(p, 0)))
    var bbase = _lam(p, bad, _absurd(p, bad, _le(p, _succ(p, n), _number(p, 1))))
    var prev = _var(p, _nat(p))
    var bh = _var(p, _pi(p, _var(p, _le(p, _succ(p, n), prev)),
        _le(p, _succ(p, n), _succ(p, prev))))
    var hh = _var(p, _le(p, _succ(p, n), _succ(p, prev)))
    var bstep = _l2(p, prev, bh, _lam(p, hh, _app2(p, ih, prev, hh)))
    var step = _l2(p, n, ih, _lam(p, bt, _nat_induction(p, bmot, bbase, bstep, bt)))
    return _d(p, _pi(p, target, _p2(p, b, _var(p, _le(p, target, b)),
        _le(p, target, _succ(p, b)))), _lam(p, target, _nat_induction(p, motive, base, step, target)))


def _le_trans[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var at = _var(p, _nat(p))
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var c = _var(p, _nat(p))
    var h = _var(p, _le(p, a, b))
    var g = _var(p, _le(p, b, c))
    var statement = _p2(p, b, c, _p2(p, h, g, _le(p, a, c)))
    var motive = _lam(p, a, statement)
    var zb = _var(p, _nat(p))
    var zc = _var(p, _nat(p))
    var zh = _var(p, _le(p, _number(p, 0), zb))
    var zg = _var(p, _le(p, zb, zc))
    var base = _l2(p, zb, zc, _l2(p, zh, zg, _unit(p)))
    var n = _var(p, _nat(p))
    var ib = _var(p, _nat(p))
    var ic = _var(p, _nat(p))
    var ihh = _var(p, _le(p, n, ib))
    var ihg = _var(p, _le(p, ib, ic))
    var ih = _var(p, _p2(p, ib, ic, _p2(p, ihh, ihg, _le(p, n, ic))))
    var bt = _var(p, _nat(p))
    var mb = _var(p, _nat(p))
    var mc = _var(p, _nat(p))
    var mh = _var(p, _le(p, _succ(p, n), mb))
    var mg = _var(p, _le(p, mb, mc))
    var bmot = _lam(p, mb, _pi(p, mc, _p2(p, mh, mg, _le(p, _succ(p, n), mc))))
    var bc = _var(p, _nat(p))
    var bh = _var(p, _le(p, _succ(p, n), _number(p, 0)))
    var bg = _var(p, _le(p, _number(p, 0), bc))
    var bbase = _lam(p, bc, _l2(p, bh, bg, _absurd(p, bh, _le(p, _succ(p, n), bc))))
    var bp = _var(p, _nat(p))
    var bix = _var(p, _nat(p))
    var bih = _var(p, _pi(p, bix, _p2(p,
        _var(p, _le(p, _succ(p, n), bp)), _var(p, _le(p, bp, bix)), _le(p, _succ(p, n), bix))))
    var ct = _var(p, _nat(p))
    var cm = _var(p, _nat(p))
    var ch = _var(p, _le(p, _succ(p, n), _succ(p, bp)))
    var cg = _var(p, _le(p, _succ(p, bp), cm))
    var cmot = _lam(p, cm, _p2(p, ch, cg, _le(p, _succ(p, n), cm)))
    var cz_h = _var(p, _le(p, _succ(p, n), _succ(p, bp)))
    var cz_g = _var(p, _le(p, _succ(p, bp), _number(p, 0)))
    var cbase = _l2(p, cz_h, cz_g, _absurd(p, cz_g, _le(p, _succ(p, n), _number(p, 0))))
    var cp = _var(p, _nat(p))
    var cih = _var(p, _p2(p, _var(p, _le(p, _succ(p, n), _succ(p, bp))),
        _var(p, _le(p, _succ(p, bp), cp)), _le(p, _succ(p, n), cp)))
    var cs_h = _var(p, _le(p, _succ(p, n), _succ(p, bp)))
    var cs_g = _var(p, _le(p, _succ(p, bp), _succ(p, cp)))
    var cstep = _l2(p, cp, cih, _l2(p, cs_h, cs_g,
        _app2(p, _app2(p, ih, bp, cp), cs_h, cs_g)))
    var bstep = _l2(p, bp, bih, _lam(p, ct, _nat_induction(p, cmot, cbase, cstep, ct)))
    var step = _l2(p, n, ih, _lam(p, bt, _nat_induction(p, bmot, bbase, bstep, bt)))
    var eb = _var(p, _nat(p))
    var ec = _var(p, _nat(p))
    var eh = _var(p, _le(p, at, eb))
    var eg = _var(p, _le(p, eb, ec))
    return _d(p, _pi(p, at, _p2(p, eb, ec, _p2(p, eh, eg, _le(p, at, ec)))),
        _lam(p, at, _nat_induction(p, motive, base, step, at)))


def _le_add[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var target = _var(p, _nat(p))
    var h = _var(p, _le(p, a, b))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _le(p, _plus(p, a, q), _plus(p, b, q)))
    var n = _var(p, _nat(p))
    var ih = _var(p, _le(p, _plus(p, a, n), _plus(p, b, n)))
    var body = _l3(p, a, b, target, _lam(p, h,
        _nat_induction(p, motive, h, _l2(p, n, ih, ih), target)))
    return _d(p, _p3(p, a, b, target, _pi(p, h,
        _le(p, _plus(p, a, target), _plus(p, b, target)))), body)


def _lt_add[C: Int](mut p: _ArithmeticBuilder[C], le_add: Int, succ_left: Int) -> Int:
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var c = _var(p, _nat(p))
    var h = _var(p, _lt(p, a, b))
    var shifted = _app(p, _app3(p, _named(p, le_add), _succ(p, a), b, c), h)
    var equality = _app2(p, _named(p, succ_left), a, c)
    var body = _l3(p, a, b, c, _lam(p, h, _le_transport_left(p,
        _plus(p, _succ(p, a), c), _plus(p, b, c), equality, shifted)))
    return _d(p, _p3(p, a, b, c, _pi(p, h,
        _lt(p, _plus(p, a, c), _plus(p, b, c)))), body)


def _mixed_trans[C: Int](mut p: _ArithmeticBuilder[C], trans: Int, strict_first: Bool) -> Int:
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var c = _var(p, _nat(p))
    var h = _var(p, _lt(p, a, b) if strict_first else _le(p, a, b))
    var g = _var(p, _le(p, b, c) if strict_first else _lt(p, b, c))
    var body = _l3(p, a, b, c, _l2(p, h, g, _app2(p,
        _app3(p, _named(p, trans), _succ(p, a), b if strict_first else _succ(p, b), c), h, g)))
    return _d(p, _p3(p, a, b, c, _p2(p, h, g, _lt(p, a, c))), body)


def _lt_to_le_succ[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var h = _var(p, _lt(p, a, b))
    return _d(p, _p3(p, a, b, h, _le(p, _plus(p, a, _number(p, 1)), b)),
        _l3(p, a, b, h, h))


def _lt_irrefl[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var h = _var(p, _lt(p, q, q))
    var motive = _lam(p, q, _pi(p, h, _false(p)))
    var zero_h = _var(p, _lt(p, _number(p, 0), _number(p, 0)))
    var base = _lam(p, zero_h, zero_h)
    var n = _var(p, _nat(p))
    var ih = _var(p, _pi(p, _var(p, _lt(p, n, n)), _false(p)))
    var next_h = _var(p, _lt(p, _succ(p, n), _succ(p, n)))
    var step = _l2(p, n, ih, _lam(p, next_h, _app(p, ih, next_h)))
    return _d(p, _pi(p, target, _pi(p, _var(p, _lt(p, target, target)), _false(p))),
        _lam(p, target, _nat_induction(p, motive, base, step, target)))


def _order_type[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int) -> Int:
    var result = _var(p, _prop(p))
    var lt_guard = _var(p, _lt(p, a, b))
    var ge_guard = _var(p, _le(p, b, a))
    var lt_case = _var(p, _pi(p, lt_guard, result))
    var ge_case = _var(p, _pi(p, ge_guard, result))
    return _p3(p, result, lt_case, ge_case, result)


def _order_cases[C: Int](mut p: _ArithmeticBuilder[C], nonneg: Int) -> Int:
    var target = _var(p, _nat(p))
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var motive = _lam(p, a, _pi(p, b, _order_type(p, a, b)))
    var bt = _var(p, _nat(p))
    var mb = _var(p, _nat(p))
    var bmot = _lam(p, mb, _order_type(p, _number(p, 0), mb))
    var pz = _var(p, _prop(p))
    var fz = _var(p, _pi(p, _var(p, _lt(p, _number(p, 0), _number(p, 0))), pz))
    var gz = _var(p, _pi(p, _var(p, _le(p, _number(p, 0), _number(p, 0))), pz))
    var zbase = _l3(p, pz, fz, gz, _app(p, gz, _unit(p)))
    var bp = _var(p, _nat(p))
    var bih = _var(p, _order_type(p, _number(p, 0), bp))
    var ps = _var(p, _prop(p))
    var fs = _var(p, _pi(p, _var(p, _lt(p, _number(p, 0), _succ(p, bp))), ps))
    var gs = _var(p, _pi(p, _var(p, _le(p, _succ(p, bp), _number(p, 0))), ps))
    var zstep = _l2(p, bp, bih, _l3(p, ps, fs, gs,
        _app(p, fs, _app(p, _named(p, nonneg), bp))))
    var base = _lam(p, bt, _nat_induction(p, bmot, zbase, zstep, bt))
    var n = _var(p, _nat(p))
    var ib = _var(p, _nat(p))
    var ih = _var(p, _pi(p, ib, _order_type(p, n, ib)))
    var st = _var(p, _nat(p))
    var sb = _var(p, _nat(p))
    var smot = _lam(p, sb, _order_type(p, _succ(p, n), sb))
    var p0 = _var(p, _prop(p))
    var f0 = _var(p, _pi(p, _var(p, _lt(p, _succ(p, n), _number(p, 0))), p0))
    var g0 = _var(p, _pi(p, _var(p, _le(p, _number(p, 0), _succ(p, n))), p0))
    var sbase = _l3(p, p0, f0, g0,
        _app(p, g0, _app(p, _named(p, nonneg), _succ(p, n))))
    var prev = _var(p, _nat(p))
    var sih = _var(p, _order_type(p, _succ(p, n), prev))
    var pp = _var(p, _prop(p))
    var ff = _var(p, _pi(p, _var(p, _lt(p, _succ(p, n), _succ(p, prev))), pp))
    var gg = _var(p, _pi(p, _var(p, _le(p, _succ(p, prev), _succ(p, n))), pp))
    var sstep = _l2(p, prev, sih, _l3(p, pp, ff, gg,
        _app3(p, _app(p, ih, prev), pp, ff, gg)))
    var step = _l2(p, n, ih, _lam(p, st, _nat_induction(p, smot, sbase, sstep, st)))
    var eb = _var(p, _nat(p))
    return _d(p, _pi(p, target, _pi(p, eb, _order_type(p, target, eb))),
        _lam(p, target, _nat_induction(p, motive, base, step, target)))


def _prefix_mono[C: Int](mut p: _ArithmeticBuilder[C], prefix: Int,
    a: Int, b: Int, h: Int, le_add: Int, comm: Int) -> Int:
    var shifted = _app(p, _app3(p, _named(p, le_add), a, b, prefix), h)
    var left_eq = _app2(p, _named(p, comm), a, prefix)
    var right_eq = _app2(p, _named(p, comm), b, prefix)
    var first = _le_transport_left(p, _plus(p, a, prefix), _plus(p, b, prefix), left_eq, shifted)
    return _le_transport_right(p, _plus(p, prefix, a), _plus(p, b, prefix), right_eq, first)


def _le_scale[C: Int](mut p: _ArithmeticBuilder[C], nonneg: Int, le_add: Int, comm: Int) -> Int:
    var k = _var(p, _nat(p))
    var at = _var(p, _nat(p))
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var h = _var(p, _le(p, a, b))
    var motive = _lam(p, a, _p2(p, b, h, _le(p, _mul(p, k, a), _mul(p, k, b))))
    var zb = _var(p, _nat(p))
    var zh = _var(p, _le(p, _number(p, 0), zb))
    var base = _l2(p, zb, zh, _app(p, _named(p, nonneg), _mul(p, k, zb)))
    var n = _var(p, _nat(p))
    var ib = _var(p, _nat(p))
    var ihh = _var(p, _le(p, n, ib))
    var ih = _var(p, _p2(p, ib, ihh, _le(p, _mul(p, k, n), _mul(p, k, ib))))
    var bt = _var(p, _nat(p))
    var bm = _var(p, _nat(p))
    var bh = _var(p, _le(p, _succ(p, n), bm))
    var bmot = _lam(p, bm, _pi(p, bh, _le(p, _mul(p, k, _succ(p, n)), _mul(p, k, bm))))
    var bad = _var(p, _le(p, _succ(p, n), _number(p, 0)))
    var bbase = _lam(p, bad, _absurd(p, bad, _le(p, _mul(p, k, _succ(p, n)), _number(p, 0))))
    var prev = _var(p, _nat(p))
    var bih = _var(p, _pi(p, _var(p, _le(p, _succ(p, n), prev)),
        _le(p, _mul(p, k, _succ(p, n)), _mul(p, k, prev))))
    var hh = _var(p, _le(p, _succ(p, n), _succ(p, prev)))
    var shifted = _prefix_mono(p, k, _mul(p, k, n), _mul(p, k, prev),
        _app2(p, ih, prev, hh), le_add, comm)
    var bstep = _l2(p, prev, bih, _lam(p, hh, shifted))
    var step = _l2(p, n, ih, _lam(p, bt, _nat_induction(p, bmot, bbase, bstep, bt)))
    var eb = _var(p, _nat(p))
    var eh = _var(p, _le(p, at, eb))
    return _d(p, _p3(p, k, at, eb, _pi(p, eh, _le(p, _mul(p, k, at), _mul(p, k, eb)))),
        _l2(p, k, at, _nat_induction(p, motive, base, step, at)))


def _scale_cancel[C: Int](mut p: _ArithmeticBuilder[C], cases: Int,
    scale: Int, mixed: Int, irrefl: Int) -> Int:
    var k = _var(p, _nat(p))
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var h = _var(p, _lt(p, _mul(p, k, a), _mul(p, k, b)))
    var first = _var(p, _lt(p, a, b))
    var other = _var(p, _le(p, b, a))
    var reversed = _app(p, _app3(p, _named(p, scale), k, b, a), other)
    var contradiction = _app2(p, _app3(p, _named(p, mixed),
        _mul(p, k, a), _mul(p, k, b), _mul(p, k, a)), h, reversed)
    var impossible = _app2(p, _named(p, irrefl), _mul(p, k, a), contradiction)
    var right = _lam(p, other, _absurd(p, impossible, _lt(p, a, b)))
    var result = _app3(p, _app2(p, _named(p, cases), a, b), _lt(p, a, b),
        _lam(p, first, first), right)
    return _d(p, _p3(p, k, a, b, _pi(p, h, _lt(p, a, b))), _l3(p, k, a, b, _lam(p, h, result)))


def _mul_succ_rhs[C: Int](mut p: _ArithmeticBuilder[C], comm: Int) -> Int:
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    return _d(p, _p2(p, a, b, _eq(p, _mul(p, a, _succ(p, b)),
        _plus(p, _mul(p, a, b), a))),
        _l2(p, a, b, _app2(p, _named(p, comm), a, _mul(p, a, b))))


def _mul_zero_left[C: Int](mut p: _ArithmeticBuilder[C], zero_left: Int) -> Int:
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var zero = _number(p, 0)
    var motive = _lam(p, q, _eq(p, _mul(p, zero, q), zero))
    var n = _var(p, _nat(p))
    var ih = _var(p, _eq(p, _mul(p, zero, n), zero))
    var unfold = _app(p, _named(p, zero_left), _mul(p, zero, n))
    var step_eq = _trans(p, _plus(p, zero, _mul(p, zero, n)), _mul(p, zero, n), zero, unfold, ih)
    return _d(p, _pi(p, target, _eq(p, _mul(p, zero, target), zero)),
        _lam(p, target, _nat_induction(p, motive, _refl(p, zero), _l2(p, n, ih, step_eq), target)))


def _mul_succ_left[C: Int](mut p: _ArithmeticBuilder[C], succ_left: Int, assoc: Int) -> Int:
    var k = _var(p, _nat(p))
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _eq(p, _mul(p, _succ(p, k), q), _plus(p, _mul(p, k, q), q)))
    var n = _var(p, _nat(p))
    var lhs_acc = _mul(p, _succ(p, k), n)
    var rhs_acc = _plus(p, _mul(p, k, n), n)
    var ih = _var(p, _eq(p, lhs_acc, rhs_acc))
    var lhs = _plus(p, _succ(p, k), lhs_acc)
    var middle = _plus(p, _succ(p, k), rhs_acc)
    var inner = _plus(p, k, rhs_acc)
    var last_inner = _plus(p, _plus(p, k, _mul(p, k, n)), n)
    var prefix_edge = _cong_right(p, _succ(p, k), lhs_acc, rhs_acc, ih)
    var successor_edge = _app2(p, _named(p, succ_left), k, rhs_acc)
    var associativity = _app3(p, _named(p, assoc), k, _mul(p, k, n), n)
    var flipped = _sym(p, last_inner, inner, associativity)
    var first = _trans(p, lhs, middle, _succ(p, inner), prefix_edge, successor_edge)
    var completed = _trans(p, lhs, _succ(p, inner), _succ(p, last_inner), first,
        _succ_cong(p, inner, last_inner, flipped))
    var body = _l2(p, k, target, _nat_induction(p, motive, _refl(p, _number(p, 0)),
        _l2(p, n, ih, completed), target))
    return _d(p, _p2(p, k, target, _eq(p, _mul(p, _succ(p, k), target),
        _plus(p, _mul(p, k, target), target))), body)


def _le_antisym[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var target = _var(p, _nat(p))
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var h = _var(p, _le(p, a, b))
    var g = _var(p, _le(p, b, a))
    var motive = _lam(p, a, _pi(p, b, _p2(p, h, g, _eq(p, a, b))))
    var bt = _var(p, _nat(p))
    var mb = _var(p, _nat(p))
    var mh = _var(p, _le(p, _number(p, 0), mb))
    var mg = _var(p, _le(p, mb, _number(p, 0)))
    var bmot = _lam(p, mb, _p2(p, mh, mg, _eq(p, _number(p, 0), mb)))
    var z0 = _var(p, _le(p, _number(p, 0), _number(p, 0)))
    var z1 = _var(p, _le(p, _number(p, 0), _number(p, 0)))
    var bbase = _l2(p, z0, z1, _refl(p, _number(p, 0)))
    var bp = _var(p, _nat(p))
    var bih = _var(p, _p2(p, _var(p, _le(p, _number(p, 0), bp)),
        _var(p, _le(p, bp, _number(p, 0))), _eq(p, _number(p, 0), bp)))
    var bs_h = _var(p, _le(p, _number(p, 0), _succ(p, bp)))
    var bs_g = _var(p, _le(p, _succ(p, bp), _number(p, 0)))
    var bstep = _l2(p, bp, bih, _l2(p, bs_h, bs_g,
        _absurd(p, bs_g, _eq(p, _number(p, 0), _succ(p, bp)))))
    var base = _lam(p, bt, _nat_induction(p, bmot, bbase, bstep, bt))
    var n = _var(p, _nat(p))
    var ib = _var(p, _nat(p))
    var ih = _var(p, _pi(p, ib, _p2(p, _var(p, _le(p, n, ib)),
        _var(p, _le(p, ib, n)), _eq(p, n, ib))))
    var st = _var(p, _nat(p))
    var sm = _var(p, _nat(p))
    var sh = _var(p, _le(p, _succ(p, n), sm))
    var sg = _var(p, _le(p, sm, _succ(p, n)))
    var smot = _lam(p, sm, _p2(p, sh, sg, _eq(p, _succ(p, n), sm)))
    var bad = _var(p, _le(p, _succ(p, n), _number(p, 0)))
    var s0 = _var(p, _le(p, _number(p, 0), _succ(p, n)))
    var sbase = _l2(p, bad, s0, _absurd(p, bad, _eq(p, _succ(p, n), _number(p, 0))))
    var prev = _var(p, _nat(p))
    var sih = _var(p, _p2(p, _var(p, _le(p, _succ(p, n), prev)),
        _var(p, _le(p, prev, _succ(p, n))), _eq(p, _succ(p, n), prev)))
    var ss_h = _var(p, _le(p, _succ(p, n), _succ(p, prev)))
    var ss_g = _var(p, _le(p, _succ(p, prev), _succ(p, n)))
    var equality = _app2(p, _app(p, ih, prev), ss_h, ss_g)
    var sstep = _l2(p, prev, sih, _l2(p, ss_h, ss_g, _succ_cong(p, n, prev, equality)))
    var step = _l2(p, n, ih, _lam(p, st, _nat_induction(p, smot, sbase, sstep, st)))
    var eb = _var(p, _nat(p))
    return _d(p, _pi(p, target, _pi(p, eb, _p2(p, _var(p, _le(p, target, eb)),
        _var(p, _le(p, eb, target)), _eq(p, target, eb)))),
        _lam(p, target, _nat_induction(p, motive, base, step, target)))


def make_nat_arithmetic_core[C: Int](mut p: _ArithmeticBuilder[C]) -> NatArithmeticCore:
    """Append untrusted checked-definition candidates; returned values are IDs."""
    var zero_left = _add_zero_left(p)
    var succ_left = _add_succ_left(p)
    var assoc = _add_assoc(p)
    var comm = _add_comm(p, zero_left, succ_left)
    var cancel = _add_cancel(p, comm)
    var mul_succ_rhs = _mul_succ_rhs(p, comm)
    var mul_succ_left = _mul_succ_left(p, succ_left, assoc)
    var mul_zero_left = _mul_zero_left(p, zero_left)
    var nonneg = _nonneg(p)
    var refl = _le_refl(p)
    var step = _le_step(p)
    var trans = _le_trans(p)
    var le_add = _le_add(p)
    var lt_add = _lt_add(p, le_add, succ_left)
    var le_lt = _mixed_trans(p, trans, False)
    var lt_le = _mixed_trans(p, trans, True)
    var lt_succ = _lt_to_le_succ(p)
    var irrefl = _lt_irrefl(p)
    var cases = _order_cases(p, nonneg)
    var scale = _le_scale(p, nonneg, le_add, comm)
    var scale_cancel = _scale_cancel(p, cases, scale, lt_le, irrefl)
    var antisym = _le_antisym(p)
    return NatArithmeticCore(zero_left, succ_left, assoc, comm, cancel,
        mul_succ_rhs, mul_succ_left, mul_zero_left, nonneg, refl, step, trans,
        le_add, lt_add, le_lt, lt_le, lt_succ, scale, scale_cancel, cases, irrefl, antisym)


def declare_nat_operator_representations[C: Int](mut p: _ArithmeticBuilder[C]) -> Array[Int, NAT_REPRESENTATIONS]:
    """Closed raw abbreviation candidates; their meaning is checked separately."""
    var definitions = Array[Int, NAT_REPRESENTATIONS](fill=-1)
    for representation in range(NAT_REPRESENTATIONS):
        var a = _var(p, _nat(p))
        var b = _var(p, _nat(p))
        var body: Int
        var result = _nat(p)
        if representation == NAT_REP_ADD:
            body = _plus(p, a, b)
        elif representation == NAT_REP_MUL or representation == NAT_REP_SCALE:
            body = _mul(p, a, b)
        elif representation == NAT_REP_LE:
            body = _le(p, a, b)
            result = _prop(p)
        elif representation == NAT_REP_LT:
            body = _lt(p, a, b)
            result = _prop(p)
        elif representation == NAT_REP_BASE:
            body = _base(p, a, b)
        else:
            var q = _var(p, _nat(p))
            var r = _var(p, _nat(p))
            var c = _var(p, _nat(p))
            var selector = _l3(p, q, r, c, q if representation == NAT_REP_DIV else r)
            body = _app(p, _app2(p, _named(p, 3), a, b), selector)
        definitions[representation] = _d(p, _p2(p, a, b, result), _l2(p, a, b, body))
    return definitions^


def make_nat_arithmetic_library[capacity: Int = NAT_ARITHMETIC_CAPACITY]() -> NatArithmeticLibrary[capacity]:
    var p = _empty_builder[capacity]()
    var declarations = declare_nat_arithmetic_definitions(p)
    var division_definition = declare_nat_arithmetic_division(p)
    var predecessor_definition = declare_nat_arithmetic_pred(p)
    var base_definition = declare_nat_arithmetic_base(p)
    var base_zero = _base_zero(p)
    var base_step = _base_step(p)
    var core = make_nat_arithmetic_core(p)
    var division = make_nat_arithmetic_division(p, core)
    var representations = declare_nat_operator_representations(p)
    var exports = Array[Int,18](fill=-1)
    exports[tk.DEC_BASE_ZERO] = base_zero
    exports[tk.DEC_BASE_STEP] = base_step
    exports[tk.DEC_NAT_NONNEG] = core.nat_nonneg
    exports[tk.DEC_LE_ADD] = core.le_add
    exports[tk.DEC_LE_TRANS] = core.le_trans
    exports[tk.DEC_ORDER_CASES] = core.order_cases
    exports[tk.DEC_LT_ADD] = core.lt_add
    exports[tk.DEC_LE_LT_TRANS] = core.le_lt_trans
    exports[tk.DEC_LT_LE_TRANS] = core.lt_le_trans
    exports[tk.DEC_LE_SCALE] = core.le_scale
    exports[tk.DEC_LT_SCALE_CANCEL] = core.lt_scale_cancel
    exports[tk.DEC_LT_TO_LE_SUCC] = core.lt_to_le_succ
    exports[tk.DEC_DIV_DECOMP] = division.decomposition
    exports[tk.DEC_MOD_LT] = division.remainder_bound
    exports[tk.DEC_DIV_QUOTIENT_UNIQUE] = division.quotient_unique
    exports[tk.DEC_DIV_REMAINDER_UNIQUE] = division.remainder_unique
    if (not declarations or division_definition != 3 or not predecessor_definition
        or base_definition != 5):
        p.bundle.count = -1
    return NatArithmeticLibrary[capacity](p.bundle.copy(), exports^, representations^)
