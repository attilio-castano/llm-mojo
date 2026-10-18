"""Untrusted constructors for ordinary Nat arithmetic proof terms.

These helpers produce syntax only. Named arithmetic operations are applications
of closed, acyclic definitions; every resulting term is checked by the same
generic dependent proof-term kernel.
"""
import llm_mojo.kernels.proof_term_syntax as tk
from std.collections import Array

comptime NAT_REP_ADD = 0
comptime NAT_REP_MUL = 1
comptime NAT_REP_LE = 2
comptime NAT_REP_LT = 3
comptime NAT_REP_SCALE = 4
comptime NAT_REP_BASE = 5
comptime NAT_REP_DIV = 6
comptime NAT_REP_MOD = 7
comptime NAT_REPRESENTATIONS = 8


@fieldwise_init
struct NatArithmeticLibrary[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var exports: Array[Int, 18]
    # These are untrusted definition indices, checked independently of exports.
    var representations: Array[Int, NAT_REPRESENTATIONS]


@fieldwise_init
struct NatArithmeticCore(ImplicitlyCopyable):
    var add_zero_left: Int
    var add_succ_left: Int
    var add_assoc: Int
    var add_comm: Int
    var add_cancel_prefix: Int
    var mul_succ_rhs: Int
    var mul_succ_left: Int
    var mul_zero_left: Int
    var nat_nonneg: Int
    var le_refl: Int
    var le_step: Int
    var le_trans: Int
    var le_add: Int
    var lt_add: Int
    var le_lt_trans: Int
    var lt_le_trans: Int
    var lt_to_le_succ: Int
    var le_scale: Int
    var lt_scale_cancel: Int
    var order_cases: Int
    var lt_irrefl: Int
    var le_antisym: Int


def _node[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _nat[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _node(p, tk.PT_NAT)


def _prop[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _node(p, tk.PT_PROP)


def _var[C: Int](mut p: tk.ProofTermBundle[C], declared: Int) -> Int:
    return _node(p, tk.PT_VAR, declared)


def _number[C: Int](mut p: tk.ProofTermBundle[C], value: Int) -> Int:
    return _node(p, tk.PT_LITERAL, value=value)


def _succ[C: Int](mut p: tk.ProofTermBundle[C], a: Int) -> Int:
    return _node(p, tk.PT_SUCC, a)


def _named[C: Int](mut p: tk.ProofTermBundle[C], definition: Int) -> Int:
    return _node(p, tk.PT_DEFINED_REF, value=definition)


def _app[C: Int](mut p: tk.ProofTermBundle[C], f: Int, a: Int) -> Int:
    return _node(p, tk.PT_APP, f, a)


def _app2[C: Int](mut p: tk.ProofTermBundle[C], f: Int, a: Int, b: Int) -> Int:
    return _app(p, _app(p, f, a), b)


def _app3[C: Int](mut p: tk.ProofTermBundle[C], f: Int, a: Int, b: Int, c: Int) -> Int:
    return _app(p, _app2(p, f, a, b), c)


def _pi[C: Int](mut p: tk.ProofTermBundle[C], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_PI, binder, body)


def _lam[C: Int](mut p: tk.ProofTermBundle[C], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, binder, body)


def _plus[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 0), a, b)


def _mul[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 1), a, b)


def _eq[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_EQ, a, b)


def _le[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 2), a, b)


def _lt[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int) -> Int:
    return _le(p, _succ(p, a), b)


def _refl[C: Int](mut p: tk.ProofTermBundle[C], a: Int) -> Int:
    return _node(p, tk.PT_REFL, a)


def _subst[C: Int](mut p: tk.ProofTermBundle[C], motive: Int,
    equality: Int, evidence: Int) -> Int:
    return _node(p, tk.PT_SUBST, motive, equality, evidence)


def _sym[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, x, a)), equality, _refl(p, a))


def _trans[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int,
    c: Int, ab: Int, bc: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, a, x)), bc, ab)


def _cong[C: Int](mut p: tk.ProofTermBundle[C], function: Int,
    a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    var first = _app(p, function, a)
    return _subst(p, _lam(p, x, _eq(p, first, _app(p, function, x))), equality,
        _refl(p, first))


def _cong_left[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int,
    equality: Int, suffix: Int) -> Int:
    var x = _var(p, _nat(p))
    return _cong(p, _lam(p, x, _plus(p, x, suffix)), a, b, equality)


def _cong_right[C: Int](mut p: tk.ProofTermBundle[C], prefix: Int,
    a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _cong(p, _lam(p, x, _plus(p, prefix, x)), a, b, equality)


def _nat_induction[C: Int](mut p: tk.ProofTermBundle[C], motive: Int,
    base: Int, step: Int, target: Int) -> Int:
    return _app(p, _app3(p, _node(p, tk.PT_ELIMINATOR, value=1),
        motive, base, step), target)


def _rec[C: Int](mut p: tk.ProofTermBundle[C], major: Int, base: Int,
    step: Int) -> Int:
    return _node(p, tk.PT_NAT_REC, major, base, step)


def _truth[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    var proposition = _var(p, _prop(p))
    var evidence = _var(p, proposition)
    return _pi(p, proposition, _pi(p, evidence, proposition))


def _unit[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    var proposition = _var(p, _prop(p))
    var evidence = _var(p, proposition)
    return _lam(p, proposition, _lam(p, evidence, evidence))


def _false[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    var proposition = _var(p, _prop(p))
    return _pi(p, proposition, proposition)


def _absurd[C: Int](mut p: tk.ProofTermBundle[C], contradiction: Int,
    proposition: Int) -> Int:
    return _app(p, contradiction, proposition)


def _and_type[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int) -> Int:
    var proposition = _var(p, _prop(p))
    var first = _var(p, a)
    var second = _var(p, b)
    var continuation = _var(p, _pi(p, first, _pi(p, second, proposition)))
    return _pi(p, proposition, _pi(p, continuation, proposition))


def _and_intro[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int,
    ha: Int, hb: Int) -> Int:
    var proposition = _var(p, _prop(p))
    var first = _var(p, a)
    var second = _var(p, b)
    var continuation = _var(p, _pi(p, first, _pi(p, second, proposition)))
    return _lam(p, proposition, _lam(p, continuation, _app2(p, continuation, ha, hb)))


def _and_left[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int,
    evidence: Int) -> Int:
    var first = _var(p, a)
    var second = _var(p, b)
    return _app2(p, evidence, a, _lam(p, first, _lam(p, second, first)))


def _and_right[C: Int](mut p: tk.ProofTermBundle[C], a: Int, b: Int,
    evidence: Int) -> Int:
    var first = _var(p, a)
    var second = _var(p, b)
    return _app2(p, evidence, b, _lam(p, first, _lam(p, second, second)))


# Private append-only producer index. Plain-bundle overloads above stay
# available for independent syntax construction and mutation controls.
# Neither these helpers nor the cache grants a typing judgment.
@fieldwise_init
struct _ArithmeticBuilder[capacity: Int](Copyable, Movable):
    # Producer-only storage. The returned library contains the bundle alone.
    var bundle: tk.ProofTermBundle[Self.capacity]
    var index: Dict[Tuple[Int, Int, Int, Int, Int], Int]


def _builder_from_bundle[C: Int](var bundle: tk.ProofTermBundle[C]) -> _ArithmeticBuilder[C]:
    var index = Dict[Tuple[Int, Int, Int, Int, Int], Int]()
    if bundle.count >= 0 and bundle.count <= C:
        for j in range(bundle.count):
            var n = bundle.nodes[j]
            if n.kind != tk.PT_VAR:
                var key = (n.kind, n.left, n.right, n.extra, n.value)
                if key not in index:
                    index[key] = j
    return _ArithmeticBuilder[C](bundle^, index^)


def _empty_builder[C: Int]() -> _ArithmeticBuilder[C]:
    return _builder_from_bundle(tk.empty_proof_terms[C]())


def _indexed_add[C: Int](mut p: _ArithmeticBuilder[C], kind: Int,
    left: Int = -1, right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    if p.bundle.count < 0 or p.bundle.count > C:
        return -1
    var key = (kind, left, right, extra, value)
    # Preserve first-match reuse at the capacity bound; variables always
    # get fresh binder identities and never enter the index.
    if kind != tk.PT_VAR and key in p.index:
        return p.index.get(key, -1)
    if p.bundle.count == C:
        return -1
    var node = p.bundle.count
    p.bundle.nodes[node] = tk.ProofTerm(kind, left, right, extra, value)
    p.bundle.count += 1
    if kind != tk.PT_VAR:
        p.index[key] = node
    return node


def _declare_definition[C: Int](mut p: _ArithmeticBuilder[C], type: Int, body: Int) -> Int:
    return tk.pt_declare_definition(p.bundle, type, body)


def _node[C: Int](mut p: _ArithmeticBuilder[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return _indexed_add(p, kind, left, right, extra, value)


def _nat[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    return _node(p, tk.PT_NAT)


def _prop[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    return _node(p, tk.PT_PROP)


def _var[C: Int](mut p: _ArithmeticBuilder[C], declared: Int) -> Int:
    return _node(p, tk.PT_VAR, declared)


def _number[C: Int](mut p: _ArithmeticBuilder[C], value: Int) -> Int:
    return _node(p, tk.PT_LITERAL, value=value)


def _succ[C: Int](mut p: _ArithmeticBuilder[C], a: Int) -> Int:
    return _node(p, tk.PT_SUCC, a)


def _named[C: Int](mut p: _ArithmeticBuilder[C], definition: Int) -> Int:
    return _node(p, tk.PT_DEFINED_REF, value=definition)


def _app[C: Int](mut p: _ArithmeticBuilder[C], f: Int, a: Int) -> Int:
    return _node(p, tk.PT_APP, f, a)


def _app2[C: Int](mut p: _ArithmeticBuilder[C], f: Int, a: Int, b: Int) -> Int:
    return _app(p, _app(p, f, a), b)


def _app3[C: Int](mut p: _ArithmeticBuilder[C], f: Int, a: Int, b: Int, c: Int) -> Int:
    return _app(p, _app2(p, f, a, b), c)


def _pi[C: Int](mut p: _ArithmeticBuilder[C], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_PI, binder, body)


def _lam[C: Int](mut p: _ArithmeticBuilder[C], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, binder, body)


def _plus[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 0), a, b)


def _mul[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 1), a, b)


def _eq[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_EQ, a, b)


def _le[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 2), a, b)


def _lt[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int) -> Int:
    return _le(p, _succ(p, a), b)


def _refl[C: Int](mut p: _ArithmeticBuilder[C], a: Int) -> Int:
    return _node(p, tk.PT_REFL, a)


def _subst[C: Int](mut p: _ArithmeticBuilder[C], motive: Int,
    equality: Int, evidence: Int) -> Int:
    return _node(p, tk.PT_SUBST, motive, equality, evidence)


def _sym[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, x, a)), equality, _refl(p, a))


def _trans[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int,
    c: Int, ab: Int, bc: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, a, x)), bc, ab)


def _cong[C: Int](mut p: _ArithmeticBuilder[C], function: Int,
    a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    var first = _app(p, function, a)
    return _subst(p, _lam(p, x, _eq(p, first, _app(p, function, x))), equality,
        _refl(p, first))


def _cong_left[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int,
    equality: Int, suffix: Int) -> Int:
    var x = _var(p, _nat(p))
    return _cong(p, _lam(p, x, _plus(p, x, suffix)), a, b, equality)


def _cong_right[C: Int](mut p: _ArithmeticBuilder[C], prefix: Int,
    a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _cong(p, _lam(p, x, _plus(p, prefix, x)), a, b, equality)


def _nat_induction[C: Int](mut p: _ArithmeticBuilder[C], motive: Int,
    base: Int, step: Int, target: Int) -> Int:
    return _app(p, _app3(p, _node(p, tk.PT_ELIMINATOR, value=1),
        motive, base, step), target)


def _rec[C: Int](mut p: _ArithmeticBuilder[C], major: Int, base: Int,
    step: Int) -> Int:
    return _node(p, tk.PT_NAT_REC, major, base, step)


def _truth[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var proposition = _var(p, _prop(p))
    var evidence = _var(p, proposition)
    return _pi(p, proposition, _pi(p, evidence, proposition))


def _unit[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var proposition = _var(p, _prop(p))
    var evidence = _var(p, proposition)
    return _lam(p, proposition, _lam(p, evidence, evidence))


def _false[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var proposition = _var(p, _prop(p))
    return _pi(p, proposition, proposition)


def _absurd[C: Int](mut p: _ArithmeticBuilder[C], contradiction: Int,
    proposition: Int) -> Int:
    return _app(p, contradiction, proposition)


def _and_type[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int) -> Int:
    var proposition = _var(p, _prop(p))
    var first = _var(p, a)
    var second = _var(p, b)
    var continuation = _var(p, _pi(p, first, _pi(p, second, proposition)))
    return _pi(p, proposition, _pi(p, continuation, proposition))


def _and_intro[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int,
    ha: Int, hb: Int) -> Int:
    var proposition = _var(p, _prop(p))
    var first = _var(p, a)
    var second = _var(p, b)
    var continuation = _var(p, _pi(p, first, _pi(p, second, proposition)))
    return _lam(p, proposition, _lam(p, continuation, _app2(p, continuation, ha, hb)))


def _and_left[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int,
    evidence: Int) -> Int:
    var first = _var(p, a)
    var second = _var(p, b)
    return _app2(p, evidence, a, _lam(p, first, _lam(p, second, first)))


def _and_right[C: Int](mut p: _ArithmeticBuilder[C], a: Int, b: Int,
    evidence: Int) -> Int:
    var first = _var(p, a)
    var second = _var(p, b)
    return _app2(p, evidence, b, _lam(p, first, _lam(p, second, second)))


def _base[C: Int](mut p: tk.ProofTermBundle[C], stride: Int, round: Int) -> Int:
    return _app2(p, _named(p, 5), stride, round)


def _base[C: Int](mut p: _ArithmeticBuilder[C], stride: Int, round: Int) -> Int:
    return _app2(p, _named(p, 5), stride, round)
