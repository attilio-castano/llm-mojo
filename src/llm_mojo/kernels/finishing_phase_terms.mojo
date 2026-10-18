"""Untrusted proof terms for independent private-record population steps.

The phase has three separate Nat-word stores and one pending/ready store.
Filling a slot writes its three symbolic words and ready=1. The certificates
quantify over every store, slot and payload; finite Nat constructor cases in
the proofs do not enumerate computation shapes or payload values. All rules
are ordinary checked Nat induction, lambda/application and equality elimination.
Native slot decoding, correspondence with Mojo's private records, and GPU
execution remain separate interpretation obligations.
"""
import llm_mojo.kernels.proof_term_syntax as tk

comptime PHASE_PROOF_CAPACITY = 4096
comptime PHASE_FALSE = 0
comptime PHASE_DISCRIMINATOR = 1
comptime PHASE_PREDECESSOR = 2
comptime PHASE_SELECT = 3
comptime PHASE_EQB = 4
comptime PHASE_UPDATE = 5
comptime PHASE_APART = 6
comptime PHASE_DECISION = 7
comptime PHASE_DATA_DEFINITIONS = 8
comptime PHASE_ZERO_SUCC = 8
comptime PHASE_SUCC_INJECTIVE = 9
comptime PHASE_EQB_REFL = 10
comptime PHASE_DECIDE = 11
comptime PHASE_APART_TO_EQB = 12
comptime PHASE_EQB_TO_APART = 13
comptime PHASE_UPDATE_HIT = 14
comptime PHASE_UPDATE_MISS = 15
comptime PHASE_UPDATE_CONGRUENCE = 16
comptime PHASE_UPDATE_SWAP = 17
comptime PHASE_PENDING_PRESERVED = 18
comptime PHASE_READY = 19
comptime PHASE_COMMUTATION = 20
comptime PHASE_ENABLED_AB = 21
comptime PHASE_ENABLED_BA = 22
comptime PHASE_DEFINITIONS = 23


@fieldwise_init
struct PhaseProof[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var left_slot: Int
    var right_slot: Int
    var commutation_root: Int
    var enabled_ab_root: Int
    var enabled_ba_root: Int
    var ready_root: Int
    var frame_root: Int
    var pair_test_root: Int
    var pair_apart_root: Int
    var selected_commutation_root: Int
    var selected_enabled_ab_root: Int
    var selected_enabled_ba_root: Int


@fieldwise_init
struct _Builder[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var nodes: Dict[Tuple[Int, Int, Int, Int, Int], Int]


def _node[C: Int](mut p: _Builder[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    if p.bundle.count < 0 or p.bundle.count >= C:
        p.bundle.count = -1
        return -1
    var key = (kind, left, right, extra, value)
    if kind != tk.PT_VAR and key in p.nodes:
        return p.nodes.get(key, -1)
    var result = p.bundle.count
    p.bundle.nodes[result] = tk.ProofTerm(kind, left, right, extra, value)
    p.bundle.count += 1
    if kind != tk.PT_VAR:
        p.nodes[key] = result
    return result


def _nat[C: Int](mut p: _Builder[C]) -> Int:
    return _node(p, tk.PT_NAT)


def _prop[C: Int](mut p: _Builder[C]) -> Int:
    return _node(p, tk.PT_PROP)


def _var[C: Int](mut p: _Builder[C], declared: Int) -> Int:
    return _node(p, tk.PT_VAR, declared)


def _zero[C: Int](mut p: _Builder[C]) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _succ[C: Int](mut p: _Builder[C], n: Int) -> Int:
    return _app(p, _node(p, tk.PT_CONSTRUCTOR,
        extra=tk.NAT_FAMILY_ID, value=1), n)


def _one[C: Int](mut p: _Builder[C]) -> Int:
    return _succ(p, _zero(p))


def _named[C: Int](mut p: _Builder[C], definition: Int) -> Int:
    return _node(p, tk.PT_DEFINED_REF, value=definition)


def _app[C: Int](mut p: _Builder[C], function: Int, argument: Int) -> Int:
    if function < 0 or argument < 0:
        p.bundle.count = -1
        return -1
    return _node(p, tk.PT_APP, function, argument)


def _apps[C: Int](mut p: _Builder[C], var function: Int, arguments: List[Int]) -> Int:
    for argument in arguments:
        function = _app(p, function, argument)
    return function


def _pi[C: Int](mut p: _Builder[C], x: Int, body: Int) -> Int:
    return _node(p, tk.PT_PI, x, body)


def _lam[C: Int](mut p: _Builder[C], x: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, x, body)


def _pis[C: Int](mut p: _Builder[C], binders: List[Int], var body: Int) -> Int:
    for i in range(len(binders)):
        body = _pi(p, binders[len(binders) - i - 1], body)
    return body


def _lams[C: Int](mut p: _Builder[C], binders: List[Int], var body: Int) -> Int:
    for i in range(len(binders)):
        body = _lam(p, binders[len(binders) - i - 1], body)
    return body


def _store[C: Int](mut p: _Builder[C]) -> Int:
    var x = _var(p, _nat(p))
    return _pi(p, x, _nat(p))


def _eq[C: Int](mut p: _Builder[C], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_EQ, a, b)


def _refl[C: Int](mut p: _Builder[C], a: Int) -> Int:
    return _node(p, tk.PT_REFL, a)


def _subst[C: Int](mut p: _Builder[C], motive: Int,
    equality: Int, evidence: Int) -> Int:
    return _node(p, tk.PT_SUBST, motive, equality, evidence)


def _sym[C: Int](mut p: _Builder[C], a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, x, a)), equality, _refl(p, a))


def _trans[C: Int](mut p: _Builder[C], a: Int, b: Int, c: Int,
    first: Int, second: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, a, x)), second, first)


def _cong[C: Int](mut p: _Builder[C], function: Int, a: Int, b: Int,
    equality: Int) -> Int:
    var x = _var(p, _nat(p))
    var start = _app(p, function, a)
    return _subst(p, _lam(p, x, _eq(p, start, _app(p, function, x))),
        equality, _refl(p, start))


def _succ_cong[C: Int](mut p: _Builder[C], a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _cong(p, _lam(p, x, _succ(p, x)), a, b, equality)


def _induct[C: Int](mut p: _Builder[C], motive: Int, base: Int,
    step: Int, target: Int) -> Int:
    return _apps(p, _node(p, tk.PT_ELIMINATOR, value=tk.NAT_FAMILY_ID),
        [motive, base, step, target])


def _false[C: Int](mut p: _Builder[C]) -> Int:
    return _named(p, PHASE_FALSE)


def _apart[C: Int](mut p: _Builder[C], a: Int, b: Int) -> Int:
    return _apps(p, _named(p, PHASE_APART), [a, b])


def _eqb[C: Int](mut p: _Builder[C], a: Int, b: Int) -> Int:
    return _apps(p, _named(p, PHASE_EQB), [a, b])


def _select[C: Int](mut p: _Builder[C], test: Int, yes: Int, no: Int) -> Int:
    return _apps(p, _named(p, PHASE_SELECT), [test, yes, no])


def _update[C: Int](mut p: _Builder[C], state: Int, key: Int, value: Int) -> Int:
    return _apps(p, _named(p, PHASE_UPDATE), [state, key, value])


def _at[C: Int](mut p: _Builder[C], state: Int, key: Int,
    value: Int, observation: Int) -> Int:
    return _app(p, _update(p, state, key, value), observation)


def _decision[C: Int](mut p: _Builder[C], a: Int, b: Int) -> Int:
    return _apps(p, _named(p, PHASE_DECISION), [a, b])


def _apart_sym[C: Int](mut p: _Builder[C], a: Int, b: Int, apart: Int) -> Int:
    var equality = _var(p, _eq(p, b, a))
    return _lam(p, equality, _app(p, apart, _sym(p, b, a, equality)))


def _declare[C: Int](mut p: _Builder[C], expected: Int, body: Int, index: Int):
    if tk.pt_declare_definition(p.bundle, expected, body) != index:
        p.bundle.count = -1


def _data[C: Int](mut p: _Builder[C]):
    # 0 False := forall P:Prop, P.
    var proposition = _var(p, _prop(p))
    _declare(p, _prop(p), _pi(p, proposition, proposition), PHASE_FALSE)
    # 1 D(0)=Eq(0,0), D(S n)=False. Its carrier is Prop : Type.
    var n = _var(p, _nat(p))
    var predecessor = _var(p, _nat(p))
    var previous = _var(p, _prop(p))
    var body = _node(p, tk.PT_NAT_REC, n, _eq(p, _zero(p), _zero(p)),
        _lams(p, [predecessor, previous], _false(p)))
    _declare(p, _pi(p, n, _prop(p)), _lam(p, n, body), PHASE_DISCRIMINATOR)
    # 2 Pred(0)=0, Pred(S n)=n.
    n = _var(p, _nat(p))
    predecessor = _var(p, _nat(p))
    previous = _var(p, _nat(p))
    body = _node(p, tk.PT_NAT_REC, n, _zero(p),
        _lams(p, [predecessor, previous], predecessor))
    _declare(p, _pi(p, n, _nat(p)), _lam(p, n, body), PHASE_PREDECESSOR)
    # 3 Select(0,y,n)=y, Select(S k,y,n)=n.
    n = _var(p, _nat(p))
    var yes = _var(p, _nat(p))
    var no = _var(p, _nat(p))
    predecessor = _var(p, _nat(p))
    previous = _var(p, _nat(p))
    body = _node(p, tk.PT_NAT_REC, n, yes,
        _lams(p, [predecessor, previous], no))
    _declare(p, _pis(p, [n, yes, no], _nat(p)),
        _lams(p, [n, yes, no], body), PHASE_SELECT)
    # 4 EqB: 0 for equal Nats, 1 otherwise. The outer recursor returns Nat->Nat.
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    predecessor = _var(p, _nat(p))
    previous = _var(p, _nat(p))
    var base = _lam(p, b, _node(p, tk.PT_NAT_REC, b, _zero(p),
        _lams(p, [predecessor, previous], _one(p))))
    var ap = _var(p, _nat(p))
    var ih = _var(p, _store(p))
    b = _var(p, _nat(p))
    var bp = _var(p, _nat(p))
    previous = _var(p, _nat(p))
    var step = _lams(p, [ap, ih, b], _node(p, tk.PT_NAT_REC, b, _one(p),
        _lams(p, [bp, previous], _app(p, ih, bp))))
    body = _node(p, tk.PT_NAT_REC, a, base, step)
    _declare(p, _pi(p, a, _store(p)), _lam(p, a, body), PHASE_EQB)
    # 5 Update(S,k,v)(x) := Select(EqB(k,x),v,S(x)).
    var state = _var(p, _store(p))
    var key = _var(p, _nat(p))
    var value = _var(p, _nat(p))
    var x = _var(p, _nat(p))
    body = _select(p, _eqb(p, key, x), value, _app(p, state, x))
    _declare(p, _pis(p, [state, key, value, x], _nat(p)),
        _lams(p, [state, key, value, x], body), PHASE_UPDATE)
    # 6 Apart(a,b) := Eq(a,b) -> False.
    a = _var(p, _nat(p))
    b = _var(p, _nat(p))
    var equal = _var(p, _eq(p, a, b))
    _declare(p, _pis(p, [a, b], _prop(p)),
        _lams(p, [a, b], _pi(p, equal, _false(p))), PHASE_APART)
    # 7 DecEq(a,b) := forall R, (Eq(a,b)->R)->(Apart(a,b)->R)->R.
    a = _var(p, _nat(p))
    b = _var(p, _nat(p))
    proposition = _var(p, _prop(p))
    equal = _var(p, _eq(p, a, b))
    var unequal = _var(p, _apart(p, a, b))
    yes = _var(p, _pi(p, equal, proposition))
    no = _var(p, _pi(p, unequal, proposition))
    _declare(p, _pis(p, [a, b], _prop(p)),
        _lams(p, [a, b], _pis(p, [proposition, yes, no], proposition)), PHASE_DECISION)


def _no_confusion[C: Int](mut p: _Builder[C]):
    var n = _var(p, _nat(p))
    var equal = _var(p, _eq(p, _zero(p), _succ(p, n)))
    var body = _subst(p, _named(p, PHASE_DISCRIMINATOR), equal, _refl(p, _zero(p)))
    _declare(p, _pis(p, [n, equal], _false(p)),
        _lams(p, [n, equal], body), PHASE_ZERO_SUCC)
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    equal = _var(p, _eq(p, _succ(p, a), _succ(p, b)))
    body = _cong(p, _named(p, PHASE_PREDECESSOR), _succ(p, a), _succ(p, b), equal)
    _declare(p, _pis(p, [a, b, equal], _eq(p, a, b)),
        _lams(p, [a, b, equal], body), PHASE_SUCC_INJECTIVE)


def _eqb_refl[C: Int](mut p: _Builder[C]):
    var a = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var motive = _lam(p, n, _eq(p, _eqb(p, n, n), _zero(p)))
    n = _var(p, _nat(p))
    var ih = _var(p, _eq(p, _eqb(p, n, n), _zero(p)))
    var body = _induct(p, motive, _refl(p, _zero(p)), _lams(p, [n, ih], ih), a)
    _declare(p, _pi(p, a, _eq(p, _eqb(p, a, a), _zero(p))),
        _lam(p, a, body), PHASE_EQB_REFL)


def _decision_yes[C: Int](mut p: _Builder[C], a: Int, b: Int, equality: Int) -> Int:
    var result = _var(p, _prop(p))
    var eq_argument = _var(p, _eq(p, a, b))
    var apart_argument = _var(p, _apart(p, a, b))
    var yes = _var(p, _pi(p, eq_argument, result))
    var no = _var(p, _pi(p, apart_argument, result))
    return _lams(p, [result, yes, no], _app(p, yes, equality))


def _decision_no[C: Int](mut p: _Builder[C], a: Int, b: Int, apart: Int) -> Int:
    var result = _var(p, _prop(p))
    var eq_argument = _var(p, _eq(p, a, b))
    var apart_argument = _var(p, _apart(p, a, b))
    var yes = _var(p, _pi(p, eq_argument, result))
    var no = _var(p, _pi(p, apart_argument, result))
    return _lams(p, [result, yes, no], _app(p, no, apart))


def _decide[C: Int](mut p: _Builder[C]):
    var a = _var(p, _nat(p))
    var an = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var outer_motive = _lam(p, an, _pi(p, b, _decision(p, an, b)))
    b = _var(p, _nat(p))
    var bn = _var(p, _nat(p))
    var inner_motive = _lam(p, bn, _decision(p, _zero(p), bn))
    bn = _var(p, _nat(p))
    var inner_ih = _var(p, _decision(p, _zero(p), bn))
    var zero_succ_equal = _var(p, _eq(p, _zero(p), _succ(p, bn)))
    var apart = _lam(p, zero_succ_equal,
        _apps(p, _named(p, PHASE_ZERO_SUCC), [bn, zero_succ_equal]))
    var inner_step = _lams(p, [bn, inner_ih],
        _decision_no(p, _zero(p), _succ(p, bn), apart))
    var base = _lam(p, b, _induct(p, inner_motive,
        _decision_yes(p, _zero(p), _zero(p), _refl(p, _zero(p))), inner_step, b))
    an = _var(p, _nat(p))
    b = _var(p, _nat(p))
    var outer_ih = _var(p, _pi(p, b, _decision(p, an, b)))
    b = _var(p, _nat(p))
    bn = _var(p, _nat(p))
    inner_motive = _lam(p, bn, _decision(p, _succ(p, an), bn))
    var succ_zero_equal = _var(p, _eq(p, _succ(p, an), _zero(p)))
    apart = _lam(p, succ_zero_equal, _apps(p, _named(p, PHASE_ZERO_SUCC),
        [an, _sym(p, _succ(p, an), _zero(p), succ_zero_equal)]))
    var inner_base = _decision_no(p, _succ(p, an), _zero(p), apart)
    bn = _var(p, _nat(p))
    inner_ih = _var(p, _decision(p, _succ(p, an), bn))
    var result = _var(p, _prop(p))
    var ss_equal = _var(p, _eq(p, _succ(p, an), _succ(p, bn)))
    var ss_apart = _var(p, _apart(p, _succ(p, an), _succ(p, bn)))
    var yes = _var(p, _pi(p, ss_equal, result))
    var no = _var(p, _pi(p, ss_apart, result))
    var eq_argument = _var(p, _eq(p, an, bn))
    var apart_argument = _var(p, _apart(p, an, bn))
    var yes_branch = _lam(p, eq_argument,
        _app(p, yes, _succ_cong(p, an, bn, eq_argument)))
    ss_equal = _var(p, _eq(p, _succ(p, an), _succ(p, bn)))
    var no_branch = _lam(p, apart_argument, _app(p, no,
        _lam(p, ss_equal, _app(p, apart_argument,
            _apps(p, _named(p, PHASE_SUCC_INJECTIVE), [an, bn, ss_equal])))))
    inner_step = _lams(p, [bn, inner_ih, result, yes, no],
        _apps(p, outer_ih, [bn, result, yes_branch, no_branch]))
    var outer_step = _lams(p, [an, outer_ih, b],
        _induct(p, inner_motive, inner_base, inner_step, b))
    b = _var(p, _nat(p))
    _declare(p, _pis(p, [a, b], _decision(p, a, b)),
        _lam(p, a, _induct(p, outer_motive, base, outer_step, a)), PHASE_DECIDE)


def _apart_to_eqb[C: Int](mut p: _Builder[C]):
    var a = _var(p, _nat(p))
    var an = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var apart = _var(p, _apart(p, an, b))
    var motive = _lam(p, an, _pis(p, [b, apart], _eq(p, _eqb(p, an, b), _one(p))))
    b = _var(p, _nat(p))
    var bn = _var(p, _nat(p))
    apart = _var(p, _apart(p, _zero(p), bn))
    var inner_motive = _lam(p, bn, _pi(p, apart,
        _eq(p, _eqb(p, _zero(p), bn), _one(p))))
    apart = _var(p, _apart(p, _zero(p), _zero(p)))
    var impossible = _app(p, apart, _refl(p, _zero(p)))
    var base = _lam(p, apart, _app(p, impossible, _eq(p, _zero(p), _one(p))))
    bn = _var(p, _nat(p))
    apart = _var(p, _apart(p, _zero(p), bn))
    var inner_ih = _var(p, _pi(p, apart, _eq(p, _eqb(p, _zero(p), bn), _one(p))))
    apart = _var(p, _apart(p, _zero(p), _succ(p, bn)))
    var inner_step = _lams(p, [bn, inner_ih, apart], _refl(p, _one(p)))
    var outer_base = _lam(p, b, _induct(p, inner_motive, base, inner_step, b))
    an = _var(p, _nat(p))
    b = _var(p, _nat(p))
    apart = _var(p, _apart(p, an, b))
    var outer_ih = _var(p, _pis(p, [b, apart], _eq(p, _eqb(p, an, b), _one(p))))
    b = _var(p, _nat(p))
    bn = _var(p, _nat(p))
    apart = _var(p, _apart(p, _succ(p, an), bn))
    inner_motive = _lam(p, bn, _pi(p, apart,
        _eq(p, _eqb(p, _succ(p, an), bn), _one(p))))
    apart = _var(p, _apart(p, _succ(p, an), _zero(p)))
    base = _lam(p, apart, _refl(p, _one(p)))
    bn = _var(p, _nat(p))
    apart = _var(p, _apart(p, _succ(p, an), bn))
    inner_ih = _var(p, _pi(p, apart, _eq(p, _eqb(p, _succ(p, an), bn), _one(p))))
    apart = _var(p, _apart(p, _succ(p, an), _succ(p, bn)))
    var equal = _var(p, _eq(p, an, bn))
    var predecessor_apart = _lam(p, equal,
        _app(p, apart, _succ_cong(p, an, bn, equal)))
    inner_step = _lams(p, [bn, inner_ih, apart],
        _apps(p, outer_ih, [bn, predecessor_apart]))
    var outer_step = _lams(p, [an, outer_ih, b],
        _induct(p, inner_motive, base, inner_step, b))
    b = _var(p, _nat(p))
    apart = _var(p, _apart(p, a, b))
    _declare(p, _pis(p, [a, b, apart], _eq(p, _eqb(p, a, b), _one(p))),
        _lam(p, a, _induct(p, motive, outer_base, outer_step, a)), PHASE_APART_TO_EQB)


def _eqb_to_apart[C: Int](mut p: _Builder[C]):
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var test = _var(p, _eq(p, _eqb(p, a, b), _one(p)))
    var equal = _var(p, _eq(p, a, b))
    var x = _var(p, _nat(p))
    var same_test = _subst(p, _lam(p, x, _eq(p, _eqb(p, a, x), _one(p))),
        _sym(p, a, b, equal), test)
    var reflexive_test = _app(p, _named(p, PHASE_EQB_REFL), a)
    var zero_one = _trans(p, _zero(p), _eqb(p, a, a), _one(p),
        _sym(p, _eqb(p, a, a), _zero(p), reflexive_test), same_test)
    var body = _apps(p, _named(p, PHASE_ZERO_SUCC), [_zero(p), zero_one])
    _declare(p, _pis(p, [a, b, test], _apart(p, a, b)),
        _lams(p, [a, b, test, equal], body), PHASE_EQB_TO_APART)


def _updates[C: Int](mut p: _Builder[C]):
    var state = _var(p, _store(p))
    var key = _var(p, _nat(p))
    var value = _var(p, _nat(p))
    var test = _eqb(p, key, key)
    var x = _var(p, _nat(p))
    var motive = _lam(p, x, _eq(p,
        _select(p, x, value, _app(p, state, key)), value))
    var body = _subst(p, motive, _sym(p, test, _zero(p),
        _app(p, _named(p, PHASE_EQB_REFL), key)), _refl(p, value))
    _declare(p, _pis(p, [state, key, value],
        _eq(p, _at(p, state, key, value, key), value)),
        _lams(p, [state, key, value], body), PHASE_UPDATE_HIT)
    state = _var(p, _store(p))
    key = _var(p, _nat(p))
    value = _var(p, _nat(p))
    var observation = _var(p, _nat(p))
    var apart = _var(p, _apart(p, key, observation))
    test = _eqb(p, key, observation)
    x = _var(p, _nat(p))
    var old = _app(p, state, observation)
    motive = _lam(p, x, _eq(p, _select(p, x, value, old), old))
    body = _subst(p, motive, _sym(p, test, _one(p),
        _apps(p, _named(p, PHASE_APART_TO_EQB), [key, observation, apart])), _refl(p, old))
    _declare(p, _pis(p, [state, key, value, observation, apart],
        _eq(p, _at(p, state, key, value, observation), old)),
        _lams(p, [state, key, value, observation, apart], body), PHASE_UPDATE_MISS)
    state = _var(p, _store(p))
    var other = _var(p, _store(p))
    key = _var(p, _nat(p))
    value = _var(p, _nat(p))
    observation = _var(p, _nat(p))
    var equal = _var(p, _eq(p, _app(p, state, observation), _app(p, other, observation)))
    x = _var(p, _nat(p))
    var left = _at(p, state, key, value, observation)
    motive = _lam(p, x, _eq(p, left, _select(p, _eqb(p, key, observation), value, x)))
    body = _subst(p, motive, equal, _refl(p, left))
    _declare(p, _pis(p, [state, other, key, value, observation, equal],
        _eq(p, left, _at(p, other, key, value, observation))),
        _lams(p, [state, other, key, value, observation, equal], body), PHASE_UPDATE_CONGRUENCE)


def _swap_side[C: Int](mut p: _Builder[C], state: Int, a: Int, b: Int,
    av: Int, bv: Int, x: Int) -> Int:
    return _at(p, _update(p, state, a, av), b, bv, x)


def _swap[C: Int](mut p: _Builder[C]):
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var apart = _var(p, _apart(p, a, b))
    var state = _var(p, _store(p))
    var av = _var(p, _nat(p))
    var bv = _var(p, _nat(p))
    var x = _var(p, _nat(p))
    var left = _swap_side(p, state, a, b, av, bv, x)
    var right = _swap_side(p, state, b, a, bv, av, x)
    var result = _eq(p, left, right)
    var equal = _var(p, _eq(p, a, x))
    var left_at_a = _swap_side(p, state, a, b, av, bv, a)
    var right_at_a = _swap_side(p, state, b, a, bv, av, a)
    var first = _apps(p, _named(p, PHASE_UPDATE_MISS),
        [_update(p, state, a, av), b, bv, a, _apart_sym(p, a, b, apart)])
    var second = _apps(p, _named(p, PHASE_UPDATE_HIT), [state, a, av])
    var left_value = _trans(p, left_at_a, _at(p, state, a, av, a), av, first, second)
    var right_value = _apps(p, _named(p, PHASE_UPDATE_HIT),
        [_update(p, state, b, bv), a, av])
    var at_a = _trans(p, left_at_a, av, right_at_a, left_value,
        _sym(p, right_at_a, av, right_value))
    var observation = _var(p, _nat(p))
    var motive = _lam(p, observation, _eq(p,
        _swap_side(p, state, a, b, av, bv, observation),
        _swap_side(p, state, b, a, bv, av, observation)))
    var yes = _lam(p, equal, _subst(p, motive, equal, at_a))
    var misses = _var(p, _apart(p, a, x))
    var base_equal = _apps(p, _named(p, PHASE_UPDATE_MISS), [state, a, av, x, misses])
    first = _apps(p, _named(p, PHASE_UPDATE_CONGRUENCE),
        [_update(p, state, a, av), state, b, bv, x, base_equal])
    second = _apps(p, _named(p, PHASE_UPDATE_MISS),
        [_update(p, state, b, bv), a, av, x, misses])
    var middle = _at(p, state, b, bv, x)
    var no = _lam(p, misses, _trans(p, left, middle, right,
        first, _sym(p, right, middle, second)))
    var body = _apps(p, _named(p, PHASE_DECIDE), [a, x, result, yes, no])
    _declare(p, _pis(p, [a, b, apart, state, av, bv, x], result),
        _lams(p, [a, b, apart, state, av, bv, x], body), PHASE_UPDATE_SWAP)


def _pending_and_ready[C: Int](mut p: _Builder[C]):
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var apart = _var(p, _apart(p, a, b))
    var state = _var(p, _store(p))
    var value = _var(p, _nat(p))
    var pending = _var(p, _eq(p, _app(p, state, b), _zero(p)))
    var filled = _at(p, state, a, value, b)
    var miss = _apps(p, _named(p, PHASE_UPDATE_MISS), [state, a, value, b, apart])
    var body = _trans(p, filled, _app(p, state, b), _zero(p), miss, pending)
    _declare(p, _pis(p, [a, b, apart, state, value, pending], _eq(p, filled, _zero(p))),
        _lams(p, [a, b, apart, state, value, pending], body), PHASE_PENDING_PRESERVED)
    state = _var(p, _store(p))
    a = _var(p, _nat(p))
    body = _apps(p, _named(p, PHASE_UPDATE_HIT), [state, a, _one(p)])
    _declare(p, _pis(p, [state, a], _eq(p, _at(p, state, a, _one(p), a), _one(p))),
        _lams(p, [state, a], body), PHASE_READY)


def _and4[C: Int](mut p: _Builder[C], propositions: List[Int]) -> Int:
    var result = _var(p, _prop(p))
    var premises = List[Int]()
    for proposition in propositions:
        premises.append(_var(p, proposition))
    var consumer = _var(p, _pis(p, premises, result))
    return _pis(p, [result, consumer], result)


def _and4_intro[C: Int](mut p: _Builder[C], propositions: List[Int],
    proofs: List[Int]) -> Int:
    var result = _var(p, _prop(p))
    var premises = List[Int]()
    for proposition in propositions:
        premises.append(_var(p, proposition))
    var consumer = _var(p, _pis(p, premises, result))
    return _lams(p, [result, consumer], _apps(p, consumer, proofs))


def _phase_roots[C: Int](mut p: _Builder[C]):
    # Fill A/B writes r,t,i to three stores and 1 to the separate ready store.
    # The universal result is a Church conjunction of four pointwise equations.
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var apart = _var(p, _apart(p, a, b))
    var states = List[Int]()
    for _ in range(4):
        states.append(_var(p, _store(p)))
    var values_a = List[Int]()
    var values_b = List[Int]()
    for _ in range(3):
        values_a.append(_var(p, _nat(p)))
    for _ in range(3):
        values_b.append(_var(p, _nat(p)))
    var x = _var(p, _nat(p))
    var propositions = List[Int]()
    var proofs = List[Int]()
    for component in range(4):
        var av = values_a[component] if component < 3 else _one(p)
        var bv = values_b[component] if component < 3 else _one(p)
        propositions.append(_eq(p,
            _swap_side(p, states[component], a, b, av, bv, x),
            _swap_side(p, states[component], b, a, bv, av, x)))
        proofs.append(_apps(p, _named(p, PHASE_UPDATE_SWAP),
            [a, b, apart, states[component], av, bv, x]))
    var binders: List[Int] = [a, b, apart]
    for state in states:
        binders.append(state)
    for value in values_a:
        binders.append(value)
    for value in values_b:
        binders.append(value)
    binders.append(x)
    _declare(p, _pis(p, binders, _and4(p, propositions)),
        _lams(p, binders, _and4_intro(p, propositions, proofs)), PHASE_COMMUTATION)
    # The two checked directions keep both initial enabledness assumptions.
    for direction in range(2):
        a = _var(p, _nat(p))
        b = _var(p, _nat(p))
        apart = _var(p, _apart(p, a, b))
        var state = _var(p, _store(p))
        var pa = _var(p, _eq(p, _app(p, state, a), _zero(p)))
        var pb = _var(p, _eq(p, _app(p, state, b), _zero(p)))
        var filled: Int
        var body: Int
        if direction == 0:
            filled = _at(p, state, a, _one(p), b)
            body = _apps(p, _named(p, PHASE_PENDING_PRESERVED),
                [a, b, apart, state, _one(p), pb])
        else:
            filled = _at(p, state, b, _one(p), a)
            body = _apps(p, _named(p, PHASE_PENDING_PRESERVED),
                [b, a, _apart_sym(p, a, b, apart), state, _one(p), pa])
        _declare(p, _pis(p, [a, b, apart, state, pa, pb], _eq(p, filled, _zero(p))),
            _lams(p, [a, b, apart, state, pa, pb], body), PHASE_ENABLED_AB + direction)


def make_finishing_phase_proof[capacity: Int = PHASE_PROOF_CAPACITY](
    left_slot: Int, right_slot: Int) -> PhaseProof[capacity]:
    var bundle = tk.empty_proof_terms[capacity]()
    var nodes = Dict[Tuple[Int, Int, Int, Int, Int], Int]()
    for j in range(bundle.count):
        var n = bundle.nodes[j]
        if n.kind != tk.PT_VAR:
            nodes[(n.kind, n.left, n.right, n.extra, n.value)] = j
    var p = _Builder[capacity](bundle^, nodes^)
    _data(p)
    _no_confusion(p)
    _eqb_refl(p)
    _decide(p)
    _apart_to_eqb(p)
    _eqb_to_apart(p)
    _updates(p)
    _swap(p)
    _pending_and_ready(p)
    _phase_roots(p)
    var commutation = _named(p, PHASE_COMMUTATION)
    var enabled_ab = _named(p, PHASE_ENABLED_AB)
    var enabled_ba = _named(p, PHASE_ENABLED_BA)
    var ready = _named(p, PHASE_READY)
    var frame = _named(p, PHASE_UPDATE_MISS)
    # Literal nodes are abbreviations only. Refl(1) must check against the
    # independently requested Eq(EqB(left,right),1) by ordinary conversion.
    # Equal or negative pairs do not gain an Apart premise from this proposal.
    var left = _node(p, tk.PT_LITERAL, value=left_slot)
    var right = _node(p, tk.PT_LITERAL, value=right_slot)
    var test = _refl(p, _one(p))
    var apart = _apps(p, _named(p, PHASE_EQB_TO_APART), [left, right, test])
    var selected = _apps(p, commutation, [left, right, apart])
    var selected_ab = _apps(p, enabled_ab, [left, right, apart])
    var selected_ba = _apps(p, enabled_ba, [left, right, apart])
    p.bundle.root = selected
    return PhaseProof[capacity](p.bundle.copy(), left_slot, right_slot,
        commutation, enabled_ab, enabled_ba, ready, frame, test, apart,
        selected, selected_ab, selected_ba)
