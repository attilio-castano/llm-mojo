"""Structural Nat maximum data and ordinary Nat-induction proof terms.

This supports a total Nat score/Boolean-flag join. It does not refine the
native UInt32 rank/token merge or GPU execution. No native comparison,
arithmetic reflection, order axiom or special maximum rule is used.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph

comptime RECORD_MAX = 55
comptime RECORD_MAX_RIGHT_ZERO = 56
comptime RECORD_MAX_LEFT_ZERO = 57
comptime RECORD_MAX_COMMUTATIVE = 58
comptime RECORD_MAX_ASSOCIATIVE = 59
comptime RECORD_MAX_IDEMPOTENT = 60
comptime RECORD_MAX_DEFINITIONS = 61


def record_max[C: Int](mut p: ph._Builder[C], a: Int, b: Int) -> Int:
    return ph._apps(p, ph._named(p, RECORD_MAX), [a, b])


def _maximum_data[C: Int](mut p: ph._Builder[C]):
    # The outer recursor returns Nat -> Nat. Its step inspects the second
    # argument: M(0,b)=b; M(S a,0)=S a; M(S a,S b)=S(M(a,b)).
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var base = ph._lam(p, b, b)
    var predecessor = ph._var(p, ph._nat(p))
    var previous = ph._var(p, ph._store(p))
    b = ph._var(p, ph._nat(p))
    var right_predecessor = ph._var(p, ph._nat(p))
    var ignored = ph._var(p, ph._nat(p))
    var right_step = ph._lams(p, [right_predecessor, ignored],
        ph._succ(p, ph._app(p, previous, right_predecessor)))
    var step = ph._lams(p, [predecessor, previous, b],
        ph._node(p, tk.PT_NAT_REC, b, ph._succ(p, predecessor), right_step))
    ph._declare(p, ph._pi(p, a, ph._store(p)),
        ph._lam(p, a, ph._node(p, tk.PT_NAT_REC, a, base, step)), RECORD_MAX)


def _right_zero[C: Int](mut p: ph._Builder[C]):
    var a = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, a, ph._eq(p, record_max(p, a, ph._zero(p)), a))
    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    # The successor/zero computation is definitional; its IH is unnecessary.
    var step = ph._lams(p, [predecessor, ih], ph._refl(p, ph._succ(p, predecessor)))
    a = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, a, ph._app(p, motive, a)),
        ph._lam(p, a, ph._induct(p, motive, ph._refl(p, ph._zero(p)), step, a)),
        RECORD_MAX_RIGHT_ZERO)


def _left_zero[C: Int](mut p: ph._Builder[C]):
    var b = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, b, ph._eq(p, record_max(p, ph._zero(p), b), b)),
        ph._lam(p, b, ph._refl(p, b)), RECORD_MAX_LEFT_ZERO)


def _commutative[C: Int](mut p: ph._Builder[C]):
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var outer_motive = ph._lam(p, a, ph._pi(p, b,
        ph._eq(p, record_max(p, a, b), record_max(p, b, a))))
    b = ph._var(p, ph._nat(p))
    var zero_right = ph._app(p, ph._named(p, RECORD_MAX_RIGHT_ZERO), b)
    var base = ph._lam(p, b,
        ph._sym(p, record_max(p, b, ph._zero(p)), b, zero_right))
    var predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, predecessor))
    b = ph._var(p, ph._nat(p))
    var successor = ph._succ(p, predecessor)
    var inner_motive = ph._lam(p, b, ph._eq(p,
        record_max(p, successor, b), record_max(p, b, successor)))
    var right_predecessor = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, right_predecessor))
    var smaller_left = record_max(p, predecessor, right_predecessor)
    var smaller_right = record_max(p, right_predecessor, predecessor)
    var smaller_equal = ph._app(p, outer_ih, right_predecessor)
    var inner_step = ph._lams(p, [right_predecessor, inner_ih],
        ph._succ_cong(p, smaller_left, smaller_right, smaller_equal))
    var outer_step = ph._lams(p, [predecessor, outer_ih, b],
        ph._induct(p, inner_motive, ph._refl(p, successor), inner_step, b))
    a = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, a, ph._app(p, outer_motive, a)),
        ph._lam(p, a, ph._induct(p, outer_motive, base, outer_step, a)),
        RECORD_MAX_COMMUTATIVE)


def _associative[C: Int](mut p: ph._Builder[C]):
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var c = ph._var(p, ph._nat(p))
    var outer_motive = ph._lam(p, a, ph._pis(p, [b, c], ph._eq(p,
        record_max(p, record_max(p, a, b), c),
        record_max(p, a, record_max(p, b, c)))))
    b = ph._var(p, ph._nat(p))
    c = ph._var(p, ph._nat(p))
    # When the first argument is zero, both sides compute to M(b,c).
    var base = ph._lams(p, [b, c], ph._refl(p, record_max(p, b, c)))

    var first_predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, first_predecessor))
    var first_successor = ph._succ(p, first_predecessor)
    b = ph._var(p, ph._nat(p))
    c = ph._var(p, ph._nat(p))
    var middle_motive = ph._lam(p, b, ph._pi(p, c, ph._eq(p,
        record_max(p, record_max(p, first_successor, b), c),
        record_max(p, first_successor, record_max(p, b, c)))))
    c = ph._var(p, ph._nat(p))
    # With middle argument zero, both sides compute to M(S a,c).
    var middle_base = ph._lam(p, c,
        ph._refl(p, record_max(p, first_successor, c)))

    var second_predecessor = ph._var(p, ph._nat(p))
    var middle_ih = ph._var(p, ph._app(p, middle_motive, second_predecessor))
    var second_successor = ph._succ(p, second_predecessor)
    c = ph._var(p, ph._nat(p))
    var inner_motive = ph._lam(p, c, ph._eq(p,
        record_max(p, record_max(p, first_successor, second_successor), c),
        record_max(p, first_successor, record_max(p, second_successor, c))))
    var pair_maximum = record_max(p, first_predecessor, second_predecessor)
    # With final argument zero, both sides compute to S(M(a,b)).
    var inner_base = ph._refl(p, ph._succ(p, pair_maximum))
    var third_predecessor = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, third_predecessor))
    var smaller_left = record_max(p, pair_maximum, third_predecessor)
    var smaller_right = record_max(p, first_predecessor,
        record_max(p, second_predecessor, third_predecessor))
    var smaller_equal = ph._apps(p, outer_ih, [second_predecessor, third_predecessor])
    var inner_step = ph._lams(p, [third_predecessor, inner_ih],
        ph._succ_cong(p, smaller_left, smaller_right, smaller_equal))
    var middle_step = ph._lams(p, [second_predecessor, middle_ih, c],
        ph._induct(p, inner_motive, inner_base, inner_step, c))
    var outer_step = ph._lams(p, [first_predecessor, outer_ih, b],
        ph._induct(p, middle_motive, middle_base, middle_step, b))
    a = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, a, ph._app(p, outer_motive, a)),
        ph._lam(p, a, ph._induct(p, outer_motive, base, outer_step, a)),
        RECORD_MAX_ASSOCIATIVE)


def _idempotent[C: Int](mut p: ph._Builder[C]):
    var a = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, a, ph._eq(p, record_max(p, a, a), a))
    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    var step = ph._lams(p, [predecessor, ih],
        ph._succ_cong(p, record_max(p, predecessor, predecessor), predecessor, ih))
    a = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, a, ph._app(p, motive, a)),
        ph._lam(p, a, ph._induct(p, motive, ph._refl(p, ph._zero(p)), step, a)),
        RECORD_MAX_IDEMPOTENT)


def append_record_max_terms[C: Int](mut p: ph._Builder[C]):
    if p.bundle.definition_count != RECORD_MAX:
        p.bundle.count = -1
        return
    _maximum_data(p)
    _right_zero(p)
    _left_zero(p)
    _commutative(p)
    _associative(p)
    _idempotent(p)
