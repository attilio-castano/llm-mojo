"""Untrusted proof terms for base linearity and finite finishing execution.

The small dependent typechecker validates these lambda, application and
substitution terms independently. Nat induction uses ordinary applications
of the eliminator derived from checked zero/successor declarations. A proof's storage
capacity bounds its syntax, not the symbolic natural counts or execution
geometry. Done is an explicitly declared indexed inductive predicate; the
generic declaration checker validates its constructor types. Arithmetic
reflection, compact numeral/constructor conversion, immutable export import,
and interpretation of Done as finite driver execution remain
trusted. Separate checks exercise
the definition-derived Prop eliminator and structural count monotonicity;
the signature generator and constructor computation remain generic kernel rules.
"""
import llm_mojo.kernels.proof_term_kernel as tk
from std.collections import Array
from llm_mojo.kernels.finishing_proof_kernel import FinishingProgram

comptime FINISHING_TYPED_PROOF_CAPACITY = 256
comptime FINISHING_DONE_DEFINITION_CAPACITY = 64
comptime FINISHING_DONE_ELIMINATION_CAPACITY = 256


@fieldwise_init
struct FinishingTypedProofs[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var base_root: Int
    var termination_root: Int


def _node[capacity: Int](mut p: tk.ProofTermBundle[capacity], kind: Int,
    left: Int = -1, right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _nat[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    return _node(p, tk.PT_NAT)


def _var[capacity: Int](mut p: tk.ProofTermBundle[capacity], declared: Int) -> Int:
    return _node(p, tk.PT_VAR, declared)


def _number[capacity: Int](mut p: tk.ProofTermBundle[capacity], value: Int) -> Int:
    return _node(p, tk.PT_LITERAL, value=value)


def _plus[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_ADD, a, b)


def _scale[capacity: Int](mut p: tk.ProofTermBundle[capacity], factor: Int, a: Int) -> Int:
    return _node(p, tk.PT_SCALE, a, value=factor)


def _base[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int) -> Int:
    return _node(p, tk.PT_BASE, q)


def _eq[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_EQ, a, b)


def _le[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_LE, a, b)


def _done[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int, n: Int) -> Int:
    return _app(p, _app(p, _node(p, tk.PT_FAMILY, value=0), q), n)


def _pi[capacity: Int](mut p: tk.ProofTermBundle[capacity], binder: Int, result: Int) -> Int:
    return _node(p, tk.PT_PI, binder, result)


def _lam[capacity: Int](mut p: tk.ProofTermBundle[capacity], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, binder, body)


def _app[capacity: Int](mut p: tk.ProofTermBundle[capacity], function: Int, argument: Int) -> Int:
    return _node(p, tk.PT_APP, function, argument)


def _decl[capacity: Int](mut p: tk.ProofTermBundle[capacity], declaration: Int) -> Int:
    # All names, including Base's equations, refer to closed definitions whose
    # actual types the kernel imports and instantiates by ordinary application.
    return _node(p, tk.PT_ARITHMETIC, value=declaration)


def _constructor[capacity: Int](mut p: tk.ProofTermBundle[capacity], constructor: Int) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, value=constructor)


def declare_finishing_done[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Bool:
    """The independently pinned driver specification, not an asserted theorem.

    Done : Nat -> Nat -> Prop
    stop : forall q,N, N <= Base(q) -> Done(q,N)
    step : forall q,N, Base(q) < N -> Done(q+1,N) -> Done(q,N)

    The producer submits this data to generic inductive admission. Finishing
    admission builds it independently and compares the checked environments:
    admissibility alone would permit other, semantically different predicates.
    The body-return and runtime-correspondence assumptions belong to the
    interpretation of this specification, not its constructor typechecking.
    """
    if not tk.pt_declare_family(p, 2):
        return False
    var q = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var stop_guard = _var(p, _le(p, n, _base(p, q)))
    var stopped = _pi(p, q, _pi(p, n,
        _pi(p, stop_guard, _done(p, q, n))))
    if tk.pt_declare_constructor(p, stopped) != 0:
        return False
    var q_step = _var(p, _nat(p))
    var n_step = _var(p, _nat(p))
    var step_guard = _var(p, _node(p, tk.PT_LT, _base(p, q_step), n_step))
    var suffix = _var(p, _done(p, _plus(p, q_step, _number(p, 1)), n_step))
    var stepped = _pi(p, q_step, _pi(p, n_step,
        _pi(p, step_guard, _pi(p, suffix, _done(p, q_step, n_step)))))
    return tk.pt_declare_constructor(p, stepped) == 1


def _subst[capacity: Int](mut p: tk.ProofTermBundle[capacity], motive: Int,
    equality: Int, evidence: Int) -> Int:
    return _node(p, tk.PT_SUBST, motive, equality, evidence)


def _affine[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_AFFINE_EQ, a, b)


def _trans[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int,
    c: Int, ab: Int, bc: Int) -> Int:
    # Eq(a,b), Eq(b,c) -> Eq(a,c), using equality elimination.
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, a, x)), bc, ab)


def _sym[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int,
    equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, x, a)), equality,
        _node(p, tk.PT_REFL, a))


def _add_congruence[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int,
    b: Int, equality: Int, c: Int) -> Int:
    var x = _var(p, _nat(p))
    var start = _plus(p, a, c)
    return _subst(p, _lam(p, x, _eq(p, start, _plus(p, x, c))), equality,
        _node(p, tk.PT_REFL, start))


def _le_right[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _le(p, a, x)), equality, evidence)


def _le_left[capacity: Int](mut p: tk.ProofTermBundle[capacity], b: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _le(p, x, b)), equality, evidence)


def base_linearity_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    """Independent required type: forall q:Nat, Base(q) = S*q."""
    var q = _var(p, _nat(p))
    return _pi(p, q, _eq(p, _base(p, q), _scale(p, program.stride, q)))


def termination_type[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    """Independent required type: forall N:Nat, Done(0,N)."""
    var n = _var(p, _nat(p))
    return _pi(p, n, _done(p, _number(p, 0), n))


def _nat_induction[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    motive: Int, base: Int, step: Int, target: Int) -> Int:
    # Ordinary application of the eliminator derived from checked Nat data.
    # Compact zero/successor arithmetic is interpreted through the pinned codec.
    return _app(p, _app(p, _app(p, _app(p,
        _node(p, tk.PT_ELIMINATOR, value=tk.NAT_FAMILY_ID), motive), base), step), target)


def _base_linearity[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var zero = _number(p, 0)
    var one = _number(p, 1)
    var stride = _number(p, program.stride)

    # The motive is an ordinary function Nat -> Prop.
    var motive_q = _var(p, _nat(p))
    var motive = _lam(p, motive_q,
        _eq(p, _base(p, motive_q), _scale(p, program.stride, motive_q)))
    var base_zero = _decl(p, tk.DEC_BASE_ZERO)
    var initial = _trans(p, _base(p, zero), zero,
        _scale(p, program.stride, zero), base_zero,
        _affine(p, zero, _scale(p, program.stride, zero)))

    # q, ih:Base(q)=S*q |- Base(q+1)=S*(q+1).
    var q = _var(p, _nat(p))
    var successor = _plus(p, q, one)
    var ih = _var(p, _app(p, motive, q))
    var base_step = _app(p, _decl(p, tk.DEC_BASE_STEP), q)
    var middle = _plus(p, _base(p, q), stride)
    var replaced = _plus(p, _scale(p, program.stride, q), stride)
    var congruence = _add_congruence(p, _base(p, q),
        _scale(p, program.stride, q), ih, stride)
    var step = _trans(p, _base(p, successor), middle, replaced, base_step, congruence)
    step = _trans(p, _base(p, successor), replaced,
        _scale(p, program.stride, successor), step,
        _affine(p, replaced, _scale(p, program.stride, successor)))
    var inductive_step = _lam(p, q, _lam(p, ih, step))
    var target = _var(p, _nat(p))
    var induction = _nat_induction(p, motive, initial, inductive_step, target)
    return _lam(p, target, induction)


def _stop[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int,
    count: Int, guard: Int) -> Int:
    return _app(p, _app(p, _app(p, _constructor(p, 0), q), count), guard)


def _termination[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram, base_linear: Int) -> Int:
    """Construct a closed dependent term for finite driver execution.

    M(k) = forall q,N, N <= S*(q+k) -> Done(q,N).
    Natural induction establishes M(k), then k=N and q=0 give Done(0,N).
    k=N is a logical witness. It neither fixes geometry nor runs the driver.
    """
    var zero = _number(p, 0)
    var one = _number(p, 1)
    var k_motive = _var(p, _nat(p))
    var q_motive = _var(p, _nat(p))
    var n_motive = _var(p, _nat(p))
    var bound_motive = _var(p, _le(p, n_motive,
        _scale(p, program.stride, _plus(p, q_motive, k_motive))))
    var motive = _lam(p, k_motive, _pi(p, q_motive,
        _pi(p, n_motive, _pi(p, bound_motive, _done(p, q_motive, n_motive)))))

    # M(0): rewrite the assumed bound to the inactive driver guard.
    var q_zero = _var(p, _nat(p))
    var n_zero = _var(p, _nat(p))
    var bound_zero_term = _scale(p, program.stride, _plus(p, q_zero, zero))
    var bound_zero = _var(p, _le(p, n_zero, bound_zero_term))
    var guard_zero = _le_right(p, n_zero,
        _affine(p, bound_zero_term, _scale(p, program.stride, q_zero)), bound_zero)
    var base_zero = _app(p, base_linear, q_zero)
    guard_zero = _le_right(p, n_zero, _sym(p, _base(p, q_zero),
        _scale(p, program.stride, q_zero), base_zero), guard_zero)
    var initial = _lam(p, q_zero, _lam(p, n_zero,
        _lam(p, bound_zero, _stop(p, q_zero, n_zero, guard_zero))))

    # M(k+1): stop immediately or prepend one body to the finite suffix.
    var k = _var(p, _nat(p))
    var ih = _var(p, _app(p, motive, k))
    var q = _var(p, _nat(p))
    var count = _var(p, _nat(p))
    var next_k = _plus(p, k, one)
    var next_q = _plus(p, q, one)
    var successor_bound_term = _scale(p, program.stride, _plus(p, q, next_k))
    var successor_bound = _var(p, _le(p, count, successor_bound_term))
    var active_guard = _var(p, _node(p, tk.PT_LT, _base(p, q), count))
    var suffix_bound = _le_right(p, count,
        _affine(p, successor_bound_term,
            _scale(p, program.stride, _plus(p, next_q, k))), successor_bound)
    var suffix = _app(p, _app(p, _app(p, ih, next_q), count), suffix_bound)
    var step = _app(p, _app(p, _app(p, _app(p,
        _constructor(p, 1), q), count), active_guard), suffix)
    var active_case = _lam(p, active_guard, step)
    var inactive_guard = _var(p, _le(p, count, _base(p, q)))
    var inactive_case = _lam(p, inactive_guard, _stop(p, q, count, inactive_guard))
    var cases = _app(p, _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_ORDER_CASES), _base(p, q)), count),
        _done(p, q, count)), active_case), inactive_case)
    var inductive_step = _lam(p, k, _lam(p, ih, _lam(p, q,
        _lam(p, count, _lam(p, successor_bound, cases)))))

    # For all N, S>=1 proves N <= S*(0+N) by natural nonnegativity.
    var n = _var(p, _nat(p))
    var induction = _nat_induction(p, motive, initial, inductive_step, n)
    var excess_factor = program.stride - 1 if program.stride >= 1 else 0
    var excess = _scale(p, excess_factor, n)
    var nonnegative = _app(p, _decl(p, tk.DEC_NAT_NONNEG), excess)
    var sufficient = _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_LE_ADD), zero), excess), n), nonnegative)
    sufficient = _le_left(p, _plus(p, excess, n),
        _affine(p, _plus(p, zero, n), n), sufficient)
    sufficient = _le_right(p, n, _affine(p, _plus(p, excess, n),
        _scale(p, program.stride, _plus(p, zero, n))), sufficient)
    var finished = _app(p, _app(p, _app(p, induction, zero), n), sufficient)
    return _lam(p, n, finished)


def make_finishing_typed_proofs(program: FinishingProgram) -> FinishingTypedProofs[FINISHING_TYPED_PROOF_CAPACITY]:
    var bundle = tk.empty_proof_terms[FINISHING_TYPED_PROOF_CAPACITY]()
    if not declare_finishing_done(bundle):
        return FinishingTypedProofs[FINISHING_TYPED_PROOF_CAPACITY](bundle^, -1, -1)
    var base_root = _base_linearity(bundle, program)
    var termination_root = _termination(bundle, program, base_root)
    bundle.root = termination_root
    return FinishingTypedProofs[FINISHING_TYPED_PROOF_CAPACITY](bundle^, base_root, termination_root)


def check_finishing_typed_proofs[capacity: Int](program: FinishingProgram,
    proofs: FinishingTypedProofs[capacity]) -> Bool:
    """Require the exact checked definition and closed theorem types.

    The checker never obtains its required type from a producer annotation.
    The checker validates the shape descriptor. Production integer-range
    safety remains a separate obligation of the coverage/interpreter gate;
    these mathematical theorems range over arbitrary natural q and N.
    """
    # This storage bounds the specification's syntax, never its Nat indices.
    var expected_definition = tk.empty_proof_terms[FINISHING_DONE_DEFINITION_CAPACITY]()
    if not declare_finishing_done(expected_definition):
        return False
    if not tk.check_inductive_definition_matches(program, proofs.bundle, expected_definition):
        return False
    # Both roots inhabit the same immutable syntax/environment. The existing
    # multi-root checker still checks all syntax and each closed expected type,
    # while sharing inference and conversion work within that one admission.
    var bundle = proofs.bundle.copy()
    bundle.root = proofs.base_root
    var judgments = Array[tk.ProofTermJudgment, 2](fill=tk.ProofTermJudgment(-1, -1))
    judgments[0] = tk.ProofTermJudgment(proofs.base_root, base_linearity_type(bundle, program))
    judgments[1] = tk.ProofTermJudgment(proofs.termination_root, termination_type(bundle))
    return tk.check_proof_terms_many(program, bundle, judgments)


def _done_property[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    motive: Int, q: Int, count: Int, witness: Int) -> Int:
    return _app(p, _app(p, _app(p, motive, q), count), witness)


def finishing_done_eliminator_type[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    """Independent expected type, not the kernel's signature generator.

    P : forall q,N, Done(q,N) -> Prop.
    STOP establishes P on a stopped witness. STEP receives the original
    suffix and evidence of P on that suffix, then establishes P on STEP.
    """
    var mq = _var(p, _nat(p))
    var mn = _var(p, _nat(p))
    var mh = _var(p, _done(p, mq, mn))
    var motive_type = _pi(p, mq, _pi(p, mn, _pi(p, mh, _node(p, tk.PT_PROP))))
    var motive = _var(p, motive_type)

    var sq = _var(p, _nat(p))
    var sn = _var(p, _nat(p))
    var sg = _var(p, _le(p, sn, _base(p, sq)))
    var stop_witness = _stop(p, sq, sn, sg)
    var stop_type = _pi(p, sq, _pi(p, sn,
        _pi(p, sg, _done_property(p, motive, sq, sn, stop_witness))))
    var stop_branch = _var(p, stop_type)

    var tq = _var(p, _nat(p))
    var tn = _var(p, _nat(p))
    var tg = _var(p, _node(p, tk.PT_LT, _base(p, tq), tn))
    var next_q = _plus(p, tq, _number(p, 1))
    var suffix = _var(p, _done(p, next_q, tn))
    var ih = _var(p, _done_property(p, motive, next_q, tn, suffix))
    var step_witness = _app(p, _app(p, _app(p, _app(p,
        _constructor(p, 1), tq), tn), tg), suffix)
    var step_type = _pi(p, tq, _pi(p, tn, _pi(p, tg,
        _pi(p, suffix, _pi(p, ih, _done_property(p, motive, tq, tn, step_witness))))))
    var step_branch = _var(p, step_type)

    var q = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var h = _var(p, _done(p, q, n))
    return _pi(p, motive, _pi(p, stop_branch, _pi(p, step_branch,
        _pi(p, q, _pi(p, n, _pi(p, h, _done_property(p, motive, q, n, h)))))))


def finishing_done_count_monotonicity_type[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    """Independent target: Done(q,N) -> forall M<=N, Done(q,M)."""
    var q = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var h = _var(p, _done(p, q, n))
    var m = _var(p, _nat(p))
    var bound = _var(p, _le(p, m, n))
    return _pi(p, q, _pi(p, n, _pi(p, h,
        _pi(p, m, _pi(p, bound, _done(p, q, m))))))


def make_finishing_done_eliminator(program: FinishingProgram) -> tk.ProofTermBundle[FINISHING_DONE_ELIMINATION_CAPACITY]:
    """Submit the generic eliminator constant under the checked Done definition."""
    var p = tk.empty_proof_terms[FINISHING_DONE_ELIMINATION_CAPACITY]()
    if declare_finishing_done(p):
        p.root = _node(p, tk.PT_ELIMINATOR, value=0)
    return p^


def make_finishing_done_elimination_proof(program: FinishingProgram) -> tk.ProofTermBundle[FINISHING_DONE_ELIMINATION_CAPACITY]:
    """Use structural induction on Done to lower a finite execution's count.

    No natural count or proof tree is enumerated. The STEP branch genuinely
    needs its generated induction hypothesis at M: the original suffix only
    supplies a finite execution at N. The existing runtime certificate is
    unchanged; this is a separate proof-kernel capability and corollary.
    """
    var p = tk.empty_proof_terms[FINISHING_DONE_ELIMINATION_CAPACITY]()
    if not declare_finishing_done(p):
        return p^

    var mq = _var(p, _nat(p))
    var mn = _var(p, _nat(p))
    var mh = _var(p, _done(p, mq, mn))
    var mm = _var(p, _nat(p))
    var mb = _var(p, _le(p, mm, mn))
    var motive = _lam(p, mq, _lam(p, mn, _lam(p, mh,
        _pi(p, mm, _pi(p, mb, _done(p, mq, mm))))))

    # STOP: M <= N <= Base(q) supplies the smaller count's inactive guard.
    var sq = _var(p, _nat(p))
    var sn = _var(p, _nat(p))
    var sg = _var(p, _le(p, sn, _base(p, sq)))
    var sm = _var(p, _nat(p))
    var sb = _var(p, _le(p, sm, sn))
    var small_guard = _app(p, _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_LE_TRANS), sm), sn), _base(p, sq)), sb), sg)
    var stop_branch = _lam(p, sq, _lam(p, sn, _lam(p, sg,
        _lam(p, sm, _lam(p, sb, _stop(p, sq, sm, small_guard))))))

    # STEP: recurse at M, then decide the actual guard for that smaller count.
    var tq = _var(p, _nat(p))
    var tn = _var(p, _nat(p))
    var tg = _var(p, _node(p, tk.PT_LT, _base(p, tq), tn))
    var next_q = _plus(p, tq, _number(p, 1))
    var suffix = _var(p, _done(p, next_q, tn))
    var ih = _var(p, _done_property(p, motive, next_q, tn, suffix))
    var tm = _var(p, _nat(p))
    var tb = _var(p, _le(p, tm, tn))
    var smaller_suffix = _app(p, _app(p, ih, tm), tb)
    var active = _var(p, _node(p, tk.PT_LT, _base(p, tq), tm))
    var stepped = _app(p, _app(p, _app(p, _app(p,
        _constructor(p, 1), tq), tm), active), smaller_suffix)
    var inactive = _var(p, _le(p, tm, _base(p, tq)))
    var cases = _app(p, _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_ORDER_CASES), _base(p, tq)), tm), _done(p, tq, tm)),
        _lam(p, active, stepped)), _lam(p, inactive, _stop(p, tq, tm, inactive)))
    var step_branch = _lam(p, tq, _lam(p, tn, _lam(p, tg,
        _lam(p, suffix, _lam(p, ih, _lam(p, tm, _lam(p, tb, cases)))))))

    var q = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var h = _var(p, _done(p, q, n))
    var eliminated = _app(p, _app(p, _app(p, _app(p, _app(p, _app(p,
        _node(p, tk.PT_ELIMINATOR, value=0), motive), stop_branch), step_branch), q), n), h)
    p.root = _lam(p, q, _lam(p, n, _lam(p, h, eliminated)))
    return p^


def _matches_finishing_done[capacity: Int](program: FinishingProgram,
    proof: tk.ProofTermBundle[capacity]) -> Bool:
    var expected = tk.empty_proof_terms[FINISHING_DONE_DEFINITION_CAPACITY]()
    if not declare_finishing_done(expected):
        return False
    return tk.check_inductive_definition_matches(program, proof, expected)


def check_finishing_done_eliminator[capacity: Int](program: FinishingProgram,
    proof: tk.ProofTermBundle[capacity]) -> Bool:
    if not _matches_finishing_done(program, proof):
        return False
    var checked = proof.copy()
    var expected = finishing_done_eliminator_type(checked)
    return tk.check_proof_terms(program, checked, expected)


def check_finishing_done_elimination_proof[capacity: Int](program: FinishingProgram,
    proof: tk.ProofTermBundle[capacity]) -> Bool:
    if not _matches_finishing_done(program, proof):
        return False
    var checked = proof.copy()
    var expected = finishing_done_count_monotonicity_type(checked)
    return tk.check_proof_terms(program, checked, expected)
