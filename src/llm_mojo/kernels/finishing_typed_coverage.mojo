"""Untrusted dependent proof terms for the finishing loop's unique coverage.

The canonical owner is explicit Euclidean data rather than an enumerated table:
q=p/S, r=p%S, i=r/T and t=r%T. Closed checked terms establish visitation,
bounds, reconstruction and uniqueness. The same checked Nat declaration and
generic Prop eliminator used by termination derive visitation. Visited itself
is submitted as inductive declaration data, not a trusted constructor signature.
Arithmetic and order laws come from checked recursive Nat definitions. The
native integer decoding, affine reflection in the other coverage roots, generic
kernel machinery and runtime correspondence remain trusted. Headroom has a
separate raw Nat certificate; its fresh checker forbids all reflection syntax.
"""
from std.collections import Array
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof_kernel import FinishingProgram
from llm_mojo.kernels.nat_headroom import (
    HeadroomProof, HEADROOM_PROOF_CAPACITY, make_headroom_proof,
)
from llm_mojo.kernels.finishing_typed_derivation import (
    _node, _nat, _var, _number, _plus, _scale, _base, _eq, _le,
    _pi, _lam, _app, _decl, _constructor, _subst, _affine, _trans, _sym,
    _le_left, _le_right, _nat_induction, _base_linearity, base_linearity_type,
)

comptime FINISHING_TYPED_COVERAGE_CAPACITY = 640
comptime FINISHING_VISITED_DEFINITION_CAPACITY = 64


@fieldwise_init
struct FinishingTypedCoverageProofs[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var base_root: Int
    var visitation_root: Int
    var owner_visitation_root: Int
    var thread_bound_root: Int
    var item_bound_root: Int
    var index_root: Int
    var round_unique_root: Int
    var thread_unique_root: Int
    var item_unique_root: Int
    # This index belongs to headroom.bundle, not the other nine roots' bundle.
    var headroom_root: Int
    var headroom: HeadroomProof[HEADROOM_PROOF_CAPACITY]


def _lt[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_LT, a, b)


def _divide[capacity: Int](mut p: tk.ProofTermBundle[capacity], n: Int, d: Int) -> Int:
    return _node(p, tk.PT_DIV, n, value=d)


def _modulo[capacity: Int](mut p: tk.ProofTermBundle[capacity], n: Int, d: Int) -> Int:
    return _node(p, tk.PT_MOD, n, value=d)


def _parameter_decl[capacity: Int](mut p: tk.ProofTermBundle[capacity], declaration: Int,
    parameter: Int) -> Int:
    return _node(p, tk.PT_ARITHMETIC, extra=parameter, value=declaration)


def _visited[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int, n: Int) -> Int:
    return _app(p, _app(p, _node(p, tk.PT_FAMILY, value=tk.PROP_FAMILY_ID), q), n)


def _index[capacity: Int](mut p: tk.ProofTermBundle[capacity], program: FinishingProgram,
    q: Int, t: Int, i: Int) -> Int:
    return _plus(p, _plus(p, _base(p, q), t), _scale(p, program.threads, i))


def declare_finishing_visited[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Bool:
    """Declare the independently pinned forward visitation specification."""
    if not tk.pt_declare_family(p, 2):
        return False
    var n = _var(p, _nat(p))
    var zero = _number(p, 0)
    var guard = _var(p, _lt(p, _base(p, zero), n))
    if tk.pt_declare_constructor(p,
        _pi(p, n, _pi(p, guard, _visited(p, zero, n)))) != 0:
        return False
    var q = _var(p, _nat(p))
    var count = _var(p, _nat(p))
    var prior = _var(p, _visited(p, q, count))
    var successor = _plus(p, q, _number(p, 1))
    var active = _var(p, _lt(p, _base(p, successor), count))
    return tk.pt_declare_constructor(p, _pi(p, q, _pi(p, count,
        _pi(p, prior, _pi(p, active, _visited(p, successor, count)))))) == 1


def visitation_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    """Required type: forall q,N, S*q<N -> Visited(q,N)."""
    var q = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var bound = _var(p, _lt(p, _scale(p, program.stride, q), n))
    return _pi(p, q, _pi(p, n, _pi(p, bound, _visited(p, q, n))))


def owner_visitation_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    """Required type: forall N,p, p<N -> Visited(p/S,N)."""
    var n = _var(p, _nat(p))
    var position = _var(p, _nat(p))
    var in_range = _var(p, _lt(p, position, n))
    return _pi(p, n, _pi(p, position, _pi(p, in_range,
        _visited(p, _divide(p, position, program.stride), n))))


def thread_bound_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var position = _var(p, _nat(p))
    var remainder = _modulo(p, position, program.stride)
    return _pi(p, position,
        _lt(p, _modulo(p, remainder, program.threads), _number(p, program.threads)))


def item_bound_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var position = _var(p, _nat(p))
    var remainder = _modulo(p, position, program.stride)
    return _pi(p, position,
        _lt(p, _divide(p, remainder, program.threads), _number(p, program.items)))


def index_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var position = _var(p, _nat(p))
    var remainder = _modulo(p, position, program.stride)
    return _pi(p, position, _eq(p, _index(p, program,
        _divide(p, position, program.stride), _modulo(p, remainder, program.threads),
        _divide(p, remainder, program.threads)), position))


def uniqueness_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram, coordinate: Int) -> Int:
    """Required bounded-owner coordinate equality, without a domain premise."""
    var position = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var t = _var(p, _nat(p))
    var i = _var(p, _nat(p))
    var tb = _var(p, _lt(p, t, _number(p, program.threads)))
    var ib = _var(p, _lt(p, i, _number(p, program.items)))
    var reconstruction = _var(p, _eq(p, _index(p, program, q, t, i), position))
    var witness = _divide(p, position, program.stride)
    var selected = q
    if coordinate == 1:
        witness = _modulo(p, _modulo(p, position, program.stride), program.threads)
        selected = t
    elif coordinate == 2:
        witness = _divide(p, _modulo(p, position, program.stride), program.threads)
        selected = i
    elif coordinate != 0:
        return -1
    return _pi(p, position, _pi(p, q, _pi(p, t, _pi(p, i,
        _pi(p, tb, _pi(p, ib, _pi(p, reconstruction, _eq(p, selected, witness))))))))


def headroom_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var n = _var(p, _nat(p))
    var cap = _var(p, _le(p, n, _number(p, program.count_limit)))
    return _pi(p, n, _pi(p, cap, _le(p,
        _plus(p, n, _number(p, program.stride)), _number(p, program.integer_limit))))


def _lt_left[capacity: Int](mut p: tk.ProofTermBundle[capacity], b: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _lt(p, x, b)), equality, evidence)


def _lt_right[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _lt(p, a, x)), equality, evidence)


def _le_lt[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int,
    c: Int, first: Int, second: Int) -> Int:
    return _app(p, _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_LE_LT_TRANS), a), b), c), first), second)


def _lt_le[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int,
    c: Int, first: Int, second: Int) -> Int:
    return _app(p, _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_LT_LE_TRANS), a), b), c), first), second)


def _add_nonnegative[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    a: Int, b: Int) -> Int:
    # 0<=b implies a<=a+b by ordinary application and equality transport.
    var zero = _number(p, 0)
    var nonnegative = _app(p, _decl(p, tk.DEC_NAT_NONNEG), b)
    var added = _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_LE_ADD), zero), b), a), nonnegative)
    added = _le_left(p, _plus(p, b, a), _affine(p, _plus(p, zero, a), a), added)
    return _le_right(p, a, _affine(p, _plus(p, b, a), _plus(p, a, b)), added)


def _visitation[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram, base_linear: Int) -> Int:
    # M(q)=forall N,S*q<N->Visited(q,N), all variables remain symbolic.
    var zero = _number(p, 0)
    var one = _number(p, 1)
    var stride = _number(p, program.stride)
    var mq = _var(p, _nat(p))
    var mn = _var(p, _nat(p))
    var mg = _var(p, _lt(p, _scale(p, program.stride, mq), mn))
    var motive = _lam(p, mq, _pi(p, mn, _pi(p, mg, _visited(p, mq, mn))))

    var n0 = _var(p, _nat(p))
    var g0 = _var(p, _lt(p, _scale(p, program.stride, zero), n0))
    var base0 = _app(p, base_linear, zero)
    var guard0 = _lt_left(p, n0, _sym(p, _base(p, zero),
        _scale(p, program.stride, zero), base0), g0)
    var start = _lam(p, n0, _lam(p, g0,
        _app(p, _app(p, _constructor(p, 0), n0), guard0)))

    var q = _var(p, _nat(p))
    var ih = _var(p, _app(p, motive, q))
    var n = _var(p, _nat(p))
    var next_q = _plus(p, q, one)
    var sq = _scale(p, program.stride, q)
    var sq_next = _scale(p, program.stride, next_q)
    var next_guard = _var(p, _lt(p, sq_next, n))
    var preceding = _add_nonnegative(p, sq, stride)
    preceding = _le_right(p, sq, _affine(p, _plus(p, sq, stride), sq_next), preceding)
    var guard = _le_lt(p, sq, sq_next, n, preceding, next_guard)
    var previous_visit = _app(p, _app(p, ih, n), guard)
    var next_base = _app(p, base_linear, next_q)
    var real_guard = _lt_left(p, n,
        _sym(p, _base(p, next_q), sq_next, next_base), next_guard)
    var visit = _app(p, _app(p, _app(p, _app(p,
        _constructor(p, 1), q), n), previous_visit), real_guard)
    var step = _lam(p, q, _lam(p, ih, _lam(p, n, _lam(p, next_guard, visit))))
    var target = _var(p, _nat(p))
    return _lam(p, target, _nat_induction(p, motive, start, step, target))


def _owner_visitation[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram, visited_below: Int) -> Int:
    var n = _var(p, _nat(p))
    var position = _var(p, _nat(p))
    var in_range = _var(p, _lt(p, position, n))
    var q = _divide(p, position, program.stride)
    var r = _modulo(p, position, program.stride)
    var sq = _scale(p, program.stride, q)
    var decomposition = _app(p, _parameter_decl(p, tk.DEC_DIV_DECOMP, program.stride), position)
    var preceding = _add_nonnegative(p, sq, r)
    preceding = _le_right(p, sq, _sym(p, position, _plus(p, sq, r), decomposition), preceding)
    var guard = _le_lt(p, sq, position, n, preceding, in_range)
    return _lam(p, n, _lam(p, position, _lam(p, in_range,
        _app(p, _app(p, _app(p, visited_below, q), n), guard))))


def _thread_bound[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var position = _var(p, _nat(p))
    var r = _modulo(p, position, program.stride)
    return _lam(p, position, _app(p,
        _parameter_decl(p, tk.DEC_MOD_LT, program.threads), r))


def _item_bound[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var position = _var(p, _nat(p))
    var r = _modulo(p, position, program.stride)
    var i = _divide(p, r, program.threads)
    var t = _modulo(p, r, program.threads)
    var ti = _scale(p, program.threads, i)
    var inner = _app(p, _parameter_decl(p, tk.DEC_DIV_DECOMP, program.threads), r)
    var preceding = _add_nonnegative(p, ti, t)
    preceding = _le_right(p, ti, _sym(p, r, _plus(p, ti, t), inner), preceding)
    var remainder_bound = _app(p, _parameter_decl(p, tk.DEC_MOD_LT, program.stride), position)
    var scaled_bound = _le_lt(p, ti, r, _number(p, program.stride), preceding, remainder_bound)
    scaled_bound = _lt_right(p, ti, _affine(p, _number(p, program.stride),
        _scale(p, program.threads, _number(p, program.items))), scaled_bound)
    return _lam(p, position, _app(p, _app(p, _app(p,
        _parameter_decl(p, tk.DEC_LT_SCALE_CANCEL, program.threads), i),
        _number(p, program.items)), scaled_bound))


def _index_reconstruction[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram, base_linear: Int) -> Int:
    var position = _var(p, _nat(p))
    var q = _divide(p, position, program.stride)
    var r = _modulo(p, position, program.stride)
    var i = _divide(p, r, program.threads)
    var t = _modulo(p, r, program.threads)
    var sq = _scale(p, program.stride, q)
    var ti = _scale(p, program.threads, i)
    var nominal = _index(p, program, q, t, i)
    var replaced = _plus(p, _plus(p, sq, t), ti)
    var reconstructed = _plus(p, sq, _plus(p, ti, t))
    var base_equality = _app(p, base_linear, q)
    var x = _var(p, _nat(p))
    var base_replaced = _subst(p, _lam(p, x,
        _eq(p, nominal, _plus(p, _plus(p, x, t), ti))), base_equality,
        _node(p, tk.PT_REFL, nominal))
    var outer = _app(p, _parameter_decl(p, tk.DEC_DIV_DECOMP, program.stride), position)
    var inner = _app(p, _parameter_decl(p, tk.DEC_DIV_DECOMP, program.threads), r)
    var y = _var(p, _nat(p))
    var position_reconstructed = _subst(p,
        _lam(p, y, _eq(p, position, _plus(p, sq, y))), inner, outer)
    var result = _trans(p, nominal, replaced, reconstructed,
        base_replaced, _affine(p, replaced, reconstructed))
    result = _trans(p, nominal, reconstructed, position, result,
        _sym(p, position, reconstructed, position_reconstructed))
    return _lam(p, position, result)


@fieldwise_init
struct _UniqueRoots(ImplicitlyCopyable):
    var round: Int
    var thread: Int
    var item: Int


def _unique[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram, base_linear: Int) -> _UniqueRoots:
    var position = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var t = _var(p, _nat(p))
    var i = _var(p, _nat(p))
    var threads = _number(p, program.threads)
    var items = _number(p, program.items)
    var stride = _number(p, program.stride)
    var tb = _var(p, _lt(p, t, threads))
    var ib = _var(p, _lt(p, i, items))
    var candidate_index = _index(p, program, q, t, i)
    var index_equality = _var(p, _eq(p, candidate_index, position))
    var ti = _scale(p, program.threads, i)
    var offset = _plus(p, ti, t)
    var next_i = _plus(p, i, _number(p, 1))
    var scaled_next = _scale(p, program.threads, next_i)
    var scaled_items = _scale(p, program.threads, items)
    var next_bound = _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_LT_TO_LE_SUCC), i), items), ib)
    var scaled_bound = _app(p, _app(p, _app(p,
        _parameter_decl(p, tk.DEC_LE_SCALE, program.threads), next_i), items), next_bound)
    var offset_bound = _app(p, _app(p, _app(p, _app(p,
        _decl(p, tk.DEC_LT_ADD), t), threads), ti), tb)
    offset_bound = _lt_left(p, _plus(p, threads, ti),
        _affine(p, _plus(p, t, ti), offset), offset_bound)
    offset_bound = _lt_right(p, offset,
        _affine(p, _plus(p, threads, ti), scaled_next), offset_bound)
    offset_bound = _lt_le(p, offset, scaled_next, scaled_items, offset_bound, scaled_bound)
    offset_bound = _lt_right(p, offset, _affine(p, scaled_items, stride), offset_bound)

    var sq = _scale(p, program.stride, q)
    var base_equality = _app(p, base_linear, q)
    var replaced = _plus(p, _plus(p, sq, t), ti)
    var decomposition = _plus(p, sq, offset)
    var x = _var(p, _nat(p))
    var replace_base = _subst(p, _lam(p, x,
        _eq(p, candidate_index, _plus(p, _plus(p, x, t), ti))), base_equality,
        _node(p, tk.PT_REFL, candidate_index))
    var candidate_decomposition = _trans(p, position, candidate_index, replaced,
        _sym(p, candidate_index, position, index_equality), replace_base)
    candidate_decomposition = _trans(p, position, replaced, decomposition,
        candidate_decomposition, _affine(p, replaced, decomposition))
    var q_unique = _app(p, _app(p, _app(p, _app(p, _app(p,
        _parameter_decl(p, tk.DEC_DIV_QUOTIENT_UNIQUE, program.stride), position), q),
        offset), offset_bound), candidate_decomposition)
    var r_unique = _app(p, _app(p, _app(p, _app(p, _app(p,
        _parameter_decl(p, tk.DEC_DIV_REMAINDER_UNIQUE, program.stride), position), q),
        offset), offset_bound), candidate_decomposition)
    var r = _modulo(p, position, program.stride)
    var i_unique = _app(p, _app(p, _app(p, _app(p, _app(p,
        _parameter_decl(p, tk.DEC_DIV_QUOTIENT_UNIQUE, program.threads), r), i), t), tb), r_unique)
    var t_unique = _app(p, _app(p, _app(p, _app(p, _app(p,
        _parameter_decl(p, tk.DEC_DIV_REMAINDER_UNIQUE, program.threads), r), i), t), tb), r_unique)
    q_unique = _sym(p, _divide(p, position, program.stride), q, q_unique)
    t_unique = _sym(p, _modulo(p, r, program.threads), t, t_unique)
    i_unique = _sym(p, _divide(p, r, program.threads), i, i_unique)
    var q_root = _lam(p, position, _lam(p, q, _lam(p, t, _lam(p, i,
        _lam(p, tb, _lam(p, ib, _lam(p, index_equality, q_unique)))))))
    var t_root = _lam(p, position, _lam(p, q, _lam(p, t, _lam(p, i,
        _lam(p, tb, _lam(p, ib, _lam(p, index_equality, t_unique)))))))
    var i_root = _lam(p, position, _lam(p, q, _lam(p, t, _lam(p, i,
        _lam(p, tb, _lam(p, ib, _lam(p, index_equality, i_unique)))))))
    return _UniqueRoots(q_root, t_root, i_root)


def make_finishing_typed_coverage[capacity: Int = FINISHING_TYPED_COVERAGE_CAPACITY](
    program: FinishingProgram) -> FinishingTypedCoverageProofs[capacity]:
    var bundle = tk.empty_proof_terms[capacity]()
    var headroom = make_headroom_proof(program.count_limit,
        program.stride, program.integer_limit)
    if not declare_finishing_visited(bundle):
        return FinishingTypedCoverageProofs[capacity](bundle^,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, headroom^)
    var base = _base_linearity(bundle, program)
    var visits = _visitation(bundle, program, base)
    var owner = _owner_visitation(bundle, program, visits)
    var threads = _thread_bound(bundle, program)
    var items = _item_bound(bundle, program)
    var index = _index_reconstruction(bundle, program, base)
    var unique = _unique(bundle, program, base)
    bundle.root = owner
    return FinishingTypedCoverageProofs[capacity](bundle^, base, visits, owner,
        threads, items, index, unique.round, unique.thread, unique.item,
        headroom.proof, headroom^)


def check_finishing_typed_coverage[capacity: Int](program: FinishingProgram,
    proofs: FinishingTypedCoverageProofs[capacity]) -> Bool:
    """Pin nine coverage roots and independently check raw Nat headroom.

    Mandatory headroom checking starts from a fresh state with all native
    shortcuts disabled. No reflected judgment or cached type from the other
    coverage roots is imported into its certificate.
    """
    if (proofs.headroom_root != proofs.headroom.proof
        or not tk.check_native_headroom_request(proofs.headroom,
            program.count_limit, program.stride, program.integer_limit)):
        return False
    var expected_definition = tk.empty_proof_terms[FINISHING_VISITED_DEFINITION_CAPACITY]()
    if not declare_finishing_visited(expected_definition):
        return False
    if not tk.check_inductive_definition_matches(program, proofs.bundle, expected_definition):
        return False
    var bundle = proofs.bundle.copy()
    var judgments = Array[tk.ProofTermJudgment, 9](fill=tk.ProofTermJudgment(-1, -1))
    judgments[0] = tk.ProofTermJudgment(proofs.base_root, base_linearity_type(bundle, program))
    judgments[1] = tk.ProofTermJudgment(proofs.visitation_root, visitation_type(bundle, program))
    judgments[2] = tk.ProofTermJudgment(proofs.owner_visitation_root, owner_visitation_type(bundle, program))
    judgments[3] = tk.ProofTermJudgment(proofs.thread_bound_root, thread_bound_type(bundle, program))
    judgments[4] = tk.ProofTermJudgment(proofs.item_bound_root, item_bound_type(bundle, program))
    judgments[5] = tk.ProofTermJudgment(proofs.index_root, index_type(bundle, program))
    judgments[6] = tk.ProofTermJudgment(proofs.round_unique_root, uniqueness_type(bundle, program, 0))
    judgments[7] = tk.ProofTermJudgment(proofs.thread_unique_root, uniqueness_type(bundle, program, 1))
    judgments[8] = tk.ProofTermJudgment(proofs.item_unique_root, uniqueness_type(bundle, program, 2))
    return tk.check_proof_terms_many(program, bundle, judgments)
