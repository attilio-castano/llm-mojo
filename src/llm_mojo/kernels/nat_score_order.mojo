"""Ordinary generic order proofs for the symbolic score bridge.

Eight closed definitions use the relocated Add/Mul/Le data and laws, Max,
equality transport and Nat induction. Le(0,b) is Church True, distinct from
the phase model's Eq(0,0). Strict order means Le(succ(a),b). No assumed score
equation or native comparison participates; every export is checked independently.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.nat_join_max as maximum
import llm_mojo.kernels.nat_score_link as score

comptime SCORE_ORDER_DEFINITIONS = 8


@fieldwise_init
struct ScoreOrderExports(ImplicitlyCopyable):
    var prefix_le: Int
    var prefix_lt: Int
    var mul_left_le: Int
    var lt_to_le: Int
    var max_right_le: Int
    var max_left_le: Int
    var max_right_lt: Int
    var max_left_lt: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _lt[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return score.score_le(p, ph._succ(p, left), right)


def _true_unit[C: Int](mut p: ph._Builder[C]) -> Int:
    # Le(0,b) reduces to forall P:Prop, P -> P, with this closed inhabitant.
    var proposition = ph._var(p, ph._prop(p))
    var evidence = ph._var(p, proposition)
    return ph._lams(p, [proposition, evidence], evidence)


def _transport_left[C: Int](mut p: ph._Builder[C], old: Int, new: Int,
    right: Int, equality: Int, evidence: Int) -> Int:
    var changed = ph._var(p, ph._nat(p))
    return ph._subst(p, ph._lam(p, changed, score.score_le(p, changed, right)),
        equality, evidence)


def _transport_right[C: Int](mut p: ph._Builder[C], left: Int,
    old: Int, new: Int, equality: Int, evidence: Int) -> Int:
    var changed = ph._var(p, ph._nat(p))
    return ph._subst(p, ph._lam(p, changed, score.score_le(p, left, changed)),
        equality, evidence)


def _prefix_le[C: Int](mut p: ph._Builder[C]) -> Int:
    # forall k a b, Le(a,b) -> Le(Add(k,a),Add(k,b)).
    var definition = p.bundle.definition_count
    var prefix = ph._var(p, ph._nat(p))
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var h = ph._var(p, score.score_le(p, a, b))
    var suffix_a = score.score_add(p, a, prefix)
    var suffix_b = score.score_add(p, b, prefix)
    var prefix_a = score.score_add(p, prefix, a)
    var prefix_b = score.score_add(p, prefix, b)
    var suffix_order = _call(p, score.SCORE_LE_ADD, [a, b, prefix, h])
    var left_eq = _call(p, score.SCORE_ADD_COMM, [a, prefix])
    var right_eq = _call(p, score.SCORE_ADD_COMM, [b, prefix])
    var first = _transport_left(p, suffix_a, prefix_a, suffix_b, left_eq, suffix_order)
    var result = _transport_right(p, prefix_a, suffix_b, prefix_b, right_eq, first)
    var binders: List[Int] = [prefix, a, b, h]
    ph._declare(p, ph._pis(p, binders, score.score_le(p, prefix_a, prefix_b)),
        ph._lams(p, binders, result), definition)
    return definition


def _prefix_lt[C: Int](mut p: ph._Builder[C], prefix_le: Int) -> Int:
    # Add recurses on its right input. Add(k,S a)=S(Add(k,a)) is
    # definitional, so prefix_le(k,S a,b) proves strictness directly.
    var definition = p.bundle.definition_count
    var prefix = ph._var(p, ph._nat(p))
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var h = ph._var(p, _lt(p, a, b))
    var result = _call(p, prefix_le, [prefix, ph._succ(p, a), b, h])
    var binders: List[Int] = [prefix, a, b, h]
    ph._declare(p, ph._pis(p, binders, _lt(p,
        score.score_add(p, prefix, a), score.score_add(p, prefix, b))),
        ph._lams(p, binders, result), definition)
    return definition


def _mul_left_le[C: Int](mut p: ph._Builder[C], prefix_le: Int) -> Int:
    # forall a b k, Le(a,b) -> Le(Mul(a,k),Mul(b,k)).
    # Induct on k, using the actual recursion Mul(a,S k)=Add(a,Mul(a,k)).
    var definition = p.bundle.definition_count
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var target = ph._var(p, ph._nat(p))
    var h = ph._var(p, score.score_le(p, a, b))
    var k = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, k, score.score_le(p,
        score.score_mul(p, a, k), score.score_mul(p, b, k)))
    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    var left_product = score.score_mul(p, a, predecessor)
    var right_product = score.score_mul(p, b, predecessor)
    var left = score.score_add(p, a, left_product)
    var middle = score.score_add(p, b, left_product)
    var right = score.score_add(p, b, right_product)
    var first = _call(p, score.SCORE_LE_ADD, [a, b, left_product, h])
    var second = _call(p, prefix_le, [b, left_product, right_product, ih])
    var result = _call(p, score.SCORE_LE_TRANS, [left, middle, right, first, second])
    var step = ph._lams(p, [predecessor, ih], result)
    var binders: List[Int] = [a, b, target, h]
    ph._declare(p, ph._pis(p, binders, ph._app(p, motive, target)),
        ph._lams(p, binders, ph._induct(p, motive, _true_unit(p), step, target)), definition)
    return definition


def _lt_to_le[C: Int](mut p: ph._Builder[C]) -> Int:
    # forall a b, Le(S a,b) -> Le(a,b). No total-order axiom or native
    # comparison decides a constructor case; both arguments use Nat induction.
    var definition = p.bundle.definition_count
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var guard = ph._var(p, _lt(p, a, b))
    var outer_motive = ph._lam(p, a, ph._pis(p, [b, guard], score.score_le(p, a, b)))
    b = ph._var(p, ph._nat(p))
    guard = ph._var(p, _lt(p, ph._zero(p), b))
    var base = ph._lams(p, [b, guard], _true_unit(p))

    var predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, predecessor))
    var successor = ph._succ(p, predecessor)
    b = ph._var(p, ph._nat(p))
    guard = ph._var(p, _lt(p, successor, b))
    var inner_motive = ph._lam(p, b,
        ph._pi(p, guard, score.score_le(p, successor, b)))
    # Both Le(S(S a),0) and Le(S a,0) reduce to Church False.
    var impossible = ph._var(p, _lt(p, successor, ph._zero(p)))
    var inner_base = ph._lam(p, impossible, impossible)
    var right_predecessor = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, right_predecessor))
    guard = ph._var(p, _lt(p, successor, ph._succ(p, right_predecessor)))
    var smaller = ph._apps(p, outer_ih, [right_predecessor, guard])
    var inner_step = ph._lams(p, [right_predecessor, inner_ih, guard], smaller)
    b = ph._var(p, ph._nat(p))
    var outer_step = ph._lams(p, [predecessor, outer_ih, b],
        ph._induct(p, inner_motive, inner_base, inner_step, b))
    a = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, a, ph._app(p, outer_motive, a)),
        ph._lam(p, a, ph._induct(p, outer_motive, base, outer_step, a)), definition)
    return definition


def _max_right_le[C: Int](mut p: ph._Builder[C]) -> Int:
    # forall a b, Le(a,b) -> Eq(Max55(a,b),b).
    var definition = p.bundle.definition_count
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var guard = ph._var(p, score.score_le(p, a, b))
    var outer_motive = ph._lam(p, a, ph._pis(p, [b, guard],
        ph._eq(p, maximum.record_max(p, a, b), b)))
    b = ph._var(p, ph._nat(p))
    guard = ph._var(p, score.score_le(p, ph._zero(p), b))
    var base = ph._lams(p, [b, guard], ph._refl(p, b))

    var predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, predecessor))
    var successor = ph._succ(p, predecessor)
    b = ph._var(p, ph._nat(p))
    guard = ph._var(p, score.score_le(p, successor, b))
    var inner_motive = ph._lam(p, b, ph._pi(p, guard,
        ph._eq(p, maximum.record_max(p, successor, b), b)))
    var impossible = ph._var(p, score.score_le(p, successor, ph._zero(p)))
    var zero_goal = ph._eq(p, maximum.record_max(p, successor, ph._zero(p)), ph._zero(p))
    var inner_base = ph._lam(p, impossible, ph._app(p, impossible, zero_goal))

    var right_predecessor = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, right_predecessor))
    guard = ph._var(p, score.score_le(p, successor, ph._succ(p, right_predecessor)))
    var smaller = ph._apps(p, outer_ih, [right_predecessor, guard])
    var lifted = ph._succ_cong(p, maximum.record_max(p, predecessor, right_predecessor),
        right_predecessor, smaller)
    var inner_step = ph._lams(p, [right_predecessor, inner_ih, guard], lifted)
    b = ph._var(p, ph._nat(p))
    var outer_step = ph._lams(p, [predecessor, outer_ih, b],
        ph._induct(p, inner_motive, inner_base, inner_step, b))
    a = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pi(p, a, ph._app(p, outer_motive, a)),
        ph._lam(p, a, ph._induct(p, outer_motive, base, outer_step, a)), definition)
    return definition


def _max_left_le[C: Int](mut p: ph._Builder[C], max_right_le: Int) -> Int:
    var definition = p.bundle.definition_count
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var guard = ph._var(p, score.score_le(p, b, a))
    var left = maximum.record_max(p, a, b)
    var swapped = maximum.record_max(p, b, a)
    var commutative = _call(p, maximum.RECORD_MAX_COMMUTATIVE, [a, b])
    var selected = _call(p, max_right_le, [b, a, guard])
    var binders: List[Int] = [a, b, guard]
    ph._declare(p, ph._pis(p, binders, ph._eq(p, left, a)),
        ph._lams(p, binders, ph._trans(p, left, swapped, a, commutative, selected)), definition)
    return definition


def _max_strict[C: Int](mut p: ph._Builder[C], lt_to_le: Int,
    weak_selection: Int, right: Bool) -> Int:
    var definition = p.bundle.definition_count
    var a = ph._var(p, ph._nat(p))
    var b = ph._var(p, ph._nat(p))
    var smaller = a if right else b
    var larger = b if right else a
    var guard = ph._var(p, _lt(p, smaller, larger))
    var weak = _call(p, lt_to_le, [smaller, larger, guard])
    var selected = _call(p, weak_selection, [a, b, weak])
    var binders: List[Int] = [a, b, guard]
    ph._declare(p, ph._pis(p, binders,
        ph._eq(p, maximum.record_max(p, a, b), larger)),
        ph._lams(p, binders, selected), definition)
    return definition


def append_nat_score_order_terms[C: Int](mut p: ph._Builder[C]) -> ScoreOrderExports:
    """Append eight ordinary helpers at the current count, preserving root94.

    Export fields are candidate declaration IDs, not checked proof handles.
    The independent consumer must reconstruct and check each theorem type.
    """
    if p.bundle.count <= 0 or p.bundle.definition_count < score.NAT_SCORE_DEFINITIONS:
        p.bundle.count = -1
        return ScoreOrderExports(-1, -1, -1, -1, -1, -1, -1, -1)
    var first = p.bundle.definition_count
    var prefix_le = _prefix_le(p)
    var prefix_lt = _prefix_lt(p, prefix_le)
    var mul_left_le = _mul_left_le(p, prefix_le)
    var lt_to_le = _lt_to_le(p)
    var max_right_le = _max_right_le(p)
    var max_left_le = _max_left_le(p, max_right_le)
    var max_right_lt = _max_strict(p, lt_to_le, max_right_le, True)
    var max_left_lt = _max_strict(p, lt_to_le, max_left_le, False)
    if (p.bundle.count < 0 or p.bundle.definition_count != first + SCORE_ORDER_DEFINITIONS):
        p.bundle.count = -1
        return ScoreOrderExports(-1, -1, -1, -1, -1, -1, -1, -1)
    return ScoreOrderExports(prefix_le, prefix_lt, mul_left_le, lt_to_le,
        max_right_le, max_left_le, max_right_lt, max_left_lt)
