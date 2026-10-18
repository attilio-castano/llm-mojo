"""Ordinary Nat-recursive division data and untrusted arithmetic proof terms.

Division by k+1 uses a quotient/remainder/countdown triple. Its definition and
proof syntax have fixed size; neither the divisor nor the dividend is unrolled
by the producer. The generic proof-term checker validates all definitions and
proofs before the coverage gate can use them.
"""
import llm_mojo.kernels.proof_term_syntax as tk
from llm_mojo.kernels.nat_arithmetic_terms import (
    _ArithmeticBuilder, _declare_definition,
    _node, _nat, _prop, _var, _number, _succ, _named, _app, _app2, _app3,
    _pi, _lam, _plus, _mul, _eq, _le, _lt, _refl, _subst, _sym, _trans,
    _cong, _cong_left, _cong_right, _nat_induction, _rec,
    _and_type, _and_intro, _and_left, _and_right, _absurd,
    NatArithmeticCore,
)


def division_triple_type[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var q = _var(p, _nat(p))
    var r = _var(p, _nat(p))
    var countdown = _var(p, _nat(p))
    var selector = _var(p, _pi(p, q, _pi(p, r, _pi(p, countdown, _nat(p)))))
    return _pi(p, selector, _nat(p))


def _triple[C: Int](mut p: _ArithmeticBuilder[C], q: Int, r: Int,
    countdown: Int) -> Int:
    var q_arg = _var(p, _nat(p))
    var r_arg = _var(p, _nat(p))
    var c_arg = _var(p, _nat(p))
    var selector = _var(p, _pi(p, q_arg, _pi(p, r_arg, _pi(p, c_arg, _nat(p)))))
    return _lam(p, selector, _app3(p, selector, q, r, countdown))


def _project[C: Int](mut p: _ArithmeticBuilder[C], state: Int,
    component: Int) -> Int:
    var q = _var(p, _nat(p))
    var r = _var(p, _nat(p))
    var countdown = _var(p, _nat(p))
    var selected = q if component == 0 else r if component == 1 else countdown
    return _app(p, state, _lam(p, q, _lam(p, r, _lam(p, countdown, selected))))


def _advance[C: Int](mut p: _ArithmeticBuilder[C], k: Int, q: Int,
    r: Int, countdown: Int) -> Int:
    var c = _var(p, _nat(p))
    var ignored = _var(p, division_triple_type(p))
    var wrap = _triple(p, _succ(p, q), _number(p, 0), k)
    var increment = _triple(p, q, _succ(p, r), c)
    return _rec(p, countdown, wrap, _lam(p, c, _lam(p, ignored, increment)))


def division_definition_type[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    return _pi(p, k, _pi(p, n, division_triple_type(p)))


def division_definition_body[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var previous = _var(p, _nat(p))
    var state = _var(p, division_triple_type(p))
    var initial = _triple(p, _number(p, 0), _number(p, 0), k)
    var advanced = _advance(p, k, _project(p, state, 0), _project(p, state, 1),
        _project(p, state, 2))
    var step = _lam(p, previous, _lam(p, state, advanced))
    return _lam(p, k, _lam(p, n, _rec(p, n, initial, step)))


def declare_nat_arithmetic_division[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    return _declare_definition(p, division_definition_type(p), division_definition_body(p))


def division_state[C: Int](mut p: _ArithmeticBuilder[C], k: Int, n: Int) -> Int:
    return _app2(p, _named(p, 3), k, n)


def division_quotient[C: Int](mut p: _ArithmeticBuilder[C], k: Int, n: Int) -> Int:
    return _project(p, division_state(p, k, n), 0)


def division_remainder[C: Int](mut p: _ArithmeticBuilder[C], k: Int, n: Int) -> Int:
    return _project(p, division_state(p, k, n), 1)


def _decomposition[C: Int](mut p: _ArithmeticBuilder[C], k: Int, n: Int,
    state: Int) -> Int:
    return _eq(p, n, _plus(p, _mul(p, _succ(p, k), _project(p, state, 0)),
        _project(p, state, 1)))


def _countdown_sum[C: Int](mut p: _ArithmeticBuilder[C], k: Int, state: Int) -> Int:
    return _eq(p, _plus(p, _project(p, state, 1), _project(p, state, 2)), k)


def _invariant[C: Int](mut p: _ArithmeticBuilder[C], k: Int, n: Int,
    state: Int) -> Int:
    return _and_type(p, _decomposition(p, k, n, state), _countdown_sum(p, k, state))


def division_invariant_type[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    return _pi(p, k, _pi(p, n, _invariant(p, k, n, division_state(p, k, n))))


def division_decomposition_type[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    return _pi(p, k, _pi(p, n, _decomposition(p, k, n, division_state(p, k, n))))


def division_remainder_bound_type[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    return _pi(p, k, _pi(p, n, _lt(p, division_remainder(p, k, n), _succ(p, k))))


def division_invariant_body[C: Int](mut p: _ArithmeticBuilder[C],
    core: NatArithmeticCore) -> Int:
    var k = _var(p, _nat(p))
    var zero = _number(p, 0)
    var d = _succ(p, k)
    var motive_n = _var(p, _nat(p))
    var motive = _lam(p, motive_n,
        _invariant(p, k, motive_n, division_state(p, k, motive_n)))
    var initial_state = division_state(p, k, zero)
    var initial = _and_intro(p, _decomposition(p, k, zero, initial_state),
        _countdown_sum(p, k, initial_state), _refl(p, zero),
        _app(p, _named(p, core.add_zero_left), k))

    var n = _var(p, _nat(p))
    var current_state = division_state(p, k, n)
    var current_d = _decomposition(p, k, n, current_state)
    var current_sum = _countdown_sum(p, k, current_state)
    var ih = _var(p, _app(p, motive, n))
    var hd = _and_left(p, current_d, current_sum, ih)
    var hs = _and_right(p, current_d, current_sum, ih)
    var q = _project(p, current_state, 0)
    var r = _project(p, current_state, 1)
    var countdown = _project(p, current_state, 2)
    var product = _mul(p, d, q)
    var successor_n = _succ(p, n)
    var successor_function_arg = _var(p, _nat(p))
    var successor_function = _lam(p, successor_function_arg, _succ(p, successor_function_arg))
    var successor_decomposition = _cong(p, successor_function, n,
        _plus(p, product, r), hd)

    # The case motive abstracts just the countdown. Quotient and remainder
    # remain neutral projections; no equality of Church triples is assumed.
    var c_arg = _var(p, _nat(p))
    var sum_premise = _var(p, _eq(p, _plus(p, r, c_arg), k))
    var case_motive = _lam(p, c_arg, _pi(p, sum_premise,
        _invariant(p, k, successor_n, _advance(p, k, q, r, c_arg))))

    var wrap_sum = _var(p, _eq(p, _plus(p, r, zero), k))
    var replacement_arg = _var(p, _nat(p))
    var replacement_function = _lam(p, replacement_arg,
        _succ(p, _plus(p, product, replacement_arg)))
    var replace_remainder = _cong(p, replacement_function, r, k, wrap_sum)
    var wrap_first = _trans(p, successor_n, _succ(p, _plus(p, product, r)),
        _succ(p, _plus(p, product, k)), successor_decomposition, replace_remainder)
    var multiplication_step = _app2(p, _named(p, core.mul_succ_rhs), d, q)
    var multiplication_backwards = _sym(p, _mul(p, d, _succ(p, q)),
        _plus(p, product, d), multiplication_step)
    var wrap_decomposition = _trans(p, successor_n, _succ(p, _plus(p, product, k)),
        _mul(p, d, _succ(p, q)), wrap_first, multiplication_backwards)
    var wrapped_state = _advance(p, k, q, r, zero)
    var wrapped = _and_intro(p, _decomposition(p, k, successor_n, wrapped_state),
        _countdown_sum(p, k, wrapped_state), wrap_decomposition,
        _app(p, _named(p, core.add_zero_left), k))
    var wrap_branch = _lam(p, wrap_sum, wrapped)

    var c = _var(p, _nat(p))
    var ignored_case_ih = _var(p, _app(p, case_motive, c))
    var increment_sum = _var(p, _eq(p, _plus(p, r, _succ(p, c)), k))
    var sum_successor_left = _app2(p, _named(p, core.add_succ_left), r, c)
    var incremented_sum = _trans(p, _plus(p, _succ(p, r), c),
        _succ(p, _plus(p, r, c)), k, sum_successor_left, increment_sum)
    var incremented_state = _advance(p, k, q, r, _succ(p, c))
    var incremented = _and_intro(p, _decomposition(p, k, successor_n, incremented_state),
        _countdown_sum(p, k, incremented_state), successor_decomposition, incremented_sum)
    var increment_branch = _lam(p, c, _lam(p, ignored_case_ih,
        _lam(p, increment_sum, incremented)))

    var cases = _nat_induction(p, case_motive, wrap_branch, increment_branch, countdown)
    var step = _lam(p, n, _lam(p, ih, _app(p, cases, hs)))
    var target = _var(p, _nat(p))
    return _lam(p, k, _lam(p, target, _nat_induction(p, motive, initial, step, target)))


def division_decomposition_body[C: Int](mut p: _ArithmeticBuilder[C], invariant: Int) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var state = division_state(p, k, n)
    var proof = _app2(p, _named(p, invariant), k, n)
    return _lam(p, k, _lam(p, n, _and_left(p, _decomposition(p, k, n, state),
        _countdown_sum(p, k, state), proof)))


def _rewrite_le_left[C: Int](mut p: _ArithmeticBuilder[C], b: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _le(p, x, b)), equality, evidence)


def _rewrite_le_right[C: Int](mut p: _ArithmeticBuilder[C], a: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _le(p, a, x)), equality, evidence)


def _rewrite_lt_left[C: Int](mut p: _ArithmeticBuilder[C], b: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _lt(p, x, b)), equality, evidence)


def _rewrite_lt_right[C: Int](mut p: _ArithmeticBuilder[C], a: Int,
    equality: Int, evidence: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _lt(p, a, x)), equality, evidence)


def _sum_ge_left[C: Int](mut p: _ArithmeticBuilder[C], core: NatArithmeticCore,
    a: Int, b: Int) -> Int:
    var zero = _number(p, 0)
    var b_nonnegative = _app(p, _named(p, core.nat_nonneg), b)
    var shifted = _app(p, _app3(p, _named(p, core.le_add), zero, b, a), b_nonnegative)
    var left = _rewrite_le_left(p, _plus(p, b, a),
        _app(p, _named(p, core.add_zero_left), a), shifted)
    return _rewrite_le_right(p, a, _app2(p, _named(p, core.add_comm), b, a), left)


def division_remainder_bound_body[C: Int](mut p: _ArithmeticBuilder[C],
    core: NatArithmeticCore, invariant: Int) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var state = division_state(p, k, n)
    var r = _project(p, state, 1)
    var countdown = _project(p, state, 2)
    var proof = _app2(p, _named(p, invariant), k, n)
    var sum_equality = _and_right(p, _decomposition(p, k, n, state),
        _countdown_sum(p, k, state), proof)
    var bound = _rewrite_le_right(p, r, sum_equality,
        _sum_ge_left(p, core, r, countdown))
    # Le(r,k) and Lt(r,succ(k)) compute to the same checked proposition.
    return _lam(p, k, _lam(p, n, bound))


def division_quotient_unique_type[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var r = _var(p, _nat(p))
    var bound = _var(p, _lt(p, r, _succ(p, k)))
    var decomposition = _var(p, _eq(p, n, _plus(p, _mul(p, _succ(p, k), q), r)))
    return _pi(p, k, _pi(p, n, _pi(p, q, _pi(p, r, _pi(p, bound,
        _pi(p, decomposition, _eq(p, division_quotient(p, k, n), q)))))))


def division_remainder_unique_type[C: Int](mut p: _ArithmeticBuilder[C]) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var r = _var(p, _nat(p))
    var bound = _var(p, _lt(p, r, _succ(p, k)))
    var decomposition = _var(p, _eq(p, n, _plus(p, _mul(p, _succ(p, k), q), r)))
    return _pi(p, k, _pi(p, n, _pi(p, q, _pi(p, r, _pi(p, bound,
        _pi(p, decomposition, _eq(p, division_remainder(p, k, n), r)))))))


def _quotient_lt_impossible[C: Int](mut p: _ArithmeticBuilder[C],
    core: NatArithmeticCore, k: Int, n: Int, qa: Int, ra: Int, qb: Int, rb: Int,
    remainder_bound: Int, decomp_a: Int, decomp_b: Int, quotient_lt: Int) -> Int:
    var d = _succ(p, k)
    var product_a = _mul(p, d, qa)
    var product_b = _mul(p, d, qb)
    var total_a = _plus(p, product_a, ra)
    var total_b = _plus(p, product_b, rb)
    var shifted = _app(p, _app3(p, _named(p, core.lt_add), ra, d, product_a), remainder_bound)
    var left_commuted = _rewrite_lt_left(p, _plus(p, d, product_a),
        _app2(p, _named(p, core.add_comm), ra, product_a), shifted)
    var both_commuted = _rewrite_lt_right(p, total_a,
        _app2(p, _named(p, core.add_comm), d, product_a), left_commuted)
    var multiply_step = _app2(p, _named(p, core.mul_succ_rhs), d, qa)
    var multiply_backwards = _sym(p, _mul(p, d, _succ(p, qa)),
        _plus(p, product_a, d), multiply_step)
    var next_product_bound = _rewrite_lt_right(p, total_a, multiply_backwards, both_commuted)
    var next_quotient_bound = _app(p, _app2(p, _named(p, core.lt_to_le_succ), qa, qb), quotient_lt)
    var scaled = _app(p, _app3(p, _named(p, core.le_scale), d, _succ(p, qa), qb),
        next_quotient_bound)
    var product_bound = _app2(p, _app3(p, _named(p, core.lt_le_trans),
        total_a, _mul(p, d, _succ(p, qa)), product_b), next_product_bound, scaled)
    var total_bound = _app2(p, _app3(p, _named(p, core.lt_le_trans),
        total_a, product_b, total_b), product_bound, _sum_ge_left(p, core, product_b, rb))
    var left_is_n = _sym(p, n, total_a, decomp_a)
    var right_is_n = _sym(p, n, total_b, decomp_b)
    var n_below_total = _rewrite_lt_left(p, total_b, left_is_n, total_bound)
    var n_below_n = _rewrite_lt_right(p, n, right_is_n, n_below_total)
    return _app2(p, _named(p, core.lt_irrefl), n, n_below_n)


def division_quotient_unique_body[C: Int](mut p: _ArithmeticBuilder[C],
    core: NatArithmeticCore, decomposition: Int, remainder_bound: Int) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var r = _var(p, _nat(p))
    var bound = _var(p, _lt(p, r, _succ(p, k)))
    var candidate = _var(p, _eq(p, n, _plus(p, _mul(p, _succ(p, k), q), r)))
    var canonical_q = division_quotient(p, k, n)
    var canonical_r = division_remainder(p, k, n)
    var canonical_decomp = _app2(p, _named(p, decomposition), k, n)
    var canonical_bound = _app2(p, _named(p, remainder_bound), k, n)
    var result_type = _eq(p, canonical_q, q)

    var canonical_smaller = _var(p, _lt(p, canonical_q, q))
    var smaller_impossible = _quotient_lt_impossible(p, core, k, n,
        canonical_q, canonical_r, q, r, canonical_bound, canonical_decomp,
        candidate, canonical_smaller)
    var smaller_branch = _lam(p, canonical_smaller, _absurd(p, smaller_impossible, result_type))

    var candidate_le = _var(p, _le(p, q, canonical_q))
    var candidate_smaller = _var(p, _lt(p, q, canonical_q))
    var reverse_impossible = _quotient_lt_impossible(p, core, k, n,
        q, r, canonical_q, canonical_r, bound, candidate, canonical_decomp,
        candidate_smaller)
    var reverse_branch = _lam(p, candidate_smaller, _absurd(p, reverse_impossible, result_type))
    var canonical_le = _var(p, _le(p, canonical_q, q))
    var equal = _app2(p, _app2(p, _named(p, core.le_antisym), canonical_q, q),
        canonical_le, candidate_le)
    var equal_branch = _lam(p, canonical_le, equal)
    var reverse_cases = _app2(p, _app3(p, _named(p, core.order_cases),
        q, canonical_q, result_type), reverse_branch, equal_branch)
    var larger_branch = _lam(p, candidate_le, reverse_cases)
    var cases = _app2(p, _app3(p, _named(p, core.order_cases),
        canonical_q, q, result_type), smaller_branch, larger_branch)
    return _lam(p, k, _lam(p, n, _lam(p, q, _lam(p, r,
        _lam(p, bound, _lam(p, candidate, cases))))))


def division_remainder_unique_body[C: Int](mut p: _ArithmeticBuilder[C],
    core: NatArithmeticCore, decomposition: Int, quotient_unique: Int) -> Int:
    var k = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var r = _var(p, _nat(p))
    var bound = _var(p, _lt(p, r, _succ(p, k)))
    var product = _mul(p, _succ(p, k), q)
    var candidate_total = _plus(p, product, r)
    var candidate = _var(p, _eq(p, n, candidate_total))
    var canonical_q = division_quotient(p, k, n)
    var canonical_r = division_remainder(p, k, n)
    var canonical_total = _plus(p, _mul(p, _succ(p, k), canonical_q), canonical_r)
    var canonical_decomp = _app2(p, _named(p, decomposition), k, n)
    var reverse_decomp = _sym(p, n, canonical_total, canonical_decomp)
    var equal_totals = _trans(p, canonical_total, n, candidate_total, reverse_decomp, candidate)
    var equal_quotients = _app2(p, _app(p, _app3(p, _named(p, quotient_unique),
        k, n, q), r), bound, candidate)
    var quotient_arg = _var(p, _nat(p))
    var motive = _lam(p, quotient_arg, _eq(p,
        _plus(p, _mul(p, _succ(p, k), quotient_arg), canonical_r), candidate_total))
    var common_prefix = _subst(p, motive, equal_quotients, equal_totals)
    var cancellation = _app(p, _app3(p, _named(p, core.add_cancel_prefix),
        product, canonical_r, r), common_prefix)
    return _lam(p, k, _lam(p, n, _lam(p, q, _lam(p, r,
        _lam(p, bound, _lam(p, candidate, cancellation))))))


@fieldwise_init
struct NatArithmeticDivision(ImplicitlyCopyable):
    var invariant: Int
    var decomposition: Int
    var remainder_bound: Int
    var quotient_unique: Int
    var remainder_unique: Int


def make_nat_arithmetic_division[C: Int](mut p: _ArithmeticBuilder[C],
    core: NatArithmeticCore) -> NatArithmeticDivision:
    var invariant = _declare_definition(p, division_invariant_type(p),
        division_invariant_body(p, core))
    var decomposition = _declare_definition(p, division_decomposition_type(p),
        division_decomposition_body(p, invariant))
    var bound = _declare_definition(p, division_remainder_bound_type(p),
        division_remainder_bound_body(p, core, invariant))
    var quotient = _declare_definition(p, division_quotient_unique_type(p),
        division_quotient_unique_body(p, core, decomposition, bound))
    var remainder = _declare_definition(p, division_remainder_unique_type(p),
        division_remainder_unique_body(p, core, decomposition, quotient))
    return NatArithmeticDivision(invariant, decomposition, bound, quotient, remainder)
