"""Untrusted proof producer for the strided finishing driver.

Every result below is an explicit formula with a derivation. The kernel checks
the producer's output independently, including substitutions and assumptions.
The fixed storage capacity bounds proof complexity, not runtime input counts or
the number of threads/items in a computation.
"""
import llm_mojo.kernels.finishing_proof_kernel as pk

comptime FINISHING_PROOF_SIZE = 192


def _c[size: Int](mut p: pk.FinishingProof[size], value: Int) -> Int:
    return pk.add_term(p, pk.TERM_CONST, value=value)


def _v[size: Int](mut p: pk.FinishingProof[size], variable: Int) -> Int:
    return pk.add_term(p, pk.TERM_VAR, value=variable)


def _add[size: Int](mut p: pk.FinishingProof[size], a: Int, b: Int) -> Int:
    return pk.add_term(p, pk.TERM_ADD, a, b)


def _scale[size: Int](mut p: pk.FinishingProof[size], factor: Int, a: Int) -> Int:
    return pk.add_term(p, pk.TERM_SCALE, a, value=factor)


def _base[size: Int](mut p: pk.FinishingProof[size], q: Int) -> Int:
    return pk.add_term(p, pk.TERM_BASE, q)


def _rel[size: Int](mut p: pk.FinishingProof[size], kind: Int, a: Int, b: Int) -> Int:
    return pk.add_formula(p, kind, a, b)


def _norm[size: Int](mut p: pk.FinishingProof[size], a: Int, b: Int) -> Int:
    return pk.add_node(p, pk.RULE_EQ_NORMALIZE, _rel(p, pk.FORM_EQ, a, b))


def _sym[size: Int](mut p: pk.FinishingProof[size], a: Int) -> Int:
    var f = p.formulas[p.nodes[a].claim]
    return pk.add_node(p, pk.RULE_EQ_SYMM, _rel(p, pk.FORM_EQ, f.right, f.left), a)


def _chain[size: Int](mut p: pk.FinishingProof[size], a: Int, b: Int) -> Int:
    var f = p.formulas[p.nodes[a].claim]
    var g = p.formulas[p.nodes[b].claim]
    var kind = pk.FORM_EQ
    var rule = pk.RULE_EQ_TRANS
    if f.kind == pk.FORM_EQ and g.kind == pk.FORM_LE:
        kind = pk.FORM_LE
        rule = pk.RULE_EQ_LE_TRANS
    elif f.kind == pk.FORM_EQ and g.kind == pk.FORM_LT:
        kind = pk.FORM_LT
        rule = pk.RULE_EQ_LT_TRANS
    elif f.kind == pk.FORM_LE and g.kind == pk.FORM_EQ:
        kind = pk.FORM_LE
        rule = pk.RULE_LE_EQ_TRANS
    elif f.kind == pk.FORM_LT and g.kind == pk.FORM_EQ:
        kind = pk.FORM_LT
        rule = pk.RULE_LT_EQ_TRANS
    elif f.kind == pk.FORM_LE and g.kind == pk.FORM_LE:
        kind = pk.FORM_LE
        rule = pk.RULE_LE_TRANS
    elif f.kind == pk.FORM_LT and g.kind == pk.FORM_LT:
        kind = pk.FORM_LT
        rule = pk.RULE_LT_TRANS
    elif f.kind == pk.FORM_LE and g.kind == pk.FORM_LT:
        kind = pk.FORM_LT
        rule = pk.RULE_LE_LT_TRANS
    elif f.kind == pk.FORM_LT and g.kind == pk.FORM_LE:
        kind = pk.FORM_LT
        rule = pk.RULE_LT_LE_TRANS
    return pk.add_node(p, rule, _rel(p, kind, f.left, g.right), a, b)


def _cong[size: Int](mut p: pk.FinishingProof[size], equality: Int, template: Int) -> Int:
    var f = p.formulas[p.nodes[equality].claim]
    var a = pk.substitute_term(p, template, 12, f.left)
    var b = pk.substitute_term(p, template, 12, f.right)
    return pk.add_node(p, pk.RULE_EQ_CONGRUENCE, _rel(p, pk.FORM_EQ, a, b),
        equality, template=template, variable=12)


def _plus[size: Int](mut p: pk.FinishingProof[size], order: Int, term: Int) -> Int:
    var f = p.formulas[p.nodes[order].claim]
    var rule = pk.RULE_LE_ADD if f.kind == pk.FORM_LE else pk.RULE_LT_ADD
    return pk.add_node(p, rule, _rel(p, f.kind, _add(p, f.left, term), _add(p, f.right, term)),
        order, term=term)


def _nonneg[size: Int](mut p: pk.FinishingProof[size], term: Int) -> Int:
    return pk.add_node(p, pk.RULE_NAT_NONNEG, _rel(p, pk.FORM_LE, _c(p, 0), term))


def _and[size: Int](mut p: pk.FinishingProof[size], a: Int, b: Int) -> Int:
    return pk.add_node(p, pk.RULE_AND_INTRO,
        _rel(p, pk.FORM_AND, p.nodes[a].claim, p.nodes[b].claim), a, b)


def _part[size: Int](mut p: pk.FinishingProof[size], a: Int, right: Bool = False) -> Int:
    var f = p.formulas[p.nodes[a].claim]
    return pk.add_node(p, pk.RULE_AND_RIGHT if right else pk.RULE_AND_LEFT,
        f.right if right else f.left, a)


def _imp[size: Int](mut p: pk.FinishingProof[size], body: Int, assumption: Int) -> Int:
    return pk.add_node(p, pk.RULE_IMP_INTRO,
        _rel(p, pk.FORM_IMP, p.nodes[assumption].claim, p.nodes[body].claim),
        body, assumption=assumption)


def _apply[size: Int](mut p: pk.FinishingProof[size], implication: Int, argument: Int) -> Int:
    return pk.add_node(p, pk.RULE_IMP_ELIM,
        p.formulas[p.nodes[implication].claim].right, implication, argument)


def _all[size: Int](mut p: pk.FinishingProof[size], body: Int, variable: Int) -> Int:
    return pk.add_node(p, pk.RULE_FORALL_INTRO,
        pk.add_formula(p, pk.FORM_FORALL, p.nodes[body].claim, variable=variable),
        body, variable=variable)


def _at[size: Int](mut p: pk.FinishingProof[size], universal: Int, term: Int) -> Int:
    var f = p.formulas[p.nodes[universal].claim]
    return pk.add_node(p, pk.RULE_FORALL_ELIM,
        pk.substitute_formula(p, f.left, f.variable, term), universal, term=term)


def _owner[size: Int](mut p: pk.FinishingProof[size], program: pk.FinishingProgram,
    q: Int, t: Int, i: Int, position: Int, count: Int) -> Int:
    var cq = _v(p, 6)
    var ct = _v(p, 7)
    var ci = _v(p, 8)
    var same = _rel(p, pk.FORM_AND, _rel(p, pk.FORM_EQ, cq, q),
        _rel(p, pk.FORM_AND, _rel(p, pk.FORM_EQ, ct, t), _rel(p, pk.FORM_EQ, ci, i)))
    var unique = _rel(p, pk.FORM_IMP, pk.add_event(p, program, cq, ct, ci, position, count), same)
    unique = pk.add_formula(p, pk.FORM_FORALL, unique, variable=8)
    unique = pk.add_formula(p, pk.FORM_FORALL, unique, variable=7)
    unique = pk.add_formula(p, pk.FORM_FORALL, unique, variable=6)
    return _rel(p, pk.FORM_AND, pk.add_event(p, program, q, t, i, position, count), unique)


def _termination[size: Int](mut p: pk.FinishingProof[size], program: pk.FinishingProgram,
    base_linear: Int, count: Int) -> Int:
    """Derive a finite execution using natural induction on a symbolic budget.

    M(k) = forall q, N <= S*(q+k) => Done(q,N).
    Done is generated only by an inactive guard or one returning iteration
    followed by a finite suffix. No rule accepts whole-loop termination.
    The budget k=N is a proof witness, not the GPU's iteration count.
    """
    var zero = _c(p, 0)
    var one = _c(p, 1)
    var q = _v(p, 2)
    var fuel = _v(p, 9)
    var next_q = _add(p, q, one)
    var next_fuel = _add(p, fuel, one)
    var done = _rel(p, pk.FORM_DONE, q, count)
    var bound = _rel(p, pk.FORM_LE, count, _scale(p, program.stride, _add(p, q, fuel)))
    var predicate = pk.add_formula(p, pk.FORM_FORALL,
        _rel(p, pk.FORM_IMP, bound, done), variable=2)

    # Zero budget: the assumed bound is precisely the false-loop-guard case.
    var zero_bound = pk.add_node(p, pk.RULE_ASSUME,
        _rel(p, pk.FORM_LE, count, _scale(p, program.stride, _add(p, q, zero))))
    var exit_guard = _chain(p, zero_bound, _norm(p,
        _scale(p, program.stride, _add(p, q, zero)), _scale(p, program.stride, q)))
    exit_guard = _chain(p, exit_guard, _sym(p, _at(p, base_linear, q)))
    var immediate = pk.add_node(p, pk.RULE_DONE_STOP, done, exit_guard)
    var base_case = _all(p, _imp(p, immediate, zero_bound), 2)

    # Successor budget: split the current guard. The active branch consumes
    # one iteration and applies the induction hypothesis at the next round.
    var ih = pk.add_node(p, pk.RULE_ASSUME, predicate)
    var successor_bound = pk.add_node(p, pk.RULE_ASSUME,
        _rel(p, pk.FORM_LE, count, _scale(p, program.stride, _add(p, q, next_fuel))))
    var active = pk.add_node(p, pk.RULE_ASSUME, _rel(p, pk.FORM_LT, _base(p, q), count))
    var suffix_bound = _chain(p, successor_bound, _norm(p,
        _scale(p, program.stride, _add(p, q, next_fuel)),
        _scale(p, program.stride, _add(p, next_q, fuel))))
    var suffix = _apply(p, _at(p, ih, next_q), suffix_bound)
    var take_step = pk.add_node(p, pk.RULE_DONE_STEP, done, active, suffix)
    var active_case = _imp(p, take_step, active)

    var inactive = pk.add_node(p, pk.RULE_ASSUME, _rel(p, pk.FORM_LE, count, _base(p, q)))
    var stop = pk.add_node(p, pk.RULE_DONE_STOP, done, inactive)
    var inactive_case = _imp(p, stop, inactive)
    var cases = pk.add_node(p, pk.RULE_NAT_ORDER_CASES, done, active_case, inactive_case)
    var successor_case = _all(p, _imp(p, cases, successor_bound), 2)
    var universal_step = _all(p, _imp(p, successor_case, ih), 9)
    var induction = pk.add_node(p, pk.RULE_NAT_INDUCTION,
        pk.add_formula(p, pk.FORM_FORALL, predicate, variable=9),
        base_case, universal_step, term=next_fuel, variable=9)

    # S>=1 makes the symbolic budget N sufficient from round zero. Reflect
    # only an affine identity; nonnegativity supplies the required inequality.
    var excess_scale = program.stride - 1 if program.stride >= 1 else 0
    var excess = _scale(p, excess_scale, count)
    var sufficient = _plus(p, _nonneg(p, excess), count)
    sufficient = _chain(p, _norm(p, count, _add(p, zero, count)), sufficient)
    sufficient = _chain(p, sufficient, _norm(p, _add(p, excess, count),
        _scale(p, program.stride, _add(p, zero, count))))
    return _apply(p, _at(p, _at(p, induction, count), zero), sufficient)


def make_finishing_proof(program: pk.FinishingProgram,
    include_termination: Bool = True) -> pk.FinishingProof[FINISHING_PROOF_SIZE]:
    var p = pk.empty_finishing_proof[FINISHING_PROOF_SIZE]()
    var zero = _c(p, 0)
    var one = _c(p, 1)
    var stride = _c(p, program.stride)
    var count = _v(p, 0)
    var position = _v(p, 1)
    var q = _v(p, 2)
    var successor = _add(p, q, one)
    var hole = _v(p, 12)

    # Induction derives Base(q) = S*q from only the driver's two equations.
    var base_zero = pk.add_node(p, pk.RULE_BASE_ZERO,
        _rel(p, pk.FORM_EQ, _base(p, zero), _c(p, program.start)))
    var base_case = _chain(p, base_zero, _norm(p, _c(p, program.start), _scale(p, program.stride, zero)))
    var base_predicate = _rel(p, pk.FORM_EQ, _base(p, q), _scale(p, program.stride, q))
    var base_ih = pk.add_node(p, pk.RULE_ASSUME, base_predicate)
    var base_step = pk.add_node(p, pk.RULE_BASE_STEP,
        _rel(p, pk.FORM_EQ, _base(p, successor), _add(p, _base(p, q), stride)), term=q)
    base_step = _chain(p, base_step, _cong(p, base_ih, _add(p, hole, stride)))
    base_step = _chain(p, base_step, _norm(p,
        _add(p, _scale(p, program.stride, q), stride), _scale(p, program.stride, successor)))
    var base_universal_step = _all(p, _imp(p, base_step, base_ih), 2)
    var base_linear = pk.add_node(p, pk.RULE_NAT_INDUCTION,
        pk.add_formula(p, pk.FORM_FORALL, base_predicate, variable=2),
        base_case, base_universal_step, term=successor, variable=2)

    # A second induction proves that every round below N is actually visited.
    var guard_zero = _rel(p, pk.FORM_LT, _scale(p, program.stride, zero), count)
    var guard_zero_assumption = pk.add_node(p, pk.RULE_ASSUME, guard_zero)
    var real_guard_zero = _chain(p, _at(p, base_linear, zero), guard_zero_assumption)
    var visit_zero = pk.add_node(p, pk.RULE_VISIT_ZERO,
        _rel(p, pk.FORM_VISITED, zero, count), real_guard_zero)
    var visit_base_case = _imp(p, visit_zero, guard_zero_assumption)
    var visit_predicate = _rel(p, pk.FORM_IMP,
        _rel(p, pk.FORM_LT, _scale(p, program.stride, q), count),
        _rel(p, pk.FORM_VISITED, q, count))
    var visit_ih = pk.add_node(p, pk.RULE_ASSUME, visit_predicate)
    var next_guard = pk.add_node(p, pk.RULE_ASSUME,
        _rel(p, pk.FORM_LT, _scale(p, program.stride, successor), count))
    var q_stride = _scale(p, program.stride, q)
    var previous_le = _plus(p, _nonneg(p, stride), q_stride)
    previous_le = _chain(p, _norm(p, q_stride, _add(p, zero, q_stride)), previous_le)
    previous_le = _chain(p, previous_le,
        _norm(p, _add(p, stride, q_stride), _scale(p, program.stride, successor)))
    var previous_guard = _chain(p, previous_le, next_guard)
    var previous_visit = _apply(p, visit_ih, previous_guard)
    var real_next_guard = _chain(p, _at(p, base_linear, successor), next_guard)
    var visit_step = pk.add_node(p, pk.RULE_VISIT_STEP,
        _rel(p, pk.FORM_VISITED, successor, count), previous_visit, real_next_guard)
    var visit_universal_step = _all(p, _imp(p, _imp(p, visit_step, next_guard), visit_ih), 2)
    var visited_below = pk.add_node(p, pk.RULE_NAT_INDUCTION,
        pk.add_formula(p, pk.FORM_FORALL, visit_predicate, variable=2),
        visit_base_case, visit_universal_step, term=successor, variable=2)

    var domain = pk.add_node(p, pk.RULE_ASSUME, pk.add_domain(p, program, count))
    var in_range = pk.add_node(p, pk.RULE_ASSUME, _rel(p, pk.FORM_LT, position, count))
    var qw = pk.add_term(p, pk.TERM_DIV, position, value=program.stride)
    var rw = pk.add_term(p, pk.TERM_MOD, position, value=program.stride)
    var iw = pk.add_term(p, pk.TERM_DIV, rw, value=program.threads)
    var tw = pk.add_term(p, pk.TERM_MOD, rw, value=program.threads)
    var outer = pk.add_node(p, pk.RULE_DIV_DECOMP,
        _rel(p, pk.FORM_EQ, position, _add(p, _scale(p, program.stride, qw), rw)),
        term=position, value=program.stride)
    var inner = pk.add_node(p, pk.RULE_DIV_DECOMP,
        _rel(p, pk.FORM_EQ, rw, _add(p, _scale(p, program.threads, iw), tw)),
        term=rw, value=program.threads)
    var remainder_bound = pk.add_node(p, pk.RULE_MOD_LT,
        _rel(p, pk.FORM_LT, rw, stride), term=position, value=program.stride)
    var thread_bound = pk.add_node(p, pk.RULE_MOD_LT,
        _rel(p, pk.FORM_LT, tw, _c(p, program.threads)), term=rw, value=program.threads)

    # Nonnegative remainder plus r<S gives i<I through positive-scale cancellation.
    var ti = _scale(p, program.threads, iw)
    var item_le = _plus(p, _nonneg(p, tw), ti)
    item_le = _chain(p, _norm(p, ti, _add(p, zero, ti)), item_le)
    item_le = _chain(p, item_le, _norm(p, _add(p, tw, ti), _add(p, ti, tw)))
    item_le = _chain(p, item_le, _sym(p, inner))
    var item_scaled_bound = _chain(p, item_le, remainder_bound)
    item_scaled_bound = _chain(p, item_scaled_bound,
        _norm(p, stride, _scale(p, program.threads, _c(p, program.items))))
    var item_bound = pk.add_node(p, pk.RULE_LT_SCALE_CANCEL,
        _rel(p, pk.FORM_LT, iw, _c(p, program.items)), item_scaled_bound, value=program.threads)

    var sq = _scale(p, program.stride, qw)
    var round_le = _plus(p, _nonneg(p, rw), sq)
    round_le = _chain(p, _norm(p, sq, _add(p, zero, sq)), round_le)
    round_le = _chain(p, round_le, _norm(p, _add(p, rw, sq), _add(p, sq, rw)))
    round_le = _chain(p, round_le, _sym(p, outer))
    var round_guard = _chain(p, round_le, in_range)
    var witness_visit = _apply(p, _at(p, visited_below, qw), round_guard)

    var reconstructed = _add(p, sq, _add(p, ti, tw))
    var position_reconstructed = _chain(p, outer, _cong(p, inner, _add(p, sq, hole)))
    var nominal_index = pk.add_index(p, program, sq, tw, iw)
    var index_equality = _cong(p, _at(p, base_linear, qw), pk.add_index(p, program, hole, tw, iw))
    index_equality = _chain(p, index_equality, _norm(p, nominal_index, reconstructed))
    index_equality = _chain(p, index_equality, _sym(p, position_reconstructed))
    var witness_event = _and(p, witness_visit, _and(p, thread_bound, _and(p, item_bound, index_equality)))

    # Any other bounded owner of p has the same two Euclidean decompositions.
    var cq = _v(p, 6)
    var ct = _v(p, 7)
    var ci = _v(p, 8)
    var candidate = pk.add_node(p, pk.RULE_ASSUME, pk.add_event(p, program, cq, ct, ci, position, count))
    var candidate_tail = _part(p, candidate, True)
    var ct_bound = _part(p, candidate_tail)
    candidate_tail = _part(p, candidate_tail, True)
    var ci_bound = _part(p, candidate_tail)
    var candidate_index = _part(p, candidate_tail, True)
    var offset = _add(p, _scale(p, program.threads, ci), ct)
    var ci_successor = pk.add_node(p, pk.RULE_LT_TO_LE_SUCC,
        _rel(p, pk.FORM_LE, _add(p, ci, one), _c(p, program.items)), ci_bound)
    var scaled_successor = pk.add_node(p, pk.RULE_LE_SCALE,
        _rel(p, pk.FORM_LE, _scale(p, program.threads, _add(p, ci, one)),
            _scale(p, program.threads, _c(p, program.items))), ci_successor, value=program.threads)
    var offset_bound = _plus(p, ct_bound, _scale(p, program.threads, ci))
    offset_bound = _chain(p, _norm(p, offset, _add(p, ct, _scale(p, program.threads, ci))), offset_bound)
    offset_bound = _chain(p, offset_bound, _norm(p,
        _add(p, _c(p, program.threads), _scale(p, program.threads, ci)),
        _scale(p, program.threads, _add(p, ci, one))))
    offset_bound = _chain(p, offset_bound, scaled_successor)
    offset_bound = _chain(p, offset_bound,
        _norm(p, _scale(p, program.threads, _c(p, program.items)), stride))
    var candidate_decomp = _cong(p, _at(p, base_linear, cq), pk.add_index(p, program, hole, ct, ci))
    candidate_decomp = _chain(p, candidate_decomp, _norm(p,
        pk.add_index(p, program, _scale(p, program.stride, cq), ct, ci),
        _add(p, _scale(p, program.stride, cq), offset)))
    candidate_decomp = _chain(p, _sym(p, candidate_index), candidate_decomp)
    var outer_unique = pk.add_node(p, pk.RULE_DIV_UNIQUE,
        _rel(p, pk.FORM_AND, _rel(p, pk.FORM_EQ, cq, qw), _rel(p, pk.FORM_EQ, offset, rw)),
        candidate_decomp, offset_bound, term=position, template=cq, witness=offset, value=program.stride)
    var inner_unique = pk.add_node(p, pk.RULE_DIV_UNIQUE,
        _rel(p, pk.FORM_AND, _rel(p, pk.FORM_EQ, ci, iw), _rel(p, pk.FORM_EQ, ct, tw)),
        _sym(p, _part(p, outer_unique, True)), ct_bound,
        term=rw, template=ci, witness=ct, value=program.threads)
    var coordinates = _and(p, _part(p, outer_unique),
        _and(p, _part(p, inner_unique, True), _part(p, inner_unique)))
    var unique = _all(p, _all(p, _all(p, _imp(p, coordinates, candidate), 8), 7), 6)
    var owner = _and(p, witness_event, unique)

    var item_body = _owner(p, program, qw, tw, _v(p, 5), position, count)
    var exists_item = pk.add_formula(p, pk.FORM_EXISTS, item_body, variable=5)
    owner = pk.add_node(p, pk.RULE_EXISTS_INTRO, exists_item, owner, term=iw)
    var thread_body = pk.add_formula(p, pk.FORM_EXISTS,
        _owner(p, program, qw, _v(p, 4), _v(p, 5), position, count), variable=5)
    var exists_thread = pk.add_formula(p, pk.FORM_EXISTS, thread_body, variable=4)
    owner = pk.add_node(p, pk.RULE_EXISTS_INTRO, exists_thread, owner, term=tw)
    var round_body = pk.add_formula(p, pk.FORM_EXISTS, pk.add_formula(p, pk.FORM_EXISTS,
        _owner(p, program, _v(p, 3), _v(p, 4), _v(p, 5), position, count), variable=5), variable=4)
    var exists_round = pk.add_formula(p, pk.FORM_EXISTS, round_body, variable=3)
    owner = pk.add_node(p, pk.RULE_EXISTS_INTRO, exists_round, owner, term=qw)
    var coverage = _all(p, _imp(p, owner, in_range), 1)

    var terminates = -1
    if include_termination:
        terminates = _termination(p, program, base_linear, count)
    var count_cap = _part(p, domain, True)
    var headroom = _plus(p, count_cap, stride)
    var cap_plus_stride = _add(p, _c(p, program.count_limit), stride)
    var cap_headroom = pk.add_node(p, pk.RULE_LE_CONST,
        _rel(p, pk.FORM_LE, cap_plus_stride, _c(p, program.integer_limit)))
    headroom = _chain(p, headroom, cap_headroom)
    var remaining = headroom
    if include_termination:
        remaining = _and(p, terminates, headroom)
    var conclusion = _and(p, coverage, remaining)
    p.root = _all(p, _imp(p, conclusion, domain), 0)
    return p^


def make_finishing_coverage_proof(program: pk.FinishingProgram) -> pk.FinishingProof[FINISHING_PROOF_SIZE]:
    """Historical coverage/headroom comparison, without legacy Done steps."""
    return make_finishing_proof(program, include_termination=False)
