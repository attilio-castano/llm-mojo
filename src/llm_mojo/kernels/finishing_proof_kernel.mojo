"""Small typed proof calculus for natural arithmetic and the finishing driver.

Terms and formulas form acyclic DAGs. Proof rules manipulate their actual
operands and exact open-assumption sets. Coverage and visitation are ordinary
derived formulas; neither is an inference rule. Arithmetic normalization,
Euclidean division, logical rules, natural induction, and the prescribed
driver's zero/successor semantics constitute the trusted rulebook.

The historical coverage/headroom profile forbids the legacy Done predicate and
its constructors. They remain only in the full comparison profile described
below. Production coverage and termination use the proof-term kernel instead.

Done(q,N) is an inductive finite-execution judgment for the driver starting at
the potential state Base(q). STOP witnesses zero body executions when the
guard is false. STEP prepends one returning body to an already finite suffix.
There is no rule that assumes termination or well-founded loop descent.
"""
from std.collections import Array

comptime FINISH_LT = 0
comptime FINISHING_TERM_CAPACITY = 128
comptime FINISHING_FORMULA_CAPACITY = 256
comptime FINISHING_VARIABLE_CAPACITY = 32
comptime NORMAL_WIDTH = FINISHING_TERM_CAPACITY + FINISHING_VARIABLE_CAPACITY + 1

comptime TERM_CONST = 0
comptime TERM_VAR = 1
comptime TERM_ADD = 2
comptime TERM_SCALE = 3
comptime TERM_DIV = 4
comptime TERM_MOD = 5
comptime TERM_BASE = 6
comptime FORM_EQ = 0
comptime FORM_LT = 1
comptime FORM_LE = 2
comptime FORM_AND = 3
comptime FORM_IMP = 4
comptime FORM_FORALL = 5
comptime FORM_EXISTS = 6
comptime FORM_VISITED = 7
comptime FORM_DONE = 9

comptime RULE_ASSUME = 100
comptime RULE_EQ_NORMALIZE = 101
comptime RULE_EQ_SYMM = 102
comptime RULE_EQ_TRANS = 103
comptime RULE_EQ_CONGRUENCE = 104
comptime RULE_LT_CONST = 105
comptime RULE_LE_CONST = 106
comptime RULE_NAT_NONNEG = 107
comptime RULE_LE_ADD = 108
comptime RULE_LT_ADD = 109
comptime RULE_LE_SCALE = 110
comptime RULE_LT_SCALE = 111
comptime RULE_LE_TRANS = 112
comptime RULE_LT_TRANS = 113
comptime RULE_LE_LT_TRANS = 114
comptime RULE_LT_LE_TRANS = 115
comptime RULE_LE_EQ_TRANS = 116
comptime RULE_LT_EQ_TRANS = 117
comptime RULE_EQ_LE_TRANS = 118
comptime RULE_EQ_LT_TRANS = 119
comptime RULE_LT_TO_LE_SUCC = 120
comptime RULE_LE_SCALE_CANCEL = 121
comptime RULE_LT_SCALE_CANCEL = 122
comptime RULE_AND_INTRO = 123
comptime RULE_AND_LEFT = 124
comptime RULE_AND_RIGHT = 125
comptime RULE_IMP_INTRO = 126
comptime RULE_IMP_ELIM = 127
comptime RULE_FORALL_INTRO = 128
comptime RULE_FORALL_ELIM = 129
comptime RULE_EXISTS_INTRO = 130
comptime RULE_NAT_INDUCTION = 131
comptime RULE_DIV_DECOMP = 132
comptime RULE_MOD_LT = 133
comptime RULE_DIV_UNIQUE = 134
comptime RULE_BASE_ZERO = 135
comptime RULE_BASE_STEP = 136
comptime RULE_VISIT_ZERO = 137
comptime RULE_VISIT_STEP = 138
comptime RULE_VISIT_GUARD = 139
comptime RULE_DONE_STOP = 141
comptime RULE_DONE_STEP = 142
comptime RULE_NAT_ORDER_CASES = 143


@fieldwise_init
struct FinishingProgram(ImplicitlyCopyable):
    var threads: Int
    var items: Int
    var stride: Int
    var start: Int
    var thread_scale: Int
    var item_scale: Int
    var index_bias: Int
    var loop_guard: Int
    var load_guard: Int
    var count_limit: Int
    var integer_limit: Int


def make_finishing_program(
    threads: Int, items: Int, stride: Int,
    count_limit: Int = 2147483647, integer_limit: Int = Int.MAX,
) -> FinishingProgram:
    return FinishingProgram(threads, items, stride, 0, 1, threads, 0,
        FINISH_LT, FINISH_LT, count_limit, integer_limit)


@fieldwise_init
struct FinishingTerm(ImplicitlyCopyable):
    var kind: Int
    var left: Int
    var right: Int
    var value: Int


@fieldwise_init
struct FinishingFormula(ImplicitlyCopyable):
    var kind: Int
    var left: Int
    var right: Int
    var variable: Int


@fieldwise_init
struct FinishingProofNode(ImplicitlyCopyable):
    var rule: Int
    var premise0: Int
    var premise1: Int
    var premise2: Int
    var premise3: Int
    var claim: Int
    var term: Int
    var variable: Int
    var assumption: Int
    var template: Int
    var value: Int
    var witness: Int


@fieldwise_init
struct FinishingProof[size: Int](Copyable, Movable):
    var terms: Array[FinishingTerm, FINISHING_TERM_CAPACITY]
    var formulas: Array[FinishingFormula, FINISHING_FORMULA_CAPACITY]
    var nodes: Array[FinishingProofNode, Self.size]
    var term_count: Int
    var formula_count: Int
    var node_count: Int
    var root: Int


def empty_finishing_proof[size: Int]() -> FinishingProof[size]:
    var terms = Array[FinishingTerm, FINISHING_TERM_CAPACITY](fill=FinishingTerm(0, -1, -1, 0))
    var formulas = Array[FinishingFormula, FINISHING_FORMULA_CAPACITY](fill=FinishingFormula(0, 0, 0, -1))
    var nodes = Array[FinishingProofNode, size](fill=FinishingProofNode(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))
    return FinishingProof[size](terms^, formulas^, nodes^, 0, 0, 0, -1)


def add_term[size: Int](mut proof: FinishingProof[size],
    kind: Int, left: Int = -1, right: Int = -1, value: Int = -1) -> Int:
    if proof.term_count < 0 or proof.term_count > FINISHING_TERM_CAPACITY:
        return -1
    for i in range(proof.term_count):
        var t = proof.terms[i]
        if t.kind == kind and t.left == left and t.right == right and t.value == value:
            return i
    if proof.term_count >= FINISHING_TERM_CAPACITY:
        return -1
    var result = proof.term_count
    proof.terms[result] = FinishingTerm(kind, left, right, value)
    proof.term_count += 1
    return result


def add_formula[size: Int](mut proof: FinishingProof[size],
    kind: Int, left: Int = -1, right: Int = -1, variable: Int = -1) -> Int:
    if proof.formula_count < 0 or proof.formula_count > FINISHING_FORMULA_CAPACITY:
        return -1
    for i in range(proof.formula_count):
        var f = proof.formulas[i]
        if f.kind == kind and f.left == left and f.right == right and f.variable == variable:
            return i
    if proof.formula_count >= FINISHING_FORMULA_CAPACITY:
        return -1
    var result = proof.formula_count
    proof.formulas[result] = FinishingFormula(kind, left, right, variable)
    proof.formula_count += 1
    return result


def add_node[size: Int](mut proof: FinishingProof[size],
    rule: Int, claim: Int, p0: Int = -1, p1: Int = -1, p2: Int = -1, p3: Int = -1,
    term: Int = -1, variable: Int = -1, assumption: Int = -1,
    template: Int = -1, value: Int = -1, witness: Int = -1) -> Int:
    if proof.node_count < 0 or proof.node_count >= size:
        return -1
    var result = proof.node_count
    proof.nodes[result] = FinishingProofNode(rule, p0, p1, p2, p3,
        claim, term, variable, assumption, template, value, witness)
    proof.node_count += 1
    return result


def substitute_term[size: Int](mut proof: FinishingProof[size],
    reference: Int, variable: Int, replacement: Int) -> Int:
    if (reference < 0 or reference >= proof.term_count or replacement < 0 or replacement >= proof.term_count
        or variable < 0 or variable >= FINISHING_VARIABLE_CAPACITY):
        return -1
    var t = proof.terms[reference]
    if t.kind == TERM_VAR:
        return replacement if t.value == variable else reference
    if t.kind == TERM_CONST:
        return reference
    var left = substitute_term(proof, t.left, variable, replacement)
    var right = -1
    if t.kind == TERM_ADD:
        right = substitute_term(proof, t.right, variable, replacement)
    return add_term(proof, t.kind, left, right, t.value)


def _term_variables[size: Int](proof: FinishingProof[size], reference: Int) -> UInt32:
    if reference < 0 or reference >= proof.term_count:
        return 0
    var t = proof.terms[reference]
    if t.kind == TERM_VAR:
        if t.value < 0 or t.value >= FINISHING_VARIABLE_CAPACITY:
            return 0
        return UInt32(1) << UInt32(t.value)
    if t.kind == TERM_CONST:
        return 0
    var variables = _term_variables(proof, t.left)
    if t.kind == TERM_ADD:
        variables |= _term_variables(proof, t.right)
    return variables


def _formula_variables[size: Int](proof: FinishingProof[size], reference: Int) -> UInt32:
    if reference < 0 or reference >= proof.formula_count:
        return 0
    var f = proof.formulas[reference]
    if f.kind == FORM_AND or f.kind == FORM_IMP:
        return _formula_variables(proof, f.left) | _formula_variables(proof, f.right)
    if f.kind == FORM_FORALL or f.kind == FORM_EXISTS:
        if f.variable < 0 or f.variable >= FINISHING_VARIABLE_CAPACITY:
            return 0
        return _formula_variables(proof, f.left) & ~(UInt32(1) << UInt32(f.variable))
    var variables = _term_variables(proof, f.left)
    variables |= _term_variables(proof, f.right)
    return variables


def substitute_formula[size: Int](mut proof: FinishingProof[size],
    reference: Int, variable: Int, replacement: Int) -> Int:
    if (reference < 0 or reference >= proof.formula_count or replacement < 0 or replacement >= proof.term_count
        or variable < 0 or variable >= FINISHING_VARIABLE_CAPACITY):
        return -1
    var f = proof.formulas[reference]
    if f.kind == FORM_FORALL or f.kind == FORM_EXISTS:
        if f.variable == variable:
            return reference
        if ((_formula_variables(proof, f.left) & (UInt32(1) << UInt32(variable))) != 0
                and (_term_variables(proof, replacement) & (UInt32(1) << UInt32(f.variable))) != 0):
            return -1
        return add_formula(proof, f.kind,
            substitute_formula(proof, f.left, variable, replacement), variable=f.variable)
    if f.kind == FORM_AND or f.kind == FORM_IMP:
        return add_formula(proof, f.kind,
            substitute_formula(proof, f.left, variable, replacement),
            substitute_formula(proof, f.right, variable, replacement))
    var right = substitute_term(proof, f.right, variable, replacement)
    return add_formula(proof, f.kind,
        substitute_term(proof, f.left, variable, replacement), right)


def add_index[size: Int](mut proof: FinishingProof[size], program: FinishingProgram,
    base: Int, thread: Int, item: Int) -> Int:
    var t = add_term(proof, TERM_SCALE, thread, value=program.thread_scale)
    var i = add_term(proof, TERM_SCALE, item, value=program.item_scale)
    var bias = add_term(proof, TERM_CONST, value=program.index_bias)
    return add_term(proof, TERM_ADD, add_term(proof, TERM_ADD,
        add_term(proof, TERM_ADD, base, t), i), bias)


def add_event[size: Int](mut proof: FinishingProof[size], program: FinishingProgram,
    round: Int, thread: Int, item: Int, position: Int, count: Int) -> Int:
    var visited = add_formula(proof, FORM_VISITED, round, count)
    var thread_bound = add_formula(proof, FORM_LT, thread, add_term(proof, TERM_CONST, value=program.threads))
    var item_bound = add_formula(proof, FORM_LT, item, add_term(proof, TERM_CONST, value=program.items))
    var index = add_index(proof, program, add_term(proof, TERM_BASE, round), thread, item)
    var equality = add_formula(proof, FORM_EQ, index, position)
    return add_formula(proof, FORM_AND, visited,
        add_formula(proof, FORM_AND, thread_bound, add_formula(proof, FORM_AND, item_bound, equality)))


def add_domain[size: Int](mut proof: FinishingProof[size], program: FinishingProgram, count: Int) -> Int:
    return add_formula(proof, FORM_AND,
        add_formula(proof, FORM_LE, add_term(proof, TERM_CONST, value=1), count),
        add_formula(proof, FORM_LE, count, add_term(proof, TERM_CONST, value=program.count_limit)))


def add_finishing_goal[size: Int](mut proof: FinishingProof[size], program: FinishingProgram,
    include_termination: Bool = True) -> Int:
    """Literal quantified target; constructing it provides no proof of it."""
    var count = add_term(proof, TERM_VAR, value=0)
    var position = add_term(proof, TERM_VAR, value=1)
    var round = add_term(proof, TERM_VAR, value=3)
    var thread = add_term(proof, TERM_VAR, value=4)
    var item = add_term(proof, TERM_VAR, value=5)
    var other_round = add_term(proof, TERM_VAR, value=6)
    var other_thread = add_term(proof, TERM_VAR, value=7)
    var other_item = add_term(proof, TERM_VAR, value=8)
    var same = add_formula(proof, FORM_AND, add_formula(proof, FORM_EQ, other_round, round),
        add_formula(proof, FORM_AND, add_formula(proof, FORM_EQ, other_thread, thread),
            add_formula(proof, FORM_EQ, other_item, item)))
    var unique = add_formula(proof, FORM_IMP,
        add_event(proof, program, other_round, other_thread, other_item, position, count), same)
    unique = add_formula(proof, FORM_FORALL, unique, variable=8)
    unique = add_formula(proof, FORM_FORALL, unique, variable=7)
    unique = add_formula(proof, FORM_FORALL, unique, variable=6)
    var witness = add_formula(proof, FORM_AND,
        add_event(proof, program, round, thread, item, position, count), unique)
    witness = add_formula(proof, FORM_EXISTS, witness, variable=5)
    witness = add_formula(proof, FORM_EXISTS, witness, variable=4)
    witness = add_formula(proof, FORM_EXISTS, witness, variable=3)
    var coverage = add_formula(proof, FORM_FORALL,
        add_formula(proof, FORM_IMP, add_formula(proof, FORM_LT, position, count), witness), variable=1)
    var headroom = add_formula(proof, FORM_LE,
        add_term(proof, TERM_ADD, count, add_term(proof, TERM_CONST, value=program.stride)),
        add_term(proof, TERM_CONST, value=program.integer_limit))
    var remaining = headroom
    if include_termination:
        remaining = add_formula(proof, FORM_AND,
            add_formula(proof, FORM_DONE, add_term(proof, TERM_CONST, value=0), count), headroom)
    var conclusion = add_formula(proof, FORM_AND, coverage, remaining)
    return add_formula(proof, FORM_FORALL,
        add_formula(proof, FORM_IMP, add_domain(proof, program, count), conclusion), variable=0)


def _same_term[a_size: Int, b_size: Int](a: FinishingProof[a_size], ar: Int,
    b: FinishingProof[b_size], br: Int) -> Bool:
    var x = a.terms[ar]
    var y = b.terms[br]
    if x.kind != y.kind or x.value != y.value:
        return False
    if x.kind == TERM_CONST or x.kind == TERM_VAR:
        return True
    if not _same_term(a, x.left, b, y.left):
        return False
    return x.kind != TERM_ADD or _same_term(a, x.right, b, y.right)


def _same_formula[a_size: Int, b_size: Int](a: FinishingProof[a_size], ar: Int,
    b: FinishingProof[b_size], br: Int) -> Bool:
    var x = a.formulas[ar]
    var y = b.formulas[br]
    if x.kind != y.kind or x.variable != y.variable:
        return False
    if x.kind == FORM_AND or x.kind == FORM_IMP:
        return _same_formula(a, x.left, b, y.left) and _same_formula(a, x.right, b, y.right)
    if x.kind == FORM_FORALL or x.kind == FORM_EXISTS:
        return _same_formula(a, x.left, b, y.left)
    return _same_term(a, x.left, b, y.left) and _same_term(a, x.right, b, y.right)


def _term_substitution_matches[size: Int](proof: FinishingProof[size], pattern: Int,
    variable: Int, replacement: Int, actual: Int) -> Bool:
    var p = proof.terms[pattern]
    var a = proof.terms[actual]
    if p.kind == TERM_VAR and p.value == variable:
        return _same_term(proof, replacement, proof, actual)
    if p.kind != a.kind or p.value != a.value:
        return False
    if p.kind == TERM_CONST or p.kind == TERM_VAR:
        return True
    return (_term_substitution_matches(proof, p.left, variable, replacement, a.left)
        and (p.kind != TERM_ADD
            or _term_substitution_matches(proof, p.right, variable, replacement, a.right)))


def _formula_substitution_matches[size: Int](proof: FinishingProof[size], pattern: Int,
    variable: Int, replacement: Int, actual: Int) -> Bool:
    var p = proof.formulas[pattern]
    var a = proof.formulas[actual]
    if p.kind != a.kind or p.variable != a.variable:
        return False
    if p.kind == FORM_FORALL or p.kind == FORM_EXISTS:
        if p.variable == variable:
            return _same_formula(proof, pattern, proof, actual)
        if ((_formula_variables(proof, p.left) & (UInt32(1) << UInt32(variable))) != 0
            and (_term_variables(proof, replacement) & (UInt32(1) << UInt32(p.variable))) != 0):
            return False
        return _formula_substitution_matches(proof, p.left, variable, replacement, a.left)
    if p.kind == FORM_AND or p.kind == FORM_IMP:
        return (_formula_substitution_matches(proof, p.left, variable, replacement, a.left)
            and _formula_substitution_matches(proof, p.right, variable, replacement, a.right))
    return (_term_substitution_matches(proof, p.left, variable, replacement, a.left)
        and _term_substitution_matches(proof, p.right, variable, replacement, a.right))


def _normal_equal(normal: Array[Int, FINISHING_TERM_CAPACITY * NORMAL_WIDTH],
    a: Int, b: Int) -> Bool:
    for k in range(NORMAL_WIDTH):
        if normal[a * NORMAL_WIDTH + k] != normal[b * NORMAL_WIDTH + k]:
            return False
    return True


def _normal_add(normal: Array[Int, FINISHING_TERM_CAPACITY * NORMAL_WIDTH],
    result: Int, a: Int, b: Int) -> Bool:
    for k in range(NORMAL_WIDTH):
        var x = normal[a * NORMAL_WIDTH + k]
        var y = normal[b * NORMAL_WIDTH + k]
        if x > Int.MAX - y or normal[result * NORMAL_WIDTH + k] != x + y:
            return False
    return True


def _normal_scale(normal: Array[Int, FINISHING_TERM_CAPACITY * NORMAL_WIDTH],
    result: Int, a: Int, scale: Int) -> Bool:
    if scale < 0:
        return False
    for k in range(NORMAL_WIDTH):
        var x = normal[a * NORMAL_WIDTH + k]
        if scale > 0 and x > Int.MAX // scale:
            return False
        if normal[result * NORMAL_WIDTH + k] != x * scale:
            return False
    return True


def _normal_succ(normal: Array[Int, FINISHING_TERM_CAPACITY * NORMAL_WIDTH],
    result: Int, a: Int) -> Bool:
    if normal[a * NORMAL_WIDTH] == Int.MAX or normal[result * NORMAL_WIDTH] != normal[a * NORMAL_WIDTH] + 1:
        return False
    for k in range(1, NORMAL_WIDTH):
        if normal[result * NORMAL_WIDTH + k] != normal[a * NORMAL_WIDTH + k]:
            return False
    return True


def _constant_value(normal: Array[Int, FINISHING_TERM_CAPACITY * NORMAL_WIDTH], term: Int) -> Int:
    for k in range(1, NORMAL_WIDTH):
        if normal[term * NORMAL_WIDTH + k] != 0:
            return -1
    return normal[term * NORMAL_WIDTH]


def _find_term[size: Int](proof: FinishingProof[size], kind: Int,
    left: Int = -1, right: Int = -1, value: Int = -1) -> Int:
    for i in range(proof.term_count):
        var t = proof.terms[i]
        if t.kind == kind and t.value == value and t.left == left and t.right == right:
            return i
    return -1


def _arity(rule: Int) -> Int:
    if (rule == RULE_ASSUME or rule == RULE_EQ_NORMALIZE or rule == RULE_LT_CONST
        or rule == RULE_LE_CONST or rule == RULE_NAT_NONNEG or rule == RULE_DIV_DECOMP
        or rule == RULE_MOD_LT or rule == RULE_BASE_ZERO or rule == RULE_BASE_STEP):
        return 0
    if (rule == RULE_EQ_SYMM or rule == RULE_EQ_CONGRUENCE or rule == RULE_LE_ADD
        or rule == RULE_LT_ADD or rule == RULE_LE_SCALE or rule == RULE_LT_SCALE
        or rule == RULE_LT_TO_LE_SUCC or rule == RULE_LE_SCALE_CANCEL
        or rule == RULE_LT_SCALE_CANCEL or rule == RULE_AND_LEFT or rule == RULE_AND_RIGHT
        or rule == RULE_IMP_INTRO or rule == RULE_FORALL_INTRO or rule == RULE_FORALL_ELIM
        or rule == RULE_EXISTS_INTRO or rule == RULE_VISIT_ZERO or rule == RULE_VISIT_GUARD
        or rule == RULE_DONE_STOP):
        return 1
    if (rule == RULE_EQ_TRANS or rule == RULE_LE_TRANS or rule == RULE_LT_TRANS
        or rule == RULE_LE_LT_TRANS or rule == RULE_LT_LE_TRANS or rule == RULE_LE_EQ_TRANS
        or rule == RULE_LT_EQ_TRANS or rule == RULE_EQ_LE_TRANS or rule == RULE_EQ_LT_TRANS
        or rule == RULE_AND_INTRO or rule == RULE_IMP_ELIM or rule == RULE_NAT_INDUCTION
        or rule == RULE_DIV_UNIQUE or rule == RULE_VISIT_STEP
        or rule == RULE_DONE_STEP or rule == RULE_NAT_ORDER_CASES):
        return 2
    return -1


def _canonical_auxiliary(node: FinishingProofNode) -> Bool:
    var use_term = (node.rule == RULE_LE_ADD or node.rule == RULE_LT_ADD
        or node.rule == RULE_FORALL_ELIM or node.rule == RULE_EXISTS_INTRO
        or node.rule == RULE_NAT_INDUCTION or node.rule == RULE_DIV_DECOMP
        or node.rule == RULE_MOD_LT or node.rule == RULE_DIV_UNIQUE
        or node.rule == RULE_BASE_STEP)
    var use_variable = (node.rule == RULE_EQ_CONGRUENCE or node.rule == RULE_FORALL_INTRO
        or node.rule == RULE_NAT_INDUCTION)
    var use_template = node.rule == RULE_EQ_CONGRUENCE or node.rule == RULE_DIV_UNIQUE
    var use_value = (node.rule == RULE_LE_SCALE or node.rule == RULE_LT_SCALE
        or node.rule == RULE_LE_SCALE_CANCEL or node.rule == RULE_LT_SCALE_CANCEL
        or node.rule == RULE_DIV_DECOMP or node.rule == RULE_MOD_LT or node.rule == RULE_DIV_UNIQUE)
    return ((use_term or node.term == -1) and (use_variable or node.variable == -1)
        and (node.rule == RULE_IMP_INTRO or node.assumption == -1)
        and (use_template or node.template == -1) and (use_value or node.value == -1)
        and (node.rule == RULE_DIV_UNIQUE or node.witness == -1))


def check_finishing_proof[size: Int](program: FinishingProgram, proof: FinishingProof[size],
    include_termination: Bool = True) -> Bool:
    """Check syntax, infer every active proof step, and require the literal closed goal.

    No producer or proof-search code is trusted. Terms are Nat; coefficient
    arithmetic is checked before addition/multiplication. Exact assumption
    identities propagate through all rules and are discharged explicitly.
    """
    if (program.threads < 1 or program.items < 1 or program.threads > 2147483647 // program.items
        or program.stride != program.threads * program.items or program.start < 0
        or program.thread_scale < 0 or program.item_scale < 0 or program.index_bias < 0
        or program.loop_guard != FINISH_LT or program.load_guard != FINISH_LT
        or program.count_limit < 1 or program.integer_limit < program.count_limit):
        return False
    if (proof.term_count < 1 or proof.term_count > FINISHING_TERM_CAPACITY
        or proof.formula_count < 1 or proof.formula_count > FINISHING_FORMULA_CAPACITY
        or proof.node_count < 1 or proof.node_count > size
        or proof.root < 0 or proof.root >= proof.node_count):
        return False
    var normal = Array[Int, FINISHING_TERM_CAPACITY * NORMAL_WIDTH](fill=0)
    var term_variables = Array[UInt32, FINISHING_TERM_CAPACITY](fill=0)
    for i in range(proof.term_count):
        var t = proof.terms[i]
        if t.kind == TERM_CONST or t.kind == TERM_VAR:
            if t.left != -1 or t.right != -1 or t.value < 0:
                return False
            if t.kind == TERM_CONST:
                normal[i * NORMAL_WIDTH] = t.value
            else:
                if t.value >= FINISHING_VARIABLE_CAPACITY:
                    return False
                normal[i * NORMAL_WIDTH + 1 + t.value] = 1
                term_variables[i] = UInt32(1) << UInt32(t.value)
        elif t.kind == TERM_ADD:
            if t.left < 0 or t.left >= i or t.right < 0 or t.right >= i or t.value != -1:
                return False
            term_variables[i] = term_variables[t.left] | term_variables[t.right]
            for k in range(NORMAL_WIDTH):
                var x = normal[t.left * NORMAL_WIDTH + k]
                var y = normal[t.right * NORMAL_WIDTH + k]
                if x > Int.MAX - y:
                    return False
                normal[i * NORMAL_WIDTH + k] = x + y
        elif t.kind == TERM_SCALE or t.kind == TERM_DIV or t.kind == TERM_MOD or t.kind == TERM_BASE:
            if t.left < 0 or t.left >= i or t.right != -1:
                return False
            term_variables[i] = term_variables[t.left]
            if t.kind == TERM_SCALE:
                if t.value < 0:
                    return False
                for k in range(NORMAL_WIDTH):
                    var x = normal[t.left * NORMAL_WIDTH + k]
                    if t.value > 0 and x > Int.MAX // t.value:
                        return False
                    normal[i * NORMAL_WIDTH + k] = x * t.value
            else:
                if ((t.kind == TERM_BASE and t.value != -1)
                    or (t.kind != TERM_BASE and t.value <= 0)):
                    return False
                # Function terms are exact symbolic atoms. Identical duplicate
                # terms share an atom; distinct arguments require congruence.
                var atom = i
                for j in range(i):
                    var prior = proof.terms[j]
                    if prior.kind == t.kind and prior.value == t.value and _same_term(proof, prior.left, proof, t.left):
                        atom = j
                        break
                normal[i * NORMAL_WIDTH + FINISHING_VARIABLE_CAPACITY + 1 + atom] = 1
        else:
            return False
    var formula_variables = Array[UInt32, FINISHING_FORMULA_CAPACITY](fill=0)
    for i in range(proof.formula_count):
        var f = proof.formulas[i]
        if not include_termination and f.kind == FORM_DONE:
            return False
        if f.kind == FORM_AND or f.kind == FORM_IMP:
            if f.left < 0 or f.left >= i or f.right < 0 or f.right >= i or f.variable != -1:
                return False
            formula_variables[i] = formula_variables[f.left] | formula_variables[f.right]
        elif f.kind == FORM_FORALL or f.kind == FORM_EXISTS:
            if (f.left < 0 or f.left >= i or f.right != -1
                or f.variable < 0 or f.variable >= FINISHING_VARIABLE_CAPACITY):
                return False
            formula_variables[i] = formula_variables[f.left] & ~(UInt32(1) << UInt32(f.variable))
        elif (f.kind == FORM_EQ or f.kind == FORM_LT or f.kind == FORM_LE
            or f.kind == FORM_VISITED or f.kind == FORM_DONE):
            if f.left < 0 or f.left >= proof.term_count or f.variable != -1:
                return False
            formula_variables[i] = term_variables[f.left]
            if f.right < 0 or f.right >= proof.term_count:
                return False
            formula_variables[i] |= term_variables[f.right]
        else:
            return False
    var contexts = Array[Bool, size * size](fill=False)
    for i in range(proof.node_count):
        var node = proof.nodes[i]
        if not include_termination and (node.rule == RULE_DONE_STOP or node.rule == RULE_DONE_STEP):
            return False
        var arity = _arity(node.rule)
        if arity < 0 or node.claim < 0 or node.claim >= proof.formula_count or not _canonical_auxiliary(node):
            return False
        var references: Array[Int, 4] = [node.premise0, node.premise1, node.premise2, node.premise3]
        for j in range(4):
            var prior = references[j]
            if j < arity:
                if prior < 0 or prior >= i:
                    return False
                for k in range(i):
                    contexts[i * size + k] = contexts[i * size + k] or contexts[prior * size + k]
            elif prior != -1:
                return False
        var f = proof.formulas[node.claim]
        var a = FinishingFormula(-1, -1, -1, -1)
        var b = FinishingFormula(-1, -1, -1, -1)
        if arity >= 1:
            a = proof.formulas[proof.nodes[node.premise0].claim]
        if arity >= 2:
            b = proof.formulas[proof.nodes[node.premise1].claim]
        # Required auxiliaries must be valid before any AST traversal.
        if node.term != -1 and (node.term < 0 or node.term >= proof.term_count):
            return False
        if node.template != -1 and (node.template < 0 or node.template >= proof.term_count):
            return False
        if node.witness != -1 and (node.witness < 0 or node.witness >= proof.term_count):
            return False
        if node.variable != -1 and (node.variable < 0 or node.variable >= FINISHING_VARIABLE_CAPACITY):
            return False
        if node.rule == RULE_ASSUME:
            contexts[i * size + i] = True
        elif node.rule == RULE_EQ_NORMALIZE:
            if f.kind != FORM_EQ or not _normal_equal(normal, f.left, f.right):
                return False
        elif node.rule == RULE_EQ_SYMM:
            if (f.kind != FORM_EQ or a.kind != FORM_EQ
                or not _normal_equal(normal, f.left, a.right) or not _normal_equal(normal, f.right, a.left)):
                return False
        elif node.rule == RULE_EQ_TRANS:
            if (f.kind != FORM_EQ or a.kind != FORM_EQ or b.kind != FORM_EQ
                or not _normal_equal(normal, a.right, b.left)
                or not _normal_equal(normal, f.left, a.left) or not _normal_equal(normal, f.right, b.right)):
                return False
        elif node.rule == RULE_EQ_CONGRUENCE:
            if (f.kind != FORM_EQ or a.kind != FORM_EQ or node.template < 0 or node.variable < 0
                or not _term_substitution_matches(proof, node.template, node.variable, a.left, f.left)
                or not _term_substitution_matches(proof, node.template, node.variable, a.right, f.right)):
                return False
        elif node.rule == RULE_LT_CONST or node.rule == RULE_LE_CONST:
            if f.kind != (FORM_LT if node.rule == RULE_LT_CONST else FORM_LE):
                return False
            var left = _constant_value(normal, f.left)
            var right = _constant_value(normal, f.right)
            if (left < 0 or right < 0 or (node.rule == RULE_LT_CONST and (f.kind != FORM_LT or left >= right))
                or (node.rule == RULE_LE_CONST and (f.kind != FORM_LE or left > right))):
                return False
        elif node.rule == RULE_NAT_NONNEG:
            if f.kind != FORM_LE or _constant_value(normal, f.left) != 0:
                return False
        elif node.rule == RULE_LE_ADD or node.rule == RULE_LT_ADD:
            var kind = FORM_LE if node.rule == RULE_LE_ADD else FORM_LT
            if (f.kind != kind or a.kind != kind or node.term < 0
                or not _normal_add(normal, f.left, a.left, node.term)
                or not _normal_add(normal, f.right, a.right, node.term)):
                return False
        elif node.rule == RULE_LE_SCALE or node.rule == RULE_LT_SCALE:
            var kind = FORM_LE if node.rule == RULE_LE_SCALE else FORM_LT
            if (f.kind != kind or a.kind != kind or node.value < 0
                or (node.rule == RULE_LT_SCALE and node.value == 0)
                or not _normal_scale(normal, f.left, a.left, node.value)
                or not _normal_scale(normal, f.right, a.right, node.value)):
                return False
        elif (node.rule == RULE_LE_TRANS or node.rule == RULE_LT_TRANS
            or node.rule == RULE_LE_LT_TRANS or node.rule == RULE_LT_LE_TRANS
            or node.rule == RULE_LE_EQ_TRANS or node.rule == RULE_LT_EQ_TRANS
            or node.rule == RULE_EQ_LE_TRANS or node.rule == RULE_EQ_LT_TRANS):
            var ak = FORM_LE
            var bk = FORM_LE
            var fk = FORM_LE
            if node.rule == RULE_LT_TRANS:
                ak = FORM_LT
                bk = FORM_LT
                fk = FORM_LT
            elif node.rule == RULE_LE_LT_TRANS:
                bk = FORM_LT
                fk = FORM_LT
            elif node.rule == RULE_LT_LE_TRANS:
                ak = FORM_LT
                fk = FORM_LT
            elif node.rule == RULE_LE_EQ_TRANS:
                bk = FORM_EQ
            elif node.rule == RULE_LT_EQ_TRANS:
                ak = FORM_LT
                bk = FORM_EQ
                fk = FORM_LT
            elif node.rule == RULE_EQ_LE_TRANS:
                ak = FORM_EQ
            elif node.rule == RULE_EQ_LT_TRANS:
                ak = FORM_EQ
                bk = FORM_LT
                fk = FORM_LT
            if (a.kind != ak or b.kind != bk or f.kind != fk
                or not _normal_equal(normal, a.right, b.left)
                or not _normal_equal(normal, f.left, a.left) or not _normal_equal(normal, f.right, b.right)):
                return False
        elif node.rule == RULE_LT_TO_LE_SUCC:
            if (a.kind != FORM_LT or f.kind != FORM_LE
                or not _normal_succ(normal, f.left, a.left) or not _normal_equal(normal, f.right, a.right)):
                return False
        elif node.rule == RULE_LE_SCALE_CANCEL or node.rule == RULE_LT_SCALE_CANCEL:
            var kind = FORM_LE if node.rule == RULE_LE_SCALE_CANCEL else FORM_LT
            if (a.kind != kind or f.kind != kind or node.value <= 0
                or not _normal_scale(normal, a.left, f.left, node.value)
                or not _normal_scale(normal, a.right, f.right, node.value)):
                return False
        elif node.rule == RULE_AND_INTRO:
            if (f.kind != FORM_AND
                or not _same_formula(proof, f.left, proof, proof.nodes[node.premise0].claim)
                or not _same_formula(proof, f.right, proof, proof.nodes[node.premise1].claim)):
                return False
        elif node.rule == RULE_AND_LEFT or node.rule == RULE_AND_RIGHT:
            var selected = a.left if node.rule == RULE_AND_LEFT else a.right
            if a.kind != FORM_AND or not _same_formula(proof, node.claim, proof, selected):
                return False
        elif node.rule == RULE_IMP_INTRO:
            if (f.kind != FORM_IMP or node.assumption < 0 or node.assumption >= i
                or proof.nodes[node.assumption].rule != RULE_ASSUME
                or not _same_formula(proof, f.left, proof, proof.nodes[node.assumption].claim)
                or not _same_formula(proof, f.right, proof, proof.nodes[node.premise0].claim)):
                return False
            contexts[i * size + node.assumption] = False
        elif node.rule == RULE_IMP_ELIM:
            if (a.kind != FORM_IMP
                or not _same_formula(proof, a.left, proof, proof.nodes[node.premise1].claim)
                or not _same_formula(proof, a.right, proof, node.claim)):
                return False
        elif node.rule == RULE_FORALL_INTRO:
            if (f.kind != FORM_FORALL or node.variable != f.variable
                or not _same_formula(proof, f.left, proof, proof.nodes[node.premise0].claim)):
                return False
            for k in range(i):
                if (contexts[i * size + k]
                    and (formula_variables[proof.nodes[k].claim] & (UInt32(1) << UInt32(node.variable))) != 0):
                    return False
        elif node.rule == RULE_FORALL_ELIM or node.rule == RULE_EXISTS_INTRO:
            if node.term < 0:
                return False
            if node.rule == RULE_FORALL_ELIM:
                if (a.kind != FORM_FORALL
                    or not _formula_substitution_matches(proof, a.left, a.variable, node.term, node.claim)):
                    return False
            elif (f.kind != FORM_EXISTS
                or not _formula_substitution_matches(proof, f.left, f.variable, node.term, proof.nodes[node.premise0].claim)):
                return False
        elif node.rule == RULE_NAT_INDUCTION:
            if f.kind != FORM_FORALL or node.variable != f.variable or node.term < 0:
                return False
            var zero = _find_term(proof, TERM_CONST, value=0)
            var succ = proof.terms[node.term]
            if (zero < 0 or succ.kind != TERM_ADD
                or proof.terms[succ.left].kind != TERM_VAR or proof.terms[succ.left].value != node.variable
                or proof.terms[succ.right].kind != TERM_CONST or proof.terms[succ.right].value != 1
                or b.kind != FORM_FORALL or b.variable != node.variable):
                return False
            var step = proof.formulas[b.left]
            if (step.kind != FORM_IMP
                or not _formula_substitution_matches(proof, f.left, node.variable, zero, proof.nodes[node.premise0].claim)
                or not _same_formula(proof, f.left, proof, step.left)
                or not _formula_substitution_matches(proof, f.left, node.variable, node.term, step.right)):
                return False
            for k in range(i):
                if (contexts[i * size + k]
                    and (formula_variables[proof.nodes[k].claim] & (UInt32(1) << UInt32(node.variable))) != 0):
                    return False
        elif node.rule == RULE_DIV_DECOMP or node.rule == RULE_MOD_LT:
            if node.term < 0 or node.value <= 0:
                return False
            var divided = _find_term(proof, TERM_DIV, node.term, value=node.value)
            var remainder = _find_term(proof, TERM_MOD, node.term, value=node.value)
            if divided < 0 or remainder < 0:
                return False
            if node.rule == RULE_MOD_LT:
                if (f.kind != FORM_LT or not _normal_equal(normal, f.left, remainder)
                    or _constant_value(normal, f.right) != node.value):
                    return False
            else:
                if f.kind != FORM_EQ or not _normal_equal(normal, f.left, node.term):
                    return False
                for k in range(NORMAL_WIDTH):
                    var q = normal[divided * NORMAL_WIDTH + k]
                    var r = normal[remainder * NORMAL_WIDTH + k]
                    if q > Int.MAX // node.value:
                        return False
                    var scaled = q * node.value
                    if scaled > Int.MAX - r or normal[f.right * NORMAL_WIDTH + k] != scaled + r:
                        return False
        elif node.rule == RULE_DIV_UNIQUE:
            if (node.term < 0 or node.template < 0 or node.witness < 0 or node.value <= 0
                or a.kind != FORM_EQ or b.kind != FORM_LT or f.kind != FORM_AND
                or not _normal_equal(normal, a.left, node.term)
                or not _normal_equal(normal, b.left, node.witness)
                or _constant_value(normal, b.right) != node.value):
                return False
            for k in range(NORMAL_WIDTH):
                var q = normal[node.template * NORMAL_WIDTH + k]
                var r = normal[node.witness * NORMAL_WIDTH + k]
                if q > Int.MAX // node.value:
                    return False
                var scaled = q * node.value
                if scaled > Int.MAX - r or normal[a.right * NORMAL_WIDTH + k] != scaled + r:
                    return False
            var divided = _find_term(proof, TERM_DIV, node.term, value=node.value)
            var remainder = _find_term(proof, TERM_MOD, node.term, value=node.value)
            var qeq = proof.formulas[f.left]
            var req = proof.formulas[f.right]
            if (divided < 0 or remainder < 0 or qeq.kind != FORM_EQ or req.kind != FORM_EQ
                or not _normal_equal(normal, qeq.left, node.template)
                or not _normal_equal(normal, qeq.right, divided)
                or not _normal_equal(normal, req.left, node.witness)
                or not _normal_equal(normal, req.right, remainder)):
                return False
        elif node.rule == RULE_BASE_ZERO:
            if f.kind != FORM_EQ or _constant_value(normal, f.right) != program.start:
                return False
            var base = proof.terms[f.left]
            if base.kind != TERM_BASE or _constant_value(normal, base.left) != 0:
                return False
        elif node.rule == RULE_BASE_STEP:
            if f.kind != FORM_EQ or node.term < 0:
                return False
            var base = proof.terms[f.left]
            if base.kind != TERM_BASE:
                return False
            var successor = proof.terms[base.left]
            if (successor.kind != TERM_ADD or not _same_term(proof, successor.left, proof, node.term)
                or _constant_value(normal, successor.right) != 1):
                return False
            var previous = _find_term(proof, TERM_BASE, node.term)
            if previous < 0:
                return False
            for k in range(NORMAL_WIDTH):
                var expected = normal[previous * NORMAL_WIDTH + k]
                if k == 0:
                    if expected > Int.MAX - program.stride:
                        return False
                    expected += program.stride
                if normal[f.right * NORMAL_WIDTH + k] != expected:
                    return False
        elif node.rule == RULE_VISIT_ZERO:
            if f.kind != FORM_VISITED or a.kind != FORM_LT or _constant_value(normal, f.left) != 0:
                return False
            var base = proof.terms[a.left]
            if (base.kind != TERM_BASE or _constant_value(normal, base.left) != 0
                or not _same_term(proof, a.right, proof, f.right)):
                return False
        elif node.rule == RULE_VISIT_STEP:
            if f.kind != FORM_VISITED or a.kind != FORM_VISITED or b.kind != FORM_LT:
                return False
            var successor = proof.terms[f.left]
            var base = proof.terms[b.left]
            if (successor.kind != TERM_ADD or not _same_term(proof, successor.left, proof, a.left)
                or _constant_value(normal, successor.right) != 1 or base.kind != TERM_BASE
                or not _same_term(proof, base.left, proof, f.left)
                or not _same_term(proof, a.right, proof, f.right)
                or not _same_term(proof, b.right, proof, f.right)):
                return False
        elif node.rule == RULE_VISIT_GUARD:
            if a.kind != FORM_VISITED or f.kind != FORM_LT:
                return False
            var base = proof.terms[f.left]
            if (base.kind != TERM_BASE or not _same_term(proof, base.left, proof, a.left)
                or not _same_term(proof, f.right, proof, a.right)):
                return False
        elif node.rule == RULE_DONE_STOP:
            # Individual operational constructor: a false guard yields a
            # finite run with zero body executions from this potential round.
            if (f.kind != FORM_DONE or a.kind != FORM_LE
                or not _same_term(proof, a.left, proof, f.right)):
                return False
            var base = proof.terms[a.right]
            if base.kind != TERM_BASE or not _same_term(proof, base.left, proof, f.left):
                return False
        elif node.rule == RULE_DONE_STEP:
            # Prepend one returning body to the finite suffix at round q+1.
            # Both the current active guard and exact successor are premises.
            if f.kind != FORM_DONE or a.kind != FORM_LT or b.kind != FORM_DONE:
                return False
            var base = proof.terms[a.left]
            var successor = proof.terms[b.left]
            if (base.kind != TERM_BASE or not _same_term(proof, base.left, proof, f.left)
                or not _same_term(proof, a.right, proof, f.right)
                or not _same_term(proof, b.right, proof, f.right)
                or successor.kind != TERM_ADD
                or not _same_term(proof, successor.left, proof, f.left)
                or _constant_value(normal, successor.right) != 1):
                return False
        elif node.rule == RULE_NAT_ORDER_CASES:
            # Generic Nat total order plus ordinary case elimination:
            # (x<y => F), (y<=x => F) establish any identical formula F.
            # Branch assumptions are discharged by ordinary IMP_INTRO; this
            # rule neither discards contexts nor recognizes loop conclusions.
            if a.kind != FORM_IMP or b.kind != FORM_IMP:
                return False
            var active = proof.formulas[a.left]
            var stopped = proof.formulas[b.left]
            if (active.kind != FORM_LT or stopped.kind != FORM_LE
                or not _normal_equal(normal, active.left, stopped.right)
                or not _normal_equal(normal, active.right, stopped.left)
                or not _same_formula(proof, a.right, proof, node.claim)
                or not _same_formula(proof, b.right, proof, node.claim)):
                return False
        else:
            return False
    for k in range(proof.node_count):
        if contexts[proof.root * size + k]:
            return False
    var model = empty_finishing_proof[1]()
    var goal = add_finishing_goal(model, program, include_termination)
    return goal >= 0 and _same_formula(proof, proof.nodes[proof.root].claim, model, goal)


def check_finishing_coverage_proof[size: Int](program: FinishingProgram,
    proof: FinishingProof[size]) -> Bool:
    """Historical coverage/headroom comparison; legacy Done rules are forbidden.

    Termination is checked independently by the proof-term kernel and the
    driver-specific checked inductive definition. The full legacy profile
    remains available only for comparison tests.
    """
    return check_finishing_proof(program, proof, include_termination=False)


@always_inline
def finishing_index[program: FinishingProgram](base: Int, thread: Int, item: Int) -> Int:
    return base + program.thread_scale * thread + program.item_scale * item + program.index_bias


@always_inline
def finishing_loop_active[program: FinishingProgram](base: Int, count: Int) -> Bool:
    comptime assert program.loop_guard == FINISH_LT, "unsupported finishing loop guard"
    return base < count


@always_inline
def finishing_load_active[program: FinishingProgram](index: Int, count: Int) -> Bool:
    comptime assert program.load_guard == FINISH_LT, "unsupported finishing load guard"
    return index < count
