"""Native proof admission and independent logical traversal checks.

Compile-time assertions check finite symbolic derivations, without enumerating
the runtime count. Host censuses exercise selected concrete instantiations.
"""
from std.collections import Array
from std.testing import assert_equal
from llm_mojo.kernels.finishing_proof import (
    FINISHING_PROOF_SIZE, FinishingProgram, FinishingProof, FinishingProofNode,
    make_finishing_program, make_finishing_proof, check_finishing_proof,
    finishing_index, finishing_loop_active, finishing_load_active,
)
from llm_mojo.kernels.finishing_proof_kernel import (
    TERM_CONST, TERM_VAR, TERM_ADD, TERM_SCALE, TERM_DIV,
    FORM_EQ, FORM_LT, FORM_LE, FORM_AND, FORM_IMP, FORM_FORALL, FORM_EXISTS, FORM_VISITED, FORM_DONE,
    RULE_ASSUME, RULE_EQ_NORMALIZE, RULE_AND_INTRO, RULE_AND_LEFT,
    RULE_FORALL_INTRO, RULE_FORALL_ELIM, RULE_EXISTS_INTRO, RULE_IMP_INTRO,
    RULE_NAT_INDUCTION, RULE_DIV_UNIQUE, RULE_LT_SCALE_CANCEL,
    RULE_DONE_STOP, RULE_DONE_STEP, RULE_NAT_ORDER_CASES, RULE_EQ_SYMM, RULE_LE_EQ_TRANS,
    RULE_VISIT_ZERO, RULE_LT_CONST, RULE_LE_CONST,
    empty_finishing_proof, add_term, add_formula, add_node, substitute_formula,
)


def check_instance[threads: Int, items: Int]() raises:
    comptime program = make_finishing_program(threads, items, threads * items)
    comptime proof = make_finishing_proof(program)
    comptime assert check_finishing_proof(program, proof), "finishing proof rejected"
    var native_program = make_finishing_program(threads, items, threads * items)
    var native_proof = make_finishing_proof(native_program)
    assert_equal(check_finishing_proof(native_program, native_proof), True)


def check_count[threads: Int, items: Int](count: Int) raises:
    comptime program = make_finishing_program(threads, items, threads * items)
    # The oracle is the flat set of runtime indices. It does not use a proof
    # rule, coefficient normalizer, or the proof builder's owner calculation.
    var reads = List[Int](capacity=count)
    for _ in range(count):
        reads.append(0)
    var base = program.start
    var iterations = 0
    while finishing_loop_active[program](base, count):
        for item in range(items):
            for thread in range(threads):
                var index = finishing_index[program](base, thread, item)
                assert_equal(index, base + item * threads + thread)
                if finishing_load_active[program](index, count):
                    reads[index] += 1
                else:
                    assert_equal(index >= count, True)
        base += program.stride
        iterations += 1
    for index in range(count):
        assert_equal(reads[index], 1)
        var round = index // (threads * items)
        var remainder = index % (threads * items)
        var item = remainder // threads
        var thread = remainder % threads
        assert_equal(finishing_loop_active[program](round * program.stride, count), True)
        assert_equal(finishing_index[program](round * program.stride, thread, item), index)
        assert_equal(finishing_load_active[program](index, count), True)
    assert_equal(iterations, 1 + (count - 1) // (threads * items))
    assert_equal(base >= count and base < count + threads * items, True)
    assert_equal(finishing_loop_active[program](base, count), False)


def check_semantics[threads: Int, items: Int]() raises:
    comptime tile = threads * items
    var counts: Array[Int, 9] = [
        1, threads, threads + 1, tile, tile + 1,
        2 * tile - 1, 2 * tile, 2 * tile + 1, 3 * tile + 7,
    ]
    for count in counts:
        check_count[threads, items](count)


def check_endpoint[threads: Int, items: Int]() raises:
    comptime program = make_finishing_program(threads, items, threads * items)
    var count = 2147483647
    var rounds = 1 + (count - 1) // program.stride
    var last = (rounds - 1) * program.stride
    var exit_base = rounds * program.stride
    assert_equal(finishing_loop_active[program](last, count), True)
    assert_equal(finishing_loop_active[program](exit_base, count), False)
    assert_equal(exit_base >= count and exit_base < count + program.stride, True)
    assert_equal(exit_base <= program.integer_limit, True)
    assert_equal(finishing_index[program](last, threads - 1, items - 1), exit_base - 1)
    var samples: Array[Int, 4] = [0, count // 2, count - 2, count - 1]
    for index in samples:
        var remainder = index % program.stride
        var thread = remainder % threads
        var item = remainder // threads
        assert_equal(finishing_index[program]((index // program.stride) * program.stride,
            thread, item), index)


def rejects_program(program: FinishingProgram, proof: FinishingProof[FINISHING_PROOF_SIZE]) raises:
    assert_equal(check_finishing_proof(program, proof), False)
    # Rebuilding cannot turn a malformed program into an accepted proof.
    var fresh = make_finishing_proof(program)
    assert_equal(check_finishing_proof(program, fresh), False)


def check_program_mutations() raises:
    var program = make_finishing_program(32, 3, 96)
    var proof = make_finishing_proof(program)
    assert_equal(check_finishing_proof(program, proof), True)
    var malformed = program
    malformed.start = 1
    rejects_program(malformed, proof)
    malformed = program
    malformed.loop_guard = 1
    rejects_program(malformed, proof)
    malformed.loop_guard = -1
    rejects_program(malformed, proof)
    malformed = program
    malformed.load_guard = 1
    rejects_program(malformed, proof)
    malformed.load_guard = -1
    rejects_program(malformed, proof)
    malformed = program
    malformed.thread_scale = 0
    rejects_program(malformed, proof)
    malformed = program
    malformed.item_scale = 31
    rejects_program(malformed, proof)
    malformed = program
    malformed.index_bias = 1
    rejects_program(malformed, proof)
    for stride in range(-1, 2):
        malformed = program
        malformed.stride = stride
        rejects_program(malformed, proof)
    malformed = program
    malformed.stride = 97
    rejects_program(malformed, proof)
    malformed = program
    malformed.stride = 95
    rejects_program(malformed, proof)
    malformed = program
    malformed.count_limit = 0
    rejects_program(malformed, proof)
    malformed.count_limit = -1
    rejects_program(malformed, proof)
    malformed.count_limit = Int.MAX
    rejects_program(malformed, proof)
    malformed = program
    malformed.integer_limit = 2147483647
    rejects_program(malformed, proof)
    malformed.integer_limit = -1
    rejects_program(malformed, proof)
    malformed = program
    malformed.threads = 0
    rejects_program(malformed, proof)
    malformed = program
    malformed.items = -1
    rejects_program(malformed, proof)
    malformed = program
    malformed.threads = Int.MAX
    rejects_program(malformed, proof)
    # Two valid statements need their own matching proof, even when both have
    # the same logical ownership shape.
    malformed = make_finishing_program(32, 3, 96, 1000)
    assert_equal(check_finishing_proof(malformed, proof), False)
    var restricted = make_finishing_proof(malformed)
    assert_equal(check_finishing_proof(malformed, restricted), True)


def extended_proof[extra: Int](program: FinishingProgram) -> FinishingProof[FINISHING_PROOF_SIZE + extra]:
    var original = make_finishing_proof(program)
    var result = empty_finishing_proof[FINISHING_PROOF_SIZE + extra]()
    for i in range(original.term_count):
        result.terms[i] = original.terms[i]
    for i in range(original.formula_count):
        result.formulas[i] = original.formulas[i]
    for i in range(original.node_count):
        result.nodes[i] = original.nodes[i]
    result.term_count = original.term_count
    result.formula_count = original.formula_count
    result.node_count = original.node_count
    result.root = original.root
    return result^


def redundant_proof(program: FinishingProgram) -> FinishingProof[FINISHING_PROOF_SIZE + 1]:
    var result = extended_proof[1](program)
    var zero = add_term(result, TERM_CONST, value=0)
    var reflexive = add_formula(result, FORM_EQ, zero, zero)
    # A correct unreferenced fact is harmless; the selected root is earlier.
    _ = add_node(result, RULE_EQ_NORMALIZE, reflexive)
    return result^


def independent_leaf(node: FinishingProofNode) -> Bool:
    return (node.premise0 == -1 and node.premise1 == -1
        and node.premise2 == -1 and node.premise3 == -1 and node.assumption == -1)


def swap_pair[size: Int](proof: FinishingProof[size]) -> Int:
    for i in range(proof.node_count - 1):
        if (independent_leaf(proof.nodes[i]) and independent_leaf(proof.nodes[i + 1])
                and (proof.nodes[i].rule != proof.nodes[i + 1].rule
                     or proof.nodes[i].claim != proof.nodes[i + 1].claim)):
            return i
    return -1


def swapped_reference(index: Int, first: Int) -> Int:
    if index == first:
        return first + 1
    if index == first + 1:
        return first
    return index


def reordered_proof(program: FinishingProgram) -> FinishingProof[FINISHING_PROOF_SIZE]:
    var original = make_finishing_proof(program)
    var result = original.copy()
    var first = swap_pair(original)
    if first < 0:
        return result^
    for i in range(original.node_count):
        result.nodes[i] = original.nodes[swapped_reference(i, first)]
        result.nodes[i].premise0 = swapped_reference(result.nodes[i].premise0, first)
        result.nodes[i].premise1 = swapped_reference(result.nodes[i].premise1, first)
        result.nodes[i].premise2 = swapped_reference(result.nodes[i].premise2, first)
        result.nodes[i].premise3 = swapped_reference(result.nodes[i].premise3, first)
        result.nodes[i].assumption = swapped_reference(result.nodes[i].assumption, first)
    result.root = swapped_reference(original.root, first)
    return result^


def find_rule[size: Int](proof: FinishingProof[size], rule: Int) -> Int:
    for i in range(proof.node_count):
        if proof.nodes[i].rule == rule:
            return i
    return -1


def find_last_rule[size: Int](proof: FinishingProof[size], rule: Int) -> Int:
    var selected = -1
    for i in range(proof.node_count):
        if proof.nodes[i].rule == rule:
            selected = i
    return selected


comptime BAD_UNSUPPORTED = 0
comptime BAD_FORWARD = 1
comptime BAD_NEGATIVE = 2
comptime BAD_DEAD = 3
comptime BAD_EQUATION = 4
comptime BAD_DESCENT = 5
comptime BAD_MISSING_ROOT = 6
comptime BAD_INVALID_ROOT = 7
comptime BAD_HOLE = 8
comptime BAD_RETIRED_EMPTY = 9
comptime BAD_RETIRED_NAMED = 10
comptime BAD_SUBSTITUTION = 11
comptime BAD_INDUCTION_BASE = 12
comptime BAD_INDUCTION_STEP = 13
comptime BAD_EIGENVARIABLE = 14
comptime BAD_DISCHARGE = 15
comptime BAD_WITNESS = 16
comptime BAD_OWNER_BOUNDARY = 17
comptime BAD_UNIQUENESS = 18
comptime BAD_CAPTURE = 19
comptime BAD_MISSING_WITNESS = 20
comptime BAD_INDUCTION_CONTEXT = 21
comptime BAD_OVERFLOW = 22
comptime BAD_NEGATIVE_NAT = 23
comptime BAD_ZERO_DIVISOR = 24
comptime BAD_VISITED_COUNT = 25
comptime BAD_CONST_CLAIM_KIND = 26
comptime BAD_OPEN_ROOT = 27
comptime BAD_RETIRED_TERMINATION_EMPTY = 28
comptime BAD_RETIRED_TERMINATION_NAMED = 29
comptime BAD_RETIRED_TERMINATION_ATOM = 30
comptime BAD_STOP_DIRECTION = 31
comptime BAD_STOP_ROUND = 32
comptime BAD_STOP_COUNT = 33
comptime BAD_STEP_GUARD = 34
comptime BAD_STEP_ROUND = 35
comptime BAD_STEP_COUNT = 36
comptime BAD_STEP_SUFFIX = 37
comptime BAD_STEP_CYCLE = 38
comptime BAD_CASE_ORDER = 39
comptime BAD_CASE_CONCLUSION = 40
comptime BAD_TERM_FUEL = 41
comptime BAD_TERM_IH_BOUND = 42
comptime BAD_TERM_BOUND = 43
comptime BAD_TERM_OPEN_ASSUMPTION = 44
comptime BAD_CASE_COUNT = 45


def appended_assumption_proof(program: FinishingProgram) -> FinishingProof[FINISHING_PROOF_SIZE + 16]:
    var proof = extended_proof[16](program)
    var round = add_term(proof, TERM_VAR, value=20)
    var count = add_term(proof, TERM_VAR, value=21)
    var visit = add_formula(proof, FORM_VISITED, round, count)
    _ = add_node(proof, RULE_ASSUME, visit)
    return proof^


def malformed_certificate(program: FinishingProgram, mode: Int) -> FinishingProof[FINISHING_PROOF_SIZE + 16]:
    # Every branch returns an ordinary data object. The compiler fixtures must
    # fail at the proof checker, never at a raising/asserting builder.
    var proof = extended_proof[16](program)
    var zero = add_term(proof, TERM_CONST, value=0)
    var one = add_term(proof, TERM_CONST, value=1)
    if mode == BAD_UNSUPPORTED:
        proof.nodes[0].rule = 999
    elif mode == BAD_FORWARD or mode == BAD_NEGATIVE:
        var selected = find_rule(proof, RULE_AND_LEFT)
        proof.nodes[selected].premise0 = selected if mode == BAD_FORWARD else -1
    elif mode == BAD_DEAD:
        _ = add_node(proof, 999, proof.nodes[proof.root].claim)
    elif mode == BAD_EQUATION:
        var false_equality = add_formula(proof, FORM_EQ, zero, one)
        _ = add_node(proof, RULE_EQ_NORMALIZE, false_equality)
    elif mode == BAD_DESCENT:
        var selected = find_rule(proof, RULE_DONE_STEP)
        # A guard cannot replace the finite suffix needed after advancing.
        proof.nodes[selected].premise1 = proof.nodes[selected].premise0
    elif mode == BAD_MISSING_ROOT:
        proof.root = 0
    elif mode == BAD_INVALID_ROOT:
        proof.root = proof.node_count
    elif mode == BAD_HOLE:
        proof.nodes[0].claim = -1
    elif mode == BAD_RETIRED_EMPTY or mode == BAD_RETIRED_NAMED:
        # These were whole-lemma rule numbers in the previous kernel. Even a
        # real target formula and named premises cannot register them now.
        var target = proof.nodes[proof.root].claim
        _ = add_node(proof, 26 if mode == BAD_RETIRED_EMPTY else 20, target,
            p0=-1 if mode == BAD_RETIRED_EMPTY else 0,
            p1=-1 if mode == BAD_RETIRED_EMPTY else 1)
    elif mode == BAD_SUBSTITUTION:
        var selected = find_rule(proof, RULE_FORALL_ELIM)
        proof.nodes[selected].term = one if proof.nodes[selected].term == zero else zero
    elif mode == BAD_INDUCTION_BASE:
        var selected = find_rule(proof, RULE_NAT_INDUCTION)
        proof.nodes[selected].premise0 = proof.nodes[selected].premise1
    elif mode == BAD_INDUCTION_STEP:
        var selected = find_rule(proof, RULE_NAT_INDUCTION)
        proof.nodes[selected].term = zero
    elif mode == BAD_EIGENVARIABLE:
        var selected = find_rule(proof, RULE_FORALL_INTRO)
        proof.nodes[selected].variable = 31
    elif mode == BAD_DISCHARGE:
        var selected = find_rule(proof, RULE_IMP_INTRO)
        proof.nodes[selected].assumption = selected
    elif mode == BAD_WITNESS or mode == BAD_MISSING_WITNESS:
        var selected = find_rule(proof, RULE_EXISTS_INTRO)
        proof.nodes[selected].term = -1 if mode == BAD_MISSING_WITNESS else one
    elif mode == BAD_OWNER_BOUNDARY:
        var selected = find_rule(proof, RULE_LT_SCALE_CANCEL)
        var original = proof.formulas[proof.nodes[selected].claim]
        var relaxed = add_formula(proof, FORM_LT, original.left,
            add_term(proof, TERM_ADD, original.right, one))
        proof.nodes[selected].claim = relaxed
    elif mode == BAD_UNIQUENESS:
        var selected = find_rule(proof, RULE_DIV_UNIQUE)
        proof.nodes[selected].witness = zero
    elif mode == BAD_CAPTURE:
        # Valid premise: forall y, y=y. It cannot justify exists x, forall y,
        # x=y using free y as the witness: that substitution captures y.
        var x = add_term(proof, TERM_VAR, value=20)
        var y = add_term(proof, TERM_VAR, value=21)
        var same = add_formula(proof, FORM_EQ, y, y)
        var refl = add_node(proof, RULE_EQ_NORMALIZE, same)
        var all_same = add_formula(proof, FORM_FORALL, same, variable=21)
        var universal = add_node(proof, RULE_FORALL_INTRO, all_same, p0=refl, variable=21)
        var different = add_formula(proof, FORM_EQ, x, y)
        var all_different = add_formula(proof, FORM_FORALL, different, variable=21)
        var existential = add_formula(proof, FORM_EXISTS, all_different, variable=20)
        _ = add_node(proof, RULE_EXISTS_INTRO, existential, p0=universal, term=y)
    elif mode == BAD_INDUCTION_CONTEXT:
        # Add an independent assumption mentioning the induction variable to
        # the step's context through conjunction/projection. The step formula
        # remains exactly right; generic induction must still reject escape.
        var selected = find_rule(proof, RULE_NAT_INDUCTION)
        var original = proof.nodes[selected]
        var variable = add_term(proof, TERM_VAR, value=original.variable)
        var count = add_term(proof, TERM_VAR, value=0)
        var open_formula = add_formula(proof, FORM_VISITED, variable, count)
        var assumption = add_node(proof, RULE_ASSUME, open_formula)
        var step = proof.nodes[original.premise1].claim
        var conjunction = add_formula(proof, FORM_AND, step, open_formula)
        var combined = add_node(proof, RULE_AND_INTRO, conjunction,
            p0=original.premise1, p1=assumption)
        var weakened_step = add_node(proof, RULE_AND_LEFT, step, p0=combined)
        original.premise1 = weakened_step
        proof.nodes[proof.node_count] = original
        proof.node_count += 1
    elif mode == BAD_OVERFLOW:
        # In machine arithmetic this could wrap to zero; over Nat it is false.
        var large = add_term(proof, TERM_CONST, value=Int.MAX // 2 + 1)
        var scaled = add_term(proof, TERM_SCALE, large, value=4)
        var false_equality = add_formula(proof, FORM_EQ, scaled, zero)
        _ = add_node(proof, RULE_EQ_NORMALIZE, false_equality)
    elif mode == BAD_NEGATIVE_NAT:
        var negative = add_term(proof, TERM_CONST, value=-1)
        var claim = add_formula(proof, FORM_EQ, negative, negative)
        _ = add_node(proof, RULE_EQ_NORMALIZE, claim)
    elif mode == BAD_ZERO_DIVISOR:
        var divided = add_term(proof, TERM_DIV, one, value=0)
        var claim = add_formula(proof, FORM_EQ, divided, divided)
        _ = add_node(proof, RULE_EQ_NORMALIZE, claim)
    elif mode == BAD_VISITED_COUNT:
        var selected = find_rule(proof, RULE_VISIT_ZERO)
        var claim = proof.nodes[selected].claim
        proof.formulas[claim].right = one
    elif mode == BAD_CONST_CLAIM_KIND:
        # A quantified claim has a formula child and right=-1, not two term
        # references. Constant arithmetic must reject before indexing terms.
        var same = add_formula(proof, FORM_EQ, zero, zero)
        var quantified = add_formula(proof, FORM_FORALL, same, variable=20)
        _ = add_node(proof, RULE_LT_CONST, quantified)
    elif mode == BAD_OPEN_ROOT:
        # The exact theorem syntax is insufficient while assumed as a premise.
        var literal_goal = proof.nodes[proof.root].claim
        proof.root = add_node(proof, RULE_ASSUME, literal_goal)
    elif mode == BAD_RETIRED_TERMINATION_EMPTY or mode == BAD_RETIRED_TERMINATION_NAMED:
        var done = proof.nodes[find_rule(proof, RULE_DONE_STOP)].claim
        _ = add_node(proof, 140, done,
            p0=-1 if mode == BAD_RETIRED_TERMINATION_EMPTY else find_rule(proof, RULE_DONE_STEP),
            p1=-1 if mode == BAD_RETIRED_TERMINATION_EMPTY else find_rule(proof, RULE_NAT_INDUCTION))
    elif mode == BAD_RETIRED_TERMINATION_ATOM:
        # The old unary termination atom cannot be used as a formula either.
        var count = add_term(proof, TERM_VAR, value=0)
        var retired = add_formula(proof, 8, count)
        _ = add_node(proof, RULE_ASSUME, retired)
    elif mode == BAD_STOP_DIRECTION or mode == BAD_STOP_ROUND or mode == BAD_STOP_COUNT:
        var original = proof.nodes[find_rule(proof, RULE_DONE_STOP)]
        var done = proof.formulas[original.claim]
        if mode == BAD_STOP_DIRECTION:
            var guard = proof.formulas[proof.nodes[original.premise0].claim]
            var reversed_guard = add_formula(proof, FORM_LE, guard.right, guard.left)
            original.premise0 = add_node(proof, RULE_ASSUME, reversed_guard)
        else:
            original.claim = add_formula(proof, FORM_DONE,
                add_term(proof, TERM_ADD, done.left, one) if mode == BAD_STOP_ROUND else done.left,
                add_term(proof, TERM_ADD, done.right, one) if mode == BAD_STOP_COUNT else done.right)
        proof.nodes[proof.node_count] = original
        proof.node_count += 1
    elif (mode == BAD_STEP_GUARD or mode == BAD_STEP_ROUND or mode == BAD_STEP_COUNT
            or mode == BAD_STEP_SUFFIX or mode == BAD_STEP_CYCLE):
        var selected = find_rule(proof, RULE_DONE_STEP)
        var original = proof.nodes[selected]
        var done = proof.formulas[original.claim]
        if mode == BAD_STEP_CYCLE:
            proof.nodes[selected].premise1 = selected
        else:
            if mode == BAD_STEP_GUARD:
                original.premise0 = proof.nodes[find_rule(proof, RULE_DONE_STOP)].premise0
            elif mode == BAD_STEP_SUFFIX:
                # A finite suffix at this same round is not the next round.
                original.premise1 = add_node(proof, RULE_ASSUME, original.claim)
            else:
                original.claim = add_formula(proof, FORM_DONE,
                    add_term(proof, TERM_ADD, done.left, one) if mode == BAD_STEP_ROUND else done.left,
                    add_term(proof, TERM_ADD, done.right, one) if mode == BAD_STEP_COUNT else done.right)
            proof.nodes[proof.node_count] = original
            proof.node_count += 1
    elif mode == BAD_CASE_ORDER or mode == BAD_CASE_CONCLUSION:
        var original = proof.nodes[find_rule(proof, RULE_NAT_ORDER_CASES)]
        if mode == BAD_CASE_ORDER:
            var active = original.premise0
            original.premise0 = original.premise1
            original.premise1 = active
        else:
            var done = proof.formulas[original.claim]
            original.claim = add_formula(proof, FORM_DONE, done.left,
                add_term(proof, TERM_ADD, done.right, one))
        proof.nodes[proof.node_count] = original
        proof.node_count += 1
    elif mode == BAD_TERM_FUEL:
        var selected = find_last_rule(proof, RULE_NAT_INDUCTION)
        proof.nodes[selected].term = zero
    elif mode == BAD_TERM_IH_BOUND:
        for i in range(proof.node_count):
            var node = proof.nodes[i]
            if node.rule == RULE_FORALL_ELIM and node.premise0 >= 0:
                var premise = proof.nodes[node.premise0]
                var quantified = proof.formulas[premise.claim]
                if premise.rule == RULE_ASSUME and quantified.kind == FORM_FORALL:
                    var body = proof.formulas[quantified.left]
                    if body.kind == FORM_IMP and proof.formulas[body.right].kind == FORM_DONE:
                        # The IH is needed at q+1, not at the current round.
                        proof.nodes[i].term = zero
                        break
    elif mode == BAD_TERM_BOUND:
        var induction = find_last_rule(proof, RULE_NAT_INDUCTION)
        for i in range(proof.node_count):
            if proof.nodes[i].rule == RULE_FORALL_ELIM and proof.nodes[i].premise0 == induction:
                # Substituting fuel zero cannot retain the claim for fuel N.
                proof.nodes[i].term = zero
                break
    elif mode == BAD_TERM_OPEN_ASSUMPTION:
        for i in range(proof.node_count):
            var claim = proof.formulas[proof.nodes[i].claim]
            if claim.kind == FORM_DONE:
                var round = proof.terms[claim.left]
                var count = proof.terms[claim.right]
                if (round.kind == TERM_CONST and round.value == 0
                        and count.kind == TERM_VAR and count.value == 0):
                    # Replace the derived execution with an undischarged
                    # assumption. The same literal root must remain open.
                    proof.nodes[i] = FinishingProofNode(RULE_ASSUME, -1, -1, -1, -1,
                        proof.nodes[i].claim, -1, -1, -1, -1, -1, -1)
                    break
    return proof^


def scope_escape_proof(program: FinishingProgram) -> FinishingProof[FINISHING_PROOF_SIZE + 16]:
    var proof = appended_assumption_proof(program)
    var selected = proof.node_count - 1
    var universal = add_formula(proof, FORM_FORALL, proof.nodes[selected].claim, variable=21)
    _ = add_node(proof, RULE_FORALL_INTRO, universal, p0=selected, variable=21)
    return proof^


def done_scope_escape_proof(program: FinishingProgram) -> FinishingProof[FINISHING_PROOF_SIZE + 16]:
    var proof = extended_proof[16](program)
    var round = add_term(proof, TERM_VAR, value=20)
    var count = add_term(proof, TERM_VAR, value=21)
    var done = add_formula(proof, FORM_DONE, round, count)
    var assumption = add_node(proof, RULE_ASSUME, done)
    var universal = add_formula(proof, FORM_FORALL, done, variable=21)
    _ = add_node(proof, RULE_FORALL_INTRO, universal, p0=assumption, variable=21)
    return proof^


def case_context_proof(program: FinishingProgram, escape: Bool) -> FinishingProof[FINISHING_PROOF_SIZE + 16]:
    var proof = extended_proof[16](program)
    var original = proof.nodes[find_rule(proof, RULE_NAT_ORDER_CASES)]
    var round = add_term(proof, TERM_VAR, value=20)
    var count = add_term(proof, TERM_VAR, value=21)
    var extra = add_formula(proof, FORM_DONE, round, count)
    var assumption = add_node(proof, RULE_ASSUME, extra)
    var branch = proof.nodes[original.premise0].claim
    var conjunction = add_formula(proof, FORM_AND, branch, extra)
    var combined = add_node(proof, RULE_AND_INTRO, conjunction,
        p0=original.premise0, p1=assumption)
    original.premise0 = add_node(proof, RULE_AND_LEFT, branch, p0=combined)
    var cases = proof.node_count
    proof.nodes[cases] = original
    proof.node_count += 1
    # The unused case remains valid under the extra assumption. Attempting
    # forall-introduction must observe its context rather than discard it.
    if escape:
        var universal = add_formula(proof, FORM_FORALL, original.claim, variable=20)
        _ = add_node(proof, RULE_FORALL_INTRO, universal, p0=cases, variable=20)
    return proof^


def stop_leaf_proof(program: FinishingProgram, round: Int, count: Int) -> FinishingProof[FINISHING_PROOF_SIZE + 16]:
    var proof = extended_proof[16](program)
    var base_induction = find_rule(proof, RULE_NAT_INDUCTION)
    var universal = proof.formulas[proof.nodes[base_induction].claim]
    var q = add_term(proof, TERM_CONST, value=round)
    var n = add_term(proof, TERM_CONST, value=count)
    var instantiated = substitute_formula(proof, universal.left, universal.variable, q)
    var equality = proof.formulas[instantiated]
    var at_round = add_node(proof, RULE_FORALL_ELIM, instantiated,
        p0=base_induction, term=q)
    var bound = add_formula(proof, FORM_LE, n, equality.right)
    var sufficient = add_node(proof, RULE_LE_CONST, bound)
    var reversed = add_formula(proof, FORM_EQ, equality.right, equality.left)
    var rewrite = add_node(proof, RULE_EQ_SYMM, reversed, p0=at_round)
    var exit_guard = add_formula(proof, FORM_LE, n, equality.left)
    var stop = add_node(proof, RULE_LE_EQ_TRANS, exit_guard, p0=sufficient, p1=rewrite)
    var done = add_formula(proof, FORM_DONE, q, n)
    _ = add_node(proof, RULE_DONE_STOP, done, p0=stop)
    return proof^


def check_termination_controls() raises:
    comptime program = make_finishing_program(32, 3, 96)
    comptime zero = stop_leaf_proof(program, 0, 0)
    comptime exact = stop_leaf_proof(program, 1, 96)
    comptime overshoot = stop_leaf_proof(program, 1, 95)
    comptime assert check_finishing_proof(program, zero), "finishing proof rejected"
    comptime assert check_finishing_proof(program, exact), "finishing proof rejected"
    comptime assert check_finishing_proof(program, overshoot), "finishing proof rejected"
    var endpoint = stop_leaf_proof(program, 1 + (2147483647 - 1) // 96, 2147483647)
    assert_equal(check_finishing_proof(program, endpoint), True)
    var later = stop_leaf_proof(program, 2147483647, 2147483647)
    assert_equal(check_finishing_proof(program, later), True)
    assert_equal(finishing_loop_active[program](0, 0), False)
    assert_equal(finishing_loop_active[program](96, 96), False)
    assert_equal(finishing_loop_active[program](96, 95), False)


def check_structural_mutations() raises:
    var program = make_finishing_program(32, 3, 96)
    for mode in range(BAD_CASE_COUNT):
        var proof = malformed_certificate(program, mode)
        if check_finishing_proof(program, proof):
            raise Error("Malformed primitive proof accepted: " + String(mode))
    var control = appended_assumption_proof(program)
    assert_equal(check_finishing_proof(program, control), True)
    var escaped = scope_escape_proof(program)
    assert_equal(check_finishing_proof(program, escaped), False)
    var done_escaped = done_scope_escape_proof(program)
    assert_equal(check_finishing_proof(program, done_escaped), False)
    var cases_control = case_context_proof(program, False)
    assert_equal(check_finishing_proof(program, cases_control), True)
    var cases_escaped = case_context_proof(program, True)
    assert_equal(check_finishing_proof(program, cases_escaped), False)
    var empty = empty_finishing_proof[0]()
    assert_equal(check_finishing_proof(program, empty), False)
    var extreme = extended_proof[1](program)
    extreme.terms[0].value = Int.MIN
    assert_equal(check_finishing_proof(program, extreme), False)
    extreme = extended_proof[1](program)
    extreme.formulas[0].left = Int.MAX
    assert_equal(check_finishing_proof(program, extreme), False)


def check_redundant_derivation() raises:
    comptime program = make_finishing_program(64, 5, 320)
    comptime proof = redundant_proof(program)
    comptime assert check_finishing_proof(program, proof), "finishing proof rejected"
    assert_equal(proof.root < FINISHING_PROOF_SIZE + 1, True)
    comptime reordered = reordered_proof(program)
    comptime assert check_finishing_proof(program, reordered), "finishing proof rejected"
    var original = make_finishing_proof(program)
    var first = swap_pair(original)
    assert_equal(first >= 0, True)
    var native_reordered = reordered_proof(program)
    assert_equal(check_finishing_proof(program, native_reordered), True)
    assert_equal(native_reordered.nodes[first].claim, original.nodes[first + 1].claim)
    assert_equal(native_reordered.nodes[first + 1].claim, original.nodes[first].claim)


def main() raises:
    check_instance[1, 1]()
    check_instance[1, 3]()
    check_instance[3, 5]()
    check_instance[3, 65]()
    check_instance[32, 1]()
    check_instance[32, 65]()
    check_instance[64, 3]()
    check_instance[128, 5]()
    check_instance[256, 1]()
    # These proofs must have constant finite size, independent of tile extent.
    # Do not census or launch these logical shapes on a GPU.
    check_instance[2147483647, 1]()
    check_instance[1, 2147483647]()
    comptime small = make_finishing_program(3, 5, 15, 17, 32)
    comptime small_proof = make_finishing_proof(small)
    comptime assert check_finishing_proof(small, small_proof), "finishing proof rejected"
    check_semantics[1, 1]()
    check_semantics[3, 5]()
    check_semantics[3, 65]()
    check_semantics[32, 1]()
    check_semantics[64, 3]()
    check_semantics[128, 5]()
    check_semantics[256, 1]()
    check_endpoint[1, 1]()
    check_endpoint[32, 65]()
    check_endpoint[64, 3]()
    check_endpoint[2147483647, 1]()
    check_endpoint[1, 2147483647]()
    check_program_mutations()
    check_structural_mutations()
    check_redundant_derivation()
    check_termination_controls()
    print("Mojo finishing proof admission and independent traversal checks passed")
