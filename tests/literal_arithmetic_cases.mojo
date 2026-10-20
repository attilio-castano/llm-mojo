"""Adversarial requests for arithmetic checked through raw Nat conversion.

Expected small results and word boundaries are written independently of the
native arithmetic that proposes production compactions. A failed construction
of an intended mutant returns True, so the rejection harness fails visibly.
"""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.nat_literal_arithmetic import make_literal_arithmetic_proof

comptime LIT_FALSE_SUM = 0
comptime LIT_FALSE_PRODUCT = 1
comptime LIT_FALSE_QUOTIENT = 2
comptime LIT_FALSE_REMAINDER = 3
comptime LIT_FALSE_BASE = 4
comptime LIT_FALSE_SCALE = 5
comptime LIT_REQUEST_OPERATION = 6
comptime LIT_REQUEST_LEFT = 7
comptime LIT_REQUEST_RIGHT = 8
comptime LIT_REQUEST_RESULT = 9
comptime LIT_SOURCE_ENDPOINT = 10
comptime LIT_RESULT_ENDPOINT = 11
comptime LIT_PROOF_ENDPOINT = 12
comptime LIT_PROOF_BOUNDARY = 13
comptime LIT_SOURCE_BOUNDARY = 14
comptime LIT_RESULT_BOUNDARY = 15
comptime LIT_NEGATIVE_COUNT = 16
comptime LIT_OVERSIZED_COUNT = 17
comptime LIT_UNUSED_MALFORMED = 18
comptime LIT_BOOTSTRAP_LITERAL = 19
comptime LIT_BOOTSTRAP_SUCCESSOR = 20
comptime LIT_BOOTSTRAP_OPERATOR = 21
comptime LIT_BOOTSTRAP_AFFINE = 22
comptime LIT_BOOTSTRAP_ORDER = 23
comptime LIT_BOOTSTRAP_EXPORT = 24
comptime LIT_BOOTSTRAP_PRIVATE = 25
comptime LIT_MISSING_DEFINITION = 26
comptime LIT_EXTRA_DEFINITION = 27
comptime LIT_CYCLIC_DEFINITION = 28
comptime LIT_FORWARD_DEFINITION = 29
comptime LIT_OPEN_DEFINITION = 30
comptime LIT_WRONG_DECLARED_TYPE = 31
comptime LIT_ALTERED_NAT = 32
comptime LIT_NEGATIVE_DEFINITIONS = 33
comptime LIT_OVERSIZED_DEFINITIONS = 34
comptime LIT_WRONG_OPERATION = 35
comptime LIT_STORED_LEFT = 36
comptime LIT_STORED_RIGHT = 37
comptime LIT_STORED_RESULT = 38
comptime LIT_NEGATIVE_SOURCE = 39
comptime LIT_NEGATIVE_RESULT = 40
comptime LIT_NEGATIVE_PROOF = 41
comptime LIT_UNKNOWN_SYNTAX = 42
comptime LIT_UNUSED_FALSE_REFLEXIVITY = 43
comptime LIT_ZERO_DIVISOR = 44
comptime LIT_NEGATIVE_DIVISOR = 45
comptime LIT_NEGATIVE_OPERAND = 46
comptime LIT_NEGATIVE_CANDIDATE = 47
comptime LIT_WRONG_ADD_DEFINITION = 48
comptime LIT_DIV_WRONG_PROJECTION = 49
comptime LIT_DIV_WRONG_PREDECESSOR = 50
comptime LIT_DIV_SOURCE_OPERAND = 51
comptime LITERAL_ARITHMETIC_MUTANTS = 52


def literal_arithmetic_accepts(operation: Int, left: Int, right: Int,
    candidate: Int, independent_proof: Bool = False) -> Bool:
    var p = make_literal_arithmetic_proof(operation, left, right, candidate)
    if independent_proof:
        # Refl may name either endpoint when ordinary raw conversion actually
        # establishes their equality. No producer reduction trace is reused.
        p.proof = tk.pt_add(p.bundle, tk.PT_REFL, p.result)
        p.bundle.root = p.proof
    return (tk.check_native_literal_arithmetic(p)
        and tk.check_native_literal_arithmetic_request(p, operation, left, right, candidate))


def literal_arithmetic_small_cases_accepts() -> Bool:
    return (literal_arithmetic_accepts(tk.PT_ADD, 3, 2, 5)
        and literal_arithmetic_accepts(tk.PT_ADD, 0, 3, 3, True)
        and literal_arithmetic_accepts(tk.PT_ADD, 3, 0, 3)
        and literal_arithmetic_accepts(tk.PT_MUL, 3, 2, 6)
        and literal_arithmetic_accepts(tk.PT_SCALE, 3, 2, 6, True)
        and literal_arithmetic_accepts(tk.PT_SCALE, 0, 3, 0)
        and literal_arithmetic_accepts(tk.PT_SCALE, 1, 3, 3)
        and literal_arithmetic_accepts(tk.PT_BASE, 3, 2, 6)
        and literal_arithmetic_accepts(tk.PT_BASE, 7, 0, 0)
        and literal_arithmetic_accepts(tk.PT_DIV, 8, 3, 2)
        and literal_arithmetic_accepts(tk.PT_MOD, 8, 3, 2, True)
        and literal_arithmetic_accepts(tk.PT_DIV, 0, 3, 0)
        and literal_arithmetic_accepts(tk.PT_MOD, 0, 3, 0)
        and literal_arithmetic_accepts(tk.PT_DIV, 2, 3, 0)
        and literal_arithmetic_accepts(tk.PT_MOD, 2, 3, 2)
        and literal_arithmetic_accepts(tk.PT_DIV, 3, 3, 1)
        and literal_arithmetic_accepts(tk.PT_MOD, 3, 3, 0)
        and literal_arithmetic_accepts(tk.PT_DIV, 4, 3, 1)
        and literal_arithmetic_accepts(tk.PT_MOD, 4, 3, 1)
        and literal_arithmetic_accepts(tk.PT_DIV, 3, 1, 3)
        and literal_arithmetic_accepts(tk.PT_MOD, 3, 1, 0))


def literal_arithmetic_mutant_accepts(mode: Int) -> Bool:
    var p = make_literal_arithmetic_proof(tk.PT_ADD, 3, 2, 5)
    # A broken positive producer/checker is never counted as mutant rejection.
    if not tk.check_native_literal_arithmetic_request(p, tk.PT_ADD, 3, 2, 5):
        return True
    if mode == LIT_FALSE_SUM:
        return literal_arithmetic_accepts(tk.PT_ADD, 3, 2, 6)
    if mode == LIT_FALSE_PRODUCT:
        return literal_arithmetic_accepts(tk.PT_MUL, 3, 2, 5)
    if mode == LIT_FALSE_QUOTIENT:
        return literal_arithmetic_accepts(tk.PT_DIV, 8, 3, 3)
    if mode == LIT_FALSE_REMAINDER:
        return literal_arithmetic_accepts(tk.PT_MOD, 8, 3, 1)
    if mode == LIT_FALSE_BASE:
        return literal_arithmetic_accepts(tk.PT_BASE, 3, 2, 5)
    if mode == LIT_FALSE_SCALE:
        return literal_arithmetic_accepts(tk.PT_SCALE, 3, 2, 5)
    if mode == LIT_REQUEST_OPERATION:
        return tk.check_native_literal_arithmetic_request(p, tk.PT_MUL, 3, 2, 5)
    if mode == LIT_REQUEST_LEFT:
        return tk.check_native_literal_arithmetic_request(p, tk.PT_ADD, 4, 2, 5)
    if mode == LIT_REQUEST_RIGHT:
        return tk.check_native_literal_arithmetic_request(p, tk.PT_ADD, 3, 3, 5)
    if mode == LIT_REQUEST_RESULT:
        return tk.check_native_literal_arithmetic_request(p, tk.PT_ADD, 3, 2, 6)
    if mode == LIT_ZERO_DIVISOR:
        return literal_arithmetic_accepts(tk.PT_DIV, 8, 0, 0)
    if mode == LIT_NEGATIVE_DIVISOR:
        return literal_arithmetic_accepts(tk.PT_MOD, 8, -1, 0)
    if mode == LIT_NEGATIVE_OPERAND:
        return literal_arithmetic_accepts(tk.PT_ADD, -1, 2, 1)
    if mode == LIT_NEGATIVE_CANDIDATE:
        return literal_arithmetic_accepts(tk.PT_ADD, 3, 2, -1)
    if mode >= LIT_DIV_WRONG_PROJECTION and mode <= LIT_DIV_SOURCE_OPERAND:
        if not literal_arithmetic_accepts(tk.PT_DIV, 7, 3, 2):
            return True
        # Keep the requested metadata and candidate fixed, but use another
        # complete, well-formed operation spine. Division is asymmetric.
        var operation = tk.PT_MOD if mode == LIT_DIV_WRONG_PROJECTION else tk.PT_DIV
        var dividend = 8 if mode == LIT_DIV_SOURCE_OPERAND else 7
        var divisor = 4 if mode == LIT_DIV_WRONG_PREDECESSOR else 3
        var forged = make_literal_arithmetic_proof(operation, dividend, divisor, 2)
        forged.operation = tk.PT_DIV
        forged.left_value = 7
        forged.right_value = 3
        return tk.check_native_literal_arithmetic_request(forged, tk.PT_DIV, 7, 3, 2)
    var nat = tk.pt_add(p.bundle, tk.PT_NAT)
    var zero = tk.pt_add(p.bundle, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    if nat < 0 or zero < 0:
        return True
    if mode == LIT_SOURCE_ENDPOINT:
        p.source = zero
    elif mode == LIT_RESULT_ENDPOINT:
        p.result = zero
    elif mode == LIT_PROOF_ENDPOINT:
        p.proof = tk.pt_add(p.bundle, tk.PT_REFL, zero)
        p.bundle.root = p.proof
    elif mode == LIT_PROOF_BOUNDARY:
        p.proof = p.bundle.count
    elif mode == LIT_SOURCE_BOUNDARY:
        p.source = p.bundle.count
    elif mode == LIT_RESULT_BOUNDARY:
        p.result = p.bundle.count
    elif mode == LIT_NEGATIVE_COUNT:
        p.bundle.count = -1
    elif mode == LIT_OVERSIZED_COUNT:
        p.bundle.count = 1025
    elif mode >= LIT_UNUSED_MALFORMED and mode <= LIT_BOOTSTRAP_PRIVATE:
        var added: Int
        if mode == LIT_UNUSED_MALFORMED:
            added = tk.pt_add(p.bundle, tk.PT_APP, nat, nat)
        elif mode == LIT_BOOTSTRAP_LITERAL:
            added = tk.pt_add(p.bundle, tk.PT_LITERAL, value=5)
        elif mode == LIT_BOOTSTRAP_SUCCESSOR:
            added = tk.pt_add(p.bundle, tk.PT_SUCC, zero)
        elif mode == LIT_BOOTSTRAP_OPERATOR:
            added = tk.pt_add(p.bundle, tk.PT_ADD, zero, zero)
        elif mode == LIT_BOOTSTRAP_AFFINE:
            added = tk.pt_add(p.bundle, tk.PT_AFFINE_EQ, zero, zero)
        elif mode == LIT_BOOTSTRAP_ORDER:
            added = tk.pt_add(p.bundle, tk.PT_LITERAL_LE, zero, zero)
        elif mode == LIT_BOOTSTRAP_EXPORT:
            added = tk.pt_add(p.bundle, tk.PT_ARITHMETIC, value=tk.DEC_NAT_NONNEG)
        else:
            added = tk.pt_add(p.bundle, 31, value=0)
        if added < 0:
            return True
    elif mode == LIT_MISSING_DEFINITION:
        p.bundle.definition_count = 8
    elif mode == LIT_EXTRA_DEFINITION:
        if tk.pt_declare_definition(p.bundle, nat, zero) != 9:
            return True
    elif mode == LIT_CYCLIC_DEFINITION or mode == LIT_FORWARD_DEFINITION:
        var index = 0 if mode == LIT_CYCLIC_DEFINITION else 8
        var reference = tk.pt_add(p.bundle, tk.PT_DEFINED_REF, value=index)
        if reference < 0:
            return True
        p.bundle.definition_bodies[0] = reference
    elif mode == LIT_OPEN_DEFINITION:
        var open_body = tk.pt_add(p.bundle, tk.PT_VAR, nat)
        if open_body < 0:
            return True
        p.bundle.definition_bodies[0] = open_body
    elif mode == LIT_WRONG_DECLARED_TYPE or mode == LIT_ALTERED_NAT:
        var proposition = tk.pt_add(p.bundle, tk.PT_PROP)
        if proposition < 0:
            return True
        if mode == LIT_WRONG_DECLARED_TYPE:
            p.bundle.definition_types[0] = proposition
        else:
            p.bundle.nat_constructor_types[0] = proposition
    elif mode == LIT_NEGATIVE_DEFINITIONS:
        p.bundle.definition_count = -1
    elif mode == LIT_OVERSIZED_DEFINITIONS:
        p.bundle.definition_count = 1025
    elif mode == LIT_WRONG_OPERATION:
        p.operation = tk.PT_LE
    elif mode == LIT_STORED_LEFT:
        p.left_value = 4
    elif mode == LIT_STORED_RIGHT:
        p.right_value = 3
    elif mode == LIT_STORED_RESULT:
        p.candidate_value = 6
    elif mode == LIT_NEGATIVE_SOURCE:
        p.source = -1
    elif mode == LIT_NEGATIVE_RESULT:
        p.result = -1
    elif mode == LIT_NEGATIVE_PROOF:
        p.proof = -1
    elif mode == LIT_UNKNOWN_SYNTAX or mode == LIT_UNUSED_FALSE_REFLEXIVITY:
        var added = tk.pt_add(p.bundle, 999) if mode == LIT_UNKNOWN_SYNTAX else tk.pt_add(p.bundle, tk.PT_REFL, nat)
        if added < 0:
            return True
    elif mode == LIT_WRONG_ADD_DEFINITION:
        # A well-typed constant-zero Add is still the wrong checked definition.
        var declared = p.bundle.nodes[p.bundle.definition_types[3]]
        var second = p.bundle.nodes[declared.right]
        if declared.kind != tk.PT_PI or second.kind != tk.PT_PI:
            return True
        var inner = tk.pt_add(p.bundle, tk.PT_LAM, second.left, zero)
        var body = tk.pt_add(p.bundle, tk.PT_LAM, declared.left, inner)
        if body < 0:
            return True
        p.bundle.definition_bodies[3] = body
    else:
        return True
    return tk.check_native_literal_arithmetic(p)


def literal_arithmetic_resource_controls_accepts() -> Bool:
    var small = make_literal_arithmetic_proof[24](tk.PT_ADD, 3, 2, 5)
    var full = make_literal_arithmetic_proof(tk.PT_ADD, 3, 2, 5)
    return (not tk.check_native_literal_arithmetic(small)
        and not tk.check_native_literal_arithmetic_request[1024, 64](full, tk.PT_ADD, 3, 2, 5)
        and tk.check_native_literal_arithmetic_request(full, tk.PT_ADD, 3, 2, 5))


def _program() -> tk.FinishingProgram:
    return tk.FinishingProgram(3, 5, 15, 0, 1, 15, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)


def literal_arithmetic_hook_accepts() -> Bool:
    var empty = tk.empty_proof_terms[128]()
    var s = tk._make_state[128, 8192](_program(), empty)
    tk._enable_native_arithmetic(s)
    if not tk._check_inductive_environment(s):
        return False
    return (tk._checked_literal_arithmetic(s, tk.PT_ADD, 3, 2, 5)
        and tk._checked_literal_arithmetic(s, tk.PT_MUL, 3, 2, 6)
        and tk._checked_literal_arithmetic(s, tk.PT_DIV, 8, 3, 2)
        and tk._checked_literal_arithmetic(s, tk.PT_MOD, 8, 3, 2)
        and not tk._checked_literal_arithmetic(s, tk.PT_ADD, 3, 2, 6)
        and not tk._checked_literal_arithmetic(s, tk.PT_MUL, 3, 2, 5)
        and not tk._checked_literal_arithmetic(s, tk.PT_DIV, 8, 3, 3)
        and not tk._checked_literal_arithmetic(s, tk.PT_MOD, 8, 3, 1)
        and not tk._checked_literal_arithmetic(s, tk.PT_ADD, 3, 2, -1)
        and s.fuel > 0)


def literal_arithmetic_preserves_raw_case(value: Int, mode: Int) -> Bool:
    var empty = tk.empty_proof_terms[128]()
    var s = tk._make_state[128, 8192](_program(), empty)
    tk._enable_native_arithmetic(s)
    if not tk._check_inductive_environment(s):
        return False
    var nat = tk._put(s, tk.PT_NAT)
    var a = tk._put(s, tk.PT_LITERAL, value=value)
    var one = tk._put(s, tk.PT_LITERAL, value=1)
    var expression = tk._put(s, tk.PT_ADD, a, one)
    if mode == 1:
        expression = tk._put(s, tk.PT_SCALE, a, value=2)
    elif mode == 2:
        expression = tk._put(s, tk.PT_ADD, a, tk._put(s, tk.PT_LITERAL, value=0))
    elif mode == 3:
        expression = tk._put(s, tk.PT_SCALE, a, value=1)
    if expression < 0 or tk._infer(s, expression) != nat:
        return False
    var result = tk._whnf(s, expression)
    if result < 0 or tk._infer(s, result) != nat:
        return False
    var n = s.syntax.nodes[result]
    if value == Int.MAX and (mode == 0 or mode == 1):
        # Nat arithmetic exceeds the host word rather than wrapping to a
        # plausible smaller literal. Finite conversion need not normalize it.
        return n.kind != tk.PT_LITERAL and s.fuel > 0
    if n.kind == tk.PT_LITERAL:
        # Independent decimal expectations at the reproduced carry boundary.
        var expected = value
        if value == Int(4294967296) and mode == 0:
            expected = Int(4294967297)
        elif value == Int(4294967296) and mode == 1:
            expected = Int(8589934592)
        elif mode == 0:
            return False
        elif mode == 1:
            return False
        return n.value == expected and s.fuel > 0
    return s.fuel > 0


def literal_arithmetic_symbolic_opacity_accepts() -> Bool:
    var empty = tk.empty_proof_terms[128]()
    var s = tk._make_state[128, 8192](_program(), empty)
    tk._enable_native_arithmetic(s)
    if not tk._check_inductive_environment(s):
        return False
    var nat = tk._put(s, tk.PT_NAT)
    var x = tk._put(s, tk.PT_VAR, nat)
    var zero = tk._put(s, tk.PT_LITERAL, value=0)
    var expressions: List[Int] = [tk._put(s, tk.PT_ADD, zero, x),
        tk._put(s, tk.PT_SCALE, x, value=0),
        tk._put(s, tk.PT_SCALE, x, value=1),
        tk._put(s, tk.PT_DIV, x, value=1),
        tk._put(s, tk.PT_MOD, x, value=1)]
    for expression in expressions:
        if tk._infer(s, expression) != nat:
            return False
        var reduced = tk._whnf(s, expression)
        if (reduced < 0 or tk._infer(s, reduced) != nat
            or s.syntax.nodes[reduced].kind == tk.PT_LITERAL or reduced == x):
            return False
    return s.fuel > 0
