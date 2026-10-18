"""Type checking, binder hygiene, and symbolic finishing proofs in Mojo.

The positive propositions quantify over naturals; concrete endpoint checks
exercise descriptor arithmetic without enumerating the count domain.
"""
from std.testing import assert_equal
from proof_state_storage_cases import check_proof_state_storage
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import FinishingProgram, make_finishing_program
from llm_mojo.kernels.finishing_typed_derivation import (
    FINISHING_TYPED_PROOF_CAPACITY, FinishingTypedProofs,
    make_finishing_typed_proofs, check_finishing_typed_proofs,
)

comptime BASIC_CAPACITY = 96
comptime BAD_APPLICATION = 0
comptime BAD_BINDER_TYPE = 1
comptime OPEN_ROOT = 2
comptime UNKNOWN_CONSTANT = 3
comptime FORWARD_REFERENCE = 4
comptime NEGATIVE_REFERENCE = 5
comptime CYCLIC_REFERENCE = 6
comptime WRONG_EQUALITY = 7
comptime WRONG_TARGET = 8
comptime CAPTURE = 9
comptime UNUSED_MALFORMED = 10
comptime BAD_INDUCTION_BASE = 11
comptime BAD_INDUCTION_STEP = 12
comptime BAD_INDUCTION_MOTIVE = 13
comptime BAD_SUBSTITUTION = 14
comptime MISSING_ROOT = 15
comptime UNKNOWN_EXPRESSION = 16
comptime BASE_NOT_AFFINE = 17
comptime NEGATIVE_NATURAL = 18
comptime EXTRA_PAYLOAD = 19
comptime BINDER_DEPENDENCY = 20
comptime TYPE_DOMAIN = 21
comptime UNTYPED_LAMBDA_RESULT = 22
comptime BAD_TYPED_BASE_ROOT = 0
comptime BAD_TYPED_TERMINATION_ROOT = 1
comptime BAD_TYPED_BASE_INDUCTION = 2
comptime BAD_TYPED_TERMINATION_INDUCTION = 3
comptime BAD_TYPED_INDUCTION_TARGET = 4
comptime BAD_TYPED_STOP_DECLARATION = 5


@fieldwise_init
struct ProofCase(Copyable, Movable):
    var bundle: tk.ProofTermBundle[BASIC_CAPACITY]
    var expected: Int


def identity_case() -> ProofCase:
    var p = tk.empty_proof_terms[BASIC_CAPACITY]()
    var nat = tk.pt_add(p, tk.PT_NAT)
    var argument = tk.pt_add(p, tk.PT_VAR, nat)
    var target_argument = tk.pt_add(p, tk.PT_VAR, nat)
    var expected = tk.pt_add(p, tk.PT_PI, target_argument, nat)
    p.root = tk.pt_add(p, tk.PT_LAM, argument, argument)
    return ProofCase(p^, expected)


def reflexivity_case() -> ProofCase:
    var p = tk.empty_proof_terms[BASIC_CAPACITY]()
    var nat = tk.pt_add(p, tk.PT_NAT)
    var argument = tk.pt_add(p, tk.PT_VAR, nat)
    var evidence = tk.pt_add(p, tk.PT_REFL, argument)
    p.root = tk.pt_add(p, tk.PT_LAM, argument, evidence)
    # A distinct binder requires alpha-equivalence of dependent Pi types.
    var target_argument = tk.pt_add(p, tk.PT_VAR, nat)
    var equation = tk.pt_add(p, tk.PT_EQ, target_argument, target_argument)
    var expected = tk.pt_add(p, tk.PT_PI, target_argument, equation)
    return ProofCase(p^, expected)


def applied_reflexivity_case() -> ProofCase:
    var result = reflexivity_case()
    var literal = tk.pt_add(result.bundle, tk.PT_LITERAL, value=37)
    result.bundle.root = tk.pt_add(result.bundle, tk.PT_APP, result.bundle.root, literal)
    result.expected = tk.pt_add(result.bundle, tk.PT_EQ, literal, literal)
    return result^


def beta_type_case() -> ProofCase:
    var p = tk.empty_proof_terms[BASIC_CAPACITY]()
    var nat = tk.pt_add(p, tk.PT_NAT)
    var argument = tk.pt_add(p, tk.PT_VAR, nat)
    var equation = tk.pt_add(p, tk.PT_EQ, argument, argument)
    var predicate = tk.pt_add(p, tk.PT_LAM, argument, equation)
    var zero = tk.pt_add(p, tk.PT_LITERAL, value=0)
    p.root = tk.pt_add(p, tk.PT_REFL, zero)
    # Expected is an application producing a proposition, rather than an
    # already reduced Eq node. Admission must compare types after beta reduction.
    var expected = tk.pt_add(p, tk.PT_APP, predicate, zero)
    return ProofCase(p^, expected)


def noncapturing_application_case() -> ProofCase:
    var p = tk.empty_proof_terms[BASIC_CAPACITY]()
    var nat = tk.pt_add(p, tk.PT_NAT)
    var outer = tk.pt_add(p, tk.PT_VAR, nat)
    var argument = tk.pt_add(p, tk.PT_VAR, nat)
    var inner = tk.pt_add(p, tk.PT_VAR, nat)
    var function = tk.pt_add(p, tk.PT_LAM, argument,
        tk.pt_add(p, tk.PT_LAM, inner, tk.pt_add(p, tk.PT_REFL, argument)))
    var application = tk.pt_add(p, tk.PT_APP, function, outer)
    p.root = tk.pt_add(p, tk.PT_LAM, outer, application)
    var a = tk.pt_add(p, tk.PT_VAR, nat)
    var b = tk.pt_add(p, tk.PT_VAR, nat)
    var expected = tk.pt_add(p, tk.PT_PI, a,
        tk.pt_add(p, tk.PT_PI, b, tk.pt_add(p, tk.PT_EQ, a, a)))
    return ProofCase(p^, expected)


def induction_case(mode: Int = -1) -> ProofCase:
    var p = tk.empty_proof_terms[BASIC_CAPACITY]()
    var nat = tk.pt_add(p, tk.PT_NAT)
    var zero = tk.pt_add(p, tk.PT_LITERAL, value=0)
    var variable = tk.pt_add(p, tk.PT_VAR, nat)
    var predicate_body = tk.pt_add(p, tk.PT_EQ, variable, variable)
    if mode == BAD_INDUCTION_MOTIVE:
        predicate_body = nat
    var motive = tk.pt_add(p, tk.PT_LAM, variable, predicate_body)
    var base = tk.pt_add(p, tk.PT_REFL, zero)
    if mode == BAD_INDUCTION_BASE:
        base = zero
    var k = tk.pt_add(p, tk.PT_VAR, nat)
    var ih_type = tk.pt_add(p, tk.PT_EQ, k, k)
    var ih = tk.pt_add(p, tk.PT_VAR, ih_type)
    var successor = tk.pt_add(p, tk.PT_CONSTRUCTOR, extra=1, value=1)
    var next = tk.pt_add(p, tk.PT_APP, successor, k)
    var step_body = tk.pt_add(p, tk.PT_REFL, next)
    if mode == BAD_INDUCTION_STEP:
        step_body = ih
    var step = tk.pt_add(p, tk.PT_LAM, k, tk.pt_add(p, tk.PT_LAM, ih, step_body))
    var target = tk.pt_add(p, tk.PT_LITERAL, value=37)
    var eliminator = tk.pt_add(p, tk.PT_ELIMINATOR, value=1)
    var application = tk.pt_add(p, tk.PT_APP, eliminator, motive)
    application = tk.pt_add(p, tk.PT_APP, application, base)
    application = tk.pt_add(p, tk.PT_APP, application, step)
    p.root = tk.pt_add(p, tk.PT_APP, application, target)
    var expected = tk.pt_add(p, tk.PT_EQ, target, target)
    return ProofCase(p^, expected)


def malformed_case(mode: Int) -> ProofCase:
    if mode == BAD_INDUCTION_BASE or mode == BAD_INDUCTION_STEP or mode == BAD_INDUCTION_MOTIVE:
        return induction_case(mode)
    var result = identity_case()
    var nat = tk.pt_add(result.bundle, tk.PT_NAT)
    var zero = tk.pt_add(result.bundle, tk.PT_LITERAL, value=0)
    var one = tk.pt_add(result.bundle, tk.PT_LITERAL, value=1)
    var root = result.bundle.root
    if mode == BAD_APPLICATION:
        result.bundle.root = tk.pt_add(result.bundle, tk.PT_APP, root, nat)
        result.expected = nat
    elif mode == BAD_BINDER_TYPE:
        result.bundle.nodes[result.bundle.nodes[root].left].left = zero
    elif mode == OPEN_ROOT:
        result.bundle.root = result.bundle.nodes[root].right
        result.expected = nat
    elif mode == UNKNOWN_CONSTANT:
        result.bundle.root = tk.pt_add(result.bundle, tk.PT_CONST, value=99999)
        result.expected = nat
    elif mode == FORWARD_REFERENCE:
        result.bundle.nodes[root].right = result.bundle.count
    elif mode == NEGATIVE_REFERENCE:
        result.bundle.nodes[root].right = -2
    elif mode == CYCLIC_REFERENCE:
        result.bundle.nodes[root].right = root
    elif mode == WRONG_EQUALITY:
        result.bundle.root = tk.pt_add(result.bundle, tk.PT_AFFINE_EQ, zero, one)
        result.expected = tk.pt_add(result.bundle, tk.PT_EQ, zero, one)
    elif mode == WRONG_TARGET:
        result = reflexivity_case()
        nat = tk.pt_add(result.bundle, tk.PT_NAT)
        one = tk.pt_add(result.bundle, tk.PT_LITERAL, value=1)
        var q = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        var successor = tk.pt_add(result.bundle, tk.PT_ADD, q, one)
        result.expected = tk.pt_add(result.bundle, tk.PT_PI, q,
            tk.pt_add(result.bundle, tk.PT_EQ, q, successor))
    elif mode == CAPTURE:
        var outer = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        var argument = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        # Reusing outer's binder token under the applied lambda would capture
        # its argument during beta substitution; binder tokens are identities.
        var inner = tk.pt_add(result.bundle, tk.PT_LAM, outer,
            tk.pt_add(result.bundle, tk.PT_REFL, argument))
        var function = tk.pt_add(result.bundle, tk.PT_LAM, argument, inner)
        var application = tk.pt_add(result.bundle, tk.PT_APP, function, outer)
        result.bundle.root = tk.pt_add(result.bundle, tk.PT_LAM, outer, application)
        var a = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        var b = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        result.expected = tk.pt_add(result.bundle, tk.PT_PI, a,
            tk.pt_add(result.bundle, tk.PT_PI, b,
                tk.pt_add(result.bundle, tk.PT_EQ, a, a)))
    elif mode == UNUSED_MALFORMED:
        _ = tk.pt_add(result.bundle, tk.PT_APP, nat, zero)
    elif mode == BAD_SUBSTITUTION:
        var q = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        var predicate = tk.pt_add(result.bundle, tk.PT_LAM, q,
            tk.pt_add(result.bundle, tk.PT_EQ, q, q))
        var equality = tk.pt_add(result.bundle, tk.PT_REFL, zero)
        var wrong_evidence = tk.pt_add(result.bundle, tk.PT_REFL, one)
        result.bundle.root = tk.pt_add(result.bundle, tk.PT_SUBST, predicate, equality, wrong_evidence)
        result.expected = tk.pt_add(result.bundle, tk.PT_EQ, zero, zero)
    elif mode == MISSING_ROOT:
        result.bundle.root = -1
    elif mode == UNKNOWN_EXPRESSION:
        _ = tk.pt_add(result.bundle, 99999)
    elif mode == BASE_NOT_AFFINE:
        var q = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        var base = tk.pt_add(result.bundle, tk.PT_BASE, q)
        var scaled = tk.pt_add(result.bundle, tk.PT_SCALE, q, value=15)
        var false_reflection = tk.pt_add(result.bundle, tk.PT_AFFINE_EQ, base, scaled)
        result.bundle.root = tk.pt_add(result.bundle, tk.PT_LAM, q, false_reflection)
        result.expected = tk.pt_add(result.bundle, tk.PT_PI, q,
            tk.pt_add(result.bundle, tk.PT_EQ, base, scaled))
    elif mode == NEGATIVE_NATURAL:
        _ = tk.pt_add(result.bundle, tk.PT_LITERAL, value=-1)
    elif mode == EXTRA_PAYLOAD:
        _ = tk.pt_add(result.bundle, tk.PT_LITERAL, value=19, extra=zero)
    elif mode == BINDER_DEPENDENCY:
        var q = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        var assumed = tk.pt_add(result.bundle, tk.PT_VAR,
            tk.pt_add(result.bundle, tk.PT_EQ, q, q))
        # An outer assumption's type still mentions q. Abstracting q while
        # retaining that assumption is invalid even in an unused expression.
        _ = tk.pt_add(result.bundle, tk.PT_LAM, q, assumed)
    elif mode == TYPE_DOMAIN:
        var universe = tk.pt_add(result.bundle, tk.PT_TYPE)
        _ = tk.pt_add(result.bundle, tk.PT_VAR, universe)
    elif mode == UNTYPED_LAMBDA_RESULT:
        var q = tk.pt_add(result.bundle, tk.PT_VAR, nat)
        # The restricted core has no higher universe for the codomain Type.
        _ = tk.pt_add(result.bundle, tk.PT_LAM, q, nat)
    return result^


def _is_nat_induction[capacity: Int](p: tk.ProofTermBundle[capacity], node: Int) -> Bool:
    # Recognize the complete ordinary application spine, not partial applies.
    var head = node
    for _ in range(4):
        if head < 0 or head >= p.count or p.nodes[head].kind != tk.PT_APP:
            return False
        head = p.nodes[head].left
    return (head >= 0 and p.nodes[head].kind == tk.PT_ELIMINATOR
        and p.nodes[head].value == 1)


def typed_nat_induction_count(proofs: FinishingTypedProofs[FINISHING_TYPED_PROOF_CAPACITY]) -> Int:
    var count = 0
    for i in range(proofs.bundle.count):
        if _is_nat_induction(proofs.bundle, i):
            count += 1
    return count


def malformed_typed_proofs(program: FinishingProgram, mode: Int) -> FinishingTypedProofs[FINISHING_TYPED_PROOF_CAPACITY]:
    var result = make_finishing_typed_proofs(program)
    if mode == BAD_TYPED_BASE_ROOT:
        result.base_root = result.termination_root
    elif mode == BAD_TYPED_TERMINATION_ROOT:
        result.termination_root = result.base_root
    else:
        var induction_count = 0
        var original_count = result.bundle.count
        for i in range(original_count):
            if _is_nat_induction(result.bundle, i):
                var step_application = result.bundle.nodes[i].left
                var base_application = result.bundle.nodes[step_application].left
                if mode == BAD_TYPED_BASE_INDUCTION and induction_count == 0:
                    result.bundle.nodes[base_application].right = tk.pt_add(result.bundle, tk.PT_LITERAL, value=0)
                elif mode == BAD_TYPED_TERMINATION_INDUCTION and induction_count == 1:
                    result.bundle.nodes[step_application].right = result.base_root
                elif mode == BAD_TYPED_INDUCTION_TARGET and induction_count == 1:
                    result.bundle.nodes[i].right = tk.pt_add(result.bundle, tk.PT_LITERAL, value=0)
                induction_count += 1
            elif (mode == BAD_TYPED_STOP_DECLARATION and result.bundle.nodes[i].kind == tk.PT_CONSTRUCTOR
                    and result.bundle.nodes[i].extra == -1 and result.bundle.nodes[i].value == 0):
                result.bundle.nodes[i].value = 1
    return result^


def check_core() raises:
    comptime program = make_finishing_program(3, 5, 15)
    comptime identity = identity_case()
    comptime assert tk.check_proof_terms(program, identity.bundle, identity.expected), "proof term rejected"
    comptime reflexivity = reflexivity_case()
    comptime assert tk.check_proof_terms(program, reflexivity.bundle, reflexivity.expected), "proof term rejected"
    comptime applied = applied_reflexivity_case()
    comptime assert tk.check_proof_terms(program, applied.bundle, applied.expected), "proof term rejected"
    comptime beta = beta_type_case()
    comptime assert tk.check_proof_terms(program, beta.bundle, beta.expected), "proof term rejected"
    comptime noncapturing = noncapturing_application_case()
    comptime assert tk.check_proof_terms(program, noncapturing.bundle, noncapturing.expected), "proof term rejected"
    comptime induction = induction_case()
    comptime assert tk.check_proof_terms(program, induction.bundle, induction.expected), "proof term rejected"
    for mode in range(UNTYPED_LAMBDA_RESULT + 1):
        var malformed = malformed_case(mode)
        assert_equal(tk.check_proof_terms(program, malformed.bundle, malformed.expected), False)
    for mode in range(BAD_TYPED_STOP_DECLARATION + 1):
        var malformed = malformed_typed_proofs(program, mode)
        assert_equal(check_finishing_typed_proofs(program, malformed), False)


def check_shape[threads: Int, items: Int]() raises:
    comptime program = make_finishing_program(threads, items, threads * items)
    comptime proofs = make_finishing_typed_proofs(program)
    comptime assert check_finishing_typed_proofs(program, proofs), "proof term rejected"
    comptime assert typed_nat_induction_count(proofs) == 2, "induction mutation target missing"
    var native_proofs = make_finishing_typed_proofs(program)
    assert_equal(check_finishing_typed_proofs(program, native_proofs), True)
    # Test concrete machine endpoints independently of proof-term induction.
    var count = 2147483647
    var rounds = 1 + (count - 1) // program.stride
    var last_base = (rounds - 1) * program.stride
    var exit_base = rounds * program.stride
    assert_equal(last_base < count, True)
    assert_equal(exit_base >= count and exit_base < count + program.stride, True)
    assert_equal(exit_base <= program.integer_limit, True)


def main() raises:
    check_proof_state_storage()
    check_core()
    check_shape[1, 1]()
    check_shape[3, 5]()
    check_shape[32, 2]()
    check_shape[64, 3]()
    check_shape[128, 1]()
    check_shape[3, 65]()
    check_shape[2147483647, 1]()
    check_shape[1, 2147483647]()
    print("Typed proof terms accepted; malformed expressions rejected")
