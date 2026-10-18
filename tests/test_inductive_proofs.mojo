"""Checked indexed Prop declarations and their exact finishing admission.

Generic admission checks the supported inductive fragment. Matching the driver
definition is a separate obligation: an admissible inductive is not necessarily
the intended finishing predicate.
"""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import FinishingProgram, make_finishing_program
from llm_mojo.kernels.finishing_typed_derivation import (
    FINISHING_TYPED_PROOF_CAPACITY, FinishingTypedProofs,
    make_finishing_typed_proofs, check_finishing_typed_proofs, termination_type,
)

comptime INDUCTIVE_CAPACITY = 128
comptime NEGATIVE_RECURSION = 0
comptime NESTED_RECURSION = 1
comptime RECURSIVE_INDEX = 2
comptime WRONG_RESULT_HEAD = 3
comptime UNSATURATED_RESULT = 4
comptime OVERSATURATED_RESULT = 5
comptime ILL_TYPED_CONSTRUCTOR = 6
comptime OPEN_CONSTRUCTOR = 7
comptime UNKNOWN_FAMILY = 8
comptime UNKNOWN_CONSTRUCTOR = 9
comptime CONSTRUCTOR_CYCLE = 10
comptime FORWARD_CONSTRUCTOR = 11
comptime UNSUPPORTED_EXPRESSION = 12
comptime BAD_FAMILY_ARITY = 13
comptime RETIRED_DONE_ATOM = 14
comptime RETIRED_DONE_STOP = 15
comptime RETIRED_DONE_STEP = 16
comptime CONSTRUCTOR_TYPE_REFERENCE = 17
comptime HIGHER_ORDER_INDEX = 18
comptime DECLARATION_PAYLOAD = 19
comptime BAD_DEFINITION_COUNT = 20
comptime MALFORMED_RECURSIVE_PREMISE = 21
comptime UNUSED_MALFORMED_CONSTRUCTOR = 22

comptime ALTERNATE_UNCONDITIONAL_STOP = 0
comptime ALTERNATE_STOP_GUARD = 1
comptime ALTERNATE_STEP_ROUND = 2
comptime ALTERNATE_STEP_COUNT = 3


@fieldwise_init
struct InductiveCase(Copyable, Movable):
    var bundle: tk.ProofTermBundle[INDUCTIVE_CAPACITY]
    var expected: Int


def _at[capacity: Int](mut p: tk.ProofTermBundle[capacity], family: Int, index: Int) -> Int:
    return tk.pt_add(p, tk.PT_APP, family, index)


def _pi[capacity: Int](mut p: tk.ProofTermBundle[capacity], binder: Int, result: Int) -> Int:
    return tk.pt_add(p, tk.PT_PI, binder, result)


def generic_indexed_case(mode: Int = -1) -> InductiveCase:
    # A family with one Nat index, unlike Done's pair of indices. Its recursive
    # constructor advances by two, unrelated to the finishing-loop stride.
    var p = tk.empty_proof_terms[INDUCTIVE_CAPACITY]()
    _ = tk.pt_declare_family(p, 1)
    var nat = tk.pt_add(p, tk.PT_NAT)
    var zero = tk.pt_add(p, tk.PT_LITERAL, value=0)
    var two = tk.pt_add(p, tk.PT_LITERAL, value=2)
    var family = tk.pt_add(p, tk.PT_FAMILY, value=0)
    var at_zero = _at(p, family, zero)
    _ = tk.pt_declare_constructor(p, at_zero)
    var index = tk.pt_add(p, tk.PT_VAR, nat)
    var at_index = _at(p, family, index)
    var recursive_type = at_index
    if mode == NEGATIVE_RECURSION:
        var h = tk.pt_add(p, tk.PT_VAR, at_index)
        recursive_type = _pi(p, h, tk.pt_add(p, tk.PT_EQ, zero, zero))
    elif mode == NESTED_RECURSION:
        var h = tk.pt_add(p, tk.PT_VAR, tk.pt_add(p, tk.PT_EQ, zero, zero))
        recursive_type = _pi(p, h, at_index)
    elif mode == MALFORMED_RECURSIVE_PREMISE:
        recursive_type = family  # Missing the required Nat index.
    var previous = tk.pt_add(p, tk.PT_VAR, recursive_type)
    var next_index = tk.pt_add(p, tk.PT_ADD, index, two)
    if mode == RECURSIVE_INDEX:
        # Well-typed after beta reduction, but proof-dependent indices belong
        # outside this direct-recursion fragment and are conservatively rejected.
        var h = tk.pt_add(p, tk.PT_VAR, at_index)
        var function = tk.pt_add(p, tk.PT_LAM, h, zero)
        next_index = tk.pt_add(p, tk.PT_APP, function, previous)
    var result = _at(p, family, next_index)
    if mode == WRONG_RESULT_HEAD:
        result = tk.pt_add(p, tk.PT_EQ, index, index)
    elif mode == UNSATURATED_RESULT:
        result = family
    elif mode == OVERSATURATED_RESULT:
        result = _at(p, result, zero)
    elif mode == ILL_TYPED_CONSTRUCTOR:
        p.nodes[previous].left = zero
    elif mode == OPEN_CONSTRUCTOR:
        # index remains free in the registered constructor type.
        _ = tk.pt_declare_constructor(p, _pi(p, previous, result))
    elif mode == HIGHER_ORDER_INDEX:
        var arg = tk.pt_add(p, tk.PT_VAR, nat)
        var function = tk.pt_add(p, tk.PT_VAR, _pi(p, arg, nat))
        var applied = tk.pt_add(p, tk.PT_APP, function, index)
        result = _pi(p, function, _at(p, family, applied))
    if mode != OPEN_CONSTRUCTOR:
        _ = tk.pt_declare_constructor(p, _pi(p, index, _pi(p, previous, result)))
    var constructor = tk.pt_add(p, tk.PT_CONSTRUCTOR, value=0)
    var step = tk.pt_add(p, tk.PT_CONSTRUCTOR, value=1)
    p.root = tk.pt_add(p, tk.PT_APP, tk.pt_add(p, tk.PT_APP, step, zero), constructor)
    var expected = _at(p, family, two)
    if mode == UNKNOWN_FAMILY:
        p.nodes[family].value = 2
    elif mode == UNKNOWN_CONSTRUCTOR:
        p.nodes[constructor].value = 2
    elif mode == CONSTRUCTOR_CYCLE:
        p.constructor_types[0] = constructor
    elif mode == FORWARD_CONSTRUCTOR:
        p.constructor_types[0] = p.count
    elif mode == UNSUPPORTED_EXPRESSION:
        _ = tk.pt_add(p, 99999)
    elif mode == BAD_FAMILY_ARITY:
        p.family_arity = -2
    elif mode == RETIRED_DONE_ATOM:
        _ = tk.pt_add(p, 14, zero, zero)
    elif mode == RETIRED_DONE_STOP:
        _ = tk.pt_add(p, tk.PT_CONST, value=5)
    elif mode == RETIRED_DONE_STEP:
        _ = tk.pt_add(p, tk.PT_CONST, value=6)
    elif mode == CONSTRUCTOR_TYPE_REFERENCE:
        p.constructor_types[0] = -1
    elif mode == DECLARATION_PAYLOAD:
        p.nodes[family].left = zero
    elif mode == BAD_DEFINITION_COUNT:
        p.constructor_count = INDUCTIVE_CAPACITY + 1
    elif mode == UNUSED_MALFORMED_CONSTRUCTOR:
        _ = tk.pt_declare_constructor(p, tk.pt_add(p, tk.PT_EQ, zero, zero))
    return InductiveCase(p^, expected)


def generic_nullary_case() -> InductiveCase:
    var p = tk.empty_proof_terms[INDUCTIVE_CAPACITY]()
    _ = tk.pt_declare_family(p, 0)
    var family = tk.pt_add(p, tk.PT_FAMILY, value=0)
    _ = tk.pt_declare_constructor(p, family)
    _ = tk.pt_declare_constructor(p, family)
    p.root = tk.pt_add(p, tk.PT_CONSTRUCTOR, value=1)
    return InductiveCase(p^, family)


def generic_empty_definition() -> tk.ProofTermBundle[INDUCTIVE_CAPACITY]:
    var p = tk.empty_proof_terms[INDUCTIVE_CAPACITY]()
    _ = tk.pt_declare_family(p, 0)
    return p^


def generic_three_index_case() -> InductiveCase:
    var p = tk.empty_proof_terms[INDUCTIVE_CAPACITY]()
    _ = tk.pt_declare_family(p, 3)
    var nat = tk.pt_add(p, tk.PT_NAT)
    var family = tk.pt_add(p, tk.PT_FAMILY, value=0)
    var a = tk.pt_add(p, tk.PT_VAR, nat)
    var b = tk.pt_add(p, tk.PT_VAR, nat)
    var c = tk.pt_add(p, tk.PT_VAR, nat)
    var result = _at(p, _at(p, _at(p, family, a), b), c)
    _ = tk.pt_declare_constructor(p, _pi(p, a, _pi(p, b, _pi(p, c, result))))
    var zero = tk.pt_add(p, tk.PT_LITERAL, value=0)
    var one = tk.pt_add(p, tk.PT_LITERAL, value=1)
    var two = tk.pt_add(p, tk.PT_LITERAL, value=2)
    var constructor = tk.pt_add(p, tk.PT_CONSTRUCTOR, value=0)
    p.root = _at(p, _at(p, _at(p, constructor, zero), one), two)
    var expected = _at(p, _at(p, _at(p, family, zero), one), two)
    return InductiveCase(p^, expected)


def alternate_done_proofs(program: FinishingProgram, mode: Int) -> FinishingTypedProofs[FINISHING_TYPED_PROOF_CAPACITY]:
    # All alternatives are admissible positive definitions; none describes the
    # exact driver. Definition matching must reject them before proof admission.
    var proofs = make_finishing_typed_proofs(program)
    var original_stop = proofs.bundle.constructor_types[0]
    if mode == ALTERNATE_UNCONDITIONAL_STOP:
        # Retain the checked base proof and discard the original termination
        # proof. The replacement root below is valid for the altered definition.
        proofs.bundle.count = proofs.base_root + 1
    var nat = tk.pt_add(proofs.bundle, tk.PT_NAT)
    var one = tk.pt_add(proofs.bundle, tk.PT_LITERAL, value=1)
    var q = tk.pt_add(proofs.bundle, tk.PT_VAR, nat)
    var n = tk.pt_add(proofs.bundle, tk.PT_VAR, nat)
    var family = tk.pt_add(proofs.bundle, tk.PT_FAMILY, value=0)
    var done = _at(proofs.bundle, _at(proofs.bundle, family, q), n)
    var base = tk.pt_add(proofs.bundle, tk.PT_BASE, q)
    var stop = done
    if mode != ALTERNATE_UNCONDITIONAL_STOP:
        var guard = tk.pt_add(proofs.bundle, tk.PT_LE, base, n)
        var h = tk.pt_add(proofs.bundle, tk.PT_VAR, guard)
        stop = _pi(proofs.bundle, h, stop)
    proofs.bundle.constructor_types[0] = _pi(proofs.bundle, q, _pi(proofs.bundle, n, stop))
    if mode == ALTERNATE_UNCONDITIONAL_STOP:
        var zero = tk.pt_add(proofs.bundle, tk.PT_LITERAL, value=0)
        var constructor = tk.pt_add(proofs.bundle, tk.PT_CONSTRUCTOR, value=0)
        var finished = _at(proofs.bundle, _at(proofs.bundle, constructor, zero), n)
        proofs.termination_root = tk.pt_add(proofs.bundle, tk.PT_LAM, n, finished)
        proofs.bundle.root = proofs.termination_root
    if mode == ALTERNATE_STEP_ROUND or mode == ALTERNATE_STEP_COUNT:
        # Keep the genuine STOP signature and alter only STEP's recursive index.
        proofs.bundle.constructor_types[0] = original_stop
        var guard = tk.pt_add(proofs.bundle, tk.PT_LT, base, n)
        var h = tk.pt_add(proofs.bundle, tk.PT_VAR, guard)
        var next_q: Int
        var next_n = n
        if mode == ALTERNATE_STEP_ROUND:
            next_q = tk.pt_add(proofs.bundle, tk.PT_ADD, q, tk.pt_add(proofs.bundle, tk.PT_LITERAL, value=2))
        else:
            next_q = tk.pt_add(proofs.bundle, tk.PT_ADD, q, one)
            next_n = tk.pt_add(proofs.bundle, tk.PT_ADD, n, one)
        var recursive = _at(proofs.bundle, _at(proofs.bundle, family, next_q), next_n)
        var g = tk.pt_add(proofs.bundle, tk.PT_VAR, recursive)
        var step = _pi(proofs.bundle, g, done)
        step = _pi(proofs.bundle, h, step)
        step = _pi(proofs.bundle, n, step)
        proofs.bundle.constructor_types[1] = _pi(proofs.bundle, q, step)
    return proofs^


def alternate_done_root_is_typed(program: FinishingProgram) -> Bool:
    var proofs = alternate_done_proofs(program, ALTERNATE_UNCONDITIONAL_STOP)
    var expected = termination_type(proofs.bundle)
    return tk.check_proof_terms(program, proofs.bundle, expected)


def check_generic_definitions() raises:
    comptime program = make_finishing_program(3, 5, 15)
    comptime indexed = generic_indexed_case()
    comptime assert tk.check_inductive_definition(program, indexed.bundle), "inductive definition rejected"
    comptime assert tk.check_proof_terms(program, indexed.bundle, indexed.expected), "proof term rejected"
    comptime nullary = generic_nullary_case()
    comptime assert tk.check_inductive_definition(program, nullary.bundle), "inductive definition rejected"
    comptime assert tk.check_proof_terms(program, nullary.bundle, nullary.expected), "proof term rejected"
    comptime empty = generic_empty_definition()
    comptime assert tk.check_inductive_definition(program, empty), "inductive definition rejected"
    # Declaring an empty predicate supplies no evidence that it is inhabited.
    comptime assert not tk.check_proof_terms(program, empty, -1), "empty definition supplied evidence"
    comptime triple = generic_three_index_case()
    comptime assert tk.check_inductive_definition(program, triple.bundle), "inductive definition rejected"
    comptime assert tk.check_proof_terms(program, triple.bundle, triple.expected), "proof term rejected"
    comptime assert alternate_done_root_is_typed(program), "alternate inductive proof unexpectedly rejected"
    for mode in range(UNUSED_MALFORMED_CONSTRUCTOR + 1):
        var malformed = generic_indexed_case(mode)
        # Some errors concern constructor proof references, rather than the
        # declaration itself. Whole-proof admission must reject both kinds.
        assert_equal(tk.check_proof_terms(program, malformed.bundle, malformed.expected), False)
    for mode in range(ALTERNATE_STEP_COUNT + 1):
        var altered = alternate_done_proofs(program, mode)
        assert_equal(tk.check_inductive_definition(program, altered.bundle), True)
        assert_equal(check_finishing_typed_proofs(program, altered), False)


def main() raises:
    check_generic_definitions()
    print("Indexed inductive definitions checked; alternate driver definitions rejected")
