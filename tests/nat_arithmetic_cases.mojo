"""Independent rejection controls for checked Nat arithmetic definitions."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.nat_arithmetic import make_nat_arithmetic_library, NAT_ARITHMETIC_CAPACITY
from llm_mojo.kernels.nat_arithmetic_terms import (
    NatArithmeticLibrary, _node, _nat, _var, _number, _named, _app,
    _lam, _pi, _truth, _false, _succ, _rec, _plus,
)

comptime ADD_CONSTANT_ZERO = 0
comptime MUL_CONSTANT_ZERO = 1
comptime LE_CONSTANT_TRUE = 2
comptime DIVISION_CONSTANT_STATE = 3
comptime PRED_IDENTITY = 4
comptime MISSING_EXPORT = 5
comptime SWAPPED_EXPORT = 6
comptime FORWARD_DEFINITION = 7
comptime CYCLIC_DEFINITION = 8
comptime OPEN_DEFINITION = 9
comptime WRONG_DECLARED_TYPE = 10
comptime UNUSED_MALFORMED = 11
comptime RETIRED_ARITHMETIC_AXIOM = 12
comptime CIRCULAR_ARITHMETIC_EXPORT = 13
comptime BOOTSTRAP_AFFINE_ORACLE = 14
comptime BOOTSTRAP_LITERAL_ORDER_ORACLE = 15
comptime WRONG_RECURSOR_STEP = 16
comptime RECURSOR_UNIVERSE_CARRIER = 17
comptime ALTERED_NAT_CONSTRUCTOR = 18
comptime DANGLING_DEFINITION_BODY = 19
comptime NEGATIVE_DEFINITION_COUNT = 20
comptime EXCESS_DEFINITION_COUNT = 21
comptime MISSING_CORE_DEFINITION = 22
comptime WRONG_DIVISION_EXPORT = 23
comptime BASE_CONSTANT_ZERO = 24
comptime BASE_SUCCESSOR_ONE = 25
comptime BASE_WRONG_CAPTURE = 26
comptime MISSING_BASE_ZERO_EXPORT = 27
comptime SWAPPED_BASE_EXPORTS = 28
comptime RETIRED_BASE_ZERO_AXIOM = 29
comptime RETIRED_BASE_STEP_AXIOM = 30
comptime CIRCULAR_BASE_EXPORT = 31
comptime BOOTSTRAP_BASE_OPERATOR = 32
comptime WRONG_BASE_DECLARED_TYPE = 33
comptime NAT_ARITHMETIC_MUTANTS = 34


def _preserve_data_for_proof_helpers(
    mut library: NatArithmeticLibrary[NAT_ARITHMETIC_CAPACITY], data: Int,
):
    # Insert an earlier clone immediately after all six data definitions.
    # Proofs refer to the correct clone; the public operation can be mutated
    # while every ordinary definition remains closed and correctly typed.
    var insertion = 6
    var correct_type = library.bundle.definition_types[data]
    var correct_body = library.bundle.definition_bodies[data]
    var previous_count = library.bundle.definition_count
    for offset in range(previous_count - insertion):
        var shifted = previous_count - offset
        library.bundle.definition_types[shifted] = library.bundle.definition_types[shifted - 1]
        library.bundle.definition_bodies[shifted] = library.bundle.definition_bodies[shifted - 1]
    library.bundle.definition_types[insertion] = correct_type
    library.bundle.definition_bodies[insertion] = correct_body
    library.bundle.definition_count += 1
    for j in range(library.bundle.count):
        var node = library.bundle.nodes[j]
        if node.kind == tk.PT_DEFINED_REF:
            if node.value == data:
                node.value = insertion
            elif node.value >= insertion:
                node.value += 1
            library.bundle.nodes[j] = node
    for declaration in range(tk.PROOF_TERM_DECLARATIONS):
        if library.exports[declaration] >= insertion:
            library.exports[declaration] += 1


def nat_arithmetic_mutant(mode: Int = -1) -> NatArithmeticLibrary[NAT_ARITHMETIC_CAPACITY]:
    var library = make_nat_arithmetic_library()
    var zero = _number(library.bundle, 0)
    if mode == ADD_CONSTANT_ZERO or mode == MUL_CONSTANT_ZERO:
        var a = _var(library.bundle, _nat(library.bundle))
        var b = _var(library.bundle, _nat(library.bundle))
        library.bundle.definition_bodies[mode] = _lam(library.bundle, a,
            _lam(library.bundle, b, zero))
    elif mode == LE_CONSTANT_TRUE:
        var a = _var(library.bundle, _nat(library.bundle))
        var b = _var(library.bundle, _nat(library.bundle))
        library.bundle.definition_bodies[2] = _lam(library.bundle, a,
            _lam(library.bundle, b, _truth(library.bundle)))
    elif mode == DIVISION_CONSTANT_STATE:
        # The original zero-state term has the right data type. Discarding
        # the recursive major still yields a well-typed, incorrect function.
        var body = library.bundle.definition_bodies[3]
        var first = library.bundle.nodes[body]
        var second = library.bundle.nodes[first.right]
        var recurrence = library.bundle.nodes[second.right]
        library.bundle.definition_bodies[3] = _lam(library.bundle, first.left,
            _lam(library.bundle, second.left, recurrence.right))
    elif mode == PRED_IDENTITY:
        # Preserve a correct predecessor for proof helpers in a new earlier
        # definition. The erroneous public data operation remains independently
        # typed, so only its canonical body pin should reject this library.
        _preserve_data_for_proof_helpers(library, 4)
        var n = _var(library.bundle, _nat(library.bundle))
        library.bundle.definition_bodies[4] = _lam(library.bundle, n, n)
    elif mode == MISSING_EXPORT:
        library.exports[tk.DEC_LE_TRANS] = -1
    elif mode == SWAPPED_EXPORT:
        library.exports[tk.DEC_NAT_NONNEG] = library.exports[tk.DEC_LE_ADD]
    elif mode == FORWARD_DEFINITION:
        # This later definition has exactly the original closed type/body.
        # Rejection therefore exercises dependency order, not a type mismatch.
        var later = tk.pt_declare_definition(library.bundle,
            library.bundle.definition_types[6], library.bundle.definition_bodies[6])
        library.bundle.definition_bodies[6] = _named(library.bundle, later)
    elif mode == CYCLIC_DEFINITION:
        library.bundle.definition_bodies[6] = _named(library.bundle, 6)
    elif mode == OPEN_DEFINITION:
        library.bundle.definition_bodies[6] = _var(library.bundle,
            library.bundle.definition_types[6])
    elif mode == WRONG_DECLARED_TYPE:
        library.bundle.definition_types[6] = _nat(library.bundle)
    elif mode == UNUSED_MALFORMED:
        _ = _app(library.bundle, _nat(library.bundle), zero)
    elif mode == RETIRED_ARITHMETIC_AXIOM:
        _ = _node(library.bundle, tk.PT_CONST, value=tk.DEC_NAT_NONNEG)
    elif mode == CIRCULAR_ARITHMETIC_EXPORT:
        _ = _node(library.bundle, tk.PT_ARITHMETIC, value=tk.DEC_NAT_NONNEG)
    elif mode == BOOTSTRAP_AFFINE_ORACLE:
        _ = _node(library.bundle, tk.PT_AFFINE_EQ, zero, zero)
    elif mode == BOOTSTRAP_LITERAL_ORDER_ORACLE:
        _ = _node(library.bundle, tk.PT_LITERAL_LE, zero, zero)
    elif mode == WRONG_RECURSOR_STEP:
        var n = _var(library.bundle, _nat(library.bundle))
        var previous = _var(library.bundle, _nat(library.bundle))
        var wrong = _lam(library.bundle, n,
            _lam(library.bundle, previous, _truth(library.bundle)))
        _ = _rec(library.bundle, zero, zero, wrong)
    elif mode == RECURSOR_UNIVERSE_CARRIER:
        var n = _var(library.bundle, _nat(library.bundle))
        var previous = _var(library.bundle, _nat(library.bundle))
        var step = _lam(library.bundle, n, _lam(library.bundle, previous, previous))
        _ = _rec(library.bundle, zero, _node(library.bundle, tk.PT_TYPE), step)
    elif mode == ALTERED_NAT_CONSTRUCTOR:
        library.bundle.nat_constructor_types[1] = _nat(library.bundle)
    elif mode == DANGLING_DEFINITION_BODY:
        library.bundle.definition_bodies[6] = -1
    elif mode == NEGATIVE_DEFINITION_COUNT:
        library.bundle.definition_count = -1
    elif mode == EXCESS_DEFINITION_COUNT:
        library.bundle.definition_count = NAT_ARITHMETIC_CAPACITY + 1
    elif mode == MISSING_CORE_DEFINITION:
        library.bundle.definition_count = 5
    elif mode == WRONG_DIVISION_EXPORT:
        library.exports[tk.DEC_DIV_QUOTIENT_UNIQUE] = library.exports[tk.DEC_DIV_REMAINDER_UNIQUE]
    elif mode == BASE_CONSTANT_ZERO or mode == BASE_SUCCESSOR_ONE or mode == BASE_WRONG_CAPTURE:
        _preserve_data_for_proof_helpers(library, 5)
        var stride = _var(library.bundle, _nat(library.bundle))
        var round = _var(library.bundle, _nat(library.bundle))
        var result = zero
        if mode != BASE_CONSTANT_ZERO:
            var predecessor = _var(library.bundle, _nat(library.bundle))
            var accumulator = _var(library.bundle, _nat(library.bundle))
            var wrong_increment = _number(library.bundle, 1) if mode == BASE_SUCCESSOR_ONE else round
            var step = _lam(library.bundle, predecessor,
                _lam(library.bundle, accumulator, _plus(library.bundle, accumulator, wrong_increment)))
            result = _rec(library.bundle, round, zero, step)
        library.bundle.definition_bodies[5] = _lam(library.bundle, stride,
            _lam(library.bundle, round, result))
    elif mode == MISSING_BASE_ZERO_EXPORT:
        library.exports[tk.DEC_BASE_ZERO] = -1
    elif mode == SWAPPED_BASE_EXPORTS:
        var first = library.exports[tk.DEC_BASE_ZERO]
        library.exports[tk.DEC_BASE_ZERO] = library.exports[tk.DEC_BASE_STEP]
        library.exports[tk.DEC_BASE_STEP] = first
    elif mode == RETIRED_BASE_ZERO_AXIOM or mode == RETIRED_BASE_STEP_AXIOM:
        _ = _node(library.bundle, tk.PT_CONST,
            value=tk.DEC_BASE_ZERO if mode == RETIRED_BASE_ZERO_AXIOM else tk.DEC_BASE_STEP)
    elif mode == CIRCULAR_BASE_EXPORT:
        _ = _node(library.bundle, tk.PT_ARITHMETIC, value=tk.DEC_BASE_STEP)
    elif mode == BOOTSTRAP_BASE_OPERATOR:
        _ = _node(library.bundle, tk.PT_BASE, zero)
    elif mode == WRONG_BASE_DECLARED_TYPE:
        var argument = _var(library.bundle, _nat(library.bundle))
        library.bundle.definition_types[5] = _pi(library.bundle, argument, _nat(library.bundle))
    return library^


def nat_arithmetic_mutant_accepts(mode: Int) -> Bool:
    var library = nat_arithmetic_mutant(mode)
    return tk.check_native_arithmetic_library(library)


def retired_arithmetic_axiom_accepts(declaration: Int) -> Bool:
    var p = tk.empty_proof_terms[32]()
    var zero = _number(p, 0)
    _ = _node(p, tk.PT_CONST, value=declaration)
    p.root = _node(p, tk.PT_REFL, zero)
    var expected = _node(p, tk.PT_EQ, zero, zero)
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms(program, p, expected)


def local_shadow_case(mode: Int = -1) -> Bool:
    # Valid caller definitions may use the same numeric IDs as the fixed
    # library. Their bodies and types are deliberately different operations.
    var p = tk.empty_proof_terms[256]()
    var zero = _number(p, 0)
    for definition in range(3):
        var a = _var(p, _nat(p))
        var b = _var(p, _nat(p))
        var result_type = _node(p, tk.PT_PROP) if definition == 2 else _nat(p)
        var type = _pi(p, a, _pi(p, b, result_type))
        var result = _false(p) if definition == 2 else zero
        _ = tk.pt_declare_definition(p, type, _lam(p, a, _lam(p, b, result)))
    var k = _var(p, _nat(p))
    var dividend = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var r = _var(p, _nat(p))
    var countdown = _var(p, _nat(p))
    var selector = _var(p, _pi(p, q, _pi(p, r, _pi(p, countdown, _nat(p)))))
    var triple_type = _pi(p, selector, _nat(p))
    var triple = _lam(p, selector, _app(p, _app(p, _app(p, selector, zero), zero), zero))
    _ = tk.pt_declare_definition(p, _pi(p, k, _pi(p, dividend, triple_type)),
        _lam(p, k, _lam(p, dividend, triple)))
    var predecessor = _var(p, _nat(p))
    _ = tk.pt_declare_definition(p, _pi(p, predecessor, _nat(p)),
        _lam(p, predecessor, predecessor))
    var local_stride = _var(p, _nat(p))
    var local_round = _var(p, _nat(p))
    _ = tk.pt_declare_definition(p, _pi(p, local_stride, _pi(p, local_round, _nat(p))),
        _lam(p, local_stride, _lam(p, local_round, zero)))
    var n = _var(p, _nat(p))
    var nonnegative = _app(p, _node(p, tk.PT_ARITHMETIC, value=tk.DEC_NAT_NONNEG), n)
    p.root = _lam(p, n, nonnegative)
    var expected = _pi(p, n, _node(p, tk.PT_LE, zero, n))
    if mode == 0:
        # A global checked theorem cannot acquire a caller's False type.
        p.root = _app(p, _node(p, tk.PT_ARITHMETIC, value=tk.DEC_NAT_NONNEG), zero)
        expected = _app(p, _app(p, _named(p, 2), zero), zero)
    elif mode == 1:
        # Core Mul refers to core Add, never to caller definition zero.
        p.root = _lam(p, n, _node(p, tk.PT_REFL, zero))
        expected = _pi(p, n, _node(p, tk.PT_EQ,
            _node(p, tk.PT_MUL, n, _number(p, 1)), zero))
    elif mode == 2:
        # Fixed library references are internal reduction syntax. A caller
        # cannot submit that namespace as a shortcut around admission.
        _ = _node(p, tk.PT_ARITHMETIC + 1, value=2)
    elif mode == 3:
        # A caller's constant DivMod cannot change the global quotient.
        p.root = _lam(p, n, _node(p, tk.PT_REFL, zero))
        expected = _pi(p, n, _node(p, tk.PT_EQ,
            _node(p, tk.PT_DIV, n, value=2), zero))
    elif mode == 4:
        # Caller definition five computes zero. It cannot replace checked Base.
        p.root = _lam(p, n, _node(p, tk.PT_REFL, zero))
        expected = _pi(p, n, _node(p, tk.PT_EQ, _node(p, tk.PT_BASE, n), zero))
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms(program, p, expected)


def aliased_constructor_case(negative: Bool = False) -> Bool:
    var p = tk.empty_proof_terms[96]()
    _ = tk.pt_declare_family(p, 0)
    var family = _node(p, tk.PT_FAMILY, value=tk.PROP_FAMILY_ID)
    var alias = _truth(p)
    if negative:
        var premise = _var(p, family)
        alias = _pi(p, premise, _false(p))
    _ = tk.pt_declare_definition(p, _node(p, tk.PT_PROP), alias)
    var argument = _var(p, _named(p, 0))
    _ = tk.pt_declare_constructor(p, _pi(p, argument, family))
    var zero = _number(p, 0)
    p.root = _node(p, tk.PT_REFL, zero)
    var expected = _node(p, tk.PT_EQ, zero, zero)
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms(program, p, expected)
