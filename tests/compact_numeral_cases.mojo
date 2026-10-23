"""Independent definitions and adversarial certificates for compact Nat numerals.

The raw definitions below use constructor applications and structural recursion.
They never use compact literals, PT_SUCC, arithmetic operators or reflection to
certify their own meaning. Every native mutant also has a positive baseline.
"""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.nat_numerals import (
    make_numeral_library, make_numeral_view, NumeralConversion, NumeralView,
    NC_REFL, NC_BETA, NC_DELTA, NC_NAT_IOTA, NC_CONG_LEFT, NC_CONG_RIGHT,
    NC_SYMM, NC_TRANS,
    NUMERAL_LIBRARY_CAPACITY, NUMERAL_VIEW_CAPACITY, NUMERAL_CONVERSION_CAPACITY,
)

comptime NUM_ZERO_WRONG = 0
comptime NUM_SUCC_IDENTITY = 1
comptime NUM_DOUBLE_IDENTITY = 2
comptime NUM_WRONG_FAMILY = 3
comptime NUM_WRONG_TYPE = 4
comptime NUM_BOOTSTRAP_LITERAL = 5
comptime NUM_BOOTSTRAP_SUCCESSOR = 6
comptime NUM_BOOTSTRAP_OPERATOR = 7
comptime NUM_BOOTSTRAP_PRIVATE = 8
comptime NUM_CYCLIC_DEFINITION = 9
comptime NUM_OPEN_DEFINITION = 10
comptime NUM_FORWARD_DEFINITION = 11
comptime NUM_UNKNOWN_RULE = 12
comptime NUM_LEFT_ENDPOINT = 13
comptime NUM_RIGHT_ENDPOINT = 14
comptime NUM_TRANS_MIDDLE = 15
comptime NUM_CONGRUENCE_ARGUMENT = 16
comptime NUM_IOTA_ENDPOINT = 17
comptime NUM_BETA_ENDPOINT = 18
comptime NUM_DELTA_ENDPOINT = 19
comptime NUM_FORWARD_PREMISE = 20
comptime NUM_CYCLIC_PREMISE = 21
comptime NUM_UNUSED_MALFORMED = 22
comptime NUM_VALUE_BINDING = 23
comptime NUM_STORED_VALUE = 24
comptime NUM_ROOT_REFERENCE = 25
comptime NUM_DENOTATION_BINDING = 26
comptime NUM_CONSTRUCTOR_BINDING = 27
comptime NUM_PREDECESSOR_BINDING = 28
comptime NUM_CERTIFICATE_DEFINITION = 29
comptime NUM_CERTIFICATE_NAT = 30
comptime NUM_CONGRUENCE_CAPTURE = 31
comptime NUM_BOOTSTRAP_NUMERAL_PRIVATE = 32
comptime NUM_FALSE_REFLEXIVITY = 33
comptime NUM_LEFT_CONGRUENCE_ARGUMENT = 34
comptime NUM_IOTA_NEUTRAL_MAJOR = 35
comptime NUM_BETA_CAPTURE = 36
comptime NUM_DELTA_WRONG_BINDING = 37
comptime NUM_DENOTATION_SOURCE_BOUNDARY = 38
comptime NUM_CONSTRUCTOR_SOURCE_BOUNDARY = 39
comptime NUM_PREDECESSOR_SOURCE_BOUNDARY = 40
comptime NUM_NEGATIVE_DERIVATION_COUNT = 41
comptime NUM_OVERSIZED_DERIVATION_COUNT = 42
comptime NUM_NEGATIVE_SOURCE_COUNT = 43
comptime NUM_OVERSIZED_SOURCE_COUNT = 44
comptime COMPACT_NUMERAL_MUTANTS = 45


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _zero[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _succ[C: Int](mut p: tk.ProofTermBundle[C], predecessor: Int) -> Int:
    return _n(p, tk.PT_APP,
        _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), predecessor)


def _independent_definitions[C: Int](mut p: tk.ProofTermBundle[C]):
    var nat = _n(p, tk.PT_NAT)
    var zero = _zero(p)
    var argument = _n(p, tk.PT_VAR, nat)
    var function_type = _n(p, tk.PT_PI, argument, nat)
    var successor = _n(p, tk.PT_LAM, argument, _succ(p, argument))
    var input = _n(p, tk.PT_VAR, nat)
    var predecessor = _n(p, tk.PT_VAR, nat)
    var accumulator = _n(p, tk.PT_VAR, nat)
    var step = _n(p, tk.PT_LAM, predecessor,
        _n(p, tk.PT_LAM, accumulator, _succ(p, _succ(p, accumulator))))
    var doubled = _n(p, tk.PT_LAM, input,
        _n(p, tk.PT_NAT_REC, input, zero, step))
    p.definition_count = 3
    p.definition_types[0] = nat
    p.definition_types[1] = function_type
    p.definition_types[2] = function_type
    p.definition_bodies[0] = zero
    p.definition_bodies[1] = successor
    p.definition_bodies[2] = doubled


def largest_positive_power_of_two() -> Int:
    var value = 1
    while value <= Int.MAX // 2:
        value *= 2
    return value


def numeral_library_accepts(mode: Int = -1, independent: Bool = False) -> Bool:
    # Spare source capacity permits independently rebuilt alpha-renamed bodies.
    var library = make_numeral_library[2 * NUMERAL_LIBRARY_CAPACITY]()
    if independent:
        _independent_definitions(library.bundle)
    var nat = _n(library.bundle, tk.PT_NAT)
    var zero = _zero(library.bundle)
    if mode == NUM_ZERO_WRONG:
        library.bundle.definition_bodies[0] = _succ(library.bundle, zero)
    elif mode == NUM_SUCC_IDENTITY or mode == NUM_DOUBLE_IDENTITY:
        var argument = _n(library.bundle, tk.PT_VAR, nat)
        library.bundle.definition_bodies[1 if mode == NUM_SUCC_IDENTITY else 2] = (
            _n(library.bundle, tk.PT_LAM, argument, argument))
    elif mode == NUM_WRONG_FAMILY:
        library.bundle.definition_bodies[0] = _n(library.bundle,
            tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID + 1, value=0)
    elif mode == NUM_WRONG_TYPE:
        library.bundle.definition_types[1] = nat
    elif mode == NUM_BOOTSTRAP_LITERAL:
        _ = _n(library.bundle, tk.PT_LITERAL, value=0)
    elif mode == NUM_BOOTSTRAP_SUCCESSOR:
        _ = _n(library.bundle, tk.PT_SUCC, zero)
    elif mode == NUM_BOOTSTRAP_OPERATOR:
        _ = _n(library.bundle, tk.PT_ADD, zero, zero)
    elif mode == NUM_BOOTSTRAP_PRIVATE:
        # Caller syntax must not manufacture an arithmetic private reference.
        _ = _n(library.bundle, 31, value=0)
    elif mode == NUM_BOOTSTRAP_NUMERAL_PRIVATE:
        _ = _n(library.bundle, 32, value=0)
    elif mode == NUM_CYCLIC_DEFINITION:
        library.bundle.definition_bodies[0] = _n(library.bundle,
            tk.PT_DEFINED_REF, value=0)
    elif mode == NUM_OPEN_DEFINITION:
        library.bundle.definition_bodies[0] = _n(library.bundle, tk.PT_VAR, nat)
    elif mode == NUM_FORWARD_DEFINITION:
        library.bundle.definition_bodies[0] = _n(library.bundle,
            tk.PT_DEFINED_REF, value=2)
    return tk.check_native_numeral_library(library)


def _first_rule[C: Int, D: Int](view: NumeralView[C, D], kind: Int) -> Int:
    for index in range(view.count):
        if view.conversions[index].kind == kind:
            return index
    return -1


def numeral_view_accepts(value: Int, mode: Int = -1,
    recompose: Bool = False) -> Bool:
    var view = make_numeral_view(value)
    if view.root < 0 or view.count <= 0:
        # A malformed producer result must fail the mutant fixture, not count
        # as evidence that the checker rejected its intended mutation.
        return mode >= 0
    # A source generator exhausting its own storage is a failed positive
    # baseline. Deliberate count mutations below still reach the checker.
    if (view.bundle.count >= NUMERAL_VIEW_CAPACITY
        or view.count + 8 >= NUMERAL_CONVERSION_CAPACITY):
        return mode >= 0
    var zero = _zero(view.bundle)
    if recompose:
        # A separately assembled proof circuit exercises symmetry and
        # transitivity rather than requiring the producer's exact transcript.
        var old_root = view.root
        var old = view.conversions[old_root]
        var first = view.count
        view.conversions[first] = NumeralConversion(NC_SYMM,
            old.right, old.left, old_root, -1)
        view.conversions[first + 1] = NumeralConversion(NC_SYMM,
            old.left, old.right, first, -1)
        view.conversions[first + 2] = NumeralConversion(NC_REFL,
            old.left, old.left, -1, -1)
        view.conversions[first + 3] = NumeralConversion(NC_TRANS,
            old.left, old.right, first + 2, first + 1)
        var successor = _n(view.bundle, tk.PT_CONSTRUCTOR,
            extra=tk.NAT_FAMILY_ID, value=1)
        var applied_left = _n(view.bundle, tk.PT_APP, successor, old.left)
        var applied_right = _n(view.bundle, tk.PT_APP, successor, old.right)
        view.conversions[first + 4] = NumeralConversion(NC_CONG_RIGHT,
            applied_left, applied_right, old_root, -1)
        view.conversions[first + 5] = NumeralConversion(NC_REFL,
            successor, successor, -1, -1)
        view.conversions[first + 6] = NumeralConversion(NC_CONG_LEFT,
            applied_left, applied_left, first + 5, -1)
        view.count += 7
        view.root = first + 3
    if mode == NUM_UNKNOWN_RULE:
        view.conversions[view.root].kind = 999
    elif mode == NUM_LEFT_ENDPOINT:
        view.conversions[view.root].left = zero
    elif mode == NUM_RIGHT_ENDPOINT:
        view.conversions[view.root].right = zero
    elif mode == NUM_TRANS_MIDDLE:
        var old = view.conversions[view.root]
        var added = view.count
        view.conversions[added] = NumeralConversion(NC_REFL, zero, zero, -1, -1)
        view.conversions[added + 1] = NumeralConversion(NC_TRANS,
            old.left, old.right, view.root, added)
        view.count += 2
        view.root = added + 1
    elif mode >= NUM_CONGRUENCE_ARGUMENT and mode <= NUM_DELTA_ENDPOINT:
        var kind = NC_CONG_RIGHT
        if mode == NUM_IOTA_ENDPOINT:
            kind = NC_NAT_IOTA
        elif mode == NUM_BETA_ENDPOINT:
            kind = NC_BETA
        elif mode == NUM_DELTA_ENDPOINT:
            kind = NC_DELTA
        var index = _first_rule(view, kind)
        if index < 0:
            return True
        view.conversions[index].right = _succ(view.bundle, zero) if mode == NUM_IOTA_ENDPOINT else zero
    elif mode == NUM_FORWARD_PREMISE or mode == NUM_CYCLIC_PREMISE:
        var old = view.conversions[view.root]
        view.conversions[view.root] = NumeralConversion(NC_SYMM,
            old.left, old.right, view.count if mode == NUM_FORWARD_PREMISE else view.root, -1)
    elif mode == NUM_UNUSED_MALFORMED:
        view.conversions[view.count] = NumeralConversion(999, zero, zero, -1, -1)
        view.count += 1
    elif mode == NUM_VALUE_BINDING:
        return tk.check_native_numeral_view(value - 1, view)
    elif mode == NUM_STORED_VALUE:
        view.value = value - 1
    elif mode == NUM_ROOT_REFERENCE:
        view.root = view.count
    elif mode == NUM_DENOTATION_BINDING:
        view.denotation = zero
    elif mode == NUM_CONSTRUCTOR_BINDING:
        view.constructor = zero
    elif mode == NUM_PREDECESSOR_BINDING:
        view.predecessor = zero
    elif mode == NUM_CERTIFICATE_DEFINITION:
        var nat = _n(view.bundle, tk.PT_NAT)
        var argument = _n(view.bundle, tk.PT_VAR, nat)
        view.bundle.definition_bodies[2] = _n(view.bundle, tk.PT_LAM, argument, argument)
    elif mode == NUM_CERTIFICATE_NAT:
        view.bundle.nat_constructor_types[0] = _n(view.bundle, tk.PT_PROP)
    elif mode == NUM_CONGRUENCE_CAPTURE:
        var nat = _n(view.bundle, tk.PT_NAT)
        var bound = _n(view.bundle, tk.PT_VAR, nat)
        var captured = _n(view.bundle, tk.PT_VAR, nat)
        var function = _n(view.bundle, tk.PT_LAM, bound, _succ(view.bundle, bound))
        var wrong_function = _n(view.bundle, tk.PT_LAM, bound, _succ(view.bundle, captured))
        var old = view.conversions[view.root]
        view.conversions[view.count] = NumeralConversion(NC_CONG_RIGHT,
            _n(view.bundle, tk.PT_APP, function, old.left),
            _n(view.bundle, tk.PT_APP, wrong_function, old.right), view.root, -1)
        view.count += 1
    elif mode == NUM_FALSE_REFLEXIVITY:
        view.conversions[view.count] = NumeralConversion(NC_REFL,
            zero, _succ(view.bundle, zero), -1, -1)
        view.count += 1
    elif mode == NUM_LEFT_CONGRUENCE_ARGUMENT:
        var successor = _n(view.bundle, tk.PT_CONSTRUCTOR,
            extra=tk.NAT_FAMILY_ID, value=1)
        var proof = view.count
        view.conversions[proof] = NumeralConversion(NC_REFL,
            successor, successor, -1, -1)
        view.conversions[proof + 1] = NumeralConversion(NC_CONG_LEFT,
            _n(view.bundle, tk.PT_APP, successor, zero),
            _n(view.bundle, tk.PT_APP, successor, _succ(view.bundle, zero)), proof, -1)
        view.count += 2
    elif mode == NUM_IOTA_NEUTRAL_MAJOR:
        var nat = _n(view.bundle, tk.PT_NAT)
        var major = _n(view.bundle, tk.PT_VAR, nat)
        var predecessor = _n(view.bundle, tk.PT_VAR, nat)
        var hypothesis = _n(view.bundle, tk.PT_VAR, nat)
        var step = _n(view.bundle, tk.PT_LAM, predecessor,
            _n(view.bundle, tk.PT_LAM, hypothesis, zero))
        var recursor = _n(view.bundle, tk.PT_NAT_REC, major, zero, step)
        view.conversions[view.count] = NumeralConversion(NC_NAT_IOTA,
            recursor, zero, -1, -1)
        view.count += 1
    elif mode == NUM_BETA_CAPTURE:
        var nat = _n(view.bundle, tk.PT_NAT)
        var bound = _n(view.bundle, tk.PT_VAR, nat)
        var captured = _n(view.bundle, tk.PT_VAR, nat)
        var identity = _n(view.bundle, tk.PT_LAM, bound, bound)
        view.conversions[view.count] = NumeralConversion(NC_BETA,
            _n(view.bundle, tk.PT_APP, identity, zero), captured, -1, -1)
        view.count += 1
    elif mode == NUM_DELTA_WRONG_BINDING:
        view.conversions[view.count] = NumeralConversion(NC_DELTA,
            _n(view.bundle, tk.PT_DEFINED_REF, value=0), _succ(view.bundle, zero), -1, -1)
        view.count += 1
    elif mode == NUM_DENOTATION_SOURCE_BOUNDARY:
        view.denotation = view.bundle.count
    elif mode == NUM_CONSTRUCTOR_SOURCE_BOUNDARY:
        view.constructor = view.bundle.count
    elif mode == NUM_PREDECESSOR_SOURCE_BOUNDARY:
        view.predecessor = view.bundle.count
    elif mode == NUM_NEGATIVE_DERIVATION_COUNT:
        view.count = -1
    elif mode == NUM_OVERSIZED_DERIVATION_COUNT:
        view.count = NUMERAL_CONVERSION_CAPACITY + 1
    elif mode == NUM_NEGATIVE_SOURCE_COUNT:
        view.bundle.count = -1
    elif mode == NUM_OVERSIZED_SOURCE_COUNT:
        view.bundle.count = NUMERAL_VIEW_CAPACITY + 1
    # These are actual proof/source sizes, not a bound on the represented Nat.
    return tk.check_native_numeral_view(value, view)


def compact_numeral_mutant_accepts(mode: Int) -> Bool:
    if mode <= NUM_FORWARD_DEFINITION or mode == NUM_BOOTSTRAP_NUMERAL_PRIVATE:
        if not numeral_library_accepts():
            return True
        return numeral_library_accepts(mode)
    if not numeral_primitive_baseline_accepts():
        return True
    return numeral_view_accepts(16, mode, recompose=True)


def numeral_primitive_baseline_accepts() -> Bool:
    var view = make_numeral_view(16)
    if view.root < 0 or view.count <= 0:
        return False
    view.conversions[view.count] = NumeralConversion(NC_REFL,
        view.denotation, view.denotation, -1, -1)
    view.count += 1
    for kind in range(8):
        if _first_rule(view, kind) < 0:
            return False
    # Require a checked representative of every rule before mutating one.
    return tk.check_native_numeral_view(16, view) and numeral_view_accepts(16, recompose=True)


def _program() -> tk.FinishingProgram:
    return tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)


def _put_succ[C: Int](mut s: tk._TypedState[C], predecessor: Int) -> Int:
    return tk._put(s, tk.PT_APP,
        tk._put(s, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), predecessor)


def _predecessor_recursor[C: Int](mut s: tk._TypedState[C], major: Int) -> Int:
    var nat = tk._put(s, tk.PT_NAT)
    var predecessor = tk._put(s, tk.PT_VAR, nat)
    var hypothesis = tk._put(s, tk.PT_VAR, nat)
    var step = tk._put(s, tk.PT_LAM, predecessor,
        tk._put(s, tk.PT_LAM, hypothesis, predecessor))
    var zero = tk._put(s, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    # Ignoring IH observes one layer, even when major is Int.MAX.
    return tk._put(s, tk.PT_NAT_REC, major, zero, step)


def _branch_observer[C: Int](mut s: tk._TypedState[C], major: Int,
    base_result: Int, successor_result: Int, proposition: Int) -> Int:
    var nat = tk._put(s, tk.PT_NAT)
    var index = tk._put(s, tk.PT_VAR, nat)
    var motive = tk._put(s, tk.PT_LAM, index, proposition)
    var predecessor = tk._put(s, tk.PT_VAR, nat)
    var hypothesis = tk._put(s, tk.PT_VAR, proposition)
    var step = tk._put(s, tk.PT_LAM, predecessor,
        tk._put(s, tk.PT_LAM, hypothesis, successor_result))
    var result = tk._put(s, tk.PT_ELIMINATOR, value=tk.NAT_FAMILY_ID)
    result = tk._put(s, tk.PT_APP, result, motive)
    result = tk._put(s, tk.PT_APP, result, base_result)
    result = tk._put(s, tk.PT_APP, result, step)
    return tk._put(s, tk.PT_APP, result, major)


def _natural_is[C: Int](mut s: tk._TypedState[C], node: Int, value: Int) -> Bool:
    if node < 0:
        return False
    var expected = tk._put(s, tk.PT_LITERAL, value=value)
    var nat = tk._put(s, tk.PT_NAT)
    return (tk._infer(s, node) == nat and tk._infer(s, expected) == nat
        and tk._numeral_meaning_equal(s, node, expected))


def _predecessor_is[C: Int](mut s: tk._TypedState[C], node: Int, value: Int) -> Bool:
    # Independent decimal oracles avoid reproducing a word subtraction defect
    # in both the implementation and its test at the 32-bit boundary.
    if value == Int(4294967296):
        return _natural_is(s, node, Int(4294967295))
    if value == Int(8589934592):
        return _natural_is(s, node, Int(8589934591))
    return _natural_is(s, node, 0 if value == 0 else value - 1)


def _successor_is[C: Int](mut s: tk._TypedState[C], node: Int, value: Int) -> Bool:
    if value == Int(4294967294):
        return _natural_is(s, node, Int(4294967295))
    if value < Int.MAX:
        return _natural_is(s, node, value + 1)
    if node < 0 or s.syntax.nodes[node].kind == tk.PT_LITERAL:
        return False
    var predecessor = tk._put(s, tk.PT_LITERAL, value=value)
    var expected = tk._put(s, tk.PT_SUCC, predecessor)
    return (tk._infer(s, node) == tk._put(s, tk.PT_NAT)
        and tk._numeral_meaning_equal(s, node, expected))


def bounded_native_numeral_path(value: Int, path: Int) -> Bool:
    var empty = tk.empty_proof_terms[128]()
    var s = tk._make_state[128, 8192](_program(), empty)
    # Numeral certification precedes and does not need arithmetic admission.
    tk._enable_native_numerals(s)
    if not tk._check_inductive_environment(s):
        return False
    var nat = tk._put(s, tk.PT_NAT)
    var literal = tk._put(s, tk.PT_LITERAL, value=value)
    var initial = s.syntax.count
    var accepted = False
    if path == 0:
        accepted = tk._infer(s, literal) == nat
    elif path == 1:
        var exposed = tk._checked_numeral_view(s, literal)
        if exposed >= 0:
            var n = s.syntax.nodes[exposed]
            if value == 0:
                accepted = n.kind == tk.PT_CONSTRUCTOR and n.extra == tk.NAT_FAMILY_ID and n.value == 0
            elif n.kind == tk.PT_APP:
                var constructor = s.syntax.nodes[n.left]
                accepted = (constructor.kind == tk.PT_CONSTRUCTOR
                    and constructor.extra == tk.NAT_FAMILY_ID and constructor.value == 1
                    and _predecessor_is(s, n.right, value))
    elif path == 2 or path == 3:
        var successor = tk._put(s, tk.PT_SUCC, literal)
        if tk._infer(s, successor) != nat:
            return False
        var reduced = tk._whnf(s, successor, False) if path == 2 else tk._operator_administrative_whnf(s, successor)
        accepted = _successor_is(s, reduced, value)
        if path == 2:
            var raw_successor = _put_succ(s, literal)
            var raw_reduced = tk._whnf(s, raw_successor, False)
            accepted = accepted and _successor_is(s, raw_reduced, value)
    elif path == 4:
        var folded = _predecessor_recursor(s, literal)
        var result = tk._whnf(s, folded, False)
        accepted = _predecessor_is(s, result, value)
        var raw_folded = _predecessor_recursor(s, _put_succ(s, literal))
        accepted = accepted and _natural_is(s, tk._whnf(s, raw_folded, False), value)
    elif path == 5:
        var zero = tk._put(s, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
        var proposition = tk._put(s, tk.PT_EQ, zero, zero)
        var base_result = tk._put(s, tk.PT_VAR, proposition)
        var successor_result = tk._put(s, tk.PT_VAR, proposition)
        var folded = _branch_observer(s, literal, base_result, successor_result, proposition)
        accepted = tk._whnf(s, folded, False) == (base_result if value == 0 else successor_result)
        var raw_folded = _branch_observer(s, _put_succ(s, literal), base_result, successor_result, proposition)
        accepted = accepted and tk._whnf(s, raw_folded, False) == successor_result
    elif path == 6:
        var symbolic = tk._put(s, tk.PT_VAR, nat)
        var successor = tk._put(s, tk.PT_SUCC, symbolic)
        var exposed = tk._checked_numeral_view(s, successor)
        if exposed >= 0:
            var n = s.syntax.nodes[exposed]
            if n.kind == tk.PT_APP:
                var constructor = s.syntax.nodes[n.left]
                accepted = (constructor.kind == tk.PT_CONSTRUCTOR
                    and constructor.extra == tk.NAT_FAMILY_ID and constructor.value == 1
                    and n.right == symbolic)
        accepted = accepted and tk._infer(s, successor) == nat
        accepted = accepted and tk._whnf(s, _predecessor_recursor(s, successor), False) == symbolic
        var administered = tk._operator_administrative_whnf(s, successor)
        accepted = (accepted and administered >= 0
            and administered == successor
            and tk._infer(s, administered) == nat)
        var zero = tk._put(s, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
        var proposition = tk._put(s, tk.PT_EQ, zero, zero)
        var base_result = tk._put(s, tk.PT_VAR, proposition)
        var successor_result = tk._put(s, tk.PT_VAR, proposition)
        var folded = _branch_observer(s, successor, base_result, successor_result, proposition)
        accepted = accepted and tk._whnf(s, folded, False) == successor_result
    elif path == 7:
        # Type the actual binary denotation; full unary evaluation is not
        # needed to certify or view this symbolic natural number.
        var denotation = tk._numeral_denotation(s, value)
        accepted = denotation >= 0 and tk._infer(s, denotation) == nat
    # This measures the checker workspace after each actual path. Its bound
    # applies equally to MAX and a power-of-two carry chain, not to value itself.
    return accepted and s.syntax.count - initial < 4096 and s.syntax.count < 8192 and s.fuel > 0


def bounded_native_numeral_paths(value: Int) -> Bool:
    for path in range(8):
        if not bounded_native_numeral_path(value, path):
            return False
    return True


def numeral_caller_namespace_accepts() -> Bool:
    var p = tk.empty_proof_terms[128]()
    var nat = _n(p, tk.PT_NAT)
    var one = _succ(p, _zero(p))
    var argument = _n(p, tk.PT_VAR, nat)
    var identity_type = _n(p, tk.PT_PI, argument, nat)
    var identity = _n(p, tk.PT_LAM, argument, argument)
    # Caller IDs0..2 intentionally disagree with all three numeral aliases.
    p.definition_count = 3
    p.definition_types[0] = nat
    p.definition_bodies[0] = one
    p.definition_types[1] = identity_type
    p.definition_types[2] = identity_type
    p.definition_bodies[1] = identity
    p.definition_bodies[2] = identity
    var s = tk._make_state[128, 8192](_program(), p)
    tk._enable_native_numerals(s)
    if not tk._check_inductive_environment(s):
        return False
    var caller = tk._put(s, tk.PT_DEFINED_REF, value=0)
    if not _natural_is(s, tk._whnf(s, caller, False), 1):
        return False
    var actual_zero = tk._numeral_denotation(s, 0)
    var actual_one = tk._numeral_denotation(s, 1)
    return (_natural_is(s, tk._whnf(s, actual_zero, False), 0)
        and _natural_is(s, tk._whnf(s, actual_one, False), 1)
        and s.syntax.count < 8192 and s.fuel > 0)


def numeral_resource_controls_accepts() -> Bool:
    # Deliberately inadequate proof storage is a resource rejection, distinct
    # from the well-formed baseline plus semantic mutations above.
    var small_library = make_numeral_library[0]()
    if tk.check_native_numeral_library(small_library):
        return False
    var value = largest_positive_power_of_two()
    var small_view = make_numeral_view[24, 16](value)
    return (not tk.check_native_numeral_view(value, small_view)
        and numeral_view_accepts(value))


def duplicate_literal_successor_accepts(value: Int = 17) -> Bool:
    if value < 0 or value == Int.MAX:
        return False
    var p = tk.empty_proof_terms[128]()
    var first = _n(p, tk.PT_LITERAL, value=value)
    # Bypass pt_add intentionally: source DAGs may legally contain duplicates.
    var duplicate = p.count
    p.nodes[duplicate] = tk.ProofTerm(tk.PT_LITERAL, -1, -1, -1, value)
    p.count += 1
    if first == duplicate:
        return False
    p.root = _n(p, tk.PT_SUCC, duplicate)
    var s = tk._make_state[128, 8192](_program(), p)
    tk._enable_native_numerals(s)
    if not tk._check_inductive_environment(s):
        return False
    var initial = s.syntax.count
    var nat = tk._put(s, tk.PT_NAT)
    return (tk._infer(s, p.root) == nat
        and _successor_is(s, tk._whnf(s, p.root, False), value)
        and _successor_is(s, tk._operator_administrative_whnf(s, p.root), value)
        and s.syntax.count - initial < 4096 and s.fuel > 0)


def independent_boundary_numerals_accepts() -> Bool:
    var empty = tk.empty_proof_terms[128]()
    var s = tk._make_state[128, 8192](_program(), empty)
    tk._enable_native_numerals(s)
    if not tk._check_inductive_environment(s):
        return False
    var initial = s.syntax.count
    var target = tk._put(s, tk.PT_LITERAL, value=Int(4294967296))
    var exposed = tk._checked_numeral_view(s, target)
    if exposed < 0 or s.syntax.nodes[exposed].kind != tk.PT_APP:
        return False
    var constructor = s.syntax.nodes[s.syntax.nodes[exposed].left]
    if (constructor.kind != tk.PT_CONSTRUCTOR
        or constructor.extra != tk.NAT_FAMILY_ID or constructor.value != 1
        or not _natural_is(s, s.syntax.nodes[exposed].right, Int(4294967295))):
        return False
    if not _natural_is(s, tk._whnf(s, _predecessor_recursor(s, target), False), Int(4294967295)):
        return False
    var below = tk._put(s, tk.PT_LITERAL, value=Int(4294967294))
    var successor = tk._put(s, tk.PT_SUCC, below)
    return (_natural_is(s, tk._whnf(s, successor, False), Int(4294967295))
        and _natural_is(s, tk._operator_administrative_whnf(s, successor), Int(4294967295))
        and s.syntax.count - initial < 4096 and s.fuel > 0)


def numeral_meaning_rejections_accept() -> Bool:
    var p = tk.empty_proof_terms[128]()
    var nat = _n(p, tk.PT_NAT)
    p.definition_count = 1
    p.definition_types[0] = nat
    p.definition_bodies[0] = _succ(p, _zero(p))
    var s = tk._make_state[128, 8192](_program(), p)
    tk._enable_native_numerals(s)
    if not tk._check_inductive_environment(s):
        return False
    var initial = s.syntax.count
    var zero = tk._put(s, tk.PT_LITERAL, value=0)
    var seventeen = tk._put(s, tk.PT_LITERAL, value=17)
    var eighteen = tk._put(s, tk.PT_LITERAL, value=18)
    var below = tk._put(s, tk.PT_LITERAL, value=Int(4294967295))
    var boundary = tk._put(s, tk.PT_LITERAL, value=Int(4294967296))
    var maximum = tk._put(s, tk.PT_LITERAL, value=Int.MAX)
    var overflow = _put_succ(s, maximum)
    var a = tk._put(s, tk.PT_VAR, nat)
    var b = tk._put(s, tk.PT_VAR, nat)
    var successor_a = tk._put(s, tk.PT_SUCC, a)
    var successor_b = tk._put(s, tk.PT_SUCC, b)
    var caller = tk._put(s, tk.PT_DEFINED_REF, value=0)
    var unfolded = tk._whnf(s, caller, False)
    if not _natural_is(s, unfolded, 1):
        return False
    return (not tk._numeral_meaning_equal(s, seventeen, eighteen)
        and not tk._numeral_meaning_equal(s, below, boundary)
        and not tk._numeral_meaning_equal(s, overflow, maximum)
        and not tk._numeral_meaning_equal(s, a, b)
        and not tk._numeral_meaning_equal(s, successor_a, successor_b)
        and not tk._numeral_meaning_equal(s, unfolded, zero)
        and s.syntax.count - initial < 4096 and s.fuel > 0)
