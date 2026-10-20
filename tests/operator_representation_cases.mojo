"""Independent raw recipes and adversarial controls for checked operator codecs.

Recipe bodies are spelled in caller syntax, using only the six certified data
definition IDs. No producer recipe or kernel expected-body builder is consulted.
"""
from std.collections import Array
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.nat_arithmetic import make_nat_arithmetic_library
from llm_mojo.kernels.nat_arithmetic_terms import (
    NAT_REP_ADD, NAT_REP_MUL, NAT_REP_LE, NAT_REP_LT, NAT_REP_SCALE,
    NAT_REP_BASE, NAT_REP_DIV, NAT_REP_MOD, NAT_REPRESENTATIONS,
)

comptime REP_ADD_SWAP = 0
comptime REP_MUL_SWAP = 1
comptime REP_LE_REVERSE = 2
comptime REP_LT_WITHOUT_SUCC = 3
comptime REP_SCALE_SWAP = 4
comptime REP_BASE_SWAP = 5
comptime REP_DIV_AS_MOD = 6
comptime REP_MOD_AS_DIV = 7
comptime REP_DIVISOR_OFFSET = 8
comptime REP_REPEATED_SELECTOR = 9
comptime REP_MISSING_INDEX = 10
comptime REP_SWAPPED_INDICES = 11
comptime REP_OPEN_ALIAS = 12
comptime REP_CIRCULAR_ALIAS = 13
comptime REP_BOOTSTRAP_NATIVE = 14
comptime REP_PROOF_EXPORT_ALIAS = 15
comptime OPERATOR_REPRESENTATION_MUTANTS = 16


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _app2[C: Int](mut p: tk.ProofTermBundle[C], f: Int, a: Int, b: Int) -> Int:
    return _n(p, tk.PT_APP, _n(p, tk.PT_APP, f, a), b)


def _named2[C: Int](mut p: tk.ProofTermBundle[C], definition: Int,
    a: Int, b: Int) -> Int:
    return _app2(p, _n(p, tk.PT_DEFINED_REF, value=definition), a, b)


def _raw_selector[C: Int](mut p: tk.ProofTermBundle[C], component: Int,
    repeated: Bool = False) -> Int:
    var nat = _n(p, tk.PT_NAT)
    var q = _n(p, tk.PT_VAR, nat)
    var r = q if repeated else _n(p, tk.PT_VAR, nat)
    var countdown = q if repeated else _n(p, tk.PT_VAR, nat)
    var body = q if component == 0 else r
    return _n(p, tk.PT_LAM, q,
        _n(p, tk.PT_LAM, r, _n(p, tk.PT_LAM, countdown, body)))


def independent_raw_recipe[C: Int](mut p: tk.ProofTermBundle[C], slot: Int,
    mode: Int = -1) -> Int:
    var nat = _n(p, tk.PT_NAT)
    var a = _n(p, tk.PT_VAR, nat)
    var b = _n(p, tk.PT_VAR, nat)
    var body = -1
    if slot == NAT_REP_ADD or slot == NAT_REP_MUL or slot == NAT_REP_SCALE:
        var definition = 0 if slot == NAT_REP_ADD else 1
        var swapped = ((slot == NAT_REP_ADD and mode == REP_ADD_SWAP)
            or (slot == NAT_REP_MUL and mode == REP_MUL_SWAP)
            or (slot == NAT_REP_SCALE and mode == REP_SCALE_SWAP))
        body = _named2(p, definition, b if swapped else a, a if swapped else b)
    elif slot == NAT_REP_LE or slot == NAT_REP_LT:
        var lhs = _n(p, tk.PT_SUCC, a) if slot == NAT_REP_LT and mode != REP_LT_WITHOUT_SUCC else a
        body = _named2(p, 2, b, lhs) if mode == REP_LE_REVERSE and slot == NAT_REP_LE else _named2(p, 2, lhs, b)
    elif slot == NAT_REP_BASE:
        body = _named2(p, 5, b, a) if mode == REP_BASE_SWAP else _named2(p, 5, a, b)
    else:
        var component = 0 if slot == NAT_REP_DIV else 1
        if ((slot == NAT_REP_DIV and mode == REP_DIV_AS_MOD)
            or (slot == NAT_REP_MOD and mode == REP_MOD_AS_DIV)):
            component = 1 - component
        var predecessor = _n(p, tk.PT_SUCC, a) if mode == REP_DIVISOR_OFFSET else a
        var state = _named2(p, 3, predecessor, b)
        body = _n(p, tk.PT_APP, state,
            _raw_selector(p, component, mode == REP_REPEATED_SELECTOR))
    return _n(p, tk.PT_LAM, a, _n(p, tk.PT_LAM, b, body))


def operator_representation_library_accepts(mode: Int = -1,
    rebind_all: Bool = False) -> Bool:
    var library = make_nat_arithmetic_library()
    if rebind_all:
        for slot in range(NAT_REPRESENTATIONS):
            var definition = library.representations[slot]
            library.bundle.definition_bodies[definition] = independent_raw_recipe(library.bundle, slot)
    if mode >= REP_ADD_SWAP and mode <= REP_REPEATED_SELECTOR:
        var slot = mode if mode < 8 else NAT_REP_DIV
        var definition = library.representations[slot]
        library.bundle.definition_bodies[definition] = independent_raw_recipe(library.bundle, slot, mode)
    elif mode == REP_MISSING_INDEX:
        library.representations[NAT_REP_ADD] = -1
    elif mode == REP_SWAPPED_INDICES:
        var first = library.representations[NAT_REP_ADD]
        library.representations[NAT_REP_ADD] = library.representations[NAT_REP_MUL]
        library.representations[NAT_REP_MUL] = first
    elif mode == REP_OPEN_ALIAS:
        var definition = library.representations[NAT_REP_ADD]
        library.bundle.definition_bodies[definition] = _n(library.bundle, tk.PT_VAR,
            library.bundle.definition_types[definition])
    elif mode == REP_CIRCULAR_ALIAS:
        var definition = library.representations[NAT_REP_ADD]
        library.bundle.definition_bodies[definition] = _n(library.bundle,
            tk.PT_DEFINED_REF, value=definition)
    elif mode == REP_BOOTSTRAP_NATIVE:
        var nat = _n(library.bundle, tk.PT_NAT)
        var a = _n(library.bundle, tk.PT_VAR, nat)
        var b = _n(library.bundle, tk.PT_VAR, nat)
        var definition = library.representations[NAT_REP_ADD]
        library.bundle.definition_bodies[definition] = _n(library.bundle, tk.PT_LAM, a,
            _n(library.bundle, tk.PT_LAM, b, _n(library.bundle, tk.PT_ADD, a, b)))
    elif mode == REP_PROOF_EXPORT_ALIAS:
        library.representations[NAT_REP_ADD] = library.exports[tk.DEC_NAT_NONNEG]
    return tk.check_native_arithmetic_library(library)


def _program(stride: Int) -> tk.FinishingProgram:
    return tk.FinishingProgram(stride, 1, stride, 0, 1, stride, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)


def native_representation_roundtrips_accepts(parameter: Int = 17) -> Bool:
    var empty = tk.empty_proof_terms[128]()
    var s = tk._make_state[128, 2048](_program(parameter), empty)
    tk._enable_native_arithmetic(s)
    if not tk._check_inductive_environment(s):
        return False
    var nat = tk._put(s, tk.PT_NAT)
    var prop = tk._put(s, tk.PT_PROP)
    var a = tk._put(s, tk.PT_VAR, nat)
    var b = tk._put(s, tk.PT_VAR, nat)
    var source = make_nat_arithmetic_library()
    for slot in range(NAT_REPRESENTATIONS):
        var kind = tk.PT_ADD
        if slot == NAT_REP_MUL:
            kind = tk.PT_MUL
        elif slot == NAT_REP_LE:
            kind = tk.PT_LE
        elif slot == NAT_REP_LT:
            kind = tk.PT_LT
        elif slot == NAT_REP_SCALE:
            kind = tk.PT_SCALE
        elif slot == NAT_REP_BASE:
            kind = tk.PT_BASE
        elif slot == NAT_REP_DIV:
            kind = tk.PT_DIV
        elif slot == NAT_REP_MOD:
            kind = tk.PT_MOD
        var binary = slot <= NAT_REP_LT
        var native = tk._put(s, kind, a, b if binary else -1,
            value=parameter if slot == NAT_REP_SCALE or slot == NAT_REP_DIV or slot == NAT_REP_MOD else -1)
        var expected_type = prop if slot == NAT_REP_LE or slot == NAT_REP_LT else nat
        if tk._infer(s, native) != expected_type:
            return False
        var expanded = tk._expand_native_arithmetic(s, native)
        if expanded < 0 or tk._infer(s, expanded) != expected_type:
            return False
        var outer = s.syntax.nodes[expanded]
        if outer.kind != tk.PT_APP:
            return False
        var inner = s.syntax.nodes[outer.left]
        if inner.kind != tk.PT_APP:
            return False
        var head = s.syntax.nodes[inner.left]
        if head.kind != tk._PT_IMMUTABLE_REF or head.value != source.representations[slot]:
            return False
        var compacted = tk._compact_arithmetic(s, expanded)
        if compacted < 0 or not tk._convert(s, native, compacted):
            return False
        if s.syntax.nodes[compacted].kind != (tk.PT_LE if slot == NAT_REP_LT else kind):
            return False
        if slot == NAT_REP_SCALE or slot == NAT_REP_DIV or slot == NAT_REP_MOD:
            if s.syntax.nodes[compacted].value != parameter:
                return False
        # Compare by administrative erasure only: both sides elaborate actual
        # aliases; no numeral recursion, affine oracle, or NatRec iota is needed.
        if not tk._operator_representation_equal(s, expanded,
            tk._expand_native_arithmetic(s, compacted)):
            return False
    return s.fuel > 0 and s.syntax.count < 2048


def _selector[C: Int](mut s: tk._TypedState[C], mode: Int) -> Int:
    var nat = tk._put(s, tk.PT_NAT)
    var q = tk._put(s, tk.PT_VAR, nat)
    var r = q if mode == 3 else tk._put(s, tk.PT_VAR, nat)
    var third_type = tk._put(s, tk.PT_PROP) if mode == 5 else nat
    var countdown = q if mode == 3 else tk._put(s, tk.PT_VAR, third_type)
    var body = q if mode == 0 or mode == 3 or mode == 5 else r if mode == 1 else countdown
    if mode == 4:
        body = tk._put(s, tk.PT_LITERAL, value=9)
    elif mode == 6:
        var extra = tk._put(s, tk.PT_VAR, nat)
        body = tk._put(s, tk.PT_LAM, extra, q)
    return tk._put(s, tk.PT_LAM, q,
        tk._put(s, tk.PT_LAM, r, tk._put(s, tk.PT_LAM, countdown, body)))


def _literal_result[C: Int](mut s: tk._TypedState[C], expression: Int,
    compact: Bool = True) -> Int:
    var reduced = tk._whnf(s, expression, compact)
    if reduced < 0:
        return -1
    var n = s.syntax.nodes[reduced]
    return n.value if n.kind == tk.PT_LITERAL else -1


def check_selector_namespace_bootstrap_controls() raises:
    var empty = tk.empty_proof_terms[128]()
    var s = tk._make_state[128, 2048](_program(7), empty)
    tk._enable_native_arithmetic(s)
    assert_equal(tk._check_inductive_environment(s), True)
    var nat = tk._put(s, tk.PT_NAT)
    var zero = tk._put(s, tk.PT_LITERAL, value=0)
    # Construct a source operation independently of native expansion/aliases.
    var state = tk._raw_app2(s, tk._put(s, tk._PT_IMMUTABLE_REF, value=3),
        tk._put(s, tk.PT_LITERAL, value=6), zero)
    # Call the semantic guard directly, independently of its heuristic. A
    # swapped addition is mathematically equal, but is not the same recipe.
    var a = tk._put(s, tk.PT_VAR, nat)
    var b = tk._put(s, tk.PT_VAR, nat)
    var actual_add = tk._raw_app2(s, tk._put(s, tk._PT_IMMUTABLE_REF, value=0), a, b)
    var correct_add = tk._put(s, tk.PT_ADD, a, b)
    var reversed_add = tk._put(s, tk.PT_ADD, b, a)
    assert_equal(tk._operator_representation_equal(s, actual_add, correct_add), True)
    assert_equal(tk._operator_representation_equal(s, actual_add, reversed_add), False)
    assert_equal(tk._checked_operator_compaction(s, actual_add, correct_add), correct_add)
    assert_equal(tk._checked_operator_compaction(s, actual_add, reversed_add), actual_add)
    # The existing canonical Nat/SUCC numeral codec needs only one bounded
    # literal increment. It never evaluates a recursor over the coefficient.
    var successor_zero = tk._put(s, tk.PT_SUCC, zero)
    var one = tk._put(s, tk.PT_LITERAL, value=1)
    assert_equal(tk._operator_representation_equal(s, successor_zero, one), True)
    var unit_mul = tk._raw_app2(s, tk._put(s, tk._PT_IMMUTABLE_REF, value=1),
        successor_zero, a)
    var unit_scale = tk._put(s, tk.PT_SCALE, a, value=1)
    var wrong_unit_scale = tk._put(s, tk.PT_SCALE, a, value=2)
    assert_equal(tk._checked_operator_compaction(s, unit_mul, unit_scale), unit_scale)
    assert_equal(tk._checked_operator_compaction(s, unit_mul, wrong_unit_scale), unit_mul)
    var predecessor_max = tk._put(s, tk.PT_LITERAL, value=Int.MAX - 1)
    var successor_max_predecessor = tk._put(s, tk.PT_SUCC, predecessor_max)
    var max_mul = tk._raw_app2(s, tk._put(s, tk._PT_IMMUTABLE_REF, value=1),
        successor_max_predecessor, a)
    var max_scale = tk._put(s, tk.PT_SCALE, a, value=Int.MAX)
    var wrong_max_scale = tk._put(s, tk.PT_SCALE, a, value=Int.MAX - 1)
    assert_equal(tk._checked_operator_compaction(s, max_mul, max_scale), max_scale)
    assert_equal(tk._checked_operator_compaction(s, max_mul, wrong_max_scale), max_mul)
    var maximum = tk._put(s, tk.PT_LITERAL, value=Int.MAX)
    var successor_maximum = tk._put(s, tk.PT_SUCC, maximum)
    assert_equal(tk._operator_administrative_whnf(s, successor_maximum), successor_maximum)
    assert_equal(tk._operator_representation_equal(s, successor_maximum, successor_maximum), True)
    var overflow_mul = tk._raw_app2(s, tk._put(s, tk._PT_IMMUTABLE_REF, value=1),
        successor_maximum, a)
    var overflow_compacted = tk._compact_arithmetic(s, overflow_mul)
    assert_equal(s.syntax.nodes[overflow_compacted].kind, tk.PT_MUL)
    assert_equal(s.syntax.nodes[overflow_compacted].left, successor_maximum)
    assert_equal(s.syntax.nodes[overflow_compacted].right, a)
    assert_equal(tk._operator_representation_equal(s, overflow_mul, overflow_compacted), True)
    assert_equal(tk._checked_operator_compaction(s, overflow_mul, max_scale), overflow_mul)
    var symbolic_successor = tk._put(s, tk.PT_SUCC, a)
    assert_equal(tk._operator_representation_equal(s, symbolic_successor, a), False)
    var symbolic_mul = tk._raw_app2(s, tk._put(s, tk._PT_IMMUTABLE_REF, value=1),
        symbolic_successor, b)
    var wrong_symbolic_mul = tk._put(s, tk.PT_MUL, a, b)
    assert_equal(tk._checked_operator_compaction(s, symbolic_mul, wrong_symbolic_mul), symbolic_mul)
    var symbolic_state = tk._raw_app2(s, tk._put(s, tk._PT_IMMUTABLE_REF, value=3),
        tk._put(s, tk.PT_LITERAL, value=6), a)
    var actual_quotient = tk._put(s, tk.PT_APP, symbolic_state, _selector(s, 0))
    var correct_div = tk._put(s, tk.PT_DIV, a, value=7)
    var wrong_mod = tk._put(s, tk.PT_MOD, a, value=7)
    assert_equal(tk._operator_representation_equal(s, actual_quotient, correct_div), True)
    assert_equal(tk._operator_representation_equal(s, actual_quotient, wrong_mod), False)
    assert_equal(tk._checked_operator_compaction(s, actual_quotient, correct_div), correct_div)
    assert_equal(tk._checked_operator_compaction(s, actual_quotient, wrong_mod), actual_quotient)
    for component in range(2):
        var projection = tk._put(s, tk.PT_APP, state, _selector(s, component))
        assert_equal(tk._infer(s, projection), nat)
        var folded = tk._compact_arithmetic(s, projection)
        assert_equal(s.syntax.nodes[folded].kind, tk.PT_DIV if component == 0 else tk.PT_MOD)
        assert_equal(s.syntax.nodes[folded].value, 7)
        assert_equal(s.syntax.nodes[folded].left, zero)
        assert_equal(_literal_result(s, folded), 0)
    for mode in range(2, 5):
        var projection = tk._put(s, tk.PT_APP, state, _selector(s, mode))
        assert_equal(tk._infer(s, projection), nat)
        assert_equal(tk._compact_arithmetic(s, projection), projection)
        assert_equal(_literal_result(s, projection, False), 9 if mode == 4 else 6)
    for mode in range(5, 7):
        var projection = tk._put(s, tk.PT_APP, state, _selector(s, mode))
        assert_equal(tk._infer(s, projection), -1)
        assert_equal(tk._compact_arithmetic(s, projection), projection)

    # Numeric ID3 in caller namespace is a different, well-typed triple.
    var caller = tk._make_state[128, 2048](_program(7), empty)
    tk._enable_native_arithmetic(caller)
    var caller_nat = tk._put(caller, tk.PT_NAT)
    var caller_zero = tk._put(caller, tk.PT_LITERAL, value=0)
    for d in range(3):
        caller.syntax.definition_types[d] = caller_nat
        caller.syntax.definition_bodies[d] = caller_zero
    var k = tk._put(caller, tk.PT_VAR, caller_nat)
    var n = tk._put(caller, tk.PT_VAR, caller_nat)
    var triple_type = tk._raw_triple_type(caller)
    var triple = tk._raw_triple(caller,
        tk._put(caller, tk.PT_LITERAL, value=9),
        tk._put(caller, tk.PT_LITERAL, value=8),
        tk._put(caller, tk.PT_LITERAL, value=7))
    caller.syntax.definition_types[3] = tk._put(caller, tk.PT_PI, k,
        tk._put(caller, tk.PT_PI, n, triple_type))
    caller.syntax.definition_bodies[3] = tk._put(caller, tk.PT_LAM, k,
        tk._put(caller, tk.PT_LAM, n, triple))
    caller.syntax.definition_count = 4
    assert_equal(tk._check_inductive_environment(caller), True)
    var caller_state = tk._raw_app2(caller, tk._put(caller, tk.PT_DEFINED_REF, value=3),
        tk._put(caller, tk.PT_LITERAL, value=6), caller_zero)
    var caller_projection = tk._put(caller, tk.PT_APP, caller_state, _selector(caller, 0))
    assert_equal(tk._infer(caller, caller_projection), caller_nat)
    assert_equal(tk._compact_arithmetic(caller, caller_projection), caller_projection)
    assert_equal(_literal_result(caller, caller_projection), 9)

    # Caller alias ID36 overlaps the first global recipe but keeps caller meaning.
    var shadow = tk._make_state[128, 2048](_program(7), empty)
    tk._enable_native_arithmetic(shadow)
    var shadow_nat = tk._put(shadow, tk.PT_NAT)
    var shadow_zero = tk._put(shadow, tk.PT_LITERAL, value=0)
    for d in range(36):
        shadow.syntax.definition_types[d] = shadow_nat
        shadow.syntax.definition_bodies[d] = shadow_zero
    var first = tk._put(shadow, tk.PT_VAR, shadow_nat)
    var second = tk._put(shadow, tk.PT_VAR, shadow_nat)
    shadow.syntax.definition_types[36] = tk._put(shadow, tk.PT_PI, first,
        tk._put(shadow, tk.PT_PI, second, shadow_nat))
    shadow.syntax.definition_bodies[36] = tk._put(shadow, tk.PT_LAM, first,
        tk._put(shadow, tk.PT_LAM, second, tk._put(shadow, tk.PT_LITERAL, value=9)))
    shadow.syntax.definition_count = 37
    assert_equal(tk._check_inductive_environment(shadow), True)
    var shadow_call = tk._raw_app2(shadow, tk._put(shadow, tk.PT_DEFINED_REF, value=36),
        shadow_zero, shadow_zero)
    assert_equal(tk._infer(shadow, shadow_call), shadow_nat)
    assert_equal(tk._compact_arithmetic(shadow, shadow_call), shadow_call)
    assert_equal(_literal_result(shadow, shadow_call), 9)

    var bootstrap = tk._make_state[128, 2048](_program(7), empty)
    assert_equal(tk._check_inductive_environment(bootstrap), True)
    var private = tk._put(bootstrap, tk._PT_IMMUTABLE_REF, value=3)
    var boot_state = tk._raw_app2(bootstrap, private,
        tk._put(bootstrap, tk.PT_LITERAL, value=6), tk._put(bootstrap, tk.PT_LITERAL, value=0))
    var boot_projection = tk._put(bootstrap, tk.PT_APP, boot_state, _selector(bootstrap, 0))
    assert_equal(tk._compact_arithmetic(bootstrap, boot_projection), boot_projection)
    assert_equal(tk._infer(bootstrap, boot_projection), -1)
    assert_equal(tk._immutable_definition_type(bootstrap, 3), -1)
    assert_equal(tk._immutable_definition_body(bootstrap, 3), -1)
    var source = materialize[tk._NAT_ARITHMETIC_LIBRARY.bundle]()
    assert_equal(tk._import_immutable_node(bootstrap, source, 0), -1)
    print("Exact selectors, raw iota, private namespace and bootstrap controls passed")
