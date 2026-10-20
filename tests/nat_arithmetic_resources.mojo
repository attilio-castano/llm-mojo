"""Resource pressure uses ordinary closed proofs, independently of geometry."""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.nat_arithmetic import make_nat_arithmetic_library, NAT_ARITHMETIC_CAPACITY


def check_arithmetic_contexts_and_lookup_index() raises:
    var library = make_nat_arithmetic_library()
    comptime C = 3 * NAT_ARITHMETIC_CAPACITY + 128
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[NAT_ARITHMETIC_CAPACITY, C](program, library.bundle)
    tk._enable_native_numerals(state)
    assert_equal(tk._check_inductive_environment(state), True)
    for j in range(library.bundle.count):
        var inferred = tk._infer(state, j)
        assert_equal(inferred == -1 or inferred == -4, False)
    for definition in range(6):
        var intended = tk._canonical_arithmetic_body(state, definition)
        assert_equal(tk._infer(state, intended) >= 0, True)
        assert_equal(tk._convert(state, state.syntax.definition_bodies[definition], intended), True)
    for declaration in range(tk.PROOF_TERM_DECLARATIONS):
        if declaration == tk.DEC_DONE_STOP or declaration == tk.DEC_DONE_STEP:
            continue
        var required = tk._required_arithmetic_export(state, declaration)
        assert_equal(tk._infer(state, required) >= 0, True)
        assert_equal(tk._convert(state, state.syntax.definition_types[library.exports[declaration]], required), True)
    # The matrix is the independent oracle for the optimized context prefix.
    comptime W = (C + 63) // 64
    assert_equal(len(state.contexts), state.syntax.count * W)
    for node in range(state.syntax.count):
        var actual_high = 0
        for word in range(W):
            if state.contexts[node * W + word] != 0:
                actual_high = word + 1
        assert_equal(state.context_high[node], actual_high)
    for node in range(state.syntax.count, C):
        assert_equal(state.context_high[node], 0)
    # Compare every interned entry with the previous linear first-match rule.
    for node in range(state.syntax.count):
        var n = state.syntax.nodes[node]
        if n.kind == tk.PT_VAR:
            continue
        var expected_first = node
        for earlier in range(node):
            var prior = state.syntax.nodes[earlier]
            if (n.kind == prior.kind and n.left == prior.left and n.right == prior.right
                and n.extra == prior.extra and n.value == prior.value):
                expected_first = earlier
                break
        assert_equal(state.interned.get((n.kind, n.left, n.right, n.extra, n.value), -1), expected_first)


def check_arithmetic_lookup_boundaries() raises:
    var p = tk.empty_proof_terms[16]()
    p.nodes[3] = p.nodes[0]
    p.count = 4
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[16, 8](program, p)
    assert_equal(tk._put(state, tk.PT_NAT), 0)
    assert_equal(tk._put(state, tk.PT_CONSTRUCTOR, extra=-1, value=0), 4)
    assert_equal(tk._put(state, tk.PT_CONSTRUCTOR, extra=1, value=0), 5)
    assert_equal(tk._put(state, tk.PT_VAR, 0), 6)
    assert_equal(tk._put(state, tk.PT_VAR, 0), 7)
    assert_equal(state.syntax.count, 8)
    assert_equal(tk._put(state, tk.PT_CONSTRUCTOR, extra=1, value=0), 5)
    assert_equal(tk._put(state, tk.PT_NAT), 0)
    assert_equal(tk._put(state, tk.PT_VAR, 0), -1)
    assert_equal(tk._put(state, tk.PT_LITERAL, value=4), -1)
    assert_equal(state.syntax.count, 8)


def check_arithmetic_workspace_fallback() raises:
    var library = make_nat_arithmetic_library()
    var nat = tk.pt_add(library.bundle, tk.PT_NAT)
    # Five fresh nodes per ordinary reflexivity definition. Derive the fill
    # count from the actual library and retain one slot for the malformed APP.
    # Geometry never sets proof syntax or checker workspace capacities.
    var fillers = (NAT_ARITHMETIC_CAPACITY - library.bundle.count - 1) // 5
    assert_equal(fillers > 0, True)
    for _ in range(fillers):
        var n = tk.pt_add(library.bundle, tk.PT_VAR, nat)
        var goal = tk.pt_add(library.bundle, tk.PT_EQ, n, n)
        var declared = tk.pt_add(library.bundle, tk.PT_PI, n, goal)
        var body = tk.pt_add(library.bundle, tk.PT_LAM, n,
            tk.pt_add(library.bundle, tk.PT_REFL, n))
        assert_equal(tk.pt_declare_definition(library.bundle, declared, body) >= 0, True)
    assert_equal(library.bundle.count < NAT_ARITHMETIC_CAPACITY, True)
    # A workspace barely containing the input still exercises exhaustion.
    # The public first attempt may succeed as generic conversion improves;
    # requiring it to exhaust would turn a valid improvement into a failure.
    comptime Tight = NAT_ARITHMETIC_CAPACITY
    var tight = tk._native_arithmetic_attempt[NAT_ARITHMETIC_CAPACITY, Tight](library)
    assert_equal(tight.accepted, False)
    assert_equal(tight.resource_exhausted, True)
    comptime Initial = 3 * NAT_ARITHMETIC_CAPACITY + 128
    comptime Full = 4 * NAT_ARITHMETIC_CAPACITY + 128
    var initial = tk._native_arithmetic_attempt[NAT_ARITHMETIC_CAPACITY, Initial](library)
    var full = tk._native_arithmetic_attempt[NAT_ARITHMETIC_CAPACITY, Full](library)
    assert_equal(initial.resource_exhausted, not initial.accepted)
    assert_equal(full.accepted, True)
    assert_equal(full.resource_exhausted, False)
    assert_equal(tk.check_native_arithmetic_library(library), True)
    # A logical error must remain rejected when more storage is available.
    var bad = tk.pt_add(library.bundle, tk.PT_APP, nat, nat)
    assert_equal(bad >= 0, True)
    library.bundle.definition_bodies[library.bundle.definition_count - 1] = bad
    var rejected = tk._native_arithmetic_attempt[NAT_ARITHMETIC_CAPACITY, Initial](library)
    assert_equal(rejected.accepted, False)
    assert_equal(rejected.resource_exhausted, False)
    var rejected_full = tk._native_arithmetic_attempt[NAT_ARITHMETIC_CAPACITY, Full](library)
    assert_equal(rejected_full.accepted, False)
    assert_equal(rejected_full.resource_exhausted, False)
    assert_equal(tk.check_native_arithmetic_library(library), False)
