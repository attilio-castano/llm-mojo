"""Storage ownership and cache authority at the proof-state boundary.

These tests use the canonical kernel initializer, not a reference initializer
or a field-by-field replica. A small proof exercises both constructor families,
an ordinary definition, closed alpha-equivalent judgments and an open dependent
context. Mutation checks distinguish deep ownership from shared cache storage.
"""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program

comptime STORAGE_SOURCE_CAPACITY = 64
comptime STORAGE_WORKSPACE = 2048


@fieldwise_init
struct _StorageCase(Copyable, Movable):
    var bundle: tk.ProofTermBundle[STORAGE_SOURCE_CAPACITY]
    var nat: Int
    var outer: Int
    var dependent: Int
    var dependent_type: Int
    var closed: Int
    var expected: Int
    var definition: Int
    var definition_type: Int
    var family_witness: Int
    var family_result: Int
    var literal: Int


def _storage_case() -> _StorageCase:
    var p = tk.empty_proof_terms[STORAGE_SOURCE_CAPACITY]()
    var nat = tk.pt_add(p, tk.PT_NAT)
    var zero = tk.pt_add(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    _ = tk.pt_declare_family(p, 1)
    var family = tk.pt_add(p, tk.PT_FAMILY, value=tk.PROP_FAMILY_ID)
    var family_result = tk.pt_add(p, tk.PT_APP, family, zero)
    _ = tk.pt_declare_constructor(p, family_result)
    var family_witness = tk.pt_add(p, tk.PT_CONSTRUCTOR, value=0)

    var argument = tk.pt_add(p, tk.PT_VAR, nat)
    var definition_type = tk.pt_add(p, tk.PT_PI, argument, nat)
    var identity = tk.pt_add(p, tk.PT_LAM, argument, argument)
    _ = tk.pt_declare_definition(p, definition_type, identity)
    var definition = tk.pt_add(p, tk.PT_DEFINED_REF, value=0)

    var outer = tk.pt_add(p, tk.PT_VAR, nat)
    var dependent_type = tk.pt_add(p, tk.PT_EQ, outer, outer)
    var dependent = tk.pt_add(p, tk.PT_VAR, dependent_type)
    var closed = tk.pt_add(p, tk.PT_LAM, outer, tk.pt_add(p, tk.PT_REFL, outer))
    var renamed = tk.pt_add(p, tk.PT_VAR, nat)
    var expected = tk.pt_add(p, tk.PT_PI, renamed,
        tk.pt_add(p, tk.PT_EQ, renamed, renamed))
    var literal = tk.pt_add(p, tk.PT_LITERAL, value=37)
    p.root = closed
    return _StorageCase(p^, nat, outer, dependent, dependent_type, closed,
        expected, definition, definition_type, family_witness, family_result, literal)


def _check_storage_judgments(mut state: tk._TypedState[STORAGE_WORKSPACE],
    sample: _StorageCase) raises:
    tk._enable_native_numerals(state)
    assert_equal(tk._check_inductive_environment(state), True)
    assert_equal(tk._infer(state, sample.family_witness), sample.family_result)
    assert_equal(tk._infer(state, sample.definition), sample.definition_type)
    assert_equal(tk._infer(state, sample.literal), sample.nat)
    var applied = tk._put(state, tk.PT_APP, sample.definition, sample.literal)
    assert_equal(tk._infer(state, applied), sample.nat)
    assert_equal(tk._convert(state, applied, sample.literal), True)
    var actual = tk._infer(state, sample.closed)
    assert_equal(actual >= 0, True)
    assert_equal(tk._infer(state, sample.expected) >= 0, True)
    assert_equal(tk._convert(state, actual, sample.expected), True)
    assert_equal(state.context_high[sample.closed], 0)
    assert_equal(state.context_high[sample.expected], 0)
    assert_equal(tk._infer(state, sample.dependent), sample.dependent_type)
    assert_equal(tk._has(state, sample.dependent, sample.outer), True)
    assert_equal(tk._has(state, sample.dependent, sample.dependent), True)
    assert_equal(state.context_high[sample.dependent] > 0, True)
    assert_equal(state.arithmetic_verified or state.fast_arithmetic, False)
    assert_equal(state.fuel > 0 and state.syntax.count < STORAGE_WORKSPACE, True)


@no_inline
def check_proof_state_storage() raises:
    var program = make_finishing_program(3, 5, 15)
    var sample = _storage_case()
    # Retain an unmodified caller bundle for public admission checks below.
    var public_bundle = sample.bundle.copy()
    var warm = tk._make_state[STORAGE_SOURCE_CAPACITY, STORAGE_WORKSPACE](program, sample.bundle)
    var fresh = tk._make_state[STORAGE_SOURCE_CAPACITY, STORAGE_WORKSPACE](program, sample.bundle)

    # Poison caller-owned syntax and declaration metadata after both copies.
    # The retained states must still admit both families and the identity.
    sample.bundle.nodes[sample.literal].value = -1
    sample.bundle.constructor_types[0] = -1
    sample.bundle.nat_constructor_types[0] = -1
    sample.bundle.definition_bodies[0] = -1
    sample.bundle.family_arity = -1
    sample.bundle.constructor_count = 0
    sample.bundle.nat_constructor_count = 0
    sample.bundle.definition_count = 0
    sample.bundle.root = -1
    sample.bundle.count = -1
    assert_equal(warm.syntax.root, sample.closed)
    _check_storage_judgments(warm, sample)
    assert_equal(warm.numeral_denotations.get(37, -1) >= 0, True)

    # Warming one state grants no environment admission or numeral authority
    # to another state, and cannot populate the other's denotation cache.
    assert_equal(fresh.environment_checked, False)
    assert_equal(fresh.nat_environment_checked, False)
    assert_equal(fresh.admitted_definitions, 0)
    assert_equal(fresh.numerals_verified or fresh.arithmetic_verified or fresh.fast_arithmetic, False)
    assert_equal(len(fresh.numeral_denotations), 0)

    # The copied syntax seeds interning, but variables remain fresh identities.
    var before = fresh.syntax.count
    assert_equal(tk._put(fresh, tk.PT_NAT), sample.nat)
    assert_equal(fresh.syntax.count, before)
    var first = tk._put(fresh, tk.PT_VAR, sample.nat)
    var second = tk._put(fresh, tk.PT_VAR, sample.nat)
    assert_equal(first != second and first != sample.outer and second != sample.outer, True)
    assert_equal(tk._infer(fresh, first), sample.nat)
    assert_equal(tk._has(fresh, first, first), True)
    assert_equal(tk._has(fresh, first, second), False)
    _check_storage_judgments(fresh, sample)

    # Copy an already checked state. Its arrays, exact-context rows and maps
    # must not share mutable ownership with the original.
    var copied = warm.copy()
    copied.syntax.nodes[sample.literal].value = 99
    copied.syntax.root = sample.dependent
    copied.numeral_denotations[37] = -1
    copied.types[sample.dependent] = -1
    tk._remove(copied, sample.dependent, sample.outer)
    assert_equal(tk._has(copied, sample.dependent, sample.outer), False)
    assert_equal(warm.syntax.nodes[sample.literal].value, 37)
    assert_equal(warm.syntax.root, sample.closed)
    assert_equal(warm.numeral_denotations.get(37, -1) >= 0, True)
    assert_equal(tk._infer(warm, sample.dependent), sample.dependent_type)
    assert_equal(tk._has(warm, sample.dependent, sample.outer), True)

    # An interned append in the copy must not create a phantom syntax node
    # in the original's map. Each owner must append its own new literal.
    before = warm.syntax.count
    assert_equal(tk._put(copied, tk.PT_LITERAL, value=101), before)
    assert_equal(copied.syntax.count, before + 1)
    assert_equal(warm.syntax.count, before)
    assert_equal(tk._put(warm, tk.PT_LITERAL, value=101), before)
    assert_equal(warm.syntax.count, before + 1)

    # A welltyped open intermediate is still inadmissible as a closed root.
    assert_equal(tk.check_proof_terms(program, public_bundle, sample.expected), True)
    public_bundle.root = sample.dependent
    assert_equal(tk.check_proof_terms(program, public_bundle, sample.dependent_type), False)
    print("Proof-state ownership, cache isolation and binder judgments passed")
