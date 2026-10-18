"""Base driver equations are checked proofs at flexible symbolic shape."""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from base_definition_cases import base_driver_equations_accepts
from nat_arithmetic_cases import (
    BASE_CONSTANT_ZERO, BASE_SUCCESSOR_ONE, BASE_WRONG_CAPTURE,
    NAT_ARITHMETIC_MUTANTS, nat_arithmetic_mutant, nat_arithmetic_mutant_accepts,
    retired_arithmetic_axiom_accepts, local_shadow_case,
)
from nat_arithmetic_resources import (
    check_arithmetic_workspace_fallback, check_arithmetic_contexts_and_lookup_index,
    check_arithmetic_lookup_boundaries,
)


def main() raises:
    comptime assert tk._NAT_ARITHMETIC_CERTIFIED, "checked Base definition rejected"
    comptime assert base_driver_equations_accepts(1), "checked Base definition rejected"
    comptime assert base_driver_equations_accepts(6), "checked Base definition rejected"
    comptime assert base_driver_equations_accepts(35), "checked Base definition rejected"
    comptime assert base_driver_equations_accepts(257), "checked Base definition rejected"
    comptime assert base_driver_equations_accepts(Int.MAX), "checked Base definition rejected"
    comptime assert base_driver_equations_accepts(17, 4), "checked Base definition rejected"
    comptime assert base_driver_equations_accepts(Int.MAX, 4), "checked Base definition rejected"
    assert_equal(base_driver_equations_accepts(17, 4), True)
    assert_equal(base_driver_equations_accepts(Int.MAX, 4), True)
    assert_equal(base_driver_equations_accepts(1), True)
    assert_equal(base_driver_equations_accepts(6), True)
    assert_equal(base_driver_equations_accepts(35), True)
    assert_equal(base_driver_equations_accepts(257), True)
    assert_equal(base_driver_equations_accepts(Int.MAX), True)
    for mode in range(4):
        assert_equal(base_driver_equations_accepts(17, mode), False)
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    for mode in range(BASE_CONSTANT_ZERO, BASE_WRONG_CAPTURE + 1):
        var wrong_base = nat_arithmetic_mutant(mode)
        # Every theorem helper still references the correct earlier clone.
        # Rejection must distinguish the public checked definition itself.
        assert_equal(tk.check_native_definitions(program, wrong_base.bundle), True)
        assert_equal(tk.check_native_arithmetic_library(wrong_base), False)
    for mode in range(BASE_CONSTANT_ZERO, NAT_ARITHMETIC_MUTANTS):
        assert_equal(nat_arithmetic_mutant_accepts(mode), False)
    for declaration in range(tk.PROOF_TERM_DECLARATIONS):
        assert_equal(retired_arithmetic_axiom_accepts(declaration), False)
    assert_equal(local_shadow_case(), True)
    assert_equal(local_shadow_case(4), False)
    check_arithmetic_workspace_fallback()
    check_arithmetic_contexts_and_lookup_index()
    check_arithmetic_lookup_boundaries()
    print("Checked Base definition, derived driver equations and rejection controls passed")
