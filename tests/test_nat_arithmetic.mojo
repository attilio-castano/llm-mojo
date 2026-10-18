"""Check actual Nat arithmetic proofs and reject weakened definitions."""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.nat_arithmetic import make_nat_arithmetic_library
from nat_arithmetic_resources import (
    check_arithmetic_workspace_fallback, check_arithmetic_contexts_and_lookup_index,
    check_arithmetic_lookup_boundaries,
)
from nat_arithmetic_cases import (
    NAT_ARITHMETIC_MUTANTS, nat_arithmetic_mutant_accepts,
    retired_arithmetic_axiom_accepts, local_shadow_case, aliased_constructor_case,
    nat_arithmetic_mutant, PRED_IDENTITY, BASE_CONSTANT_ZERO, BASE_SUCCESSOR_ONE, BASE_WRONG_CAPTURE,
)


def main() raises:
    comptime assert tk._NAT_ARITHMETIC_CERTIFIED, "native Nat arithmetic rejected"
    var runtime_library = make_nat_arithmetic_library()
    assert_equal(tk.check_native_arithmetic_library(runtime_library), True)
    var wrong_pred = nat_arithmetic_mutant(PRED_IDENTITY)
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    assert_equal(tk.check_native_definitions(program, wrong_pred.bundle), True)
    assert_equal(tk.check_native_arithmetic_library(wrong_pred), False)
    for mode in range(BASE_CONSTANT_ZERO, BASE_WRONG_CAPTURE + 1):
        var wrong_base = nat_arithmetic_mutant(mode)
        assert_equal(tk.check_native_definitions(program, wrong_base.bundle), True)
        assert_equal(tk.check_native_arithmetic_library(wrong_base), False)
    for mode in range(NAT_ARITHMETIC_MUTANTS):
        assert_equal(nat_arithmetic_mutant_accepts(mode), False)
    for declaration in range(tk.PROOF_TERM_DECLARATIONS):
        assert_equal(retired_arithmetic_axiom_accepts(declaration), False)
    assert_equal(local_shadow_case(), True)
    for mode in range(5):
        assert_equal(local_shadow_case(mode), False)
    assert_equal(aliased_constructor_case(), True)
    assert_equal(aliased_constructor_case(True), False)
    check_arithmetic_workspace_fallback()
    check_arithmetic_contexts_and_lookup_index()
    check_arithmetic_lookup_boundaries()
    print("Checked Nat arithmetic definitions, derived laws and rejection controls passed")
