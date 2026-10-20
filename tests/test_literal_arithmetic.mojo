"""Check literal computations against raw Nat definitions at compile time."""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from literal_arithmetic_cases import (
    LITERAL_ARITHMETIC_MUTANTS, literal_arithmetic_accepts,
    literal_arithmetic_small_cases_accepts, literal_arithmetic_mutant_accepts,
    literal_arithmetic_resource_controls_accepts, literal_arithmetic_hook_accepts,
    literal_arithmetic_preserves_raw_case, literal_arithmetic_symbolic_opacity_accepts,
)


def main() raises:
    comptime assert literal_arithmetic_accepts(tk.PT_ADD, 3, 2, 5), "checked literal arithmetic conversion rejected"
    comptime assert literal_arithmetic_accepts(tk.PT_MUL, 3, 2, 6), "checked literal arithmetic conversion rejected"
    comptime assert literal_arithmetic_accepts(tk.PT_DIV, 8, 3, 2), "checked literal arithmetic conversion rejected"
    comptime assert literal_arithmetic_accepts(tk.PT_MOD, 8, 3, 2, True), "checked literal arithmetic conversion rejected"
    comptime assert literal_arithmetic_hook_accepts(), "checked literal arithmetic conversion rejected"
    comptime assert literal_arithmetic_preserves_raw_case(Int.MAX, 0), "checked literal arithmetic conversion rejected"
    comptime assert literal_arithmetic_preserves_raw_case(Int(4294967296), 1), "checked literal arithmetic conversion rejected"
    comptime assert literal_arithmetic_symbolic_opacity_accepts(), "checked literal arithmetic conversion rejected"
    assert_equal(literal_arithmetic_small_cases_accepts(), True)
    for mode in range(LITERAL_ARITHMETIC_MUTANTS):
        assert_equal(literal_arithmetic_mutant_accepts(mode), False)
    assert_equal(literal_arithmetic_resource_controls_accepts(), True)
    assert_equal(literal_arithmetic_hook_accepts(), True)
    for mode in range(4):
        assert_equal(literal_arithmetic_preserves_raw_case(Int.MAX, mode), True)
        assert_equal(literal_arithmetic_preserves_raw_case(Int(4294967296), mode), True)
    assert_equal(literal_arithmetic_symbolic_opacity_accepts(), True)
    print("Checked raw Nat literal arithmetic, request binding, opacity and rejection controls")
