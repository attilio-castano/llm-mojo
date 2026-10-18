"""Check operator elaborations against certified Nat data definitions."""
from std.testing import assert_equal
from operator_representation_cases import (
    OPERATOR_REPRESENTATION_MUTANTS, operator_representation_library_accepts,
    native_representation_roundtrips_accepts, check_selector_namespace_bootstrap_controls,
)
from export_instantiation_cases import mixed_export_parameters_accept, export_contracts_accept
from test_typed_coverage import independent_coverage_accepts
from llm_mojo.kernels.finishing_proof import make_finishing_program
import llm_mojo.kernels.finishing_typed_coverage as coverage


def main() raises:
    comptime assert operator_representation_library_accepts(), "checked operator representation rejected"
    comptime assert operator_representation_library_accepts(rebind_all=True), "checked operator representation rejected"
    comptime assert native_representation_roundtrips_accepts(17), "checked operator representation rejected"
    comptime assert native_representation_roundtrips_accepts(Int.MAX), "checked operator representation rejected"
    comptime assert export_contracts_accept(1, 1), "checked operator representation rejected"
    comptime assert export_contracts_accept(2, 17), "checked operator representation rejected"
    comptime assert export_contracts_accept(65537, 17), "checked operator representation rejected"
    comptime program = make_finishing_program(64, 3, 192)
    comptime proofs = coverage.make_finishing_typed_coverage(program)
    comptime assert independent_coverage_accepts(program, proofs), "checked operator representation rejected"
    assert_equal(operator_representation_library_accepts(), True)
    assert_equal(operator_representation_library_accepts(rebind_all=True), True)
    assert_equal(native_representation_roundtrips_accepts(1), True)
    assert_equal(native_representation_roundtrips_accepts(17), True)
    assert_equal(native_representation_roundtrips_accepts(65537), True)
    assert_equal(native_representation_roundtrips_accepts(Int.MAX), True)
    for mode in range(OPERATOR_REPRESENTATION_MUTANTS):
        assert_equal(operator_representation_library_accepts(mode), False)
    assert_equal(export_contracts_accept(1, 1), True)
    assert_equal(export_contracts_accept(2, 17), True)
    assert_equal(export_contracts_accept(65537, 17), True)
    assert_equal(export_contracts_accept(Int.MAX, Int.MAX), True)
    assert_equal(export_contracts_accept(17, 17, zero_scale=True), True)
    assert_equal(mixed_export_parameters_accept(), True)
    assert_equal(mixed_export_parameters_accept(True), True)
    assert_equal(mixed_export_parameters_accept(False, 0), False)
    assert_equal(mixed_export_parameters_accept(True, 1), False)
    var native_proofs = coverage.make_finishing_typed_coverage(program)
    assert_equal(independent_coverage_accepts(program, native_proofs), True)
    assert_equal(coverage.check_finishing_typed_coverage(program, native_proofs), True)
    check_selector_namespace_bootstrap_controls()
    print("Checked all eight operator recipes, symbolic MAX specializations and rejection controls")
