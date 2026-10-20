"""Check immutable export types through ordinary application and conversion."""
from std.testing import assert_equal
from base_definition_cases import base_driver_equations_accepts
from export_instantiation_cases import (
    EXPORT_CONTRACT_MUTANTS, export_contracts_accept,
    mixed_export_parameters_accept, export_capture_accepts,
    caller_export_spoof_accepts, export_open_root_accepts,
)


def main() raises:
    comptime assert export_contracts_accept(1, 1), "checked export instantiation rejected"
    comptime assert export_contracts_accept(2, 17), "checked export instantiation rejected"
    comptime assert export_contracts_accept(65537, 17), "checked export instantiation rejected"
    comptime assert export_contracts_accept(Int.MAX, Int.MAX), "checked export instantiation rejected"
    comptime assert export_contracts_accept(17, 17, zero_scale=True), "checked export instantiation rejected"
    comptime assert export_contracts_accept(17, 17, shadows_and_aliases=True), "checked export instantiation rejected"
    comptime assert export_contracts_accept(Int.MAX, Int.MAX, shadows_and_aliases=True), "checked export instantiation rejected"
    comptime assert mixed_export_parameters_accept(), "checked export instantiation rejected"
    comptime assert mixed_export_parameters_accept(True), "checked export instantiation rejected"
    comptime assert export_capture_accepts(), "checked export instantiation rejected"
    comptime assert base_driver_equations_accepts(Int.MAX, 4), "checked export instantiation rejected"
    assert_equal(export_contracts_accept(1, 1), True)
    assert_equal(export_contracts_accept(2, 17), True)
    assert_equal(export_contracts_accept(65537, 17), True)
    assert_equal(export_contracts_accept(Int.MAX, Int.MAX), True)
    assert_equal(export_contracts_accept(17, 17, zero_scale=True), True)
    assert_equal(export_contracts_accept(17, 17, shadows_and_aliases=True), True)
    assert_equal(export_contracts_accept(Int.MAX, Int.MAX, shadows_and_aliases=True), True)
    assert_equal(mixed_export_parameters_accept(), True)
    assert_equal(mixed_export_parameters_accept(True), True)
    assert_equal(export_capture_accepts(), True)
    assert_equal(base_driver_equations_accepts(Int.MAX, 4), True)
    for mode in range(EXPORT_CONTRACT_MUTANTS):
        assert_equal(export_contracts_accept(17, 17, mode), False)
    for mode in range(2):
        assert_equal(mixed_export_parameters_accept(False, mode), False)
        assert_equal(mixed_export_parameters_accept(True, mode), False)
    assert_equal(export_capture_accepts(True), False)
    assert_equal(caller_export_spoof_accepts(), False)
    assert_equal(export_open_root_accepts(), False)
    print("Checked all 16 immutable export contracts, ordinary parameter application and rejection controls")
