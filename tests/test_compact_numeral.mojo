"""Check raw Nat numeral meanings and their compact execution bridge."""
from std.testing import assert_equal
from compact_numeral_cases import (
    COMPACT_NUMERAL_MUTANTS, compact_numeral_mutant_accepts,
    numeral_library_accepts, numeral_view_accepts, largest_positive_power_of_two,
    bounded_native_numeral_paths, numeral_caller_namespace_accepts,
    numeral_primitive_baseline_accepts,
    numeral_resource_controls_accepts,
    duplicate_literal_successor_accepts,
    independent_boundary_numerals_accepts,
    numeral_meaning_rejections_accept,
)


def main() raises:
    comptime assert numeral_library_accepts(), "checked compact numeral bridge rejected"
    comptime assert numeral_library_accepts(independent=True), "checked compact numeral bridge rejected"
    comptime assert numeral_view_accepts(0), "checked compact numeral bridge rejected"
    comptime assert numeral_view_accepts(1), "checked compact numeral bridge rejected"
    comptime assert numeral_view_accepts(17, recompose=True), "checked compact numeral bridge rejected"
    comptime assert numeral_primitive_baseline_accepts(), "checked compact numeral bridge rejected"
    comptime assert numeral_view_accepts(Int(1) << 32), "checked compact numeral bridge rejected"
    comptime assert numeral_view_accepts(Int(1) << 33), "checked compact numeral bridge rejected"
    comptime assert independent_boundary_numerals_accepts(), "checked compact numeral bridge rejected"
    comptime assert numeral_meaning_rejections_accept(), "checked compact numeral bridge rejected"
    comptime assert numeral_view_accepts(Int.MAX), "checked compact numeral bridge rejected"
    comptime assert numeral_view_accepts(largest_positive_power_of_two()), "checked compact numeral bridge rejected"
    comptime assert bounded_native_numeral_paths(Int.MAX), "checked compact numeral bridge rejected"
    comptime assert bounded_native_numeral_paths(largest_positive_power_of_two()), "checked compact numeral bridge rejected"
    comptime assert numeral_caller_namespace_accepts(), "checked compact numeral bridge rejected"
    assert_equal(numeral_library_accepts(), True)
    assert_equal(numeral_library_accepts(independent=True), True)
    var values: List[Int] = [0, 1, 17, 65537, Int(1) << 32, Int(1) << 33,
        Int.MAX - 1, Int.MAX, largest_positive_power_of_two()]
    for value in values:
        assert_equal(numeral_view_accepts(value), True)
        assert_equal(numeral_view_accepts(value, recompose=True), True)
        assert_equal(bounded_native_numeral_paths(value), True)
    for mode in range(COMPACT_NUMERAL_MUTANTS):
        assert_equal(compact_numeral_mutant_accepts(mode), False)
    assert_equal(numeral_view_accepts(-1), False)
    assert_equal(numeral_caller_namespace_accepts(), True)
    assert_equal(numeral_resource_controls_accepts(), True)
    assert_equal(duplicate_literal_successor_accepts(17), True)
    assert_equal(duplicate_literal_successor_accepts(Int.MAX - 1), True)
    assert_equal(independent_boundary_numerals_accepts(), True)
    assert_equal(numeral_meaning_rejections_accept(), True)
    print("Checked compact numeral definitions, conversion proofs, MAX and carry-boundary paths")
