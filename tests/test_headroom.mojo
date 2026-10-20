"""Check raw Nat headroom, carry boundaries and rejection controls."""
from std.testing import assert_equal
from headroom_alpha_cases import headroom_alpha_controls_accepts
from headroom_word_cases import headroom_word_denotations_accepts
from headroom_cases import (
    headroom_accepts, headroom_small_cases_accepts,
    headroom_word_boundaries_accepts, headroom_rejection_controls_accepts,
    headroom_resource_controls_accepts, headroom_alternate_proofs_accepts,
)


def main() raises:
    comptime assert headroom_word_denotations_accepts(), "checked Nat word denotation rejected"
    comptime assert headroom_accepts(37, 15, 52), "checked Nat headroom rejected"
    comptime assert headroom_accepts(Int(4294967295), 1, Int(4294967296)), "checked Nat headroom rejected"
    comptime assert headroom_accepts(Int(9223372036854775806), 1, Int(9223372036854775807)), "checked Nat headroom rejected"
    assert_equal(headroom_alpha_controls_accepts(), True)
    assert_equal(headroom_word_denotations_accepts(), True)
    assert_equal(headroom_small_cases_accepts(), True)
    assert_equal(headroom_word_boundaries_accepts(), True)
    assert_equal(headroom_alternate_proofs_accepts(), True)
    assert_equal(headroom_rejection_controls_accepts(), True)
    assert_equal(headroom_resource_controls_accepts(), True)
    print("Checked Nat headroom, carry boundaries, raw syntax and rejection controls")
