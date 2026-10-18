"""Checked total Nat join and its specialized canonical trace theorem."""
from std.testing import assert_equal
from finishing_join_cases import (
    FINISHING_JOIN_MUTANTS, FINISHING_JOIN_RESOURCE_CONTROLS,
    finishing_join_accepts, finishing_join_alternate_accepts,
    finishing_join_mutant_accepts, finishing_join_resource_mutant_accepts,
)


def main() raises:
    comptime assert finishing_join_accepts(), "checked finishing Nat join rejected"
    assert_equal(finishing_join_accepts(), True)
    assert_equal(finishing_join_alternate_accepts(), True)
    for mode in range(FINISHING_JOIN_MUTANTS):
        if finishing_join_mutant_accepts(mode):
            raise Error("Finishing Nat join semantic mutant accepted: " + String(mode))
    for mode in range(FINISHING_JOIN_RESOURCE_CONTROLS):
        if finishing_join_resource_mutant_accepts(mode):
            raise Error("Finishing Nat join resource preflight failed: " + String(mode))
    print("Checked total Nat join semantic rejection controls:", FINISHING_JOIN_MUTANTS)
    print("Checked total Nat join resource preflight controls:", FINISHING_JOIN_RESOURCE_CONTROLS)
    print("Checked total Nat join and canonical finishing theorem passed")
