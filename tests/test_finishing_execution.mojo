"""Universal checked trace theorem and independent complete execution oracles."""
from std.testing import assert_equal
from finishing_execution_cases import (
    check_finishing_execution_native, check_finishing_execution_controls,
    check_finishing_execution_replay_controls,
)
from finishing_execution_proof_cases import (
    execution_certificate_accepts, independent_execution_completion_accepts,
    independent_execution_canonical_accepts, execution_alternate_proof_accepts,
    FINISHING_EXECUTION_PROOF_MUTANTS, finishing_execution_proof_mutant_accepts,
)


def main() raises:
    comptime assert execution_certificate_accepts(), "checked finishing execution rejected"
    assert_equal(independent_execution_completion_accepts(), True)
    assert_equal(independent_execution_canonical_accepts(), True)
    assert_equal(execution_alternate_proof_accepts(), True)
    for mode in range(FINISHING_EXECUTION_PROOF_MUTANTS):
        if finishing_execution_proof_mutant_accepts(mode):
            raise Error("Finishing execution proof mutant accepted: " + String(mode))
    check_finishing_execution_native()
    check_finishing_execution_controls()
    check_finishing_execution_replay_controls()
    print("Finishing execution malformed proof controls:", FINISHING_EXECUTION_PROOF_MUTANTS)
    print("Checked whole finishing execution and independent controls passed")
