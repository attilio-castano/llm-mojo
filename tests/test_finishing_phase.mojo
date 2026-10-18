"""Universal private-Fill proof checks and an independent native oracle."""
from std.testing import assert_equal
from finishing_phase_cases import (
    FINISHING_PHASE_MUTANTS, phase_pair_accepts, phase_mutant_accepts,
    independent_phase_contract_accepts, phase_alternate_proof_accepts,
    check_phase_native_controls, check_phase_counterexamples,
)


def main() raises:
    comptime assert phase_pair_accepts(3, 5, 0, 7, 0, 0, 2, 2), "checked finishing phase commutation rejected"
    comptime assert phase_pair_accepts(64, 3, 1, 4, 0, 0, 63, 2), "checked finishing phase commutation rejected"
    comptime assert phase_pair_accepts(Int(4294967296), 2, 0, 1, 0, 0, 0, 1), "checked finishing phase commutation rejected"
    assert_equal(independent_phase_contract_accepts(), True)
    assert_equal(phase_alternate_proof_accepts(), True)
    for mode in range(FINISHING_PHASE_MUTANTS):
        assert_equal(phase_mutant_accepts(mode), False)
    check_phase_native_controls()
    check_phase_counterexamples()
    print("Checked private finishing-phase commutation and execution controls passed")
