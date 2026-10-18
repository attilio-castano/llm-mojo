from finishing_phase_cases import phase_mutant_accepts, PHASE_SCRATCH_DEFINITIONS


def main():
    comptime assert phase_mutant_accepts(PHASE_SCRATCH_DEFINITIONS), "checked finishing phase commutation rejected"
