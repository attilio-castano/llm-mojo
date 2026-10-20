from finishing_phase_cases import phase_mutant_accepts, PHASE_STALE_FOOTPRINT


def main():
    comptime assert phase_mutant_accepts(PHASE_STALE_FOOTPRINT), "checked finishing phase commutation rejected"
