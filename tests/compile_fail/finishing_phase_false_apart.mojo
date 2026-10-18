from finishing_phase_cases import phase_mutant_accepts, PHASE_FALSE_APART


def main():
    comptime assert phase_mutant_accepts(PHASE_FALSE_APART), "checked finishing phase commutation rejected"
