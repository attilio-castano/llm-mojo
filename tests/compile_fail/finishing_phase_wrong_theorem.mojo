from finishing_phase_cases import phase_mutant_accepts, PHASE_WRONG_THEOREM


def main():
    comptime assert phase_mutant_accepts(PHASE_WRONG_THEOREM), "checked finishing phase commutation rejected"
