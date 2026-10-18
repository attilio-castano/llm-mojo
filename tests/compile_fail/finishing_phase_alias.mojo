from finishing_phase_cases import phase_mutant_accepts, PHASE_ALIAS


def main():
    comptime assert phase_mutant_accepts(PHASE_ALIAS), "checked finishing phase commutation rejected"
