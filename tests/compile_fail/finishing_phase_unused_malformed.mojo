from finishing_phase_cases import phase_mutant_accepts, PHASE_UNUSED_MALFORMED


def main():
    comptime assert phase_mutant_accepts(PHASE_UNUSED_MALFORMED), "checked finishing phase commutation rejected"
