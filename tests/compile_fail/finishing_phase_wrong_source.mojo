from finishing_phase_cases import phase_mutant_accepts, PHASE_WRONG_SOURCE


def main():
    comptime assert phase_mutant_accepts(PHASE_WRONG_SOURCE), "checked finishing phase commutation rejected"
