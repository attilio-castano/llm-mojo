from finishing_phase_cases import phase_mutant_accepts, PHASE_REQUEST_BINDING


def main():
    comptime assert phase_mutant_accepts(PHASE_REQUEST_BINDING), "checked finishing phase commutation rejected"
