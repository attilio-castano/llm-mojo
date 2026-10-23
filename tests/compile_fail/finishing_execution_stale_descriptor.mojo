from finishing_execution_proof_cases import finishing_execution_proof_mutant_accepts, EXECUTION_STALE_DESCRIPTOR


def main():
    comptime assert finishing_execution_proof_mutant_accepts(EXECUTION_STALE_DESCRIPTOR), "checked finishing execution rejected"
