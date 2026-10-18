from finishing_execution_proof_cases import finishing_execution_proof_mutant_accepts, EXECUTION_FALSE_FULL_SUPPORT


def main():
    comptime assert finishing_execution_proof_mutant_accepts(EXECUTION_FALSE_FULL_SUPPORT), "checked finishing execution rejected"
