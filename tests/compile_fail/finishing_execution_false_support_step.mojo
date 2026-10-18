from finishing_execution_proof_cases import finishing_execution_proof_mutant_accepts, EXECUTION_FALSE_SUPPORT_STEP


def main():
    comptime assert finishing_execution_proof_mutant_accepts(EXECUTION_FALSE_SUPPORT_STEP), "checked finishing execution rejected"
