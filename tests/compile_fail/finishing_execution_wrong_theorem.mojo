from finishing_execution_proof_cases import finishing_execution_proof_mutant_accepts, EXECUTION_WRONG_THEOREM


def main():
    comptime assert finishing_execution_proof_mutant_accepts(EXECUTION_WRONG_THEOREM), "checked finishing execution rejected"
