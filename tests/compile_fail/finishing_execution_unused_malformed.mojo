from finishing_execution_proof_cases import finishing_execution_proof_mutant_accepts, EXECUTION_UNUSED_MALFORMED


def main():
    comptime assert finishing_execution_proof_mutant_accepts(EXECUTION_UNUSED_MALFORMED), "checked finishing execution rejected"
