from finishing_execution_proof_cases import finishing_execution_proof_mutant_accepts, EXECUTION_WRONG_SOURCE


def main():
    comptime assert finishing_execution_proof_mutant_accepts(EXECUTION_WRONG_SOURCE), "checked finishing execution rejected"
