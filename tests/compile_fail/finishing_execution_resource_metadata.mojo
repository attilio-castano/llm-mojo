from finishing_execution_proof_cases import finishing_execution_proof_mutant_accepts, EXECUTION_RESOURCE_METADATA


def main():
    comptime assert finishing_execution_proof_mutant_accepts(EXECUTION_RESOURCE_METADATA), "checked finishing execution rejected"
