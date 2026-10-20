"""Malformed finite-execution derivation must fail at proof admission."""
from llm_mojo.kernels.finishing_proof import make_finishing_program, check_finishing_proof
from test_finishing_proof import BAD_RETIRED_TERMINATION_NAMED, malformed_certificate


def main():
    comptime program = make_finishing_program(32, 3, 96)
    comptime proof = malformed_certificate(program, BAD_RETIRED_TERMINATION_NAMED)
    comptime assert check_finishing_proof(program, proof), "finishing proof rejected"
    print(proof.root)
