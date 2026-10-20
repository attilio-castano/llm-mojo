"""Reject a certificate for a different or inadmissible finishing program."""
from llm_mojo.kernels.finishing_proof import (
    FinishingProgram, make_finishing_program, make_finishing_proof, check_finishing_proof,
)


def malformed_program() -> FinishingProgram:
    var program = make_finishing_program(32, 3, 96)
    program.loop_guard = 1
    return program


def main():
    comptime changed = malformed_program()
    comptime proof = make_finishing_proof(changed)
    comptime assert check_finishing_proof(changed, proof), "finishing proof rejected"
    print(changed.stride)
