"""An independent open finite-execution assumption cannot escape."""
from llm_mojo.kernels.finishing_proof import make_finishing_program, check_finishing_proof
from test_finishing_proof import case_context_proof


def main():
    comptime program = make_finishing_program(32, 3, 96)
    comptime proof = case_context_proof(program, True)
    comptime assert check_finishing_proof(program, proof), "finishing proof rejected"
    print(proof.root)
