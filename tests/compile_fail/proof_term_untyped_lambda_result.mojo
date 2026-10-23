"""A lambda cannot synthesize a Pi type whose codomain Type is untyped."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from test_proof_terms import UNTYPED_LAMBDA_RESULT, malformed_case


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime proof_case = malformed_case(UNTYPED_LAMBDA_RESULT)
    comptime assert tk.check_proof_terms(program, proof_case.bundle, proof_case.expected), "proof term rejected"
    print(proof_case.bundle.root)
