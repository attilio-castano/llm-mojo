"""Rejected by the checked inductive eliminator, rather than a parse failure."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from test_inductive_eliminator import malformed_eliminator_case


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime proof_case = malformed_eliminator_case(19)
    comptime assert tk.check_proof_terms(program, proof_case.bundle, proof_case.expected), "eliminator admission rejected"
