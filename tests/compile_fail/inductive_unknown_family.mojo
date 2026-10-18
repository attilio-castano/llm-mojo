"""Malformed inductive declaration or constructor term must fail admission."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from test_inductive_proofs import UNKNOWN_FAMILY, generic_indexed_case


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime proof_case = generic_indexed_case(UNKNOWN_FAMILY)
    comptime assert tk.check_proof_terms(program, proof_case.bundle, proof_case.expected), "inductive admission rejected"
    print(proof_case.expected)
