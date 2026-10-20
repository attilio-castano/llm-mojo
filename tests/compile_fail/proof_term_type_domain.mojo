"""Malformed proof term must fail at compile-time admission."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from test_proof_terms import TYPE_DOMAIN, malformed_case


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime proof_case = malformed_case(TYPE_DOMAIN)
    comptime assert tk.check_proof_terms(program, proof_case.bundle, proof_case.expected), "proof term rejected"
    print(proof_case.bundle.root)
