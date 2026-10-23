"""Compiler must reject a malformed Nat definition or eliminator term."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from test_nat_inductive import malformed_nat_case, MISSING_CONSTRUCTOR


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime malformed = malformed_nat_case(MISSING_CONSTRUCTOR)
    comptime assert tk.check_proof_terms(program, malformed.bundle, malformed.expected), "Nat inductive admission rejected"
