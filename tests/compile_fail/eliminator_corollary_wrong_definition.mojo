"""The structural induction corollary must use its IH and intended definition."""
from llm_mojo.kernels.finishing_proof import make_finishing_program
from llm_mojo.kernels.finishing_typed_derivation import check_finishing_done_elimination_proof
from test_inductive_eliminator import malformed_monotonicity_proof


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime proof = malformed_monotonicity_proof(1)
    comptime assert check_finishing_done_elimination_proof(program, proof), "eliminator admission rejected"
