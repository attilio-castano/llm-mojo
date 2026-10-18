"""An independent open finite-execution assumption cannot escape."""
from llm_mojo.kernels.finishing_proof import make_finishing_program, check_finishing_proof
from test_finishing_proof import done_scope_escape_proof


def main():
    comptime program = make_finishing_program(32, 3, 96)
    comptime proof = done_scope_escape_proof(program)
    comptime assert check_finishing_proof(program, proof), "finishing proof rejected"
    print(proof.root)
