"""An admissible positive inductive must still match the actual driver."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from llm_mojo.kernels.finishing_typed_derivation import check_finishing_typed_proofs
from test_inductive_proofs import ALTERNATE_UNCONDITIONAL_STOP, alternate_done_proofs, alternate_done_root_is_typed


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime proofs = alternate_done_proofs(program, ALTERNATE_UNCONDITIONAL_STOP)
    comptime assert tk.check_inductive_definition(program, proofs.bundle), "generic inductive unexpectedly rejected"
    comptime assert alternate_done_root_is_typed(program), "alternate inductive proof unexpectedly rejected"
    comptime assert check_finishing_typed_proofs(program, proofs), "inductive admission rejected"
    print(proofs.termination_root)
