"""Malformed typed driver derivation must fail at compile-time admission."""
from llm_mojo.kernels.finishing_proof import make_finishing_program
from llm_mojo.kernels.finishing_typed_derivation import check_finishing_typed_proofs
from test_proof_terms import BAD_TYPED_BASE_INDUCTION, malformed_typed_proofs


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime proofs = malformed_typed_proofs(program, BAD_TYPED_BASE_INDUCTION)
    comptime assert check_finishing_typed_proofs(program, proofs), "proof term rejected"
    print(proofs.termination_root)
