"""Generic finishing proof API and its untrusted canonical proof producer.

The historical profiles check coverage, visitation and integer headroom.
The full profile also derives finite execution from trusted operational
transitions. Both remain available for comparison tests. Production coverage
and termination use checked inductive definitions in the proof-term kernel.
Producing a target formula provides no proof of it.
"""
from llm_mojo.kernels.finishing_proof_kernel import *
from llm_mojo.kernels.finishing_derivation import (
    FINISHING_PROOF_SIZE, make_finishing_proof, make_finishing_coverage_proof,
)
