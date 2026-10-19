"""Compilation-cost probe; every selected certificate remains mandatory.

PROFILE_STAGE: 0 arithmetic bootstrap, 1 coverage, 2 termination, 3 execution,
4 production fold and execution binding. PROFILE_COPIES makes distinct caller
specializations with an identical mathematical shape, like different layouts.
This is a host compiler diagnostic, not a GPU correctness/performance test.
"""
from std.sys import get_defined_int
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from llm_mojo.kernels.finishing_typed_coverage import (
    make_finishing_typed_coverage, check_finishing_typed_coverage,
)
from llm_mojo.kernels.finishing_typed_derivation import (
    make_finishing_typed_proofs, check_finishing_typed_proofs,
)
from llm_mojo.kernels.finishing_execution_check import (
    check_finishing_execution_rules, make_finishing_execution_certificate,
    check_finishing_execution_certificate_request,
)
from llm_mojo.kernels.argmax_reduction import sequential_argmax_plan
from llm_mojo.kernels.token_selection import (
    make_argmax_fold_plan, check_argmax_fold_plan,
    make_argmax_finishing_execution, check_argmax_finishing_execution,
    _argmax_execution_descriptor,
)
from llm_mojo.kernels.finishing_checked_exports import (
    FinishingTraversalLibrary, canonical_finishing_execution_admitted,
    check_canonical_finishing_execution,
)


def check_case[copy_index: Int]():
    comptime stage = get_defined_int["PROFILE_STAGE", default=4]()
    comptime reuse = get_defined_int["PROFILE_REUSE", default=1]()
    comptime assert 0 <= stage <= 4, "unknown compilation profile stage"
    comptime program = make_finishing_program(64, 3, 192)
    comptime if stage == 0:
        comptime assert tk._NAT_ARITHMETIC_CERTIFIED
    elif stage == 1:
        comptime if reuse:
            comptime assert FinishingTraversalLibrary[program].coverage_accepted
        else:
            comptime proof = make_finishing_typed_coverage(program)
            comptime assert check_finishing_typed_coverage(program, proof)
    elif stage == 2:
        comptime if reuse:
            comptime assert FinishingTraversalLibrary[program].termination_accepted
        else:
            comptime proof = make_finishing_typed_proofs(program)
            comptime assert check_finishing_typed_proofs(program, proof)
    elif stage == 3:
        comptime if reuse:
            comptime assert canonical_finishing_execution_admitted()
        else:
            comptime assert check_finishing_execution_rules()
    else:
        comptime plan = sequential_argmax_plan[3]()
        comptime if reuse:
            comptime descriptor = _argmax_execution_descriptor[64, 3, 7](materialize[plan]())
            comptime assert FinishingTraversalLibrary[descriptor.program].accepted
            comptime assert check_canonical_finishing_execution(descriptor)
        else:
            comptime fold = make_argmax_fold_plan[64, 3, 7, plan]()
            comptime assert check_argmax_fold_plan(fold)
            comptime descriptor = make_argmax_finishing_execution(fold)
            comptime assert check_argmax_finishing_execution(fold, descriptor)
            comptime certificate = make_finishing_execution_certificate(descriptor)
            comptime assert check_finishing_execution_certificate_request(certificate, descriptor)
    print("Checked compilation profile caller", copy_index, "stage", stage)


def main():
    comptime copies = get_defined_int["PROFILE_COPIES", default=1]()
    comptime assert 1 <= copies <= 32, "compilation profile copies out of bounds"
    comptime for copy_index in range(copies):
        check_case[copy_index]()
