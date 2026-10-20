"""Valid typed coverage cannot bypass termination proof admission."""
from llm_mojo.kernels.argmax_reduction import sequential_argmax_plan
from llm_mojo.kernels.token_selection import (
    ArgmaxFoldPlan, make_argmax_fold_plan, check_argmax_fold_plan,
)
from llm_mojo.kernels.finishing_typed_coverage import check_finishing_typed_coverage


def malformed_fold() -> ArgmaxFoldPlan[32, 3, 7]:
    comptime plan = sequential_argmax_plan[3]()
    var fold = make_argmax_fold_plan[32, 3, 7, plan]()
    fold.typed_proof.termination_root = fold.typed_proof.base_root
    return fold^


def main():
    comptime fold = malformed_fold()
    comptime assert check_finishing_typed_coverage(fold.program, fold.proof), "coverage proof unexpectedly rejected"
    comptime assert check_argmax_fold_plan(fold), "proof term rejected"
    print(fold.stride)
