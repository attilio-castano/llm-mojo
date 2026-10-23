"""A valid restricted count proof cannot certify the public count domain."""
from llm_mojo.kernels.argmax_reduction import sequential_argmax_plan
from llm_mojo.kernels.finishing_proof import (
    make_finishing_program,
)
from llm_mojo.kernels.finishing_typed_coverage import (
    make_finishing_typed_coverage, check_finishing_typed_coverage,
)
from llm_mojo.kernels.token_selection import (
    ArgmaxFoldPlan, make_argmax_fold_plan, check_argmax_fold_plan,
)


def restricted_fold() -> ArgmaxFoldPlan[32, 3, 7]:
    comptime plan = sequential_argmax_plan[3]()
    var fold = make_argmax_fold_plan[32, 3, 7, plan]()
    fold.program = make_finishing_program(32, 3, 96, 1000)
    fold.proof = make_finishing_typed_coverage(fold.program)
    return fold^


def main():
    comptime fold = restricted_fold()
    comptime assert check_finishing_typed_coverage(fold.program, fold.proof)
    comptime assert check_argmax_fold_plan(fold), "finishing proof rejected"
    print(fold.stride)
