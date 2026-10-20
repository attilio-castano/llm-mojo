"""Fold admission must reject a stride larger than the tile omits inputs."""
from llm_mojo.kernels.argmax_reduction import sequential_argmax_plan, empty_node
from llm_mojo.kernels.token_selection import (
    ArgmaxFoldPlan, make_argmax_fold_plan, check_argmax_fold_plan,
)


def malformed_fold() -> ArgmaxFoldPlan[32, 3, 7]:
    comptime plan = sequential_argmax_plan[3]()
    var fold = make_argmax_fold_plan[32, 3, 7, plan]()
    fold.stride = 97
    return fold^


def main():
    comptime fold = malformed_fold()
    comptime assert check_argmax_fold_plan(fold), "argmax fold plan rejected"
    print(fold.stride)
