"""An unused legacy Done constructor cannot enter production admission."""
from llm_mojo.kernels.argmax_reduction import sequential_argmax_plan
from llm_mojo.kernels.token_selection import (
    ArgmaxFoldPlan, make_argmax_fold_plan, check_argmax_fold_plan,
)
import llm_mojo.kernels.proof_term_kernel as tk


def malformed_fold() -> ArgmaxFoldPlan[32, 3, 7]:
    comptime plan = sequential_argmax_plan[3]()
    var fold = make_argmax_fold_plan[32, 3, 7, plan]()
    var zero = tk.pt_add(fold.proof.bundle, tk.PT_LITERAL, value=0)
    _ = tk.pt_add(fold.proof.bundle, tk.PT_DONE, zero, zero)
    return fold^


def main():
    comptime plan = sequential_argmax_plan[3]()
    comptime accepted = make_argmax_fold_plan[32, 3, 7, plan]()
    comptime assert check_argmax_fold_plan(accepted), "valid production plan rejected"
    comptime rejected = malformed_fold()
    comptime assert check_argmax_fold_plan(rejected), "inductive admission rejected"
    print(rejected.stride)
