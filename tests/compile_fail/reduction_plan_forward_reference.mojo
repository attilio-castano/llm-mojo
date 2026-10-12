"""Execution admission must reject a merge that refers to itself."""
from std.collections import Array
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord
from llm_mojo.kernels.argmax_reduction import ArgmaxPlan, merge_node, evaluate_argmax_plan
from reduction_plan_cases import sequential_plan


def invalid_plan() -> ArgmaxPlan[17]:
    var plan = sequential_plan()
    plan.nodes[9] = merge_node(9, 0)
    return plan^


def main():
    comptime bad = invalid_plan()
    var inputs: Array[ArgmaxRecord, 8] = [
        ArgmaxRecord(0, 0, 0), ArgmaxRecord(0, 1, 0),
        ArgmaxRecord(0, 2, 0), ArgmaxRecord(0, 3, 0),
        ArgmaxRecord(0, 4, 0), ArgmaxRecord(0, 5, 0),
        ArgmaxRecord(0, 6, 0), ArgmaxRecord(0, 7, 0),
    ]
    var result = evaluate_argmax_plan[17, bad](inputs)
    print(result.rank)
