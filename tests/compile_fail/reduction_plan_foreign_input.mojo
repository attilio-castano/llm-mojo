"""Admission must reject a foreign leaf beyond a 65-input domain."""
from std.collections import Array
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord
from llm_mojo.kernels.argmax_reduction import ArgmaxPlan, input_node, evaluate_argmax_plan
from reduction_plan_cases import sequential_plan


def invalid_plan() -> ArgmaxPlan[65, 131]:
    var plan = sequential_plan[65]()
    plan.nodes[64] = input_node(65)
    return plan^


def main():
    comptime bad = invalid_plan()
    var inputs = Array[ArgmaxRecord, 65](uninitialized=True)
    for i in range(65):
        inputs[i] = ArgmaxRecord(0, UInt32(i), 0)
    var result = evaluate_argmax_plan[65, 131, bad](inputs)
    print(result.rank)
