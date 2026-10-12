"""Concrete plan descriptions shared by acceptance and rejection tests."""
from std.collections import Array
from llm_mojo.kernels.argmax_reduction import (
    ReductionNode, ArgmaxPlan, input_node, empty_node, merge_node,
)


def sequential_plan() -> ArgmaxPlan[17]:
    var nodes = Array[ReductionNode, 17](uninitialized=True)
    for i in range(8):
        nodes[i] = input_node(i)
    nodes[8] = empty_node()
    for i in range(8):
        nodes[9 + i] = merge_node(8 + i, i)
    return ArgmaxPlan[17](nodes^, 16)


def balanced_plan() -> ArgmaxPlan[17]:
    var plan = sequential_plan()
    for i in range(4):
        plan.nodes[9 + i] = merge_node(2 * i, 2 * i + 1)
    plan.nodes[13] = merge_node(9, 10)
    plan.nodes[14] = merge_node(11, 12)
    plan.nodes[15] = merge_node(13, 14)
    plan.nodes[16] = merge_node(15, 8)
    return plan^


def repeated_permuted_plan() -> ArgmaxPlan[17]:
    var plan = balanced_plan()
    for i in range(8):
        plan.nodes[i] = input_node(7 - i)
    plan.nodes[16] = merge_node(15, 15)
    return plan^


def earlier_root_plan() -> ArgmaxPlan[17]:
    var plan = balanced_plan()
    plan.root = 15
    plan.nodes[16] = input_node(0)
    return plan^
