"""Concrete plan descriptions shared by acceptance and rejection tests."""
from std.collections import Array
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, ReductionNode, input_node, merge_node,
    sequential_argmax_plan, balanced_argmax_plan,
)


def sequential_plan[inputs: Int]() -> ArgmaxPlan[inputs, 2 * inputs + 1]:
    return sequential_argmax_plan[inputs]()


def balanced_plan[inputs: Int]() -> ArgmaxPlan[inputs, 2 * inputs + 1]:
    return balanced_argmax_plan[inputs]()


def compact_plan[inputs: Int]() -> ArgmaxPlan[inputs, 2 * inputs - 1]:
    """Remove identity storage and the final identity merge from a balanced plan."""
    var original = balanced_plan[inputs]()
    var nodes = Array[ReductionNode, 2 * inputs - 1](uninitialized=True)
    for i in range(inputs):
        nodes[i] = input_node(i)
    for i in range(inputs - 1):
        var node = original.nodes[inputs + 1 + i]
        # The balanced tree's internal merges never refer to its Empty node.
        if node.left > inputs:
            node.left -= 1
        if node.right > inputs:
            node.right -= 1
        nodes[inputs + i] = node
    return ArgmaxPlan[inputs, 2 * inputs - 1](nodes^, 2 * inputs - 2)


def repeated_permuted_plan[inputs: Int]() -> ArgmaxPlan[inputs, 2 * inputs + 1]:
    var plan = balanced_plan[inputs]()
    for i in range(inputs):
        plan.nodes[i] = input_node(inputs - 1 - i)
    var complete = 0 if inputs == 1 else 2 * inputs - 1
    plan.nodes[2 * inputs] = merge_node(complete, complete)
    return plan^


def earlier_root_plan[inputs: Int]() -> ArgmaxPlan[inputs, 2 * inputs + 1]:
    var plan = balanced_plan[inputs]()
    plan.root = 0 if inputs == 1 else 2 * inputs - 1
    plan.nodes[2 * inputs] = input_node(0)
    return plan^
