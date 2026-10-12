"""Closed argmax reductions over eight fixed symbolic input records.

Acceptance checks syntax and support. Its semantic consequence depends on
the shared merge's join laws and this checker/interpreter's correctness.
"""
from std.collections import Array
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, empty_argmax, merge_argmax,
)


comptime ARGMAX_INPUTS = 8
comptime EMPTY_NODE = 0
comptime INPUT_NODE = 1
comptime MERGE_NODE = 2


@fieldwise_init
struct ReductionNode(ImplicitlyCopyable):
    var operation: Int
    var left: Int
    var right: Int


def input_node(index: Int) -> ReductionNode:
    return ReductionNode(INPUT_NODE, index, 0)


def empty_node() -> ReductionNode:
    return ReductionNode(EMPTY_NODE, 0, 0)


def merge_node(left: Int, right: Int) -> ReductionNode:
    return ReductionNode(MERGE_NODE, left, right)


@fieldwise_init
struct ArgmaxPlan[size: Int](Copyable, Movable):
    var nodes: Array[ReductionNode, Self.size]
    var root: Int


def check_argmax_plan[size: Int](plan: ArgmaxPlan[size]) -> Bool:
    """Check every node, backward references, and full root support.

    Unreachable nodes must also be well formed. Root need not be the final
    node. Repeated leaves are accepted because the fixed join is idempotent.
    """
    if size < 1 or plan.root < 0 or plan.root >= size:
        return False
    var support = Array[UInt32, size](uninitialized=True)
    for i in range(size):
        var node = plan.nodes[i]
        if node.operation == EMPTY_NODE:
            if node.left != 0 or node.right != 0:
                return False
            support[i] = 0
        elif node.operation == INPUT_NODE:
            if node.left < 0 or node.left >= ARGMAX_INPUTS or node.right != 0:
                return False
            support[i] = UInt32(1) << UInt32(node.left)
        elif node.operation == MERGE_NODE:
            if node.left < 0 or node.left >= i or node.right < 0 or node.right >= i:
                return False
            support[i] = support[node.left] | support[node.right]
        else:
            return False
    return support[plan.root] == UInt32(0xFF)


def check_argmax_equivalence[left_size: Int, right_size: Int](
    left: ArgmaxPlan[left_size], right: ArgmaxPlan[right_size],
) -> Bool:
    """Both accepted roots denote the join of the same eight symbolic inputs.

    This uses the fixed primitive's trusted ACI/identity laws; it does not
    establish those laws by enumerating concrete input assignments.
    """
    return check_argmax_plan(left) and check_argmax_plan(right)


def evaluate_argmax_plan[size: Int, plan: ArgmaxPlan[size]](
    inputs: Array[ArgmaxRecord, ARGMAX_INPUTS],
) -> ArgmaxRecord:
    """Specialize execution from the exact compile-time description checked."""
    comptime assert check_argmax_plan(plan), "argmax reduction plan rejected"
    var values = Array[ArgmaxRecord, size](uninitialized=True)
    comptime for i in range(size):
        comptime node = plan.nodes[i]
        comptime if node.operation == EMPTY_NODE:
            values[i] = empty_argmax()
        elif node.operation == INPUT_NODE:
            values[i] = inputs[node.left]
        else:
            values[i] = merge_argmax(values[node.left], values[node.right])
    return values[plan.root]
