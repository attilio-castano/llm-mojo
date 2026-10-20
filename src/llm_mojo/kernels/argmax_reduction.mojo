"""Closed argmax reductions over a compile-time count of symbolic records.

Acceptance checks syntax and support. Its semantic consequence depends on
the shared merge's join laws and this checker/interpreter's correctness.
"""
from std.collections import Array
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, empty_argmax, merge_argmax,
)


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
struct ArgmaxPlan[inputs: Int, size: Int](Copyable, Movable):
    var nodes: Array[ReductionNode, Self.size]
    var root: Int


def sequential_argmax_plan[inputs: Int]() -> ArgmaxPlan[inputs, 2 * inputs + 1]:
    """A left fold including one identity node, for any positive input count."""
    comptime assert inputs > 0, "argmax plan requires positive inputs"
    var nodes = Array[ReductionNode, 2 * inputs + 1](uninitialized=True)
    for i in range(inputs):
        nodes[i] = input_node(i)
    nodes[inputs] = empty_node()
    for i in range(inputs):
        nodes[inputs + 1 + i] = merge_node(inputs + i, i)
    return ArgmaxPlan[inputs, 2 * inputs + 1](nodes^, 2 * inputs)


def balanced_argmax_plan[inputs: Int]() -> ArgmaxPlan[inputs, 2 * inputs + 1]:
    """Pair each level, carrying an odd final node without duplicating it."""
    comptime assert inputs > 0, "argmax plan requires positive inputs"
    var nodes = Array[ReductionNode, 2 * inputs + 1](uninitialized=True)
    var active = Array[Int, inputs](uninitialized=True)
    for i in range(inputs):
        nodes[i] = input_node(i)
        active[i] = i
    nodes[inputs] = empty_node()
    var count = inputs
    var next_node = inputs + 1
    while count > 1:
        var pairs = count // 2
        for i in range(pairs):
            nodes[next_node] = merge_node(active[2 * i], active[2 * i + 1])
            active[i] = next_node
            next_node += 1
        if count % 2 != 0:
            active[pairs] = active[count - 1]
        count = (count + 1) // 2
    nodes[2 * inputs] = merge_node(active[0], inputs)
    return ArgmaxPlan[inputs, 2 * inputs + 1](nodes^, 2 * inputs)


def check_argmax_plan[inputs: Int, size: Int](plan: ArgmaxPlan[inputs, size]) -> Bool:
    """Check every node, backward references, and full root support.

    Unreachable nodes must also be well formed. Root need not be the final
    node. Repeated leaves are accepted because the fixed join is idempotent.
    """
    if inputs < 1 or size < 1 or plan.root < 0 or plan.root >= size:
        return False
    # One Boolean per (node, input) has no integer-mask width limit. This
    # matrix is checker state only; evaluate_argmax_plan does not allocate it.
    var support = Array[Bool, size * inputs](uninitialized=True)
    for i in range(size):
        var node = plan.nodes[i]
        if node.operation == EMPTY_NODE:
            if node.left != 0 or node.right != 0:
                return False
            for j in range(inputs):
                support[i * inputs + j] = False
        elif node.operation == INPUT_NODE:
            if node.left < 0 or node.left >= inputs or node.right != 0:
                return False
            for j in range(inputs):
                support[i * inputs + j] = j == node.left
        elif node.operation == MERGE_NODE:
            if node.left < 0 or node.left >= i or node.right < 0 or node.right >= i:
                return False
            for j in range(inputs):
                support[i * inputs + j] = (support[node.left * inputs + j]
                                          or support[node.right * inputs + j])
        else:
            return False
    for j in range(inputs):
        if not support[plan.root * inputs + j]:
            return False
    return True


def check_argmax_equivalence[inputs: Int, left_size: Int, right_size: Int](
    left: ArgmaxPlan[inputs, left_size], right: ArgmaxPlan[inputs, right_size],
) -> Bool:
    """Both accepted roots denote the join of the same symbolic inputs.

    This uses the fixed primitive's trusted ACI/identity laws; it does not
    establish those laws by enumerating concrete input assignments.
    """
    return check_argmax_plan(left) and check_argmax_plan(right)


@always_inline
def evaluate_argmax_plan[inputs: Int, size: Int, plan: ArgmaxPlan[inputs, size]](
    records: Array[ArgmaxRecord, inputs],
) -> ArgmaxRecord:
    """Specialize execution from the exact compile-time description checked."""
    comptime assert check_argmax_plan(plan), "argmax reduction plan rejected"
    var values = Array[ArgmaxRecord, size](uninitialized=True)
    comptime for i in range(size):
        comptime node = plan.nodes[i]
        comptime if node.operation == EMPTY_NODE:
            values[i] = empty_argmax()
        elif node.operation == INPUT_NODE:
            values[i] = records[node.left]
        else:
            values[i] = merge_argmax(values[node.left], values[node.right])
    return values[plan.root]
