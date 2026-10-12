"""Compile-time acceptance and native execution of closed argmax plans."""
from std.collections import Array
from std.testing import assert_equal
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord, empty_argmax
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, ReductionNode, input_node, empty_node, merge_node,
    check_argmax_plan, check_argmax_equivalence, evaluate_argmax_plan,
)
from reduction_plan_cases import (
    sequential_plan, balanced_plan, repeated_permuted_plan, earlier_root_plan,
)


def specification(inputs: Array[ArgmaxRecord, 8]) -> ArgmaxRecord:
    # Two-pass oracle uses no merge primitive or plan interpretation.
    var top: UInt32 = 0
    var invalid: UInt32 = 0
    for i in range(8):
        if inputs[i].rank > top:
            top = inputs[i].rank
        if inputs[i].invalid != 0:
            invalid = 1
    var token: UInt32 = 0xFFFFFFFF
    for i in range(8):
        if inputs[i].rank == top and inputs[i].token < token:
            token = inputs[i].token
    return ArgmaxRecord(top, token, invalid)


def same_record(got: ArgmaxRecord, expected: ArgmaxRecord) raises:
    assert_equal(got.rank, expected.rank)
    assert_equal(got.token, expected.token)
    assert_equal(got.invalid, expected.invalid)


def check_execution(inputs: Array[ArgmaxRecord, 8]) raises:
    comptime sequential = sequential_plan()
    comptime balanced = balanced_plan()
    comptime repeated = repeated_permuted_plan()
    comptime earlier = earlier_root_plan()
    var expected = specification(inputs)
    same_record(evaluate_argmax_plan[17, sequential](inputs), expected)
    same_record(evaluate_argmax_plan[17, balanced](inputs), expected)
    same_record(evaluate_argmax_plan[17, repeated](inputs), expected)
    same_record(evaluate_argmax_plan[17, earlier](inputs), expected)


def check_malformed_plans() raises:
    # Exercise native checking as well as the negative compilation fixtures.
    var plan = balanced_plan()
    plan.root = 15  # A complete root may precede later well-formed nodes.
    assert_equal(check_argmax_plan(plan), True)
    plan.nodes[16] = input_node(8)  # Even dead foreign nodes are rejected.
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan()
    plan.nodes[8] = ReductionNode(99, 0, 0)
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan()
    plan.nodes[8] = empty_node()
    plan.nodes[8].left = 1
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan()
    plan.nodes[0] = input_node(0)
    plan.nodes[0].right = 1
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan()
    plan.nodes[0] = input_node(-1)
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan()
    plan.nodes[0] = input_node(32)  # Reject before a shift could alias a bit.
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan()
    plan.nodes[9] = merge_node(0, -1)
    assert_equal(check_argmax_plan(plan), False)
    plan.nodes[9] = merge_node(0, 9)
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan()
    plan.root = -1
    assert_equal(check_argmax_plan(plan), False)
    plan.root = 17
    assert_equal(check_argmax_plan(plan), False)
    var no_nodes = Array[ReductionNode, 0](uninitialized=True)
    var empty = ArgmaxPlan[0](no_nodes^, 0)
    assert_equal(check_argmax_plan(empty), False)


def main() raises:
    comptime sequential = sequential_plan()
    comptime balanced = balanced_plan()
    comptime repeated = repeated_permuted_plan()
    comptime assert check_argmax_equivalence(sequential, balanced), "argmax plans not equivalent"
    comptime assert check_argmax_equivalence(balanced, repeated), "argmax plans not equivalent"
    var inputs: Array[ArgmaxRecord, 8] = [
        ArgmaxRecord(3, 100, 0), ArgmaxRecord(1, 0, 1),
        ArgmaxRecord(3, 31, 0), ArgmaxRecord(3, 1024, 0),
        ArgmaxRecord(0, 7, 0), ArgmaxRecord(2, 42, 0),
        ArgmaxRecord(3, 32, 0), ArgmaxRecord(1, 13, 0),
    ]
    same_record(evaluate_argmax_plan[17, sequential](inputs), ArgmaxRecord(3, 31, 1))
    check_execution(inputs)
    # Endpoint/interior ranks and IDs, every winner position, ties, and poison.
    var values: Array[UInt32, 6] = [0, 1, 31, 1024, 0xFFFFFFFE, 0xFFFFFFFF]
    for r in range(6):
        for t in range(6):
            for position in range(8):
                for i in range(8):
                    inputs[i] = ArgmaxRecord(values[r], values[t], UInt32(i % 2))
                inputs[position] = ArgmaxRecord(values[(r + 1) % 6], values[(t + 1) % 6], 0)
                check_execution(inputs)
    for i in range(8):
        inputs[i] = ArgmaxRecord(31, UInt32(100 + i), 0)
    check_execution(inputs)
    # A single losing leaf must still poison every tree's result.
    for position in range(8):
        for i in range(8):
            inputs[i] = ArgmaxRecord(31, UInt32(100 + i), 0)
        inputs[position] = ArgmaxRecord(0, 0, 1)
        check_execution(inputs)
    for i in range(8):
        inputs[i] = empty_argmax()
    check_execution(inputs)
    var runtime_plan = sequential_plan()
    assert_equal(check_argmax_plan(runtime_plan), True)
    check_malformed_plans()
    print("Closed argmax plan coverage and composition checks passed")
