"""Compile-time acceptance and native execution of flexible closed argmax plans."""
from std.collections import Array
from std.testing import assert_equal
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord, empty_argmax
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, ReductionNode, input_node, empty_node, merge_node,
    check_argmax_plan, check_argmax_equivalence, evaluate_argmax_plan,
)
from reduction_plan_cases import (
    sequential_plan, balanced_plan, compact_plan, repeated_permuted_plan, earlier_root_plan,
)


def specification[inputs: Int](records: Array[ArgmaxRecord, inputs]) -> ArgmaxRecord:
    # Two-pass oracle uses no merge primitive or plan interpretation.
    var top: UInt32 = 0
    var invalid: UInt32 = 0
    for i in range(inputs):
        if records[i].rank > top:
            top = records[i].rank
        if records[i].invalid != 0:
            invalid = 1
    var token: UInt32 = 0xFFFFFFFF
    for i in range(inputs):
        if records[i].rank == top and records[i].token < token:
            token = records[i].token
    return ArgmaxRecord(top, token, invalid)


def same_record(got: ArgmaxRecord, expected: ArgmaxRecord) raises:
    assert_equal(got.rank, expected.rank)
    assert_equal(got.token, expected.token)
    assert_equal(got.invalid, expected.invalid)


def check_execution[inputs: Int](records: Array[ArgmaxRecord, inputs]) raises:
    comptime sequential = sequential_plan[inputs]()
    comptime balanced = balanced_plan[inputs]()
    comptime compact = compact_plan[inputs]()
    comptime repeated = repeated_permuted_plan[inputs]()
    comptime earlier = earlier_root_plan[inputs]()
    var expected = specification(records)
    same_record(evaluate_argmax_plan[inputs, 2 * inputs + 1, sequential](records), expected)
    same_record(evaluate_argmax_plan[inputs, 2 * inputs + 1, balanced](records), expected)
    same_record(evaluate_argmax_plan[inputs, 2 * inputs - 1, compact](records), expected)
    same_record(evaluate_argmax_plan[inputs, 2 * inputs + 1, repeated](records), expected)
    same_record(evaluate_argmax_plan[inputs, 2 * inputs + 1, earlier](records), expected)


def check_malformed_plans[inputs: Int]() raises:
    # Exercise native checking as well as the negative compilation fixtures.
    var plan = earlier_root_plan[inputs]()
    assert_equal(check_argmax_plan(plan), True)
    plan.nodes[2 * inputs] = input_node(inputs)  # Dead foreign nodes are rejected.
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    plan.nodes[inputs] = ReductionNode(99, 0, 0)
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    plan.nodes[inputs] = empty_node()
    plan.nodes[inputs].left = 1
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    plan.nodes[0].right = 1
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    plan.nodes[0] = input_node(-1)
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    plan.nodes[0] = input_node(inputs)
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    plan.nodes[inputs + 1] = merge_node(0, -1)
    assert_equal(check_argmax_plan(plan), False)
    plan.nodes[inputs + 1] = merge_node(0, inputs + 1)
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    plan.root = -1
    assert_equal(check_argmax_plan(plan), False)
    plan.root = 2 * inputs + 1
    assert_equal(check_argmax_plan(plan), False)
    plan = balanced_plan[inputs]()
    # Counts above 32/64 must still detect an omitted final symbolic input.
    plan.nodes[inputs - 1] = empty_node()
    assert_equal(check_argmax_plan(plan), False)
    var no_nodes = Array[ReductionNode, 0](uninitialized=True)
    var empty = ArgmaxPlan[inputs, 0](no_nodes^, 0)
    assert_equal(check_argmax_plan(empty), False)


def check_input_count[inputs: Int]() raises:
    comptime sequential = sequential_plan[inputs]()
    comptime balanced = balanced_plan[inputs]()
    comptime compact = compact_plan[inputs]()
    comptime repeated = repeated_permuted_plan[inputs]()
    comptime earlier = earlier_root_plan[inputs]()
    comptime assert check_argmax_equivalence(sequential, balanced), "argmax plans not equivalent"
    comptime assert check_argmax_equivalence(balanced, compact), "argmax plans not equivalent"
    comptime assert check_argmax_equivalence(balanced, repeated), "argmax plans not equivalent"
    comptime assert check_argmax_equivalence(repeated, earlier), "argmax plans not equivalent"
    var records = Array[ArgmaxRecord, inputs](uninitialized=True)
    # Endpoint/interior ranks and IDs, every winner position, ties, and poison.
    var values: Array[UInt32, 6] = [0, 1, 31, 1024, 0xFFFFFFFE, 0xFFFFFFFF]
    for r in range(6):
        for t in range(6):
            for position in range(inputs):
                for i in range(inputs):
                    records[i] = ArgmaxRecord(values[r], values[t], UInt32(i % 2))
                records[position] = ArgmaxRecord(values[(r + 1) % 6], values[(t + 1) % 6], 0)
                check_execution(records)
    for i in range(inputs):
        records[i] = ArgmaxRecord(31, UInt32(100 + i), 0)
    check_execution(records)
    # A single losing leaf must still poison every tree's result.
    for position in range(inputs):
        for i in range(inputs):
            records[i] = ArgmaxRecord(31, UInt32(100 + i), 0)
        records[position] = ArgmaxRecord(0, 0, 1)
        check_execution(records)
    for i in range(inputs):
        records[i] = empty_argmax()
    check_execution(records)
    var runtime_plan = sequential_plan[inputs]()
    assert_equal(check_argmax_plan(runtime_plan), True)
    check_malformed_plans[inputs]()


def main() raises:
    check_input_count[1]()
    check_input_count[3]()
    check_input_count[8]()
    check_input_count[33]()
    check_input_count[65]()
    var nodes = Array[ReductionNode, 1](uninitialized=True)
    nodes[0] = empty_node()
    var no_inputs = ArgmaxPlan[0, 1](nodes^, 0)
    assert_equal(check_argmax_plan(no_inputs), False)
    print("Flexible closed argmax plan coverage and composition checks passed")
