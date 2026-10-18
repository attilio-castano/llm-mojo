"""Finite tile checks and runtime ownership/oracle checks for closed tiled folds."""
from std.collections import Array
from std.testing import assert_equal
from llm_mojo.kernels.argmax_reduction import (
    sequential_argmax_plan, balanced_argmax_plan, evaluate_argmax_plan, empty_node,
)
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord, empty_argmax, merge_argmax
from llm_mojo.kernels.token_selection import (
    ArgmaxFoldPlan, make_argmax_fold_plan, check_argmax_fold_plan,
    check_argmax_schedule, argmax_tile_offset, argmax_fold_index,
)


def check_instance[threads: Int, items: Int]() raises:
    comptime assert check_argmax_schedule[threads, items](), "argmax schedule rejected"
    # Independent ownership census: visit (item, thread), mark each destination
    # once, and check the actual production offset helper's strided mapping.
    var visited = List[Bool](capacity=threads * items)
    for _ in range(threads * items):
        visited.append(False)
    for item in range(items):
        for thread in range(threads):
            var offset = argmax_tile_offset[threads](thread, item)
            assert_equal(offset, item * threads + thread)
            assert_equal(visited[offset], False)
            visited[offset] = True
    for present in visited:
        assert_equal(present, True)


def specification(records: List[ArgmaxRecord]) -> ArgmaxRecord:
    # Independent two-pass oracle: no scalar merge, AST or tile traversal.
    var top: UInt32 = 0
    var invalid: UInt32 = 0
    for record in records:
        if record.rank > top:
            top = record.rank
        if record.invalid != 0:
            invalid = 1
    var token: UInt32 = 0xFFFFFFFF
    for record in records:
        if record.rank == top and record.token < token:
            token = record.token
    return ArgmaxRecord(top, token, invalid)


def same_record(got: ArgmaxRecord, expected: ArgmaxRecord) raises:
    assert_equal(got.rank, expected.rank)
    assert_equal(got.token, expected.token)
    assert_equal(got.invalid, expected.invalid)


def simulate_fold[
    threads: Int, items: Int, nodes: Int, fold: ArgmaxFoldPlan[threads, items, nodes],
](records: List[ArgmaxRecord]) -> ArgmaxRecord:
    """Host semantic check of guarded tiles and their accepted local expression."""
    comptime assert check_argmax_fold_plan(fold), "argmax fold plan rejected"
    comptime plan = fold.reduction
    var best = empty_argmax()
    for thread in range(threads):
        var local_best = empty_argmax()
        var base = 0
        while base < len(records):
            var local = Array[ArgmaxRecord, items](uninitialized=True)
            for item in range(items):
                var index = argmax_fold_index[threads](base, thread, item)
                local[item] = empty_argmax()
                if index < len(records):
                    local[item] = records[index]
            local_best = merge_argmax(local_best, evaluate_argmax_plan[items, nodes, plan](local))
            base += fold.stride
        best = merge_argmax(best, local_best)
    return best


def check_count[
    threads: Int, items: Int, nodes: Int, fold: ArgmaxFoldPlan[threads, items, nodes],
](count: Int) raises:
    # Count each guarded read over multiple tiles, including a ragged final tile.
    var visited = List[Bool](capacity=count)
    for _ in range(count):
        visited.append(False)
    var base = 0
    while base < count:
        for thread in range(threads):
            for item in range(items):
                var index = argmax_fold_index[threads](base, thread, item)
                assert_equal(index >= base and index < base + fold.stride, True)
                if index < count:
                    assert_equal(visited[index], False)
                    visited[index] = True
        base += fold.stride
    assert_equal(base >= count and base < count + fold.stride, True)
    for index in range(count):
        assert_equal(visited[index], True)
        # An independent inverse identifies the unique guarded source slot.
        var tile = index // fold.stride
        var offset = index % fold.stride
        var thread = offset % threads
        var item = offset // threads
        assert_equal(item >= 0 and item < items, True)
        assert_equal(argmax_fold_index[threads](tile * fold.stride, thread, item), index)

    for pattern in range(5):
        var records = List[ArgmaxRecord](capacity=count)
        for index in range(count):
            var record = ArgmaxRecord(UInt32((index * 17) % 37), UInt32(index), 0)
            if pattern == 1:
                record = ArgmaxRecord(31, UInt32(count - 1 - index), 0)
            elif pattern == 2:
                record = ArgmaxRecord(0, UInt32(index), 0)
                if index == count - 1:
                    record.rank = 0xFFFFFFFF
            elif pattern == 3:
                record = ArgmaxRecord(31, UInt32(index), 0)
                if index == count - 1:
                    record = ArgmaxRecord(0, UInt32(index), 1)
            elif pattern == 4:
                record = empty_argmax()
            records.append(record)
        same_record(simulate_fold[threads, items, nodes, fold](records), specification(records))


def check_endpoint[
    threads: Int, items: Int, nodes: Int, fold: ArgmaxFoldPlan[threads, items, nodes],
]() raises:
    # Algebra at the allowed runtime-count endpoint; no INT32_MAX-sized census.
    var count = 2147483647
    var tiles = 1 + (count - 1) // fold.stride
    var last_base = (tiles - 1) * fold.stride
    assert_equal(last_base < count, True)
    assert_equal(tiles * fold.stride >= count, True)
    assert_equal(tiles * fold.stride > 2147483647, True)
    assert_equal(tiles * fold.stride < count + fold.stride, True)
    var samples: Array[Int, 7] = [
        0, threads - 1, fold.stride - 1, fold.stride,
        last_base, count - 2, count - 1,
    ]
    for index in samples:
        var tile = index // fold.stride
        var offset = index % fold.stride
        var thread = offset % threads
        var item = offset // threads
        assert_equal(tile >= 0 and tile < tiles, True)
        assert_equal(thread >= 0 and thread < threads, True)
        assert_equal(item >= 0 and item < items, True)
        assert_equal(argmax_fold_index[threads](tile * fold.stride, thread, item), index)
    var final_index = argmax_fold_index[threads](last_base, threads - 1, items - 1)
    assert_equal(final_index, tiles * fold.stride - 1)
    assert_equal(final_index < count + fold.stride, True)
    assert_equal(final_index >= count, True)
    assert_equal(count + fold.stride <= Int.MAX, True)


def check_fold_instance[threads: Int, items: Int]() raises:
    comptime sequential = sequential_argmax_plan[items]()
    comptime balanced = balanced_argmax_plan[items]()
    comptime fold = make_argmax_fold_plan[threads, items, 2 * items + 1, sequential]()
    comptime alternate = make_argmax_fold_plan[threads, items, 2 * items + 1, balanced]()
    comptime assert check_argmax_fold_plan(fold), "argmax fold plan rejected"
    comptime assert check_argmax_fold_plan(alternate), "argmax fold plan rejected"
    comptime tile = threads * items
    var counts: Array[Int, 11] = [
        1, threads - 1, threads, threads + 1, tile - 1, tile, tile + 1,
        2 * tile - 1, 2 * tile, 2 * tile + 1, 3 * tile + 7,
    ]
    for count in counts:
        check_count[threads, items, 2 * items + 1, fold](count)
        check_count[threads, items, 2 * items + 1, alternate](count)
    check_endpoint[threads, items, 2 * items + 1, fold]()
    var malformed = make_argmax_fold_plan[threads, items, 2 * items + 1, sequential]()
    assert_equal(check_argmax_fold_plan(malformed), True)
    malformed.stride = 0
    assert_equal(check_argmax_fold_plan(malformed), False)
    malformed.stride = -tile
    assert_equal(check_argmax_fold_plan(malformed), False)
    malformed.stride = tile - 1
    assert_equal(check_argmax_fold_plan(malformed), False)
    malformed.stride = tile + 1
    assert_equal(check_argmax_fold_plan(malformed), False)
    malformed = make_argmax_fold_plan[threads, items, 2 * items + 1, sequential]()
    malformed.reduction.nodes[items - 1] = empty_node()
    assert_equal(check_argmax_fold_plan(malformed), False)


def main() raises:
    check_instance[32, 1]()
    check_instance[64, 3]()
    check_instance[128, 8]()
    check_instance[256, 16]()
    check_instance[32, 65]()
    check_fold_instance[32, 1]()
    check_fold_instance[64, 3]()
    check_fold_instance[128, 8]()
    check_fold_instance[256, 16]()
    check_fold_instance[32, 65]()
    comptime assert not check_argmax_schedule[0, 8]()
    comptime assert not check_argmax_schedule[33, 8]()
    comptime assert not check_argmax_schedule[128, 0]()
    comptime assert not check_argmax_schedule[128, -1]()
    comptime assert not check_argmax_schedule[32, 2147483647]()
    print("Parameterized tile and runtime argmax fold ownership/oracle checks passed")
