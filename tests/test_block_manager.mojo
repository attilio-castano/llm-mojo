"""Block manager: allocation at block boundaries, block states and invariants, on the host.

Seeded random operations over pools of 1 to 512 blocks and block sizes from 32
to 4,096 check every invariant after every operation, with scattered and
in-order free lists. Every kind of rejection leaves the manager unchanged.
"""
from std.math import ceildiv
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from llm_mojo.serving.blocks import BlockManager, COMPLETE, PARTIAL, RESET


def _assert_list(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        assert_equal(actual[i], expected[i])


def _state(manager: BlockManager) -> String:
    """Everything a rejected operation must leave unchanged."""
    var text = String("free")
    for block in manager.free:
        text += " " + String(block)
    text += " | states"
    for state in manager.states:
        text += " " + String(state)
    for s in range(len(manager.active)):
        text += " | " + String(s) + (" live " if manager.active[s] else " released ")
        text += String(manager.lengths[s]) + "/" + String(manager.reserved[s]) + ":"
        for block in manager.tables[s]:
            text += " " + String(block)
    return text


struct Random(Movable):
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed) * 2654435761 + 7

    def below(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def _exercise(blocks: Int, size: Int, max_length: Int, seed: Int, steps: Int, mut tally: List[Int]) raises:
    """Random adds, reservations, commits, truncations and releases, with the invariants checked after each.

    tally counts adds, allocating reservations, rejected reservations, commits,
    truncations, releases, reserved and committed steps, and other rejections.
    """
    var manager = BlockManager(blocks, size, max_length, seed)
    manager.check()
    var random = Random(seed)
    var live = List[Int]()
    for _ in range(steps):
        var before = _state(manager)
        var choice = random.below(8)
        if choice == 0 or len(live) == 0:
            live.append(manager.add())
            tally[0] += 1
            manager.check()
            continue
        var pick = random.below(len(live))
        var s = live[pick]
        var length = manager.length(s)
        var held = len(manager.table(s))
        var free = manager.free_blocks()
        if choice <= 2:
            # A few positions or a few blocks more, sometimes past the pool or the limit.
            var target = length + random.below(3 * size + 2)
            var needed = max(ceildiv(target, size) - held, 0)
            if target <= max_length and needed <= free:
                manager.reserve(s, target)
                assert_equal(len(manager.table(s)), held + needed)
                assert_equal(manager.free_blocks(), free - needed)
                if needed > 0:
                    tally[1] += 1
            else:
                with assert_raises():
                    manager.reserve(s, target)
                assert_equal(_state(manager), before)
                tally[2] += 1
        elif choice == 3:
            var target = length + random.below(manager.reserved[s] - length + 1)
            manager.commit(s, target)
            assert_equal(manager.length(s), target)
            tally[3] += 1
        elif choice == 4:
            var target = random.below(length + 1)
            manager.truncate(s, target)
            assert_equal(len(manager.table(s)), ceildiv(target, size))
            assert_equal(manager.free_blocks(), free + held - ceildiv(target, size))
            tally[4] += 1
        elif choice == 5:
            manager.release(s)
            _ = live.pop(pick)
            assert_equal(manager.free_blocks(), free + held)
            tally[5] += 1
        elif choice == 6:
            # One step: room for a chunk, then its commit, as a client does around a forward.
            var target = min(length + 1 + random.below(2 * size), max_length)
            if target > length and max(ceildiv(target, size) - held, 0) <= free:
                manager.reserve(s, target)
                manager.commit(s, target)
                assert_equal(manager.length(s), target)
                tally[6] += 1
        else:
            # Commits past the reservation and truncations past the length change nothing.
            with assert_raises():
                manager.commit(s, manager.reserved[s] + 1)
            with assert_raises():
                manager.truncate(s, length + 1)
            assert_equal(_state(manager), before)
            tally[7] += 1
        manager.check()
    # Releasing every sequence leaves every block free and Reset.
    manager.reset()
    manager.check()
    assert_equal(manager.free_blocks(), blocks)
    for state in manager.states:
        assert_equal(state, RESET)


def test_random_operations_keep_every_invariant() raises:
    var tally = List[Int](capacity=8)
    for _ in range(8):
        tally.append(0)
    for size in [32, 64, 128, 4096]:
        for blocks in [1, 2, 7, 64, 512]:
            _exercise(blocks, size, 4096, blocks * 31 + size, 2000, tally)
            _exercise(blocks, size, 4096, 0, 500, tally)
    # A per-sequence limit that is not a whole number of blocks.
    _exercise(40, 32, 100, 5, 2000, tally)
    # Every kind of operation happened often, so the invariants were checked after each.
    for count in tally:
        assert_true(count >= 500)


def test_blocks_are_allocated_at_boundaries_and_released_last_first() raises:
    var manager = BlockManager(8, 32, 4096)
    var a = manager.add()
    var b = manager.add()
    manager.reserve(a, 1)
    _assert_list(manager.table(a), [0])
    assert_equal(manager.states[0], PARTIAL)
    manager.commit(a, 1)
    # A block's 32nd position needs no new block; the 33rd does.
    manager.reserve(a, 32)
    _assert_list(manager.table(a), [0])
    manager.commit(a, 32)
    assert_equal(manager.states[0], COMPLETE)
    manager.reserve(a, 33)
    _assert_list(manager.table(a), [0, 1])
    assert_equal(manager.states[1], PARTIAL)
    # Sequences growing in turn interleave their blocks.
    manager.reserve(b, 70)
    manager.commit(b, 70)
    _assert_list(manager.table(b), [2, 3, 4])
    assert_equal(manager.states[2], COMPLETE)
    assert_equal(manager.states[3], COMPLETE)
    assert_equal(manager.states[4], PARTIAL)
    manager.check()
    # Truncation inside a Complete block makes it Partial and frees the blocks after it.
    manager.truncate(b, 40)
    _assert_list(manager.table(b), [2, 3])
    assert_equal(manager.states[3], PARTIAL)
    assert_equal(manager.states[4], RESET)
    # Release frees the last block first, so the first block is allocated next.
    manager.release(b)
    var c = manager.add()
    assert_equal(c, b)
    manager.reserve(c, 1)
    _assert_list(manager.table(c), [2])
    manager.check()


def test_a_seed_scatters_the_allocation_order() raises:
    var manager = BlockManager(64, 32, 4096, 9)
    var s = manager.add()
    manager.reserve(s, 64 * 32)
    var table = manager.table(s)
    assert_equal(len(table), 64)
    var in_order = 0
    var seen = List[Bool](capacity=64)
    for _ in range(64):
        seen.append(False)
    for i in range(64):
        if table[i] == i:
            in_order += 1
        seen[table[i]] = True
    for i in range(64):
        assert_true(seen[i])
    assert_true(in_order < 8)
    manager.check()


def test_rejections_leave_the_manager_unchanged() raises:
    for blocks in [0, -1]:
        with assert_raises():
            _ = BlockManager(blocks, 32, 4096)
    for size in [0, 4097]:
        with assert_raises():
            _ = BlockManager(4, size, 4096)
    with assert_raises():
        _ = BlockManager(4, 32, 0)
    var manager = BlockManager(4, 32, 100)
    var a = manager.add()
    var b = manager.add()
    manager.reserve(a, 40)
    manager.commit(a, 40)
    manager.reserve(b, 64)
    var before = _state(manager)
    # More blocks than are free, a length past the per-sequence limit, and one below the length.
    with assert_raises():
        manager.reserve(a, 65)
    with assert_raises():
        manager.reserve(b, 101)
    with assert_raises():
        manager.reserve(a, 39)
    # Commits past the reservation or below the length.
    with assert_raises():
        manager.commit(a, 41)
    with assert_raises():
        manager.commit(b, 65)
    manager.commit(b, 10)
    with assert_raises():
        manager.commit(b, 9)
    manager.truncate(b, 0)
    manager.reserve(b, 64)
    assert_equal(_state(manager), before)
    # Truncations past the length or below zero.
    with assert_raises():
        manager.truncate(a, 41)
    with assert_raises():
        manager.truncate(a, -1)
    # Sequences never added, negative, or released.
    with assert_raises():
        manager.reserve(2, 1)
    with assert_raises():
        _ = manager.length(-1)
    assert_equal(_state(manager), before)
    manager.release(a)
    with assert_raises():
        manager.commit(a, 0)
    with assert_raises():
        manager.release(a)
    manager.check()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
