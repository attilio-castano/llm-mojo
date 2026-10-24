"""Exact host ownership/index oracles, separate from GPU numerical qualification.

Tests model completion by committing full blocks before publication. The engine
tests separately check clearing pool metadata for returned eviction/free IDs.
"""
from std.math import ceildiv
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from llm_mojo.serving.blocks import BlockManager, COMPLETE, PARTIAL, REGISTERED
from llm_mojo.serving.prefix_cache import PrefixCache


def _tokens(count: Int, first: Int = 0) -> List[Int]:
    var result = List[Int](capacity=count)
    for i in range(count):
        result.append(first + i)
    return result^


def _assert_list(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        assert_equal(actual[i], expected[i])


def _snapshot(manager: BlockManager, cache: PrefixCache) -> String:
    var result = String("free")
    for block in manager.free:
        result += " " + String(block)
    for block in range(manager.blocks):
        result += " | " + String(manager.states[block]) + ":" + String(manager.refs[block])
    for s in range(len(manager.active)):
        result += " | seq " + String(s) + ":" + String(manager.active[s])
        result += ":" + String(manager.lengths[s]) + "/" + String(manager.reserved[s])
        for block in manager.tables[s]:
            result += " " + String(block)
    result += " | clock " + String(cache.clock) + " event " + String(cache.event_id)
    for entry in cache.entries:
        result += " | entry " + entry.namespace + ":" + entry.salt
        result += ":" + String(entry.block) + ":" + String(entry.last_used)
        for token in entry.tokens:
            result += " " + String(token)
    for event in cache.events:
        result += " | event " + String(event.event_id) + ":" + event.kind
        result += ":" + event.namespace + ":" + event.salt
        for token in event.tokens:
            result += " " + String(token)
    return result


def _producer(mut manager: BlockManager, length: Int) raises -> Int:
    var sequence = manager.add()
    manager.reserve(sequence, length)
    manager.commit(sequence, length)
    return sequence


def test_exact_history_namespace_salt_and_last_prompt_row_bound() raises:
    var manager = BlockManager(8, 4, 32)
    var cache = PrefixCache(8, 4, 32)
    var tokens = _tokens(9)
    var sequence = _producer(manager, 8)
    _assert_list(cache.publish(manager, sequence, tokens, "model/reference", "tenant-a"), [])
    _assert_list(cache.match(manager, tokens, "model/reference", "tenant-a"), [0, 1])
    _assert_list(cache.match(manager, _tokens(8), "model/reference", "tenant-a"), [0])
    _assert_list(cache.match(manager, _tokens(4), "model/reference", "tenant-a"), [])
    _assert_list(cache.match(manager, tokens, "other-model", "tenant-a"), [])
    _assert_list(cache.match(manager, tokens, "model/reference", "tenant-b"), [])
    _assert_list(cache.match(manager, tokens, "model/reference", "tenant-a", maximum=3), [])
    _assert_list(cache.match(manager, tokens, "model/reference", "tenant-a", maximum=4), [0])
    var changed = tokens.copy()
    changed[5] = 99
    _assert_list(cache.match(manager, changed, "model/reference", "tenant-a"), [0])
    changed = tokens.copy()
    changed[0] = 99
    _assert_list(cache.match(manager, changed, "model/reference", "tenant-a"), [])
    cache.check(manager)
    manager.check()


def test_duplicate_publication_keeps_private_blocks_without_live_remapping() raises:
    var manager = BlockManager(6, 4, 24)
    var cache = PrefixCache(6, 4, 24)
    var tokens = _tokens(9)
    var original = _producer(manager, 8)
    _ = cache.publish(manager, original, tokens, "model", "")
    _ = cache.take_events()
    var duplicate = _producer(manager, 8)
    var original_table = manager.table(duplicate)
    _assert_list(original_table, [2, 3])
    _assert_list(cache.publish(manager, duplicate, tokens, "model", ""), [])
    _assert_list(manager.table(duplicate), original_table)
    assert_equal(manager.states[2], COMPLETE)
    assert_equal(manager.states[3], COMPLETE)
    assert_equal(len(cache.entries), 2)
    assert_equal(len(cache.events), 0)
    _assert_list(manager.detach(duplicate), [3, 2])
    _assert_list(cache.match(manager, tokens, "model", ""), [0, 1])
    cache.check(manager)
    manager.check()


def test_admission_protects_matched_blocks_and_returns_reused_eviction_ids() raises:
    var manager = BlockManager(4, 4, 16)
    var cache = PrefixCache(4, 4, 16)
    var tokens = _tokens(9)
    var original = _producer(manager, 8)
    _ = cache.publish(manager, original, tokens, "model", "")
    _ = manager.detach(original)
    var unrelated = _producer(manager, 8)
    _ = cache.publish(manager, unrelated, _tokens(9, 100), "model", "")
    _ = manager.detach(unrelated)
    assert_equal(manager.free_blocks(), 0)
    var matched = cache.match(manager, tokens, "model", "")
    var admitted = cache.admit(manager, matched, 12)
    assert_equal(len(admitted.cleared), 1)
    assert_equal(manager.length(admitted.sequence), 8)
    assert_equal(manager.reserved[admitted.sequence], 12)
    var table = manager.table(admitted.sequence)
    assert_equal(table[0], 0)
    assert_equal(table[1], 1)
    assert_equal(table[2], admitted.cleared[0])
    assert_equal(manager.states[table[2]], PARTIAL)
    assert_equal(manager.refs[0], 1)
    assert_equal(manager.refs[1], 1)
    cache.check(manager)
    manager.check()


def test_idle_lru_and_protected_matches_do_not_evict_live_owners() raises:
    var manager = BlockManager(4, 4, 16)
    var cache = PrefixCache(4, 4, 16)
    var first_tokens = _tokens(5)
    var first = _producer(manager, 4)
    _ = cache.publish(manager, first, first_tokens, "model", "")
    _ = manager.detach(first)
    var second = _producer(manager, 4)
    _ = cache.publish(manager, second, _tokens(5, 100), "model", "")
    _ = manager.detach(second)
    var matched = cache.match(manager, first_tokens, "model", "")
    var reuse = cache.admit(manager, matched, 8)
    _ = manager.detach(reuse.sequence)
    _assert_list(cache.reclaim(manager, 3, []), [1])
    var before = _snapshot(manager, cache)
    with assert_raises():
        _ = cache.reclaim(manager, 4, matched)
    assert_equal(_snapshot(manager, cache), before)
    _assert_list(cache.reclaim(manager, 4, []), [0])
    assert_equal(len(cache.entries), 0)
    manager.check()
    cache.check(manager)


def test_rejected_admission_and_reclaim_are_atomic() raises:
    var manager = BlockManager(3, 4, 20)
    var cache = PrefixCache(3, 4, 20)
    var original = manager.add()
    manager.reserve(original, 8)
    manager.commit(original, 4)
    _ = cache.publish(manager, original, _tokens(5), "model", "")
    var blocker = _producer(manager, 4)
    var matched = cache.match(manager, _tokens(5), "model", "")
    var before = _snapshot(manager, cache)
    with assert_raises():
        _ = cache.admit(manager, matched, 12)
    with assert_raises():
        _ = cache.admit(manager, matched, 3)
    with assert_raises():
        _ = cache.admit(manager, matched, 8, watermark=-1)
    with assert_raises():
        _ = cache.admit(manager, [0, 0], 8)
    with assert_raises():
        _ = cache.admit(manager, [3], 8)
    with assert_raises():
        _ = cache.reclaim(manager, 1, [])
    assert_equal(_snapshot(manager, cache), before)
    manager.check()
    cache.check(manager)
    assert_true(manager.active[blocker])


def test_missing_ancestor_stops_lookup_and_reused_blocks_have_no_stale_key() raises:
    var manager = BlockManager(4, 4, 16)
    var cache = PrefixCache(4, 4, 16)
    var tokens = _tokens(9)
    var original = _producer(manager, 8)
    _ = cache.publish(manager, original, tokens, "model", "")
    _ = manager.detach(original)
    _assert_list(cache.reclaim(manager, 3, []), [0])
    assert_equal(len(cache.entries), 1)
    _assert_list(cache.match(manager, tokens, "model", ""), [])
    var replacement = _producer(manager, 4)
    _assert_list(manager.table(replacement), [0])
    _ = cache.publish(manager, replacement, _tokens(5, 100), "model", "")
    _ = manager.detach(replacement)
    _assert_list(cache.match(manager, tokens, "model", ""), [])
    _assert_list(cache.match(manager, _tokens(5, 100), "model", ""), [0])
    manager.check()
    cache.check(manager)


def test_admission_rejects_reordered_or_mixed_prefix_tables_before_eviction() raises:
    var manager = BlockManager(6, 4, 24)
    var cache = PrefixCache(6, 4, 24)
    var first = _producer(manager, 8)
    _ = cache.publish(manager, first, _tokens(9), "model", "first")
    _ = manager.detach(first)
    var second = _producer(manager, 8)
    _ = cache.publish(manager, second, _tokens(9, 100), "model", "second")
    _ = manager.detach(second)
    var before = _snapshot(manager, cache)
    with assert_raises():
        _ = cache.admit(manager, [1, 0], 12)
    with assert_raises():
        _ = cache.admit(manager, [0, 3], 12)
    assert_equal(_snapshot(manager, cache), before)
    # Even equal namespace/salt cannot join blocks from different histories.
    cache.entries[3].salt = "first"
    before = _snapshot(manager, cache)
    with assert_raises():
        _ = cache.admit(manager, [0, 3], 12)
    assert_equal(_snapshot(manager, cache), before)
    cache.check(manager)
    manager.check()


def test_full_index_with_only_referenced_entries_skips_publication() raises:
    var manager = BlockManager(4, 4, 16)
    var cache = PrefixCache(1, 4, 16)
    var original = _producer(manager, 8)
    _assert_list(cache.publish(manager, original, _tokens(9), "model", ""), [])
    assert_equal(len(cache.entries), 1)
    assert_equal(manager.states[0], REGISTERED)
    assert_equal(manager.states[1], COMPLETE)
    _assert_list(manager.detach(original), [1])
    var replacement = _producer(manager, 4)
    _assert_list(cache.publish(manager, replacement, _tokens(5, 100), "model", ""), [0])
    assert_equal(len(cache.entries), 1)
    manager.check()
    cache.check(manager)


def test_clear_is_atomic_with_readers_and_events_have_exact_logical_keys() raises:
    var manager = BlockManager(4, 4, 16)
    var cache = PrefixCache(4, 4, 16)
    var tokens = _tokens(9)
    var original = _producer(manager, 8)
    _ = cache.publish(manager, original, tokens, "model", "tenant")
    var stored = cache.take_events()
    assert_equal(len(stored), 2)
    assert_equal(stored[0].event_id, 0)
    assert_equal(stored[1].event_id, 1)
    assert_equal(stored[0].kind, "stored")
    assert_equal(stored[0].namespace, "model")
    assert_equal(stored[0].salt, "tenant")
    _assert_list(stored[0].tokens, _tokens(4))
    _assert_list(stored[1].tokens, _tokens(8))
    assert_equal(len(cache.events), 0)
    var before = _snapshot(manager, cache)
    with assert_raises():
        _ = cache.clear(manager)
    assert_equal(_snapshot(manager, cache), before)
    _ = manager.detach(original)
    _assert_list(cache.clear(manager), [0, 1])
    var removed = cache.take_events()
    assert_equal(len(removed), 3)
    for index in range(3):
        assert_equal(removed[index].event_id, index + 2)
    assert_equal(removed[0].kind, "removed")
    _assert_list(removed[0].tokens, _tokens(4))
    assert_equal(removed[1].kind, "removed")
    _assert_list(removed[1].tokens, _tokens(8))
    assert_equal(removed[2].kind, "cleared")
    assert_equal(len(removed[2].tokens), 0)
    assert_equal(manager.free_blocks(), 4)
    cache.check(manager)
    manager.check()


@fieldwise_init
struct KnownPrefix(Copyable, Movable):
    var tokens: List[Int]
    var salt: String


def _same(left: List[Int], right: List[Int]) -> Bool:
    if len(left) != len(right):
        return False
    for i in range(len(left)):
        if left[i] != right[i]:
            return False
    return True


def _consume_events(mut cache: PrefixCache, mut known: List[KnownPrefix]) raises:
    # Independent logical oracle: no block IDs, refcounts, or index slots.
    var events = cache.take_events()
    for event in events:
        var existing = -1
        for i in range(len(known)):
            if event.salt == known[i].salt and _same(event.tokens, known[i].tokens):
                existing = i
        if event.kind == "stored":
            assert_equal(existing, -1)
            known.append(KnownPrefix(event.tokens.copy(), event.salt.copy()))
        elif event.kind == "removed":
            assert_true(existing >= 0)
            _ = known.pop(existing)
        else:
            assert_equal(event.kind, "cleared")
            assert_equal(len(known), 0)
    assert_equal(len(known), len(cache.entries))


def _known_extent(known: List[KnownPrefix], tokens: List[Int], salt: String, size: Int) -> Int:
    var result = 0
    for end in range(size, len(tokens), size):
        var found = False
        for entry in known:
            if entry.salt != salt or len(entry.tokens) != end:
                continue
            var same = True
            for i in range(end):
                if entry.tokens[i] != tokens[i]:
                    same = False
            found = found or same
        if not found:
            break
        result = end
    return result


def test_seeded_interleaving_matches_logical_index_and_capacity_oracles() raises:
    var admitted_count = 0
    var rejected_count = 0
    var shared_count = 0
    for seed in [7, 29, 113]:
        var state = UInt64(seed)
        var manager = BlockManager(8, 4, 32, seed)
        var cache = PrefixCache(8, 4, 32)
        var live = List[Int]()
        var known = List[KnownPrefix]()
        for _ in range(500):
            state = state * 6364136223846793005 + 1442695040888963407
            var random = Int((state >> 33) % 100000)
            if len(live) > 0 and (len(live) >= 3 or random % 4 == 0):
                var selected = random % len(live)
                _ = manager.detach(live[selected])
                _ = live.pop(selected)
            else:
                var prompt = 1 + random % 25
                var tokens = _tokens(prompt + 1, (random // 31 % 3) * 100)
                var salt = String("a" if random % 2 == 0 else "b")
                # The request's last known token supplies logits, independently
                # of whether its prefix ends at an aligned full-block boundary.
                var request = List[Int](capacity=prompt)
                for i in range(prompt):
                    request.append(tokens[i])
                var matched = cache.match(manager, request, "model", salt)
                assert_equal(len(matched) * 4, _known_extent(known, request, salt, 4))
                var room = manager.free_blocks()
                for block in range(manager.blocks):
                    if manager.states[block] != REGISTERED or manager.refs[block] != 0:
                        continue
                    var protected = False
                    for borrowed in matched:
                        protected = protected or borrowed == block
                    if not protected:
                        room += 1
                var watermark = random % 2 if len(live) > 0 else 0
                var needed = ceildiv(prompt, 4) - len(matched) + watermark
                if needed > room:
                    var before = _snapshot(manager, cache)
                    with assert_raises():
                        _ = cache.admit(manager, matched, prompt, watermark)
                    assert_equal(_snapshot(manager, cache), before)
                    rejected_count += 1
                else:
                    var admission = cache.admit(manager, matched, prompt, watermark)
                    for block in matched:
                        shared_count += 1 if manager.refs[block] > 1 else 0
                    manager.commit(admission.sequence, prompt)
                    _ = cache.publish(manager, admission.sequence, tokens, "model", salt)
                    live.append(admission.sequence)
                    admitted_count += 1
            _consume_events(cache, known)
            manager.check()
            cache.check(manager)
        for sequence in live:
            _ = manager.detach(sequence)
        _ = cache.clear(manager)
        _consume_events(cache, known)
        assert_equal(manager.free_blocks(), 8)
        manager.check()
        cache.check(manager)
    assert_true(admitted_count > 100)
    assert_true(rejected_count > 100)
    assert_true(shared_count > 10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
