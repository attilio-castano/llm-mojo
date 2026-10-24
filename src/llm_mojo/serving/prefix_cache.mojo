"""Bounded exact prefix index over immutable full blocks.

Each entry names one physical block by the entire token history through its
boundary, a qualified model namespace, and a request salt. Matching requires
every preceding block to remain indexed and resident. No hashes, disk state,
partial-block sharing, or outstanding GPU pins are involved. The synchronous
caller establishes completion before publication and clears KV written counts
only for the block IDs returned by eviction or admission.
"""
from std.math import ceildiv
from llm_mojo.serving.blocks import BlockManager, COMPLETE, REGISTERED


@fieldwise_init
struct PrefixEntry(Copyable, Movable):
    var namespace: String
    var salt: String
    var tokens: List[Int]
    var block: Int
    var last_used: Int


@fieldwise_init
struct PrefixCacheEvent(Copyable, Movable):
    """Logical index change; physical block IDs are deliberately absent."""
    var event_id: Int
    var kind: String
    var namespace: String
    var salt: String
    var tokens: List[Int]


@fieldwise_init
struct PrefixAdmission(Movable):
    var sequence: Int
    var cleared: List[Int]


struct PrefixCache(Movable):
    var capacity: Int
    var block_size: Int
    var max_length: Int
    var entries: List[PrefixEntry]
    var clock: Int
    var events: List[PrefixCacheEvent]
    var event_id: Int

    def __init__(out self, capacity: Int, block_size: Int, max_length: Int = 4096) raises:
        if capacity < 0 or block_size < 1 or block_size > 4096 or max_length < 1:
            raise Error("invalid bounded prefix-cache geometry")
        self.capacity = capacity
        self.block_size = block_size
        self.max_length = max_length
        self.entries = List[PrefixEntry](capacity=capacity)
        self.clock = 0
        self.events = List[PrefixCacheEvent]()
        self.event_id = 0

    def _geometry(self, manager: BlockManager) raises:
        if (self.capacity > manager.blocks or self.block_size != manager.block_size
                or self.max_length != manager.max_length):
            raise Error("prefix cache and block manager disagree about geometry")

    def _event_room(self, count: Int) raises:
        if count < 0 or self.event_id > Int.MAX - count:
            raise Error("prefix-cache event identities exhausted")

    def _emit(mut self, kind: String, namespace: String, salt: String, tokens: List[Int]):
        self.events.append(PrefixCacheEvent(self.event_id, kind.copy(), namespace.copy(),
                                            salt.copy(), tokens.copy()))
        self.event_id += 1

    def _tick(mut self) -> Int:
        # Saturation preserves deterministic LRU tie-breaking without wrapping.
        self.clock = min(self.clock, Int.MAX - 1) + 1
        return self.clock

    def _find(self, tokens: List[Int], length: Int, namespace: String, salt: String) -> Int:
        for index in range(len(self.entries)):
            if (self.entries[index].namespace != namespace or self.entries[index].salt != salt
                    or len(self.entries[index].tokens) != length):
                continue
            var same = True
            for i in range(length):
                if self.entries[index].tokens[i] != tokens[i]:
                    same = False
                    break
            if same:
                return index
        return -1

    def _block_entry(self, block: Int) -> Int:
        for index in range(len(self.entries)):
            if self.entries[index].block == block:
                return index
        return -1

    def _protected(self, block: Int, protected: List[Int]) -> Bool:
        for candidate in protected:
            if block == candidate:
                return True
        return False

    def _validate_protected(self, manager: BlockManager, protected: List[Int]) raises:
        for block in protected:
            if block < 0 or block >= manager.blocks:
                raise Error("protected prefix-cache block is outside the pool")

    def _victim(self, manager: BlockManager, protected: List[Int]) -> Int:
        var victim = -1
        for index in range(len(self.entries)):
            var block = self.entries[index].block
            if manager.refs[block] != 0 or self._protected(block, protected):
                continue
            if victim < 0 or self.entries[index].last_used < self.entries[victim].last_used:
                victim = index
        return victim

    def _remove(mut self, mut manager: BlockManager, index: Int) raises -> Int:
        var block = self.entries[index].block
        var namespace = self.entries[index].namespace.copy()
        var salt = self.entries[index].salt.copy()
        var tokens = self.entries[index].tokens.copy()
        manager.evict_registered(block)
        self._emit("removed", namespace, salt, tokens)
        _ = self.entries.pop(index)
        return block

    def match(self, manager: BlockManager, tokens: List[Int], namespace: String, salt: String,
              maximum: Int = -1) raises -> List[Int]:
        """Find contiguous resident full blocks, always leaving the last token to run."""
        self._geometry(manager)
        if maximum < -1 or namespace.byte_length() == 0:
            raise Error("prefix matching needs a qualified namespace and valid limit")
        var limit = min(len(tokens) - 1, self.max_length)
        if maximum >= 0:
            limit = min(limit, maximum)
        var result = List[Int](capacity=max(limit // self.block_size, 0))
        for length in range(self.block_size, limit + 1, self.block_size):
            var index = self._find(tokens, length, namespace, salt)
            if index < 0:
                break
            var block = self.entries[index].block
            if block < 0 or block >= manager.blocks or manager.states[block] != REGISTERED:
                break
            # Repeated physical blocks cannot represent different positions.
            if self._protected(block, result):
                break
            result.append(block)
        return result^

    def reclaimable(self, manager: BlockManager, protected: List[Int]) raises -> Int:
        self._geometry(manager)
        self._validate_protected(manager, protected)
        var count = 0
        for entry in self.entries:
            if manager.refs[entry.block] == 0 and not self._protected(entry.block, protected):
                count += 1
        return count

    def reclaim(mut self, mut manager: BlockManager, required_free: Int,
                protected: List[Int]) raises -> List[Int]:
        """Evict idle LRU entries until `required_free` blocks are free, or reject atomically."""
        var available = self.reclaimable(manager, protected)
        if required_free < 0 or required_free > manager.free_blocks() + available:
            raise Error("not enough reclaimable idle prefix-cache blocks")
        var count = max(required_free - manager.free_blocks(), 0)
        self._event_room(count)
        var cleared = List[Int](capacity=count)
        for _ in range(count):
            var victim = self._victim(manager, protected)
            cleared.append(self._remove(manager, victim))
        return cleared^

    def admit(mut self, mut manager: BlockManager, cached: List[Int], reserve_length: Int,
              watermark: Int = 0) raises -> PrefixAdmission:
        """Prevalidate attachment, idle eviction and private reservation as one admission."""
        self._geometry(manager)
        var cached_length = len(cached) * self.block_size
        if (watermark < 0 or watermark > manager.blocks or reserve_length < cached_length
                or reserve_length > self.max_length):
            raise Error("invalid cached admission reservation or watermark")
        for i in range(len(cached)):
            var block = cached[i]
            if (block < 0 or block >= manager.blocks or manager.states[block] != REGISTERED
                    or self._block_entry(block) < 0):
                raise Error("cached admission refers to a missing Registered block")
            for previous in range(i):
                if cached[previous] == block:
                    raise Error("cached admission repeats a physical block")
            var entry = self._block_entry(block)
            if len(self.entries[entry].tokens) != (i + 1) * self.block_size:
                raise Error("cached admission is not an ordered prefix table")
            if i > 0:
                var previous = self._block_entry(cached[i - 1])
                if (self.entries[entry].namespace != self.entries[previous].namespace
                        or self.entries[entry].salt != self.entries[previous].salt):
                    raise Error("cached admission mixes prefix identities")
                for token in range(i * self.block_size):
                    if self.entries[entry].tokens[token] != self.entries[previous].tokens[token]:
                        raise Error("cached admission mixes different token histories")
        var needed = ceildiv(reserve_length, self.block_size) - len(cached)
        # Protected matches cannot be counted as room for their own private tail.
        var cleared = self.reclaim(manager, needed + watermark, cached)
        var sequence = manager.attach(cached)
        manager.reserve(sequence, reserve_length)
        var stamp = self._tick()
        for block in cached:
            self.entries[self._block_entry(block)].last_used = stamp
        return PrefixAdmission(sequence, cleared^)

    def publish(mut self, mut manager: BlockManager, sequence: Int, tokens: List[Int],
                namespace: String, salt: String) raises -> List[Int]:
        """Register healthy completed full blocks; an existing exact key stays canonical."""
        self._geometry(manager)
        var length = manager.length(sequence)
        if namespace.byte_length() == 0 or len(tokens) < length:
            raise Error("publication needs exact known history and a qualified namespace")
        var full = length // self.block_size
        # Validate every candidate before changing the index or block ownership.
        for ordinal in range(full):
            var block = manager.tables[sequence][ordinal]
            if manager.states[block] == REGISTERED:
                var existing = self._find(tokens, (ordinal + 1) * self.block_size, namespace, salt)
                if existing < 0 or self.entries[existing].block != block:
                    raise Error("Registered history disagrees with the publication key")
            elif manager.states[block] != COMPLETE or manager.refs[block] != 1:
                raise Error("publication needs full private Complete blocks")
        self._event_room(2 * full)
        var cleared = List[Int](capacity=full)
        for ordinal in range(full):
            var end = (ordinal + 1) * self.block_size
            if self._find(tokens, end, namespace, salt) >= 0:
                # Never remap or register a duplicate private block in a live table.
                continue
            if self.capacity == 0:
                continue
            if len(self.entries) == self.capacity:
                var victim = self._victim(manager, List[Int]())
                if victim < 0:
                    continue
                cleared.append(self._remove(manager, victim))
            var key = List[Int](capacity=end)
            for i in range(end):
                key.append(tokens[i])
            manager.register(sequence, ordinal)
            var block = manager.tables[sequence][ordinal]
            var stamp = self._tick()
            var event_key = key.copy()
            self.entries.append(PrefixEntry(namespace.copy(), salt.copy(), key^, block, stamp))
            self._emit("stored", namespace, salt, event_key)
        return cleared^

    def clear(mut self, mut manager: BlockManager) raises -> List[Int]:
        """Drop the index only when none of its blocks has an active reader."""
        self._geometry(manager)
        for entry in self.entries:
            if manager.states[entry.block] != REGISTERED or manager.refs[entry.block] != 0:
                raise Error("cannot clear a prefix cache with active readers")
        self._event_room(len(self.entries) + 1)
        var cleared = List[Int](capacity=len(self.entries))
        while len(self.entries) > 0:
            cleared.append(self._remove(manager, 0))
        self._emit("cleared", "", "", List[Int]())
        return cleared^

    def take_events(mut self) -> List[PrefixCacheEvent]:
        var result = self.events.copy()
        self.events.clear()
        return result^

    def check(self, manager: BlockManager) raises:
        self._geometry(manager)
        if len(self.entries) > self.capacity or self.clock < 0 or self.event_id < 0:
            raise Error("prefix-cache bounded state is invalid")
        var indexed = List[Int](length=manager.blocks, fill=0)
        for index in range(len(self.entries)):
            var block = self.entries[index].block
            var length = len(self.entries[index].tokens)
            if (block < 0 or block >= manager.blocks or manager.states[block] != REGISTERED
                    or self.entries[index].namespace.byte_length() == 0 or length < self.block_size
                    or length > self.max_length or length % self.block_size != 0
                    or self.entries[index].last_used < 0 or self.entries[index].last_used > self.clock):
                raise Error("prefix-cache entry is not a resident exact full-prefix key")
            indexed[block] += 1
            for previous in range(index):
                if (len(self.entries[previous].tokens) == length
                        and self.entries[previous].namespace == self.entries[index].namespace
                        and self.entries[previous].salt == self.entries[index].salt):
                    var same = True
                    for i in range(length):
                        if self.entries[previous].tokens[i] != self.entries[index].tokens[i]:
                            same = False
                            break
                    if same:
                        raise Error("prefix cache repeats an exact key")
        for block in range(manager.blocks):
            if indexed[block] != (1 if manager.states[block] == REGISTERED else 0):
                raise Error("Registered blocks and prefix index entries disagree")
