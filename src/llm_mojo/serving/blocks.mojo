"""Which KV blocks each sequence holds: allocation, block states and invariants.

The manager owns a pool's blocks by number, not their bytes. A sequence's table
lists its blocks in position order: position t lives in block table[t // block_size]
at slot t % block_size. A step first reserves room for its new positions,
which allocates a block at each block boundary, and commits the new length once
its writes are enqueued. Truncation and release return whole blocks, last block
first. The manager submits no GPU work; the model checks each step against the
pool's written counts, which must agree with the lengths committed here.

Block states distinguish private writes from immutable cached history:
- Reset: free, held by no sequence;
- Partial: held by one sequence and not yet full, including blocks reserved for
  a step that has not been committed;
- Complete: held by one sequence, every slot submitted;
- Registered: full and retained by a prefix index, with any number of readers.

`refs` counts active sequence-table references, independently of cache retention.
The caller clears KV written counts only for private blocks actually returned to
the free list or idle Registered blocks explicitly evicted from the index.

Every rejected operation raises before it changes any state.
"""
from std.math import ceildiv
from llm_mojo.serving.kv_pool import KVPool

comptime RESET = 0
comptime PARTIAL = 1
comptime COMPLETE = 2
comptime REGISTERED = 3


struct BlockManager(Movable):
    var blocks: Int
    var block_size: Int
    var max_length: Int
    # Free blocks; allocation takes the last one.
    var free: List[Int]
    var states: List[Int]
    var refs: List[Int]
    # Per sequence slot: its blocks in position order, the positions whose
    # writes committed steps have enqueued, the positions its blocks were
    # reserved for, and whether the slot holds a sequence.
    var tables: List[List[Int]]
    var lengths: List[Int]
    var reserved: List[Int]
    var active: List[Bool]

    def __init__(out self, blocks: Int, block_size: Int, max_length: Int, seed: Int = 0) raises:
        """`blocks` free blocks. Seed 0 allocates block 0 first and the rest in order; any
        other seed allocates in a seeded permutation, as a busy pool's free list would."""
        if blocks < 1 or block_size < 1 or block_size > 4096 or max_length < 1:
            raise Error("invalid block manager geometry")
        self.blocks = blocks
        self.block_size = block_size
        self.max_length = max_length
        var order = List[Int](capacity=blocks)
        for block in range(blocks):
            order.append(block)
        if seed != 0:
            var state = UInt64(seed) * 2654435761 + 1
            for i in range(blocks - 1, 0, -1):
                state = state * 6364136223846793005 + 1442695040888963407
                var j = Int((state >> 33) % UInt64(i + 1))
                var swap = order[i]
                order[i] = order[j]
                order[j] = swap
        self.free = List[Int](capacity=blocks)
        for i in range(blocks - 1, -1, -1):
            self.free.append(order[i])
        self.states = List[Int](capacity=blocks)
        self.refs = List[Int](capacity=blocks)
        for _ in range(blocks):
            self.states.append(RESET)
            self.refs.append(0)
        self.tables = List[List[Int]]()
        self.lengths = List[Int]()
        self.reserved = List[Int]()
        self.active = List[Bool]()

    def add(mut self) -> Int:
        """A new empty sequence, in the first slot a released sequence left."""
        for s in range(len(self.active)):
            if not self.active[s]:
                self.active[s] = True
                return s
        self.tables.append(List[Int]())
        self.lengths.append(0)
        self.reserved.append(0)
        self.active.append(True)
        return len(self.active) - 1

    def _known(self, sequence: Int) raises:
        if sequence < 0 or sequence >= len(self.active) or not self.active[sequence]:
            raise Error("unknown sequence")

    def length(self, sequence: Int) raises -> Int:
        self._known(sequence)
        return self.lengths[sequence]

    def table(self, sequence: Int) raises -> List[Int]:
        self._known(sequence)
        return self.tables[sequence].copy()

    def free_blocks(self) -> Int:
        return len(self.free)

    def attach(mut self, cached: List[Int]) raises -> Int:
        """Borrow a complete immutable prefix, validating before creating its slot."""
        if len(cached) > self.max_length // self.block_size:
            raise Error("cached prefix exceeds the sequence limit")
        for i in range(len(cached)):
            var block = cached[i]
            if block < 0 or block >= self.blocks or self.states[block] != REGISTERED:
                raise Error("cached prefix must contain Registered blocks")
            for previous in range(i):
                if cached[previous] == block:
                    raise Error("cached prefix repeats a physical block")
        var sequence = self.add()
        for block in cached:
            self.tables[sequence].append(block)
            self.refs[block] += 1
        self.lengths[sequence] = len(cached) * self.block_size
        self.reserved[sequence] = self.lengths[sequence]
        return sequence

    def register(mut self, sequence: Int, ordinal: Int) raises:
        """Retain one full committed block; the caller establishes GPU completion."""
        self._known(sequence)
        if ordinal < 0 or ordinal >= self.lengths[sequence] // self.block_size:
            raise Error("only full committed blocks can be Registered")
        var block = self.tables[sequence][ordinal]
        if self.states[block] == REGISTERED:
            return
        if self.states[block] != COMPLETE or self.refs[block] != 1:
            raise Error("registration needs a private Complete block")
        self.states[block] = REGISTERED

    def evict_registered(mut self, block: Int) raises:
        """Return one idle cache block; its index entry and KV count retire together."""
        if (block < 0 or block >= self.blocks or self.states[block] != REGISTERED
                or self.refs[block] != 0):
            raise Error("only an idle Registered block can be evicted")
        self.states[block] = RESET
        self.free.append(block)

    def reserve(mut self, sequence: Int, length: Int) raises:
        """Hold blocks for `length` positions of a sequence, allocating at each block boundary."""
        self._known(sequence)
        if length < self.lengths[sequence] or length > self.max_length:
            raise Error("a reservation must lie between the sequence's length and the per-sequence limit")
        var needed = ceildiv(length, self.block_size) - len(self.tables[sequence])
        if needed > len(self.free):
            raise Error("not enough free KV blocks")
        for _ in range(needed):
            var block = self.free.pop()
            self.states[block] = PARTIAL
            self.refs[block] = 1
            self.tables[sequence].append(block)
        self.reserved[sequence] = max(self.reserved[sequence], length)

    def commit(mut self, sequence: Int, length: Int) raises:
        """Record that enqueued writes now cover the sequence's positions below `length`."""
        self._known(sequence)
        if length < self.lengths[sequence] or length > self.reserved[sequence]:
            raise Error("a committed length must lie between the sequence's length and its reservation")
        self.lengths[sequence] = length
        for i in range(length // self.block_size):
            var block = self.tables[sequence][i]
            if self.states[block] != REGISTERED:
                self.states[block] = COMPLETE

    def truncate(mut self, sequence: Int, length: Int) raises:
        """Shorten a sequence to `length` positions, freeing whole blocks past it, last block first."""
        self._known(sequence)
        if length < 0 or length > self.lengths[sequence]:
            raise Error("a truncation must lie between zero and the sequence's length")
        var keep = ceildiv(length, self.block_size)
        if length % self.block_size != 0 and self.states[self.tables[sequence][keep - 1]] == REGISTERED:
            raise Error("cannot truncate inside an immutable Registered block")
        while len(self.tables[sequence]) > keep:
            var block = self.tables[sequence].pop()
            self.refs[block] -= 1
            if self.states[block] != REGISTERED:
                self.states[block] = RESET
                self.free.append(block)
        if length % self.block_size != 0:
            self.states[self.tables[sequence][keep - 1]] = PARTIAL
        self.lengths[sequence] = length
        self.reserved[sequence] = length

    def release(mut self, sequence: Int) raises:
        """Drop all sequence references, retaining immutable cache blocks."""
        self.truncate(sequence, 0)
        self.active[sequence] = False

    def detach(mut self, sequence: Int) raises -> List[Int]:
        """Release a sequence and return only IDs whose KV counts must be cleared."""
        self._known(sequence)
        var freed = List[Int](capacity=len(self.tables[sequence]))
        for i in range(len(self.tables[sequence]) - 1, -1, -1):
            var block = self.tables[sequence][i]
            if self.states[block] != REGISTERED:
                freed.append(block)
        self.release(sequence)
        return freed^

    def reset(mut self) raises:
        """Release every sequence; the prefix index separately evicts idle cache."""
        for s in range(len(self.active)):
            if self.active[s]:
                self.release(s)

    def check(self) raises:
        """Raise unless every invariant holds.

        Reset blocks are free exactly once and unreferenced. Private Partial or
        Complete blocks have one owner. Registered blocks are never free and
        appear only in fully committed history; the prefix index retains them
        even with no active readers. Stored refs equal exact table occurrences.
        A released slot holds nothing.
        """
        if (len(self.states) != self.blocks or len(self.refs) != self.blocks
                or len(self.tables) != len(self.active) or len(self.lengths) != len(self.active)
                or len(self.reserved) != len(self.active)):
            raise Error("block-manager storage geometry disagrees")
        var free = List[Int](capacity=self.blocks)
        var held = List[Int](capacity=self.blocks)
        for _ in range(self.blocks):
            free.append(0)
            held.append(0)
        for block in self.free:
            if block < 0 or block >= self.blocks:
                raise Error("a free block is outside the pool")
            free[block] += 1
        for s in range(len(self.active)):
            if not self.active[s]:
                if len(self.tables[s]) != 0 or self.lengths[s] != 0 or self.reserved[s] != 0:
                    raise Error("a released sequence still holds blocks or positions")
                continue
            var length = self.lengths[s]
            var reserved = self.reserved[s]
            if length < 0 or reserved < length or reserved > self.max_length:
                raise Error("a sequence's length or reservation is out of range")
            if len(self.tables[s]) != ceildiv(reserved, self.block_size):
                raise Error("a sequence's table does not match its reservation")
            for i in range(len(self.tables[s])):
                var block = self.tables[s][i]
                if block < 0 or block >= self.blocks:
                    raise Error("a table entry is outside the pool")
                held[block] += 1
                var expected = COMPLETE if i < length // self.block_size else PARTIAL
                if self.states[block] == REGISTERED:
                    for previous in range(i):
                        if self.tables[s][previous] == block:
                            raise Error("one sequence repeats an immutable physical block")
                    if expected != COMPLETE:
                        raise Error("Registered block lies outside full committed history")
                elif self.states[block] != expected:
                    raise Error("a held block's state disagrees with its sequence's length")
        for block in range(self.blocks):
            if self.refs[block] != held[block] or self.refs[block] < 0:
                raise Error("stored block references disagree with sequence tables")
            if self.states[block] == RESET:
                if free[block] != 1 or held[block] != 0:
                    raise Error("a Reset block is not exactly free and unreferenced")
            elif self.states[block] == REGISTERED:
                if free[block] != 0:
                    raise Error("a Registered block appears on the free list")
            elif self.states[block] == PARTIAL or self.states[block] == COMPLETE:
                if free[block] != 0 or held[block] != 1:
                    raise Error("a private block does not have exactly one owner")
            else:
                raise Error("unknown block state")

    def check_pool(self, pool: KVPool) raises:
        """Raise unless the pool's written slots agree with every committed length.

        Blocks wholly below a sequence's length are full, its next block holds
        the rest, and every free or unwritten reserved block is empty. Idle
        Registered blocks remain full. Holds
        between steps: during a step the model has advanced the pool before the
        step is committed here.
        """
        self.check()
        if pool.blocks != self.blocks or pool.block_size != self.block_size:
            raise Error("the pool and the block manager disagree about blocks")
        var expected = List[Int](capacity=self.blocks)
        for block in range(self.blocks):
            expected.append(self.block_size if self.states[block] == REGISTERED else 0)
        for s in range(len(self.active)):
            for i in range(len(self.tables[s])):
                expected[self.tables[s][i]] = min(max(self.lengths[s] - i * self.block_size, 0), self.block_size)
        for block in range(self.blocks):
            if pool.length(block) != expected[block]:
                raise Error("block " + String(block) + "'s written slots disagree with its sequence's length")
