"""Which KV blocks each sequence holds: allocation, block states and invariants.

The manager owns a pool's blocks by number, not their bytes. A sequence's table
lists its blocks in position order: position t lives in block table[t // block_size]
at slot t % block_size. A step first reserves room for its new positions,
which allocates a block at each block boundary, and commits the new length once
its writes are enqueued. Truncation and release return whole blocks, last block
first. The manager submits no GPU work; the model checks each step against the
pool's written counts, which must agree with the lengths committed here.

Block states follow the serving plan, without Registered, which arrives with
shared blocks in phase 4:
- Reset: free, held by no sequence;
- Partial: held by one sequence and not yet full, including blocks reserved for
  a step that has not been committed;
- Complete: held by one sequence, every slot submitted.

Every rejected operation raises before it changes any state.
"""
from std.math import ceildiv
from llm_mojo.serving.kv_pool import KVPool

comptime RESET = 0
comptime PARTIAL = 1
comptime COMPLETE = 2


struct BlockManager(Movable):
    var blocks: Int
    var block_size: Int
    var max_length: Int
    # Free blocks; allocation takes the last one.
    var free: List[Int]
    var states: List[Int]
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
        for _ in range(blocks):
            self.states.append(RESET)
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
            self.tables[sequence].append(block)
        self.reserved[sequence] = max(self.reserved[sequence], length)

    def commit(mut self, sequence: Int, length: Int) raises:
        """Record that enqueued writes now cover the sequence's positions below `length`."""
        self._known(sequence)
        if length < self.lengths[sequence] or length > self.reserved[sequence]:
            raise Error("a committed length must lie between the sequence's length and its reservation")
        self.lengths[sequence] = length
        for i in range(length // self.block_size):
            self.states[self.tables[sequence][i]] = COMPLETE

    def truncate(mut self, sequence: Int, length: Int) raises:
        """Shorten a sequence to `length` positions, freeing whole blocks past it, last block first."""
        self._known(sequence)
        if length < 0 or length > self.lengths[sequence]:
            raise Error("a truncation must lie between zero and the sequence's length")
        var keep = ceildiv(length, self.block_size)
        while len(self.tables[sequence]) > keep:
            var block = self.tables[sequence].pop()
            self.states[block] = RESET
            self.free.append(block)
        if length % self.block_size != 0:
            self.states[self.tables[sequence][keep - 1]] = PARTIAL
        self.lengths[sequence] = length
        self.reserved[sequence] = length

    def release(mut self, sequence: Int) raises:
        """Free every block of a sequence, last block first, and forget the sequence."""
        self.truncate(sequence, 0)
        self.active[sequence] = False

    def reset(mut self) raises:
        """Release every sequence."""
        for s in range(len(self.active)):
            if self.active[s]:
                self.release(s)

    def check(self) raises:
        """Raise unless every invariant holds.

        Free and held blocks partition the pool: each block is either on the free
        list once and Reset, or in exactly one table entry of one sequence and
        Partial or Complete, so every reference count is 0 or 1. A sequence's
        table covers its reserved length, which lies between its length and the
        per-sequence limit; blocks wholly below its length are Complete and the
        others Partial. A released slot holds nothing.
        """
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
                if self.states[block] != expected:
                    raise Error("a held block's state disagrees with its sequence's length")
        for block in range(self.blocks):
            if free[block] + held[block] != 1:
                raise Error("block " + String(block) + " is not exactly free or held once")
            if free[block] == 1 and self.states[block] != RESET:
                raise Error("a free block is not Reset")

    def check_pool(self, pool: KVPool) raises:
        """Raise unless the pool's written slots agree with every committed length.

        Blocks wholly below a sequence's length are full, its next block holds
        the rest, and every other block, free or reserved, is empty. Holds
        between steps: during a step the model has advanced the pool before the
        step is committed here.
        """
        self.check()
        if pool.blocks != self.blocks or pool.block_size != self.block_size:
            raise Error("the pool and the block manager disagree about blocks")
        var expected = List[Int](capacity=self.blocks)
        for _ in range(self.blocks):
            expected.append(0)
        for s in range(len(self.active)):
            for i in range(len(self.tables[s])):
                expected[self.tables[s][i]] = min(max(self.lengths[s] - i * self.block_size, 0), self.block_size)
        for block in range(self.blocks):
            if pool.length(block) != expected[block]:
                raise Error("block " + String(block) + "'s written slots disagree with its sequence's length")
