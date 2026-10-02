"""One engine step: which tokens run, at which positions, and where their KV goes.

A step concatenates the scheduled tokens of its sequences, one-token decode
sequences first. Each sequence's block table maps its positions to physical KV
blocks; write slots are the physical rows receiving each new token's K and V.
Only `logits_rows` need vocabulary logits. Validation is host-side and exact, so
a rejected batch changes no state and submits no GPU work.
"""


struct StepBatch(Movable):
    var token_ids: List[Int]
    var positions: List[Int]
    var query_start: List[Int]
    var decode_count: Int
    var seq_lens: List[Int]
    var max_blocks: Int
    var block_table: List[Int]
    var slot_mapping: List[Int]
    var logits_rows: List[Int]

    def __init__(
        out self,
        var token_ids: List[Int],
        var positions: List[Int],
        var query_start: List[Int],
        decode_count: Int,
        var seq_lens: List[Int],
        max_blocks: Int,
        var block_table: List[Int],
        var slot_mapping: List[Int],
        var logits_rows: List[Int],
    ):
        self.token_ids = token_ids^
        self.positions = positions^
        self.query_start = query_start^
        self.decode_count = decode_count
        self.seq_lens = seq_lens^
        self.max_blocks = max_blocks
        self.block_table = block_table^
        self.slot_mapping = slot_mapping^
        self.logits_rows = logits_rows^

    @staticmethod
    def sequence(ids: List[Int], past: Int, table: List[Int], block_size: Int) raises -> StepBatch:
        """One sequence writing `ids` at positions `past, past + 1, ...` in the blocks its table lists.

        Position p lives in block table[p // block_size]; a table of one entry
        holds the whole sequence in one block.
        """
        var rows = len(ids)
        var width = len(table)
        if rows < 1 or past < 0 or width < 1 or block_size < 1 or past + rows > width * block_size:
            raise Error("a sequence step must fit in its table")
        for block in table:
            if block < 0:
                raise Error("a sequence step must fit in its table")
        var positions = List[Int](capacity=rows)
        var slots = List[Int](capacity=rows)
        for i in range(rows):
            var position = past + i
            positions.append(position)
            slots.append(table[position // block_size] * block_size + position % block_size)
        return StepBatch(ids.copy(), positions^, [0, rows], 1 if rows == 1 else 0,
            [past + rows], width, table.copy(), slots^, [rows - 1])

    def rows(self) -> Int:
        return len(self.token_ids)

    def sequences(self) -> Int:
        return len(self.seq_lens)

    def validate(self, blocks: Int, block_size: Int, vocabulary: Int) raises:
        var n = len(self.token_ids)
        var s = len(self.seq_lens)
        if n < 1 or s < 1 or self.max_blocks < 1 or blocks < 1 or block_size < 1 or vocabulary < 1:
            raise Error("step batch is empty or its limits are invalid")
        if (len(self.query_start) != s + 1 or len(self.positions) != n
            or len(self.slot_mapping) != n or len(self.block_table) != s * self.max_blocks):
            raise Error("step batch fields disagree in length")
        if self.query_start[0] != 0 or self.query_start[s] != n:
            raise Error("step batch offsets must span every token")
        for id in self.token_ids:
            if id < 0 or id >= vocabulary:
                raise Error("step batch token ID out of range")
        for block in self.block_table:
            if block < 0 or block >= blocks:
                raise Error("step batch block ID out of range")
        var leading = 0
        for seq in range(s):
            var begin = self.query_start[seq]
            var count = self.query_start[seq + 1] - begin
            if count < 1:
                raise Error("every sequence in a step needs at least one token")
            if count == 1 and leading == seq:
                leading += 1
            var length = self.seq_lens[seq]
            if length < count or length > self.max_blocks * block_size:
                raise Error("sequence length does not fit its tokens and block table")
            var used = (length + block_size - 1) // block_size
            for a in range(used):
                for b in range(a + 1, used):
                    if self.block_table[seq * self.max_blocks + a] == self.block_table[seq * self.max_blocks + b]:
                        raise Error("a sequence maps two of its positions' blocks to one block")
            var past = length - count
            for i in range(count):
                var position = past + i
                if self.positions[begin + i] != position:
                    raise Error("step positions must be contiguous and end at the sequence length")
                var block = self.block_table[seq * self.max_blocks + position // block_size]
                if self.slot_mapping[begin + i] != block * block_size + position % block_size:
                    raise Error("write slot disagrees with the block table")
        if self.decode_count != leading:
            raise Error("decode count must equal the leading one-token sequences")
        # A block receiving writes in this step belongs to exactly one sequence.
        for seq in range(s):
            var first = (self.seq_lens[seq] - (self.query_start[seq + 1] - self.query_start[seq])) // block_size
            var last = (self.seq_lens[seq] - 1) // block_size
            for index in range(first, last + 1):
                var written = self.block_table[seq * self.max_blocks + index]
                for other in range(s):
                    if other == seq:
                        continue
                    var used = (self.seq_lens[other] + block_size - 1) // block_size
                    for j in range(used):
                        if self.block_table[other * self.max_blocks + j] == written:
                            raise Error("a block written in this step belongs to another sequence")
        var previous = -1
        for row in self.logits_rows:
            if row <= previous or row >= n:
                raise Error("logit rows must increase within the step")
            var ends_sequence = False
            for seq in range(s):
                if row == self.query_start[seq + 1] - 1:
                    ends_sequence = True
            if not ends_sequence:
                raise Error("a logit row must be the last row of its sequence")
            previous = row
