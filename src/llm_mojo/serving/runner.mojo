"""A synchronous runner boundary shared by the engine and timing simulations.

The runner returns one token per logits_rows entry and completes every use of
the step's KV before returning. A GPU adapter lives with its model, not here.
The scripted runner is a lifecycle oracle, not numerical or GPU evidence.
"""
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool


trait ModelRunner(Movable):
    def now_ns(self) -> Int:
        ...

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        ...


struct SimulatedRunner(ModelRunner):
    var script: List[Int]
    var vocabulary: Int
    var clock_ns: Int
    var fixed_ns: Int
    var per_token_ns: Int
    var per_position_ns: Int
    var steps: Int

    def __init__(out self, script: List[Int], vocabulary: Int, fixed_ns: Int = 1000,
                 per_token_ns: Int = 10, per_position_ns: Int = 1) raises:
        if len(script) == 0 or vocabulary < 1 or fixed_ns < 0 or per_token_ns < 0 or per_position_ns < 0:
            raise Error("invalid simulated runner configuration")
        for token in script:
            if token < 0 or token >= vocabulary:
                raise Error("simulated token is outside the vocabulary")
        self.script = script.copy()
        self.vocabulary = vocabulary
        self.clock_ns = 0
        self.fixed_ns = fixed_ns
        self.per_token_ns = per_token_ns
        self.per_position_ns = per_position_ns
        self.steps = 0

    def now_ns(self) -> Int:
        return self.clock_ns

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        batch.validate(kv.blocks, kv.block_size, self.vocabulary)
        var attended = 0
        # Validate all append boundaries before advancing any block.
        for s in range(batch.sequences()):
            var past = batch.positions[batch.query_start[s]]
            var length = batch.seq_lens[s]
            for b in range((length + kv.block_size - 1) // kv.block_size):
                var written = kv.written[batch.block_table[s * batch.max_blocks + b]]
                var expected = min(max(past - b * kv.block_size, 0), kv.block_size)
                if written != expected:
                    raise Error("simulated append does not start at the cached length")
            var count = batch.query_start[s + 1] - batch.query_start[s]
            attended += count * past + count * (count + 1) // 2
        for s in range(batch.sequences()):
            var past = batch.positions[batch.query_start[s]]
            var length = batch.seq_lens[s]
            for b in range(past // kv.block_size, (length + kv.block_size - 1) // kv.block_size):
                kv.written[batch.block_table[s * batch.max_blocks + b]] = min(kv.block_size, length - b * kv.block_size)
        var selected = List[Int](capacity=len(batch.logits_rows))
        for row in batch.logits_rows:
            # No global token cursor: replay and interleaving preserve this oracle.
            selected.append(self.script[(batch.token_ids[row] + batch.positions[row]) % len(self.script)])
        self.clock_ns += self.fixed_ns + self.per_token_ns * batch.rows() + self.per_position_ns * attended
        self.steps += 1
        return selected^
