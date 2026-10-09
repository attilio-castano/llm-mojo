"""Bounded synchronous request scheduling over a caller-owned KV pool.

One step takes running decodes first and at most one prompt/replay chunk.
Incremental admission drops KV under pressure and recomputes retained history.
Optional lifetime reservation instead admits only requests whose declared peak
KV demand fits and keeps their blocks until completion. Prefix caching is a
later mechanism. Lists have declared capacities, but this implementation does
not claim allocation-free host scheduling. The runner must finish before return.
"""
from std.math import ceildiv
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.blocks import BlockManager
from llm_mojo.serving.kv_pool import KVPool
from llm_mojo.serving.runner import ModelRunner

comptime WAITING = 0
comptime PREFILL = 1
comptime DECODE = 2
comptime FINISHED = 3
comptime TOKEN_EVENT = 0
comptime FINISH_EVENT = 1


def _saturated_sum(left: Int, right: Int) -> Int:
    return Int.MAX if left > Int.MAX - right else left + right


def _saturated_product(left: Int, right: Int) -> Int:
    if right == 0:
        return 0
    return Int.MAX if left > Int.MAX // right else left * right


struct StepCost(ImplicitlyCopyable, Movable):
    """Frozen nonnegative integer coefficients; a prediction, not a hard SLO.

    Partitions count attention launches per layer: leading singleton queries
    and the optional multi-row tail. A one-token prefill joins the singletons.
    Overflow saturates rather than wrapping a large prediction below target.
    """
    var fixed_ns: Int
    var per_row_ns: Int
    var per_position_ns: Int
    var per_partition_ns: Int
    var per_logit_ns: Int
    var target_ns: Int

    def __init__(out self, fixed_ns: Int, per_row_ns: Int, per_position_ns: Int,
                 per_partition_ns: Int, per_logit_ns: Int, target_ns: Int = 25_000_000):
        self.fixed_ns = fixed_ns
        self.per_row_ns = per_row_ns
        self.per_position_ns = per_position_ns
        self.per_partition_ns = per_partition_ns
        self.per_logit_ns = per_logit_ns
        self.target_ns = target_ns

    def validate(self) raises:
        if (self.fixed_ns < 0 or self.per_row_ns < 0 or self.per_position_ns < 0
                or self.per_partition_ns < 0 or self.per_logit_ns < 0 or self.target_ns < 1):
            raise Error("step cost needs nonnegative coefficients and a positive target")

    def estimate(self, rows: Int, positions: Int, partitions: Int, logits: Int) -> Int:
        var result = self.fixed_ns
        result = _saturated_sum(result, _saturated_product(self.per_row_ns, rows))
        result = _saturated_sum(result, _saturated_product(self.per_position_ns, positions))
        result = _saturated_sum(result, _saturated_product(self.per_partition_ns, partitions))
        return _saturated_sum(result, _saturated_product(self.per_logit_ns, logits))


struct EngineRequest(Movable):
    var request_id: Int
    var tokens: List[Int]
    var stop_ids: List[Int]
    var prompt_length: Int
    var maximum: Int
    var generated: Int
    var state: Int
    var sequence: Int
    var ticket: Int
    var abort_requested: Bool
    var reason: String
    var preemptions: Int
    var arrival_ns: Int

    def __init__(out self, request_id: Int, prompt: List[Int], maximum: Int,
                 stops: List[Int], ticket: Int, arrival_ns: Int):
        self.request_id = request_id
        self.tokens = List[Int](capacity=len(prompt) + maximum)
        for token in prompt:
            self.tokens.append(token)
        self.stop_ids = stops.copy()
        self.prompt_length = len(prompt)
        self.maximum = maximum
        self.generated = 0
        self.state = WAITING
        self.sequence = -1
        self.ticket = ticket
        self.abort_requested = False
        self.reason = ""
        self.preemptions = 0
        self.arrival_ns = arrival_ns


@fieldwise_init
struct EngineEvent(Copyable, Movable):
    var kind: Int
    var request_id: Int
    var token_id: Int
    var reason: String
    var prompt_tokens: Int
    var generated_tokens: Int
    var arrival_ns: Int
    var emitted_ns: Int


struct EngineStep(Movable):
    var step_id: Int
    var decode_seqs: Int
    var prefill_seqs: Int
    var prefill_tokens: Int
    var total_tokens: Int
    var attended_positions: Int
    var admitted: Int
    var preempted: Int
    var finished: Int
    var aborted: Int
    var waiting: Int
    var blocks_free: Int
    var begin_ns: Int
    var schedule_ns: Int
    var build_ns: Int
    var execute_ns: Int
    var postprocess_ns: Int
    var end_ns: Int
    var predicted_ns: Int
    var budget_limited: Int
    var events: List[EngineEvent]

    def __init__(out self, step_id: Int, capacity: Int):
        self.step_id = step_id
        self.decode_seqs = 0
        self.prefill_seqs = 0
        self.prefill_tokens = 0
        self.total_tokens = 0
        self.attended_positions = 0
        self.admitted = 0
        self.preempted = 0
        self.finished = 0
        self.aborted = 0
        self.waiting = 0
        self.blocks_free = 0
        self.begin_ns = 0
        self.schedule_ns = 0
        self.build_ns = 0
        self.execute_ns = 0
        self.postprocess_ns = 0
        self.end_ns = 0
        self.predicted_ns = 0
        self.budget_limited = 0
        self.events = List[EngineEvent](capacity=capacity)


struct EngineCore(Movable):
    var blocks: BlockManager
    var requests: List[EngineRequest]
    var max_context: Int
    var vocabulary: Int
    var token_budget: Int
    var max_sequences: Int
    var max_requests: Int
    var watermark_blocks: Int
    var mixed_prefill: Bool
    var reserve_lifetime: Bool
    var cost_policy: StepCost
    var cost_policy_enabled: Bool
    var next_ticket: Int
    var step_id: Int
    var failed: Bool
    var failure_events: List[EngineEvent]

    def __init__(out self, blocks: Int, block_size: Int, max_context: Int, vocabulary: Int,
                 token_budget: Int = 256, max_sequences: Int = 64, max_requests: Int = 128,
                 watermark_blocks: Int = 0, mixed_prefill: Bool = True,
                 reserve_lifetime: Bool = False) raises:
        if (blocks < 1 or block_size < 1 or max_context < 1 or max_context > 4096 or vocabulary < 1
                or token_budget < 1 or token_budget > 4096 or max_sequences < 1 or max_sequences > 64
                or max_requests < 1 or watermark_blocks < 0 or watermark_blocks > blocks):
            raise Error("invalid bounded engine configuration")
        self.blocks = BlockManager(blocks, block_size, max_context)
        self.requests = List[EngineRequest](capacity=max_requests)
        self.max_context = max_context
        self.vocabulary = vocabulary
        self.token_budget = min(token_budget, max_context)
        self.max_sequences = max_sequences
        self.max_requests = max_requests
        self.watermark_blocks = watermark_blocks
        self.mixed_prefill = mixed_prefill
        self.reserve_lifetime = reserve_lifetime
        self.cost_policy = StepCost(0, 0, 0, 0, 0)
        self.cost_policy_enabled = False
        self.next_ticket = 0
        self.step_id = 0
        self.failed = False
        self.failure_events = List[EngineEvent](capacity=max_requests)

    def set_cost_policy(mut self, policy: StepCost) raises:
        """Enable a frozen cost model only after validating all its fields."""
        policy.validate()
        if self.failed:
            raise Error("a failed engine cannot change policy")
        self.cost_policy = policy
        self.cost_policy_enabled = True

    def live(self) -> Int:
        var count = 0
        for i in range(len(self.requests)):
            if self.requests[i].state != FINISHED:
                count += 1
        return count

    def add(mut self, request_id: Int, prompt: List[Int], max_new_tokens: Int,
            stop_ids: List[Int], arrival_ns: Int = -1) raises -> Int:
        """Validate before mutation; a reused finished slot receives a fresh ticket."""
        if self.failed or request_id < 0 or len(prompt) < 1 or max_new_tokens < 0 or arrival_ns < -1:
            raise Error("invalid engine request")
        if len(prompt) > self.max_context or max_new_tokens > self.max_context - len(prompt):
            raise Error("request exceeds its declared context")
        # The last selected token need not enter KV: generation ends immediately.
        var extent = max(len(prompt), len(prompt) + max_new_tokens - 1)
        if ceildiv(extent, self.blocks.block_size) > self.blocks.blocks:
            raise Error("request cannot fit its declared KV pool even alone")
        for token in prompt:
            if token < 0 or token >= self.vocabulary:
                raise Error("prompt token is outside the vocabulary")
        for token in stop_ids:
            if token < 0 or token >= self.vocabulary:
                raise Error("stop token is outside the vocabulary")
        var slot = -1
        for i in range(len(self.requests)):
            if self.requests[i].state != FINISHED and self.requests[i].request_id == request_id:
                raise Error("duplicate live request ID")
            if self.requests[i].state == FINISHED and slot < 0:
                slot = i
        if slot < 0 and len(self.requests) == self.max_requests:
            raise Error("engine request queue is full")
        var request = EngineRequest(request_id, prompt, max_new_tokens, stop_ids, self.next_ticket, arrival_ns)
        if slot < 0:
            slot = len(self.requests)
            self.requests.append(request^)
        else:
            self.requests[slot] = request^
        self.next_ticket += 1
        return slot

    def abort(mut self, request_id: Int):
        """Idempotent; take effect at the next synchronous step boundary."""
        for i in range(len(self.requests)):
            if self.requests[i].request_id == request_id and self.requests[i].state != FINISHED:
                self.requests[i].abort_requested = True

    def _held(self, index: Int) -> Bool:
        return self.requests[index].sequence >= 0

    def _has_resident(self) -> Bool:
        for i in range(len(self.requests)):
            if self._held(i):
                return True
        return False

    def _peak_extent(self, index: Int) -> Int:
        # The last emitted token is never processed when the request finishes.
        if self.requests[index].maximum == 0:
            return 0
        return self.requests[index].prompt_length + self.requests[index].maximum - 1

    def _release(mut self, index: Int, mut kv: KVPool) raises:
        if self._held(index):
            var sequence = self.requests[index].sequence
            kv.truncate_table(self.blocks.table(sequence), 0)
            self.blocks.release(sequence)
            self.requests[index].sequence = -1

    def _finish(mut self, index: Int, reason: String, mut kv: KVPool, mut record: EngineStep, at_ns: Int) raises:
        self._release(index, kv)
        self.requests[index].state = FINISHED
        self.requests[index].reason = reason
        record.events.append(EngineEvent(FINISH_EVENT, self.requests[index].request_id, -1, reason,
                                        self.requests[index].prompt_length, self.requests[index].generated,
                                        self.requests[index].arrival_ns, at_ns))
        record.finished += 1
        if reason == "abort":
            record.aborted += 1

    def _preempt(mut self, index: Int, mut kv: KVPool, mut record: EngineStep) raises:
        self._release(index, kv)
        self.requests[index].state = WAITING
        self.requests[index].preemptions += 1
        record.preempted += 1

    def _oldest(self, state: Int, excluded: List[Int]) -> Int:
        var index = -1
        for i in range(len(self.requests)):
            if self.requests[i].state != state:
                continue
            var skip = False
            for done in excluded:
                if done == i:
                    skip = True
            if not skip and (index < 0 or self.requests[i].ticket < self.requests[index].ticket):
                index = i
        return index

    def _make_room(mut self, index: Int, length: Int, selected: List[Int],
                   mut kv: KVPool, mut record: EngineStep) raises -> Bool:
        if self.reserve_lifetime:
            # Admission already owns every future block. Growing the written
            # extent cannot evict another owner or require a new allocation.
            if length > self.blocks.reserved[self.requests[index].sequence]:
                raise Error("a step exceeds its lifetime KV reservation")
            return True
        var needed = ceildiv(length, self.blocks.block_size) - len(self.blocks.tables[self.requests[index].sequence])
        while needed > self.blocks.free_blocks():
            # Already scheduled holders cannot be invalidated. Among the rest,
            # dropping the newest preserves progress of the oldest request.
            var victim = -1
            for i in range(len(self.requests)):
                if not self._held(i):
                    continue
                var skip = False
                for done in selected:
                    if done == i:
                        skip = True
                if not skip and (victim < 0 or self.requests[i].ticket > self.requests[victim].ticket):
                    victim = i
            if victim < 0:
                raise Error("no reclaimable owner for an unsatisfiable reservation")
            self._preempt(victim, kv, record)
            if victim == index:
                return False
        self.blocks.reserve(self.requests[index].sequence, length)
        return True

    def _prefill_count(self, index: Int, budget: Int, selected: List[Int],
                       mut record: EngineStep) raises -> Int:
        var past = self.blocks.length(self.requests[index].sequence) if self._held(index) else 0
        var remaining = len(self.requests[index].tokens) - past
        var maximum = min(budget, remaining)
        if not self.cost_policy_enabled:
            return maximum
        var decodes = len(selected)
        var positions = 0
        for done in selected:
            positions += self.blocks.length(self.requests[done].sequence) + 1
        var low = 1
        var high = maximum
        var best = 0
        while low <= high:
            var count = low + (high - low) // 2
            var partitions = (1 if decodes > 0 or count == 1 else 0) + (1 if count > 1 else 0)
            var logits = decodes + (1 if count == remaining else 0)
            var cost = self.cost_policy.estimate(decodes + count,
                positions + count * past + count * (count + 1) // 2, partitions, logits)
            if cost <= self.cost_policy.target_ns:
                best = count
                low = count + 1
            else:
                high = count - 1
        if best < maximum:
            record.budget_limited = 1
        # Mandatory decodes always progress. With none, one prompt token must
        # progress even when fixed cost alone exceeds the research target.
        return max(best, 1) if decodes == 0 else best

    def _schedule_prefill(mut self, budget: Int, mut selected: List[Int], mut counts: List[Int],
                          mut kv: KVPool, mut record: EngineStep) raises -> Bool:
        if budget < 1 or len(selected) == self.max_sequences:
            return False
        var index = self._oldest(PREFILL, List[Int]())
        var count = 0
        if index >= 0:
            var past = self.blocks.length(self.requests[index].sequence)
            count = self._prefill_count(index, budget, selected, record)
            if count == 0:
                return False
            if not self._make_room(index, past + count, selected, kv, record):
                # A standalone prefill that cannot extend must give older
                # decodes a turn instead of immediately replaying itself.
                if not self.mixed_prefill:
                    return False
                index = -1
        if index < 0:
            index = self._oldest(WAITING, List[Int]())
            if index >= 0:
                var free = self.blocks.free_blocks()
                # The conservative margin applies while any request resides,
                # even if none was selected. Ignore it when the pool is empty
                # so every request validated to fit alone can eventually run.
                var residents = self._has_resident() if self.reserve_lifetime else len(selected) > 0
                var watermark = self.watermark_blocks if residents else 0
                var available = max(free - watermark, 0) * self.blocks.block_size
                count = self._prefill_count(index, budget, selected, record)
                var extent: Int
                if self.reserve_lifetime:
                    extent = self._peak_extent(index)
                    # Strict FIFO: do not bypass a larger waiting request.
                    # Existing residents have bounded work and reserved growth,
                    # so they finish without replay and eventually free room.
                    if ceildiv(extent, self.blocks.block_size) > max(free - watermark, 0):
                        return False
                else:
                    count = min(count, available)
                    extent = count
                if count > 0:
                    self.requests[index].sequence = self.blocks.add()
                    self.requests[index].state = PREFILL
                    self.blocks.reserve(self.requests[index].sequence, extent)
                    record.admitted += 1
                else:
                    return False
        if index < 0:
            return False
        # Use the chosen chunk, not the larger lifetime reservation, to retain
        # both fixed row budgets and fitted-policy limits on execution work.
        selected.append(index)
        counts.append(count)
        record.prefill_seqs = 1
        record.prefill_tokens = count
        return True

    def _batch(self, selected: List[Int], counts: List[Int]) raises -> StepBatch:
        var ids = List[Int](capacity=self.token_budget)
        var positions = List[Int](capacity=self.token_budget)
        var slots = List[Int](capacity=self.token_budget)
        var starts = List[Int](capacity=len(selected) + 1)
        var lengths = List[Int](capacity=len(selected))
        var logits = List[Int](capacity=len(selected))
        var width = 1
        for index in selected:
            width = max(width, len(self.blocks.tables[self.requests[index].sequence]))
        var tables = List[Int](capacity=len(selected) * width)
        var leading = 0
        starts.append(0)
        for s in range(len(selected)):
            var index = selected[s]
            var sequence = self.requests[index].sequence
            var past = self.blocks.length(sequence)
            var count = counts[s]
            var table = self.blocks.table(sequence)
            for i in range(count):
                var position = past + i
                ids.append(self.requests[index].tokens[position])
                positions.append(position)
                slots.append(table[position // self.blocks.block_size] * self.blocks.block_size + position % self.blocks.block_size)
            for b in range(width):
                tables.append(table[b] if b < len(table) else table[0])
            lengths.append(past + count)
            starts.append(len(ids))
            if count == 1 and leading == s:
                leading += 1
            if past + count == len(self.requests[index].tokens):
                logits.append(len(ids) - 1)
        return StepBatch(ids^, positions^, starts^, leading, lengths^, width, tables^, slots^, logits^)

    def check(self, kv: KVPool) raises:
        self.blocks.check()
        self.blocks.check_pool(kv)
        if self.live() > self.max_requests:
            raise Error("live requests exceed the bound")
        for sequence in range(len(self.blocks.active)):
            var owners = 0
            for i in range(len(self.requests)):
                if self.requests[i].sequence == sequence:
                    owners += 1
            if owners != (1 if self.blocks.active[sequence] else 0):
                raise Error("a live block-manager sequence does not have exactly one request owner")
        for i in range(len(self.requests)):
            if (len(self.requests[i].tokens) != self.requests[i].prompt_length + self.requests[i].generated
                    or self.requests[i].generated > self.requests[i].maximum):
                raise Error("request token accounting disagrees")
            if (self.requests[i].state == WAITING or self.requests[i].state == FINISHED) and self.requests[i].sequence != -1:
                raise Error("inactive request still holds KV")
            if self.requests[i].state == PREFILL or self.requests[i].state == DECODE:
                if self.requests[i].sequence < 0 or not self.blocks.active[self.requests[i].sequence]:
                    raise Error("running request has no live KV owner")
                var cached = self.blocks.length(self.requests[i].sequence)
                if self.reserve_lifetime:
                    var peak = self._peak_extent(i)
                    if (peak < 1 or self.blocks.reserved[self.requests[i].sequence] != peak
                            or len(self.blocks.tables[self.requests[i].sequence]) != ceildiv(peak, self.blocks.block_size)):
                        raise Error("a running request does not own its full lifetime KV reservation")
                if (cached > len(self.requests[i].tokens)
                        or (self.requests[i].state == DECODE and cached != len(self.requests[i].tokens) - 1)):
                    raise Error("request phase disagrees with its cached history")

    def step[Runner: ModelRunner](mut self, mut runner: Runner, mut kv: KVPool) raises -> EngineStep:
        if self.failed or kv.blocks != self.blocks.blocks or kv.block_size != self.blocks.block_size:
            raise Error("failed engine or incompatible pool")
        self.check(kv)
        var record = EngineStep(self.step_id, self.max_requests + 2 * self.max_sequences)
        record.begin_ns = runner.now_ns()
        self.step_id += 1
        for i in range(len(self.requests)):
            if self.requests[i].state != FINISHED and self.requests[i].arrival_ns < 0:
                self.requests[i].arrival_ns = record.begin_ns
            if self.requests[i].state != FINISHED and self.requests[i].abort_requested:
                self._finish(i, "abort", kv, record, record.begin_ns)
            elif self.requests[i].state != FINISHED and self.requests[i].maximum == 0:
                self._finish(i, "length", kv, record, record.begin_ns)
        var selected = List[Int](capacity=self.max_sequences)
        var counts = List[Int](capacity=self.max_sequences)
        var considered = List[Int](capacity=self.max_requests)
        var budget = self.token_budget
        var standalone = False
        if not self.mixed_prefill:
            standalone = self._schedule_prefill(budget, selected, counts, kv, record)
        while not standalone and budget > 0 and len(selected) < self.max_sequences:
            var index = self._oldest(DECODE, considered)
            if index < 0:
                break
            considered.append(index)
            var length = self.blocks.length(self.requests[index].sequence) + 1
            if self._make_room(index, length, selected, kv, record):
                selected.append(index)
                counts.append(1)
                record.decode_seqs += 1
                budget -= 1
        if self.mixed_prefill:
            _ = self._schedule_prefill(budget, selected, counts, kv, record)
        record.schedule_ns = runner.now_ns() - record.begin_ns
        var build_begin = runner.now_ns()
        var postprocess_begin = build_begin
        if len(selected) > 0:
            var batch = self._batch(selected, counts)
            batch.validate(kv.blocks, kv.block_size, self.vocabulary)
            record.total_tokens = batch.rows()
            for s in range(len(selected)):
                var past = batch.positions[batch.query_start[s]]
                record.attended_positions += counts[s] * past + counts[s] * (counts[s] + 1) // 2
            if self.cost_policy_enabled:
                var partitions = (1 if batch.decode_count > 0 else 0) + (1 if batch.rows() > batch.decode_count else 0)
                record.predicted_ns = self.cost_policy.estimate(batch.rows(), record.attended_positions,
                                                               partitions, len(batch.logits_rows))
            record.build_ns = runner.now_ns() - build_begin
            var execute_begin = runner.now_ns()
            var tokens: List[Int]
            try:
                tokens = runner.execute(batch, kv)
                if len(tokens) != len(batch.logits_rows):
                    raise Error("runner returned the wrong number of selected tokens")
                for token in tokens:
                    if token < 0 or token >= self.vocabulary:
                        raise Error("runner returned an invalid token")
                for s in range(batch.sequences()):
                    var length = batch.seq_lens[s]
                    for b in range((length + kv.block_size - 1) // kv.block_size):
                        if kv.written[batch.block_table[s * batch.max_blocks + b]] != min(kv.block_size, length - b * kv.block_size):
                            raise Error("runner did not commit the step's declared KV extent")
            except error:
                self.failed = True
                # The adapter drains pending work before raising. This engine
                # is permanently invalidated; these releases only drop logical
                # ownership and are not a recovery operation.
                for i in range(len(self.requests)):
                    if self.requests[i].state != FINISHED:
                        self._finish(i, "error", kv, record, runner.now_ns())
                self.failure_events = record.events.copy()
                raise error
            postprocess_begin = runner.now_ns()
            record.execute_ns = postprocess_begin - execute_begin
            var sampled = 0
            for s in range(len(selected)):
                var index = selected[s]
                self.blocks.commit(self.requests[index].sequence, batch.seq_lens[s])
                if batch.seq_lens[s] == len(self.requests[index].tokens):
                    var token = tokens[sampled]
                    sampled += 1
                    self.requests[index].tokens.append(token)
                    self.requests[index].generated += 1
                    self.requests[index].state = DECODE
                    record.events.append(EngineEvent(TOKEN_EVENT, self.requests[index].request_id, token, "",
                        self.requests[index].prompt_length, self.requests[index].generated,
                        self.requests[index].arrival_ns, postprocess_begin))
                    var stopped = False
                    for stop in self.requests[index].stop_ids:
                        if token == stop:
                            stopped = True
                    if stopped:
                        self._finish(index, "stop", kv, record, postprocess_begin)
                    elif self.requests[index].generated == self.requests[index].maximum:
                        self._finish(index, "length", kv, record, postprocess_begin)
        record.blocks_free = self.blocks.free_blocks()
        for i in range(len(self.requests)):
            if self.requests[i].state == WAITING:
                record.waiting += 1
        self.check(kv)
        record.end_ns = runner.now_ns()
        record.postprocess_ns = record.end_ns - postprocess_begin
        return record^
