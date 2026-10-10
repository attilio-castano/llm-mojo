"""Bounded request scheduling over a caller-owned KV pool.

One step takes running decodes first and at most one prompt/replay chunk.
Incremental admission drops KV under pressure and recomputes retained history.
Optional lifetime reservation instead admits only requests whose declared peak
KV demand fits and keeps their blocks until completion. Prefix caching is a
later mechanism. Lists have declared capacities, but this implementation does
not claim allocation-free host scheduling. Synchronous step finishes every GPU
use before return. Optional step_async queues at most one successor before
collecting the oldest result and retains each referenced KV owner until its
last outstanding use completes. Delivered token IDs remain separate from
symbolic GPU dependencies and enqueued KV extents.
"""
from std.math import ceildiv
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.blocks import BlockManager
from llm_mojo.serving.kv_pool import KVPool
from llm_mojo.serving.runner import EngineClock, ModelRunner, AsyncModelRunner

comptime WAITING = 0
comptime PREFILL = 1
comptime DECODE = 2
comptime FINISHED = 3
comptime DRAINING = 4
comptime TOKEN_EVENT = 0
comptime FINISH_EVENT = 1


def _saturated_sum(left: Int, right: Int) -> Int:
    return Int.MAX if left > Int.MAX - right else left + right


def _saturated_product(left: Int, right: Int) -> Int:
    if right == 0:
        return 0
    return Int.MAX if left > Int.MAX // right else left * right


def _peak_kv_extent(prompt_length: Int, maximum: Int) -> Int:
    # Zero output finishes before scheduling; the final emitted token never enters KV.
    return 0 if maximum == 0 else prompt_length + maximum - 1


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
    # Uncollected samples are symbolic GPU dependencies, never delivered IDs.
    var pending_samples: Int
    var pending_uses: Int

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
        self.pending_samples = 0
        self.pending_uses = 0


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


@fieldwise_init
struct EngineKVObservation(ImplicitlyCopyable, Movable):
    """Opt-in host observations; a written count is not a GPU write timestamp.

    Phases 0..3 are step start, scheduled, runner returned, and step end.
    Pool written counts after the runner returns include its validated writes,
    before token sampling can finish a request and release its table.
    """
    var phase: Int
    var at_ns: Int
    var allocated_blocks: Int
    var written_blocks: Int
    var written_tokens: Int
    var reserved_tokens: Int
    var waiting_requests: Int
    var resident_requests: Int


@fieldwise_init
struct EngineAsyncSubmission(ImplicitlyCopyable, Movable):
    var ticket: Int
    var buffer_slot: Int
    var decode_seqs: Int
    var prefill_seqs: Int
    var prefill_tokens: Int
    var total_tokens: Int
    var attended_positions: Int
    var selected_logits: Int
    var begin_ns: Int
    var submitted_ns: Int
    var pending: Int


@fieldwise_init
struct EngineAsyncHead(ImplicitlyCopyable, Movable):
    var ticket: Int
    var head: Int
    var request_id: Int
    var generated_tokens: Int


@fieldwise_init
struct EngineAsyncCompletion(ImplicitlyCopyable, Movable):
    var ticket: Int
    var begin_ns: Int
    var completed_ns: Int
    var pending: Int


@fieldwise_init
struct EngineAsyncResult(Copyable, Movable):
    var ticket: Int
    var head: Int
    var request_id: Int
    var token_id: Int
    var generated_tokens: Int
    var disposition: String
    var observed_ns: Int


struct EngineStep(Copyable, Movable):
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
    var kv_observations: List[EngineKVObservation]
    var admitted_request_id: Int
    var admitted_ns: Int
    var execute_begin_ns: Int
    var execute_end_ns: Int
    var async_submitted_ns: Int
    var async_collect_begin_ns: Int
    var async_collect_end_ns: Int
    var async_inflight: Int
    var async_chained_tokens: Int
    var async_discarded_tokens: Int
    var async_pressure_drain: Int
    var async_ticket: Int
    var async_submissions: List[EngineAsyncSubmission]
    var async_heads: List[EngineAsyncHead]
    var async_completions: List[EngineAsyncCompletion]
    var async_results: List[EngineAsyncResult]

    def __init__(out self, step_id: Int, capacity: Int, observe_kv: Bool = False,
                 async_step: Bool = False):
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
        self.kv_observations = List[EngineKVObservation](capacity=4 if observe_kv else 0)
        self.admitted_request_id = -1
        self.admitted_ns = -1
        self.execute_begin_ns = 0
        self.execute_end_ns = 0
        self.async_submitted_ns = 0
        self.async_collect_begin_ns = 0
        self.async_collect_end_ns = 0
        self.async_inflight = 0
        self.async_chained_tokens = 0
        self.async_discarded_tokens = 0
        self.async_pressure_drain = 0
        self.async_ticket = -1
        self.async_submissions = List[EngineAsyncSubmission](capacity=2 if async_step else 0)
        self.async_heads = List[EngineAsyncHead](capacity=2 * capacity if async_step else 0)
        self.async_completions = List[EngineAsyncCompletion](capacity=2 if async_step else 0)
        self.async_results = List[EngineAsyncResult](capacity=capacity if async_step else 0)


@fieldwise_init
struct EngineSelection(Movable):
    var selected: List[Int]
    var counts: List[Int]


@fieldwise_init
struct EngineAsyncInput(Movable):
    var batch: StepBatch
    var source_indices: List[Int]
    var sample_indices: List[Int]


@fieldwise_init
struct EngineAsyncPending(Movable):
    """Immutable request identities and work belonging to one runner ticket."""
    var runner_ticket: Int
    var selected: List[Int]
    var request_tickets: List[Int]
    var sample_indices: List[Int]
    var sample_generations: List[Int]
    var batch: StepBatch
    var record: EngineStep


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
    var observe_kv: Bool
    var cost_policy: StepCost
    var cost_policy_enabled: Bool
    var next_ticket: Int
    var step_id: Int
    var failed: Bool
    var failure_events: List[EngineEvent]
    var async_pending: List[EngineAsyncPending]
    var async_enabled: Bool
    var async_last_ticket: Int

    def __init__(out self, blocks: Int, block_size: Int, max_context: Int, vocabulary: Int,
                 token_budget: Int = 256, max_sequences: Int = 64, max_requests: Int = 128,
                 watermark_blocks: Int = 0, mixed_prefill: Bool = True,
                 reserve_lifetime: Bool = False, observe_kv: Bool = False) raises:
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
        self.observe_kv = observe_kv
        self.cost_policy = StepCost(0, 0, 0, 0, 0)
        self.cost_policy_enabled = False
        self.next_ticket = 0
        self.step_id = 0
        self.failed = False
        self.failure_events = List[EngineEvent](capacity=max_requests)
        self.async_pending = List[EngineAsyncPending](capacity=2)
        self.async_enabled = False
        self.async_last_ticket = -1

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
        var extent = _peak_kv_extent(len(prompt), max_new_tokens)
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
        """Idempotent; stop delivery at the next host step boundary."""
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
        return _peak_kv_extent(self.requests[index].prompt_length, self.requests[index].maximum)

    def _release(mut self, index: Int, mut kv: KVPool) raises:
        if self._held(index):
            var sequence = self.requests[index].sequence
            kv.truncate_table(self.blocks.table(sequence), 0)
            self.blocks.release(sequence)
            self.requests[index].sequence = -1

    def _finish(mut self, index: Int, reason: String, mut kv: KVPool, mut record: EngineStep, at_ns: Int) raises:
        if self.requests[index].state == FINISHED or self.requests[index].state == DRAINING:
            return
        # A stop/abort can be known while its successor still uses these blocks.
        # Terminal events are emitted once; physical ownership retires later.
        if self.requests[index].pending_uses > 0:
            self.requests[index].state = DRAINING
        else:
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
        if self.requests[index].pending_uses > 0:
            raise Error("cannot preempt an outstanding asynchronous KV owner")
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
                if not self._held(i) or self.requests[i].pending_uses > 0:
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

    def _history_length(self, index: Int) -> Int:
        return len(self.requests[index].tokens) + self.requests[index].pending_samples

    def _prefill_count(self, index: Int, budget: Int, selected: List[Int],
                       mut record: EngineStep) raises -> Int:
        var past = self.blocks.length(self.requests[index].sequence) if self._held(index) else 0
        var remaining = self._history_length(index) - past
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

    def _schedule_prefill[Runner: EngineClock](mut self, budget: Int, mut selected: List[Int], mut counts: List[Int],
                          mut runner: Runner, mut kv: KVPool, mut record: EngineStep) raises -> Bool:
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
                    if self.observe_kv:
                        # Capture the successful ownership transition even if
                        # this request finishes in the same execution step.
                        record.admitted_request_id = self.requests[index].request_id
                        record.admitted_ns = runner.now_ns()
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

    def _schedule[Runner: EngineClock](mut self, mut runner: Runner, mut kv: KVPool,
                                     mut record: EngineStep) raises -> EngineSelection:
        """The same FIFO/mixed selection for synchronous and projected histories."""
        var selected = List[Int](capacity=self.max_sequences)
        var counts = List[Int](capacity=self.max_sequences)
        var considered = List[Int](capacity=self.max_requests)
        var budget = self.token_budget
        var standalone = False
        if not self.mixed_prefill:
            standalone = self._schedule_prefill(budget, selected, counts, runner, kv, record)
        while not standalone and budget > 0 and len(selected) < self.max_sequences:
            var index = self._oldest(DECODE, considered)
            if index < 0:
                break
            considered.append(index)
            # The known limit never needs a speculative terminal input token.
            if self.requests[index].generated + self.requests[index].pending_samples >= self.requests[index].maximum:
                continue
            var length = self.blocks.length(self.requests[index].sequence) + 1
            if self._make_room(index, length, selected, kv, record):
                selected.append(index)
                counts.append(1)
                record.decode_seqs += 1
                budget -= 1
        if self.mixed_prefill:
            _ = self._schedule_prefill(budget, selected, counts, runner, kv, record)
        return EngineSelection(selected^, counts^)

    def _batch(self, selected: List[Int], counts: List[Int],
               predecessor_selected: List[Int], predecessor_samples: List[Int]) raises -> EngineAsyncInput:
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
        var sources = List[Int](capacity=self.token_budget)
        var samples = List[Int](capacity=len(selected))
        starts.append(0)
        for s in range(len(selected)):
            var index = selected[s]
            var sequence = self.requests[index].sequence
            var past = self.blocks.length(sequence)
            var count = counts[s]
            var table = self.blocks.table(sequence)
            for i in range(count):
                var position = past + i
                if position < len(self.requests[index].tokens):
                    ids.append(self.requests[index].tokens[position])
                    sources.append(-1)
                else:
                    # Only a single unresolved predecessor is addressable.
                    var source = -1
                    for prior in range(len(predecessor_selected)):
                        if predecessor_selected[prior] == index:
                            source = predecessor_samples[prior]
                    if (position != len(self.requests[index].tokens)
                            or self.requests[index].pending_samples != 1 or source < 0):
                        raise Error("a symbolic token has no immediately preceding selected result")
                    ids.append(0)
                    sources.append(source)
                positions.append(position)
                slots.append(table[position // self.blocks.block_size] * self.blocks.block_size + position % self.blocks.block_size)
            for b in range(width):
                tables.append(table[b] if b < len(table) else table[0])
            lengths.append(past + count)
            starts.append(len(ids))
            if count == 1 and leading == s:
                leading += 1
            samples.append(-1)
            if past + count == self._history_length(index):
                samples[s] = len(logits)
                logits.append(len(ids) - 1)
        var batch = StepBatch(ids^, positions^, starts^, leading, lengths^, width, tables^, slots^, logits^)
        return EngineAsyncInput(batch^, sources^, samples^)

    def _copy_batch(self, batch: StepBatch) -> StepBatch:
        # Keep aggregate values intact across raising scheduler/runner calls.
        return StepBatch(batch.token_ids.copy(), batch.positions.copy(), batch.query_start.copy(),
            batch.decode_count, batch.seq_lens.copy(), batch.max_blocks, batch.block_table.copy(),
            batch.slot_mapping.copy(), batch.logits_rows.copy())

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
            var uses = 0
            var samples = 0
            for p in range(len(self.async_pending)):
                for s in range(len(self.async_pending[p].selected)):
                    if self.async_pending[p].selected[s] == i:
                        if self.async_pending[p].request_tickets[s] != self.requests[i].ticket:
                            raise Error("a pending step refers to a reused request slot")
                        uses += 1
                        samples += 1 if self.async_pending[p].sample_indices[s] >= 0 else 0
            if uses != self.requests[i].pending_uses or samples != self.requests[i].pending_samples:
                raise Error("pending request accounting disagrees with queued tickets")
            if (len(self.requests[i].tokens) != self.requests[i].prompt_length + self.requests[i].generated
                    or self.requests[i].generated > self.requests[i].maximum
                    or self.requests[i].pending_samples < 0 or self.requests[i].pending_samples > self.requests[i].pending_uses
                    or self.requests[i].pending_uses < 0 or self.requests[i].pending_uses > 2
                    or self.requests[i].generated + self.requests[i].pending_samples > self.requests[i].maximum):
                raise Error("request token accounting disagrees")
            if (self.requests[i].state == WAITING or self.requests[i].state == FINISHED) and self.requests[i].sequence != -1:
                raise Error("inactive request still holds KV")
            if self.requests[i].state == PREFILL or self.requests[i].state == DECODE or self.requests[i].state == DRAINING:
                if self.requests[i].sequence < 0 or not self.blocks.active[self.requests[i].sequence]:
                    raise Error("running request has no live KV owner")
                var cached = self.blocks.length(self.requests[i].sequence)
                if self.reserve_lifetime:
                    var peak = self._peak_extent(i)
                    if (peak < 1 or self.blocks.reserved[self.requests[i].sequence] != peak
                            or len(self.blocks.tables[self.requests[i].sequence]) != ceildiv(peak, self.blocks.block_size)):
                        raise Error("a running request does not own its full lifetime KV reservation")
                if (cached > self._history_length(i)
                        or (self.requests[i].state == DECODE and cached != self._history_length(i) - 1)):
                    raise Error("request phase disagrees with its cached history")
            if self.requests[i].state == DRAINING and self.requests[i].pending_uses == 0:
                raise Error("a draining request has no outstanding GPU use")
        if len(self.async_pending) > 2:
            raise Error("asynchronous submission exceeds two tickets")

    def pending_steps(self) -> Int:
        return len(self.async_pending)

    def _boundary(mut self, mut kv: KVPool, mut record: EngineStep, at_ns: Int) raises:
        for i in range(len(self.requests)):
            if self.requests[i].state == FINISHED or self.requests[i].state == DRAINING:
                continue
            if self.requests[i].arrival_ns < 0:
                self.requests[i].arrival_ns = at_ns
            if self.requests[i].abort_requested:
                self._finish(i, "abort", kv, record, at_ns)
            elif self.requests[i].maximum == 0:
                self._finish(i, "length", kv, record, at_ns)

    def _pressure_requires_drain(self) raises -> Bool:
        """Conservative growth check before any incremental eviction is possible.

        Smaller fitted chunks may fit without draining; using the fixed upper
        bound here is safe and keeps this guard independent of model estimates.
        Waiting admission only consumes free blocks and never evicts an owner.
        """
        if self.reserve_lifetime:
            return False
        var budget = self.token_budget
        var needed = 0
        var selected = 0
        var considered = List[Int](capacity=self.max_requests)
        var prefill = self._oldest(PREFILL, List[Int]())
        if not self.mixed_prefill and prefill >= 0:
            var sequence = self.requests[prefill].sequence
            var count = min(budget, self._history_length(prefill) - self.blocks.length(sequence))
            needed = ceildiv(self.blocks.length(sequence) + count, self.blocks.block_size) - len(self.blocks.tables[sequence])
            return needed > self.blocks.free_blocks()
        while budget > 0 and selected < self.max_sequences:
            var index = self._oldest(DECODE, considered)
            if index < 0:
                break
            considered.append(index)
            if self.requests[index].generated + self.requests[index].pending_samples >= self.requests[index].maximum:
                continue
            var sequence = self.requests[index].sequence
            needed += ceildiv(self.blocks.length(sequence) + 1, self.blocks.block_size) - len(self.blocks.tables[sequence])
            budget -= 1
            selected += 1
        if self.mixed_prefill and budget > 0 and selected < self.max_sequences and prefill >= 0:
            var sequence = self.requests[prefill].sequence
            var count = min(budget, self._history_length(prefill) - self.blocks.length(sequence))
            needed += ceildiv(self.blocks.length(sequence) + count, self.blocks.block_size) - len(self.blocks.tables[sequence])
        return needed > self.blocks.free_blocks()

    def _submit_async[Runner: AsyncModelRunner](mut self, mut runner: Runner, mut kv: KVPool,
                    predecessor_ticket: Int, predecessor_selected: List[Int],
                    predecessor_samples: List[Int], inflight: Int) raises -> EngineAsyncPending:
        var record = EngineStep(self.step_id, self.max_requests + 2 * self.max_sequences, self.observe_kv, True)
        self.step_id += 1
        record.begin_ns = runner.now_ns()
        self._observe_kv(kv, record, 0, record.begin_ns)
        var selection = self._schedule(runner, kv, record)
        record.schedule_ns = runner.now_ns() - record.begin_ns
        self._observe_kv(kv, record, 1, record.begin_ns + record.schedule_ns)
        var selected = selection.selected.copy()
        var counts = selection.counts.copy()
        var owners = List[Int](capacity=len(selected))
        var generations = List[Int](capacity=len(selected))
        if len(selected) == 0:
            var batch = StepBatch(List[Int](), List[Int](), List[Int](), 0, List[Int](), 1,
                                  List[Int](), List[Int](), List[Int]())
            return EngineAsyncPending(-1, selected^, owners^, List[Int](), generations^, batch^, record^)
        var build_begin = runner.now_ns()
        var input = self._batch(selected, counts, predecessor_selected, predecessor_samples)
        var batch = self._copy_batch(input.batch)
        batch.validate(kv.blocks, kv.block_size, self.vocabulary)
        record.total_tokens = batch.rows()
        for s in range(len(selected)):
            var past = batch.positions[batch.query_start[s]]
            record.attended_positions += counts[s] * past + counts[s] * (counts[s] + 1) // 2
            owners.append(self.requests[selected[s]].ticket)
            generations.append(self.requests[selected[s]].generated + self.requests[selected[s]].pending_samples + 1
                               if input.sample_indices[s] >= 0 else -1)
        if self.cost_policy_enabled:
            var partitions = (1 if batch.decode_count > 0 else 0) + (1 if batch.rows() > batch.decode_count else 0)
            record.predicted_ns = self.cost_policy.estimate(batch.rows(), record.attended_positions,
                                                           partitions, len(batch.logits_rows))
        var has_source = False
        for source in input.source_indices:
            if source >= 0:
                has_source = True
                record.async_chained_tokens += 1
        record.build_ns = runner.now_ns() - build_begin
        record.execute_begin_ns = runner.now_ns()
        var ticket = runner.submit(batch, input.source_indices, predecessor_ticket if has_source else -1, kv)
        if ticket < 0 or ticket <= self.async_last_ticket:
            raise Error("asynchronous runner tickets must increase")
        self.async_last_ticket = ticket
        record.async_ticket = ticket
        record.async_submitted_ns = runner.now_ns()
        record.async_inflight = inflight + 1
        record.async_submissions.append(EngineAsyncSubmission(ticket, ticket % 2, record.decode_seqs,
            record.prefill_seqs, record.prefill_tokens, record.total_tokens, record.attended_positions,
            len(batch.logits_rows), record.execute_begin_ns, record.async_submitted_ns, inflight + 1))
        for s in range(len(selected)):
            var index = selected[s]
            var length = batch.seq_lens[s]
            for b in range(ceildiv(length, kv.block_size)):
                if kv.written[batch.block_table[s * batch.max_blocks + b]] != min(kv.block_size, length - b * kv.block_size):
                    raise Error("asynchronous runner did not enqueue its declared KV extent")
            self.blocks.commit(self.requests[index].sequence, length)
            self.requests[index].pending_uses += 1
            if input.sample_indices[s] >= 0:
                self.requests[index].pending_samples += 1
                self.requests[index].state = DECODE
                record.async_heads.append(EngineAsyncHead(ticket, input.sample_indices[s],
                    self.requests[index].request_id, generations[s]))
        return EngineAsyncPending(ticket, selected^, owners^, input.sample_indices.copy(),
                                  generations^, batch^, record^)

    def _retire_async[Runner: AsyncModelRunner](mut self, mut runner: Runner, mut kv: KVPool,
                                              var pending: EngineAsyncPending) raises -> EngineStep:
        pending.record.async_collect_begin_ns = runner.now_ns()
        var tokens = runner.collect(pending.runner_ticket)
        pending.record.async_collect_end_ns = runner.now_ns()
        if len(tokens) != len(pending.batch.logits_rows):
            raise Error("asynchronous runner returned the wrong number of selected tokens")
        for token in tokens:
            if token < 0 or token >= self.vocabulary:
                raise Error("asynchronous runner returned an invalid token")
        pending.record.async_completions.append(EngineAsyncCompletion(pending.runner_ticket,
            pending.record.async_collect_begin_ns, pending.record.async_collect_end_ns, len(self.async_pending)))
        var at_ns = pending.record.async_collect_end_ns
        for s in range(len(pending.selected)):
            var index = pending.selected[s]
            if self.requests[index].ticket != pending.request_tickets[s]:
                raise Error("asynchronous result refers to a reused request slot")
            self.requests[index].pending_uses -= 1
            var sample = pending.sample_indices[s]
            if sample >= 0:
                self.requests[index].pending_samples -= 1
                var token = tokens[sample]
                var disposition = String("delivered")
                if self.requests[index].state == DRAINING:
                    disposition = "discarded-" + self.requests[index].reason
                    pending.record.async_discarded_tokens += 1
                else:
                    self.requests[index].tokens.append(token)
                    self.requests[index].generated += 1
                    pending.record.events.append(EngineEvent(TOKEN_EVENT, self.requests[index].request_id, token, "",
                        self.requests[index].prompt_length, self.requests[index].generated,
                        self.requests[index].arrival_ns, at_ns))
                    var stopped = False
                    for stop in self.requests[index].stop_ids:
                        if token == stop:
                            stopped = True
                    if stopped:
                        self._finish(index, "stop", kv, pending.record, at_ns)
                    elif self.requests[index].generated == self.requests[index].maximum:
                        self._finish(index, "length", kv, pending.record, at_ns)
                pending.record.async_results.append(EngineAsyncResult(pending.runner_ticket, sample,
                    self.requests[index].request_id, token, pending.sample_generations[s], disposition, at_ns))
            if self.requests[index].state == DRAINING and self.requests[index].pending_uses == 0:
                self._release(index, kv)
                self.requests[index].state = FINISHED
        pending.record.execute_end_ns = at_ns
        pending.record.execute_ns = at_ns - pending.record.execute_begin_ns
        self._observe_kv(kv, pending.record, 2, at_ns)
        pending.record.blocks_free = self.blocks.free_blocks()
        for i in range(len(self.requests)):
            if self.requests[i].state == WAITING:
                pending.record.waiting += 1
        self.check(kv)
        pending.record.end_ns = runner.now_ns()
        pending.record.postprocess_ns = pending.record.end_ns - at_ns
        self._observe_kv(kv, pending.record, 3, pending.record.end_ns)
        return pending.record.copy()

    def _fail_async[Runner: AsyncModelRunner](mut self, mut runner: Runner, mut kv: KVPool,
                                            mut record: EngineStep) raises:
        self.failed = True
        # If drain itself fails, retain ownership in the invalid process. It
        # cannot be reused, and device-loss recovery belongs to a supervisor.
        runner.drain()
        self.async_pending.clear()
        for i in range(len(self.requests)):
            self.requests[i].pending_uses = 0
            self.requests[i].pending_samples = 0
            if self.requests[i].state == DRAINING:
                self._release(i, kv)
                self.requests[i].state = FINISHED
            elif self.requests[i].state != FINISHED:
                self._finish(i, "error", kv, record, runner.now_ns())
        self.failure_events = record.events.copy()

    def step_async[Runner: AsyncModelRunner](mut self, mut runner: Runner, mut kv: KVPool) raises -> EngineStep:
        """Queue one successor before collecting the oldest step's exact result.

        The public boundary has at most one pending ticket, and submission may
        temporarily raise the bound to two. Abort suppresses uncollected output
        at this boundary. Incremental pressure first retires the outstanding
        ticket, then a later call can safely preempt/replay through the common
        scheduler. Lifetime reservations keep the normal pipeline resident.
        """
        if self.failed or kv.blocks != self.blocks.blocks or kv.block_size != self.blocks.block_size:
            raise Error("failed engine or incompatible pool")
        self.check(kv)
        self.async_enabled = True
        var failure_record = EngineStep(self.step_id, self.max_requests + 2 * self.max_sequences)
        try:
            # Preserve terminal boundary events even if a later submit/collect
            # fails after a waiting request has already become FINISHED.
            self._boundary(kv, failure_record, runner.now_ns())
            if len(self.async_pending) == 0:
                var first = self._submit_async(runner, kv, -1, List[Int](), List[Int](), 0)
                if first.runner_ticket < 0:
                    first.record.events = failure_record.events.copy()
                    first.record.finished += failure_record.finished
                    first.record.aborted += failure_record.aborted
                    first.record.end_ns = runner.now_ns()
                    first.record.blocks_free = self.blocks.free_blocks()
                    for i in range(len(self.requests)):
                        if self.requests[i].state == WAITING:
                            first.record.waiting += 1
                    self.check(kv)
                    return first.record.copy()
                self.async_pending.append(first^)
            var current = self.async_pending.pop()
            for event in failure_record.events:
                current.record.events.append(event.copy())
            current.record.finished += failure_record.finished
            current.record.aborted += failure_record.aborted
            failure_record = current.record.copy()
            if self._pressure_requires_drain():
                current.record.async_pressure_drain = 1
            else:
                var successor = self._submit_async(runner, kv, current.runner_ticket,
                                                  current.selected, current.sample_indices, 1)
                if successor.runner_ticket >= 0:
                    # Expose the actual submission order in this host call,
                    # then erase these observations from the successor record.
                    for observation in successor.record.async_submissions:
                        current.record.async_submissions.append(observation)
                    for head in successor.record.async_heads:
                        current.record.async_heads.append(head)
                    successor.record.async_submissions.clear()
                    successor.record.async_heads.clear()
                    self.async_pending.append(successor^)
                else:
                    self.step_id -= 1
            failure_record = current.record.copy()
            return self._retire_async(runner, kv, current^)
        except error:
            self._fail_async(runner, kv, failure_record)
            raise error

    def drain_async[Runner: AsyncModelRunner](mut self, mut runner: Runner, mut kv: KVPool) raises -> List[EngineStep]:
        """Retire outstanding output without submitting any successor."""
        var records = List[EngineStep](capacity=2)
        while len(self.async_pending) > 0:
            var pending = self.async_pending.pop()
            var failure_record = EngineStep(self.step_id, self.max_requests + 2 * self.max_sequences)
            try:
                self._boundary(kv, failure_record, runner.now_ns())
                for event in failure_record.events:
                    pending.record.events.append(event.copy())
                pending.record.finished += failure_record.finished
                pending.record.aborted += failure_record.aborted
                failure_record = pending.record.copy()
                records.append(self._retire_async(runner, kv, pending^))
            except error:
                self._fail_async(runner, kv, failure_record)
                raise error
        self.check(kv)
        return records^

    def _observe_kv(self, kv: KVPool, mut record: EngineStep, phase: Int, at_ns: Int):
        """Read existing host metadata only; submit and synchronize no work."""
        if not self.observe_kv:
            return
        var written_blocks = 0
        var written_tokens = 0
        for written in kv.written:
            written_blocks += 1 if written > 0 else 0
            written_tokens += written
        var reserved_tokens = 0
        for s in range(len(self.blocks.active)):
            if self.blocks.active[s]:
                reserved_tokens += self.blocks.reserved[s]
        var waiting = 0
        var residents = 0
        for i in range(len(self.requests)):
            waiting += 1 if self.requests[i].state == WAITING else 0
            residents += 1 if self._held(i) else 0
        record.kv_observations.append(EngineKVObservation(phase, at_ns,
            self.blocks.blocks - self.blocks.free_blocks(), written_blocks, written_tokens,
            reserved_tokens, waiting, residents))

    def step[Runner: ModelRunner](mut self, mut runner: Runner, mut kv: KVPool) raises -> EngineStep:
        if self.failed or self.async_enabled or kv.blocks != self.blocks.blocks or kv.block_size != self.blocks.block_size:
            raise Error("failed engine or incompatible pool")
        self.check(kv)
        var record = EngineStep(self.step_id, self.max_requests + 2 * self.max_sequences, self.observe_kv)
        record.begin_ns = runner.now_ns()
        self._observe_kv(kv, record, 0, record.begin_ns)
        self.step_id += 1
        for i in range(len(self.requests)):
            if self.requests[i].state != FINISHED and self.requests[i].arrival_ns < 0:
                self.requests[i].arrival_ns = record.begin_ns
            if self.requests[i].state != FINISHED and self.requests[i].abort_requested:
                self._finish(i, "abort", kv, record, record.begin_ns)
            elif self.requests[i].state != FINISHED and self.requests[i].maximum == 0:
                self._finish(i, "length", kv, record, record.begin_ns)
        var selection = self._schedule(runner, kv, record)
        var selected = selection.selected.copy()
        var counts = selection.counts.copy()
        record.schedule_ns = runner.now_ns() - record.begin_ns
        self._observe_kv(kv, record, 1, record.begin_ns + record.schedule_ns)
        var build_begin = runner.now_ns()
        var postprocess_begin = build_begin
        if len(selected) > 0:
            var input = self._batch(selected, counts, List[Int](), List[Int]())
            var batch = self._copy_batch(input.batch)
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
            if self.observe_kv:
                record.execute_begin_ns = execute_begin
                record.execute_end_ns = postprocess_begin
            self._observe_kv(kv, record, 2, postprocess_begin)
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
        if len(selected) == 0:
            if self.observe_kv:
                record.execute_begin_ns = postprocess_begin
                record.execute_end_ns = postprocess_begin
            self._observe_kv(kv, record, 2, postprocess_begin)
        record.blocks_free = self.blocks.free_blocks()
        for i in range(len(self.requests)):
            if self.requests[i].state == WAITING:
                record.waiting += 1
        self.check(kv)
        record.end_ns = runner.now_ns()
        record.postprocess_ns = record.end_ns - postprocess_begin
        self._observe_kv(kv, record, 3, record.end_ns)
        return record^
