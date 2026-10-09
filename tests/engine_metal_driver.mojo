"""Explicit prepared-checkpoint engine acceptance checks.

This changes a private resident tied-head weight after a successful step. It
does not change prepared files or simulate device loss. The real model submits
the next Metal step, GPU argmax detects a nonfinite logit, and QwenRunner must
drain before EngineCore releases logical ownership and rejects future work.
The optional admission check compares natural greedy histories under pressure,
then checks reserved-request cancellation and numeric fault cleanup.
"""
from std.math import isfinite
from std.memory import bitcast
from std.sys import argv
from std.testing import assert_equal, assert_raises, assert_true
from llm_mojo.layers.decoder_layer import DECODER_FUSED_DECODE, DECODER_MIXED
from llm_mojo.models.qwen2.model import HIDDEN, LAYERS, VOCABULARY
from llm_mojo.models.qwen2.runner import QwenRunner
from llm_mojo.models.qwen2.plan import MAX_CONTEXT, configured_plan
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.engine import EngineCore, PREFILL, DECODE, WAITING, FINISHED, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVPool


def _assert_list(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def numeric_fault_cleanup(path: String, reserve_lifetime: Bool = False) raises:
    var runner = QwenRunner(path, 128, 8, 3)
    print("engine device", runner.ctx.name(), "backend", runner.ctx.api())
    assert_equal(runner.ctx.api(), "metal")
    var pool = KVPool(runner.ctx, 4, 32, runner.model.kv_geometry())
    var engine = EngineCore(4, 32, 128, VOCABULARY, token_budget=8,
                            max_sequences=3, max_requests=3, reserve_lifetime=reserve_lifetime)
    var a = engine.add(0, [1, 2, 3], 3, List[Int]())
    var first = engine.step(runner, pool)
    assert_equal(first.total_tokens, 3)
    assert_equal(first.prefill_tokens, 3)
    assert_equal(len(first.events), 1)
    assert_equal(first.events[0].kind, TOKEN_EVENT)
    assert_equal(engine.requests[a].state, DECODE)
    assert_equal(engine.requests[a].generated, 1)
    assert_equal(runner.model.valid, True)
    assert_equal(runner.model.submitted_rows, 3 * LAYERS)
    var history = engine.requests[a].tokens.copy()
    var prompt: List[Int] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    if reserve_lifetime:
        # A lifetime reservation includes a second, unwritten physical block.
        prompt = List[Int]()
        for i in range(34):
            prompt.append(100 + i)
    var b = engine.add(1, prompt, 2, List[Int]())
    var c = engine.add(2, [20, 21], 2, List[Int]())
    assert_equal(engine.requests[b].state, WAITING)
    assert_equal(engine.requests[c].state, WAITING)
    # The tied vocabulary weight is used by both embedding and the LM head.
    # Choose a row absent from the pending input so the deliberate NaN reaches
    # the vocabulary projection without corrupting the decoder inputs.
    var poisoned_token = 0 if history[len(history) - 1] != 0 else 15
    with runner.model.embedding.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=poisoned_token * HIDDEN] = bitcast[DType.bfloat16](UInt16(0x7fc0))
    var submitted = runner.model.submitted_rows
    with assert_raises():
        _ = engine.step(runner, pool)
    # The mixed forward enqueued one decode and seven prompt rows through all
    # 24 layers before argmax readback rejected the deliberately nonfinite head.
    assert_equal(runner.model.last_route.configuration, DECODER_MIXED)
    assert_equal(runner.model.submitted_rows - submitted, 8 * LAYERS)
    assert_equal(runner.model.sampled_rows, 1)
    assert_equal(runner.model.valid, False)
    assert_true(engine.failed)
    assert_equal(engine.live(), 0)
    assert_equal(engine.blocks.free_blocks(), 4)
    assert_equal(len(engine.failure_events), 3)
    var finish_counts = List[Int](length=3, fill=0)
    for event in engine.failure_events:
        assert_equal(event.kind, FINISH_EVENT)
        assert_equal(event.reason, "error")
        assert_true(event.request_id >= 0 and event.request_id < 3)
        finish_counts[event.request_id] += 1
        assert_equal(event.generated_tokens, 1 if event.request_id == 0 else 0)
    for count in finish_counts:
        assert_equal(count, 1)
    for slot in [a, b, c]:
        assert_equal(engine.requests[slot].state, FINISHED)
        assert_equal(engine.requests[slot].reason, "error")
        assert_equal(engine.requests[slot].sequence, -1)
    _assert_list(engine.requests[a].tokens, history)
    assert_equal(engine.requests[a].generated, 1)
    assert_equal(engine.requests[b].generated, 0)
    assert_equal(engine.requests[c].generated, 0)
    for written in pool.written:
        assert_equal(written, 0)
    engine.check(pool)
    runner.ctx.synchronize()
    # Invalid engines retain their terminal events and cannot deliver another
    # token or enqueue another forward after ownership has been released.
    var failed_rows = runner.model.submitted_rows
    var failed_step = engine.step_id
    with assert_raises():
        _ = engine.step(runner, pool)
    with assert_raises():
        _ = engine.add(3, [30], 1, List[Int]())
    assert_equal(runner.model.submitted_rows, failed_rows)
    assert_equal(engine.step_id, failed_step)
    assert_equal(len(engine.failure_events), 3)
    engine.check(pool)
    print("engine numeric-fault cleanup passed: submitted rows", failed_rows,
          "terminal requests", len(engine.failure_events), "free blocks", engine.blocks.free_blocks())
    if reserve_lifetime:
        print("engine reserved numeric-fault cleanup passed: unwritten reservation tail released")


def _natural_histories(mut runner: QwenRunner, reserve_lifetime: Bool) raises -> List[List[Int]]:
    runner.model.reset(runner.ctx)
    var pool = KVPool(runner.ctx, 3, 32, runner.model.kv_geometry())
    var engine = EngineCore(3, 32, 128, VOCABULARY, token_budget=8,
                            max_sequences=3, max_requests=3, reserve_lifetime=reserve_lifetime)
    var a = List[Int]()
    var b = List[Int]()
    for i in range(30):
        a.append(i + 1)
    for i in range(38):
        b.append(101 + i)
    _ = engine.add(0, a, 7, List[Int]())
    _ = engine.add(1, b, 3, List[Int]())
    _ = engine.add(2, [301, 302, 303, 304, 305], 3, List[Int]())
    var delivered = List[Int](length=3, fill=0)
    var finishes = List[Int](length=3, fill=0)
    var steps = 0
    var preemptions = 0
    var computed_rows = 0
    while engine.live() > 0 and steps < 1000:
        var record = engine.step(runner, pool)
        preemptions += record.preempted
        computed_rows += record.total_tokens
        for event in record.events:
            var id = event.request_id
            assert_true(id >= 0 and id < 3)
            if event.kind == TOKEN_EVENT:
                assert_equal(finishes[id], 0)
                delivered[id] += 1
                assert_equal(event.generated_tokens, delivered[id])
            else:
                assert_equal(event.kind, FINISH_EVENT)
                assert_equal(event.reason, "length")
                finishes[id] += 1
        engine.check(pool)
        steps += 1
    assert_equal(engine.live(), 0)
    assert_equal(engine.blocks.free_blocks(), 3)
    assert_equal(delivered[0], 7)
    assert_equal(delivered[1], 3)
    assert_equal(delivered[2], 3)
    for finish in finishes:
        assert_equal(finish, 1)
    if reserve_lifetime:
        assert_equal(preemptions, 0)
        assert_equal(computed_rows, 83)
    else:
        assert_true(preemptions > 0)
    var histories = List[List[Int]](capacity=3)
    for i in range(3):
        histories.append(engine.requests[i].tokens.copy())
    for written in pool.written:
        assert_equal(written, 0)
    print("engine admission natural passed:", "reserved" if reserve_lifetime else "incremental",
          "steps", steps, "preemptions", preemptions, "computed rows", computed_rows)
    return histories^


def _reserved_abort_cleanup(mut runner: QwenRunner) raises:
    runner.model.reset(runner.ctx)
    var pool = KVPool(runner.ctx, 3, 32, runner.model.kv_geometry())
    var engine = EngineCore(3, 32, 128, VOCABULARY, token_budget=8,
                            max_sequences=3, max_requests=2, reserve_lifetime=True)
    var a = engine.add(0, List[Int](length=30, fill=42), 7, List[Int]())
    var b = engine.add(1, List[Int](length=38, fill=17), 3, List[Int]())
    var partial = engine.step(runner, pool)
    assert_equal(partial.prefill_tokens, 8)
    assert_equal(len(partial.events), 0)
    assert_equal(engine.requests[a].state, PREFILL)
    assert_equal(engine.blocks.reserved[engine.requests[a].sequence], 36)
    assert_equal(engine.blocks.free_blocks(), 1)
    assert_equal(engine.requests[b].state, WAITING)
    assert_equal(engine.requests[b].sequence, -1)
    engine.abort(0)
    engine.abort(0)
    var aborted = engine.step(runner, pool)
    assert_equal(aborted.aborted, 1)
    assert_equal(aborted.finished, 1)
    assert_equal(aborted.prefill_tokens, 8)
    assert_equal(len(aborted.events), 1)
    assert_equal(aborted.events[0].kind, FINISH_EVENT)
    assert_equal(aborted.events[0].request_id, 0)
    assert_equal(aborted.events[0].reason, "abort")
    assert_equal(aborted.events[0].generated_tokens, 0)
    assert_equal(engine.requests[a].state, FINISHED)
    assert_equal(engine.requests[a].sequence, -1)
    assert_equal(engine.requests[a].generated, 0)
    assert_equal(engine.blocks.length(engine.requests[b].sequence), 8)
    assert_equal(engine.blocks.reserved[engine.requests[b].sequence], 40)
    engine.check(pool)
    engine.abort(0)
    var steps = 0
    var finishes = 0
    while engine.live() > 0 and steps < 1000:
        var record = engine.step(runner, pool)
        for event in record.events:
            assert_equal(event.request_id, 1)
            if event.kind == FINISH_EVENT:
                finishes += 1
                assert_equal(event.reason, "length")
        engine.check(pool)
        steps += 1
    assert_equal(engine.live(), 0)
    assert_equal(finishes, 1)
    assert_equal(engine.requests[b].generated, 3)
    assert_equal(engine.blocks.free_blocks(), 3)
    for written in pool.written:
        assert_equal(written, 0)
    print("engine reserved abort cleanup passed: cached and unwritten reservation blocks released")


def admission(path: String) raises:
    var runner = QwenRunner(path, 128, 8, 3)
    print("engine device", runner.ctx.name(), "backend", runner.ctx.api())
    assert_equal(runner.ctx.api(), "metal")
    var incremental = _natural_histories(runner, False)
    var reserved = _natural_histories(runner, True)
    assert_equal(len(incremental), 3)
    assert_equal(len(reserved), 3)
    for i in range(3):
        _assert_list(incremental[i], reserved[i])
    print("engine admission equality passed: histories 3 generated tokens 13")
    _reserved_abort_cleanup(runner)
    numeric_fault_cleanup(path, True)


# This development-only gate compares identical histories and configurations.
# It never requires configuration 26 to equal configuration 27.
comptime QUAL_CONTEXT = 128
comptime QUAL_ROWS = 256
comptime QUAL_SEQUENCES = 8
comptime QUAL_POISON = 123


struct FastChecks(ImplicitlyCopyable, Movable):
    var checked: Int
    var mismatches: Int
    var unexpected_nonfinite: Int

    def __init__(out self):
        self.checked = 0
        self.mismatches = 0
        self.unexpected_nonfinite = 0

    def equal(mut self, actual: Int, expected: Int, label: String) raises:
        self.checked += 1
        if actual != expected:
            self.mismatches += 1
            raise Error(label + ": got " + String(actual) + ", expected " + String(expected))

    def truth(mut self, actual: Bool, label: String) raises:
        self.equal(1 if actual else 0, 1, label)

    def reason(mut self, actual: String, expected: String) raises:
        self.checked += 1
        if actual != expected:
            self.mismatches += 1
            raise Error("terminal reason " + actual + ", expected " + expected)

    def element(mut self, actual: Scalar[DType.bfloat16], expected: Scalar[DType.bfloat16],
                label: String) raises:
        self.checked += 1
        if not isfinite(actual.cast[DType.float32]()) or not isfinite(expected.cast[DType.float32]()):
            self.unexpected_nonfinite += 1
            raise Error(label + ": unexpected nonfinite value")
        if bitcast[DType.uint16](actual) != bitcast[DType.uint16](expected):
            self.mismatches += 1
            raise Error(label + ": BF16 bytes differ")

    def finite(mut self, actual: Scalar[DType.bfloat16], label: String) raises:
        self.checked += 1
        if not isfinite(actual.cast[DType.float32]()):
            self.unexpected_nonfinite += 1
            raise Error(label + ": unexpected nonfinite value")

    def emit(self, name: String) raises:
        assert_true(self.checked > 0)
        assert_equal(self.mismatches, 0)
        assert_equal(self.unexpected_nonfinite, 0)
        print("check",name,"checked",self.checked,"mismatches",self.mismatches,
              "unexpected-nonfinite",self.unexpected_nonfinite)


def _qualification_configuration(fast_decode: Bool) -> Int:
    return DECODER_FUSED_DECODE if fast_decode else DECODER_MIXED


def _qualification_token(sequence: Int, position: Int) -> Int:
    return (position*103+sequence*17+42)%151643


def _qualification_ids(sequence: Int, start: Int, rows: Int) -> List[Int]:
    var result = List[Int](capacity=rows)
    for position in range(start,start+rows):
        result.append(_qualification_token(sequence,position))
    return result^


def _qualification_table(sequence: Int, sequences: Int) -> List[Int]:
    # Interleaved reverse tables include unused future blocks.
    var result = List[Int](capacity=4)
    for block in range(4):
        result.append(4*sequences-1-(block*sequences+sequence))
    return result^


def _qualification_pool_equal(paged: KVPool, solo: KVPool, sequences: Int,
                              lengths: List[Int], mut check: FastChecks) raises:
    check.equal(paged.geometry.layers,LAYERS,"checkpoint layer count")
    check.equal(solo.geometry.layers,LAYERS,"solo checkpoint layer count")
    # Mapping happens only in this untimed numerical gate after both runners
    # have completed. The comparison traverses every layer and every K/V row.
    with paged.storage.map_to_host() as a:
        with solo.storage.map_to_host() as b:
            for sequence in range(sequences):
                var table = _qualification_table(sequence,sequences)
                for layer in range(LAYERS):
                    for kind in range(2):
                        for block in range(4):
                            var base = paged.key_offset(table[block],layer)+kind*paged.region()
                            var one = solo.key_offset(sequence,layer)+kind*solo.region()
                            for slot in range(32):
                                var position = block*32+slot
                                for head in range(2):
                                    var offset = (head*32+slot)*64 if paged.head_major else (slot*2+head)*64
                                    for dimension in range(64):
                                        var actual = a.unsafe_ptr()[unsafe_offset=base+offset+dimension]
                                        var expected = b.unsafe_ptr()[unsafe_offset=one+(position*2+head)*64+dimension]
                                        if position < lengths[sequence]:
                                            check.element(actual,expected,"same-route all-layer KV")
                                        else:
                                            check.element(actual,Scalar[DType.bfloat16](QUAL_POISON),"unwritten paged KV")
                                            check.element(expected,Scalar[DType.bfloat16](QUAL_POISON),"unwritten solo KV")
            for spare in range(4*sequences,paged.blocks):
                for layer in range(LAYERS):
                    for kind in range(2):
                        var base = paged.key_offset(spare,layer)+kind*paged.region()
                        for i in range(paged.region()):
                            check.element(a.unsafe_ptr()[unsafe_offset=base+i],
                                          Scalar[DType.bfloat16](QUAL_POISON),"spare KV block")


def _qualification_logits(mut batched: QwenRunner, sequence: Int,
                          mut solo: QwenRunner, mut check: FastChecks) raises:
    with batched.model.logits.map_to_host() as a:
        with solo.model.logits.map_to_host() as b:
            for i in range(VOCABULARY):
                check.element(a.unsafe_ptr()[unsafe_offset=sequence*VOCABULARY+i],
                              b.unsafe_ptr()[unsafe_offset=i],"same-route checkpoint logits")


def _qualification_finite_logits(mut runner: QwenRunner, rows: Int, mut check: FastChecks) raises:
    with runner.model.logits.map_to_host() as a:
        for i in range(rows*VOCABULARY):
            check.finite(a.unsafe_ptr()[unsafe_offset=i],"checkpoint logits")


def _qualification_head_untouched(mut runner: QwenRunner, mut check: FastChecks) raises:
    with runner.model.logits.map_to_host() as a:
        for i in range(len(runner.model.logits)):
            check.element(a.unsafe_ptr()[unsafe_offset=i],Scalar[DType.bfloat16](QUAL_POISON),"zero-head logits")
    with runner.model.normalized.map_to_host() as a:
        for i in range(len(runner.model.normalized)):
            check.element(a.unsafe_ptr()[unsafe_offset=i],Scalar[DType.bfloat16](QUAL_POISON),"zero-head final norm")


def _qualification_drained(engine: EngineCore, pool: KVPool, mut check: FastChecks) raises:
    engine.check(pool)
    check.equal(engine.live(),0,"terminal live requests")
    check.equal(engine.blocks.free_blocks(),pool.blocks,"terminal block release")
    for written in pool.written:
        check.equal(written,0,"terminal valid KV extent")
    for i in range(len(engine.requests)):
        check.equal(engine.requests[i].state,FINISHED,"terminal request state")
        check.equal(engine.requests[i].sequence,-1,"terminal ownership")


def _qualification_singleton(mut runner: QwenRunner, mut check: FastChecks) raises:
    runner.model.reset(runner.ctx)
    var pool = KVPool(runner.ctx,4,32,runner.model.kv_geometry())
    var engine = EngineCore(4,32,QUAL_CONTEXT,VOCABULARY,token_budget=1,
                            max_sequences=1,max_requests=1,reserve_lifetime=True)
    runner.model.logits.enqueue_fill(QUAL_POISON)
    runner.model.normalized.enqueue_fill(QUAL_POISON)
    var slot = engine.add(0,[42,43],2,List[Int]())
    var partial = engine.step(runner,pool)
    check.equal(partial.total_tokens,1,"singleton partial rows")
    check.equal(partial.prefill_tokens,1,"singleton partial prefill")
    check.equal(len(partial.events),0,"singleton partial samples")
    check.equal(runner.model.last_route.configuration,DECODER_MIXED,"singleton partial route")
    check.equal(runner.model.sampled_rows,0,"singleton partial selected logits")
    check.equal(engine.requests[slot].generated,0,"singleton partial generated")
    _qualification_head_untouched(runner,check)
    var final_prefill = engine.step(runner,pool)
    check.equal(final_prefill.prefill_tokens,1,"singleton final prefill")
    check.equal(len(final_prefill.events),1,"singleton final token event")
    check.equal(final_prefill.events[0].kind,TOKEN_EVENT,"singleton final event kind")
    check.equal(runner.model.last_route.configuration,_qualification_configuration(runner.fast_decode),
                "singleton final-prefill route")
    check.equal(runner.model.sampled_rows,1,"singleton final selected logits")
    _qualification_finite_logits(runner,1,check)
    var decode = engine.step(runner,pool)
    check.equal(decode.decode_seqs,1,"singleton decode count")
    check.equal(decode.total_tokens,1,"singleton decode rows")
    check.equal(decode.finished,1,"singleton terminal count")
    check.equal(runner.model.last_route.configuration,_qualification_configuration(runner.fast_decode),
                "singleton decode route")
    check.reason(engine.requests[slot].reason,"length")
    check.equal(engine.requests[slot].generated,2,"singleton total outputs")
    check.equal(runner.model.submitted_rows,3*LAYERS,"singleton submitted layer rows")
    _qualification_finite_logits(runner,1,check)
    _qualification_drained(engine,pool,check)


def _qualification_prefill(mut batched: QwenRunner, mut solo: QwenRunner,
                           mut paged: KVPool, mut one: KVPool, prefixes: List[Int],
                           mut check: FastChecks) raises:
    for s in range(len(prefixes)):
        var a = StepBatch.sequence(_qualification_ids(s,0,prefixes[s]),0,
                                   _qualification_table(s,len(prefixes)),32)
        a.logits_rows = List[Int]()
        check.equal(len(batched.execute(a,paged)),0,"hybrid prefix selected rows")
        check.equal(batched.model.last_route.configuration,DECODER_MIXED,"hybrid prefix route")
        var b = StepBatch.sequence(_qualification_ids(s,0,prefixes[s]),0,[s],QUAL_CONTEXT)
        b.logits_rows = List[Int]()
        solo.model.forward(solo.ctx,b,one,configured_plan(DECODER_MIXED,prefixes[s],prefixes[s]))
        check.equal(len(solo.model.greedy_tokens(solo.ctx)),0,"solo prefix selected rows")


def _qualification_decode(prefixes: List[Int], step: Int) raises -> StepBatch:
    var ids = List[Int]()
    var positions = List[Int]()
    var starts = List[Int]()
    var lengths = List[Int]()
    var tables = List[Int]()
    var slots = List[Int]()
    var heads = List[Int]()
    for s in range(len(prefixes)):
        var p = prefixes[s]+step
        var table = _qualification_table(s,len(prefixes))
        ids.append(_qualification_token(s,p))
        positions.append(p)
        starts.append(s)
        lengths.append(p+1)
        tables.extend(table.copy())
        slots.append(table[p//32]*32+p%32)
        heads.append(s)
    starts.append(len(prefixes))
    return StepBatch(ids^,positions^,starts^,len(prefixes),lengths^,4,tables^,slots^,heads^)


def _qualification_batched[HEAD_MAJOR: Bool](mut batched: QwenRunner, mut solo: QwenRunner,
                                           mut check: FastChecks) raises:
    var boundaries: List[Int] = [31,32,33,63,64,65]
    for sequences in [1,2,3,8]:
        # Every boundary appears at every requested sequence count. The 8-row
        # cell repeats 31 and 32 after covering all six boundary values.
        for cell in range((len(boundaries)+sequences-1)//sequences):
            batched.model.reset(batched.ctx)
            solo.model.reset(solo.ctx)
            var paged = KVPool(batched.ctx,4*sequences+2,32,batched.model.kv_geometry(),HEAD_MAJOR)
            var one = KVPool(solo.ctx,sequences,QUAL_CONTEXT,solo.model.kv_geometry())
            paged.storage.enqueue_fill(QUAL_POISON)
            one.storage.enqueue_fill(QUAL_POISON)
            var prefixes = List[Int]()
            for s in range(sequences):
                prefixes.append(boundaries[(cell*sequences+s)%len(boundaries)])
            _qualification_prefill(batched,solo,paged,one,prefixes,check)
            _qualification_pool_equal(paged,one,sequences,prefixes,check)
            for step in range(2):
                var lengths = List[Int]()
                for prefix in prefixes:
                    lengths.append(prefix+step+1)
                var batch = _qualification_decode(prefixes,step)
                var tokens = batched.execute(batch,paged)
                var configuration = _qualification_configuration(batched.fast_decode)
                check.equal(batched.model.last_route.configuration,configuration,"eligible batched route")
                check.equal(batched.model.last_route.sequences,sequences,"eligible batched sequences")
                check.equal(batched.model.sampled_rows,sequences,"eligible batched selected rows")
                check.equal(len(tokens),sequences,"eligible batched token count")
                for s in range(sequences):
                    var p = prefixes[s]+step
                    solo.model.forward(solo.ctx,StepBatch.sequence([_qualification_token(s,p)],p,[s],QUAL_CONTEXT),one,
                                       configured_plan(configuration,1,p+1))
                    var selected = solo.model.greedy_tokens(solo.ctx)
                    check.equal(len(selected),1,"eligible solo token count")
                    check.equal(tokens[s],selected[0],"same-route checkpoint greedy token")
                    _qualification_logits(batched,s,solo,check)
                _qualification_pool_equal(paged,one,sequences,lengths,check)


def _qualification_mixed[HEAD_MAJOR: Bool](mut batched: QwenRunner, mut solo: QwenRunner,
                                         mut check: FastChecks) raises:
    batched.model.reset(batched.ctx)
    solo.model.reset(solo.ctx)
    var paged = KVPool(batched.ctx,10,32,batched.model.kv_geometry(),HEAD_MAJOR)
    var one = KVPool(solo.ctx,2,QUAL_CONTEXT,solo.model.kv_geometry())
    paged.storage.enqueue_fill(QUAL_POISON)
    one.storage.enqueue_fill(QUAL_POISON)
    _qualification_prefill(batched,solo,paged,one,[31,32],check)
    var a = _qualification_table(0,2)
    var b = _qualification_table(1,2)
    var tables = a.copy()
    tables.extend(b.copy())
    var mixed = StepBatch([_qualification_token(0,31),_qualification_token(1,32),
                           _qualification_token(1,33),_qualification_token(1,34)],
                          [31,32,33,34],[0,1,4],1,[32,35],4,tables.copy(),
                          [a[0]*32+31,b[1]*32,b[1]*32+1,b[1]*32+2],[0,3])
    var tokens = batched.execute(mixed,paged)
    check.equal(batched.model.last_route.configuration,DECODER_MIXED,"multirow fallback route")
    check.equal(len(tokens),2,"multirow selected tokens")
    for s in range(2):
        var past = 31 if s == 0 else 32
        var count = 1 if s == 0 else 3
        solo.model.forward(solo.ctx,StepBatch.sequence(_qualification_ids(s,past,count),past,[s],QUAL_CONTEXT),one,
                           configured_plan(DECODER_MIXED,count,past+count))
        var selected = solo.model.greedy_tokens(solo.ctx)
        check.equal(tokens[s],selected[0],"mixed fallback greedy token")
        _qualification_logits(batched,s,solo,check)
    _qualification_pool_equal(paged,one,2,[32,35],check)
    # The second singleton is unfinished: one selected head for two queries.
    var subset = StepBatch([_qualification_token(0,32),_qualification_token(1,35)],
                           [32,35],[0,1,2],2,[33,36],4,tables.copy(),[a[1]*32,b[1]*32+3],[0])
    var subset_tokens = batched.execute(subset,paged)
    check.equal(batched.model.last_route.configuration,DECODER_MIXED,"incomplete singleton batch fallback")
    check.equal(batched.model.sampled_rows,1,"incomplete singleton selected rows")
    check.equal(len(subset_tokens),1,"incomplete singleton token count")
    for s in range(2):
        var past = 32 if s == 0 else 35
        var batch = StepBatch.sequence(_qualification_ids(s,past,1),past,[s],QUAL_CONTEXT)
        if s == 1:
            batch.logits_rows = List[Int]()
        solo.model.forward(solo.ctx,batch,one,configured_plan(DECODER_MIXED,1,past+1))
        var selected = solo.model.greedy_tokens(solo.ctx)
        check.equal(len(selected),1 if s == 0 else 0,"incomplete solo selected rows")
        if s == 0:
            check.equal(subset_tokens[0],selected[0],"incomplete singleton greedy token")
            _qualification_logits(batched,0,solo,check)
    _qualification_pool_equal(paged,one,2,[33,36],check)


def _qualification_terminals(mut runner: QwenRunner, mut check: FastChecks) raises:
    runner.model.reset(runner.ctx)
    var probe = KVPool(runner.ctx,4,32,runner.model.kv_geometry())
    var stop_prompt: List[Int] = [42]
    var prompt: List[Int] = [42,17,91]
    var first = runner.execute(StepBatch.sequence(stop_prompt,0,[0],32),probe)
    check.equal(len(first),1,"stop probe token count")
    check.equal(runner.model.last_route.configuration,_qualification_configuration(runner.fast_decode),
                "stop probe eligible singleton route")
    _qualification_finite_logits(runner,1,check)
    runner.model.reset(runner.ctx)
    var pool = KVPool(runner.ctx,4,32,runner.model.kv_geometry())
    var engine = EngineCore(4,32,QUAL_CONTEXT,VOCABULARY,token_budget=256,
                            max_sequences=8,max_requests=1,reserve_lifetime=True)
    var slot = engine.add(0,stop_prompt,3,[first[0]])
    var stopped = engine.step(runner,pool)
    check.equal(stopped.total_tokens,1,"stop prompt rows")
    check.equal(runner.model.last_route.configuration,_qualification_configuration(runner.fast_decode),
                "stop completion eligible singleton route")
    check.equal(stopped.finished,1,"stop terminal count")
    check.equal(stopped.events[0].kind,TOKEN_EVENT,"stop token precedes finish")
    check.equal(stopped.events[0].token_id,first[0],"actual stop token")
    check.reason(engine.requests[slot].reason,"stop")
    check.equal(engine.requests[slot].generated,1,"stop outputs")
    _qualification_drained(engine,pool,check)
    slot = engine.add(1,prompt,3,List[Int]())
    var rows = 0
    var finishes = 0
    var steps = 0
    while engine.live() > 0 and steps < 8:
        var record = engine.step(runner,pool)
        rows += record.total_tokens
        finishes += record.finished
        _qualification_finite_logits(runner,runner.model.sampled_rows,check)
        steps += 1
    check.equal(rows,5,"normal turn P+G-1 rows")
    check.equal(finishes,1,"length terminal count")
    check.reason(engine.requests[slot].reason,"length")
    check.equal(engine.requests[slot].generated,3,"length outputs")
    _qualification_drained(engine,pool,check)
    var old_ticket = engine.requests[slot].ticket
    slot = engine.add(2,prompt,1,List[Int]())
    check.equal(slot,0,"finished request slot reused")
    check.equal(engine.requests[slot].ticket,old_ticket+1,"fresh request ticket")
    var reused = engine.step(runner,pool)
    check.equal(reused.finished,1,"reused slot terminal count")
    check.equal(reused.events[0].request_id,2,"reused slot request identity")
    check.equal(reused.total_tokens,3,"reused full prompt rows")
    _qualification_drained(engine,pool,check)
    var submitted = runner.model.submitted_rows
    slot = engine.add(3,prompt,0,List[Int]())
    var zero = engine.step(runner,pool)
    check.equal(zero.total_tokens,0,"zero-output execution rows")
    check.equal(zero.finished,1,"zero-output terminal count")
    check.equal(runner.model.submitted_rows,submitted,"zero-output enqueued no rows")
    check.equal(engine.requests[slot].generated,0,"zero-output generated")
    _qualification_drained(engine,pool,check)
    # One resident partial prompt and one waiting request. Waiting abort occurs
    # while the resident progresses; then repeated resident abort releases its
    # future unused reservation without another model execution.
    runner.model.reset(runner.ctx)
    var abort_pool = KVPool(runner.ctx,4,32,runner.model.kv_geometry())
    var aborted = EngineCore(4,32,QUAL_CONTEXT,VOCABULARY,token_budget=1,
                             max_sequences=1,max_requests=2,reserve_lifetime=True)
    var a = aborted.add(0,List[Int](length=34,fill=42),2,List[Int]())
    var b = aborted.add(1,List[Int](length=40,fill=17),2,List[Int]())
    var partial = aborted.step(runner,abort_pool)
    check.equal(partial.total_tokens,1,"partial abort setup rows")
    check.equal(aborted.requests[a].state,PREFILL,"partial abort setup phase")
    check.equal(aborted.requests[b].state,WAITING,"waiting abort setup phase")
    check.equal(aborted.requests[b].sequence,-1,"waiting request has no owner")
    aborted.abort(1)
    var waiting = aborted.step(runner,abort_pool)
    check.equal(waiting.aborted,1,"waiting abort count")
    check.equal(waiting.total_tokens,1,"resident progresses on waiting abort")
    check.reason(aborted.requests[b].reason,"abort")
    check.equal(aborted.requests[b].generated,0,"waiting abort delivered no token")
    aborted.abort(0)
    aborted.abort(0)
    submitted = runner.model.submitted_rows
    var released = aborted.step(runner,abort_pool)
    check.equal(released.aborted,1,"partial-prefill abort terminal count")
    check.equal(released.total_tokens,0,"partial-prefill abort executed no rows")
    check.equal(runner.model.submitted_rows,submitted,"partial-prefill abort enqueued no rows")
    _qualification_drained(aborted,abort_pool,check)
    runner.model.reset(runner.ctx)
    var decode_pool = KVPool(runner.ctx,4,32,runner.model.kv_geometry())
    var decode = EngineCore(4,32,QUAL_CONTEXT,VOCABULARY,token_budget=256,
                            max_sequences=8,max_requests=1,reserve_lifetime=True)
    var d = decode.add(0,prompt,4,List[Int]())
    _ = decode.step(runner,decode_pool)
    check.equal(decode.requests[d].state,DECODE,"running decode abort setup")
    var decoded = decode.step(runner,decode_pool)
    check.equal(decoded.total_tokens,1,"decode abort executed singleton rows")
    check.equal(decoded.decode_seqs,1,"decode abort executed sequence count")
    check.equal(runner.model.last_route.configuration,_qualification_configuration(runner.fast_decode),
                "decode abort executed eligible singleton route")
    check.equal(decode.requests[d].generated,2,"decode abort setup delivered outputs")
    _qualification_finite_logits(runner,1,check)
    var history = decode.requests[d].tokens.copy()
    submitted = runner.model.submitted_rows
    decode.abort(0)
    decode.abort(0)
    var cancelled = decode.step(runner,decode_pool)
    check.equal(cancelled.aborted,1,"decode abort terminal count")
    check.equal(cancelled.total_tokens,0,"decode abort executed no rows")
    check.equal(runner.model.submitted_rows,submitted,"decode abort enqueued no rows")
    check.equal(decode.requests[d].generated,2,"decode abort preserves output count")
    check.equal(len(decode.requests[d].tokens),len(history),"decode abort preserves history length")
    for i in range(len(history)):
        check.equal(decode.requests[d].tokens[i],history[i],"decode abort preserves delivered history")
    _qualification_drained(decode,decode_pool,check)


def _qualification_fault(mut runner: QwenRunner, mut check: FastChecks) raises:
    runner.model.reset(runner.ctx)
    var pool = KVPool(runner.ctx,4,32,runner.model.kv_geometry())
    var engine = EngineCore(4,32,QUAL_CONTEXT,VOCABULARY,token_budget=256,
                            max_sequences=8,max_requests=3,reserve_lifetime=True)
    var a = engine.add(0,[42,17,91],4,List[Int]())
    var first = engine.step(runner,pool)
    check.equal(first.total_tokens,3,"fault finite prefix rows")
    _qualification_finite_logits(runner,1,check)
    var history = engine.requests[a].tokens.copy()
    # Four blocks fit alone but not beside the resident's block. Strict FIFO
    # also leaves the following small request waiting, so the failing batch is
    # one fully sampled singleton, with no multi-row tail.
    var b = engine.add(1,List[Int](length=97,fill=100),2,List[Int]())
    var c = engine.add(2,[20,21],2,List[Int]())
    check.equal(engine.requests[b].state,WAITING,"fault waiting long request")
    check.equal(engine.requests[c].state,WAITING,"fault waiting small request")
    var poisoned_token = 0 if history[len(history)-1] != 0 else 15
    check.truth(poisoned_token != history[len(history)-1],"fault row absent from pending input")
    with runner.model.embedding.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=poisoned_token*HIDDEN] = bitcast[DType.bfloat16](UInt16(0x7fc0))
    var submitted = runner.model.submitted_rows
    with assert_raises(contains="nonfinite"):
        _ = engine.step(runner,pool)
    check.checked += 1  # A real assertion of the expected intentional fault.
    check.equal(runner.model.last_route.configuration,_qualification_configuration(runner.fast_decode),
                "fault eligible singleton route")
    check.equal(runner.model.last_route.sequences,1,"fault singleton sequence count")
    check.equal(runner.model.sampled_rows,1,"fault singleton selected logits")
    check.equal(runner.model.submitted_rows-submitted,LAYERS,"fault enqueued all checkpoint layers")
    check.truth(not runner.model.valid,"fault invalidated model")
    check.truth(engine.failed,"fault invalidated engine")
    check.equal(len(engine.failure_events),3,"fault terminal requests")
    var counts = List[Int](length=3,fill=0)
    for event in engine.failure_events:
        check.equal(event.kind,FINISH_EVENT,"fault finish event kind")
        check.reason(event.reason,"error")
        check.truth(event.request_id >= 0 and event.request_id < 3,"fault request identity")
        counts[event.request_id] += 1
        check.equal(event.generated_tokens,1 if event.request_id == 0 else 0,"fault delivered count")
    for count in counts:
        check.equal(count,1,"fault exactly-once terminal result")
    check.equal(len(engine.requests[a].tokens),len(history),"fault preserved history length")
    for i in range(len(history)):
        check.equal(engine.requests[a].tokens[i],history[i],"fault preserved delivered history")
    _qualification_drained(engine,pool,check)
    var failed_rows = runner.model.submitted_rows
    var failed_step = engine.step_id
    with assert_raises():
        _ = engine.step(runner,pool)
    check.checked += 1
    with assert_raises():
        _ = engine.add(3,[30],1,List[Int]())
    check.checked += 1
    check.equal(runner.model.submitted_rows,failed_rows,"failed engine submitted no further rows")
    check.equal(engine.step_id,failed_step,"failed engine advanced no further step")
    _qualification_drained(engine,pool,check)
    # The injected NaN is expected and was rejected. It is not counted as an
    # unexpected finite-boundary failure and no corrupted logits are accepted.


def fast_qualification(path: String, kind: String) raises:
    if kind != "reference" and kind != "fast-decode":
        raise Error("fast-qualification requires reference|fast-decode")
    var runner = QwenRunner(path,MAX_CONTEXT,QUAL_ROWS,QUAL_SEQUENCES,fast_decode=kind == "fast-decode")
    var solo = QwenRunner(path,MAX_CONTEXT,QUAL_ROWS,QUAL_SEQUENCES)
    assert_equal(runner.ctx.name(),"Apple M4 Pro")
    assert_equal(runner.ctx.api(),"metal")
    assert_equal(solo.ctx.name(),runner.ctx.name())
    assert_equal(solo.ctx.api(),runner.ctx.api())
    assert_equal(runner.model.kv_geometry().layers,LAYERS)
    print("device",runner.ctx.name()+"/"+runner.ctx.api())
    print("qualification","engine-fast-checkpoint-v1")
    print("runner",kind)
    var singleton = FastChecks()
    _qualification_singleton(runner,singleton)
    singleton.emit("singleton-partial-zero-head")
    var batched = FastChecks()
    _qualification_batched[False](runner,solo,batched)
    _qualification_batched[True](runner,solo,batched)
    batched.emit("batched-solo-hybrid-kv-logits")
    var mixed = FastChecks()
    _qualification_mixed[False](runner,solo,mixed)
    _qualification_mixed[True](runner,solo,mixed)
    mixed.emit("mixed-fallback")
    var terminals = FastChecks()
    _qualification_terminals(runner,terminals)
    terminals.emit("terminal-reuse")
    var fault = FastChecks()
    _qualification_fault(runner,fault)
    fault.emit("fault-cleanup")


def main() raises:
    var args = argv()
    if len(args) == 2:
        numeric_fault_cleanup(args[1])
    elif len(args) == 3 and args[2] == "admission":
        admission(args[1])
    elif len(args) == 4 and args[2] == "fast-qualification":
        fast_qualification(args[1],args[3])
    else:
        raise Error("usage: engine_metal_driver VERIFIED_PREPARED_PATH [admission|fast-qualification reference|fast-decode]")
