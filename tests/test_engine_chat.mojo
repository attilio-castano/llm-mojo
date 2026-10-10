"""Exact engine chat lifecycle with a scripted runner, without model weights.

These tests call the production EngineChatSession methods. The script advances
logical KV extents in a tiny existing KVPool allocation; it provides no model
numerical or GPU performance evidence. Token framing uses the same verified
checkpoint tokenizer tables as test_chat.mojo.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.chat import ChatHistory, IM_END, NEWLINE
from llm_mojo.models.qwen2.engine_chat import EngineChatSession
from llm_mojo.models.qwen2.model import VOCABULARY
from llm_mojo.models.qwen2.plan import KV_BLOCK_SIZE
from llm_mojo.models.qwen2.runner import EngineChatRunner
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.engine import PREFILL, DECODE, FINISHED, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVGeometry, KVPool
from llm_mojo.serving.runner import SimulatedRunner

comptime TABLES = "build/checkpoints/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775/prepared-v1/tables.bin"


struct ChatScriptRunner(EngineChatRunner):
    var ctx: DeviceContext
    var oracle: SimulatedRunner
    var valid: Bool
    var submitted: Int
    var fail_at_step: Int
    var synchronize_count: Int
    var executed_decodes: Int

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.oracle = SimulatedRunner([42],VOCABULARY)
        self.valid = True
        self.submitted = 0
        self.fail_at_step = -1
        self.synchronize_count = 0
        self.executed_decodes = 0

    @staticmethod
    def create_chat_runner(prepared: String, capacity: Int, chunk_rows: Int) raises -> Self:
        # The path is intentionally unused: this lifecycle oracle has no weights.
        return Self()

    def create_chat_kv(self, capacity: Int) raises -> KVPool:
        var blocks = (capacity+KV_BLOCK_SIZE-1)//KV_BLOCK_SIZE
        return KVPool(self.ctx,blocks,KV_BLOCK_SIZE,KVGeometry(1,1,1))

    def chat_valid(self) -> Bool:
        return self.valid

    def chat_submitted_rows(self) -> Int:
        # Count actual logical rows, rather than Qwen's submitted layer rows.
        return self.submitted

    def chat_device_name(self) -> String:
        return "scripted"

    def chat_device_api(self) -> String:
        return "scripted"

    def chat_synchronize(mut self) raises:
        self.ctx.synchronize()
        self.synchronize_count += 1

    def chat_invalidate(mut self):
        self.valid = False

    def now_ns(self) -> Int:
        return self.oracle.now_ns()

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        if not self.valid:
            raise Error("scripted runner is invalid")
        try:
            var selected = self.oracle.execute(batch,kv)
            self.submitted += batch.rows()
            self.executed_decodes += batch.decode_count
            if self.oracle.steps == self.fail_at_step:
                # Fail after written extents changed. EngineCore must invalidate
                # ownership even when execution has made partial progress.
                raise Error("injected engine chat execution failure")
            return selected^
        except error:
            self.valid = False
            self.ctx.synchronize()
            raise error


comptime ScriptedSession = EngineChatSession[ChatScriptRunner]


def _ids(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual),len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i],expected[i])


def _session(tokenizer: Tokenizer, mut work: TokenizerWorkspace,
             script: List[Int], chunk: Int = 256, capacity: Int = 512) raises -> ScriptedSession:
    var session = ScriptedSession("scripted-lifecycle",tokenizer,work,"test",chunk,capacity)
    session.runner.oracle = SimulatedRunner(script,VOCABULARY)
    return session^


def _closed(prompt: List[Int], reply: List[Int]) -> List[Int]:
    var expected = prompt.copy()
    expected.extend(reply.copy())
    if len(reply) == 0 or reply[len(reply)-1] != IM_END:
        expected.append(IM_END)
    expected.append(NEWLINE)
    return expected^


def _drained(session: ScriptedSession) raises:
    session.check_drained()
    assert_equal(session.engine.live(),0)
    assert_equal(session.engine.blocks.free_blocks(),session.kv.blocks)
    for written in session.kv.written:
        assert_equal(written,0)
    for i in range(len(session.engine.requests)):
        assert_equal(session.engine.requests[i].state,FINISHED)
        assert_equal(session.engine.requests[i].sequence,-1)
    assert_equal(session.history.generating,False)


def _drain(mut session: ScriptedSession) raises -> Int:
    var finishes = 0
    var steps = 0
    var request_id = session.request_id
    while session.history.generating and steps < 1024:
        var record = session.step()
        for event in record.events:
            assert_equal(event.request_id,request_id)
            if event.kind == FINISH_EVENT:
                finishes += 1
        steps += 1
    _drained(session)
    assert_true(steps < 1024)
    assert_equal(finishes,1)
    return steps


struct ChatSnapshot(Movable):
    var tokens: List[Int]
    var maximum: Int
    var generated: Int
    var generating: Bool
    var reason: String
    var request_id: Int
    var turn_rows: Int
    var turn_steps: Int
    var submitted_before: Int
    var submitted: Int
    var runner_steps: Int
    var next_ticket: Int
    var engine_step: Int
    var request_slots: Int

    def __init__(out self, session: ScriptedSession):
        self.tokens = session.history.tokens.copy()
        self.maximum = session.history.maximum
        self.generated = session.history.generated
        self.generating = session.history.generating
        self.reason = session.history.reason.copy()
        self.request_id = session.request_id
        self.turn_rows = session.turn_rows
        self.turn_steps = session.turn_steps
        self.submitted_before = session.submitted_before
        self.submitted = session.runner.chat_submitted_rows()
        self.runner_steps = session.runner.oracle.steps
        self.next_ticket = session.engine.next_ticket
        self.engine_step = session.engine.step_id
        self.request_slots = len(session.engine.requests)

    def unchanged(self, session: ScriptedSession) raises:
        _ids(session.history.tokens,self.tokens)
        assert_equal(session.history.maximum,self.maximum)
        assert_equal(session.history.generated,self.generated)
        assert_equal(session.history.generating,self.generating)
        assert_equal(session.history.reason,self.reason)
        assert_equal(session.request_id,self.request_id)
        assert_equal(session.turn_rows,self.turn_rows)
        assert_equal(session.turn_steps,self.turn_steps)
        assert_equal(session.submitted_before,self.submitted_before)
        assert_equal(session.runner.chat_submitted_rows(),self.submitted)
        assert_equal(session.runner.oracle.steps,self.runner_steps)
        assert_equal(session.engine.next_ticket,self.next_ticket)
        assert_equal(session.engine.step_id,self.engine_step)
        assert_equal(len(session.engine.requests),self.request_slots)


def test_engine_chat_scripted_stop_and_limit_exact_history_and_release() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    for stop in [IM_END,151643]:
        var session = _session(tokenizer,work,[stop],4)
        session.begin(tokenizer,work,"hi",3)
        var prompt = session.history.tokens.copy()
        _ = _drain(session)
        assert_equal(session.history.reason,"stop")
        assert_equal(session.history.generated,1)
        assert_equal(session.engine.requests[0].reason,"stop")
        assert_equal(session.engine.requests[0].generated,1)
        assert_equal(session.turn_rows,len(prompt))
        assert_equal(session.runner.chat_submitted_rows(),len(prompt))
        assert_equal(session.request_id,1)
        _ids(session.history.tokens,_closed(prompt,[stop]))
        var delivered = prompt.copy()
        delivered.append(stop)
        _ids(session.engine.requests[0].tokens,delivered)
    var limited = _session(tokenizer,work,[42],4)
    limited.begin(tokenizer,work,"hi",3)
    var prompt = limited.history.tokens.copy()
    _ = _drain(limited)
    assert_equal(limited.history.reason,"limit")
    assert_equal(limited.history.generated,3)
    assert_equal(limited.engine.requests[0].reason,"length")
    assert_equal(limited.turn_rows,len(prompt)+3-1)
    _ids(limited.history.tokens,_closed(prompt,[42,42,42]))


def test_engine_chat_cancel_after_partial_prefill_releases_future_reservation() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    var session = _session(tokenizer,work,[42],1)
    session.begin(tokenizer,work,"a partial prompt",4)
    var prompt = session.history.tokens.copy()
    assert_true(len(prompt) > 1)
    var partial = session.step()
    assert_equal(partial.total_tokens,1)
    assert_equal(len(partial.events),0)
    assert_equal(session.engine.requests[0].state,PREFILL)
    assert_equal(session.history.generated,0)
    var owned = session.kv.blocks-session.engine.blocks.free_blocks()
    assert_equal(owned,(len(prompt)+4-1+KV_BLOCK_SIZE-1)//KV_BLOCK_SIZE)
    var written = 0
    for extent in session.kv.written:
        written += extent
    assert_equal(written,1)
    assert_true(owned*KV_BLOCK_SIZE > written)
    var submitted = session.runner.chat_submitted_rows()
    var executed = session.runner.oracle.steps
    session.abort()
    session.abort()
    var cancelled = session.step()
    assert_equal(cancelled.aborted,1)
    assert_equal(cancelled.total_tokens,0)
    assert_equal(len(cancelled.events),1)
    assert_equal(cancelled.events[0].kind,FINISH_EVENT)
    assert_equal(cancelled.events[0].reason,"abort")
    assert_equal(session.runner.chat_submitted_rows(),submitted)
    assert_equal(session.runner.oracle.steps,executed)
    assert_equal(session.history.reason,"interrupted")
    assert_equal(session.history.generated,0)
    _ids(session.history.tokens,_closed(prompt,List[Int]()))
    _drained(session)


def test_engine_chat_cancel_after_executed_decode_preserves_delivered_tokens() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    var session = _session(tokenizer,work,[42])
    session.begin(tokenizer,work,"hi",4)
    var prompt = session.history.tokens.copy()
    var prefill = session.step()
    assert_equal(prefill.prefill_tokens,len(prompt))
    assert_equal(prefill.decode_seqs,0)
    assert_equal(session.engine.requests[0].state,DECODE)
    var decoded = session.step()
    assert_equal(decoded.decode_seqs,1)
    assert_equal(decoded.total_tokens,1)
    assert_equal(decoded.events[0].kind,TOKEN_EVENT)
    assert_equal(session.runner.executed_decodes,1)
    assert_equal(session.history.generated,2)
    var delivered = session.engine.requests[0].tokens.copy()
    var submitted = session.runner.chat_submitted_rows()
    var executed = session.runner.oracle.steps
    session.abort()
    session.abort()
    var cancelled = session.step()
    assert_equal(cancelled.aborted,1)
    assert_equal(cancelled.total_tokens,0)
    assert_equal(session.runner.chat_submitted_rows(),submitted)
    assert_equal(session.runner.oracle.steps,executed)
    assert_equal(session.history.reason,"interrupted")
    assert_equal(session.history.generated,2)
    assert_equal(session.turn_rows,len(prompt)+1)
    _ids(session.engine.requests[0].tokens,delivered)
    _ids(session.history.tokens,_closed(prompt,[42,42]))
    _drained(session)


def test_engine_chat_slot_reuse_reset_and_full_history_reprefill() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    var session = _session(tokenizer,work,[42],64,1024)
    var initial = session.history.tokens.copy()
    session.begin(tokenizer,work,"hi",1)
    var first_prompt = session.history.tokens.copy()
    _ = _drain(session)
    var closed = session.history.tokens.copy()
    assert_equal(session.turn_rows,len(first_prompt))
    assert_equal(session.engine.requests[0].ticket,0)
    session.begin(tokenizer,work,"again",2)
    var second_prompt = session.history.tokens.copy()
    assert_true(len(second_prompt) > len(closed))
    for i in range(len(closed)):
        assert_equal(second_prompt[i],closed[i])
    _ids(session.engine.requests[0].tokens,second_prompt)
    assert_equal(len(session.engine.requests),1)
    assert_equal(session.engine.requests[0].ticket,1)
    assert_equal(session.engine.requests[0].sequence,-1)
    assert_equal(session.engine.blocks.free_blocks(),session.kv.blocks)
    _ = _drain(session)
    assert_equal(session.turn_rows,len(second_prompt)+1)
    assert_equal(session.request_id,2)
    var submitted = session.runner.chat_submitted_rows()
    var executed = session.runner.oracle.steps
    session.reset()
    _ids(session.history.tokens,initial)
    assert_equal(session.history.reason,"ready")
    assert_equal(session.history.generated,0)
    assert_equal(session.runner.chat_submitted_rows(),submitted)
    assert_equal(session.runner.oracle.steps,executed)
    assert_true(session.runner.chat_valid())
    _drained(session)
    session.begin(tokenizer,work,"hi",1)
    _ids(session.history.tokens,first_prompt)
    assert_equal(session.engine.requests[0].ticket,2)
    _ = _drain(session)
    assert_equal(session.turn_rows,len(first_prompt))
    assert_equal(session.request_id,3)


def test_engine_chat_context_and_invalid_turn_rejection_are_atomic() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    var session = _session(tokenizer,work,[42])
    session.begin(tokenizer,work,"hi",1)
    _ = _drain(session)
    var previous = ChatSnapshot(session)
    with assert_raises():
        session.begin(tokenizer,work,"",1)
    previous.unchanged(session)
    for maximum in [-1,0,4097,512]:
        with assert_raises():
            session.begin(tokenizer,work,"hi",maximum)
        previous.unchanged(session)
    _drained(session)
    # Derive the exact context boundary from native tokenizer framing.
    var probe = ChatHistory(tokenizer,work,"test")
    probe.begin(tokenizer,work,"hi",1,512)
    var capacity = len(probe.tokens)+1+2
    var exact = _session(tokenizer,work,[42],min(16,capacity),capacity)
    exact.begin(tokenizer,work,"hi",1)
    _ = _drain(exact)
    var short = _session(tokenizer,work,[42],min(16,capacity-1),capacity-1)
    var untouched = ChatSnapshot(short)
    with assert_raises():
        short.begin(tokenizer,work,"hi",1)
    untouched.unchanged(short)
    _drained(short)


def test_engine_chat_add_rejection_rolls_back_completed_history_and_counters() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    var session = _session(tokenizer,work,[42])
    session.begin(tokenizer,work,"hi",1)
    _ = _drain(session)
    var previous = ChatSnapshot(session)
    # Fault injection: a declared vocabulary that excludes the fixed IM_END
    # stop lets ChatHistory.begin succeed, then makes EngineCore.add reject.
    # This targets the adapter's rollback branch, not normal Qwen input.
    session.engine.vocabulary = IM_END
    with assert_raises():
        session.begin(tokenizer,work,"again",2)
    previous.unchanged(session)
    _drained(session)
    session.engine.vocabulary = VOCABULARY
    session.begin(tokenizer,work,"again",2)
    assert_equal(session.engine.requests[0].ticket,previous.next_ticket)
    assert_equal(session.request_id,previous.request_id)
    _ = _drain(session)
    assert_equal(session.history.generated,2)


def test_engine_chat_execution_failure_drops_written_decode_and_requires_restart() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    var session = _session(tokenizer,work,[42])
    session.begin(tokenizer,work,"hi",4)
    var prompt = session.history.tokens.copy()
    _ = session.step()
    assert_equal(session.history.generated,1)
    session.runner.fail_at_step = 2
    with assert_raises(contains="injected engine chat execution failure"):
        _ = session.step()
    assert_equal(session.runner.executed_decodes,1)
    assert_equal(session.runner.oracle.steps,2)
    assert_true(session.engine.failed)
    assert_equal(session.runner.chat_valid(),False)
    assert_equal(len(session.engine.failure_events),1)
    assert_equal(session.engine.failure_events[0].kind,FINISH_EVENT)
    assert_equal(session.engine.failure_events[0].reason,"error")
    assert_equal(session.engine.failure_events[0].generated_tokens,1)
    assert_equal(session.history.generated,1)
    session.fail()
    assert_equal(session.runner.synchronize_count,1)
    assert_equal(session.history.reason,"error")
    _ids(session.history.tokens,_closed(prompt,[42]))
    _drained(session)
    var closed = session.history.tokens.copy()
    var submitted = session.runner.chat_submitted_rows()
    var executed = session.runner.oracle.steps
    for action in [0,1,2]:
        with assert_raises():
            if action == 0:
                session.begin(tokenizer,work,"again",1)
            elif action == 1:
                session.reset()
            else:
                _ = session.step()
    assert_equal(session.runner.chat_submitted_rows(),submitted)
    assert_equal(session.runner.oracle.steps,executed)
    _ids(session.history.tokens,closed)
    session.fail()
    assert_equal(session.runner.synchronize_count,2)
    _ids(session.history.tokens,closed)
    assert_equal(len(session.engine.failure_events),1)
    _drained(session)


def test_engine_chat_output_failure_aborts_live_decode_and_closes_history_once() raises:
    var tokenizer = Tokenizer(String(TABLES))
    var work = TokenizerWorkspace()
    var session = _session(tokenizer,work,[42])
    session.begin(tokenizer,work,"hi",4)
    var prompt = session.history.tokens.copy()
    _ = session.step()
    var decoded = session.step()
    assert_equal(decoded.decode_seqs,1)
    assert_equal(session.history.generated,2)
    var delivered = session.engine.requests[0].tokens.copy()
    var submitted = session.runner.chat_submitted_rows()
    var executed = session.runner.oracle.steps
    session.fail()
    assert_equal(session.runner.synchronize_count,1)
    assert_equal(session.engine.requests[0].reason,"abort")
    assert_true(session.engine.failed)
    assert_equal(session.runner.chat_valid(),False)
    assert_equal(session.history.reason,"error")
    assert_equal(session.history.generated,2)
    assert_equal(session.runner.chat_submitted_rows(),submitted)
    assert_equal(session.runner.oracle.steps,executed)
    _ids(session.engine.requests[0].tokens,delivered)
    _ids(session.history.tokens,_closed(prompt,[42,42]))
    _drained(session)
    var closed = session.history.tokens.copy()
    session.fail()
    _ids(session.history.tokens,closed)
    with assert_raises():
        session.reset()
    _drained(session)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
