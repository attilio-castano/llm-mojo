"""One request engine chat; weights resident, complete history recomputed per turn.

Terminal EngineCore requests release all logical KV ownership. ChatHistory
retains exact token IDs and framing; it does not claim persistent conversation KV.
Execution failure invalidates this process, including reset.
"""
from llm_mojo.models.qwen2.chat import ChatHistory
from llm_mojo.models.qwen2.model import VOCABULARY
from llm_mojo.models.qwen2.plan import MAX_CONTEXT, KV_BLOCK_SIZE
from llm_mojo.models.qwen2.runner import EngineChatRunner, QwenRunner
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.serving.engine import EngineCore, EngineStep, TOKEN_EVENT, FINISH_EVENT
from llm_mojo.serving.kv_pool import KVPool


struct EngineChatSession[Runner: EngineChatRunner = QwenRunner](Movable):
    var runner: Self.Runner
    var kv: KVPool
    var engine: EngineCore
    var history: ChatHistory
    var request_id: Int
    var turn_rows: Int
    var turn_steps: Int
    var submitted_before: Int

    def __init__(out self, prepared: String, tokenizer: Tokenizer,
                 mut work: TokenizerWorkspace, system: String,
                 chunk_rows: Int = 256, capacity: Int = MAX_CONTEXT) raises:
        self.history = ChatHistory(tokenizer,work,system)
        if (capacity < 1 or capacity > MAX_CONTEXT or chunk_rows < 1
                or chunk_rows > capacity or len(self.history.tokens)+3 >= capacity):
            raise Error("invalid engine chat capacity or system message")
        self.runner = Self.Runner.create_chat_runner(prepared,capacity,chunk_rows)
        var blocks = (capacity+KV_BLOCK_SIZE-1)//KV_BLOCK_SIZE
        self.kv = self.runner.create_chat_kv(capacity)
        self.engine = EngineCore(blocks,KV_BLOCK_SIZE,capacity,VOCABULARY,
                                 token_budget=chunk_rows,max_sequences=1,max_requests=1,
                                 reserve_lifetime=True)
        self.request_id = 0
        self.turn_rows = 0
        self.turn_steps = 0
        self.submitted_before = 0

    def check_drained(self) raises:
        self.engine.check(self.kv)
        if self.engine.live() != 0 or self.engine.blocks.free_blocks() != self.kv.blocks:
            raise Error("engine chat terminal retained request or block ownership")
        for written in self.kv.written:
            if written != 0:
                raise Error("engine chat terminal retained valid KV rows")

    def begin(mut self, tokenizer: Tokenizer, mut work: TokenizerWorkspace,
              message: String, maximum: Int) raises:
        if self.engine.failed or not self.runner.chat_valid():
            raise Error("Engine execution failed: restart the chat process.")
        self.check_drained()
        var previous = self.history.tokens.copy()
        var old_maximum = self.history.maximum
        var old_generated = self.history.generated
        var old_reason = self.history.reason
        # ChatHistory validates context before mutation, reserving closure too.
        self.history.begin(tokenizer,work,message,maximum,self.engine.max_context)
        var stops: List[Int] = [151643,151645]
        try:
            _ = self.engine.add(self.request_id,self.history.tokens,maximum,stops)
        except error:
            self.history.tokens = previous^
            self.history.maximum = old_maximum
            self.history.generated = old_generated
            self.history.generating = False
            self.history.reason = old_reason
            raise error
        self.turn_rows = 0
        self.turn_steps = 0
        self.submitted_before = self.runner.chat_submitted_rows()

    def abort(mut self):
        self.engine.abort(self.request_id)

    def step(mut self) raises -> EngineStep:
        var record = self.engine.step(self.runner,self.kv)
        self.turn_rows += record.total_tokens
        self.turn_steps += 1
        for event in record.events:
            if event.request_id != self.request_id:
                raise Error("engine chat delivered another request's event")
            if event.kind == TOKEN_EVENT:
                self.history.accept(event.token_id)
            elif event.kind == FINISH_EVENT:
                if event.reason == "abort":
                    self.history.finish("interrupted")
                elif event.reason == "error":
                    self.history.finish("error")
                elif event.reason != "stop" and event.reason != "length":
                    raise Error("invalid engine chat terminal reason")
        if self.engine.live() == 0:
            self.check_drained()
            if self.history.generating:
                raise Error("engine chat terminal did not finish exact history")
            self.request_id += 1
        return record^

    def reset(mut self) raises:
        if self.engine.failed or not self.runner.chat_valid():
            raise Error("Engine execution failed: restart the chat process.")
        self.check_drained()
        self.history.reset()

    def fail(mut self) raises:
        # Decoder/report failures also drop the sole live request before exit.
        if self.engine.live() > 0 and not self.engine.failed:
            self.abort()
            _ = self.engine.step(self.runner,self.kv)
        self.runner.chat_synchronize()
        self.history.finish("error")
        self.history.reason = "error"
        self.engine.failed = True
        self.runner.chat_invalidate()
        self.check_drained()
