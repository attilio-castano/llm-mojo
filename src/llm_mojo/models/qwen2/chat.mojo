"""Qwen text-chat framing and a persistent native batch-one session.

History is authoritative; the length its block manager has committed identifies
the already submitted prefix, whose K/V rows live in the blocks of the session's
table. No rendered-text round trip is used for generated assistant tokens.
"""
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.model import QwenModel, VOCABULARY
from llm_mojo.models.qwen2.plan import KV_BLOCK_SIZE, KV_HEAD_MAJOR, MAX_CONTEXT, fast_plan
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.models.qwen2.tokens import IM_END, is_stop
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.blocks import BlockManager
from llm_mojo.serving.kv_pool import KVPool

comptime DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
comptime NEWLINE = 198


struct ChatHistory(Movable):
    var tokens: List[Int]
    var initial: List[Int]
    var generating: Bool
    var maximum: Int
    var generated: Int
    var reason: String

    def __init__(out self, tokenizer: Tokenizer, mut work: TokenizerWorkspace, system: String) raises:
        self.initial = tokenizer.encode("<|im_start|>system\n"+system+"<|im_end|>\n",work)
        self.tokens = self.initial.copy()
        self.generating = False
        self.maximum = 0
        self.generated = 0
        self.reason = "ready"

    def reset(mut self):
        self.tokens = self.initial.copy()
        self.generating = False
        self.generated = 0
        self.reason = "ready"

    def begin(mut self, tokenizer: Tokenizer, mut work: TokenizerWorkspace,
              message: String, maximum: Int, capacity: Int) raises:
        if self.generating or maximum < 1 or maximum > MAX_CONTEXT or message.byte_length() == 0:
            raise Error("invalid chat turn")
        var suffix = tokenizer.encode("<|im_start|>user\n"+message+"<|im_end|>\n<|im_start|>assistant\n",work)
        # Reserve the entire reply plus a forced end marker and its newline.
        # Rejection is atomic: neither history nor model state has changed.
        if len(self.tokens)+len(suffix)+maximum+2 > capacity:
            raise Error("Conversation full: use /reset or a shorter message/reply limit.")
        self.tokens.extend(suffix^)
        self.maximum = maximum
        self.generated = 0
        self.generating = True
        self.reason = "generating"

    def finish(mut self, reason: String):
        if not self.generating:
            return
        if self.generated == 0 or self.tokens[len(self.tokens)-1] != IM_END:
            self.tokens.append(IM_END)
        self.tokens.append(NEWLINE)
        self.generating = False
        self.reason = reason

    def accept(mut self, token: Int) raises:
        if not self.generating or token < 0 or token >= VOCABULARY:
            raise Error("invalid generated chat token")
        self.tokens.append(token)
        self.generated += 1
        if is_stop(token):
            self.finish("stop")
        elif self.generated == self.maximum:
            self.finish("limit")


struct ChatSession(Movable):
    var model: QwenModel
    var kv: KVPool
    var blocks: BlockManager
    var sequence: Int
    var history: ChatHistory

    def __init__(out self, ctx: DeviceContext, prepared: String,
                 tokenizer: Tokenizer, mut work: TokenizerWorkspace,
                 system: String, chunk_rows: Int = 256, capacity: Int = MAX_CONTEXT,
                 block_size: Int = KV_BLOCK_SIZE, head_major: Bool = KV_HEAD_MAJOR) raises:
        """A session whose conversation fills blocks of block_size slots, by default the plan's
        layout; 0 holds it in one block."""
        self.history = ChatHistory(tokenizer,work,system)
        if len(self.history.tokens)+3 >= capacity:
            raise Error("system message exceeds chat capacity")
        self.model = QwenModel(ctx,prepared,capacity,chunk_rows)
        var size = block_size if block_size > 0 else capacity
        var count = (capacity+size-1)//size
        self.kv = KVPool(ctx,count,size,self.model.kv_geometry(),head_major)
        self.blocks = BlockManager(count,size,capacity)
        self.sequence = self.blocks.add()

    def length(self) raises -> Int:
        """Conversation tokens whose KV writes have been submitted."""
        return self.blocks.length(self.sequence)

    def table(self) raises -> List[Int]:
        """The session's blocks in position order."""
        return self.blocks.table(self.sequence)

    def begin(mut self, tokenizer: Tokenizer, mut work: TokenizerWorkspace,
              message: String, maximum: Int) raises:
        if not self.model.valid:
            raise Error("Session needs /reset after an execution failure.")
        self.history.begin(tokenizer,work,message,maximum,self.model.capacity)

    def submit_next(mut self, ctx: DeviceContext) raises:
        var cached = self.length()
        if not self.history.generating or not self.model.valid or cached >= len(self.history.tokens):
            raise Error("no valid pending chat input")
        var rows = min(self.model.max_rows,len(self.history.tokens)-cached)
        var ids = List[Int](capacity=rows)
        for i in range(rows):
            ids.append(self.history.tokens[cached+i])
        self.blocks.reserve(self.sequence,cached+rows)
        self.model.forward(ctx,StepBatch.sequence(ids,cached,self.table(),self.kv.block_size),self.kv,
            fast_plan(rows,cached+rows,ctx.name()))
        self.blocks.commit(self.sequence,cached+rows)

    def sample(mut self, ctx: DeviceContext) raises -> Int:
        if not self.history.generating or self.length() != len(self.history.tokens):
            raise Error("chat sample requires a completely cached prefix")
        var token = self.model.greedy(ctx)
        self.history.accept(token)
        return token

    def truncate(mut self, length: Int) raises:
        """Forget submitted positions from `length` on; their blocks return to the manager.

        The caller guarantees that no queued work still depends on them.
        """
        self.kv.truncate_table(self.table(),length)
        self.blocks.truncate(self.sequence,length)

    def reset(mut self, ctx: DeviceContext) raises:
        self.model.reset(ctx)
        self.kv.reset(ctx)
        self.blocks.reset()
        self.sequence = self.blocks.add()
        self.history.reset()

    def fail(mut self):
        self.model.valid = False
        self.history.generating = False
        self.history.reason = "error"
