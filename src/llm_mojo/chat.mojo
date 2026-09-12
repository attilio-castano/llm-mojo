"""Qwen text-chat framing and a persistent native batch-one session.

History is authoritative; model.length identifies its already submitted prefix.
No rendered-text round trip is used for generated assistant tokens.
"""
from max.gpu.host import DeviceContext
from llm_mojo.model import QwenModel, select_configuration, select_token_selection
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.generate_cli import is_stop

comptime DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
comptime IM_END = 151645
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
        if self.generating or maximum < 1 or maximum > 4096 or message.byte_length() == 0:
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
        if not self.generating or token < 0 or token >= 151936:
            raise Error("invalid generated chat token")
        self.tokens.append(token)
        self.generated += 1
        if is_stop(token):
            self.finish("stop")
        elif self.generated == self.maximum:
            self.finish("limit")


struct ChatSession(Movable):
    var model: QwenModel
    var history: ChatHistory
    var policy: String

    def __init__(out self, ctx: DeviceContext, prepared: String,
                 tokenizer: Tokenizer, mut work: TokenizerWorkspace,
                 system: String, chunk_rows: Int = 256, capacity: Int = 4096) raises:
        self.history = ChatHistory(tokenizer,work,system)
        if len(self.history.tokens)+3 >= capacity:
            raise Error("system message exceeds chat capacity")
        self.model = QwenModel(ctx,prepared,capacity,chunk_rows)
        self.policy = "fast"

    def begin(mut self, tokenizer: Tokenizer, mut work: TokenizerWorkspace,
              message: String, maximum: Int) raises:
        if not self.model.valid:
            raise Error("Session needs /reset after an execution failure.")
        self.history.begin(tokenizer,work,message,maximum,self.model.capacity)

    def submit_next(mut self, ctx: DeviceContext) raises:
        if not self.history.generating or not self.model.valid or self.model.length >= len(self.history.tokens):
            raise Error("no valid pending chat input")
        var rows = min(self.model.max_rows,len(self.history.tokens)-self.model.length)
        var ids = List[Int](capacity=rows)
        for i in range(rows):
            ids.append(self.history.tokens[self.model.length+i])
        var config = select_configuration(self.policy,rows,self.model.length+rows,ctx.name())
        self.model.forward(ctx,ids,config,"",select_token_selection(self.policy,rows,ctx.name()))

    def sample(mut self, ctx: DeviceContext) raises -> Int:
        if not self.history.generating or self.model.length != len(self.history.tokens):
            raise Error("chat sample requires a completely cached prefix")
        var token = self.model.greedy(ctx)
        self.history.accept(token)
        return token

    def reset(mut self, ctx: DeviceContext) raises:
        self.model.reset(ctx)
        self.history.reset()

    def fail(mut self):
        self.model.valid = False
        self.history.generating = False
        self.history.reason = "error"
