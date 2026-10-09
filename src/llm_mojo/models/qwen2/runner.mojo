"""Synchronous Metal adapter for the model-independent request engine."""
from max.gpu.host import DeviceContext
from llm_mojo.layers.decoder_layer import DECODER_FUSED_DECODE, DECODER_MIXED
from llm_mojo.models.qwen2.model import QwenModel
from llm_mojo.models.qwen2.plan import KV_BLOCK_SIZE, configured_plan
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool
from llm_mojo.serving.runner import ModelRunner


def select_engine_configuration(batch: StepBatch, fast_decode: Bool = False) -> Int:
    """Optional decode composition; unfinished singleton prefill stays mixed.

    StepBatch validation still belongs to the model. A completed singleton
    prefill has the same one-query shape as decode and may use configuration 26.
    An unfinished prompt has no selected logit, which configuration 26 requires.
    """
    if (fast_decode and batch.rows() > 0 and batch.rows() == batch.sequences()
            and len(batch.logits_rows) == batch.sequences()):
        return DECODER_FUSED_DECODE
    return DECODER_MIXED


trait EngineChatRunner(ModelRunner, Deinitable):
    """Only the lifecycle operations used by the one-request chat adapter."""

    @staticmethod
    def create_chat_runner(prepared: String, capacity: Int, chunk_rows: Int) raises -> Self:
        ...

    def create_chat_kv(self, capacity: Int) raises -> KVPool:
        ...

    def chat_valid(self) -> Bool:
        ...

    def chat_submitted_rows(self) -> Int:
        ...

    def chat_synchronize(mut self) raises:
        ...

    def chat_invalidate(mut self):
        ...


struct QwenRunner(EngineChatRunner):
    var ctx: DeviceContext
    var model: QwenModel
    var origin_ns: Int
    var fast_decode: Bool

    def __init__(out self, path: String, capacity: Int, max_rows: Int, max_sequences: Int,
                 fast_decode: Bool = False) raises:
        self.ctx = DeviceContext()
        self.model = QwenModel(self.ctx,path,capacity,max_rows,max_sequences)
        self.fast_decode = fast_decode
        self.origin_ns = Int(now())

    @staticmethod
    def create_chat_runner(prepared: String, capacity: Int, chunk_rows: Int) raises -> Self:
        return Self(prepared,capacity,chunk_rows,1)

    def create_chat_kv(self, capacity: Int) raises -> KVPool:
        var blocks = (capacity+KV_BLOCK_SIZE-1)//KV_BLOCK_SIZE
        return KVPool(self.ctx,blocks,KV_BLOCK_SIZE,self.model.kv_geometry())

    def chat_valid(self) -> Bool:
        return self.model.valid

    def chat_submitted_rows(self) -> Int:
        return self.model.submitted_rows

    def chat_synchronize(mut self) raises:
        self.ctx.synchronize()

    def chat_invalidate(mut self):
        self.model.valid = False

    def now_ns(self) -> Int:
        return Int(now())-self.origin_ns

    def reset_clock(mut self):
        self.origin_ns = Int(now())

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        var total = 0
        for length in batch.seq_lens:
            total = max(total,length)
        try:
            var configuration = select_engine_configuration(batch,self.fast_decode)
            self.model.forward(self.ctx,batch,kv,configured_plan(configuration,batch.rows(),total,batch.sequences()))
            # The readback (or explicit synchronization for no sampled rows)
            # completes all queued writes before block ownership can change.
            return self.model.greedy_tokens(self.ctx)
        except error:
            self.model.valid = False
            self.ctx.synchronize()
            raise error
