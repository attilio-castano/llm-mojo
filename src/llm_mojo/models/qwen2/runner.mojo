"""Synchronous Metal adapter for the model-independent request engine."""
from max.gpu.host import DeviceContext
from llm_mojo.layers.decoder_layer import DECODER_MIXED
from llm_mojo.models.qwen2.model import QwenModel
from llm_mojo.models.qwen2.plan import configured_plan
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool
from llm_mojo.serving.runner import ModelRunner


struct QwenRunner(ModelRunner):
    var ctx: DeviceContext
    var model: QwenModel
    var origin_ns: Int

    def __init__(out self, path: String, capacity: Int, max_rows: Int, max_sequences: Int) raises:
        self.ctx = DeviceContext()
        self.model = QwenModel(self.ctx,path,capacity,max_rows,max_sequences)
        self.origin_ns = Int(now())

    def now_ns(self) -> Int:
        return Int(now())-self.origin_ns

    def reset_clock(mut self):
        self.origin_ns = Int(now())

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        var total = 0
        for length in batch.seq_lens:
            total = max(total,length)
        try:
            self.model.forward(self.ctx,batch,kv,configured_plan(DECODER_MIXED,batch.rows(),total,batch.sequences()))
            # The readback (or explicit synchronization for no sampled rows)
            # completes all queued writes before block ownership can change.
            return self.model.greedy_tokens(self.ctx)
        except error:
            self.model.valid = False
            self.ctx.synchronize()
            raise error
