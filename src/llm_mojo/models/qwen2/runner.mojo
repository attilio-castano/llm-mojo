"""Synchronous and two-context Metal adapters for the request engine."""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from llm_mojo.layers.decoder_layer import DECODER_FUSED_DECODE, DECODER_MIXED
from llm_mojo.models.qwen2.model import ForwardRoute, QwenModel, VOCABULARY
from llm_mojo.models.qwen2.plan import KV_BLOCK_SIZE, configured_plan
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool
from llm_mojo.serving.runner import AsyncModelRunner, ModelRunner
from llm_mojo.serving.engine import EngineCore, EngineStep


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

    def chat_device_name(self) -> String:
        ...

    def chat_device_api(self) -> String:
        ...

    def chat_synchronize(mut self) raises:
        ...

    def chat_invalidate(mut self):
        ...

    def chat_async(self) -> Bool:
        return False

    def chat_step(mut self, mut engine: EngineCore, mut kv: KVPool) raises -> EngineStep:
        return engine.step(self,kv)


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

    def chat_device_name(self) -> String:
        return self.ctx.name()

    def chat_device_api(self) -> String:
        return self.ctx.api()

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


struct QwenAsyncSlot(Movable):
    var metadata: DeviceBuffer[DType.int32]
    var staging: HostBuffer[DType.int32]
    var selected: DeviceBuffer[DType.uint32]
    var readback: HostBuffer[DType.uint32]
    var ticket: Int
    var samples: Int
    var pending: Bool
    var route: ForwardRoute

    def __init__(out self, ctx: DeviceContext, metadata_size: Int, sequences: Int) raises:
        self.metadata = ctx.enqueue_create_buffer[DType.int32](metadata_size)
        self.staging = ctx.enqueue_create_host_buffer[DType.int32](metadata_size)
        self.selected = ctx.enqueue_create_buffer[DType.uint32](sequences*3)
        self.readback = ctx.enqueue_create_host_buffer[DType.uint32](sequences*3)
        self.ticket = -1
        self.samples = 0
        self.pending = False
        self.route = ForwardRoute(-1,0,0,0,0,0,0,False,False,0,0)


struct QwenAsyncRunner(EngineChatRunner, AsyncModelRunner):
    var ctx: DeviceContext
    var next_ctx: DeviceContext
    var model: QwenModel
    var slots: List[QwenAsyncSlot]
    var origin_ns: Int
    var fast_decode: Bool
    var next_ticket: Int
    var collect_ticket: Int
    var drain_failure: String

    def __init__(out self, path: String, capacity: Int, max_rows: Int, max_sequences: Int,
                 fast_decode: Bool = False) raises:
        var ctx = DeviceContext()
        var model = QwenModel(ctx,path,capacity,max_rows,max_sequences)
        self = Self(ctx,model^,fast_decode)

    def __init__(out self, ctx: DeviceContext, var model: QwenModel,
                 fast_decode: Bool = False) raises:
        self.ctx = ctx
        self.next_ctx = DeviceContext()
        if self.ctx.api() != "metal" or self.next_ctx.api() != "metal" or self.ctx.name() != self.next_ctx.name():
            raise Error("Async Qwen requires two contexts on the same actual Metal device")
        self.slots = List[QwenAsyncSlot](capacity=2)
        self.slots.append(QwenAsyncSlot(self.ctx,len(model.step_input),model.max_sequences))
        self.slots.append(QwenAsyncSlot(self.next_ctx,len(model.step_input),model.max_sequences))
        self.model = model^
        self.fast_decode = fast_decode
        self.next_ticket = 0
        self.collect_ticket = 0
        self.drain_failure = String()
        self.origin_ns = Int(now())
        # Allocation must complete before any CPU staging access.
        self.ctx.synchronize()
        self.next_ctx.synchronize()
        for bank in range(2):
            for index in range(len(self.slots[bank].staging)):
                self.slots[bank].staging[index] = Int32(0)

    def __deinit__(deinit self):
        # Field destruction can otherwise free context-0 model workspaces
        # while context 1 still reads them. Attempt both queues independently.
        try:
            self.ctx.synchronize()
        except error:
            print("Async Qwen context 0 final drain failed:",error)
        try:
            self.next_ctx.synchronize()
        except error:
            print("Async Qwen context 1 final drain failed:",error)

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

    def chat_device_name(self) -> String:
        return self.ctx.name()

    def chat_device_api(self) -> String:
        return self.ctx.api()

    def chat_synchronize(mut self) raises:
        self.drain()

    def chat_invalidate(mut self):
        self.model.valid = False

    def chat_async(self) -> Bool:
        return True

    def chat_step(mut self, mut engine: EngineCore, mut kv: KVPool) raises -> EngineStep:
        return engine.step_async(self,kv)

    def now_ns(self) -> Int:
        return Int(now())-self.origin_ns

    def reset_clock(mut self) raises:
        """Start a fresh measurement after every earlier ticket has drained."""
        if self.pending() != 0:
            raise Error("Async clock reset requires no pending tickets")
        self.drain()
        self.next_ticket = 0
        self.collect_ticket = 0
        for bank in range(2):
            self.slots[bank].ticket = -1
            self.slots[bank].samples = 0
            self.slots[bank].route = ForwardRoute(-1,0,0,0,0,0,0,False,False,0,0)
        self.origin_ns = Int(now())

    def pending(self) -> Int:
        return self.next_ticket-self.collect_ticket

    def submit(mut self, batch: StepBatch, source_indices: List[Int],
               source_ticket: Int, mut kv: KVPool) raises -> Int:
        # All host rejection precedes enqueue and preserves healthy state.
        if not self.model.valid or self.pending() >= 2 or len(source_indices) != batch.rows():
            raise Error("Invalid async Qwen state, source extent or pending depth")
        var has_source = False
        for source in source_indices:
            if source < -1:
                raise Error("Async source index must be literal -1 or a sampled-logit index")
            has_source = has_source or source >= 0
        var previous = (self.next_ticket+1)%2
        if has_source:
            if source_ticket != self.next_ticket-1 or source_ticket < 0 or self.slots[previous].ticket != source_ticket:
                raise Error("Async source must name the immediately preceding submitted ticket")
            for source in source_indices:
                if source >= self.slots[previous].samples:
                    raise Error("Async token source is outside its sampled results")
        elif source_ticket != -1:
            raise Error("Literal async metadata must not name a source ticket")
        var bank = self.next_ticket%2
        if self.slots[bank].pending:
            raise Error("Async staging bank still belongs to an uncollected ticket")
        var ctx = self.ctx if bank == 0 else self.next_ctx
        var predecessor_ctx = self.next_ctx if bank == 0 else self.ctx
        var total = 0
        for length in batch.seq_lens:
            total = max(total,length)
        var configuration = select_engine_configuration(batch,self.fast_decode)
        var plan = configured_plan(configuration,batch.rows(),total,batch.sequences())
        self.model.preflight(ctx,batch,kv,plan)
        self.model.stage_metadata(batch,source_indices,self.slots[bank].staging)
        try:
            if self.next_ticket > 0:
                # Captures only the predecessor prefix already queued now.
                # The just-collected bank can subsequently enqueue n+2 safely.
                ctx.enqueue_wait_for(predecessor_ctx)
            self.model.forward_staged(ctx,batch,kv,plan,self.slots[bank].staging,
                self.slots[bank].metadata,self.slots[bank].selected,
                self.slots[previous].selected,has_source)
            if len(batch.logits_rows) > 0:
                ctx.enqueue_copy(dst_buf=self.slots[bank].readback,src_buf=self.slots[bank].selected)
            var ticket = self.next_ticket
            self.slots[bank].ticket = ticket
            self.slots[bank].samples = len(batch.logits_rows)
            self.slots[bank].route = self.model.last_route
            self.slots[bank].pending = True
            self.next_ticket += 1
            return ticket
        except error:
            self.model.valid = False
            try:
                self.drain()
            except drain_error:
                raise Error("Async submission failed: " + String(error) + "; drain failed: " + String(drain_error))
            raise error

    def collect(mut self, ticket: Int) raises -> List[Int]:
        if not self.model.valid or ticket != self.collect_ticket or ticket >= self.next_ticket:
            raise Error("Async Qwen collection requires the oldest pending healthy ticket")
        var bank = ticket%2
        if not self.slots[bank].pending or self.slots[bank].ticket != ticket:
            raise Error("Async Qwen ticket no longer owns its bank")
        try:
            var ctx = self.ctx if bank == 0 else self.next_ctx
            ctx.synchronize()
            var tokens = List[Int](capacity=self.slots[bank].samples)
            for index in range(self.slots[bank].samples):
                if self.slots[bank].readback[index*3+2] != 0:
                    raise Error("Nonfinite async model logits in sequence " + String(index))
                var selected = Int(self.slots[bank].readback[index*3+1])
                if selected < 0 or selected >= VOCABULARY:
                    raise Error("Invalid async GPU token result")
                tokens.append(selected)
            self.model.last_route = self.slots[bank].route
            self.slots[bank].pending = False
            self.collect_ticket += 1
            return tokens^
        except error:
            self.model.valid = False
            try:
                self.drain()
            except drain_error:
                raise Error("Async collection failed: " + String(error) + "; drain failed: " + String(drain_error))
            raise error

    def drain(mut self) raises:
        # Both contexts may still use shared workspace or the other's results.
        # No caller may release KV allocations until these waits succeed.
        var failure = String()
        try:
            self.ctx.synchronize()
        except error:
            failure = "context 0: " + String(error)
        try:
            self.next_ctx.synchronize()
        except error:
            failure += "; context 1: " + String(error)
        if failure.byte_length() > 0:
            self.model.valid = False
            self.drain_failure = failure.copy()
            # Failed waits cannot establish retirement of either bank.
            raise Error("Async Qwen drain could not prove completion: " + failure)
        for index in range(2):
            self.slots[index].pending = False
        self.collect_ticket = self.next_ticket

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        if self.pending() != 0:
            raise Error("Synchronous compatibility execution cannot bypass pending async tickets")
        var sources = List[Int](capacity=batch.rows())
        for _ in range(batch.rows()):
            sources.append(-1)
        var ticket = self.submit(batch,sources,-1,kv)
        return self.collect(ticket)
