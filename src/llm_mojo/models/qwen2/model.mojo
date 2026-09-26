"""Fixed Qwen model ownership. Native execution; prepared files are verified by tooling.

The model owns weights and workspaces. KV storage and sequence lengths belong to the
caller's KVPool; each forward receives a StepBatch describing the step. The
cross-layer copy or owner swap keeps the decoder alias contract intact. Numerical
comparisons are diagnostics; storage and lifecycle invariants remain exact.
Which kernels a call uses is decided by an ExecutionPlan (models/qwen2/plan.mojo):
configuration 26 runs the decode composition for one row per sequence, and every
other configuration runs one sequence through the generic layer dispatch.
"""
from std.memory import bitcast
from std.gpu import global_idx
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.layers.attention_sublayer import AttentionWeights, AttentionWorkspace
from llm_mojo.layers.mlp import MLPWeights, MLPWorkspace
from llm_mojo.layers.decoder_layer import (
    DECODER_FUSED_DECODE, decoder_mappings, enqueue_decode_batch_layer, enqueue_decoder_layer_configuration,
    validate_decode_batch_layer, validate_decoder_configuration,
)
from llm_mojo.kernels.residual_norm import enqueue_residual_norm
from llm_mojo.kernels.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.kernels.linear import enqueue_linear_apple_gpu, enqueue_linear_decode_rows_apple_gpu
from llm_mojo.kernels.token_selection import enqueue_argmax
from llm_mojo.models.qwen2.plan import DECODE_PROJECTION, ExecutionPlan, MAX_CONTEXT
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVGeometry, KVPool

comptime HIDDEN = 896
comptime VOCABULARY = 151936
comptime LAYERS = 24
comptime QUERY_HEADS = 14
comptime KV_HEADS = 2
comptime HEAD_DIM = 64
comptime KV_WIDTH = KV_HEADS * HEAD_DIM
# One (score, token, nonfinite) record per 1024-logit argmax group.
comptime ARGMAX_GROUPS = (VOCABULARY + 1023) // 1024

# Host observation slots, recorded only in OBSERVE specializations.
comptime MARK_START = 0
comptime MARK_PREFLIGHT = 1
comptime MARK_TOKENS = 2
comptime MARK_EMBEDDING = 3
comptime MARK_LAYERS = 4
comptime MARK_HEAD = 5
comptime MARK_GREEDY = 6
comptime MARK_MAPPED = 7
comptime MARK_SELECTED = 8
comptime MARK_RETURN = 9


def generation_budget(prompt_length: Int, maximum: Int) raises -> Int:
    """New tokens a reply may add: the request, capped by the remaining context."""
    if prompt_length < 1 or prompt_length > MAX_CONTEXT or maximum < 0 or maximum > MAX_CONTEXT:
        raise Error("invalid prompt or generation limit")
    return min(maximum,MAX_CONTEXT-prompt_length)


def load_bf16(buffer: DeviceBuffer[DType.bfloat16], path: String, source_elements: Int = 0) raises:
    """Initialization only. Caller has verified manifest and all file hashes."""
    var data = open(path, "r").read_bytes()
    var expected = source_elements if source_elements > 0 else len(buffer)
    if expected < len(buffer) or len(data) != expected * 2:
        raise Error("prepared model tensor byte extent mismatch: " + path)
    with buffer.map_to_host() as mapped:
        var dst = mapped.unsafe_ptr()
        for i in range(len(buffer)):
            dst[unsafe_offset=i] = bitcast[DType.bfloat16](UInt16(data[2*i]) | (UInt16(data[2*i+1]) << 8))


def save_bf16(buffer: DeviceBuffer[DType.bfloat16], path: String, count: Int, start: Int = 0) raises:
    """Diagnostic-only readback. Deliberately synchronizes before observation."""
    if start < 0 or count < 0 or start > len(buffer) or count > len(buffer)-start:
        raise Error("invalid diagnostic extent")
    var data = List[UInt8](capacity=count*2)
    with buffer.map_to_host() as mapped:
        for i in range(count):
            var bits = bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=start+i])
            data.append(UInt8(bits & 255))
            data.append(UInt8(bits >> 8))
    var file = open(path,"w")
    file.write_bytes(data)


def _embedding[IL: TensorLayout, WL: TensorLayout, OL: TensorLayout](
    ids: TileTensor[DType.int32, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin], count: Int32,
):
    comptime assert ids.flat_rank == 1 and weight.flat_rank == 2 and output.flat_rank == 2
    var i = global_idx.x
    if i < Int(count) * HIDDEN:
        var row = i // HIDDEN
        output[row,i % HIDDEN] = weight[Int(ids[row]),i % HIDDEN]


def _copy_rows[IL: TensorLayout, OL: TensorLayout](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin], count: Int32,
):
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    var i = global_idx.x
    if i < Int(count)*HIDDEN:
        output[i//HIDDEN,i%HIDDEN] = input[i//HIDDEN,i%HIDDEN]


def swap_hidden_buffers(mut left: DeviceBuffer[DType.bfloat16], mut right: DeviceBuffer[DType.bfloat16]):
    """Exchange owners, keeping both allocations alive for queued GPU readers."""
    var previous = left^
    left = right^
    right = previous^


@fieldwise_init
struct CaptureRequest(Copyable, Movable):
    """Diagnostic capture: every layer boundary is written under directory."""
    var directory: String
    var include_norms: Bool


@fieldwise_init
struct ForwardRoute(ImplicitlyCopyable, Movable):
    """What the last forward actually enqueued, counted where each dispatch is issued.

    decode_launches counts the decode composition's compute launches; it stays
    zero on the generic layer path.
    """
    var configuration: Int
    var layers: Int
    var normalized_inputs: Int
    var layer_residual_norms: Int
    var deferred_residual_norms: Int
    var owner_swaps: Int
    var hidden_copies: Int
    var final_rms_norm: Bool
    var gpu_argmax: Bool
    var sequences: Int
    var decode_launches: Int

    def describe(self) -> String:
        return ("configuration=" + String(self.configuration) + " layers=" + String(self.layers)
                + " normalized_inputs=" + String(self.normalized_inputs)
                + " residual_norms=" + String(self.layer_residual_norms + self.deferred_residual_norms)
                + " swaps=" + String(self.owner_swaps) + " copies=" + String(self.hidden_copies)
                + " final_rms_norm=" + String(Int(self.final_rms_norm))
                + " gpu_argmax=" + String(Int(self.gpu_argmax)))


struct ModelLayer(Movable):
    var attention: AttentionWeights
    var mlp: MLPWeights

    def __init__(out self, ctx: DeviceContext) raises:
        self.attention = AttentionWeights(ctx)
        self.mlp = MLPWeights(ctx)

    def __init__(out self, ctx: DeviceContext, path: String, index: Int) raises:
        self = Self(ctx)
        self.load(path,index)

    def load(mut self, path: String, index: Int) raises:
        var prefix = path + "/layer_" + String(index) + "_"
        load_bf16(self.attention.norm,prefix+"attention_norm.bin")
        load_bf16(self.attention.qkv,prefix+"qkv.bin")
        load_bf16(self.attention.bias,prefix+"bias.bin")
        load_bf16(self.attention.output,prefix+"wo.bin")
        load_bf16(self.mlp.norm,prefix+"mlp_norm.bin")
        load_bf16(self.mlp.gate,prefix+"gate.bin")
        load_bf16(self.mlp.up,prefix+"up.bin")
        load_bf16(self.mlp.down,prefix+"down.bin")


struct QwenModel(Movable):
    var layers: List[ModelLayer]
    var embedding: DeviceBuffer[DType.bfloat16]
    var norm: DeviceBuffer[DType.bfloat16]
    var attention: AttentionWorkspace
    var mlp: MLPWorkspace
    var input: DeviceBuffer[DType.bfloat16]
    var normalized: DeviceBuffer[DType.bfloat16]
    var logits: DeviceBuffer[DType.bfloat16]
    var selection_partials: DeviceBuffer[DType.uint32]
    var selection_result: DeviceBuffer[DType.uint32]
    var gpu_argmax: Bool
    # The step's one upload: max_rows token IDs, then each sequence's position and block.
    var step_input: DeviceBuffer[DType.int32]
    var capacity: Int
    var max_rows: Int
    var max_sequences: Int
    # A successful forward has produced logits that greedy can read.
    var ready: Bool
    var valid: Bool
    var submitted_rows: Int
    var last_route: ForwardRoute
    # Host-only observation storage. Default specializations contain no clocks.
    var observation: List[UInt64]

    def __init__(out self, ctx: DeviceContext, layer_count: Int, capacity: Int, max_rows: Int,
                 max_sequences: Int = 1) raises:
        """Allocate without loading; use the path constructor for the prepared checkpoint."""
        if (ctx.api() != "metal" or layer_count < 1 or capacity < 1 or capacity > MAX_CONTEXT
                or max_rows < 1 or max_rows > capacity or max_sequences < 1 or max_sequences > max_rows):
            raise Error("Qwen requires Metal and valid layer, row, sequence and context capacity")
        self.capacity = capacity
        self.max_rows = max_rows
        self.max_sequences = max_sequences
        self.ready = False
        self.valid = True
        self.submitted_rows = 0
        self.last_route = ForwardRoute(-1, 0, 0, 0, 0, 0, 0, False, False, 0, 0)
        self.observation = List[UInt64](capacity=10)
        for _ in range(10):
            self.observation.append(0)
        self.embedding = ctx.enqueue_create_buffer[DType.bfloat16](VOCABULARY*HIDDEN)
        self.norm = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
        self.layers = List[ModelLayer](capacity=layer_count)
        for _ in range(layer_count):
            self.layers.append(ModelLayer(ctx))
        self.attention = AttentionWorkspace(ctx,max_rows,capacity,
            materialized=False,fp32_materialized=False,prefill_splits=8)
        self.mlp = MLPWorkspace(ctx,max_rows)
        self.input = ctx.enqueue_create_buffer[DType.bfloat16](max_rows*HIDDEN)
        self.normalized = ctx.enqueue_create_buffer[DType.bfloat16](max_sequences*HIDDEN)
        self.logits = ctx.enqueue_create_buffer[DType.bfloat16](max_sequences*VOCABULARY)
        self.selection_partials = ctx.enqueue_create_buffer[DType.uint32](max_sequences*ARGMAX_GROUPS*3)
        self.selection_result = ctx.enqueue_create_buffer[DType.uint32](max_sequences*3)
        self.gpu_argmax = False
        self.step_input = ctx.enqueue_create_buffer[DType.int32](max_rows+2*max_sequences)

    def __init__(out self, ctx: DeviceContext, path: String, capacity: Int, max_rows: Int,
                 max_sequences: Int = 1) raises:
        """Allocate all 24 layers and load the verified prepared checkpoint at path."""
        self = Self(ctx, LAYERS, capacity, max_rows, max_sequences)
        load_bf16(self.embedding,path+"/embedding.bin")
        load_bf16(self.norm,path+"/final_norm.bin")
        for i in range(LAYERS):
            self.layers[i].load(path,i)
        # Preparation stores all 4096 positions; only capacity rows are resident.
        load_bf16(self.attention.cosine,path+"/cosine.bin",MAX_CONTEXT*64)
        load_bf16(self.attention.sine,path+"/sine.bin",MAX_CONTEXT*64)
        ctx.synchronize()

    @staticmethod
    def allocate(ctx: DeviceContext, layer_count: Int, capacity: Int, max_rows: Int,
                 max_sequences: Int = 1) raises -> QwenModel:
        """Unloaded model storage for tests that supply their own weights."""
        return QwenModel(ctx, layer_count, capacity, max_rows, max_sequences)

    def kv_geometry(self) -> KVGeometry:
        """What each token stores in every layer; pools serving this model use it."""
        return KVGeometry(len(self.layers), self.layers[0].attention.kv_heads, self.layers[0].attention.head_dim)

    def reset(mut self, ctx: DeviceContext) raises:
        """Finish queued work and forget the last logits. Callers reset their KV pools."""
        self.valid = False
        ctx.synchronize()
        self.ready = False
        self.valid = True
        self.submitted_rows = 0

    def preflight(mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan) raises:
        batch.validate(kv.blocks,kv.block_size,VOCABULARY)
        var rows = batch.rows()
        var sequences = batch.sequences()
        # Every sequence is held in one block; only configuration 26 steps several (plan.validate).
        if batch.max_blocks != 1:
            raise Error("Qwen steps sequences held in one block each")
        if not self.valid or rows > self.max_rows or sequences > self.max_sequences:
            raise Error("invalid Qwen state, row extent or context overflow")
        plan.validate(rows, sequences)
        if len(self.layers) < 1:
            raise Error("empty Qwen layer stack")
        if kv.geometry != self.kv_geometry() or kv.block_size != self.capacity:
            raise Error("KV pool geometry does not match the model")
        if (len(self.embedding) != VOCABULARY*HIDDEN or len(self.norm) != HIDDEN
            or len(self.input) != self.max_rows*HIDDEN or len(self.step_input) != self.max_rows+2*self.max_sequences
            or len(self.normalized) != self.max_sequences*HIDDEN or len(self.logits) != self.max_sequences*VOCABULARY
            or len(self.selection_partials) != self.max_sequences*ARGMAX_GROUPS*3
            or len(self.selection_result) != self.max_sequences*3
            or self.attention.capacity != self.capacity
            or self.attention.max_rows != self.max_rows or self.mlp.max_rows != self.max_rows):
            raise Error("inconsistent model allocation geometry")
        # Each sequence's step must start exactly where every layer of its block ends.
        for s in range(sequences):
            var base = kv.index(batch.block_table[s],0)
            var past = batch.positions[batch.query_start[s]]
            for i in range(len(self.layers)):
                if kv.caches[base+i].length != past or kv.caches[base+i].capacity != self.capacity:
                    raise Error("inconsistent model cache lengths")
        if plan.configuration == DECODER_FUSED_DECODE:
            var positions = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows),row_major(sequences))
            var blocks = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows+self.max_sequences),
                                    row_major(sequences))
            for i in range(len(self.layers)):
                validate_decode_batch_layer[QUERY_HEADS,KV_HEADS,HEAD_DIM](ctx,self.layers[i].attention,
                    self.attention,self.layers[i].mlp,self.mlp,self.input,kv.storage,
                    positions,blocks,i,len(self.layers),kv.block_size)
            return
        var mappings = decoder_mappings(plan.configuration,rows)
        var base = kv.index(batch.block_table[0],0)
        for i in range(len(self.layers)):
            validate_decoder_configuration(ctx,self.layers[i].attention,kv.caches[base+i],self.attention,
                self.layers[i].mlp,self.mlp,TileTensor(self.input,row_major(rows,HIDDEN)),
                Int(mappings[0]),Int(mappings[1]),Int(mappings[2]))

    @always_inline
    def _mark[OBSERVE: Bool](mut self, slot: Int):
        comptime if OBSERVE:
            self.observation[slot] = now()

    def forward[OBSERVE: Bool = False, PROJECTION: Int = DECODE_PROJECTION](
            mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan) raises:
        """Submit all layers for one step under plan. The step upload synchronizes; layer execution does not.

        The batch names each sequence's block in `kv`, whose layer views must all
        hold exactly the rows before that sequence's first position. With
        plan.gpu_argmax the vocabulary projection is followed by GPU argmax;
        otherwise greedy scans the materialized logits on the CPU. PROJECTION is
        the decode composition's batched projection arrangement; it changes how
        rows share weight loads, not results.
        """
        self._forward[OBSERVE, False, PROJECTION](ctx, batch, kv, plan, CaptureRequest("", False))

    def forward_captured(mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan,
                         request: CaptureRequest) raises:
        """Diagnostic forward that synchronously writes every boundary under request.directory."""
        if request.directory.byte_length() == 0:
            raise Error("capture requires a directory")
        if batch.sequences() != 1:
            raise Error("capture covers one sequence")
        self._forward[False, True, DECODE_PROJECTION](ctx, batch, kv, plan, request)

    def _forward[OBSERVE: Bool, CAPTURE: Bool, PROJECTION: Int](mut self, ctx: DeviceContext, batch: StepBatch,
                                                              mut kv: KVPool, plan: ExecutionPlan,
                                                              request: CaptureRequest) raises:
        self._mark[OBSERVE](MARK_START)
        self.preflight(ctx,batch,kv,plan)
        self._mark[OBSERVE](MARK_PREFLIGHT)
        try:
            var route: ForwardRoute
            if plan.configuration == DECODER_FUSED_DECODE:
                route = self._decode_step[OBSERVE, CAPTURE, PROJECTION](ctx, batch, kv, plan, request)
            else:
                route = self._layer_step[OBSERVE, CAPTURE](ctx, batch, kv, plan, request)
            self.ready = True
            self.submitted_rows += batch.rows()*len(self.layers)
            self.last_route = route
        except error:
            self.valid = False
            raise error

    def _embed(mut self, ctx: DeviceContext, rows: Int) raises:
        var token_view = TileTensor(self.step_input,row_major(rows))
        var weight_view = TileTensor(self.embedding,row_major(VOCABULARY,HIDDEN))
        var input_view = TileTensor(self.input,row_major(rows,HIDDEN))
        comptime embedding_kernel = _embedding[type_of(token_view.layout),type_of(weight_view.layout),type_of(input_view.layout)]
        ctx.enqueue_function[embedding_kernel](token_view,weight_view,input_view,Int32(rows),
            grid_dim=(rows*HIDDEN+255)//256,block_dim=256)

    def _decode_step[OBSERVE: Bool, CAPTURE: Bool, PROJECTION: Int](mut self, ctx: DeviceContext, batch: StepBatch,
                                                                  mut kv: KVPool, plan: ExecutionPlan,
                                                                  request: CaptureRequest) raises -> ForwardRoute:
        """Configuration 26: one row per sequence through the decode composition, GPU argmax per row."""
        var sequences = batch.sequences()
        var layer_count = len(self.layers)
        var route = ForwardRoute(plan.configuration, 0, 0, 0, 0, 0, 0, False, False, sequences, 0)
        var capture = request.directory
        with self.step_input.map_to_host() as mapped:
            for s in range(sequences):
                mapped.unsafe_ptr()[unsafe_offset=s] = Int32(batch.token_ids[s])
                mapped.unsafe_ptr()[unsafe_offset=self.max_rows+s] = Int32(batch.positions[s])
                mapped.unsafe_ptr()[unsafe_offset=self.max_rows+self.max_sequences+s] = Int32(batch.block_table[s])
        self._mark[OBSERVE](MARK_TOKENS)
        var positions = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows),row_major(sequences))
        var blocks = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows+self.max_sequences),
                                row_major(sequences))
        self._embed(ctx, sequences)
        route.decode_launches += 1
        self._mark[OBSERVE](MARK_EMBEDDING)
        comptime if CAPTURE:
            save_bf16(self.input,capture+"/hidden_0.bin",sequences*HIDDEN)
        for i in range(layer_count):
            var normalized_input = i > 0
            route.decode_launches += enqueue_decode_batch_layer[QUERY_HEADS,KV_HEADS,HEAD_DIM,PROJECTION](ctx,
                self.layers[i].attention,self.attention,self.layers[i].mlp,self.mlp,self.input,kv.storage,
                positions,blocks,i,layer_count,kv.block_size,normalized_input)
            for s in range(sequences):
                kv.caches[kv.index(batch.block_table[s],i)].length = batch.positions[s]+1
            route.layers += 1
            if normalized_input:
                route.normalized_inputs += 1
            route.layer_residual_norms += 1
            comptime if CAPTURE:
                if request.include_norms:
                    save_bf16(self.attention.normalized,capture+"/attention_norm_"+String(i)+".bin",sequences*HIDDEN)
                    save_bf16(self.mlp.normalized,capture+"/mlp_norm_"+String(i)+".bin",sequences*HIDDEN)
                    save_bf16(self.attention.output,capture+"/attention_residual_"+String(i)+".bin",sequences*HIDDEN)
            # The MLP residual feeds the next layer's attention norm, or the final norm.
            if i+1 < layer_count:
                enqueue_residual_norm[HIDDEN](ctx,TileTensor(self.attention.output,row_major(sequences,HIDDEN)),
                    TileTensor(self.mlp.down,row_major(sequences,HIDDEN)),
                    TileTensor(self.layers[i+1].attention.norm,row_major(HIDDEN)),
                    TileTensor(self.mlp.output,row_major(sequences,HIDDEN)),
                    TileTensor(self.attention.normalized,row_major(sequences,HIDDEN)))
            else:
                enqueue_residual_norm[HIDDEN](ctx,TileTensor(self.attention.output,row_major(sequences,HIDDEN)),
                    TileTensor(self.mlp.down,row_major(sequences,HIDDEN)),TileTensor(self.norm,row_major(HIDDEN)),
                    TileTensor(self.mlp.output,row_major(sequences,HIDDEN)),
                    TileTensor(self.normalized,row_major(sequences,HIDDEN)))
            route.deferred_residual_norms += 1
            route.decode_launches += 1
            comptime if CAPTURE:
                save_bf16(self.mlp.output,capture+"/hidden_"+String(i+1)+".bin",sequences*HIDDEN)
                # Captures cover one sequence: its appended row and its block's whole cache.
                var view = kv.index(batch.block_table[0],i)
                var appended = (kv.caches[view].length-1)*KV_WIDTH
                save_bf16(kv.caches[view].key,capture+"/append_key_"+String(i)+".bin",KV_WIDTH,appended)
                save_bf16(kv.caches[view].value,capture+"/append_value_"+String(i)+".bin",KV_WIDTH,appended)
                save_bf16(kv.caches[view].key,capture+"/cache_key_"+String(i)+".bin",self.capacity*KV_WIDTH)
                save_bf16(kv.caches[view].value,capture+"/cache_value_"+String(i)+".bin",self.capacity*KV_WIDTH)
            if i+1 < layer_count:
                swap_hidden_buffers(self.input,self.mlp.output)
                route.owner_swaps += 1
        self._mark[OBSERVE](MARK_LAYERS)
        var logits = TileTensor(self.logits,row_major(sequences,VOCABULARY))
        enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx,TileTensor(self.normalized,row_major(sequences,HIDDEN)),
            TileTensor(self.embedding,row_major(VOCABULARY,HIDDEN)),logits)
        enqueue_argmax(ctx,logits,TileTensor(self.selection_partials,row_major(sequences*ARGMAX_GROUPS,3)),
            TileTensor(self.selection_result,row_major(sequences,3)))
        route.decode_launches += 3
        route.gpu_argmax = True
        self.gpu_argmax = True
        self._mark[OBSERVE](MARK_HEAD)
        comptime if CAPTURE:
            save_bf16(self.normalized,capture+"/final_norm.bin",sequences*HIDDEN)
            save_bf16(self.logits,capture+"/logits.bin",sequences*VOCABULARY)
        return route

    def _layer_step[OBSERVE: Bool, CAPTURE: Bool](mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool,
                                                  plan: ExecutionPlan, request: CaptureRequest) raises -> ForwardRoute:
        """Every other configuration: one sequence through the generic layer dispatch and a CPU greedy."""
        var rows = batch.rows()
        var base = kv.index(batch.block_table[0],0)
        var layer_count = len(self.layers)
        var route = ForwardRoute(plan.configuration, 0, 0, 0, 0, 0, 0, False, False, 1, 0)
        var capture = request.directory
        with self.step_input.map_to_host() as mapped:
            for i in range(rows):
                mapped.unsafe_ptr()[unsafe_offset=i] = Int32(batch.token_ids[i])
        self._mark[OBSERVE](MARK_TOKENS)
        self._embed(ctx, rows)
        self._mark[OBSERVE](MARK_EMBEDDING)
        comptime if CAPTURE:
            save_bf16(self.input,capture+"/hidden_0.bin",rows*HIDDEN)
        var input_view = TileTensor(self.input,row_major(rows,HIDDEN))
        for i in range(layer_count):
            _ = enqueue_decoder_layer_configuration(ctx,self.layers[i].attention,
                kv.caches[base+i],self.attention,self.layers[i].mlp,self.mlp,
                TileTensor(self.input,row_major(rows,HIDDEN)),plan.configuration)
            route.layers += 1
            comptime if CAPTURE:
                if request.include_norms:
                    save_bf16(self.attention.normalized,capture+"/attention_norm_"+String(i)+".bin",rows*HIDDEN)
                    save_bf16(self.mlp.normalized,capture+"/mlp_norm_"+String(i)+".bin",rows*HIDDEN)
                    save_bf16(self.attention.output,capture+"/attention_residual_"+String(i)+".bin",rows*HIDDEN)
                save_bf16(self.mlp.output,capture+"/hidden_"+String(i+1)+".bin",rows*HIDDEN)
                save_bf16(self.attention.rotated_key,capture+"/append_key_"+String(i)+".bin",rows*KV_WIDTH)
                save_bf16(self.attention.raw_value,capture+"/append_value_"+String(i)+".bin",rows*KV_WIDTH)
                save_bf16(kv.caches[base+i].key,capture+"/cache_key_"+String(i)+".bin",self.capacity*KV_WIDTH)
                save_bf16(kv.caches[base+i].value,capture+"/cache_value_"+String(i)+".bin",self.capacity*KV_WIDTH)
            if i+1 < layer_count:
                ctx.enqueue_function[_copy_rows[type_of(input_view.layout),type_of(input_view.layout)]](
                    TileTensor(self.mlp.output,row_major(rows,HIDDEN)),
                    TileTensor(self.input,row_major(rows,HIDDEN)),Int32(rows),
                    grid_dim=(rows*HIDDEN+255)//256,block_dim=256)
                route.hidden_copies += 1
        self._mark[OBSERVE](MARK_LAYERS)
        enqueue_rms_norm_apple_gpu(ctx,
            TileTensor(self.mlp.output.unsafe_ptr().unsafe_offset((rows-1)*HIDDEN),row_major(1,HIDDEN)),
            TileTensor(self.norm,row_major(HIDDEN)),TileTensor(self.normalized,row_major(1,HIDDEN)))
        route.final_rms_norm = True
        enqueue_linear_apple_gpu(ctx,TileTensor(self.normalized,row_major(1,HIDDEN)),
            TileTensor(self.embedding,row_major(VOCABULARY,HIDDEN)),TileTensor(self.logits,row_major(1,VOCABULARY)))
        self.gpu_argmax = False
        self._mark[OBSERVE](MARK_HEAD)
        comptime if CAPTURE:
            save_bf16(self.normalized,capture+"/final_norm.bin",HIDDEN)
            save_bf16(self.logits,capture+"/logits.bin",VOCABULARY)
        return route

    def greedy_tokens[OBSERVE: Bool = False](mut self, ctx: DeviceContext) raises -> List[Int]:
        """One greedy token per sequence of the last forward, in batch order.

        Lowest ID on ties. A nonfinite logit in any row invalidates the model, as
        it does for a single sequence, and no token is returned. OBSERVE records
        the same host marks as greedy.
        """
        self._mark[OBSERVE](MARK_GREEDY)
        if not self.valid or not self.ready:
            raise Error("no valid next-token logits")
        var sequences = self.last_route.sequences
        var tokens = List[Int](capacity=sequences)
        if not self.gpu_argmax:
            tokens.append(self.greedy[OBSERVE](ctx))
            return tokens^
        try:
            with self.selection_result.map_to_host() as mapped:
                self._mark[OBSERVE](MARK_MAPPED)
                for s in range(sequences):
                    if mapped.unsafe_ptr()[unsafe_offset=s*3+2] != 0:
                        raise Error("nonfinite model logits in sequence " + String(s))
                    var selected = Int(mapped.unsafe_ptr()[unsafe_offset=s*3+1])
                    if selected < 0 or selected >= VOCABULARY:
                        raise Error("invalid GPU token result")
                    tokens.append(selected)
                self._mark[OBSERVE](MARK_SELECTED)
        except error:
            self.valid = False
            raise error
        self._mark[OBSERVE](MARK_RETURN)
        return tokens^

    def greedy[OBSERVE: Bool = False](mut self, ctx: DeviceContext) raises -> Int:
        """Read the selected route: lowest ID on ties; reject any nonfinite logit."""
        self._mark[OBSERVE](MARK_GREEDY)
        if not self.valid or not self.ready:
            raise Error("no valid next-token logits")
        if self.last_route.sequences != 1:
            raise Error("a step of several sequences selects with greedy_tokens")
        try:
            if self.gpu_argmax:
                var selected: Int
                with self.selection_result.map_to_host() as mapped:
                    self._mark[OBSERVE](MARK_MAPPED)
                    if mapped.unsafe_ptr()[unsafe_offset=2] != 0:
                        raise Error("nonfinite model logits")
                    selected = Int(mapped.unsafe_ptr()[unsafe_offset=1])
                    if selected < 0 or selected >= VOCABULARY:
                        raise Error("invalid GPU token result")
                    self._mark[OBSERVE](MARK_SELECTED)
                self._mark[OBSERVE](MARK_RETURN)
                return selected
            var winner = 0
            var best = Float32(-3.402823466e38)
            with self.logits.map_to_host() as mapped:
                self._mark[OBSERVE](MARK_MAPPED)
                for i in range(VOCABULARY):
                    var value = mapped.unsafe_ptr()[unsafe_offset=i].cast[DType.float32]()
                    if value != value or value > Float32(3.402823466e38) or value < Float32(-3.402823466e38):
                        raise Error("nonfinite model logits")
                    if value > best:
                        best = value
                        winner = i
                self._mark[OBSERVE](MARK_SELECTED)
            self._mark[OBSERVE](MARK_RETURN)
            return winner
        except error:
            self.valid = False
            raise error
