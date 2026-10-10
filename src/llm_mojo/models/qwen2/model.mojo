"""Fixed Qwen model ownership. Native execution; prepared files are verified by tooling.

The model owns weights and workspaces. KV storage and its written slots belong to the
caller's KVPool; each forward receives a StepBatch describing the step, whose block
tables say where each sequence's rows live in the pool. The
cross-layer copy or owner swap keeps the decoder alias contract intact. Numerical
comparisons are diagnostics; storage and lifecycle invariants remain exact.
Which kernels a call uses is decided by an ExecutionPlan (models/qwen2/plan.mojo):
configuration 26 runs the decode composition for one row per sequence;
configuration 27 is an explicit mixed reference composition; other
configurations run one sequence through the generic layer dispatch.
"""
from std.memory import bitcast
from max.gpu import global_idx
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from llm_mojo.layers.attention_sublayer import AttentionWeights, AttentionWorkspace
from llm_mojo.layers.mlp import MLPWeights, MLPWorkspace
from llm_mojo.layers.decoder_layer import (
    DECODER_FUSED_DECODE, DECODER_MIXED, enqueue_decode_batch_layer, enqueue_decoder_layer_configuration_paged,
    validate_decode_batch_layer, validate_decoder_configuration_paged, validate_mixed_layer, enqueue_mixed_layer,
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


def _chain_tokens[ML: TensorLayout, RL: TensorLayout](
    metadata: TileTensor[DType.int32, ML, MutAnyOrigin],
    previous: TileTensor[DType.uint32, RL, ImmutAnyOrigin], rows: Int32,
):
    """Resolve validated negative token references in the ordered device queue.

    A failed previous selection has no usable token. Zero is a safe bounded
    drain input; the previous ticket's collector reports that numerical fault
    before this dependent ticket can deliver any token.
    """
    comptime assert metadata.flat_rank == 1 and previous.flat_rank == 2
    var row = Int(global_idx.x)
    if row < Int(rows):
        var encoded = rebind[Int32](metadata[row])
        if encoded < 0:
            var source = -Int(encoded)-1
            var token = rebind[UInt32](previous[source,1])
            var invalid = rebind[UInt32](previous[source,2])
            metadata[row] = Int32(token) if invalid == 0 and token < UInt32(VOCABULARY) else Int32(0)


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


def _gather_rows[IL: TensorLayout, RL: TensorLayout, OL: TensorLayout](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    rows: TileTensor[DType.int32, RL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin], count: Int32,
):
    comptime assert input.flat_rank == 2 and rows.flat_rank == 1 and output.flat_rank == 2
    var i = global_idx.x
    if i < Int(count)*HIDDEN:
        output[i//HIDDEN,i%HIDDEN] = input[Int(rows[i//HIDDEN]),i%HIDDEN]


def swap_hidden_buffers(mut left: DeviceBuffer[DType.bfloat16], mut right: DeviceBuffer[DType.bfloat16]):
    """Exchange owners, keeping both allocations alive for queued GPU readers."""
    var previous = left^
    left = right^
    right = previous^


def save_sequence_rows(kv: KVPool, table: List[Int], layer: Int, kv_index: Int, rows: Int, path: String,
                       first: Int = 0) raises:
    """Diagnostic readback of a sequence's K (kv_index 0) or V rows first .. first + rows - 1 in one layer.

    Rows come out in position order, each as its KV heads' values in head order,
    whatever the pool's order within a block: for a sequence in one slot-major
    block, the block's region as it lies in memory. Deliberately synchronizes.
    """
    var heads = kv.geometry.kv_heads
    var dim = kv.geometry.head_dim
    var size = kv.block_size
    if first < 0 or rows < 0 or first + rows > len(table) * size:
        raise Error("invalid diagnostic extent")
    var data = List[UInt8](capacity=rows*heads*dim*2)
    var t = first
    while t < first + rows:
        var end = min(first + rows, (t // size + 1) * size)
        var view = kv.view(table[t // size], layer, kv_index)
        with view.map_to_host() as mapped:
            for position in range(t, end):
                var slot = position % size
                for head in range(heads):
                    var start = (head*size + slot)*dim if kv.head_major else (slot*heads + head)*dim
                    for d in range(dim):
                        var bits = bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=start+d])
                        data.append(UInt8(bits & 255))
                        data.append(UInt8(bits >> 8))
        t = end
    var file = open(path,"w")
    file.write_bytes(data)


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
    # One upload: token IDs, positions, physical slots, tables, then sampled rows.
    var step_input: DeviceBuffer[DType.int32]
    # A staged forward has already queued metadata before the layer dispatch.
    # Ordinary synchronous forwards retain their original mapped upload.
    var metadata_queued: Bool
    var capacity: Int
    var max_rows: Int
    var max_sequences: Int
    # Blocks in the widest table: the context in the smallest block size, 32.
    var table_width: Int
    # A successful forward has produced logits that greedy can read.
    var ready: Bool
    var valid: Bool
    var submitted_rows: Int
    var sampled_rows: Int
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
        self.table_width = (capacity + 31) // 32
        self.ready = False
        self.valid = True
        self.submitted_rows = 0
        self.sampled_rows = 0
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
        self.metadata_queued = False
        self.step_input = ctx.enqueue_create_buffer[DType.int32](3*max_rows+max_sequences*self.table_width+max_sequences)

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
        self.sampled_rows = 0

    def preflight(mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan) raises:
        batch.validate(kv.blocks,kv.block_size,VOCABULARY)
        var rows = batch.rows()
        var sequences = batch.sequences()
        var size = kv.block_size
        if not self.valid or rows > self.max_rows or sequences > self.max_sequences:
            raise Error("invalid Qwen state, row extent or context overflow")
        plan.validate(rows, sequences)
        if len(self.layers) < 1:
            raise Error("empty Qwen layer stack")
        if kv.geometry != self.kv_geometry():
            raise Error("KV pool geometry does not match the model")
        # A table of several blocks needs blocks of a multiple of 32 slots, so that no
        # 32-row tile straddles two blocks, and fits the step buffer.
        if batch.max_blocks > self.table_width or (batch.max_blocks > 1 and size % 32 != 0):
            raise Error("block tables need blocks of a multiple of 32 slots, at most one per 32 positions")
        if (len(self.embedding) != VOCABULARY*HIDDEN or len(self.norm) != HIDDEN
            or len(self.input) != self.max_rows*HIDDEN
            or len(self.step_input) != 3*self.max_rows+self.max_sequences*self.table_width+self.max_sequences
            or len(self.normalized) != self.max_sequences*HIDDEN or len(self.logits) != self.max_sequences*VOCABULARY
            or len(self.selection_partials) != self.max_sequences*ARGMAX_GROUPS*3
            or len(self.selection_result) != self.max_sequences*3
            or self.attention.capacity != self.capacity
            or self.attention.max_rows != self.max_rows or self.mlp.max_rows != self.max_rows):
            raise Error("inconsistent model allocation geometry")
        # Each sequence's step starts exactly where its blocks' written slots end:
        # blocks before its first position full, that block holding the positions
        # before it, and later blocks empty.
        for s in range(sequences):
            var past = batch.positions[batch.query_start[s]]
            var length = batch.seq_lens[s]
            if length > self.capacity:
                raise Error("a sequence exceeds the model's context")
            for b in range((length + size - 1) // size):
                var written = kv.written[batch.block_table[s*batch.max_blocks+b]]
                if written != (size if b < past // size else (past % size if b == past // size else 0)):
                    raise Error("inconsistent model cache lengths")
        if plan.configuration == DECODER_MIXED:
            var tail = rows-batch.decode_count
            if (tail == 1 or sequences != batch.decode_count+(1 if tail > 0 else 0)):
                raise Error("mixed execution supports leading singleton sequences and at most one multi-row tail")
            var positions = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows),row_major(rows))
            var slots = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(2*self.max_rows),row_major(rows))
            var tables = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(3*self.max_rows),
                                    row_major(sequences,batch.max_blocks))
            for i in range(len(self.layers)):
                validate_mixed_layer[QUERY_HEADS,KV_HEADS,HEAD_DIM](ctx,self.layers[i].attention,
                    self.attention,self.layers[i].mlp,self.mlp,self.input,kv.storage,
                    positions,slots,tables,batch.decode_count,i,len(self.layers),kv.block_size)
            return
        if plan.configuration == DECODER_FUSED_DECODE:
            if len(batch.logits_rows) != sequences:
                raise Error("decode route requires a logit row for every sequence")
            var positions = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows),row_major(sequences))
            var tables = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(3*self.max_rows),
                                    row_major(sequences,batch.max_blocks))
            for i in range(len(self.layers)):
                validate_decode_batch_layer[QUERY_HEADS,KV_HEADS,HEAD_DIM](ctx,self.layers[i].attention,
                    self.attention,self.layers[i].mlp,self.mlp,self.input,kv.storage,
                    positions,tables,i,len(self.layers),kv.block_size)
            return
        if len(batch.logits_rows) != 1:
            raise Error("single-sequence route requires its final logit row")
        for i in range(len(self.layers)):
            validate_decoder_configuration_paged(ctx,self.layers[i].attention,kv.storage,batch.positions[0],
                self.attention,self.layers[i].mlp,self.mlp,TileTensor(self.input,row_major(rows,HIDDEN)),
                plan.configuration)

    def _upload(mut self, batch: StepBatch) raises:
        """Upload validated per-row metadata, tables, and the requested head rows."""
        if self.metadata_queued:
            return
        var rows = batch.rows()
        var entries = batch.sequences()*batch.max_blocks
        with self.step_input.map_to_host() as mapped:
            for i in range(rows):
                mapped.unsafe_ptr()[unsafe_offset=i] = Int32(batch.token_ids[i])
                mapped.unsafe_ptr()[unsafe_offset=self.max_rows+i] = Int32(batch.positions[i])
                mapped.unsafe_ptr()[unsafe_offset=2*self.max_rows+i] = Int32(batch.slot_mapping[i])
            for i in range(entries):
                mapped.unsafe_ptr()[unsafe_offset=3*self.max_rows+i] = Int32(batch.block_table[i])
            for i in range(len(batch.logits_rows)):
                mapped.unsafe_ptr()[unsafe_offset=3*self.max_rows+self.max_sequences*self.table_width+i] = Int32(batch.logits_rows[i])

    def stage_metadata(self, batch: StepBatch, sources: List[Int], staging: HostBuffer[DType.int32]) raises:
        """Fill an idle pinned source after the caller has preflighted its batch.

        Sources are -1 for a literal token or a validated selected-logit index
        of the preceding ticket. Negative encoded tokens exist only in this
        staging record; StepBatch retains valid placeholder IDs on the host.
        """
        var rows = batch.rows()
        var entries = batch.sequences()*batch.max_blocks
        if (len(staging) != len(self.step_input) or len(sources) != rows or rows > self.max_rows
                or entries > self.max_sequences*self.table_width):
            raise Error("Invalid staged metadata allocation or row extent")
        for i in range(rows):
            staging[i] = Int32(-sources[i]-1 if sources[i] >= 0 else batch.token_ids[i])
            staging[self.max_rows+i] = Int32(batch.positions[i])
            staging[2*self.max_rows+i] = Int32(batch.slot_mapping[i])
        for i in range(entries):
            staging[3*self.max_rows+i] = Int32(batch.block_table[i])
        for i in range(len(batch.logits_rows)):
            staging[3*self.max_rows+self.max_sequences*self.table_width+i] = Int32(batch.logits_rows[i])

    def forward_staged(mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan,
                       staging: HostBuffer[DType.int32], metadata: DeviceBuffer[DType.int32],
                       selected: DeviceBuffer[DType.uint32], previous: DeviceBuffer[DType.uint32],
                       has_source: Bool) raises:
        """Queue an already staged step; its adapter owns prefix dependencies.

        The upload source and destination and both result banks remain alive
        until their context completion. All shared layer workspaces are ordered
        behind the preceding context prefix by QwenAsyncRunner.
        """
        self.preflight(ctx,batch,kv,plan)
        if (self.metadata_queued or (plan.configuration != DECODER_MIXED and plan.configuration != DECODER_FUSED_DECODE)
                or len(metadata) != len(self.step_input) or len(staging) != len(self.step_input)
                or len(selected) != self.max_sequences*3 or len(previous) != self.max_sequences*3):
            raise Error("Invalid staged forward storage or execution route")
        self.step_input = metadata
        self.selection_result = selected
        ctx.enqueue_copy(dst_buf=self.step_input,src_buf=staging)
        if has_source:
            var tokens = TileTensor(self.step_input,row_major(batch.rows()))
            var earlier = TileTensor(previous,row_major(self.max_sequences,3))
            ctx.enqueue_function[_chain_tokens[type_of(tokens.layout),type_of(earlier.layout)]](
                tokens,earlier,Int32(batch.rows()),grid_dim=(batch.rows()+127)//128,block_dim=128)
        self.metadata_queued = True
        try:
            self.forward(ctx,batch,kv,plan)
        except error:
            self.metadata_queued = False
            self.valid = False
            raise error
        self.metadata_queued = False

    @always_inline
    def _mark[OBSERVE: Bool](mut self, slot: Int):
        comptime if OBSERVE:
            self.observation[slot] = now()

    def forward[OBSERVE: Bool = False, PROJECTION: Int = DECODE_PROJECTION](
            mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan) raises:
        """Submit all layers for one step under plan. The step upload synchronizes; layer execution does not.

        The batch's tables name each sequence's blocks in `kv`, whose written
        slots must hold exactly the rows before that sequence's first position;
        the step advances them. With
        plan.gpu_argmax the vocabulary projection is followed by GPU argmax;
        otherwise greedy scans the materialized logits on the CPU. PROJECTION is
        the decode composition's projection arrangement for every row. Exact
        arrangements change how rows share weight loads, not results; reordered
        ones (kernels/linear.mojo) also change each sum's order, the same for
        one row as for many.
        """
        self._forward[OBSERVE, False, PROJECTION](ctx, batch, kv, plan, CaptureRequest("", False))

    def forward_captured[PROJECTION: Int = DECODE_PROJECTION](mut self, ctx: DeviceContext, batch: StepBatch,
                                                              mut kv: KVPool, plan: ExecutionPlan,
                                                              request: CaptureRequest) raises:
        """Diagnostic forward that synchronously writes every boundary under request.directory."""
        if request.directory.byte_length() == 0:
            raise Error("capture requires a directory")
        if batch.sequences() != 1:
            raise Error("capture covers one sequence")
        self._forward[False, True, PROJECTION](ctx, batch, kv, plan, request)

    def _forward[OBSERVE: Bool, CAPTURE: Bool, PROJECTION: Int](mut self, ctx: DeviceContext, batch: StepBatch,
                                                              mut kv: KVPool, plan: ExecutionPlan,
                                                              request: CaptureRequest) raises:
        self._mark[OBSERVE](MARK_START)
        self.preflight(ctx,batch,kv,plan)
        self._mark[OBSERVE](MARK_PREFLIGHT)
        try:
            var route: ForwardRoute
            if plan.configuration == DECODER_MIXED:
                if kv.head_major:
                    route = self._mixed_step[OBSERVE, CAPTURE, PROJECTION, True](ctx, batch, kv, plan, request)
                else:
                    route = self._mixed_step[OBSERVE, CAPTURE, PROJECTION, False](ctx, batch, kv, plan, request)
            elif plan.configuration == DECODER_FUSED_DECODE:
                if kv.head_major:
                    route = self._decode_step[OBSERVE, CAPTURE, PROJECTION, True](ctx, batch, kv, plan, request)
                else:
                    route = self._decode_step[OBSERVE, CAPTURE, PROJECTION, False](ctx, batch, kv, plan, request)
            elif kv.head_major:
                route = self._layer_step[OBSERVE, CAPTURE, True](ctx, batch, kv, plan, request)
            else:
                route = self._layer_step[OBSERVE, CAPTURE, False](ctx, batch, kv, plan, request)
            # Every layer's writes are enqueued: each sequence's blocks now hold its new length.
            for s in range(batch.sequences()):
                var past = batch.positions[batch.query_start[s]]
                var length = batch.seq_lens[s]
                for b in range(past // kv.block_size, (length + kv.block_size - 1) // kv.block_size):
                    kv.written[batch.block_table[s*batch.max_blocks+b]] = min(kv.block_size, length - b*kv.block_size)
            self.ready = True
            self.sampled_rows = len(batch.logits_rows)
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

    def _decode_step[OBSERVE: Bool, CAPTURE: Bool, PROJECTION: Int, HEAD_MAJOR: Bool](
            mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan,
            request: CaptureRequest) raises -> ForwardRoute:
        """Configuration 26: one row per sequence through the decode composition, GPU argmax per row."""
        var sequences = batch.sequences()
        var layer_count = len(self.layers)
        var route = ForwardRoute(plan.configuration, 0, 0, 0, 0, 0, 0, False, False, sequences, 0)
        var capture = request.directory
        self._upload(batch)
        self._mark[OBSERVE](MARK_TOKENS)
        var positions = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows),row_major(sequences))
        var tables = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(3*self.max_rows),
                                row_major(sequences,batch.max_blocks))
        self._embed(ctx, sequences)
        route.decode_launches += 1
        self._mark[OBSERVE](MARK_EMBEDDING)
        comptime if CAPTURE:
            save_bf16(self.input,capture+"/hidden_0.bin",sequences*HIDDEN)
        for i in range(layer_count):
            var normalized_input = i > 0
            route.decode_launches += enqueue_decode_batch_layer[QUERY_HEADS,KV_HEADS,HEAD_DIM,PROJECTION,HEAD_MAJOR](ctx,
                self.layers[i].attention,self.attention,self.layers[i].mlp,self.mlp,self.input,kv.storage,
                positions,tables,i,layer_count,kv.block_size,normalized_input)
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
                # Captures cover one sequence: its appended row and every row of its blocks.
                var table = batch.block_table.copy()
                var held = len(table)*kv.block_size
                save_sequence_rows(kv,table,i,0,1,capture+"/append_key_"+String(i)+".bin",batch.positions[0])
                save_sequence_rows(kv,table,i,1,1,capture+"/append_value_"+String(i)+".bin",batch.positions[0])
                save_sequence_rows(kv,table,i,0,held,capture+"/cache_key_"+String(i)+".bin")
                save_sequence_rows(kv,table,i,1,held,capture+"/cache_value_"+String(i)+".bin")
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

    def _mixed_step[OBSERVE: Bool, CAPTURE: Bool, PROJECTION: Int, HEAD_MAJOR: Bool](
            mut self, ctx: DeviceContext, batch: StepBatch, mut kv: KVPool, plan: ExecutionPlan,
            request: CaptureRequest) raises -> ForwardRoute:
        """Reference composition for leading singleton rows and one optional prompt chunk.

        Only requested sequence endings reach the vocabulary head. Partial
        prompt chunks still advance KV and can return an empty token list.
        """
        var rows = batch.rows()
        var sampled = len(batch.logits_rows)
        var layer_count = len(self.layers)
        var route = ForwardRoute(plan.configuration, 0, 0, 0, 0, 0, 0, False, False, batch.sequences(), 0)
        self._upload(batch)
        self._mark[OBSERVE](MARK_TOKENS)
        var positions = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows),row_major(rows))
        var slots = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(2*self.max_rows),row_major(rows))
        var tables = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(3*self.max_rows),
                                row_major(batch.sequences(),batch.max_blocks))
        self._embed(ctx, rows)
        route.decode_launches += 1
        self._mark[OBSERVE](MARK_EMBEDDING)
        comptime if CAPTURE:
            save_bf16(self.input,request.directory+"/hidden_0.bin",rows*HIDDEN)
        for i in range(layer_count):
            var normalized_input = i > 0
            route.decode_launches += enqueue_mixed_layer[QUERY_HEADS,KV_HEADS,HEAD_DIM,PROJECTION,HEAD_MAJOR](ctx,
                self.layers[i].attention,self.attention,self.layers[i].mlp,self.mlp,self.input,kv.storage,
                positions,slots,tables,batch.decode_count,i,layer_count,kv.block_size,normalized_input)
            route.layers += 1
            if normalized_input:
                route.normalized_inputs += 1
            route.layer_residual_norms += 1
            comptime if CAPTURE:
                if request.include_norms:
                    save_bf16(self.attention.normalized,request.directory+"/attention_norm_"+String(i)+".bin",rows*HIDDEN)
            if i+1 < layer_count:
                enqueue_residual_norm[HIDDEN](ctx,TileTensor(self.attention.output,row_major(rows,HIDDEN)),
                    TileTensor(self.mlp.down,row_major(rows,HIDDEN)),
                    TileTensor(self.layers[i+1].attention.norm,row_major(HIDDEN)),
                    TileTensor(self.mlp.output,row_major(rows,HIDDEN)),
                    TileTensor(self.attention.normalized,row_major(rows,HIDDEN)))
            else:
                enqueue_residual_norm[HIDDEN](ctx,TileTensor(self.attention.output,row_major(rows,HIDDEN)),
                    TileTensor(self.mlp.down,row_major(rows,HIDDEN)),TileTensor(self.norm,row_major(HIDDEN)),
                    TileTensor(self.mlp.output,row_major(rows,HIDDEN)),
                    TileTensor(self.attention.normalized,row_major(rows,HIDDEN)))
            route.deferred_residual_norms += 1
            route.decode_launches += 1
            comptime if CAPTURE:
                save_bf16(self.mlp.output,request.directory+"/hidden_"+String(i+1)+".bin",rows*HIDDEN)
                if request.include_norms:
                    save_bf16(self.mlp.normalized,request.directory+"/mlp_norm_"+String(i)+".bin",rows*HIDDEN)
                    save_bf16(self.attention.output,request.directory+"/attention_residual_"+String(i)+".bin",rows*HIDDEN)
                save_sequence_rows(kv,batch.block_table,i,0,rows,request.directory+"/append_key_"+String(i)+".bin",batch.positions[0])
                save_sequence_rows(kv,batch.block_table,i,1,rows,request.directory+"/append_value_"+String(i)+".bin",batch.positions[0])
                save_sequence_rows(kv,batch.block_table,i,0,batch.max_blocks*kv.block_size,request.directory+"/cache_key_"+String(i)+".bin")
                save_sequence_rows(kv,batch.block_table,i,1,batch.max_blocks*kv.block_size,request.directory+"/cache_value_"+String(i)+".bin")
            if i+1 < layer_count:
                swap_hidden_buffers(self.input,self.mlp.output)
                route.owner_swaps += 1
        self._mark[OBSERVE](MARK_LAYERS)
        self.gpu_argmax = True
        if sampled > 0:
            var head_rows = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(
                3*self.max_rows+self.max_sequences*self.table_width),row_major(sampled))
            var all_normal = TileTensor(self.attention.normalized,row_major(rows,HIDDEN))
            var selected = TileTensor(self.normalized,row_major(sampled,HIDDEN))
            comptime gather = _gather_rows[type_of(all_normal.layout),type_of(head_rows.layout),type_of(selected.layout)]
            ctx.enqueue_function[gather](all_normal,head_rows,selected,Int32(sampled),
                                         grid_dim=(sampled*HIDDEN+255)//256,block_dim=256)
            var logits = TileTensor(self.logits,row_major(sampled,VOCABULARY))
            enqueue_linear_decode_rows_apple_gpu[PROJECTION](ctx,selected,TileTensor(self.embedding,row_major(VOCABULARY,HIDDEN)),logits)
            enqueue_argmax(ctx,logits,TileTensor(self.selection_partials,row_major(sampled*ARGMAX_GROUPS,3)),
                TileTensor(self.selection_result,row_major(sampled,3)))
            route.decode_launches += 4
            route.gpu_argmax = True
            comptime if CAPTURE:
                save_bf16(self.normalized,request.directory+"/final_norm.bin",sampled*HIDDEN)
                save_bf16(self.logits,request.directory+"/logits.bin",sampled*VOCABULARY)
        self._mark[OBSERVE](MARK_HEAD)
        return route

    def _layer_step[OBSERVE: Bool, CAPTURE: Bool, HEAD_MAJOR: Bool](mut self, ctx: DeviceContext, batch: StepBatch,
                                                                   mut kv: KVPool, plan: ExecutionPlan,
                                                                   request: CaptureRequest) raises -> ForwardRoute:
        """Every other configuration: one sequence through the generic layer dispatch and a CPU greedy."""
        var rows = batch.rows()
        var layer_count = len(self.layers)
        var route = ForwardRoute(plan.configuration, 0, 0, 0, 0, 0, 0, False, False, 1, 0)
        var capture = request.directory
        var past = batch.positions[0]
        self._upload(batch)
        self._mark[OBSERVE](MARK_TOKENS)
        var positions = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(self.max_rows),row_major(rows))
        var table = TileTensor(self.step_input.unsafe_ptr().unsafe_offset(3*self.max_rows),row_major(batch.max_blocks))
        self._embed(ctx, rows)
        self._mark[OBSERVE](MARK_EMBEDDING)
        comptime if CAPTURE:
            save_bf16(self.input,capture+"/hidden_0.bin",rows*HIDDEN)
        var input_view = TileTensor(self.input,row_major(rows,HIDDEN))
        for i in range(layer_count):
            _ = enqueue_decoder_layer_configuration_paged[HEAD_MAJOR](ctx,self.layers[i].attention,kv.storage,
                table,positions,past,i,layer_count,kv.block_size,self.attention,self.layers[i].mlp,self.mlp,
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
                save_sequence_rows(kv,batch.block_table,i,0,batch.max_blocks*kv.block_size,
                                   capture+"/cache_key_"+String(i)+".bin")
                save_sequence_rows(kv,batch.block_table,i,1,batch.max_blocks*kv.block_size,
                                   capture+"/cache_value_"+String(i)+".bin")
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
        """One greedy token per requested logit row, in increasing input-row order.

        Lowest ID on ties. A nonfinite logit in any row invalidates the model, as
        it does for a single sequence, and no token is returned. OBSERVE records
        the same host marks as greedy.
        """
        self._mark[OBSERVE](MARK_GREEDY)
        if not self.valid or not self.ready:
            raise Error("no valid next-token logits")
        var sequences = self.sampled_rows
        var tokens = List[Int](capacity=sequences)
        if sequences == 0:
            try:
                ctx.synchronize()
            except error:
                self.valid = False
                raise error
            return tokens^
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
        if self.sampled_rows != 1:
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
