"""Fixed Qwen model ownership. Native execution; prepared files are verified by tooling.

The cross-layer copy or owner swap keeps the decoder alias contract intact. Numerical
comparisons are diagnostics; storage and lifecycle invariants remain exact.
Which kernels a call uses is decided by an ExecutionPlan (models/qwen2/plan.mojo).
"""
from std.memory import bitcast
from std.gpu import global_idx
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.layers.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace
from llm_mojo.layers.mlp import MLPWeights, MLPWorkspace
from llm_mojo.layers.decoder_layer import (
    DECODER_FUSED_DECODE, decoder_mappings, enqueue_decoder_layer_configuration,
    validate_decoder_configuration,
)
from llm_mojo.kernels.residual_norm import enqueue_residual_norm
from llm_mojo.kernels.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.kernels.linear import enqueue_linear_apple_gpu
from llm_mojo.kernels.token_selection import enqueue_argmax
from llm_mojo.models.qwen2.plan import ExecutionPlan
from llm_mojo.runtime.clock import now

comptime HIDDEN = 896
comptime VOCABULARY = 151936
comptime LAYERS = 24
comptime KV_WIDTH = 128
comptime MAX_CONTEXT = 4096
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
    """What the last forward actually enqueued, counted where each dispatch is issued."""
    var configuration: Int
    var layers: Int
    var normalized_inputs: Int
    var layer_residual_norms: Int
    var deferred_residual_norms: Int
    var owner_swaps: Int
    var hidden_copies: Int
    var final_rms_norm: Bool
    var gpu_argmax: Bool

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
    var cache: AttentionCache

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        self.attention = AttentionWeights(ctx)
        self.mlp = MLPWeights(ctx)
        self.cache = AttentionCache(ctx,capacity)

    def __init__(out self, ctx: DeviceContext, path: String, index: Int, capacity: Int) raises:
        self = Self(ctx,capacity)
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
    var tokens: DeviceBuffer[DType.int32]
    var capacity: Int
    var max_rows: Int
    var length: Int
    var valid: Bool
    var submitted_rows: Int
    var last_route: ForwardRoute
    # Host-only observation storage. Default specializations contain no clocks.
    var observation: List[UInt64]

    def __init__(out self, ctx: DeviceContext, layer_count: Int, capacity: Int, max_rows: Int) raises:
        """Allocate without loading; use the path constructor for the prepared checkpoint."""
        if (ctx.api() != "metal" or layer_count < 1 or capacity < 1 or capacity > MAX_CONTEXT
                or max_rows < 1 or max_rows > capacity):
            raise Error("Qwen requires Metal and valid layer, row and context capacity")
        self.capacity = capacity
        self.max_rows = max_rows
        self.length = 0
        self.valid = True
        self.submitted_rows = 0
        self.last_route = ForwardRoute(-1, 0, 0, 0, 0, 0, 0, False, False)
        self.observation = List[UInt64](capacity=10)
        for _ in range(10):
            self.observation.append(0)
        self.embedding = ctx.enqueue_create_buffer[DType.bfloat16](VOCABULARY*HIDDEN)
        self.norm = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
        self.layers = List[ModelLayer](capacity=layer_count)
        for _ in range(layer_count):
            self.layers.append(ModelLayer(ctx,capacity))
        self.attention = AttentionWorkspace(ctx,max_rows,capacity,
            materialized=False,fp32_materialized=False,prefill_splits=8)
        self.mlp = MLPWorkspace(ctx,max_rows)
        self.input = ctx.enqueue_create_buffer[DType.bfloat16](max_rows*HIDDEN)
        self.normalized = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
        self.logits = ctx.enqueue_create_buffer[DType.bfloat16](VOCABULARY)
        self.selection_partials = ctx.enqueue_create_buffer[DType.uint32](ARGMAX_GROUPS*3)
        self.selection_result = ctx.enqueue_create_buffer[DType.uint32](3)
        self.gpu_argmax = False
        self.tokens = ctx.enqueue_create_buffer[DType.int32](max_rows)

    def __init__(out self, ctx: DeviceContext, path: String, capacity: Int, max_rows: Int) raises:
        """Allocate all 24 layers and load the verified prepared checkpoint at path."""
        self = Self(ctx, LAYERS, capacity, max_rows)
        load_bf16(self.embedding,path+"/embedding.bin")
        load_bf16(self.norm,path+"/final_norm.bin")
        for i in range(LAYERS):
            self.layers[i].load(path,i)
        # Preparation stores all 4096 positions; only capacity rows are resident.
        load_bf16(self.attention.cosine,path+"/cosine.bin",MAX_CONTEXT*64)
        load_bf16(self.attention.sine,path+"/sine.bin",MAX_CONTEXT*64)
        ctx.synchronize()

    @staticmethod
    def allocate(ctx: DeviceContext, layer_count: Int, capacity: Int, max_rows: Int) raises -> QwenModel:
        """Unloaded model storage for tests that supply their own weights."""
        return QwenModel(ctx, layer_count, capacity, max_rows)

    def reset(mut self, ctx: DeviceContext) raises:
        self.valid = False
        ctx.synchronize()
        for i in range(len(self.layers)):
            self.layers[i].cache.length = 0
        self.length = 0
        self.valid = True
        self.submitted_rows = 0

    def preflight(mut self, ctx: DeviceContext, ids: List[Int], plan: ExecutionPlan) raises:
        var rows = len(ids)
        if not self.valid or rows < 1 or rows > self.max_rows or rows > self.capacity-self.length:
            raise Error("invalid Qwen state, row extent or context overflow")
        plan.validate(rows)
        if (len(self.embedding) != VOCABULARY*HIDDEN or len(self.norm) != HIDDEN
            or len(self.input) != self.max_rows*HIDDEN or len(self.tokens) != self.max_rows
            or len(self.normalized) != HIDDEN or len(self.logits) != VOCABULARY
            or len(self.selection_partials) != ARGMAX_GROUPS*3 or len(self.selection_result) != 3
            or self.attention.capacity != self.capacity
            or self.attention.max_rows != self.max_rows or self.mlp.max_rows != self.max_rows):
            raise Error("inconsistent model allocation geometry")
        for id in ids:
            if id < 0 or id >= VOCABULARY:
                raise Error("model token ID out of range")
        var mappings = decoder_mappings(plan.configuration,rows)
        if len(self.layers) < 1:
            raise Error("empty Qwen layer stack")
        for i in range(len(self.layers)):
            if self.layers[i].cache.length != self.length or self.layers[i].cache.capacity != self.capacity:
                raise Error("inconsistent model cache lengths")
            validate_decoder_configuration(ctx,self.layers[i].attention,self.layers[i].cache,self.attention,
                self.layers[i].mlp,self.mlp,TileTensor(self.input,row_major(rows,HIDDEN)),
                Int(mappings[0]),Int(mappings[1]),Int(mappings[2]))

    @always_inline
    def _mark[OBSERVE: Bool](mut self, slot: Int):
        comptime if OBSERVE:
            self.observation[slot] = now()

    def forward[OBSERVE: Bool = False](mut self, ctx: DeviceContext, ids: List[Int], plan: ExecutionPlan) raises:
        """Submit all layers under plan. Token upload synchronizes; layer execution does not.

        With plan.gpu_argmax the vocabulary projection is followed by GPU argmax;
        otherwise greedy scans the materialized logits on the CPU.
        """
        self._forward[OBSERVE, False](ctx, ids, plan, CaptureRequest("", False))

    def forward_captured(mut self, ctx: DeviceContext, ids: List[Int], plan: ExecutionPlan,
                         request: CaptureRequest) raises:
        """Diagnostic forward that synchronously writes every boundary under request.directory."""
        if request.directory.byte_length() == 0:
            raise Error("capture requires a directory")
        self._forward[False, True](ctx, ids, plan, request)

    def _forward[OBSERVE: Bool, CAPTURE: Bool](mut self, ctx: DeviceContext, ids: List[Int],
                                               plan: ExecutionPlan, request: CaptureRequest) raises:
        self._mark[OBSERVE](MARK_START)
        self.preflight(ctx,ids,plan)
        self._mark[OBSERVE](MARK_PREFLIGHT)
        var rows = len(ids)
        var layer_count = len(self.layers)
        var fuse_norm = plan.fuse_residual_norm
        var route = ForwardRoute(plan.configuration, 0, 0, 0, 0, 0, 0, False, False)
        var capture = request.directory
        try:
            with self.tokens.map_to_host() as mapped:
                for i in range(rows):
                    mapped.unsafe_ptr()[unsafe_offset=i] = Int32(ids[i])
            self._mark[OBSERVE](MARK_TOKENS)
            var token_view = TileTensor(self.tokens,row_major(rows))
            var weight_view = TileTensor(self.embedding,row_major(VOCABULARY,HIDDEN))
            var input_view = TileTensor(self.input,row_major(rows,HIDDEN))
            comptime embedding_kernel = _embedding[type_of(token_view.layout),type_of(weight_view.layout),type_of(input_view.layout)]
            ctx.enqueue_function[embedding_kernel](token_view,weight_view,input_view,Int32(rows),
                grid_dim=(rows*HIDDEN+255)//256,block_dim=256)
            self._mark[OBSERVE](MARK_EMBEDDING)
            comptime if CAPTURE:
                save_bf16(self.input,capture+"/hidden_0.bin",rows*HIDDEN)
            for i in range(layer_count):
                var normalized_input = fuse_norm and i > 0
                _ = enqueue_decoder_layer_configuration(ctx,self.layers[i].attention,
                    self.layers[i].cache,self.attention,self.layers[i].mlp,self.mlp,
                    TileTensor(self.input,row_major(rows,HIDDEN)),plan.configuration,fuse_norm,normalized_input)
                route.layers += 1
                if normalized_input:
                    route.normalized_inputs += 1
                if fuse_norm:
                    route.layer_residual_norms += 1
                comptime if CAPTURE:
                    if request.include_norms:
                        save_bf16(self.attention.normalized,capture+"/attention_norm_"+String(i)+".bin",rows*HIDDEN)
                        save_bf16(self.mlp.normalized,capture+"/mlp_norm_"+String(i)+".bin",rows*HIDDEN)
                        save_bf16(self.attention.output,capture+"/attention_residual_"+String(i)+".bin",rows*HIDDEN)
                if fuse_norm:
                    # The MLP residual feeds the next layer's attention norm, or the final norm.
                    if i+1 < layer_count:
                        enqueue_residual_norm(ctx,TileTensor(self.attention.output,row_major(1,HIDDEN)),
                            TileTensor(self.mlp.down,row_major(1,HIDDEN)),
                            TileTensor(self.layers[i+1].attention.norm,row_major(HIDDEN)),
                            TileTensor(self.mlp.output,row_major(1,HIDDEN)),
                            TileTensor(self.attention.normalized,row_major(1,HIDDEN)))
                    else:
                        enqueue_residual_norm(ctx,TileTensor(self.attention.output,row_major(1,HIDDEN)),
                            TileTensor(self.mlp.down,row_major(1,HIDDEN)),TileTensor(self.norm,row_major(HIDDEN)),
                            TileTensor(self.mlp.output,row_major(1,HIDDEN)),
                            TileTensor(self.normalized,row_major(1,HIDDEN)))
                    route.deferred_residual_norms += 1
                comptime if CAPTURE:
                    save_bf16(self.mlp.output,capture+"/hidden_"+String(i+1)+".bin",rows*HIDDEN)
                    if plan.configuration == DECODER_FUSED_DECODE or plan.configuration == 25:
                        # Fusion intentionally leaves the unpack/rotated scratch untouched.
                        var appended = (self.layers[i].cache.length-1)*KV_WIDTH
                        save_bf16(self.layers[i].cache.key,capture+"/append_key_"+String(i)+".bin",KV_WIDTH,appended)
                        save_bf16(self.layers[i].cache.value,capture+"/append_value_"+String(i)+".bin",KV_WIDTH,appended)
                    else:
                        save_bf16(self.attention.rotated_key,capture+"/append_key_"+String(i)+".bin",rows*KV_WIDTH)
                        save_bf16(self.attention.raw_value,capture+"/append_value_"+String(i)+".bin",rows*KV_WIDTH)
                    save_bf16(self.layers[i].cache.key,capture+"/cache_key_"+String(i)+".bin",self.capacity*KV_WIDTH)
                    save_bf16(self.layers[i].cache.value,capture+"/cache_value_"+String(i)+".bin",self.capacity*KV_WIDTH)
                if i+1 < layer_count:
                    if plan.swap_buffers:
                        swap_hidden_buffers(self.input,self.mlp.output)
                        route.owner_swaps += 1
                    else:
                        ctx.enqueue_function[_copy_rows[type_of(input_view.layout),type_of(input_view.layout)]](
                            TileTensor(self.mlp.output,row_major(rows,HIDDEN)),
                            TileTensor(self.input,row_major(rows,HIDDEN)),Int32(rows),
                            grid_dim=(rows*HIDDEN+255)//256,block_dim=256)
                        route.hidden_copies += 1
            self._mark[OBSERVE](MARK_LAYERS)
            if not fuse_norm:
                enqueue_rms_norm_apple_gpu(ctx,
                    TileTensor(self.mlp.output.unsafe_ptr().unsafe_offset((rows-1)*HIDDEN),row_major(1,HIDDEN)),
                    TileTensor(self.norm,row_major(HIDDEN)),TileTensor(self.normalized,row_major(1,HIDDEN)))
                route.final_rms_norm = True
            var logits = TileTensor(self.logits,row_major(1,VOCABULARY))
            enqueue_linear_apple_gpu(ctx,TileTensor(self.normalized,row_major(1,HIDDEN)),
                TileTensor(self.embedding,row_major(VOCABULARY,HIDDEN)),logits)
            if plan.gpu_argmax:
                enqueue_argmax(ctx,logits,TileTensor(self.selection_partials,row_major(ARGMAX_GROUPS,3)),
                    TileTensor(self.selection_result,row_major(1,3)))
                route.gpu_argmax = True
            self.gpu_argmax = plan.gpu_argmax
            self._mark[OBSERVE](MARK_HEAD)
            comptime if CAPTURE:
                save_bf16(self.normalized,capture+"/final_norm.bin",HIDDEN)
                save_bf16(self.logits,capture+"/logits.bin",VOCABULARY)
            self.length += rows
            self.submitted_rows += rows*layer_count
            self.last_route = route
        except error:
            self.valid = False
            raise error

    def greedy[OBSERVE: Bool = False](mut self, ctx: DeviceContext) raises -> Int:
        """Read the selected route: lowest ID on ties; reject any nonfinite logit."""
        self._mark[OBSERVE](MARK_GREEDY)
        if not self.valid or self.length == 0:
            raise Error("no valid next-token logits")
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
