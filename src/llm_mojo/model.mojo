"""Fixed Qwen model ownership. Native execution; prepared files are verified by tooling.

The cross-layer copy or owner swap keeps the decoder alias contract intact. Numerical
comparisons are diagnostics; storage and lifecycle invariants remain exact.
"""
from std.memory import bitcast
from std.ffi import external_call
from std.gpu import global_idx
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace
from llm_mojo.mlp import MLPWeights, MLPWorkspace
from llm_mojo.decoder_layer import _decoder_preflight, decoder_mappings, enqueue_decoder_layer_configuration
from llm_mojo.residual_norm import enqueue_residual_norm
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.linear import enqueue_linear_apple_gpu
from llm_mojo.token_selection import enqueue_argmax, enqueue_head_argmax


def _observation_clock() -> UInt64:
    return external_call["clock_gettime_nsec_np", UInt64](UInt32(8))


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
    if i < Int(count) * 896:
        var row = i // 896
        output[row,i % 896] = weight[Int(ids[row]),i % 896]


def _copy_rows[IL: TensorLayout, OL: TensorLayout](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin], count: Int32,
):
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    var i = global_idx.x
    if i < Int(count)*896:
        output[i//896,i%896] = input[i//896,i%896]


def swap_hidden_buffers(mut left: DeviceBuffer[DType.bfloat16], mut right: DeviceBuffer[DType.bfloat16]):
    """Exchange owners, keeping both allocations alive for queued GPU readers."""
    var previous = left^
    left = right^
    right = previous^


def select_copy_free(policy: String, rows: Int, device: String) -> Bool:
    return (policy == "fast" or policy == "auto" or policy == "buffer-swap" or policy == "swap-argmax" or policy == "all-three") and rows == 1 and device == "Apple M4 Pro"


def select_residual_norm(policy: String, rows: Int, device: String) -> Bool:
    return (policy == "fast" or policy == "auto" or policy == "residual-norm" or policy == "all-three") and rows == 1 and device == "Apple M4 Pro"


def candidate_configuration(rows: Int, total: Int) -> Int:
    """Split8 choices measured in the real 24-layer model on Apple M4 Pro."""
    if (rows == 16 and (total == 1024 or total == 4096)) or (
        total == 256 and (rows == 15 or rows == 17)
    ):
        return 2
    if ((rows == 64 or rows == 256) and (total == 1024 or total == 4096)) or (
        total == 4096 and (rows == 65 or rows == 255)
    ):
        return 3
    return 0


def select_configuration(policy: String, rows: Int, total: Int, device: String) raises -> Int:
    if rows < 1 or total < rows or total > 4096:
        raise Error("invalid configuration-selection dimensions")
    if policy == "gpu-argmax" or policy == "fused-head" or policy == "buffer-swap" or policy == "residual-norm" or policy == "swap-argmax" or policy == "all-three":
        return select_configuration("fast",rows,total,device)
    if policy == "fusion" or policy == "combined" or policy == "unfused":
        if rows == 1 and device == "Apple M4 Pro":
            if policy == "unfused":
                return 0
            return 26 if policy == "combined" else 25
        return select_configuration("fast", rows, total, device)
    if policy == "baseline":
        return 0
    if policy == "auto" or policy == "fast":
        if device != "Apple M4 Pro":
            return 0
        # Exact QKV + activation fusion; paired full-model gate at 64/1024/3968.
        if rows == 1:
            return 26
        # Full-model paired measurements, including baseline self-comparisons.
        # Every other measured winner is in the shared split8 lookup below.
        if rows == 16 and total == 256:
            return 21
        return candidate_configuration(rows,total)
    if policy == "consistent" or policy == "20":
        return 20
    if policy == "candidate":
        return candidate_configuration(rows,total) if device == "Apple M4 Pro" else 0
    if policy == "0" or policy == "2" or policy == "3" or policy == "21":
        return Int(policy)
    raise Error("unknown generation configuration policy")


def select_token_selection(policy: String, rows: Int, device: String) raises -> Int:
    if rows < 1:
        raise Error("invalid selection row count")
    if rows == 1 and device == "Apple M4 Pro":
        if policy == "fast" or policy == "auto" or policy == "gpu-argmax" or policy == "swap-argmax" or policy == "all-three":
            return 1
        if policy == "fused-head":
            return 2
    return 0


struct ModelLayer(Movable):
    var attention: AttentionWeights
    var mlp: MLPWeights
    var cache: AttentionCache

    def __init__(out self, ctx: DeviceContext, path: String, index: Int, capacity: Int) raises:
        self.attention = AttentionWeights(ctx)
        self.mlp = MLPWeights(ctx)
        self.cache = AttentionCache(ctx,capacity)
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
    var selection: Int
    var tokens: DeviceBuffer[DType.int32]
    var capacity: Int
    var max_rows: Int
    var length: Int
    var valid: Bool
    var submitted_rows: Int
    # Host-only observation storage. Default specializations contain no clocks.
    var observation: List[UInt64]

    def __init__(out self, ctx: DeviceContext, path: String, capacity: Int, max_rows: Int) raises:
        if ctx.api() != "metal" or capacity < 1 or capacity > 4096 or max_rows < 1 or max_rows > capacity:
            raise Error("Qwen requires Metal and valid row/context capacity")
        self.capacity = capacity
        self.max_rows = max_rows
        self.length = 0
        self.valid = True
        self.submitted_rows = 0
        self.observation = List[UInt64](capacity=10)
        for _ in range(10):
            self.observation.append(0)
        self.embedding = ctx.enqueue_create_buffer[DType.bfloat16](151936*896)
        self.norm = ctx.enqueue_create_buffer[DType.bfloat16](896)
        load_bf16(self.embedding,path+"/embedding.bin")
        load_bf16(self.norm,path+"/final_norm.bin")
        self.layers = List[ModelLayer](capacity=24)
        for i in range(24):
            self.layers.append(ModelLayer(ctx,path,i,capacity))
        self.attention = AttentionWorkspace(ctx,max_rows,capacity,
            materialized=False,fp32_materialized=False,prefill_splits=8)
        self.mlp = MLPWorkspace(ctx,max_rows)
        self.input = ctx.enqueue_create_buffer[DType.bfloat16](max_rows*896)
        self.normalized = ctx.enqueue_create_buffer[DType.bfloat16](896)
        self.logits = ctx.enqueue_create_buffer[DType.bfloat16](151936)
        self.selection_partials = ctx.enqueue_create_buffer[DType.uint32](2374*3)
        self.selection_result = ctx.enqueue_create_buffer[DType.uint32](3)
        self.selection = 0
        self.tokens = ctx.enqueue_create_buffer[DType.int32](max_rows)
        # Preparation stores all 4096 positions; only capacity rows are resident.
        load_bf16(self.attention.cosine,path+"/cosine.bin",4096*64)
        load_bf16(self.attention.sine,path+"/sine.bin",4096*64)
        ctx.synchronize()

    def reset(mut self, ctx: DeviceContext) raises:
        self.valid = False
        ctx.synchronize()
        for i in range(len(self.layers)):
            self.layers[i].cache.length = 0
        self.length = 0
        self.valid = True
        self.submitted_rows = 0

    def preflight(mut self, ctx: DeviceContext, ids: List[Int], configuration: Int) raises:
        var rows = len(ids)
        if not self.valid or rows < 1 or rows > self.max_rows or rows > self.capacity-self.length:
            raise Error("invalid Qwen state, row extent or context overflow")
        if (len(self.embedding) != 151936*896 or len(self.norm) != 896
            or len(self.input) != self.max_rows*896 or len(self.tokens) != self.max_rows
            or len(self.normalized) != 896 or len(self.logits) != 151936
            or len(self.selection_partials) != 2374*3 or len(self.selection_result) != 3
            or self.attention.capacity != self.capacity
            or self.attention.max_rows != self.max_rows or self.mlp.max_rows != self.max_rows):
            raise Error("inconsistent model allocation geometry")
        for id in ids:
            if id < 0 or id >= 151936:
                raise Error("model token ID out of range")
        var mappings = decoder_mappings(configuration,rows)
        if len(self.layers) != 24:
            raise Error("incomplete Qwen layer stack")
        for i in range(len(self.layers)):
            if self.layers[i].cache.length != self.length or self.layers[i].cache.capacity != self.capacity:
                raise Error("inconsistent model cache lengths")
            _decoder_preflight(ctx,self.layers[i].attention,self.layers[i].cache,self.attention,self.layers[i].mlp,self.mlp,
                TileTensor(self.input,row_major(rows,896)),True,
                Int(mappings[0]),Int(mappings[1]),Int(mappings[2]))

    def forward[OBSERVE: Bool = False](mut self, ctx: DeviceContext, ids: List[Int], configuration: Int = 0, capture: String = "", selection: Int = 0, materialize: Bool = False, copy_free: Bool = False, fuse_residual_norm: Bool = False, capture_norm: Bool = False) raises:
        """Submit all layers. ID upload synchronizes; layer execution does not.

        Native inference API. The host token
        staging boundary is measured separately from a future enqueue API.
        Selection 0 materializes logits for CPU greedy; 1 adds GPU argmax;
        2 fuses projection/local selection and leaves logits untouched unless
        materialize or capture is requested. greedy reads the matching result.
        """
        comptime if OBSERVE:
            self.observation[0] = _observation_clock()
        if fuse_residual_norm and (len(ids) != 1 or configuration != 26 or selection == 2):
            raise Error("residual RMSNorm fusion requires one configuration-26 row and ordinary vocabulary projection")
        if copy_free and len(ids) != 1:
            raise Error("buffer swapping is restricted to single-row decode")
        if selection < 0 or selection > 2:
            raise Error("unknown token selection mode")
        self.preflight(ctx,ids,configuration)
        comptime if OBSERVE:
            self.observation[1] = _observation_clock()
        var rows = len(ids)
        try:
            with self.tokens.map_to_host() as mapped:
                for i in range(rows):
                    mapped.unsafe_ptr()[unsafe_offset=i] = Int32(ids[i])
            comptime if OBSERVE:
                self.observation[2] = _observation_clock()
            var token_view = TileTensor(self.tokens,row_major(rows))
            var weight_view = TileTensor(self.embedding,row_major(151936,896))
            var input_view = TileTensor(self.input,row_major(rows,896))
            comptime embedding_kernel = _embedding[type_of(token_view.layout),type_of(weight_view.layout),type_of(input_view.layout)]
            ctx.enqueue_function[embedding_kernel](token_view,weight_view,input_view,Int32(rows),
                grid_dim=(rows*896+255)//256,block_dim=256)
            comptime if OBSERVE:
                self.observation[3] = _observation_clock()
            if capture.byte_length() > 0:
                save_bf16(self.input,capture+"/hidden_0.bin",rows*896)
            for i in range(24):
                _ = enqueue_decoder_layer_configuration(ctx,self.layers[i].attention,
                    self.layers[i].cache,self.attention,self.layers[i].mlp,self.mlp,
                    TileTensor(self.input,row_major(rows,896)),configuration,fuse_residual_norm,fuse_residual_norm and i > 0)
                if capture_norm and capture.byte_length() > 0:
                    save_bf16(self.attention.normalized,capture+"/attention_norm_"+String(i)+".bin",rows*896)
                    save_bf16(self.mlp.normalized,capture+"/mlp_norm_"+String(i)+".bin",rows*896)
                    save_bf16(self.attention.output,capture+"/attention_residual_"+String(i)+".bin",rows*896)
                if fuse_residual_norm:
                    if i < 23:
                        enqueue_residual_norm(ctx,TileTensor(self.attention.output,row_major(1,896)),
                            TileTensor(self.mlp.down,row_major(1,896)),TileTensor(self.layers[i+1].attention.norm,row_major(896)),
                            TileTensor(self.mlp.output,row_major(1,896)),TileTensor(self.attention.normalized,row_major(1,896)))
                    else:
                        enqueue_residual_norm(ctx,TileTensor(self.attention.output,row_major(1,896)),
                            TileTensor(self.mlp.down,row_major(1,896)),TileTensor(self.norm,row_major(896)),
                            TileTensor(self.mlp.output,row_major(1,896)),TileTensor(self.normalized,row_major(1,896)))
                if capture.byte_length() > 0:
                    save_bf16(self.mlp.output,capture+"/hidden_"+String(i+1)+".bin",rows*896)
                    if configuration == 25 or configuration == 26:
                        # Fusion intentionally leaves the unpack/rotated scratch untouched.
                        save_bf16(self.layers[i].cache.key,capture+"/append_key_"+String(i)+".bin",128,(self.layers[i].cache.length-1)*128)
                        save_bf16(self.layers[i].cache.value,capture+"/append_value_"+String(i)+".bin",128,(self.layers[i].cache.length-1)*128)
                    else:
                        save_bf16(self.attention.rotated_key,capture+"/append_key_"+String(i)+".bin",rows*128)
                        save_bf16(self.attention.raw_value,capture+"/append_value_"+String(i)+".bin",rows*128)
                    save_bf16(self.layers[i].cache.key,capture+"/cache_key_"+String(i)+".bin",self.capacity*128)
                    save_bf16(self.layers[i].cache.value,capture+"/cache_value_"+String(i)+".bin",self.capacity*128)
                if i < 23 and copy_free:
                    swap_hidden_buffers(self.input,self.mlp.output)
                elif i < 23:
                    ctx.enqueue_function[_copy_rows[type_of(input_view.layout),type_of(input_view.layout)]](TileTensor(self.mlp.output,row_major(rows,896)),
                        TileTensor(self.input,row_major(rows,896)),Int32(rows),
                        grid_dim=(rows*896+255)//256,block_dim=256)
            comptime if OBSERVE:
                self.observation[4] = _observation_clock()
            if not fuse_residual_norm:
                enqueue_rms_norm_apple_gpu(ctx,
                    TileTensor(self.mlp.output.unsafe_ptr().unsafe_offset((rows-1)*896),row_major(1,896)),
                    TileTensor(self.norm,row_major(896)),TileTensor(self.normalized,row_major(1,896)))
            var partials = TileTensor(self.selection_partials,row_major(2374,3))
            var result = TileTensor(self.selection_result,row_major(1,3))
            var logits = TileTensor(self.logits,row_major(1,151936))
            if selection == 2:
                if materialize or capture.byte_length() > 0:
                    enqueue_head_argmax[True](ctx,TileTensor(self.normalized,row_major(1,896)),
                        TileTensor(self.embedding,row_major(151936,896)),logits,partials,result)
                else:
                    enqueue_head_argmax[False](ctx,TileTensor(self.normalized,row_major(1,896)),
                        TileTensor(self.embedding,row_major(151936,896)),logits,partials,result)
            else:
                enqueue_linear_apple_gpu(ctx,TileTensor(self.normalized,row_major(1,896)),
                    TileTensor(self.embedding,row_major(151936,896)),logits)
                if selection == 1:
                    enqueue_argmax(ctx,logits,partials,result)
            self.selection = selection
            comptime if OBSERVE:
                self.observation[5] = _observation_clock()
            if capture.byte_length() > 0:
                save_bf16(self.normalized,capture+"/final_norm.bin",896)
                save_bf16(self.logits,capture+"/logits.bin",151936)
            self.length += rows
            self.submitted_rows += rows*24
        except error:
            self.valid = False
            raise error

    def greedy[OBSERVE: Bool = False](mut self, ctx: DeviceContext) raises -> Int:
        """Read the selected route: lowest ID on ties; reject any nonfinite logit."""
        comptime if OBSERVE:
            self.observation[6] = _observation_clock()
        if not self.valid or self.length == 0:
            raise Error("no valid next-token logits")
        try:
            if self.selection != 0:
                var selected: Int
                with self.selection_result.map_to_host() as mapped:
                    comptime if OBSERVE:
                        self.observation[7] = _observation_clock()
                    if mapped.unsafe_ptr()[unsafe_offset=2] != 0:
                        raise Error("nonfinite model logits")
                    selected = Int(mapped.unsafe_ptr()[unsafe_offset=1])
                    if selected < 0 or selected >= 151936:
                        raise Error("invalid GPU token result")
                    comptime if OBSERVE:
                        self.observation[8] = _observation_clock()
                comptime if OBSERVE:
                    self.observation[9] = _observation_clock()
                return selected
            var winner = 0
            var best = Float32(-3.402823466e38)
            with self.logits.map_to_host() as mapped:
                comptime if OBSERVE:
                    self.observation[7] = _observation_clock()
                for i in range(151936):
                    var value = mapped.unsafe_ptr()[unsafe_offset=i].cast[DType.float32]()
                    if value != value or value > Float32(3.402823466e38) or value < Float32(-3.402823466e38):
                        raise Error("nonfinite model logits")
                    if value > best:
                        best = value
                        winner = i
                comptime if OBSERVE:
                    self.observation[8] = _observation_clock()
            comptime if OBSERVE:
                self.observation[9] = _observation_clock()
            return winner
        except error:
            self.valid = False
            raise error
