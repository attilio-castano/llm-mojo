"""Fixed Qwen model ownership. Native execution; prepared files are verified by tooling.

This initial implementation is a development candidate, not accepted full-model
inference. Its cross-layer copy keeps the existing decoder alias contract intact.
"""
from std.memory import bitcast
from std.gpu import global_idx
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace
from llm_mojo.mlp import MLPWeights, MLPWorkspace
from llm_mojo.decoder_layer import _decoder_preflight, decoder_mappings, enqueue_decoder_layer_configuration
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.linear import enqueue_linear_apple_gpu


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


def save_bf16(buffer: DeviceBuffer[DType.bfloat16], path: String, count: Int) raises:
    """Diagnostic-only readback. Deliberately synchronizes before observation."""
    if count < 0 or count > len(buffer):
        raise Error("invalid diagnostic extent")
    var data = List[UInt8](capacity=count*2)
    with buffer.map_to_host() as mapped:
        for i in range(count):
            var bits = bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i])
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


def candidate_configuration(rows: Int, total: Int) -> Int:
    """Prior shared layer lookup. These candidates await full-model confirmation."""
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
    # Auto remains the control until model-level promotion earns lookup entries.
    if policy == "baseline" or policy == "auto":
        return 0
    if policy == "consistent" or policy == "20":
        return 20
    if policy == "candidate":
        return candidate_configuration(rows,total) if device == "Apple M4 Pro" else 0
    if policy == "0" or policy == "2" or policy == "3":
        return Int(policy)
    raise Error("unknown generation configuration policy")


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
    var tokens: DeviceBuffer[DType.int32]
    var capacity: Int
    var max_rows: Int
    var length: Int
    var valid: Bool
    var submitted_rows: Int

    def __init__(out self, ctx: DeviceContext, path: String, capacity: Int, max_rows: Int) raises:
        if ctx.api() != "metal" or capacity < 1 or capacity > 4096 or max_rows < 1 or max_rows > capacity:
            raise Error("Qwen requires Metal and valid row/context capacity")
        self.capacity = capacity
        self.max_rows = max_rows
        self.length = 0
        self.valid = True
        self.submitted_rows = 0
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

    def forward(mut self, ctx: DeviceContext, ids: List[Int], configuration: Int = 0, capture: String = "") raises:
        """Submit all layers. ID upload synchronizes; layer execution does not.

        Public development API; not yet full-model accepted. The host token
        staging boundary is measured separately from a future enqueue API.
        """
        self.preflight(ctx,ids,configuration)
        var rows = len(ids)
        try:
            with self.tokens.map_to_host() as mapped:
                for i in range(rows):
                    mapped.unsafe_ptr()[unsafe_offset=i] = Int32(ids[i])
            var token_view = TileTensor(self.tokens,row_major(rows))
            var weight_view = TileTensor(self.embedding,row_major(151936,896))
            var input_view = TileTensor(self.input,row_major(rows,896))
            comptime embedding_kernel = _embedding[type_of(token_view.layout),type_of(weight_view.layout),type_of(input_view.layout)]
            ctx.enqueue_function[embedding_kernel](token_view,weight_view,input_view,Int32(rows),
                grid_dim=(rows*896+255)//256,block_dim=256)
            if capture.byte_length() > 0:
                save_bf16(self.input,capture+"/hidden_0.bin",rows*896)
            for i in range(24):
                _ = enqueue_decoder_layer_configuration(ctx,self.layers[i].attention,
                    self.layers[i].cache,self.attention,self.layers[i].mlp,self.mlp,
                    TileTensor(self.input,row_major(rows,896)),configuration)
                if capture.byte_length() > 0:
                    save_bf16(self.mlp.output,capture+"/hidden_"+String(i+1)+".bin",rows*896)
                    save_bf16(self.attention.rotated_key,capture+"/append_key_"+String(i)+".bin",rows*128)
                    save_bf16(self.attention.raw_value,capture+"/append_value_"+String(i)+".bin",rows*128)
                    save_bf16(self.layers[i].cache.key,capture+"/cache_key_"+String(i)+".bin",self.capacity*128)
                    save_bf16(self.layers[i].cache.value,capture+"/cache_value_"+String(i)+".bin",self.capacity*128)
                if i < 23:
                    ctx.enqueue_function[_copy_rows[type_of(input_view.layout),type_of(input_view.layout)]](TileTensor(self.mlp.output,row_major(rows,896)),
                        TileTensor(self.input,row_major(rows,896)),Int32(rows),
                        grid_dim=(rows*896+255)//256,block_dim=256)
            enqueue_rms_norm_apple_gpu(ctx,
                TileTensor(self.mlp.output.unsafe_ptr().unsafe_offset((rows-1)*896),row_major(1,896)),
                TileTensor(self.norm,row_major(896)),TileTensor(self.normalized,row_major(1,896)))
            enqueue_linear_apple_gpu(ctx,TileTensor(self.normalized,row_major(1,896)),
                TileTensor(self.embedding,row_major(151936,896)),TileTensor(self.logits,row_major(1,151936)))
            if capture.byte_length() > 0:
                save_bf16(self.normalized,capture+"/final_norm.bin",896)
                save_bf16(self.logits,capture+"/logits.bin",151936)
            self.length += rows
            self.submitted_rows += rows*24
        except error:
            self.valid = False
            raise error

    def greedy(mut self, ctx: DeviceContext) raises -> Int:
        """Host reference argmax: lowest ID on ties; reject nonfinite logits."""
        if not self.valid or self.length == 0:
            raise Error("no valid next-token logits")
        try:
            var winner = 0
            var best = Float32(-3.402823466e38)
            with self.logits.map_to_host() as mapped:
                for i in range(151936):
                    var value = mapped.unsafe_ptr()[unsafe_offset=i].cast[DType.float32]()
                    if value != value or value > Float32(3.402823466e38) or value < Float32(-3.402823466e38):
                        raise Error("nonfinite model logits")
                    if value > best:
                        best = value
                        winner = i
            return winner
        except error:
            self.valid = False
            raise error
