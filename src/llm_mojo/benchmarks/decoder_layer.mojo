"""One fixed, receipt-bound decoder workload; host fixture work precedes timing."""
from llm_mojo.decoder_layer import enqueue_decoder_layer
from llm_mojo.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace
from llm_mojo.mlp import MLPWeights, MLPWorkspace
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.python import Python, PythonObject
from std.sys import argv, is_defined, get_defined_int
from std.time import perf_counter_ns, sleep


def _load(mut buffer: DeviceBuffer[DType.bfloat16], label: String, count: Int,
          adversarial: Bool = False, layer: Int = 0) raises:
    var support = Python.import_module("llm_mojo.benchmarks.decoder_layer_contract")
    with buffer.map_to_host() as mapped:
        support.load(Int(mapped.unsafe_ptr()),label,count,adversarial,layer)


def _check(mut buffer: DeviceBuffer[DType.bfloat16], stage: String, p: Int, r: Int,
           adversarial: Bool, layer: Int) raises:
    var support = Python.import_module("llm_mojo.benchmarks.decoder_layer_contract")
    with buffer.map_to_host() as mapped:
        support.check(Int(mapped.unsafe_ptr()),stage,p,r,adversarial,layer)


def _enqueue(ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
             mut a: AttentionWorkspace, mut mw: MLPWeights, mut m: MLPWorkspace,
             mut input: DeviceBuffer[DType.bfloat16], r: Int, t: Int) raises:
    cache.length = t-r
    var actual = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,
        TileTensor(input.unsafe_ptr().unsafe_offset((t-r)*896),row_major(r,896)),
        7 if r > 1 else 0)
    if actual != (6 if r > 1 else 4) or cache.length != t:
        raise Error("decoder benchmark route/cache identity mismatch")


def main() raises:
    var args = List[String]()
    comptime if is_defined["GQA_PROFILE_ROWS"]():
        args = ["decoder",String(get_defined_int["GQA_PROFILE_QUERY_ROWS"]()),
                String(get_defined_int["GQA_PROFILE_ROWS"]()),"1","0","0","0","4001","profile",
                String(get_defined_int["GQA_PROFILE_ITERATIONS"]()),String(get_defined_int["GQA_PROFILE_WARMUP"]())]
    else:
        for arg in argv():
            args.append(String(arg))
    if len(args) != 11:
        raise Error("expected R T layers candidate control first seed mode samples warmup")
    var r = Int(args[1])
    var t = Int(args[2])
    var layers = Int(args[3])
    var candidate = Int(args[4])
    var control = Int(args[5])
    var first = Int(args[6])
    var seed = Int(args[7])
    var mode = args[8]
    var repetitions = Int(args[9])
    var warmup = Int(args[10])
    if (r < 1 or r > t or t > 4096 or (layers != 1 and layers != 24)
        or candidate != 0 or control != 0 or seed != 4001
        or (first != 0 and first != 1) or repetitions < 1 or warmup < 0 or warmup > 100
        or (mode != "bench" and mode != "profile" and mode != "adversarial")
        or (mode == "profile" and (layers != 1 or repetitions*16 > 5000))):
        raise Error("invalid decoder benchmark shape, mapping or budget")
    var support = Python.import_module("llm_mojo.benchmarks.decoder_layer_contract")
    support.fixture_identity()
    var adversarial = mode == "adversarial"
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("decoder benchmark requires Metal")
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("operation: decoder_layer")
    print("measurement: whole_decoder")
    print("query rows:",r)
    print("shape:",t,layers,"seed:",seed)
    print("variants:",control,candidate,"candidate-first:",first)
    var a = AttentionWorkspace(ctx,t,t,14,2,64,False,False)
    var m = MLPWorkspace(ctx,t)
    _load(a.cosine,"full_cosine",t*64)
    _load(a.sine,"full_sine",t*64)
    var att_weights = List[AttentionWeights]()
    var mlp_weights = List[MLPWeights]()
    var caches = List[AttentionCache]()
    var inputs = List[DeviceBuffer[DType.bfloat16]]()
    for layer in range(layers):
        var aw = AttentionWeights(ctx)
        var mw = MLPWeights(ctx)
        var cache = AttentionCache(ctx,t)
        var input = ctx.enqueue_create_buffer[DType.bfloat16](t*896)
        _load(input,"input_X",t*896,adversarial,layer)
        _load(aw.norm,"input_input_norm",896)
        _load(aw.qkv,"input_qkv",1152*896,adversarial,layer)
        _load(aw.bias,"input_bias",1152)
        _load(aw.output,"input_wo",896*896,adversarial,layer)
        _load(mw.norm,"input_post_norm",896)
        _load(mw.gate,"input_gate",4864*896,adversarial,layer)
        _load(mw.up,"input_up",4864*896,adversarial,layer)
        _load(mw.down,"input_down",896*4864,adversarial,layer)
        if t > r:
            _ = enqueue_decoder_layer(ctx,aw,cache,a,mw,m,
                TileTensor(input,row_major(t-r,896)),7 if t-r > 1 else 0)
            ctx.synchronize()
        _enqueue(ctx,aw,cache,a,mw,m,input,r,t)
        ctx.synchronize()
        _check(a.projected,"B_att",t-r,r,adversarial,layer)
        _check(a.output,"Z",t-r,r,adversarial,layer)
        _check(m.down,"B_mlp",t-r,r,adversarial,layer)
        _check(m.output,"Y",t-r,r,adversarial,layer)
        att_weights.append(aw^)
        mlp_weights.append(mw^)
        caches.append(cache^)
        inputs.append(input^)
    # Verify the complete shared-workspace ring, preserving every output.
    var saved = List[DeviceBuffer[DType.bfloat16]]()
    for _ in range(layers):
        saved.append(ctx.enqueue_create_buffer[DType.bfloat16](t*896))
    for layer in range(layers):
        _enqueue(ctx,att_weights[layer],caches[layer],a,mlp_weights[layer],m,inputs[layer],r,t)
        ctx.enqueue_copy(dst_buf=saved[layer],src_buf=m.output)
    ctx.synchronize()
    for layer in range(layers):
        _check(saved[layer],"Y",t-r,r,adversarial,layer)
    print("correctness: passed")
    if adversarial:
        print("BENCHMARK_COMPLETE")
        return
    if mode == "profile":
        for _ in range(warmup):
            _enqueue(ctx,att_weights[0],caches[0],a,mlp_weights[0],m,inputs[0],r,t)
        ctx.synchronize()
        print("profile implementation: enqueue_decoder_layer")
        print("rows:",r)
        print("hidden: 896")
        print("key value rows:",t)
        print("query heads: 14")
        print("key value heads: 2")
        print("intermediate size: 4864")
        print("mlp mapping:",7 if r > 1 else 0)
        print("profile workload:","decoder-r"+String(r)+"-t"+String(t)+"-v0")
        print("profile dispatches per iteration: 16")
        print("warmup iterations:",warmup)
        print("profile iterations:",repetitions)
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(repetitions):
            _enqueue(ctx,att_weights[0],caches[0],a,mlp_weights[0],m,inputs[0],r,t)
        ctx.synchronize()
        print("PROFILE_REGION_END")
        sleep(0.25)
        return
    for arm in range(2):
        var is_candidate = (arm == 0) == (first == 1)
        var label = "candidate" if is_candidate else "control"
        for sample in range(warmup+repetitions):
            var start = perf_counter_ns()
            for layer in range(layers):
                _enqueue(ctx,att_weights[layer],caches[layer],a,mlp_weights[layer],m,inputs[layer],r,t)
            ctx.synchronize()
            var us = Float64(perf_counter_ns()-start)/Float64(1000*layers)
            if sample >= warmup:
                print("SAMPLE",label,0,sample-warmup,us)
    print("BENCHMARK_COMPLETE")
