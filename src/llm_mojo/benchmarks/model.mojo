"""Real Qwen decode on the Fast route: fixed history, normal stream, optional host observations.

Completed decode experiments (fusion, selection, buffer swap, composition,
projection arrangement, scheduling and launch probes) are replay-only; their
collectors exist through commit edb610a. See studies/model_generation/README.md.
"""
from std.sys import argv, is_defined, get_defined_int, get_defined_string
from std.time import sleep
from std.memory import bitcast
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from layout import TileTensor, TensorLayout, row_major
from std.gpu import global_idx
from llm_mojo.models.qwen2.model import QwenModel, select_configuration, save_bf16, _observation_clock
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace


def rewind(mut model: QwenModel, prefix: Int):
    # Previous greedy readback completed model computation. Only the logical suffix is rewound.
    model.length = prefix
    model.submitted_rows = prefix * 24
    for layer in range(24):
        model.layers[layer].cache.length = prefix


def snapshot(mut model: QwenModel, path: String) raises:
    save_bf16(model.logits,path+"-logits.bin",151936)
    for layer in range(24):
        save_bf16(model.layers[layer].cache.key,path+"-k"+String(layer)+".bin",4096*128)
        save_bf16(model.layers[layer].cache.value,path+"-v"+String(layer)+".bin",4096*128)


def poison_outputs(mut model: QwenModel, prefix: Int) raises:
    """Untimed verification: stale logits/cache appends must not pass parity."""
    var sentinel = bitcast[DType.bfloat16](UInt16(0x7FC0))
    model.logits.enqueue_fill(sentinel)
    model.mlp.activated.enqueue_fill(sentinel)
    model.mlp.gated.enqueue_fill(sentinel)
    for layer in range(24):
        with model.layers[layer].cache.key.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel
        with model.layers[layer].cache.value.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel


def step[OBSERVE: Bool](mut model: QwenModel, ctx: DeviceContext, ids: List[Int]) raises -> Int:
    """One single-row Fast decode step: configuration 26, GPU argmax, swap, fused norms."""
    model.forward[OBSERVE](ctx,ids,26,"",1,False,True,True)
    return model.greedy[OBSERVE](ctx)


def _batch_advance[LT: TensorLayout](x: TileTensor[DType.int32, LT, MutAnyOrigin]):
    comptime assert x.flat_rank == 1
    if global_idx.x == 0:
        x[0] = x[0]*2+1

def batch_support() raises:
    var ctx = DeviceContext()
    print("device:",ctx.name()); print("api:",ctx.api())
    if ctx.name() != "Apple M4 Pro" or ctx.api() != "metal": raise Error("requires M4 Pro / Metal")
    var buf = ctx.enqueue_create_buffer[DType.int32](1)
    buf.enqueue_fill(3)
    var x = TileTensor(buf,row_major(1))
    var compiled = ctx.compile_function[_batch_advance[type_of(x.layout)]]()
    ctx.enqueue_function(compiled,x,grid_dim=1,block_dim=32)
    ctx.enqueue_function(compiled,x,grid_dim=1,block_dim=32)
    with buf.map_to_host() as mapped:
        if mapped.unsafe_ptr()[unsafe_offset=0] != 15: raise Error("eager control failed")
    print("BATCH_EAGER_PASS 15")
    buf.enqueue_fill(3);ctx.synchronize()
    def build(mut builder: DeviceGraphBuilder) raises {mut x, imm compiled}:
        print("BATCH_BUILDER_ENTERED")
        var first = builder.add_function(compiled,x,grid_dim=1,block_dim=32,dependencies=[])
        _ = builder.add_function(compiled,x,grid_dim=1,block_dim=32,dependencies=[first])
    try:
        var graph = DeviceGraph.create(ctx,build)
        graph.replay();ctx.synchronize()
        with buf.map_to_host() as mapped:
            if mapped.unsafe_ptr()[unsafe_offset=0] != 15: raise Error("graph first replay failed")
        graph.replay();ctx.synchronize()
        with buf.map_to_host() as mapped:
            if mapped.unsafe_ptr()[unsafe_offset=0] != 63: raise Error("graph second replay failed")
        print("BATCH_GRAPH_PASS 15 63")
    except e:
        print("BATCH_GRAPH_ERROR",e)
    print("BATCH_SUPPORT_COMPLETE")


def main() raises:
    comptime if is_defined["MODEL_BATCH_SUPPORT"]():
        batch_support()
        return
    var args = List[String]()
    comptime if is_defined["MODEL_PROFILE_PREFIX"]():
        args = ["model","profile",String(get_defined_string["MODEL_PREPARED"]()),
                String(get_defined_string["MODEL_TABLES"]()),
                String(get_defined_int["MODEL_PROFILE_PREFIX"]()),"0","0",""]
    else:
        for arg in argv():
            args.append(String(arg))
    if len(args) != 8:
        raise Error("model mode prepared tables prefix first comparison output-directory")
    var mode = args[1]
    var prefix = Int(args[4])
    var first = Int(args[5])
    var comparison = Int(args[6])
    if ((mode != "bench" and mode != "verify" and mode != "profile")
        or (prefix != 64 and prefix != 1024 and prefix != 3968)
        or first < 0 or first > 1 or comparison < 0 or comparison > 1):
        raise Error("invalid frozen model profiling workload")
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var tokenizer = Tokenizer(args[3])
    var work = TokenizerWorkspace()
    var text = String()
    for _ in range(200):
        text += "A train travels sixty kilometers in forty-five minutes. Explain how to calculate its average speed, keeping track of distance, time, and units. The passengers compare their calculations and check each step.\n"
    var history = tokenizer.encode(text,work)
    if len(history) < prefix+1:
        raise Error("insufficient frozen token history")
    var model = QwenModel(ctx,args[2],4096,256)
    # Define all inactive storage for exact before/after comparisons.
    for layer in range(24):
        model.layers[layer].cache.key.enqueue_fill(0)
        model.layers[layer].cache.value.enqueue_fill(0)
    var offset = 0
    while offset < prefix:
        var count = min(256,prefix-offset)
        var chunk = List[Int](capacity=count)
        for i in range(count):
            chunk.append(history[offset+i])
        model.forward(ctx,chunk,select_configuration("fast",count,offset+count,ctx.name()))
        offset += count
    ctx.synchronize()
    var ids: List[Int] = [history[prefix]]
    var winner = step[False](model,ctx,ids)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("prefix:",prefix,"token:",ids[0],"winner:",winner)
    if mode == "verify":
        rewind(model,prefix)
        snapshot(model,args[7]+"/before")
        poison_outputs(model,prefix)
        var plain = step[False](model,ctx,ids)
        snapshot(model,args[7]+"/plain")
        rewind(model,prefix)
        poison_outputs(model,prefix)
        var observed = step[True](model,ctx,ids)
        snapshot(model,args[7]+"/observed")
        if plain != observed or model.length != prefix+1 or model.submitted_rows != 24*(prefix+1):
            raise Error("instrumentation changed token or accounting")
        var record = String()
        for i in range(prefix+1):
            record += String(history[i])+"\n"
        var file = open(args[7]+"/history.txt","w")
        file.write(record)
        print("VERIFY_COMPLETE")
        return
    if mode == "profile":
        for _ in range(10):
            rewind(model,prefix)
            if step[False](model,ctx,ids) != winner:
                raise Error("unstable profile prediction")
        print("correctness: passed")
        print("profile implementation:","QwenModel.forward+greedy-all-three")
        print("rows: 1")
        print("hidden: 896")
        print("key value rows:",prefix+1)
        print("profile workload:","model-p"+String(prefix)+"-all-three")
        print("profile dispatches per iteration:",245)
        print("warmup iterations: 10")
        print("profile iterations: 8")
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(8):
            rewind(model,prefix)
            if step[False](model,ctx,ids) != winner:
                raise Error("unstable profile prediction")
        print("PROFILE_REGION_END")
        sleep(0.25)
        return
    var records = String()
    for arm_index in range(2):
        var arm = (first+arm_index)%2
        var observe = comparison == 1 and arm == 1
        for sample in range(20):
            rewind(model,prefix)
            var start = _observation_clock()
            var selected: Int
            if observe:
                selected = step[True](model,ctx,ids)
            else:
                selected = step[False](model,ctx,ids)
            var elapsed = _observation_clock()-start
            if selected != winner or model.submitted_rows != 24*(prefix+1):
                raise Error("measurement prediction/accounting changed")
            if sample >= 10:
                records += "SAMPLE "+String(arm)+" "+String(sample-10)+" "+String(elapsed)
                if observe:
                    for i in range(10):
                        records += " "+String(model.observation[i]-start)
                records += "\n"
    print(records,end="")
    print("BENCHMARK_COMPLETE")
