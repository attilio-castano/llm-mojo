"""Real Qwen decode: fixed history, normal stream, optional host observations."""
from std.sys import argv, is_defined, get_defined_int, get_defined_string
from std.time import sleep
from std.memory import bitcast
from max.gpu.host import DeviceContext
from llm_mojo.model import QwenModel, select_configuration, save_bf16, _observation_clock
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace


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


def step[OBSERVE: Bool](mut model: QwenModel, ctx: DeviceContext, ids: List[Int], configuration: Int = -1, selection: Int = 0, materialize: Bool = False) raises -> Int:
    model.forward[OBSERVE](ctx,ids,configuration if configuration >= 0 else select_configuration("fast",1,model.length+1,ctx.name()),"",selection,materialize)
    return model.greedy[OBSERVE](ctx)


def main() raises:
    comptime SELECTION = is_defined["MODEL_SELECTION_STUDY"]()
    comptime COMBINED = is_defined["MODEL_COMBINED_STUDY"]()
    comptime FUSION = is_defined["MODEL_FUSION_STUDY"]()
    comptime PROFILE_FUSED = is_defined["MODEL_FUSION_PROFILE"]()
    comptime assert not COMBINED or FUSION or PROFILE_FUSED, "combined study requires an explicit study or profile route"
    var candidate = 26 if COMBINED else 25
    var control = 26 if SELECTION else (0 if FUSION else -1)
    var selection_control = 0
    var selection_candidate = 0
    comptime PROFILE_SELECTION = get_defined_int["MODEL_SELECTION_PROFILE", 0]()
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
        or first < 0 or first > 1 or comparison < 0 or comparison > (3 if SELECTION else (2 if COMBINED else 1))):
        raise Error("invalid frozen model profiling workload")
    if SELECTION and mode == "verify" and comparison != 1 and comparison != 2:
        raise Error("selection verification requires candidate 1 or 2")
    if SELECTION:
        candidate = 26
        selection_control = 1 if comparison == 3 else 0
        selection_candidate = 2 if comparison >= 2 else comparison
    if COMBINED and comparison == 2:
        control = 25
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
    var winner = step[False](model,ctx,ids,control)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("prefix:",prefix,"token:",ids[0],"winner:",winner)
    if mode == "verify":
        rewind(model,prefix)
        snapshot(model,args[7]+"/before")
        poison_outputs(model,prefix)
        var plain = step[False](model,ctx,ids,control)
        snapshot(model,args[7]+"/plain")
        rewind(model,prefix)
        poison_outputs(model,prefix)
        var observed: Int
        comptime if SELECTION:
            observed = step[False](model,ctx,ids,26,selection_candidate,True)
        elif FUSION:
            observed = step[False](model,ctx,ids,candidate)
        else:
            observed = step[True](model,ctx,ids)
        snapshot(model,args[7]+"/observed")
        if plain != observed or model.length != prefix+1 or model.submitted_rows != 24*(prefix+1):
            raise Error("instrumentation changed token or accounting")
        comptime if SELECTION:
            rewind(model,prefix)
            poison_outputs(model,prefix)
            model.selection_partials.enqueue_fill(0xDEADBEEF)
            model.selection_result.enqueue_fill(0xDEADBEEF)
            if step[False](model,ctx,ids,26,selection_candidate) != plain:
                raise Error("nonmaterializing selection changed winner")
            snapshot(model,args[7]+"/actual")
            if selection_candidate == 2:
                with model.logits.map_to_host() as mapped:
                    for i in range(151936):
                        if bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]) != 0x7FC0:
                            raise Error("fused timed path wrote logits")
            # The public greedy lifecycle must reject a nonfinite flag.
            with model.selection_result.map_to_host() as mapped:
                mapped.unsafe_ptr()[unsafe_offset=2] = 1
            var rejected = False
            try:
                _ = model.greedy(ctx)
            except:
                rejected = True
            if not rejected or model.valid:
                raise Error("nonfinite selection did not invalidate model")
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
            if step[False](model,ctx,ids,candidate if PROFILE_FUSED else control,PROFILE_SELECTION) != winner:
                raise Error("unstable profile prediction")
        print("correctness: passed")
        comptime if SELECTION:
            print("profile implementation:","QwenModel.forward+greedy-"+("gpu-argmax" if PROFILE_SELECTION == 1 else ("fused-head" if PROFILE_SELECTION == 2 else "combined")))
        else:
            print("profile implementation:", ("QwenModel.forward+greedy-combined" if COMBINED else "QwenModel.forward+greedy-fused") if PROFILE_FUSED else "QwenModel.forward+greedy")
        print("rows: 1")
        print("hidden: 896")
        print("key value rows:",prefix+1)
        comptime if SELECTION:
            print("profile workload:","model-p"+String(prefix)+("-gpu-argmax" if PROFILE_SELECTION == 1 else ("-fused-head" if PROFILE_SELECTION == 2 else "-combined")))
            print("profile dispatches per iteration:",314+(2 if PROFILE_SELECTION == 1 else (1 if PROFILE_SELECTION == 2 else 0)))
        else:
            print("profile workload:","model-p"+String(prefix)+(("-combined" if COMBINED else "-fused") if PROFILE_FUSED else ""))
            print("profile dispatches per iteration:",(314 if COMBINED else 338) if PROFILE_FUSED else 410)
        print("warmup iterations: 10")
        print("profile iterations: 8")
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(8):
            rewind(model,prefix)
            if step[False](model,ctx,ids,candidate if PROFILE_FUSED else control,PROFILE_SELECTION) != winner:
                raise Error("unstable profile prediction")
        print("PROFILE_REGION_END")
        sleep(0.25)
        return
    var records = String()
    for arm_index in range(2):
        var arm = (first+arm_index)%2
        var observe = comparison == 1 and arm == 1 and not FUSION and not SELECTION
        for sample in range(20):
            rewind(model,prefix)
            var start = _observation_clock()
            var selected: Int
            if observe:
                selected = step[True](model,ctx,ids)
            else:
                selected = step[False](model,ctx,ids,candidate if FUSION and comparison >= 1 and arm == 1 else control,selection_candidate if arm == 1 else selection_control)
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
