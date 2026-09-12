"""Real Qwen decode: fixed history, normal stream, optional host observations."""
from std.sys import argv, is_defined, get_defined_int, get_defined_string
from std.time import sleep
from std.memory import bitcast
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from layout import TileTensor, TensorLayout, row_major
from std.gpu import global_idx
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


def poison_outputs(mut model: QwenModel, prefix: Int, poison_norm: Bool = False) raises:
    """Untimed verification: stale logits/cache appends must not pass parity."""
    var sentinel = bitcast[DType.bfloat16](UInt16(0x7FC0))
    model.logits.enqueue_fill(sentinel)
    model.mlp.activated.enqueue_fill(sentinel)
    model.mlp.gated.enqueue_fill(sentinel)
    if poison_norm:
        model.attention.normalized.enqueue_fill(sentinel)
        model.attention.output.enqueue_fill(sentinel)
        model.mlp.normalized.enqueue_fill(sentinel)
        model.mlp.output.enqueue_fill(sentinel)
        model.normalized.enqueue_fill(sentinel)
    for layer in range(24):
        with model.layers[layer].cache.key.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel
        with model.layers[layer].cache.value.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel


def step[OBSERVE: Bool](mut model: QwenModel, ctx: DeviceContext, ids: List[Int], configuration: Int = -1, selection: Int = 0, materialize: Bool = False, copy_free: Bool = False, fuse_norm: Bool = False, projection: Int = 0) raises -> Int:
    # Only current-default builds set this; explicit historical study arms stay fixed.
    comptime DEFAULT_VARIANT = get_defined_int["MODEL_DEFAULT_VARIANT", 0]()
    model.forward[OBSERVE](ctx,ids,configuration if configuration >= 0 else select_configuration("fast",1,model.length+1,ctx.name()),"",
        1 if DEFAULT_VARIANT >= 2 else selection,materialize,
        copy_free or DEFAULT_VARIANT >= 2,
        fuse_norm or DEFAULT_VARIANT == 1 or DEFAULT_VARIANT == 3,False,projection)
    return model.greedy[OBSERVE](ctx)


def verify_swap_lifecycle(mut model: QwenModel, ctx: DeviceContext, history: List[Int], path: String, copy_free: Bool = True, selection: Int = 0, fuse_norm: Bool = False) raises:
    for arm in range(2):
        model.reset(ctx)
        for layer in range(24):
            model.layers[layer].cache.key.enqueue_fill(0)
            model.layers[layer].cache.value.enqueue_fill(0)
        var record = String()
        var lengths: List[Int] = [3,1,1,2,1,0,1,2,1]
        for index in range(len(lengths)):
            var rows = lengths[index]
            if rows == 0:
                model.reset(ctx)
                continue
            var before = Int(model.input.unsafe_ptr())
            var destination = Int(model.mlp.output.unsafe_ptr())
            var prefix = model.length
            var ids = List[Int]()
            for j in range(rows):
                ids.append(history[prefix+j])
            var swap = arm == 1 and rows == 1 and copy_free
            model.forward(ctx,ids,select_configuration("combined",rows,prefix+rows,ctx.name()),"",selection if arm == 1 and rows == 1 else 0,False,swap,fuse_norm and arm == 1 and rows == 1)
            var token = model.greedy(ctx)
            if (Int(model.input.unsafe_ptr()) != (destination if swap else before)
                or Int(model.mlp.output.unsafe_ptr()) != (before if swap else destination)
                or before == destination or model.length != prefix+rows
                or model.submitted_rows != 24*(prefix+rows)):
                raise Error("buffer owner or accounting invariant failed")
            save_bf16(model.logits,path+"/lifecycle-"+String(arm)+"-"+String(index)+".bin",151936)
            record += String(index)+" "+String(rows)+" "+String(model.length)+" "+String(token)+"\n"
            # Reject invalid IDs before moving any owner or modifying caches.
            var rejected = False
            var bad_ids: List[Int] = [-1]
            var saved = Int(model.input.unsafe_ptr())
            try:
                model.forward(ctx,bad_ids,26,"",selection,False,arm == 1 and copy_free,arm == 1 and fuse_norm)
            except:
                rejected = True
            if not rejected or not model.valid or Int(model.input.unsafe_ptr()) != saved or model.length != prefix+rows:
                raise Error("rejected forward changed valid ownership/state")
            if arm == 1:
                rejected = False
                var bad_rows: List[Int] = [0,1]
                try:
                    model.forward(ctx,bad_rows,0,"",0,False,True)
                except:
                    rejected = True
                if not rejected or Int(model.input.unsafe_ptr()) != saved or model.length != prefix+rows:
                    raise Error("multi-row swap rejection changed state")
                if fuse_norm:
                    rejected = False
                    try:
                        model.forward(ctx,bad_rows,0,"",0,False,False,True)
                    except:
                        rejected = True
                    if not rejected or not model.valid or Int(model.input.unsafe_ptr()) != saved or model.length != prefix+rows:
                        raise Error("multi-row residual/norm rejection changed state")
        snapshot(model,path+"/lifecycle-final-"+String(arm))
        var file = open(path+"/lifecycle-"+String(arm)+".txt","w")
        file.write(record)
    print("SWAP_LIFECYCLE_COMPLETE")


def scheduling_pair[OBSERVE: Bool](mut model: QwenModel, ctx: DeviceContext,
        seed: Int, prefix: Int, first: Int, comparison: Int, advance: Bool,
        output: String) raises:
    # Only these benchmark modes change. Production inference is untouched.
    var records = String()
    for arm_index in range(2):
        var arm = (first+arm_index)%2
        var variant = comparison if arm else 0
        var ids: List[Int] = [seed]
        for _ in range(16):
            rewind(model,prefix)
            _ = step[OBSERVE](model,ctx,ids,26,1,False,True,True,variant)
        rewind(model,prefix)
        var times = List[UInt64](capacity=64*11)
        var tokens = List[Int](capacity=64)
        var starts = List[UInt64](capacity=64 if is_defined["MODEL_LAUNCH_PROBE"]() else 0)
        for sample in range(64):
            if not advance: rewind(model,prefix)
            var start = _observation_clock()
            var selected = step[OBSERVE](model,ctx,ids,26,1,False,True,True,variant)
            var elapsed = _observation_clock()-start
            times.append(elapsed)
            comptime if is_defined["MODEL_LAUNCH_PROBE"](): starts.append(start)
            comptime if OBSERVE:
                for j in range(10): times.append(model.observation[j]-start)
            tokens.append(selected)
            if advance: ids[0] = selected
        var expected = prefix+(64 if advance else 1)
        if model.length != expected or model.submitted_rows != 24*expected:
            raise Error("scheduling cache accounting changed")
        for layer in range(24):
            if model.layers[layer].cache.length != expected:
                raise Error("scheduling layer length changed")
        for sample in range(64):
            records += "SCHED_SAMPLE "+String(arm)+" "+String(sample)+" "+String(tokens[sample])
            for j in range(11 if OBSERVE else 1):
                records += " "+String(times[sample*(11 if OBSERVE else 1)+j])
            records += "\n"
        comptime if is_defined["MODEL_LAUNCH_PROBE"]():
            for sample in range(64):
                records += "ENQUEUE_WINDOW "+String(arm)+" "+String(sample)+" "+String(starts[sample])+"\n"
        if output.byte_length():
            snapshot(model,output+("/observed" if arm else "/plain"))
    print(records,end="")
    print("SCHEDULING_COMPLETE")


def launch_micro[DOWN: Bool](batch: Int, first: Int, comparison: Int) raises:
    """Same projection and views; isolate explicit compiled-handle reuse."""
    from layout import TileTensor, row_major
    from llm_mojo.linear import _linear_rowwise_apple_gpu_kernel
    from std.math import ceildiv
    comptime K = 4864 if DOWN else 32
    comptime N = 896 if DOWN else 1
    var ctx = DeviceContext()
    if ctx.name() != "Apple M4 Pro" or ctx.api() != "metal": raise Error("launch probe requires M4 Pro / Metal")
    var xb = ctx.enqueue_create_buffer[DType.bfloat16](K)
    var wb = ctx.enqueue_create_buffer[DType.bfloat16](N*K)
    var bb = ctx.enqueue_create_buffer[DType.bfloat16](N)
    var yb = ctx.enqueue_create_buffer[DType.bfloat16](N+2)
    xb.enqueue_fill(1); wb.enqueue_fill(1); yb.enqueue_fill(-77)
    var x = TileTensor(xb,row_major(1,K))
    var w = TileTensor(wb,row_major(N,K))
    var b = TileTensor(bb,row_major(N))
    var y = TileTensor(yb.unsafe_ptr().unsafe_offset(1),row_major(1,N))
    comptime kernel = _linear_rowwise_apple_gpu_kernel[type_of(x.layout),type_of(w.layout),type_of(b.layout),type_of(y.layout),False]
    var compiled = ctx.compile_function[kernel]()
    ctx.synchronize()
    print("device:",ctx.name());print("api:",ctx.api())
    var record = String()
    for arm_index in range(2):
        var arm = (first+arm_index)%2
        var cached = comparison == 1 and arm == 1
        var times = List[UInt64](capacity=30)
        for sample in range(20):
            var start = _observation_clock()
            for _ in range(batch):
                if cached:
                    ctx.enqueue_function(compiled,x,w,b,y,Int32(1),Int32(K),Int32(N),grid_dim=ceildiv(N,4),block_dim=128)
                else:
                    ctx.enqueue_function[kernel](x,w,b,y,Int32(1),Int32(K),Int32(N),grid_dim=ceildiv(N,4),block_dim=128)
            var queued = _observation_clock()
            ctx.synchronize()
            var done = _observation_clock()
            if sample >= 10:
                times.append(start);times.append(queued);times.append(done)
        with yb.map_to_host() as mapped:
            if mapped.unsafe_ptr()[unsafe_offset=0] != -77 or mapped.unsafe_ptr()[unsafe_offset=N+1] != -77:
                raise Error("launch probe guard changed")
            for i in range(N):
                if mapped.unsafe_ptr()[unsafe_offset=i+1] != Scalar[DType.bfloat16](K):
                    raise Error("launch probe output differs from exact integer oracle")
        for sample in range(10):
            record += "LAUNCH_SAMPLE "+String(arm)+" "+String(sample)
            for j in range(3): record += " "+String(times[sample*3+j])
            record += "\n"
    print(record,end="");print("LAUNCH_MICRO_COMPLETE")


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
    comptime LAUNCH_PROBE = is_defined["MODEL_LAUNCH_PROBE"]()
    comptime SCHEDULING = is_defined["MODEL_SCHEDULING_STUDY"]()
    comptime PROJECTION = is_defined["MODEL_PROJECTION_STUDY"]()
    comptime assert not SCHEDULING or PROJECTION
    comptime assert not LAUNCH_PROBE or SCHEDULING
    comptime PROFILE_PROJECTION = get_defined_int["MODEL_PROJECTION_PROFILE",0]()
    comptime COMPOSITION = is_defined["MODEL_COMPOSITION_STUDY"]()
    comptime DEFAULT_VARIANT = get_defined_int["MODEL_DEFAULT_VARIANT", 0]()
    comptime PROFILE_COMPOSITION = get_defined_int["MODEL_COMPOSITION_PROFILE", DEFAULT_VARIANT]()
    comptime COPY_FREE = is_defined["MODEL_COPY_FREE_STUDY"]()
    comptime SELECTION = is_defined["MODEL_SELECTION_STUDY"]()
    comptime COMBINED = is_defined["MODEL_COMBINED_STUDY"]()
    comptime FUSION = is_defined["MODEL_FUSION_STUDY"]()
    comptime PROFILE_FUSED = is_defined["MODEL_FUSION_PROFILE"]()
    comptime assert not COMBINED or FUSION or PROFILE_FUSED, "combined study requires an explicit study or profile route"
    var candidate = 26 if COMBINED or COPY_FREE else 25
    var control = 26 if SELECTION or COPY_FREE or COMPOSITION else (0 if FUSION else -1)
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
    if LAUNCH_PROBE and args[1] == "launch":
        var batch = Int(args[3]);var first = Int(args[4]);var comparison = Int(args[5])
        if (batch != 1 and batch != 256) or first < 0 or first > 1 or comparison < 0 or comparison > 1:
            raise Error("invalid launch microbenchmark")
        if args[2] == "down": launch_micro[True](batch,first,comparison)
        elif args[2] == "tiny": launch_micro[False](batch,first,comparison)
        else: raise Error("unknown launch microbenchmark")
        return
    var mode = args[1]
    var prefix = Int(args[4])
    var first = Int(args[5])
    var comparison = Int(args[6])
    var scheduling = SCHEDULING and (mode == "fixed" or mode == "advance" or mode == "observed-fixed" or mode == "observed-advance")
    if ((not scheduling and mode != "bench" and mode != "verify" and mode != "profile")
        or (prefix != 64 and prefix != 1024 and prefix != 3968)
        or first < 0 or first > 1 or comparison < 0 or comparison > (5 if COMPOSITION or PROJECTION else (3 if SELECTION else (2 if COMBINED else 1)))):
        raise Error("invalid frozen model profiling workload")
    if scheduling and comparison > 1:
        raise Error("scheduling compares projection variants 0/1 only")
    if PROJECTION and mode == "verify" and comparison == 0:
        raise Error("projection verification requires candidate 1..5")
    if COMPOSITION and mode == "verify" and (comparison < 1 or comparison > 3):
        raise Error("composition verification requires candidate 1..3")
    var composition_control = 2 if comparison == 4 else (1 if comparison == 5 else 0)
    var composition_candidate = min(comparison,3)
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
    var winner = step[False](model,ctx,ids,26 if PROJECTION else control,1 if PROJECTION else 0,False,PROJECTION,PROJECTION)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("prefix:",prefix,"token:",ids[0],"winner:",winner)
    if scheduling:
        if args[7].byte_length():
            snapshot(model,args[7]+"/before")
            var record = String()
            for i in range(prefix+1): record += String(history[i])+"\n"
            var file = open(args[7]+"/history.txt","w")
            file.write(record)
        if mode == "observed-fixed" or mode == "observed-advance":
            scheduling_pair[True](model,ctx,ids[0],prefix,first,comparison,mode == "observed-advance",args[7])
        else:
            scheduling_pair[False](model,ctx,ids[0],prefix,first,comparison,mode == "advance",args[7])
        return
    if mode == "verify":
        rewind(model,prefix)
        snapshot(model,args[7]+"/before")
        poison_outputs(model,prefix,COMPOSITION or PROJECTION)
        var plain = step[False](model,ctx,ids,26 if PROJECTION else control,1 if PROJECTION else 0,False,PROJECTION,PROJECTION)
        snapshot(model,args[7]+"/plain")
        rewind(model,prefix)
        poison_outputs(model,prefix,COMPOSITION or PROJECTION)
        var observed: Int
        comptime if PROJECTION:
            observed = step[False](model,ctx,ids,26,1,False,True,True,comparison)
        elif COMPOSITION:
            observed = step[False](model,ctx,ids,26,1 if comparison >= 2 else 0,False,comparison >= 2,comparison != 2)
        elif SELECTION:
            observed = step[False](model,ctx,ids,26,selection_candidate,True)
        elif FUSION:
            observed = step[False](model,ctx,ids,candidate,0,False,COPY_FREE)
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
        comptime if PROJECTION:
            if prefix == 64:
                for arm in range(2):
                    rewind(model,prefix)
                    poison_outputs(model,prefix,True)
                    model.forward(ctx,ids,26,args[7]+("/layers-candidate" if arm else "/layers-control"),1,False,True,True,True,comparison if arm else 0)
                    if model.greedy(ctx) != plain: raise Error("projection layer capture changed winner")
        comptime if COPY_FREE or COMPOSITION:
            if prefix == 64:
                for arm in range(2):
                    rewind(model,prefix)
                    model.forward(ctx,ids,26,args[7]+("/layers-candidate" if arm else "/layers-control"),1 if COMPOSITION and comparison >= 2 and arm == 1 else 0,False,arm == 1 and (COPY_FREE or comparison >= 2),COMPOSITION and comparison != 2 and arm == 1,COMPOSITION)
                    if model.greedy(ctx) != plain:
                        raise Error("layer capture changed winner")
                verify_swap_lifecycle(model,ctx,history,args[7],COPY_FREE or comparison >= 2,1 if COMPOSITION and comparison >= 2 else 0,COMPOSITION and comparison != 2)
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
            if step[SCHEDULING](model,ctx,ids,26 if PROJECTION else (candidate if PROFILE_FUSED else control),1 if PROJECTION or (COMPOSITION and PROFILE_COMPOSITION >= 2) else PROFILE_SELECTION,False,PROJECTION or (COMPOSITION and PROFILE_COMPOSITION >= 2) or (COPY_FREE and PROFILE_FUSED),PROJECTION or (COMPOSITION and (PROFILE_COMPOSITION == 1 or PROFILE_COMPOSITION == 3)),PROFILE_PROJECTION if PROJECTION else 0) != winner:
                raise Error("unstable profile prediction")
        print("correctness: passed")
        comptime if PROJECTION:
            print("profile implementation:","QwenModel.forward+greedy-all-three")
        elif COMPOSITION or DEFAULT_VARIANT != 0:
            print("profile implementation:","QwenModel.forward+greedy-"+("combined" if PROFILE_COMPOSITION == 0 else ("residual-norm" if PROFILE_COMPOSITION == 1 else ("swap-argmax" if PROFILE_COMPOSITION == 2 else "all-three"))))
        elif COPY_FREE:
            print("profile implementation:","QwenModel.forward+greedy-"+("buffer-swap" if PROFILE_FUSED else "combined"))
        elif SELECTION:
            print("profile implementation:","QwenModel.forward+greedy-"+("gpu-argmax" if PROFILE_SELECTION == 1 else ("fused-head" if PROFILE_SELECTION == 2 else "combined")))
        else:
            print("profile implementation:", ("QwenModel.forward+greedy-combined" if COMBINED else "QwenModel.forward+greedy-fused") if PROFILE_FUSED else "QwenModel.forward+greedy")
        print("rows: 1")
        print("hidden: 896")
        print("key value rows:",prefix+1)
        comptime if PROJECTION:
            print("profile workload:","model-p"+String(prefix)+"-all-three")
            print("profile dispatches per iteration:",245)
            print("projection arrangement:",PROFILE_PROJECTION)
        elif COMPOSITION or DEFAULT_VARIANT != 0:
            print("profile workload:","model-p"+String(prefix)+("-combined" if PROFILE_COMPOSITION == 0 else ("-residual-norm" if PROFILE_COMPOSITION == 1 else ("-swap-argmax" if PROFILE_COMPOSITION == 2 else "-all-three"))))
            print("profile dispatches per iteration:",314 if PROFILE_COMPOSITION == 0 else (266 if PROFILE_COMPOSITION == 1 else (293 if PROFILE_COMPOSITION == 2 else 245)))
        elif COPY_FREE:
            print("profile workload:","model-p"+String(prefix)+("-buffer-swap" if PROFILE_FUSED else "-combined"))
            print("profile dispatches per iteration:",291 if PROFILE_FUSED else 314)
        elif SELECTION:
            print("profile workload:","model-p"+String(prefix)+("-gpu-argmax" if PROFILE_SELECTION == 1 else ("-fused-head" if PROFILE_SELECTION == 2 else "-combined")))
            print("profile dispatches per iteration:",314+(2 if PROFILE_SELECTION == 1 else (1 if PROFILE_SELECTION == 2 else 0)))
        else:
            print("profile workload:","model-p"+String(prefix)+(("-combined" if COMBINED else "-fused") if PROFILE_FUSED else ""))
            print("profile dispatches per iteration:",(314 if COMBINED else 338) if PROFILE_FUSED else 410)
        print("warmup iterations: 10")
        print("profile iterations: 8")
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        var host_records = List[UInt64](capacity=8*11)
        for _ in range(8):
            rewind(model,prefix)
            var host_start: UInt64 = 0
            comptime if SCHEDULING: host_start = _observation_clock()
            if step[SCHEDULING](model,ctx,ids,26 if PROJECTION else (candidate if PROFILE_FUSED else control),1 if PROJECTION or (COMPOSITION and PROFILE_COMPOSITION >= 2) else PROFILE_SELECTION,False,PROJECTION or (COMPOSITION and PROFILE_COMPOSITION >= 2) or (COPY_FREE and PROFILE_FUSED),PROJECTION or (COMPOSITION and (PROFILE_COMPOSITION == 1 or PROFILE_COMPOSITION == 3)),PROFILE_PROJECTION if PROJECTION else 0) != winner:
                raise Error("unstable profile prediction")
            comptime if SCHEDULING:
                var elapsed = _observation_clock()-host_start
                host_records.append(elapsed)
                for j in range(10): host_records.append(model.observation[j]-host_start)
        print("PROFILE_REGION_END")
        comptime if SCHEDULING:
            var text = String()
            for iteration in range(8):
                text += "SCHED_HOST "+String(iteration)
                for j in range(11): text += " "+String(host_records[iteration*11+j])
                text += "\n"
            print(text,end="")
        sleep(0.25)
        return
    var records = String()
    for arm_index in range(2):
        var arm = (first+arm_index)%2
        var observe = comparison == 1 and arm == 1 and not FUSION and not SELECTION and not COMPOSITION and not PROJECTION
        for sample in range(20):
            rewind(model,prefix)
            var start = _observation_clock()
            var selected: Int
            comptime if PROJECTION:
                selected = step[False](model,ctx,ids,26,1,False,True,True,comparison if arm else 0)
            elif COMPOSITION:
                var variant = composition_candidate if arm else composition_control
                selected = step[False](model,ctx,ids,26,1 if variant >= 2 else 0,False,variant >= 2,variant == 1 or variant == 3)
            else:
                if observe:
                    selected = step[True](model,ctx,ids)
                else:
                    selected = step[False](model,ctx,ids,candidate if FUSION and comparison >= 1 and arm == 1 else control,selection_candidate if arm == 1 else selection_control,False,COPY_FREE and comparison == 1 and arm == 1)
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
