"""Real Qwen decode on the Fast route: fixed history, normal stream, optional host observations.

The token-profile modes (bench, verify, profile) time one sequence. The batch
mode and the MODEL_BATCH_PROFILE build time decode steps of B sequences: the
batch-size study (1c) pairs row tiles, and the projection study (1d) pairs
batched projection arrangements (docs/batched-decode-plan.md).
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
from llm_mojo.kernels.linear import DECODE_ARRANGEMENTS
from llm_mojo.models.qwen2.model import QwenModel, save_bf16
from llm_mojo.models.qwen2.plan import fast_plan
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool


def rewind(mut model: QwenModel, mut kv: KVPool, prefix: Int) raises:
    # Previous greedy readback completed model computation. Only the logical suffix is rewound.
    model.submitted_rows = prefix * 24
    kv.truncate(0, prefix)


def snapshot(mut model: QwenModel, kv: KVPool, path: String) raises:
    save_bf16(model.logits,path+"-logits.bin",151936)
    for layer in range(24):
        save_bf16(kv.caches[kv.index(0,layer)].key,path+"-k"+String(layer)+".bin",4096*128)
        save_bf16(kv.caches[kv.index(0,layer)].value,path+"-v"+String(layer)+".bin",4096*128)


def poison_outputs(mut model: QwenModel, mut kv: KVPool, prefix: Int) raises:
    """Untimed verification: stale logits/cache appends must not pass parity."""
    var sentinel = bitcast[DType.bfloat16](UInt16(0x7FC0))
    model.logits.enqueue_fill(sentinel)
    model.mlp.activated.enqueue_fill(sentinel)
    model.mlp.gated.enqueue_fill(sentinel)
    for layer in range(24):
        with kv.caches[kv.index(0,layer)].key.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel
        with kv.caches[kv.index(0,layer)].value.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel


def step[OBSERVE: Bool](mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext, ids: List[Int]) raises -> Int:
    """One single-row Fast decode step: configuration 26, GPU argmax, swap, fused norms."""
    var cached = kv.length(0)
    model.forward[OBSERVE](ctx,StepBatch.sequence(ids,cached,0,kv.block_size),kv,fast_plan(1,cached+1,ctx.name()))
    return model.greedy[OBSERVE](ctx)


comptime BATCH_POOL = 64
comptime BATCH_MIXED = 32


def frozen_history(tables: String) raises -> List[Int]:
    var tokenizer = Tokenizer(tables)
    var work = TokenizerWorkspace()
    var text = String()
    for _ in range(200):
        text += "A train travels sixty kilometers in forty-five minutes. Explain how to calculate its average speed, keeping track of distance, time, and units. The passengers compare their calculations and check each step.\n"
    return tokenizer.encode(text,work)


def batch_contexts(context: Int) -> List[Int]:
    """Cached tokens per sequence: the declared context, or 64 to 3968 spread evenly (context 0)."""
    var result = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        if context == 0:
            result.append(64 + (min(s,BATCH_MIXED-1)*(3968-64))//(BATCH_MIXED-1))
        else:
            result.append(context)
    return result^


def batch_setup(ctx: DeviceContext, mut model: QwenModel, mut kv: KVPool, history: List[Int], longest: Int) raises:
    """Prefill the frozen history into block 0 with Fast chunks, then copy block 0 to every block."""
    kv.storage.enqueue_fill(0)
    var offset = 0
    while offset < longest:
        var count = min(256,longest-offset)
        var chunk = List[Int](capacity=count)
        for i in range(count):
            chunk.append(history[offset+i])
        model.forward(ctx,StepBatch.sequence(chunk,offset,0,kv.block_size),kv,fast_plan(count,offset+count,ctx.name()))
        offset += count
    ctx.synchronize()
    var block = len(kv.storage)//kv.blocks
    var first = kv.storage.create_sub_buffer[DType.bfloat16](0,block)
    for b in range(1,kv.blocks):
        var target = kv.storage.create_sub_buffer[DType.bfloat16](b*block,block)
        ctx.enqueue_copy(dst_buf=target,src_buf=first)
    ctx.synchronize()


def batch_rewind(mut model: QwenModel, mut kv: KVPool, contexts: List[Int], sequences: Int) raises:
    # Earlier readback completed every step. Only logical lengths are rewound.
    model.submitted_rows = 0
    for s in range(sequences):
        for layer in range(kv.geometry.layers):
            kv.caches[kv.index(s,layer)].length = contexts[s]


def batch_step[OBSERVE: Bool, ARRANGEMENT: Int](mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext,
                                                tokens: List[Int], contexts: List[Int], sequences: Int) raises -> List[Int]:
    """One decode token for each of the first `sequences` blocks, then one token per sequence."""
    var ids = List[Int](capacity=sequences)
    var positions = List[Int](capacity=sequences)
    var starts = List[Int](capacity=sequences+1)
    var seq_lens = List[Int](capacity=sequences)
    var blocks = List[Int](capacity=sequences)
    var slots = List[Int](capacity=sequences)
    var rows = List[Int](capacity=sequences)
    var longest = 0
    for s in range(sequences):
        ids.append(tokens[s])
        positions.append(contexts[s])
        starts.append(s)
        seq_lens.append(contexts[s]+1)
        blocks.append(s)
        slots.append(s*kv.block_size+contexts[s])
        rows.append(s)
        longest = max(longest,contexts[s]+1)
    starts.append(sequences)
    var batch = StepBatch(ids^,positions^,starts^,sequences,seq_lens^,1,blocks^,slots^,rows^)
    model.forward[OBSERVE, ARRANGEMENT](ctx,batch,kv,fast_plan(sequences,longest,ctx.name(),sequences))
    return model.greedy_tokens[OBSERVE](ctx)


def batch_arms(study: String) raises -> List[Int]:
    """Each comparison's candidate arrangement against arrangement 0; -1 is arrangement 0 with host marks.

    size: 1c's tiles 4, 8 and 16 and the observed arm. projections: 1d's screen of
    arrangements 3-6. confirm:A: 1d's confirmation of the selected arrangement A.
    """
    if study == "size":
        return [0, 1, 2, -1]
    if study == "projections":
        return [0, 3, 4, 5, 6]
    var parts = study.split(":")
    if len(parts) == 2 and String(parts[0]) == "confirm":
        var arrangement = Int(String(parts[1]))
        if arrangement >= 3 and arrangement < DECODE_ARRANGEMENTS:
            return [0, arrangement]
    raise Error("unknown batch study")


def batch_arm(arrangement: Int, mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext,
              tokens: List[Int], contexts: List[Int], sequences: Int) raises -> List[Int]:
    if arrangement == -1:
        return batch_step[True, 0](model,kv,ctx,tokens,contexts,sequences)
    comptime for a in range(DECODE_ARRANGEMENTS):
        if arrangement == a:
            return batch_step[False, a](model,kv,ctx,tokens,contexts,sequences)
    raise Error("unknown projection arrangement")


def batch_check(model: QwenModel, selected: List[Int], expected: List[Int], sequences: Int) raises:
    if (len(selected) != sequences or model.submitted_rows != 24*sequences
            or model.last_route.decode_launches != 245 or model.last_route.sequences != sequences):
        raise Error("batched step changed its route or accounting")
    for s in range(sequences):
        if selected[s] != expected[s]:
            raise Error("batched step changed a sequence's token")


def batch_bench(prepared: String, tables: String, context: Int, first: Int, study: String) raises:
    """Paired arms per batch size: arrangement 0 against itself, then each of the study's candidates.

    Tiles 4, 8 and 16 are projection arrangements 0, 1 and 2.
    """
    var arms = batch_arms(study)
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(tables)
    var contexts = batch_contexts(context)
    var sizes: List[Int] = [1, 2, 4, 8, 16, 32, 64]
    if context == 0:
        sizes = [BATCH_MIXED]
    var longest = 0
    for c in contexts:
        longest = max(longest,c)
    if len(history) < longest+BATCH_POOL+1:
        raise Error("insufficient frozen token history")
    var tokens = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        tokens.append(history[contexts[s]+s])
    var model = QwenModel(ctx,prepared,4096,256,BATCH_POOL)
    var kv = KVPool(ctx,BATCH_POOL,4096,model.kv_geometry())
    batch_setup(ctx,model,kv,history,longest)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("context:",context,"pool blocks:",BATCH_POOL)
    print("study:",study)
    var records = String()
    for index in range(len(sizes)):
        var sequences = sizes[len(sizes)-1-index] if first == 1 else sizes[index]
        batch_rewind(model,kv,contexts,sequences)
        var expected = batch_step[False, 0](model,kv,ctx,tokens,contexts,sequences)
        batch_check(model,expected,expected,sequences)
        print("sequences:",sequences,"first token:",expected[0],"last token:",expected[sequences-1])
        for position in range(len(arms)):
            var comparison = len(arms)-1-position if first == 1 else position
            for arm_index in range(2):
                var arm = (first+arm_index)%2
                for sample in range(20):
                    batch_rewind(model,kv,contexts,sequences)
                    var start = now()
                    var arrangement = arms[comparison] if arm == 1 else 0
                    var selected = batch_arm(arrangement,model,kv,ctx,tokens,contexts,sequences)
                    var elapsed = now()-start
                    batch_check(model,selected,expected,sequences)
                    if sample >= 10:
                        records += ("BATCH "+String(sequences)+" "+String(comparison)+" "+String(arm)+" "
                                    +String(sample-10)+" "+String(elapsed))
                        if arrangement == -1:
                            for i in range(10):
                                records += " "+String(model.observation[i]-start)
                        records += "\n"
    print(records,end="")
    print("BATCH_COMPLETE")


def batch_profile[ARRANGEMENT: Int](prepared: String, tables: String, context: Int, sequences: Int,
                                    workload: String) raises:
    """Trace target: ten warmups and eight plain batched steps inside the profile region."""
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(tables)
    var contexts = batch_contexts(context)
    var tokens = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        tokens.append(history[contexts[s]+s])
    var model = QwenModel(ctx,prepared,4096,256,BATCH_POOL)
    var kv = KVPool(ctx,BATCH_POOL,4096,model.kv_geometry())
    batch_setup(ctx,model,kv,history,context)
    batch_rewind(model,kv,contexts,sequences)
    var expected = batch_step[False, ARRANGEMENT](model,kv,ctx,tokens,contexts,sequences)
    for _ in range(10):
        batch_rewind(model,kv,contexts,sequences)
        batch_check(model,batch_step[False, ARRANGEMENT](model,kv,ctx,tokens,contexts,sequences),expected,sequences)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("correctness: passed")
    print("profile implementation:","QwenModel.forward+greedy_tokens")
    print("rows:",sequences)
    print("hidden: 896")
    print("key value rows:",context+1)
    print("profile workload:",workload)
    print("profile dispatches per iteration:",245)
    print("warmup iterations: 10")
    print("profile iterations: 8")
    print("post-profile idle milliseconds: 250")
    print("PROFILE_REGION_BEGIN")
    for _ in range(8):
        batch_rewind(model,kv,contexts,sequences)
        batch_check(model,batch_step[False, ARRANGEMENT](model,kv,ctx,tokens,contexts,sequences),expected,sequences)
    print("PROFILE_REGION_END")
    sleep(0.25)


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
    comptime if is_defined["MODEL_BATCH_PROFILE"]():
        batch_profile[get_defined_int["MODEL_BATCH_ARRANGEMENT"]()](String(get_defined_string["MODEL_PREPARED"]()),
            String(get_defined_string["MODEL_TABLES"]()),get_defined_int["MODEL_BATCH_PROFILE"](),
            get_defined_int["MODEL_BATCH_SEQUENCES"](),String(get_defined_string["MODEL_BATCH_WORKLOAD"]()))
        return
    var cli = argv()
    if len(cli) == 7 and String(cli[1]) == "batch":
        var context = Int(String(cli[4]))
        var first = Int(String(cli[5]))
        if (context != 0 and context != 64 and context != 1024 and context != 3968) or first < 0 or first > 1:
            raise Error("invalid batch-size workload")
        batch_bench(String(cli[2]),String(cli[3]),context,first,String(cli[6]))
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
    var history = frozen_history(args[3])
    if len(history) < prefix+1:
        raise Error("insufficient frozen token history")
    var model = QwenModel(ctx,args[2],4096,256)
    var kv = KVPool(ctx,1,4096,model.kv_geometry())
    # Define all inactive storage for exact before/after comparisons.
    for layer in range(24):
        kv.caches[kv.index(0,layer)].key.enqueue_fill(0)
        kv.caches[kv.index(0,layer)].value.enqueue_fill(0)
    var offset = 0
    while offset < prefix:
        var count = min(256,prefix-offset)
        var chunk = List[Int](capacity=count)
        for i in range(count):
            chunk.append(history[offset+i])
        model.forward(ctx,StepBatch.sequence(chunk,offset,0,4096),kv,fast_plan(count,offset+count,ctx.name()))
        offset += count
    ctx.synchronize()
    var ids: List[Int] = [history[prefix]]
    var winner = step[False](model,kv,ctx,ids)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("prefix:",prefix,"token:",ids[0],"winner:",winner)
    if mode == "verify":
        rewind(model,kv,prefix)
        snapshot(model,kv,args[7]+"/before")
        poison_outputs(model,kv,prefix)
        var plain = step[False](model,kv,ctx,ids)
        snapshot(model,kv,args[7]+"/plain")
        rewind(model,kv,prefix)
        poison_outputs(model,kv,prefix)
        var observed = step[True](model,kv,ctx,ids)
        snapshot(model,kv,args[7]+"/observed")
        if plain != observed or kv.length(0) != prefix+1 or model.submitted_rows != 24*(prefix+1):
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
            rewind(model,kv,prefix)
            if step[False](model,kv,ctx,ids) != winner:
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
            rewind(model,kv,prefix)
            if step[False](model,kv,ctx,ids) != winner:
                raise Error("unstable profile prediction")
        print("PROFILE_REGION_END")
        sleep(0.25)
        return
    var records = String()
    for arm_index in range(2):
        var arm = (first+arm_index)%2
        var observe = comparison == 1 and arm == 1
        for sample in range(20):
            rewind(model,kv,prefix)
            var start = now()
            var selected: Int
            if observe:
                selected = step[True](model,kv,ctx,ids)
            else:
                selected = step[False](model,kv,ctx,ids)
            var elapsed = now()-start
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
