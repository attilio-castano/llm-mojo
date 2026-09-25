"""Development-only full-model entrypoint; requires externally verified assets."""
from std.sys import argv
from std.memory import bitcast
from std.testing import assert_equal, assert_raises
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.model import CaptureRequest, LAYERS, QwenModel, VOCABULARY
from llm_mojo.models.qwen2.plan import MAX_CONTEXT, baseline_plan, configured_plan, execution_plan
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVGeometry, KVPool
from model_operation_support import capture_operations


def lifecycle(path: String) raises:
    var ctx = DeviceContext()
    print("model device",ctx.name(),"backend",ctx.api())
    var model = QwenModel(ctx,path,4,3)
    var kv = KVPool(ctx,1,4,model.kv_geometry())
    var ids: List[Int] = [42,17,91]
    with assert_raises():
        _ = model.greedy(ctx)
    var configurations: List[Int] = [0,2,3,21]
    var one = baseline_plan(1,4)
    for configuration in configurations:
        model.reset(ctx)
        kv.reset(ctx)
        model.forward(ctx,StepBatch.sequence(ids,0,0,4),kv,configured_plan(configuration,3,3))
        _ = model.greedy(ctx)
        assert_equal(kv.length(0),3)
        assert_equal(model.submitted_rows,3*LAYERS)
        with assert_raises():
            model.forward(ctx,StepBatch.sequence(List[Int](),3,0,4),kv,one)
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([-1],3,0,4),kv,one)
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([VOCABULARY],3,0,4),kv,one)
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1,2],3,0,4),kv,baseline_plan(2,5))
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1],3,0,4),kv,configured_plan(999,1,4))
        # A step must start exactly where every layer's cache ends.
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1],2,0,4),kv,baseline_plan(1,3))
        kv.caches[kv.index(0,LAYERS-1)].length = 2
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1],3,0,4),kv,one)
        kv.caches[kv.index(0,LAYERS-1)].length = 3
        # The pool must match the model's geometry, and one call covers one sequence.
        var mismatched = KVPool(ctx,1,8,model.kv_geometry())
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1],0,0,8),mismatched,baseline_plan(1,1))
        assert_equal(mismatched.length(0),0)
        var narrow = KVPool(ctx,1,4,KVGeometry(LAYERS,1,64))
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1],0,0,4),narrow,baseline_plan(1,1))
        assert_equal(narrow.length(0),0)
        var pair = KVPool(ctx,2,4,model.kv_geometry())
        with assert_raises():
            model.forward(ctx,StepBatch([5,6],[0,0],[0,1,2],2,[1,1],1,[0,1],[0,4],[0,1]),pair,baseline_plan(2,2))
        assert_equal(pair.length(0),0)
        assert_equal(pair.length(1),0)
        assert_equal(kv.length(0),3)
        assert_equal(model.submitted_rows,3*LAYERS)
        assert_equal(model.valid,True)
        model.forward(ctx,StepBatch.sequence([2],3,0,4),kv,configured_plan(0,1,4))
        _ = model.greedy(ctx)
        for i in range(LAYERS):
            assert_equal(kv.caches[kv.index(0,i)].length,4)
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1],4,0,4),kv,one)
    model.reset(ctx)
    kv.reset(ctx)
    model.forward(ctx,StepBatch.sequence(ids,0,0,4),kv,configured_plan(0,3,3))
    var first = model.greedy(ctx)
    var bits = List[UInt16]()
    with model.logits.map_to_host() as mapped:
        for i in range(VOCABULARY):
            bits.append(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]))
    model.reset(ctx)
    kv.reset(ctx)
    model.forward(ctx,StepBatch.sequence(ids,0,0,4),kv,configured_plan(0,3,3))
    assert_equal(model.greedy(ctx),first)
    with model.logits.map_to_host() as mapped:
        for i in range(VOCABULARY):
            assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]),bits[i])
    model.logits.enqueue_fill(-3)
    with model.logits.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=7] = 5
        mapped.unsafe_ptr()[unsafe_offset=9] = 5
    assert_equal(model.greedy(ctx),7)
    var patterns: List[UInt16] = [0x7fc0,0x7f80,0xff80]
    for pattern in patterns:
        model.logits.enqueue_fill(0)
        with model.logits.map_to_host() as mapped:
            mapped.unsafe_ptr()[unsafe_offset=VOCABULARY-1] = bitcast[DType.bfloat16](pattern)
        with assert_raises():
            _ = model.greedy(ctx)
        assert_equal(model.valid,False)
        with assert_raises():
            model.forward(ctx,StepBatch.sequence([1],kv.length(0),0,4),kv,one)
        model.reset(ctx)
        kv.reset(ctx)
        assert_equal(kv.length(0),0)
        assert_equal(model.submitted_rows,0)
        model.forward(ctx,StepBatch.sequence([1],0,0,4),kv,one)
    print("lifecycle passed: configurations invalid-input overflow reset replay ties nonfinite invalidation")


def benchmark(path: String, plan: String, warmups: Int, samples: Int) raises:
    var ctx = DeviceContext()
    print("model device",ctx.name(),"backend",ctx.api())
    var model = QwenModel(ctx,path,MAX_CONTEXT,MAX_CONTEXT)
    var kv = KVPool(ctx,1,MAX_CONTEXT,model.kv_geometry())
    var active_prefix = -1
    for line in open(plan,"r").read().splitlines():
        var spec = integers(String(line))
        var prefix = spec[0]
        var rows = spec[1]
        var config = spec[2]
        if prefix != active_prefix or prefix == 0:
            model.reset(ctx)
            kv.reset(ctx)
            if prefix > 0:
                var ids = List[Int]()
                for i in range(prefix):
                    ids.append((i*103+42)%151643)
                model.forward(ctx,StepBatch.sequence(ids,0,0,MAX_CONTEXT),kv,configured_plan(0,prefix,prefix))
                ctx.synchronize()
            active_prefix = prefix
        var ids = List[Int]()
        for i in range(rows):
            ids.append(((prefix+i)*103+42)%151643)
        for sample in range(-warmups,samples):
            # Reuse the unchanged real prefix, overwriting only the suffix.
            model.submitted_rows = prefix*LAYERS
            kv.truncate(0,prefix)
            var batch = StepBatch.sequence(ids,prefix,0,MAX_CONTEXT)
            var started = now()
            model.forward(ctx,batch,kv,configured_plan(config,rows,prefix+rows))
            ctx.synchronize()
            var elapsed = now()-started
            _ = model.greedy(ctx)
            if sample >= 0:
                print("sample",spec[3],spec[4],prefix,rows,config,sample,elapsed)


def integers(text: String) raises -> List[Int]:
    var values = List[Int]()
    for item in text.split(","):
        values.append(Int(String(item)))
    return values^


def main() raises:
    var args = argv()
    if len(args) == 3 and args[1] == "--lifecycle":
        lifecycle(args[2])
        return
    if len(args) == 6 and args[1] == "--bench":
        benchmark(args[2],args[3],Int(args[4]),Int(args[5]))
        return
    if len(args) == 5 and args[1] == "--operations":
        var ctx = DeviceContext()
        capture_operations(ctx,args[2],args[3],args[4])
        return
    if len(args) != 6:
        raise Error("model_driver prepared-dir comma-token-ids schedule configurations capture-root")
    var ids = integers(args[2])
    var schedule = integers(args[3])
    var dynamic = args[4] == "fast" or args[4] == "baseline" or args[4] == "consistent"
    var configurations = List[Int](length=len(schedule),fill=0) if dynamic else integers(args[4])
    if len(schedule) != len(configurations):
        raise Error("one configuration is required per call")
    var maximum = 0
    var total = 0
    for rows in schedule:
        if rows < 1:
            raise Error("empty schedule call")
        maximum = max(maximum,rows)
        total += rows
    if total != len(ids) or total > MAX_CONTEXT:
        raise Error("schedule does not cover token IDs")
    var ctx = DeviceContext()
    print("model device",ctx.name(),"backend",ctx.api())
    var capacity = min(MAX_CONTEXT,len(ids)+3)
    var model = QwenModel(ctx,args[1],capacity,maximum)
    var kv = KVPool(ctx,1,capacity,model.kv_geometry())
    # Exact untouched-cache checks use a finite recognizable poison pattern.
    for i in range(LAYERS):
        kv.caches[kv.index(0,i)].key.enqueue_fill(123)
        kv.caches[kv.index(0,i)].value.enqueue_fill(123)
    var offset = 0
    for i in range(len(schedule)):
        var chunk = List[Int]()
        for j in range(schedule[i]):
            chunk.append(ids[offset+j])
        var cached = offset+schedule[i]
        var plan = execution_plan(args[4],schedule[i],cached,ctx.name()) if dynamic else configured_plan(configurations[i],schedule[i],cached)
        var batch = StepBatch.sequence(chunk,offset,0,capacity)
        if args[5] != "-":
            model.forward_captured(ctx,batch,kv,plan,CaptureRequest(args[5]+"/call_"+String(i),False))
        else:
            model.forward(ctx,batch,kv,plan)
        print("call",i,"token",model.greedy(ctx),"cache_length",kv.length(0),"submitted_layer_rows",model.submitted_rows,"configuration",plan.configuration)
        offset += schedule[i]
