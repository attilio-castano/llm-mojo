"""Development-only full-model entrypoint; requires externally verified assets."""
from std.sys import argv
from std.memory import bitcast
from std.testing import assert_equal, assert_raises
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.model import CaptureRequest, LAYERS, QwenModel, VOCABULARY
from llm_mojo.models.qwen2.plan import MAX_CONTEXT, baseline_plan, configured_plan, execution_plan
from llm_mojo.runtime.clock import now
from model_operation_support import capture_operations


def lifecycle(path: String) raises:
    var ctx = DeviceContext()
    print("model device",ctx.name(),"backend",ctx.api())
    var model = QwenModel(ctx,path,4,3)
    var ids: List[Int] = [42,17,91]
    with assert_raises():
        _ = model.greedy(ctx)
    var configurations: List[Int] = [0,2,3,21]
    var one = baseline_plan(1,4)
    for configuration in configurations:
        model.reset(ctx)
        model.forward(ctx,ids,configured_plan(configuration,3,3))
        _ = model.greedy(ctx)
        assert_equal(model.length,3)
        assert_equal(model.submitted_rows,3*LAYERS)
        with assert_raises():
            model.forward(ctx,List[Int](),one)
        with assert_raises():
            model.forward(ctx,[-1],one)
        with assert_raises():
            model.forward(ctx,[VOCABULARY],one)
        with assert_raises():
            model.forward(ctx,[1,2],baseline_plan(2,5))
        with assert_raises():
            model.forward(ctx,[1],configured_plan(999,1,4))
        model.layers[LAYERS-1].cache.length = 2
        with assert_raises():
            model.forward(ctx,[1],one)
        model.layers[LAYERS-1].cache.length = 3
        assert_equal(model.length,3)
        assert_equal(model.submitted_rows,3*LAYERS)
        assert_equal(model.valid,True)
        model.forward(ctx,[2],configured_plan(0,1,4))
        _ = model.greedy(ctx)
        for i in range(LAYERS):
            assert_equal(model.layers[i].cache.length,4)
        with assert_raises():
            model.forward(ctx,[1],one)
    model.reset(ctx)
    model.forward(ctx,ids,configured_plan(0,3,3))
    var first = model.greedy(ctx)
    var bits = List[UInt16]()
    with model.logits.map_to_host() as mapped:
        for i in range(VOCABULARY):
            bits.append(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]))
    model.reset(ctx)
    model.forward(ctx,ids,configured_plan(0,3,3))
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
            model.forward(ctx,[1],one)
        model.reset(ctx)
        assert_equal(model.length,0)
        assert_equal(model.submitted_rows,0)
        model.forward(ctx,[1],one)
    print("lifecycle passed: configurations invalid-input overflow reset replay ties nonfinite invalidation")


def benchmark(path: String, plan: String, warmups: Int, samples: Int) raises:
    var ctx = DeviceContext()
    print("model device",ctx.name(),"backend",ctx.api())
    var model = QwenModel(ctx,path,MAX_CONTEXT,MAX_CONTEXT)
    var active_prefix = -1
    for line in open(plan,"r").read().splitlines():
        var spec = integers(String(line))
        var prefix = spec[0]
        var rows = spec[1]
        var config = spec[2]
        if prefix != active_prefix or prefix == 0:
            model.reset(ctx)
            if prefix > 0:
                var ids = List[Int]()
                for i in range(prefix):
                    ids.append((i*103+42)%151643)
                model.forward(ctx,ids,configured_plan(0,prefix,prefix))
                ctx.synchronize()
            active_prefix = prefix
        var ids = List[Int]()
        for i in range(rows):
            ids.append(((prefix+i)*103+42)%151643)
        for sample in range(-warmups,samples):
            # Reuse the unchanged real prefix, overwriting only the suffix.
            model.length = prefix
            model.submitted_rows = prefix*LAYERS
            for i in range(LAYERS):
                model.layers[i].cache.length = prefix
            var started = now()
            model.forward(ctx,ids,configured_plan(config,rows,prefix+rows))
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
    var model = QwenModel(ctx,args[1],min(MAX_CONTEXT,len(ids)+3),maximum)
    # Exact untouched-cache checks use a finite recognizable poison pattern.
    for i in range(LAYERS):
        model.layers[i].cache.key.enqueue_fill(123)
        model.layers[i].cache.value.enqueue_fill(123)
    var offset = 0
    for i in range(len(schedule)):
        var chunk = List[Int]()
        for j in range(schedule[i]):
            chunk.append(ids[offset+j])
        var cached = offset+schedule[i]
        var plan = execution_plan(args[4],schedule[i],cached,ctx.name()) if dynamic else configured_plan(configurations[i],schedule[i],cached)
        if args[5] != "-":
            model.forward_captured(ctx,chunk,plan,CaptureRequest(args[5]+"/call_"+String(i),False))
        else:
            model.forward(ctx,chunk,plan)
        print("call",i,"token",model.greedy(ctx),"cache_length",model.length,"submitted_layer_rows",model.submitted_rows,"configuration",plan.configuration)
        offset += schedule[i]
