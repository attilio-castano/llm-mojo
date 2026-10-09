"""One bounded arrival trace through the native engine, with complete line records.

Greedy runs the real Qwen checkpoint. Scripted runs a virtual-clock lifecycle
oracle. Input is generated and validated by model_profile's engine collector.
"""
from std.sys import argv
from std.time import sleep
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.model import VOCABULARY
from llm_mojo.models.qwen2.runner import QwenRunner
from llm_mojo.serving.engine import EngineCore, EngineStep, StepCost, TOKEN_EVENT
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.runner import ModelRunner, SimulatedRunner
from llm_mojo.serving.kv_pool import KVGeometry, KVPool


trait TraceRunner(ModelRunner):
    def reset_clock(mut self):
        ...

    def wait_until(mut self, at_ns: Int) raises:
        ...


@fieldwise_init
struct TraceMetal(TraceRunner):
    var inner: QwenRunner

    def now_ns(self) -> Int:
        return self.inner.now_ns()

    def reset_clock(mut self):
        self.inner.reset_clock()

    def wait_until(mut self, at_ns: Int) raises:
        sleep(Float64(max(at_ns-self.now_ns(),0))/1e9)

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        return self.inner.execute(batch,kv)


@fieldwise_init
struct TraceSimulation(TraceRunner):
    var inner: SimulatedRunner

    def now_ns(self) -> Int:
        return self.inner.now_ns()

    def reset_clock(mut self):
        self.inner.clock_ns = 0

    def wait_until(mut self, at_ns: Int):
        self.inner.clock_ns = max(self.inner.clock_ns,at_ns)

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        return self.inner.execute(batch,kv)


struct TraceRequest(Movable):
    var request_id: Int
    var arrival_ns: Int
    var maximum: Int
    var stops: List[Int]
    var prompt: List[Int]
    var observed_ns: Int
    var added: Bool
    var aborted: Bool

    def __init__(out self, request_id: Int, arrival_ns: Int, maximum: Int,
                 var stops: List[Int], var prompt: List[Int]):
        self.request_id = request_id
        self.arrival_ns = arrival_ns
        self.maximum = maximum
        self.stops = stops^
        self.prompt = prompt^
        self.observed_ns = -1
        self.added = False
        self.aborted = False


@fieldwise_init
struct TraceAbort(ImplicitlyCopyable, Movable):
    var request_id: Int
    var offset_ns: Int
    var applied: Bool


def integers(value: String) raises -> List[Int]:
    var result = List[Int]()
    if value == "-":
        return result^
    for part in value.split(","):
        result.append(Int(part))
    return result^


def emit(record: EngineStep):
    print("step",record.step_id,record.decode_seqs,record.prefill_seqs,record.prefill_tokens,
          record.total_tokens,record.attended_positions,record.admitted,record.preempted,
          record.finished,record.aborted,record.waiting,record.blocks_free,record.begin_ns,
          record.schedule_ns,record.build_ns,record.execute_ns,record.postprocess_ns,record.end_ns,
          record.predicted_ns,record.budget_limited)
    for event in record.events:
        if event.kind == TOKEN_EVENT:
            print("token",event.request_id,event.token_id,event.prompt_tokens,event.generated_tokens,
                  event.arrival_ns,event.emitted_ns)
        else:
            print("finish",event.request_id,event.reason,event.prompt_tokens,event.generated_tokens,
                  event.arrival_ns,event.emitted_ns)


def run_trace[Runner: TraceRunner](
    mut runner: Runner, mut kv: KVPool, mut requests: List[TraceRequest], mut aborts: List[TraceAbort],
    arm: String, budget: Int, maximum_sequences: Int, warmup_steps: Int, cost: StepCost,
    reserve_lifetime: Bool, admission_explicit: Bool,
) raises:
    var serial = arm == "serial"
    var static = arm == "static"
    var chunked = arm == "chunked" or arm == "adaptive"
    var sequences = 1 if serial else maximum_sequences
    var effective_budget = budget if chunked else 4096
    # Warm-up is untimed and produces no retained request/step observations.
    if warmup_steps > 0:
        var warm = EngineCore(kv.blocks,32,4096,VOCABULARY,effective_budget,sequences,128,0,chunked,
                              reserve_lifetime=reserve_lifetime)
        _ = warm.add(0,[11,13,17,19],warmup_steps,[])
        while warm.live() > 0:
            _ = warm.step(runner,kv)
        warm.check(kv)
    var engine = EngineCore(kv.blocks,32,4096,VOCABULARY,effective_budget,sequences,128,0,chunked,
                            reserve_lifetime=reserve_lifetime)
    if arm == "adaptive":
        engine.set_cost_policy(cost)
        print("policy",cost.target_ns,cost.fixed_ns,cost.per_row_ns,cost.per_position_ns,
              cost.per_partition_ns,cost.per_logit_ns)
    print("config",arm,kv.blocks,effective_budget,sequences)
    if admission_explicit:
        print("admission","reserved" if reserve_lifetime else "incremental")
    runner.reset_clock()
    var admitted = 0
    var iterations = 0
    while admitted < len(requests) or engine.live() > 0:
        var at = runner.now_ns()
        # Ingress is observed even when serial/static execution is queued.
        for i in range(len(requests)):
            if requests[i].observed_ns < 0 and requests[i].arrival_ns <= at:
                requests[i].observed_ns = at
                print("arrival",requests[i].request_id,requests[i].arrival_ns,at)
        for a in range(len(aborts)):
            if not aborts[a].applied and aborts[a].offset_ns <= at:
                aborts[a].applied = True
                engine.abort(aborts[a].request_id)
                for i in range(len(requests)):
                    if requests[i].request_id == aborts[a].request_id:
                        requests[i].aborted = True
        # Cancellation of an arrived queued request takes effect at this
        # boundary even when an execution cohort is still draining.
        for i in range(len(requests)):
            if not requests[i].added and requests[i].observed_ns >= 0 and requests[i].aborted:
                _ = engine.add(requests[i].request_id,requests[i].prompt,requests[i].maximum,
                               requests[i].stops,requests[i].arrival_ns)
                engine.abort(requests[i].request_id)
                requests[i].added = True
                admitted += 1
        var can_admit = not (serial or static) or engine.live() == 0
        if can_admit:
            var count = 0
            for i in range(len(requests)):
                if not requests[i].added and requests[i].observed_ns >= 0:
                    _ = engine.add(requests[i].request_id,requests[i].prompt,requests[i].maximum,
                                   requests[i].stops,requests[i].arrival_ns)
                    if requests[i].aborted:
                        engine.abort(requests[i].request_id)
                    requests[i].added = True
                    admitted += 1
                    count += 1
                    if serial or (static and count == sequences):
                        break
        if engine.live() > 0:
            var record = engine.step(runner,kv)
            emit(record)
            iterations += 1
            if iterations > 1000000:
                raise Error("bounded finite trace did not drain")
        elif admitted < len(requests):
            var next = Int.MAX
            for i in range(len(requests)):
                if requests[i].observed_ns < 0:
                    next = min(next,requests[i].arrival_ns)
                elif not requests[i].added:
                    next = at
            if next == Int.MAX:
                raise Error("no future arrival can advance the trace")
            runner.wait_until(next)
    engine.check(kv)
    if engine.blocks.free_blocks() != kv.blocks:
        raise Error("trace did not release every block")
    print("drained",admitted,iterations,engine.blocks.free_blocks(),runner.now_ns())


def main() raises:
    var args = argv()
    if len(args) < 9 or len(args) > 11:
        raise Error("engine_trace prepared trace.tsv arm blocks budget max_sequences warmup_steps greedy|scripted [policy.tsv] [incremental|reserved]")
    var arm = args[3]
    var blocks = Int(args[4])
    var budget = Int(args[5])
    var sequences = Int(args[6])
    var warmup = Int(args[7])
    var mode = args[8]
    if (arm != "serial" and arm != "static" and arm != "continuous" and arm != "chunked" and arm != "adaptive"):
        raise Error("invalid trace arm")
    if (blocks < 1 or blocks > 8192 or budget < 1 or budget > 4096 or sequences < 1 or sequences > 64
            or warmup < 0 or warmup > 4092 or (mode != "greedy" and mode != "scripted")):
        raise Error("invalid bounded trace configuration")
    var cost = StepCost(0,0,0,0,0)
    var admission = "incremental"
    var admission_explicit = False
    if arm == "adaptive":
        if len(args) < 10 or mode != "greedy":
            raise Error("adaptive trace requires Metal and a frozen cost policy")
        var fields = open(args[9],"r").read().split()
        if len(fields) != 7 or String(fields[0]) != "cost":
            raise Error("invalid cost policy record")
        cost = StepCost(Int(fields[1]),Int(fields[2]),Int(fields[3]),Int(fields[4]),Int(fields[5]),Int(fields[6]))
        cost.validate()
        if len(args) == 11:
            admission = args[10]
            admission_explicit = True
    elif len(args) == 10:
        admission = args[9]
        admission_explicit = True
    elif len(args) == 11:
        raise Error("a cost policy belongs only to the adaptive arm")
    if admission != "incremental" and admission != "reserved":
        raise Error("invalid trace admission policy")
    var reserve_lifetime = admission == "reserved"
    var requests = List[TraceRequest]()
    var aborts = List[TraceAbort]()
    var script: List[Int] = [11,13,17,19,23]
    var previous = -1
    for line in open(args[2],"r").read().splitlines():
        var fields = String(line).split()
        if len(fields) == 0:
            continue
        if fields[0] == "request" and len(fields) == 7:
            var id = Int(fields[1])
            var arrival = Int(fields[2])
            var maximum = Int(fields[3])
            if String(fields[6]) != "-" or arrival < previous or arrival < 0 or maximum < 0 or id < 0:
                raise Error("invalid request trace or unsupported teacher forcing")
            for i in range(len(requests)):
                if requests[i].request_id == id:
                    raise Error("duplicate trace request ID")
            requests.append(TraceRequest(id,arrival,maximum,integers(String(fields[4])),integers(String(fields[5]))))
            previous = arrival
        elif fields[0] == "abort" and len(fields) == 3:
            var offset = Int(fields[2])
            if offset < 0:
                raise Error("negative abort offset")
            aborts.append(TraceAbort(Int(fields[1]),offset,False))
        elif fields[0] == "script" and len(fields) == 2:
            script = integers(String(fields[1]))
        else:
            raise Error("invalid trace row")
    if len(requests) < 1 or len(requests) > 128:
        raise Error("trace requires 1..128 requests")
    # Preflight every request before allocating a model or running any work.
    var check = EngineCore(blocks,32,4096,VOCABULARY,budget,sequences,128,
                           reserve_lifetime=reserve_lifetime)
    for i in range(len(requests)):
        _ = check.add(requests[i].request_id,requests[i].prompt,requests[i].maximum,requests[i].stops,requests[i].arrival_ns)
    print("mode",mode)
    if mode == "scripted":
        var ctx = DeviceContext()
        var kv = KVPool(ctx,blocks,32,KVGeometry(1,1,1))
        var runner = TraceSimulation(SimulatedRunner(script,VOCABULARY))
        print("device simulated/virtual")
        run_trace(runner,kv,requests,aborts,arm,budget,sequences,warmup,cost,reserve_lifetime,admission_explicit)
    else:
        var rows = budget if arm == "chunked" or arm == "adaptive" else 4096
        var runner = TraceMetal(QwenRunner(args[1],4096,rows,min(sequences,rows)))
        var kv = KVPool(runner.inner.ctx,blocks,32,runner.inner.model.kv_geometry())
        print("device",runner.inner.ctx.name()+"/"+runner.inner.ctx.api())
        run_trace(runner,kv,requests,aborts,arm,budget,sequences,warmup,cost,reserve_lifetime,admission_explicit)
