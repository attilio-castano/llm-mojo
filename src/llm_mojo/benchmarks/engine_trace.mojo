"""One bounded arrival trace through the native engine, with complete line records.

Greedy runs the real Qwen checkpoint. Scripted runs a virtual-clock lifecycle
oracle. Input is generated and validated by model_profile's engine collector.
"""
from std.sys import argv
from std.time import sleep
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.model import VOCABULARY
from llm_mojo.models.qwen2.runner import QwenRunner, QwenAsyncRunner, select_engine_configuration
from llm_mojo.models.qwen2.plan import configured_plan
from llm_mojo.serving.engine import EngineCore, EngineStep, StepCost, TOKEN_EVENT
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.runner import ModelRunner, SimulatedRunner
from llm_mojo.serving.kv_pool import KVGeometry, KVPool


trait TraceRunner(ModelRunner):
    def reset_clock(mut self):
        ...

    def wait_until(mut self, at_ns: Int) raises:
        ...

    def emit_route(self, step_id: Int, rows: Int):
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

    def emit_route(self, step_id: Int, rows: Int):
        # Actual completed dispatch, with no new device observation.
        print("route",step_id,self.inner.model.last_route.configuration,rows,
              self.inner.model.last_route.sequences,self.inner.model.sampled_rows)


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

    def emit_route(self, step_id: Int, rows: Int):
        # engine-fast-v1 is Metal-only; simulation makes no route claim.
        return


struct TraceSyncObservation(ModelRunner):
    """Same synchronous reference adapter, with actual enqueue/readback marks."""
    var inner: QwenRunner
    var submit_begin_ns: Int
    var submitted_ns: Int
    var collect_begin_ns: Int
    var completed_ns: Int

    def __init__(out self, var runner: QwenRunner):
        self.inner = runner^
        self.submit_begin_ns = 0
        self.submitted_ns = 0
        self.collect_begin_ns = 0
        self.completed_ns = 0

    def now_ns(self) -> Int:
        return self.inner.now_ns()

    def execute(mut self, batch: StepBatch, mut kv: KVPool) raises -> List[Int]:
        var total = 0
        for length in batch.seq_lens:
            total = max(total,length)
        self.submit_begin_ns = self.now_ns()
        try:
            self.inner.model.forward(self.inner.ctx,batch,kv,
                configured_plan(select_engine_configuration(batch),batch.rows(),total,batch.sequences()))
            self.submitted_ns = self.now_ns()
            self.collect_begin_ns = self.now_ns()
            var tokens = self.inner.model.greedy_tokens(self.inner.ctx)
            self.completed_ns = self.now_ns()
            return tokens^
        except error:
            self.inner.model.valid = False
            self.inner.ctx.synchronize()
            raise error


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


def emit(record: EngineStep, observe_kv: Bool = False):
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
    if observe_kv:
        if record.admitted_request_id >= 0:
            print("admit",record.step_id,record.admitted_request_id,record.admitted_ns)
        for point in record.kv_observations:
            var phase = "start"
            if point.phase == 1:
                phase = "scheduled"
            elif point.phase == 2:
                phase = "executed"
            elif point.phase == 3:
                phase = "end"
            print("kv",record.step_id,phase,point.at_ns,point.allocated_blocks,point.written_blocks,
                  point.written_tokens,point.reserved_tokens,point.waiting_requests,point.resident_requests)
        print("kv_execute",record.step_id,record.execute_begin_ns,record.execute_end_ns)


def emit_async_events(record: EngineStep, ticket: Int):
    for event in record.events:
        if event.kind == TOKEN_EVENT:
            print("async_token",event.request_id,event.token_id,event.prompt_tokens,event.generated_tokens,
                  event.arrival_ns,event.emitted_ns,ticket)
        else:
            # Queued aborts and zero limits are host boundaries without a
            # selected head. Do not attach them to an unrelated GPU ticket.
            var source = -1
            for token in record.events:
                if token.kind == TOKEN_EVENT and token.request_id == event.request_id:
                    source = ticket
            print("async_finish",event.request_id,event.reason,event.prompt_tokens,event.generated_tokens,
                  event.arrival_ns,event.emitted_ns,source)


def emit_async(record: EngineStep):
    for submission in record.async_submissions:
        print("submit",submission.ticket,submission.buffer_slot,submission.decode_seqs,submission.prefill_seqs,
              submission.prefill_tokens,submission.total_tokens,submission.attended_positions,
              submission.selected_logits,submission.begin_ns,submission.submitted_ns,submission.pending)
        for head in record.async_heads:
            if head.ticket == submission.ticket:
                print("head",head.ticket,head.head,head.request_id,head.generated_tokens)
    for completion in record.async_completions:
        print("complete",completion.ticket,completion.begin_ns,completion.completed_ns,completion.pending)
    for result in record.async_results:
        print("result",result.ticket,result.head,result.request_id,result.token_id,result.generated_tokens,
              result.disposition,result.observed_ns)
    emit_async_events(record,record.async_ticket)


def emit_sync_comparison(record: EngineStep, runner: TraceSyncObservation, ticket: Int):
    if record.total_tokens > 0:
        var selected = 0
        for event in record.events:
            selected += 1 if event.kind == TOKEN_EVENT else 0
        print("submit",ticket,ticket % 2,record.decode_seqs,record.prefill_seqs,record.prefill_tokens,
              record.total_tokens,record.attended_positions,selected,
              runner.submit_begin_ns,runner.submitted_ns,1)
        var head = 0
        for event in record.events:
            if event.kind == TOKEN_EVENT:
                print("head",ticket,head,event.request_id,event.generated_tokens)
                head += 1
        print("complete",ticket,runner.collect_begin_ns,runner.completed_ns,0)
        head = 0
        for event in record.events:
            if event.kind == TOKEN_EVENT:
                print("result",ticket,head,event.request_id,event.token_id,event.generated_tokens,"delivered",event.emitted_ns)
                head += 1
    emit_async_events(record,ticket if record.total_tokens > 0 else -1)


def async_ingress(at: Int, mut engine: EngineCore, mut requests: List[TraceRequest],
                  mut aborts: List[TraceAbort], mut added: Int) raises:
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
    for i in range(len(requests)):
        if not requests[i].added and requests[i].observed_ns >= 0:
            _ = engine.add(requests[i].request_id,requests[i].prompt,requests[i].maximum,
                           requests[i].stops,requests[i].arrival_ns)
            if requests[i].aborted:
                engine.abort(requests[i].request_id)
            requests[i].added = True
            added += 1


def wait_async_arrival(at: Int, requests: List[TraceRequest]) raises:
    var next = Int.MAX
    for index in range(len(requests)):
        if requests[index].observed_ns < 0:
            next = min(next,requests[index].arrival_ns)
    if next == Int.MAX:
        raise Error("no future arrival can advance the async trace")
    sleep(Float64(max(next-at,0))/1e9)


def async_identity(blocks: Int, execution: String):
    print("config","chunked",blocks,256,8)
    print("admission","reserved")
    print("study","engine-async-v1")
    print("work_capacity",256,8)
    print("execution",execution)
    print("completion_mode","two-context-prefix-wait" if execution == "async" else "synchronous-readback")
    print("async_capacity",2)


def emit_async_drain(engine: EngineCore, kv: KVPool, requests: Int, submitted: Int,
                     completed: Int, delivered: Int, discarded: Int, at: Int) raises:
    engine.check(kv)
    var written = 0
    for count in kv.written:
        written += count
    var free = engine.blocks.free_blocks()
    if free != kv.blocks or written or engine.live() or engine.pending_steps():
        raise Error("async trace did not drain all submissions and KV ownership")
    print("async_drained",requests,submitted,completed,delivered,discarded,free,
          kv.blocks-free,written,engine.live(),engine.pending_steps(),at)


def run_sync_comparison(mut runner: TraceSyncObservation, mut kv: KVPool, mut requests: List[TraceRequest],
                        mut aborts: List[TraceAbort], warmup_steps: Int) raises:
    if warmup_steps > 0:
        var warm = EngineCore(kv.blocks,32,4096,VOCABULARY,256,8,128,0,True,reserve_lifetime=True)
        _ = warm.add(0,[11,13,17,19],warmup_steps,[])
        while warm.live() > 0:
            _ = warm.step(runner,kv)
        warm.check(kv)
    var engine = EngineCore(kv.blocks,32,4096,VOCABULARY,256,8,128,0,True,reserve_lifetime=True)
    async_identity(kv.blocks,"sync")
    runner.inner.reset_clock()
    var added = 0
    var submitted = 0
    var delivered = 0
    var iterations = 0
    while added < len(requests) or engine.live() > 0:
        var at = runner.now_ns()
        async_ingress(at,engine,requests,aborts,added)
        if engine.live() > 0:
            var record = engine.step(runner,kv)
            emit_sync_comparison(record,runner,submitted)
            submitted += 1 if record.total_tokens > 0 else 0
            for event in record.events:
                delivered += 1 if event.kind == TOKEN_EVENT else 0
            iterations += 1
            if iterations > 1000000:
                raise Error("bounded synchronous comparison did not drain")
        elif added < len(requests):
            wait_async_arrival(at,requests)
    emit_async_drain(engine,kv,added,submitted,submitted,delivered,0,runner.now_ns())


def run_async_comparison(mut runner: QwenAsyncRunner, mut kv: KVPool, mut requests: List[TraceRequest],
                         mut aborts: List[TraceAbort], warmup_steps: Int) raises:
    if warmup_steps > 0:
        var warm = EngineCore(kv.blocks,32,4096,VOCABULARY,256,8,128,0,True,reserve_lifetime=True)
        _ = warm.add(0,[11,13,17,19],warmup_steps,[])
        while warm.live() > 0:
            _ = warm.step_async(runner,kv)
        _ = warm.drain_async(runner,kv)
        warm.check(kv)
    var engine = EngineCore(kv.blocks,32,4096,VOCABULARY,256,8,128,0,True,reserve_lifetime=True)
    async_identity(kv.blocks,"async")
    runner.reset_clock()
    var added = 0
    var submitted = 0
    var completed = 0
    var delivered = 0
    var discarded = 0
    var iterations = 0
    while added < len(requests) or engine.live() > 0 or engine.pending_steps() > 0:
        var at = runner.now_ns()
        async_ingress(at,engine,requests,aborts,added)
        if engine.live() > 0 or engine.pending_steps() > 0:
            var record = engine.step_async(runner,kv)
            emit_async(record)
            submitted += len(record.async_submissions)
            completed += len(record.async_completions)
            for result in record.async_results:
                discarded += 1 if result.disposition != "delivered" else 0
            for event in record.events:
                delivered += 1 if event.kind == TOKEN_EVENT else 0
            iterations += 1
            if iterations > 1000000:
                raise Error("bounded asynchronous comparison did not drain")
        elif added < len(requests):
            wait_async_arrival(at,requests)
    emit_async_drain(engine,kv,added,submitted,completed,delivered,discarded,runner.now_ns())


def run_trace[Runner: TraceRunner](
    mut runner: Runner, mut kv: KVPool, mut requests: List[TraceRequest], mut aborts: List[TraceAbort],
    arm: String, budget: Int, maximum_sequences: Int, warmup_steps: Int, cost: StepCost,
    reserve_lifetime: Bool, admission_explicit: Bool, observe_kv: Bool, budget_study: Bool,
    fast_study: Bool, runner_kind: String,
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
                            reserve_lifetime=reserve_lifetime,observe_kv=observe_kv)
    if arm == "adaptive":
        engine.set_cost_policy(cost)
        print("policy",cost.target_ns,cost.fixed_ns,cost.per_row_ns,cost.per_position_ns,
              cost.per_partition_ns,cost.per_logit_ns)
    print("config",arm,kv.blocks,effective_budget,sequences)
    if admission_explicit:
        print("admission","reserved" if reserve_lifetime else "incremental")
    if observe_kv:
        print("telemetry","admission-range-v1")
        # The pool's actual BF16 geometry binds byte-time even in scripted mode.
        print("kv_geometry",kv.geometry.layers,kv.geometry.kv_heads,kv.geometry.head_dim,2)
    if budget_study or fast_study:
        print("study","engine-fast-v1" if fast_study else "engine-budget-v1")
        print("work_capacity",256,8)
    if fast_study:
        print("runner",runner_kind)
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
            emit(record,observe_kv)
            if fast_study and record.total_tokens > 0:
                runner.emit_route(record.step_id,record.total_tokens)
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
    if len(args) < 9 or len(args) > 13:
        raise Error("engine_trace prepared trace.tsv arm blocks budget max_sequences warmup_steps greedy|scripted [policy.tsv] [incremental|reserved [admission-range-v1|engine-budget-v1|engine-fast-v1|engine-async-v1 [reference|fast-decode|sync|async]]]")
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
    var observe_kv = False
    var selector = ""
    var selector_explicit = False
    var runner_kind = "reference"
    var runner_explicit = False
    if arm == "adaptive":
        if len(args) < 10 or mode != "greedy":
            raise Error("adaptive trace requires Metal and a frozen cost policy")
        var fields = open(args[9],"r").read().split()
        if len(fields) != 7 or String(fields[0]) != "cost":
            raise Error("invalid cost policy record")
        cost = StepCost(Int(fields[1]),Int(fields[2]),Int(fields[3]),Int(fields[4]),Int(fields[5]),Int(fields[6]))
        cost.validate()
        if len(args) >= 11:
            admission = args[10]
            admission_explicit = True
        if len(args) >= 12:
            selector = args[11]
            selector_explicit = True
        if len(args) == 13:
            runner_kind = args[12]
            runner_explicit = True
    else:
        if len(args) > 12:
            raise Error("a cost policy belongs only to the adaptive arm")
        if len(args) >= 10:
            admission = args[9]
            admission_explicit = True
        if len(args) >= 11:
            selector = args[10]
            selector_explicit = True
        if len(args) == 12:
            runner_kind = args[11]
            runner_explicit = True
    if selector_explicit:
        if (selector != "admission-range-v1" and selector != "engine-budget-v1"
                and selector != "engine-fast-v1" and selector != "engine-async-v1"):
            raise Error("invalid trace study selector")
        observe_kv = selector == "admission-range-v1"
    var budget_study = selector == "engine-budget-v1"
    var fast_study = selector == "engine-fast-v1"
    var async_study = selector == "engine-async-v1"
    if runner_explicit and not ((fast_study and (runner_kind == "reference" or runner_kind == "fast-decode"))
                               or (async_study and (runner_kind == "sync" or runner_kind == "async"))):
        raise Error("runner selector differs from its explicit Fast or async study")
    if fast_study and (mode != "greedy" or not runner_explicit):
        raise Error("engine-fast-v1 requires greedy execution and an explicit runner selector")
    if async_study and (not runner_explicit or mode != "greedy" or arm != "chunked"
                        or budget != 256 or sequences != 8 or not admission_explicit or admission != "reserved"):
        raise Error("engine-async-v1 requires fixed256/eight slots, reserved admission, greedy and sync|async")
    if budget_study or fast_study:
        if (not admission_explicit or sequences != 8
                or (arm != "chunked" and arm != "adaptive")
                or (budget != 32 and budget != 64 and budget != 128 and budget != 256)
                or (arm == "adaptive" and budget != 256)):
            raise Error("invalid engine study fixed work-capacity configuration")
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
    if async_study:
        if runner_kind == "sync":
            var runner = TraceSyncObservation(QwenRunner(args[1],4096,256,8))
            var kv = KVPool(runner.inner.ctx,blocks,32,runner.inner.model.kv_geometry())
            print("device",runner.inner.ctx.name()+"/"+runner.inner.ctx.api())
            if runner.inner.ctx.api() != "metal":
                raise Error("async comparison requires a verified Metal backend")
            run_sync_comparison(runner,kv,requests,aborts,warmup)
        else:
            var runner = QwenAsyncRunner(args[1],4096,256,8)
            var kv = KVPool(runner.ctx,blocks,32,runner.model.kv_geometry())
            print("device",runner.ctx.name()+"/"+runner.ctx.api())
            if runner.ctx.api() != "metal":
                raise Error("async comparison requires a verified Metal backend")
            run_async_comparison(runner,kv,requests,aborts,warmup)
    elif mode == "scripted":
        var ctx = DeviceContext()
        var kv = KVPool(ctx,blocks,32,KVGeometry(1,1,1))
        var runner = TraceSimulation(SimulatedRunner(script,VOCABULARY))
        print("device simulated/virtual")
        run_trace(runner,kv,requests,aborts,arm,budget,sequences,warmup,cost,reserve_lifetime,admission_explicit,observe_kv,budget_study,fast_study,runner_kind)
    else:
        # Budget studies vary scheduler work while holding the model's physical
        # row/sequence workspace capacity and metadata strides fixed.
        var rows = budget if arm == "chunked" or arm == "adaptive" else 4096
        var runner_sequences = min(sequences,rows)
        if budget_study or fast_study:
            rows = 256
            runner_sequences = 8
        var runner = TraceMetal(QwenRunner(args[1],4096,rows,runner_sequences,fast_decode=runner_kind == "fast-decode"))
        var kv = KVPool(runner.inner.ctx,blocks,32,runner.inner.model.kv_geometry())
        print("device",runner.inner.ctx.name()+"/"+runner.inner.ctx.api())
        run_trace(runner,kv,requests,aborts,arm,budget,sequences,warmup,cost,reserve_lifetime,admission_explicit,observe_kv,budget_study,fast_study,runner_kind)
