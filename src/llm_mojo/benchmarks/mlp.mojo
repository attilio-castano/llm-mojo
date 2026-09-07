"""Seven materialized MLP stages, checked before paired timing or tracing."""
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.mlp import (
    MLPWeights,
    MLPWorkspace,
    enqueue_mlp_apple_gpu,
    enqueue_mlp_stage_apple_gpu,
)
from std.python import Python
from std.sys import argv
from std.sys.defines import is_defined, get_defined_int
from std.time import perf_counter_ns, sleep


def load(
    mut buffer: DeviceBuffer[DType.bfloat16], name: String, count: Int
) raises:
    var helper = Python.import_module("llm_mojo.benchmarks.mlp_contract")
    with buffer.map_to_host() as mapped:
        helper.load(Int(mapped.unsafe_ptr()), name, count)


def check(
    mut buffer: DeviceBuffer[DType.bfloat16],
    name: String,
    count: Int,
    composed: Bool = False,
) raises:
    var helper = Python.import_module("llm_mojo.benchmarks.mlp_contract")
    with buffer.map_to_host() as mapped:
        helper.check(Int(mapped.unsafe_ptr()), name, count, composed)


def dispatch(
    ctx: DeviceContext,
    mut weights: MLPWeights,
    mut work: MLPWorkspace,
    mut x: DeviceBuffer[DType.bfloat16],
    rows: Int,
    stage: Int,
) raises:
    var view = TileTensor(x, row_major(rows, 896))
    if stage == -1:
        enqueue_mlp_apple_gpu(ctx, weights, work, view)
    else:
        enqueue_mlp_stage_apple_gpu(ctx, weights, work, view, stage)


def main() raises:
    var args = List[String]()
    comptime if is_defined["GQA_PROFILE_ROWS"]():
        args = [
            "profile",
            String(get_defined_int["GQA_PROFILE_ROWS"]()),
            "1",
            "0",
            "0",
            "0",
            "1601",
            "profile",
            String(get_defined_int["GQA_PROFILE_ITERATIONS"]()),
            String(get_defined_int["GQA_PROFILE_WARMUP", default=10]()),
        ]
    else:
        for arg in argv():
            args.append(String(arg))
    if len(args) != 10:
        raise Error(
            "expected rows layers candidate control first seed mode repetitions warmup"
        )
    var rows = Int(args[1])
    var layers = Int(args[2])
    var first = Int(args[5])
    var mode = args[7]
    var repetitions = Int(args[8])
    var warmup = Int(args[9])
    var stage = -1
    if mode != "bench" and mode != "profile":
        for j in range(7):
            if mode == "stage" + String(j):
                stage = j
        if stage == -1:
            raise Error("unknown MLP measurement boundary")
    if (
        rows < 1
        or rows > 4096
        or (layers != 1 and layers != 24)
        or Int(args[3]) != 0
        or Int(args[4]) != 0
        or (first != 0 and first != 1)
        or Int(args[6]) != 1601
        or repetitions < 1
        or warmup < 0
        or warmup > 100
        or (stage >= 0 and layers != 1)
        or (mode == "profile" and (layers != 1 or repetitions * 7 > 5000))
    ):
        raise Error("invalid MLP measurement request")
    var helper = Python.import_module("llm_mojo.benchmarks.mlp_contract")
    helper.fixture_identity()
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("MLP benchmark requires Metal")
    print("device:", ctx.name())
    print("api:", ctx.api())
    print("operation: mlp")
    print("shape:", rows, layers, "seed: 1601")
    print("variants: 0 0 candidate-first:", first)
    var boundary = "whole_mlp" if stage == -1 else "mlp_stage_" + String(stage)
    print("measurement:", boundary)
    var work = MLPWorkspace(ctx, rows)
    var weights = List[MLPWeights]()
    var inputs = List[DeviceBuffer[DType.bfloat16]]()
    for _ in range(layers):
        var w = MLPWeights(ctx)
        load(w.norm, "norm", 896)
        load(w.gate, "gate", 896 * 4864)
        load(w.up, "up", 896 * 4864)
        load(w.down, "down", 896 * 4864)
        weights.append(w^)
        var x = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
        load(x, "X", rows * 896)
        inputs.append(x^)
    if stage >= 0:
        load(work.normalized, "N", rows * 896)
        load(work.gate, "G", rows * 4864)
        load(work.up, "U", rows * 4864)
        load(work.activated, "A", rows * 4864)
        load(work.gated, "S", rows * 4864)
        load(work.down, "D", rows * 896)
    for layer in range(layers):
        var poison = Float32(FloatLiteral.nan).cast[DType.bfloat16]()
        if stage == -1:
            work.down.enqueue_fill(poison)
            work.output.enqueue_fill(poison)
        elif stage == 0:
            work.normalized.enqueue_fill(poison)
        elif stage == 1:
            work.gate.enqueue_fill(poison)
        elif stage == 2:
            work.up.enqueue_fill(poison)
        elif stage == 3:
            work.activated.enqueue_fill(poison)
        elif stage == 4:
            work.gated.enqueue_fill(poison)
        elif stage == 5:
            work.down.enqueue_fill(poison)
        else:
            work.output.enqueue_fill(poison)
        dispatch(ctx, weights[layer], work, inputs[layer], rows, stage)
        if stage == -1:
            check(work.down, "D", rows * 896, True)
            check(work.output, "Y", rows * 896, True)
        elif stage == 0:
            check(work.normalized, "N", rows * 896)
        elif stage == 1:
            check(work.gate, "G", rows * 4864)
        elif stage == 2:
            check(work.up, "U", rows * 4864)
        elif stage == 3:
            check(work.activated, "A", rows * 4864)
        elif stage == 4:
            check(work.gated, "S", rows * 4864)
        elif stage == 5:
            check(work.down, "D", rows * 896)
        else:
            check(work.output, "Y", rows * 896)
    ctx.synchronize()
    print("correctness: passed")
    if mode == "profile":
        for _ in range(warmup):
            dispatch(ctx, weights[0], work, inputs[0], rows, -1)
        ctx.synchronize()
        print("profile implementation: enqueue_mlp_apple_gpu")
        print("entrypoint: enqueue_mlp_apple_gpu")
        print("rows:", rows)
        print("hidden: 896")
        print("intermediate size: 4864")
        print("profile workload:", "mlp-r" + String(rows) + "-v0")
        print("profile dispatches per iteration: 7")
        print("warmup iterations:", warmup)
        print("profile iterations:", repetitions)
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(repetitions):
            dispatch(ctx, weights[0], work, inputs[0], rows, -1)
        ctx.synchronize()
        print("PROFILE_REGION_END")
        sleep(0.25)
        return
    var samples = List[Float64]()
    for _ in range(2):
        for rep in range(warmup + repetitions):
            var start = perf_counter_ns()
            for layer in range(layers):
                dispatch(ctx, weights[layer], work, inputs[layer], rows, stage)
            ctx.synchronize()
            var elapsed = (
                Float64(perf_counter_ns() - start) / 1000.0 / Float64(layers)
            )
            if rep >= warmup:
                samples.append(elapsed)
    # Keep output overhead outside both arms.
    for arm in range(2):
        var label = "candidate" if ((arm == 0) == (first == 1)) else "control"
        for rep in range(repetitions):
            print("SAMPLE", label, 0, rep, samples[arm * repetitions + rep])
    print("BENCHMARK_COMPLETE")
