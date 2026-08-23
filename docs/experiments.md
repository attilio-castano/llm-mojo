# Experimental method

This document defines how the project plans, runs, records, and interprets
performance experiments. It is the canonical protocol for experimental
evidence; individual benchmark programs document their own executable boundary,
and individual experiment records document what happened in a particular set of
runs.

The central distinction is:

> A benchmark produces measurements. An experiment interprets measurements. A
> project document states the current accepted understanding.

Keeping those roles separate lets the project preserve failed and superseded
work without confusing historical observations with the current implementation
or its supported performance claims.

## Vocabulary

Use these terms consistently in code, documentation, and results:

| Term | Meaning |
| --- | --- |
| **Contract** | The values an operation must compute, including arithmetic, dtypes, shapes, layouts, and numerical acceptance. |
| **Implementation** | A particular host or device path that attempts to satisfy a contract. |
| **Workload** | Exact inputs and execution conditions presented to an implementation. |
| **Benchmark** | Reusable executable code that measures an implementation under one or more workloads. |
| **Run** | One execution of a benchmark at a particular commit, on a particular machine, under recorded conditions. |
| **Experiment** | A declared question or hypothesis, a frozen comparison protocol, and one or more runs. |
| **Finding** | A bounded interpretation supported by an experiment's results. |
| **Decision** | A project choice, such as retaining an implementation or introducing a dispatch rule, informed by one or more findings. |

The evidence path is:

```text
operation contract
        ↓
implementations × workloads
        ↓
executable benchmark
        ↓
individual runs
        ↓
experiment finding
        ↓
accepted design or dispatch decision
```

A benchmark result does not by itself establish why performance changed. A
microkernel finding does not by itself establish decoder-block or model-level
performance.

## Evidence levels

Every result must name the narrowest level it directly measures:

1. **Operation:** one operation or kernel, such as RMSNorm.
2. **Composition:** a decoder block or another composition of operations.
3. **Model phase:** prefill, incremental prefill, or decode within the complete
   model path.
4. **End to end:** user-visible inference behavior such as time to first token
   or time per output token.

Evidence may move upward only through a separate measurement at the higher
level. A faster operation is a reason to measure its caller, not evidence that
the caller became faster.

## Levels of experimental formality

Not every diagnostic needs a durable record. Match the process to the claim:

- **Exploration** is local profiling, debugging, or parameter probing. It may
  run from a dirty worktree, but it supports no retained performance claim.
- **Recorded experiment** has a stable experiment identifier, a protocol
  declared before its decisive runs, correctness evidence, raw samples,
  provenance, and a bounded conclusion.
- **Reference result** is a recorded experiment repeated well enough to become
  the project's current baseline or to support a design decision. Canonical
  operation or performance documentation links to its experiment identifier.

Record a negative or inconclusive experiment when it rules out a plausible
hypothesis, changes a design decision, or would otherwise be likely to be
repeated. Ordinary compiler errors and debugging attempts remain exploration.

## Experiment lifecycle

### 1. State the question

Declare one primary question or hypothesis before collecting the decisive
measurements. A characterization experiment may define the quantities and
workload matrix without predicting a direction. An optimization experiment
should name the expected mechanism and the workloads where the effect should
appear. For example:

```text
Replacing a full-threadgroup shared-memory reduction with SIMD-group
reductions will reduce synchronized RMSNorm latency for rows=1 by removing
threadgroup barriers, without changing program-requested global traffic.
```

A useful hypothesis can be rejected. "Try vectorization" is an activity, not a
hypothesis.

### 2. Freeze the comparison

Before the decisive run, specify:

- baseline and variant implementations;
- correctness contract and acceptance tolerance;
- workload matrix;
- measurements and timing boundaries;
- warmup, repetition, and ordering policy;
- environment requirements;
- acceptance, rejection, and stop conditions;
- the evidence level and explicit nonclaims.

If the protocol changes after results are visible, record the change as a
deviation or create a new experiment. Do not silently redefine the question to
fit the observation.

### 3. Establish correctness and identity

Both baseline and variant must pass the same numerical contract before their
timings are compared. A benchmark's lightweight guard is useful but does not
replace the operation's provenance-bearing oracle tests.

GPU evidence must prove the selected device and backend at runtime. Silent CPU
fallback, successful compilation, or the presence of GPU tooling is not GPU
execution evidence.

For trace comparisons, filenames, directory names, command-line positions, and
human labels are not implementation identity. Generate a unique capture ID at
launch, carry it through the capture receipt, profiler run name, and launched
process name, then require the analyzer to match those observations. The
receipt must bind the runtime entrypoint, device, backend, shape, and iteration
counts to the binary and provenance hashes. A comparator derives semantic roles
from that verified identity; it does not accept caller-assigned baseline or
variant roles.

### 4. Estimate noise

Warm the compiled execution path before measurement. Retain individual samples
and report a distribution, not only the best run or one arithmetic mean.

When comparing implementations, prefer paired or interleaved baseline and
variant blocks on the same machine and toolchain. This reduces bias from
temperature, background load, power state, and run order. Repeat enough
baseline blocks to establish whether the proposed effect is larger than normal
run-to-run variation.

Do not remove an observation merely because it is inconvenient. If an external
event invalidates a sample, record the exclusion and its predeclared rule.

### 5. Benchmark and profile separately

The headline benchmark should contain only the synchronization and
instrumentation required by its declared timing boundary. A profiling run may
repeat a workload for longer, collect hardware counters, serialize work, or
otherwise perturb normal execution.

Use the benchmark to establish **whether** performance changed. Use traces,
counters, and generated code to investigate **why**. Do not publish an
instrumented duration as if it were the ordinary benchmark duration unless the
profiling overhead is part of the declared measurement.

### 6. Interpret and decide

An optimization may support an empirical project decision only when:

- correctness and device-identity gates pass;
- the measured effect exceeds the established noise for a relevant workload;
- important neighboring workloads do not regress beyond the declared bound;
- the conclusion stays within the measured evidence level.

A causal finding additionally requires profile or generated-code evidence
consistent with the proposed mechanism. The project may retain an empirically
faster implementation while explicitly marking its mechanism unresolved; it
may not promote the timing difference into a causal explanation.

If implementations cross over across repeatable workloads, retain multiple
paths and introduce a simple dispatch rule only when the crossover matters to a
real caller. Do not build speculative dispatch machinery from one isolated
microbenchmark.

## Workload contract

A workload must be reproducible without relying on a label such as "decode" or
"prefill" alone. Record:

- a stable workload identifier;
- semantic role, such as batch-1 decode or first-turn prefill;
- all logical shapes and physical strides;
- dtypes and accumulation dtypes;
- deterministic input source or seed;
- implementation parameters such as threadgroup size or vector width;
- allocation ownership and whether buffers are reused;
- operations included and excluded from timing;
- synchronization before and after the timed region.

For an operation applied independently to token rows, record both the physical
row count and its model interpretation. At a high level:

```text
decode rows  ≈ batch size
prefill rows ≈ batch size × active sequence positions
```

The same row count can represent different model situations. The numerical
shape remains the primary workload identity; the semantic label explains why
the project cares about it.

## Timing boundaries

Keep these measurements distinct when they are relevant:

- compilation or pipeline creation;
- allocation and initialization;
- host submission or enqueue overhead;
- synchronized device execution;
- host-device mapping or explicit copies;
- operation-level latency;
- composed model-phase latency;
- end-to-end inference latency.

Excluding a cost is valid when the caller amortizes or owns it, but the result
must say so. Synchronization must be explicit enough that asynchronous enqueue
time is never mistaken for completed device work.

Report raw samples together with at least a central value and spread. Throughput
must retain the corresponding elapsed-time samples and exact work definition so
it can be recomputed.

## Memory and traffic claims

Apple Silicon's unified memory does not make all memory quantities equivalent.
Distinguish:

1. **Allocated footprint:** bytes owned by live buffers.
2. **Logical tensor traffic:** bytes implied by the declared input, weight, and
   output tensors for a chosen comparison convention.
3. **Program-requested traffic:** loads and stores expressed by the
   implementation, including repeated reads.
4. **Observed hardware traffic:** cache, fabric, or DRAM activity reported by a
   named counter and profiling procedure.

Derived byte throughput must name which numerator it uses. Never label logical
or source-derived bytes as measured DRAM bandwidth. Cache reuse, coalescing,
prefetching, register spills, and counter semantics can all make observed
traffic differ from source-level accounting.

## Run record

A recorded run must include:

- experiment and run identifiers;
- UTC start time;
- repository commit, branch, and dirty state;
- implementation symbol or path and all parameters;
- workload identifier and full workload contract;
- exact command;
- hardware profile and runtime device/backend identity;
- macOS, Xcode, Metal, `uv`, Mojo, and MAX versions;
- power source and mode, thermal state, attached displays, and available memory;
- warmup, repetitions, execution order, and synchronization boundaries;
- correctness result and declared tolerance;
- individual timing samples and derived statistics;
- relevant profiler, compiler-output, and external-artifact checksums;
- any protocol deviation or invalidated sample.

Exploratory runs may identify a dirty state. A run promoted to reference evidence
or used to accept an optimization must correspond to a clean, immutable commit.
Do not record serial numbers, hardware UUIDs, usernames, provisioning
identifiers, or volume UUIDs.

## Durable experiment records

Create the top-level `experiments/` directory with the first recorded
experiment, not before. Use monotonically increasing project-wide identifiers:

```text
experiments/
    README.md
    EXP-0001-rmsnorm-baseline/
        manifest.json
        samples.jsonl
        report.md
```

`experiments/README.md` is the ledger. Each row should give the identifier,
date, subject, hypothesis, status, evidence level, bounded outcome, relevant
commits, and any resulting decision. Generate the ledger from manifests if
manual maintenance later becomes a second source of truth.

Within an experiment:

- `manifest.json` contains the frozen protocol, run provenance, artifact
  references, checksums, and machine-readable status;
- `samples.jsonl` contains raw observations without rounding away the original
  measurements. Each observation identifies its run, implementation, workload,
  iteration or repetition, measured value, unit, and validity;
- `report.md` contains the question, results, interpretation, alternative
  explanations, conclusion, nonclaims, and follow-up.

An experiment may be `planned`, `complete`, `inconclusive`, or `superseded`.
Once decisive runs are complete, preserve their protocol and raw evidence.
Corrections belong in a dated amendment; a new hypothesis or materially changed
protocol receives a new experiment identifier.

Large Xcode traces, compiler dumps, model artifacts, generated binaries, and
other bulky outputs remain outside Git. Record the generating command, relevant
tool versions, byte size, SHA-256 digest, and a non-sensitive provenance note.
Commit compact manifests, raw timing samples, and reports.

## Documentation authority

Avoid copying the same facts across documents:

| Location | Authority |
| --- | --- |
| `docs/project.md` | Project goals, method, roadmap, and evidence gates |
| `docs/model.md` | Model, numerical, artifact, and runtime contracts |
| `docs/layouts.md` | Value, storage, work, and execution-order language |
| `docs/development.md` | Toolchain, reference machine, and development procedures |
| `docs/experiments.md` | Shared experimental protocol and vocabulary |
| `benchmarks/` | Executable measurement instruments and their timing boundaries |
| `experiments/` | Dated protocols, runs, raw evidence, and bounded findings |
| future operation documents | Current synthesis and decisions, linked to experiment identifiers |

Stable contracts should link to experimental evidence rather than embed a
chronological result log. Completed experiments remain as historical evidence
even when superseded. When an operation develops multiple implementations or
nontrivial dispatch decisions, add a canonical operation document that
summarizes the current understanding and cites the relevant experiment
identifiers without duplicating their raw numbers.

## First application

The first recorded experiment should characterize the existing Apple GPU
RMSNorm implementation before changing it. It should establish correctness,
runtime device identity, a row-count workload matrix, synchronized latency and
noise, and representative Metal profiling evidence. Only then should a second
experiment compare one reduction or vectorization change against that frozen
baseline.

This first record is also the test of the documentation format. Keep fields that
support reproduction or interpretation, and remove ceremony that does neither.
