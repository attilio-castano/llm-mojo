# Experimental method

A study should let a reader answer: what does the operation compute, which
thread owns the work, what changed, what was measured, and what remains unknown?
Keep the evidence needed to answer those questions. Git preserves the research
history; the current checkout presents a curated explanation of each topic.

## The optimization loop

1. Write the contract: dtype, dimensions, strides, rounding, aliasing, ownership,
   and the caller's responsibilities. Establish an independent numerical oracle.
2. State a mechanism and a bounded comparison. For example, reuse one K/V load
   across four related query heads, accepting more live accumulator state.
3. Validate the implementation and the actual benchmark route on Metal. Include
   nonuniform data, ragged boundaries, and overwritten output/workspace checks.
4. Freeze a clean source commit, compile once, verify the binary, and measure
   the control against itself as well as the proposed alternatives.
5. Benchmark and profile separately. Interpret the whole operation's latency;
   use traces to test a specific explanation, not to manufacture a speed claim.
6. Record the result, including regressions and uncertainty, then stop at the
   comparison budget. Promote a dispatch rule only with evidence for its domain.

Correctness is a gate, not a speed measurement. Operation correctness does not
prove decoder-block composition: projection, RoPE positions, KV-cache mutation,
head merge, residual connections and the MLP introduce separate failure modes.

## Current measurement contract

The explicit matrix is in `benchmarks/study.py`. Every study uses the same
four-block paired procedure; blocks 2 and 3 reverse workload and arm order.
Each arm has ten warmups followed by ten samples. We keep all observations,
including slow ones, and use each block's median rather than treating every
sample as an independent experiment.

The candidate/control ratio is computed separately within each block. A gain
requires all four ratios below one and a median reduction exceeding both 5%
and the largest absolute deviation of the control self-pair ratios from one.
A regression uses the symmetric rule. Everything else is inconclusive. These
are conservative decision rules, not statistical confidence intervals. The
plot shading shows the range of block ratios, not a confidence band.

Calibration lives in the same sample table and run as the comparisons. The
loader requires a complete grid, including calibration, and verifies the raw
file hash. Calibration cannot be imported from an unrelated binary or session.
The frozen specification in `run.json` allows future code to change the next
matrix without reinterpreting an existing run.

AC power and Low Power Mode off reduce avoidable variation. Before and after
each block, record power, thermal warnings, memory and displays. Record the
hardware, software, source commit, source hashes and binary hash. Prove the
runtime device and Metal backend; tooling availability alone proves nothing
about where a kernel executed. These checks do not fix GPU clocks or exclude
all background activity. Do not silently discard a noisy run or repeat until a
preferred outcome appears.

## Boundaries and physical claims

Hot latency is host enqueue through device completion for one operation.
Ring24 latency is a sweep over 24 distinct input/weight buffers, synchronized
once and divided by 24. Allocation, compilation, initialization and numerical
checks are outside timing. Output/scratch reuse is explicit. Ring24 changes
both reuse distance and synchronization amortization; it is not guaranteed
cold DRAM and is not a complete 24-layer model.

Keep these quantities separate: allocated footprint, source-requested loads,
cache transactions and measured DRAM traffic. A byte count divided by time is
not measured memory bandwidth unless the counter and boundary support it.
Similarly, source accumulator count is not a physical register allocation.
Name the counter, unit, aggregation, target interval and capture conditions
when using profiler evidence. Counter absence is not zero; device-wide
measurements cannot automatically be assigned to one kernel. Isolated stage
times need not sum to end-to-end time.

## What belongs in Git

Each topic has an explanation, a compact `run.json`, `samples.csv.gz`, derived
`summary.csv`, and a PNG used by its report. One plotting command reconstructs
the summary and image from retained samples without GPU execution. Add a
compact profile table only when it substantiates a report's explanation.
Avoid repeated metadata per sample, hash lists of disposable logs, duplicate
image formats, and a new runner or directory for each parameter choice.

Generated oracle arrays belong in `build/`; their independent generators,
pinned dependencies and frozen data hashes belong in tests. Model weights,
compiled binaries, full traces and temporary logs remain outside Git.

The historical campaign is preserved at merged revision `a86f4db`; see
[the study index](../studies/README.md). Old numerical results remain historical
observations. Fresh runs use the current protocol and can weaken or change a
previous conclusion. No compatibility layer is required for retired file formats.
