# Projection scheduling diagnosis

Question: why did fixed-width projection kernels reduce active GPU time, but
show little complete-token improvement at short context and more at long context?
Compare only original runtime/128 (variant 0) and fixed-width/128 (variant 1)
over already promoted all-three Fast. No kernel or production-path edits, tuning,
default promotion, or repetition until a preferred result.

Hypotheses: (1) CPU submission limits short-context completion, so reduced GPU
execution becomes uncovered time between target commands; longer attention changes
that balance. (2) repeated-position benchmarking or transient conditions contribute
to the pattern. Cache effects remain an alternative, not an assumed mechanism.

Use existing model runner with four modes: fixed position, advancing greedy
sequence, and versions of both with the existing ten host observation clocks.
Each arm warms up 16 times at the starting position, rewinds logically once, then
records 64 tokens. Fixed mode rewinds each token; advancing mode feeds the selected
token into the next step. Histories start at 64/1024/3968; advancing ends at
128/1088/4032 cached tokens. One resident model per comparison; both arms start
from the same prefix. IDs, timing buffers and output recording are preallocated;
string formatting occurs after each arm. Preparation, rewind, result checks and
recording are untimed. Token upload through greedy completion remains timed.

Four balanced blocks reverse workload and arm order in blocks 2/3. Each mode and
context has control/self and candidate/control pairs: 4*4*3*2*2*64 = 12288 samples.
Keep all samples and conditions. Compare block medians, first/last 16 samples,
self-pair deviations, fixed versus advancing, and observed versus plain. Report
absolute milliseconds as well as ratios. Apply the existing descriptive 5%/noise
and all-four-block rule, but this diagnosis cannot itself promote a default.
The observed modes diagnose host forward submission and greedy readback wait;
these are wall intervals with overlapping GPU execution, not exclusive CPU time.

Correctness: frozen original/fixed kernel tests remain valid; rerun the existing
native projection test and Python suite. At each context and mode retain complete
logits/KV byte comparisons after 64 samples, with finite/prefix/inactive checks.
All 64 winner IDs must match between arms in every pair, and layer/model cache
lengths must match the declared progression. No stop at EOS in bounded diagnosis.

Capture two balanced passes of the six fixed-position context/variant traces,
10 warmups and 8 measured tokens each. Use observed host clocks and retain their
raw output bound to the capture receipt. Expect 245 compute + 4 blits per token,
23904 retained commands and 96 host observation records. Record active interval
unions, per-stage active durations, target compute span, uncovered time, submission
span and preemption fragments. Uncovered time is not necessarily globally idle;
other GPU work and profiler perturbation remain possible. Compare traced host
observations with untraced observed modes before drawing a mechanism conclusion.

Freeze clean source and binary/assets identity before collecting. Require M4 Pro /
Metal, BF16/FP32 boundaries, AC, normal power and nominal thermal state. Retain
compact raw samples, full numerical records, command fragments and provenance in
one lossless archive with replay checks and a chart; full traces/binaries stay
outside Git. No kernel tuning or extra cache experiment is included in this
bounded first diagnosis; if the evidence cannot distinguish causes, say so.
