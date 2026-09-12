# Exact-width decode projections and thread-block arrangement

Baseline is current single-row M4 Pro Fast: configuration 26 plus residual/RMSNorm,
owner swapping and separate GPU argmax. Test the same 121 rowwise projections
per token (packed QKV, attention output, three MLP projections per layer, head).
Keep one 32-thread SIMD group per output and sequential FP32 accumulation,
warp reduction, BF16 bias addition/output boundary, allocations and other kernels.

Six arms: 0 original runtime-width/128 threads; 1 fixed width with four-iteration
prefetch/unroll/128; 2 runtime width/64; 3 runtime width/256; 4 fixed width/64;
5 fixed width/256. Fixed widths are 896 and 4864. Specialization loads four
lane-strided input/weight operands before four sequential accumulator updates.
It does not introduce vector reduction or partial accumulators. Dynamic 64/256
arms retain the original loop. No repacking, shared-memory staging, quantization,
cooperative reduction or gate/up epilogue fusion. Prefill is unchanged.

Validate 60 primitive cases per arm with exact BF16 comparison, bias/no bias,
signs, subnormals, rounding-adjacent values, nonuniform data, ragged output blocks,
protected output guards and unchanged inputs/weights/bias. Invalid variants,
unsupported widths and multi-row candidate requests must reject before mutation.
Compare full logits and 48 complete caches for every candidate at prefixes
64/1024/3968; compare 195 layer/normalization/storage tensors at prefix64.
Stream three fixed 128-token replies with resets for all arms, four balanced
blocks (72 replies). All text/tokens/history must match. Use existing validation
and model profiling tools; retain all checks and samples.

Screen six paired comparisons per context/block: control self-pair and each
candidate versus control. Four blocks, three contexts, ten warmups and ten
samples per arm: 1440 samples. Reverse workload and arm order in blocks 2/3.
Use one resident model executable, runtime candidate selection; allocation,
preparation, logical rewind and recording excluded. Time token upload through
greedy completion. Require verified M4 Pro/Metal, source/binary/assets, AC,
normal power and nominal thermal state. No tuning after acceptance measurements.

A screen qualifier must have all four ratios below 1 and median reduction above
max(5%, largest absolute self-pair deviation), at every context. If any qualify,
choose the lowest worst-context median ratio, then mean ratio, then variant ID.
Confirm only that fixed choice against control in an independent four-block
three-context paired/self-paired run (480 samples), with the same gate. No
fallback candidate or repeated screen after confirmation failure. Promotion to
Fast/auto requires confirmation plus public-path and lifecycle validation.
Otherwise retain current Fast and the documented negative/inconclusive result.

Capture all six history1024 traces separately, ten warmups/eight steps; expect
245 compute and four mapping blits per step. Verify 121 projection commands
use their declared arrangement through the compiled source and runtime selector.
Active GPU intervals explain mechanism but do not establish exclusive CPU/GPU,
occupancy or DRAM causes without counters. Retain lossless timing, numerical,
terminal and command records with provenance; full traces/binaries stay outside Git.
