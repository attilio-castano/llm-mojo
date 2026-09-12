# Runtime enqueue diagnosis

Investigate whether the approximately 6.7 ms model submission interval is spent
inside the pinned MAX enqueue runtime or outside it. Test explicit compiled
kernel handles as the smallest application-level acceleration. These are
measurement tools only: no engine/default/kernel changes and no promotion from
an isolated launch benchmark.

Use the existing model benchmark with `MODEL_LAUNCH_PROBE`. Record the absolute
start already read for each token; no extra clock inside the measured token.
The existing host marks delimit upload completion to head/argmax submission.
A process-local DYLD interposer brackets the exported
`AsyncRT_DeviceContext_enqueueFunctionDirect` call with CLOCK_UPTIME_RAW, the
same clock as Mojo. It forwards every argument/result unchanged, preallocates
and pretouches storage, and writes only at process exit. Verify its ABI against
the pinned MAX 26.5.0 macOS arm64 binary. Do not modify installed libraries.
Expect exactly 245 nonoverlapping runtime calls on one thread per token window;
reject missing/dropped/error calls or straddling intervals. These are wall times,
including any runtime blocking; they are not exclusive CPU execution times.

Freeze clean source, C source separately (the shared source receipt excludes C),
clang version, runtime dylib hash, model/probe binary hashes, prepared assets,
Mojo/MAX versions and hardware. Require Apple M4 Pro / Metal, AC, normal power,
and nominal thermal state. All GPU measurements run serially.

Full model: original runtime-width/128 and fixed-width/128 projections, with
promoted residual norm, buffer swapping and separate GPU argmax. BF16 weights
and activations, original FP32 reductions; one row, width 896, MLP width 4864,
151936 logits, 24 layers, KV capacity 4096. Prefixes 64/1024/3968, fixed position,
16 warmups and 64 retained tokens per arm. Three wrapper states: absent, loaded
but inactive, and recording. Four blocks reverse workload/arm order in blocks
2 and 3. Total 4*3*3*2*64 = 4608 token samples, 376320 measured runtime calls.
Keep every sample. Compare within-block medians across wrapper states to expose
perturbation; aggregate block medians equally. At block 1 retain full logits and
48 KV array comparisons for all nine context/state pairs, finite/prefix/inactive
checks, cache accounting and identical token trajectories across all runs.

Launch microbenchmark: original projection kernel, tiny [1,32] x [1,32] weights
and representative down projection [1,4864] x [896,4864] weights. All inputs and
weights are one, so exact BF16 outputs 32/4864 plus two untouched guards are an
independent integer oracle. No bias, one 32-thread group per output, 128 threads
per block. Both paths use the same precreated views and kernel. Compile a handle
once before either arm, then compare explicit handle reuse to the normal generic
enqueue overload (which already caches compilation). Batches 1 and 256, synchronize
after each batch, ten warmup and ten measured batches per arm. Four blocks,
self-pairs and candidate pairs: 640 retained batch samples. Report per-launch
submission and batch completion, keeping synchronization amortization explicit.
A descriptive win requires all four block ratios below one and median improvement
above max(5%, largest self-pair deviation). This cannot promote a model default.

Queue diagnostic: two passes, both shapes, batch 256, original/original arms,
recording enabled. 80 batch samples and 20480 measured calls. Compare first/last
16 call durations to test whether heavier GPU work makes later enqueues wait.
Do not infer an exact queue limit or backend implementation from this alone.

Retain complete stdout, all measured call intervals/geometries, raw probe file
hash/count, conditions, numerical receipts and provenance in one lossless gzip
archive. Prefill and warmup calls are outside the declared retained windows;
full original probe logs remain outside Git. Replay the entire census, boundary
partition, numerical and trajectory invariants, hashes and summary. Add corruption
and boundary tests; run the full Python suite and native route smoke before the
implementation commit. Engine and existing numerical contracts are unchanged,
so follow docs/development.md's tooling-only validation rule. No Instruments
capture is needed to answer this boundary question. State limitations and next
backend-level options if compiled handles cannot close the gap.
