# Qwen attention sublayer

The complete attention sublayer now has a validated FP32 attention baseline,
whole-block timings and a twelve-stage profile. The bottleneck changes with
the workload: decode is dominated by softmax and PV; projections dominate
1024-token prefill; QK and PV dominate 4096-token prefill. The next small
experiment should compare output-projection mappings with everything else fixed.

The measured unit is
`X → RMSNorm → Q/K/V → RoPE → KV append → GQA → Wo → residual X+branch`.
It ends before the decoder MLP. Batch is one, H=896, Nq=14, Nkv=2 and D=64.
The attention output `[R,14,64]` is viewed as `[R,896]` for Wo without a copy.
For R new rows and T visible positions, the cache prefix has length T-R.

## Accuracy comes first

The primary reference is pinned Transformers 4.43.1 `Qwen2SdpaAttention` on
Torch 2.4.0 CPU, with explicitly FP32 Q/K/V and mask at the SDPA boundary,
and BF16 attention output before Wo. Weights, activations, cache and final
output remain BF16. This is our agreed inference precision policy; it does
not identify Qwen's original training kernels.

The default Mojo route retains FP32 scores, probabilities and accumulation.
Validation passed 81 Mojo tests and 38 Python tests, all 510 frozen synthetic
arrays, three checkpoint attention cases and repeated asynchronous cache
reuse. The instrument additionally passed full 4096-token ring24 checks
with a poisoned cache suffix. It checks the projected branch separately so
the residual cannot conceal an attention error.

Historical BF16 eager compatibility failures remain reproducible. The
[contract and numerical investigation](../../docs/attention-sublayer.md),
[validation](validation.json), [original numerical record](numerics.json)
and [FP32 comparison](precision_numerics.json) retain the precision decision,
independent gates and provenance. The measured engine and fixtures match the
validation hashes; only the runner's per-process timeout changed afterward.

## Whole-block latency

These are control-arm medians of four block medians, in milliseconds per
sublayer. Both timing arms run the same implementation, providing noise
calibration rather than an optimized comparison.

| Workload | R | T | Hot | Ring24 per call |
| --- | ---: | ---: | ---: | ---: |
| Decode | 1 | 4096 | 2.494 | 2.689 |
| Full prefill | 1024 | 1024 | 39.127 | 39.037 |
| Full prefill | 4096 | 4096 | 355.793 | 355.288 |
| Cached chunk | 64 | 4096 | 9.590 | 9.597 |

![Whole-sublayer latency](latency.png)

The complete 15-shape matrix and calibration ranges are in [summary.csv](summary.csv).
Keep the short cases in perspective: the largest self-pair deviations were
177.3% for hot `(1,64)`, 171.4% for hot `(4,64)` and 54.4% for hot `(16,16)`.
All samples are retained. These points cannot support small optimization
claims under this run's decision rule. Open plot marks identify self-pair
variation above 5%. The four workloads in the table retained the 5% minimum
decision threshold in both modes. Future comparisons need fresh matching
calibration; this run's calibration must not be imported into another run.

## Where the GPU time goes

The following percentages divide a stage's total recorded active duration
by the total active duration in that same capture. They are not percentages
of the separate host-to-completion latency measurement.

| Workload | Main active GPU work | Median stage durations |
| --- | --- | --- |
| Decode `(1,4096)` | Softmax 48.1%, PV 46.5% | 1.110 ms, 1.009 ms |
| Full `(1024,1024)` | All four projections 58.3%; Q 25.6%, Wo 25.4% | Q 9.813 ms, Wo 9.725 ms |
| Full `(4096,4096)` | QK 40.5%, PV 28.6% | 143.409 ms, 101.088 ms |
| Chunk `(64,4096)` | PV 39.9%, QK 31.8%, softmax 14.8% | 3.833 ms, 3.088 ms, 1.429 ms |

![Time spent in each kernel](profile.png)

The [stage table](profile_summary.csv) retains medians and ranges. Each
shape has one separate instrumented capture, with 100, 25, 10 and 25 measured
iterations respectively. Stage names follow verified enqueue order. Instruments
fragmented 54 full-context and 17 chunked dispatches; their execution segments
were joined before assigning stages. Durations exclude preemption and host gaps.
Adding these stage medians would not reconstruct whole-block latency.

The source explains why one optimization will not solve every workload:

- QK assigns one score to a thread; PV assigns one output component to a
  thread. Their serial reduction work grows with the visible key count.
- Softmax assigns an entire row to one thread. Decode has only fourteen
  active softmax threads, each scanning up to 4096 scores. More query rows
  provide more independent softmax work, changing its relative importance.
- Each projection currently uses one SIMD group per output dot product.
  It does not yet use the existing prefill mapping that reuses an input tile
  across several output columns and rows.

Device-wide counter medians reinforce this as a hypothesis rather than a
per-kernel diagnosis: decode's Kernel Occupancy was 1.46%, while full-1024
prefill was 58.61%. The corresponding Last Level Cache Limiter medians were
2.25% and 100%. These samples cover the enclosing target window, include
other GPU activity and are not measured DRAM bandwidth. No target compiler
spill event was reported in these four captures; this is a capture-bounded observation.

Full-4096 optional counter analysis is explicitly unavailable. Its XML export
was stopped after growing beyond 4 GiB, because the existing analyzer loads a
whole table into memory. The original trace and complete stage-timing exports
are preserved externally. No partial counter XML contributes to the report,
and the absent counter summaries are not recorded as zero.

## Work and storage

The number of visible query-key pairs per head is
`C = R(T-R) + R(R+1)/2`. The reference allocates `4*14*R*T` bytes for FP32
scores, overwritten in place by probabilities. QK plus PV performs roughly
`4*64*14*C` floating-point operations; the four projections perform
`2*R*896*(1152+896)`. Softmax, masking and elementwise operations are additional.

| R | T | FP32 scratch | Projection GFLOPs | QK+PV GFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 224 KiB | 0.00367 | 0.01468 |
| 1024 | 1024 | 56 MiB | 3.758 | 1.881 |
| 4096 | 4096 | 896 MiB | 15.032 | 30.072 |
| 64 | 4096 | 14 MiB | 0.235 | 0.932 |

Weights, norm and biases occupy 3,674,112 bytes per ring entry; both KV caches
together occupy `512*T` bytes. These are source-derived counts and allocated
storage, not measured memory traffic. The materialized FP32 path is an
inspectable accuracy baseline with a substantial memory cost.

## Next bounded experiment

Start with the **bias-free output projection**, comparing its current mapping
against the existing 8×16 Apple MMA mapping. It is a single seam, already in
the approved candidate plan, and Wo accounts for 25.4% of full-1024 and 11.7%
of full-4096 active GPU time. Those shares motivate a trial; they do not predict
its whole-block speedup. Keep the FP32 attention computation, other projections,
RoPE, cache and residual fixed, including BF16 rounding before residual addition.

Use the existing independent operation/composition gates and poisoned-buffer
instrument checks, then the approved five-shape screen. Advance a configuration
only if the paired evidence warrants the full matrix. No optimization candidate
has been screened in this baseline record and no dispatch rule is promoted.

Packing QKV, direct RoPE/cache writes and projection/residual fusion remain
possible later experiments. The latter two address small active-time shares
in these captures; very short calls could behave differently and have noisy
calibration here. Separately, FP32-preserving GQA parallelism and tiling are
larger opportunities for decode and long prefill. The older optimized BF16
routes cannot establish a speedup under the selected FP32 policy without
their own accuracy validation.

## Reproduction and evidence

The latency run used clean source `8d8c8540c5f9de2d7d7cf16fc00f502702a3a941`
on 2026-09-06, 16:22:57–16:53:05 UTC. All four profile binaries use that same
source. Hardware was Apple M4 Pro / Metal, Mac16,7 with 24 GiB memory;
Mojo 1.0.0, MAX 26.5.0, macOS 26.6.2 and Xcode 26.6. Every latency block and
capture recorded AC power, Low Power Mode off and no thermal/performance warning.
These checks do not pin GPU clocks or exclude background activity.

The workload uses frozen synthetic seed 53 with a CPU-derived cache prefix.
Ring24 owns 24 distinct weights, inputs and caches with two sign patterns;
it shares scratch/output and is not a decoder stack. Timing includes the host
length rewind, twelve enqueues, cache append and completion. Allocation,
fixture reads, uploads, prefix setup and correctness checks are excluded.
Each call overwrites the same suffix, keeping R and T fixed. There are ten
warmups and ten samples per arm in four blocks; blocks two and three reverse
workload and arm order.

[run.json](run.json) and [samples.csv.gz](samples.csv.gz) retain all 2,400
latency observations and their provenance. [profiles.json](profiles.json)
and [profile_samples.csv.gz](profile_samples.csv.gz) retain all 1,920 measured
dispatch durations, capture identities, selected counters and the explicit
counter-analysis omission. The source of the curator is hashed in that record.
The post-measurement curation/plot changes handle absent optional counters and
mark elevated calibration variation; they change no measured engine code.
All 38 Python checks pass with the retained evidence, including duplicate and
missing-dispatch rejection for the new profile schema. Both tables and both
figures regenerate byte-for-byte from the files in this directory.

Rebuild tables and figures without a GPU:

```sh
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot studies/attention_sublayer
```

For fresh measurements, use a clean checkout, regenerate and validate fixtures,
then use the [package-owned build/run and trace commands](../../src/llm_mojo/benchmarks/README.md).
Use the recorded commit to reproduce the exact measured source. Raw traces,
XML, binaries, checkpoint assets and oracle arrays stay outside Git.
