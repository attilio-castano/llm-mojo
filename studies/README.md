# Inference on this MacBook Pro

Start with the working [terminal chat](../docs/chat.md), then follow the evidence
from a complete model down to its kernels. These studies explain native Mojo
inference on Apple M4 Pro / Metal through numerical observations, retained raw
measurements and explicit timing boundaries. The [model contract](../docs/model.md)
and [layouts](../docs/layouts.md) define the values, storage and ownership.

## Completed Fast runtime and chat

| Study | What it establishes |
| --- | --- |
| [Residual/RMSNorm alone and composed](model_generation/residual-norm.md) | Promoted all-three Fast route: 17.1–24.5% lower latency, 107–115 tokens/s; exact independent and composed checks |
| [Inter-layer buffer swapping](model_generation/buffer-swap.md) | Exact outputs with 23 fewer compute commands; 5.3–8.3% median paired reductions and 90–94 tokens/s streaming, but the full promotion gate fails |
| [GPU token selection](model_generation/token-selection.md) | Standalone argmax gains 4.5–7% but misses the full promotion gate; the tested fused head is slower; exact outputs and complete retained evidence |
| [Combined QKV and activation fusion](model_generation/combined-fusion.md) | Promoted M4 Pro single-token Fast route: 16.9–19.4% lower latency, 86–90 tokens/s streaming medians, exact outputs and cache storage |
| [QKV fusion experiment](model_generation/qkv-fusion.md) | Exact candidate with 12–15% measured token-latency reductions; kept experimental because short-context calibration prevented promotion |
| [Complete token profile](model_generation/token-profile.md) | Where a roughly 17 ms decode step spends time, with host observations and complete Metal command traces |
| [Native terminal chat](model_generation/chat.md) | Multi-turn cache reuse, exact history and terminal lifecycle, with paired prefill and actual interaction timings |
| [Fast full-model runtime](model_generation/runtime.md) | All 24 layers, native generation, numerical diagnostics and 11 measured dispatch choices |
| [Decoder selection](decoder_layer/selection.md) | Which combinations improve complete prefill and decode workloads |

The runtime measurements are against our optimized baseline; chat measurements
separately compare suffix-only prefill with full-history replay. Neither is an
HF speed comparison. Full-model differences are diagnostic, while exact cache
and lifecycle invariants remain required. [Numerical investigations](model_generation/README.md)
retain the earlier failed qualification policies. Schedule determinism is a
follow-up, supported by the existing decoder policy and full-model studies.

## Supporting operation and composition studies

| Topic | Question |
| --- | --- |
| [RMSNorm](rms_norm/README.md) | Where should a row's sum of squares be reduced? |
| [Linear decode](linear_decode/README.md) | What do packing QKV and reusing input across outputs buy? |
| [Linear prefill](linear_prefill/README.md) | How should more token rows change tile and lane ownership? |
| [RoPE](rope/README.md) | How much work and data movement does rotating a dimension pair require? |
| [GQA decode](gqa_decode/README.md) | How do fusion, sequence parallelism and shared KV heads interact? |
| [GQA prefill](gqa_prefill/README.md) | How do query tiling, online softmax and Apple matrix instructions interact? |
| [MLP sublayer](mlp_sublayer/README.md) | How do tiled projections change complete SwiGLU latency under its frozen BF16 rounding contract? |
| [Attention sublayer](attention_sublayer/README.md) | Where does time go in the complete block under the selected FP32 attention policy? |
| [Decoder layer](decoder_layer/README.md) | How do complete layer costs shift between prefill, cached chunks and decode? |
| [Decoder policies](decoder_layer/policies.md) | How much does schedule-invariant execution cost, and which optimizations preserve it? |
| [CPU tokenizer](tokenizer/README.md) | When does heap BPE improve complete text encoding? |

The CPU tokenizer study establishes exact Rust parity in Mojo and retains 9,680
CPU observations. Heap merging improves long single pieces but costs more than
scanning the short pieces in the ordinary text workloads. Its report has a
separate regeneration command and keeps this result distinct from GPU inference.

The decoder study composes both residual branches without an added copy or
synchronization. Its seven reserved cases pass, and it retains 960 baseline
latency observations and 2,400 measured dispatches. MLP contributes 79% of
active time at full R=T=256; attention contributes 67% for R=64,T=4096. Decode
noise and diagnostic gaps remain explicit. The follow-up
[configuration study](decoder_layer/selection.md) retains 16,800 additional
latency observations and confirms 5.6–52.9% lower decoder latency on the primary
cached-prefill grid. Full/short prefill and decode retain the existing baseline.
The subsequent [policy campaign](decoder_layer/policies.md) separates Fast and
Deterministic execution. It retains 17,280 timing observations, 4,975 profiled
dispatches and 9,583,121 core numerical checks. Four-row reuse lowers deterministic
prefill latency by 24–37%; the selected deterministic prefill policy still takes
2.2–6.0 times Fast latency. Decode comparisons remain inconclusive. Both final
deterministic lookup settings pass all fresh schedule/cache comparisons.
The [full-model consistency investigation](model_generation/consistency.md)
qualified a canonical HF reference but stopped at native numerical gates.
Those results remain distinct from the completed Fast diagnostic milestone;
native full-model schedule invariance has not been established.

The attention-sublayer study uses the explicit CPU FP32 attention policy as
its accuracy baseline. It has 17 synthetic and three checkpoint cases; the
earlier BF16 eager holdout remains a recorded compatibility failure, with an
explicit command to reproduce its original strict gate. The composed study
retains the baseline, Wo, FP32 decode and FP32 prefill comparisons, and integrates
the existing packed QKV and Wo mappings through one public Mojo entrypoint.
It separates incremental QKV value from the whole block's combined gain.
Small hot-call noise and omitted optional counter analysis remain explicit.

The MLP study retains its materialized baseline and a completed projection
campaign under the same BF16 rounding rules. All eight configurations pass
53 regression cases; original and final also pass seven fresh holdouts. The
direct 3,200-observation comparison gives 10.7x/11.2x hot speedups at
1,024/4,096 rows for 16x16 gate/up/down. One-row ring24 is 2.23x slower and
one-row hot is inconclusive, so selection remains explicit and rowwise stays
the default. Eight final profiles retain 8,890 measured dispatches; projections
still occupy about 91% of large-row active GPU time. The complete optimization
campaign retains 12,160 latency observations, including screens and increments.
The measurements below belong to the six operation studies.

The six topics retain **31,200 latency observations**, eleven report figures,
and fifteen focused GQA profiles containing 10,200 measured dispatch durations.
Their recorded validation passed 71 Mojo tests and 36 Python tooling/evidence checks,
including every prefill measurement route in hot and ring24 modes.

The original five topics characterize existing implementations at source
`1267a7a`. GQA prefill adds a bounded optimization screen at `fe418cc` and a
direct comparison of finalists at corrected source `bf4277c`. That study also
found a trace-analysis defect: Instruments may split a preempted dispatch into
several active intervals. All nine original GQA captures were reanalyzed with
the segment-joining fix; the separate latency observations are unchanged.

The results are deliberately mixed: RMSNorm remains inconclusive; packed QKV
improves the ring sweep; linear prefill benefits from register/MMA reuse at
larger sampled row counts; and RoPE has baseline characterization. GQA decode
benefits from fusion and parallelism. GQA prefill's 32x32 MMA path is about
11.2× faster than materialized attention at full 4,096-token prefill, while
sharing four query heads demonstrates no gain over that optimized control.
The resource follow-up finds six confirmed gains from rolled QK, sixteen
inconclusive comparisons and no demonstrated regressions. It also reduces
the maximum compiler-reported spill size per event from 144 to 48 bytes in
all three diagnostic workloads.
Read each study's limits alongside its plot.

The matrices use selected sizes, hot and ring24 timing, four paired blocks,
and matching self-pair calibration. See the
[measurement command](../src/llm_mojo/benchmarks/README.md) and [decision rule](../docs/experiments.md).

Each finished study keeps `run.json`, all observations in `samples.csv.gz`, a
small `summary.csv`, and the PNGs used in its explanation. GQA prefill retains
its original screen with a `screen_` filename prefix and the bounded resource
follow-up with `resources_screen_` and `resources_` prefixes in the same folder. GQA
decode and prefill also retain compact profile records and dispatch samples.
The larger [attention topic](attention_sublayer/README.md) separates its current
overview, detailed experiments, numerical history and predeclared plans. It
keeps records/CSVs in `data/` and generated PNGs in `figures/`.
Rebuild the operation and attention tables and figures without a GPU:

```bash
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot
```

The project reset starts from merged PR #11 at
[`a86f4db`](https://github.com/attilio-castano/llm-mojo/tree/a86f4dbadb4b8c9255aacb004ad30fdfefeaa8fd).
All fourteen original experiment folders, their reports and committed artifacts
remain available in that revision's
[experiments directory](https://github.com/attilio-castano/llm-mojo/tree/a86f4dbadb4b8c9255aacb004ad30fdfefeaa8fd/experiments).
No history is rewritten. The current checkout retires their packaging and
campaign-specific runners. Several older raw-sample paths were temporary and
are now absent at those locations; Git preserves only what was committed.
The original GQA campaign did commit its individual samples.

The reset preserves all engine implementations and all numerical cases. Large
oracle arrays are regenerated and checked against their landed hashes. Fresh
results identify their own source commit and conditions; old crossover claims
are not silently carried forward. The composed decoder and its configuration
study now extend those results.
The completed model and chat studies extend this work to the usable engine.
Replay their retained evidence and regenerate their tables without weights or a GPU:

```sh
uv run --locked python studies/model_generation/summarize.py
```
