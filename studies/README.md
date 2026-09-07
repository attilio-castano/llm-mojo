# Inference on this MacBook Pro

These studies explain how the existing Mojo operations map work onto the Apple
M4 Pro. Start with the value and storage contracts in [model](../docs/model.md)
and [layouts](../docs/layouts.md), then read a topic below. Each comparison is
operation-level except for the composed attention and MLP sublayers; a working decoder
block and full-model generation remain next.

| Topic | Question |
| --- | --- |
| [RMSNorm](rms_norm/README.md) | Where should a row's sum of squares be reduced? |
| [Linear decode](linear_decode/README.md) | What do packing QKV and reusing input across outputs buy? |
| [Linear prefill](linear_prefill/README.md) | How should more token rows change tile and lane ownership? |
| [RoPE](rope/README.md) | How much work and data movement does rotating a dimension pair require? |
| [GQA decode](gqa_decode/README.md) | How do fusion, sequence parallelism and shared KV heads interact? |
| [GQA prefill](gqa_prefill/README.md) | How do query tiling, online softmax and Apple matrix instructions interact? |
| [MLP sublayer](mlp_sublayer/README.md) | Where does the materialized SwiGLU block spend time, under its frozen BF16 rounding contract? |
| [Attention sublayer](attention_sublayer/README.md) | Where does time go in the complete block under the selected FP32 attention policy? |

The attention-sublayer study uses the explicit CPU FP32 attention policy as
its accuracy baseline. It has 17 synthetic and three checkpoint cases; the
earlier BF16 eager holdout remains a recorded compatibility failure, with an
explicit command to reproduce its original strict gate. The composed study
retains the baseline, Wo, FP32 decode and FP32 prefill comparisons, and integrates
the existing packed QKV and Wo mappings through one public Mojo entrypoint.
It separates incremental QKV value from the whole block's combined gain.
Small hot-call noise and omitted optional counter analysis remain explicit.
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
Rebuild every table and figure without a GPU:

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
are not silently carried forward. After these bounded studies, the next engine
milestone is composing and verifying one complete decoder block.
