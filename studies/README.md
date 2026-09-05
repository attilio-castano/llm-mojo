# Inference on this MacBook Pro

These studies explain how the existing Mojo operations map work onto the Apple
M4 Pro. Start with the value and storage contracts in [model](../docs/model.md)
and [layouts](../docs/layouts.md), then read a topic below. Each comparison is
operation-level; a working decoder block and full-model generation remain next.

| Topic | Question |
| --- | --- |
| [RMSNorm](rms_norm/README.md) | Where should a row's sum of squares be reduced? |
| [Linear decode](linear_decode/README.md) | What do packing QKV and reusing input across outputs buy? |
| [Linear prefill](linear_prefill/README.md) | How should more token rows change tile and lane ownership? |
| [RoPE](rope/README.md) | How much work and data movement does rotating a dimension pair require? |
| [GQA decode](gqa_decode/README.md) | How do fusion, sequence parallelism and shared KV heads interact? |

Fresh characterization is complete at measured source `1267a7a`: **10,560
latency observations**, five report figures, and three focused GQA profiles.
The full fixture migration passed 68 Mojo tests. The 29 shared tooling, trace and evidence checks, all retained operation benchmark routes and all thirteen GQA routes
were validated separately. No engine code, numerical tolerances or dependency
lock changed during the reset.

The results are deliberately mixed: RMSNorm remains inconclusive in this run;
packed QKV improves the ring sweep; prefill benefits from register/MMA reuse at
larger sampled row counts; RoPE has baseline characterization; and GQA's fusion
and parallelism gains remain clear. Read each study's limits alongside its plot.

The matrix is deliberately bounded: existing implementations, selected sizes,
hot and ring24 timing, four blocks, and self-pair calibration. We are refreshing
characterization, not searching a new tuning space. See the
[measurement command](../benchmarks/README.md) and [decision rule](../docs/experiments.md).

Each finished study keeps `run.json`, all observations in `samples.csv.gz`, a
small `summary.csv`, and one report image. Rebuild every figure without a GPU:

```bash
uv run --no-project --with matplotlib==3.10.8 python benchmarks/plot.py
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
