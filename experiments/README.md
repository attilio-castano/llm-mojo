# Experiment ledger

This directory contains dated performance experiments. Reusable measurement
code remains under `benchmarks/`; each experiment directory keeps its frozen
protocol, raw observations, provenance, bounded finding, and resulting decision.

The canonical method and vocabulary are defined in
[`docs/experiments.md`](../docs/experiments.md). A planned experiment has no
result. Treat only a complete experiment's bounded outcome as retained evidence.

| ID | Date | Subject | Question or hypothesis | Status | Evidence level | Bounded outcome | Relevant commits | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [EXP-0001](EXP-0001-rmsnorm-baseline/report.md) | 2026-08-21 | Apple GPU RMSNorm baseline | How does the existing shared-tree implementation scale across row counts, and what limits representative regimes? | Complete | Operation | Dispatch-floor scaling transitions at rows=512; saturation was not observed through rows=4096; Metal counters remain unavailable. | Baseline `148dbaa`; timing `b59717d`; analyzer `6b7d4ac` | Characterization complete; baseline unchanged |
| [EXP-0002](EXP-0002-rmsnorm-simdgroup-reduction/report.md) | 2026-08-22 | Apple GPU RMSNorm SIMD-group reduction | Does one hybrid SIMD-group reduction materially improve a real decode or prefill regime without a relevant regression? | Complete | Operation | Rows=512 improved 5.38% in 4/4 paired blocks; no relevant material regression; post-timing xctrace failed before target launch. | Protocol `d062148`; variant `e947397`; timing `476fc29`; promotion `9bf73f6` | SIMD-group default; shared tree retained only as benchmark baseline; no dispatch rule |
| [EXP-0003](EXP-0003-rmsnorm-limiter-counters/report.md) | 2026-08-22 | Apple GPU RMSNorm limiter counters | Do matched Performance Limiters traces show a repeatable device-counter difference between the shared-tree and SIMD-group implementations at rows=512? | Planned | Operation | Pending matched counter captures. | Profiler gate `de6aca0` | No production decision; diagnostic mechanism follow-up only |
