# Experiment ledger

This directory contains dated performance experiments. Reusable measurement
code remains under `benchmarks/`; each experiment directory keeps its frozen
protocol, raw observations, provenance, bounded finding, and resulting decision.

The canonical method and vocabulary are defined in
[`docs/experiments.md`](../docs/experiments.md). A planned experiment has no
result. Treat only a complete experiment's bounded outcome as retained evidence.

| ID | Date | Subject | Question or hypothesis | Status | Evidence level | Bounded outcome | Relevant commits | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [EXP-0001](EXP-0001-rmsnorm-baseline/report.md) | 2026-08-21 | Apple GPU RMSNorm baseline | How does the existing shared-tree implementation scale across row counts, and what limits representative regimes? | In progress | Operation | Profiling method calibrated with counters unresolved; decisive AC timing pending. | Baseline `148dbaa`; instrument `5fdc511`; analyzer `2b23275` | Baseline unchanged |
