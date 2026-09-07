# Qwen attention sublayer

The attention block is integrated and tested through residual addition. Packed
QKV, FP32 GQA, and Wo run through one public Mojo enqueue. The latest experiment
also composes split8 GQA with both 16×16 projection tiles. These remain explicit
mapping choices: short chunks can regress, so there is no automatic selector.

`X → RMSNorm → packed QKV → unpack → Q/K RoPE → KV append → GQA → Wo → X + branch`

This is one attention block, ending before the decoder MLP. Batch is one;
H=896, Nq=14, Nkv=2, D=64. For R new rows and T visible positions, the existing
cache prefix has T−R rows. Ring24 repeats the block with distinct buffers and two
sign patterns; it is not a 24-layer decoder benchmark.

## What we learned

All percentages below are reductions in paired, unprofiled whole-block latency
on Apple M4 Pro / Metal. Hot and ring24 results remain separate in the full tables.
Each row names its own control; gains from different rows must not be multiplied.

| Comparison | Result | Limit / next implication |
| --- | --- | --- |
| Integrated block vs original materialized FP32 attention and rowwise projections | About 88% at full 1024, 91% at full 4096, 80% at R=64/T=4096 | Integration mattered as much as isolated kernels. |
| Both 16×16 projections vs integrated 8×16 projections, unsplit GQA | 16.5–18.8% at full 256; 15.4–16.1% at full 1024; about 9.1% at full 4096 | R=16/T=256 regressed 13.91% in ring24. |
| Both 16×16 projections vs 8×16 projections, split8 GQA fixed | 7.5–9.5% at R=64/T=1024; 11.3–11.8% at R=256/T=1024 | Some 16-row chunks regress; at T=4096 several gains are inconclusive. |
| Split8 vs unsplit GQA, both 16×16 projections fixed | All 14 tested shape/mode cells qualified | Bounded cached-chunk domain; no general full-prefill claim. |

The combined-projection profiles put unsplit GQA at roughly 71% of active GPU
time at full 4096 and 89% at R=64/T=4096. At full 1024 its share is comparable
to the two projections. Profiles explain where work remains; stage medians do
not construct whole-block latency. The latest split8 composition has paired
timing evidence but no new stage captures. This study establishes neither a
hardware ceiling nor full-model performance.

## Correctness and ownership

The reference is Transformers 4.43.1 `Qwen2SdpaAttention` on Torch 2.4.0 CPU,
with explicitly FP32 SDPA inputs and BF16 output before Wo. This is the selected
inference precision policy, not a claim about Qwen's original training kernels.
Weights, activations, cache and output are BF16; GQA scores, probabilities and
accumulation remain FP32. Gates and exact cache checks are unchanged.

Weights, cache and workspace belong to the caller. Enqueue performs no allocation
or synchronization. The [contract](../../docs/attention-sublayer.md) explains
rounding boundaries, tested mappings and fixture generation.

## Reading and reproducing the study

- [Experiment results](experiments.md): complete comparisons, figures, regressions and provenance.
- [Numerical investigation](numerics.md): reference choice, historical failures, holdouts and checkpoint reproduction.
- [Predeclared plans](plans.md): original comparison budgets and stop conditions.
- [Data](data/): compact raw samples, numerical records, provenance and regenerated tables.
- [Figures](figures/): generated views of the retained evidence.

Run `uv run --locked llm-mojo-validate` for correctness. Regenerate the tables
and figures with `uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot`.
See the [benchmark commands](../../src/llm_mojo/benchmarks/README.md) before
collecting new measurements; historical records keep their measured source IDs.

Four large numerical records use adjacent `.json.gz` files. Their small `.json`
summaries hash both the compressed file and the original JSON bytes and list
record counts. `load_numerical_record()` verifies and reads them; `gzip -dc`
also recovers the original JSON verbatim. All unsuccessful checks are retained.
