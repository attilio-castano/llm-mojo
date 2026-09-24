# Decode experiment summaries

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives.

Moved from the [model contract](../model.md) on 2026-09-23. Each summary links its study.

### GPU selection experiment

The [token selection study](../../studies/model_generation/token-selection.md)
compares configuration 26 with CPU greedy, a separate GPU argmax, and a fused
vocabulary projection/local argmax. Both GPU routes preserve rounded BF16
scores, lowest-ID ties and rejection of any nonfinite score. The fused route
leaves logits untouched except during explicit diagnostic materialization.
Neither candidate passed that standalone promotion rule. The later
[composed study](../../studies/model_generation/residual-norm.md) promotes the
separate GPU argmax together with residual/RMSNorm fusion and buffer swapping.
The `gpu-argmax` and `fused-head` study policies, and the fused head itself,
exist through `edb610a`.

### Inter-layer buffer ownership experiment

The [buffer-swap study](../../studies/model_generation/buffer-swap.md) exchanges the
input and MLP-output DeviceBuffer owners between layers, removing 23 compute
copies while retaining both allocations and the decoder's disjoint input/output
contract. Each next layer rebuilds its input view from the current owner. The
last layer does not swap, so final normalization still reads `mlp.output`.
Exact hidden-state, cache, lifecycle and streaming comparisons pass. Median
paired reductions are 5.3–8.3%, but that standalone promotion gate fails. The
later composed study qualifies and promotes swapping as part of all three.
Fast swaps only on single-row Apple M4 Pro calls; the `buffer-swap` study policy
exists through `edb610a`. Multi-row calls retain the copy path, including after
a swapping call; a plan cannot request multi-row swapping.


### Residual addition and normalization

The [independent and composed study](../../studies/model_generation/residual-norm.md)
fuses 48 residual/RMSNorm boundaries per decode token while retaining both the
BF16 residual sum and normalized result. Fast now enables this together with
buffer swapping and separate GPU argmax on single-row M4 Pro / Metal calls.
The measured combined route uses 245 compute commands versus 314 previously,
with 17.1–24.5% lower complete-token latency and 107–115 tokens/s streaming
medians. Multi-row behavior is unchanged. The four measured arms were explicit
policies (`combined`, `residual-norm`, `swap-argmax`, `all-three`) through
`edb610a`; Fast is the promoted `all-three` arm.

### Projection load scheduling and block size

The [six-arrangement study](../../studies/model_generation/projection-arrangements.md)
compares fixed-width 896/4864 loading and 64/128/256-thread blocks over the current
Fast path. Fixed-width arms reduced projection active time in Metal traces and
streamed about 119–120 tokens/s, but none met the frozen full-token acceptance
gate across all three histories. Fast retains the original projection kernel.
The measured arms, policies `projection-0` through `projection-5`, exist through
`edb610a`.

The [scheduling diagnosis](../../studies/model_generation/projection-scheduling.md)
reproduces the larger long-context gain in advancing decoding. Host forward
submission remains about 6.7 ms while long-context readback wait drops from
2.3–2.4 ms to 1.2 ms. Fixed-width projection active time stays near 4.9 ms across
contexts, supporting a submission/backlog explanation without changing MLP shapes.
Profiler perturbation prevents treating traced gaps as exclusive CPU time.
