# GQA decode: fusion, parallelism and KV reuse

One query token has 14 heads of 64 BF16 values. K and V each hold T×2×64
values: seven query heads share one KV head. Query head h uses KV head h//7,
attends to all T visible tokens, and scales its dot product by 1/8. The caller
owns projections, RoPE, KV-cache updates, allocation and synchronization.
The [materialized path](../../src/llm_mojo/attention.mojo) exposes probability
scratch; the [decode alternatives](../../src/llm_mojo/attention_decode.mojo)
return only O[1,14,64].

Online softmax keeps a running maximum m, normalizer z, and unnormalized
weighted output u. Scores round to BF16 in registers; m/z/u stay FP32. Final
output rounds to BF16. Scores and probabilities need not be written to a
device buffer, but K/V still traverse the memory hierarchy. Independent tests
compare the distinct probability-rounding paths at frozen tolerances.

| Mapping | Work ownership | What is traded |
| --- | --- | --- |
| Materialized | Separate QK, softmax and probability×V stages | Three dispatches and BF16 scratch; inspectable correctness control |
| Simple fusion G1 | One 32-lane SIMD group per query head; each lane owns dimensions l and l+32 | One dispatch, register-resident state, only 14 groups |
| Parallel fusion G32 | 32 SIMD groups per query head partition the KV sequence | 14 threadgroups of 1024 threads; 8,448 shared bytes, one barrier and an internal merge |
| Split64 H4 | 64 sequence splits, up to four related heads per group | 256 threadgroups; more independent work and KV reuse; second dispatch merges partial states |

To combine partial states, compute M=max(m_j), then
Z=sum(exp(m_j−M) z_j), U=sum(exp(m_j−M) u_j), and O=U/Z.
This is why splitting can remain numerically stable without materializing
attention probabilities.

At T=4096, unique K+V occupy 2 MiB. One-head ownership requests those values
seven times per KV group. Four-head reuse needs two passes, reducing requested
K/V loads by 3.5×. It also needs 24 FP32 source state values per lane before
temporaries. Those counts are neither measured DRAM traffic nor a physical
register allocation. Split workspace is 231 KiB, larger than the baseline's
112 KiB score/probability scratch at this length. More workspace can still win
by exposing parallel work and reuse.

The maintained matrix compares these four existing designs at
T=1,16,64,256,1024,4096 in hot and ring24 modes. It keeps a materialized self-pair
inside the same run. All twelve original optimized configurations remain in
numerical tests, including ragged/empty splits, large/tied scores, cancellation,
head mapping, output poisoning and T=4095/4096. Conditional exponential
rescaling and every intermediate tuning choice are documented in the
[historical campaign](https://github.com/attilio-castano/llm-mojo/blob/a86f4dbadb4b8c9255aacb004ad30fdfefeaa8fd/experiments/EXP-0014-gqa-decode/report.md).

The historical campaign found that fusion alone helped, additional parallelism
helped much more, and long-context split/reuse improved further. Reducing the
number of exponential calls did not improve its strong controls. A routing
bug also taught us to validate the branch actually launched: numerical parity
alone cannot detect a benchmark accidentally timing another correct kernel.
Those lessons motivate the retained route and calibration checks.

The algorithm draws on online softmax and work partitioning in
[FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf) and sequence
splitting in [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).
They are algorithm references, not performance controls. These results compare
with this repository's materialized baseline. They do not establish MLX parity,
a universal crossover, decoder-block correctness or prefill performance. The
new measurements leave both optimized finalists as explicit choices.
