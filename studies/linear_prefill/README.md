# Linear prefill: reuse weights across rows

Prefill computes X[M,896] times W[1152,896] transposed, plus bias, yielding
O[M,1152]. All arrays are BF16 and contiguous along their last dimension;
dot products accumulate in FP32. Increasing M supplies more token rows and
creates opportunities to reuse weights. See [the source](../../src/llm_mojo/linear.mojo).

The work is approximately 2 × M × 896 × 1152 floating-point operations.
At M=256 that is about 528 million operations per packed projection. The
weight footprint remains about 1.97 MiB; reuse across rows can therefore
change the limiting resource compared with decode.

| Mapping | Tile / owner | Reuse and cost |
| --- | --- | --- |
| Rowwise | One SIMD group per output | K is partitioned over lanes; one FP32 accumulator per lane and one reduction |
| Direct | 8×16 output tile, 128 threads | Each thread owns one output and streams K; no shared staging or barriers |
| Shared BK16 | Same 8×16 tile | Cooperatively stage 8×16 input and 16×16 weight values per K phase; 768 bytes shared, 56 phases, 112 barriers |
| Register 2×2 | 8×16 output tile, 32 threads | Four FP32 accumulators per lane; reuse loaded operands across four outputs; no shared storage |
| Apple MMA | 8×16 output tile, 32 threads | Two collective 8×8 fragments per K=8 phase; four distributed FP32 accumulators per lane |

The register mapping computes four complete output accumulators per lane;
it is distinct from the rowwise K reduction. MMA distributes fragment
ownership across the same hardware SIMD group. It is one arithmetic option,
not a prerequisite for correct tiling. The shared candidate illustrates why
reducing source loads can still lose when barriers and staging cost too much.

The maintained comparison uses M=1,8,16,64,256 in hot and ring24 modes. Every
candidate is paired directly with rowwise, including a rowwise self-pair.
This gives a readable ownership comparison without replaying every historical
BK sweep. Independent tests retain exact and ragged M/N/K cases, safe shared
barriers, collective zero fill and comparisons with the host reference.

Historical EXP-0007 through EXP-0013 found different crossovers for these
mappings. Those observations are preserved in Git. This study establishes new
measurements with one protocol; it does not inherit an old crossover threshold
or introduce an automatic selector. Packed N=1152 evidence is not proof for
N=128, arbitrary M, other devices, or complete-model throughput.
