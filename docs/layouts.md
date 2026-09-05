# Layout language

This project uses a small, concrete language to describe how tensor operations
map onto memory and hardware. The language separates four questions that are
easy to conflate:

1. **Value:** What are the logical axes, extents, dtypes, and arithmetic?
2. **Storage:** Where does each logical coordinate live in linear memory?
3. **Work:** Which CPU iteration, SIMD lane, or GPU thread owns each element?
4. **Order:** In what order are reductions performed, where do casts occur, and
   what synchronization makes intermediate values visible?

A storage layout does not answer the work or order questions. Likewise, a
thread partition does not establish a numerical reduction order or prove which
device executed it.

## Storage notation

Write a strided layout as:

```text
shape : stride
```

For a two-dimensional layout `(R, H) : (sr, sh)`, logical coordinate `(r, h)`
maps to linear offset:

```text
r * sr + h * sh
```

For example, `(R, H) : (H, 1)` is row-major storage with a contiguous hidden
axis. A zero stride denotes a projected or broadcast view: `(R, H) : (0, 1)`
reuses one length-`H` row for every logical row without materializing copies.

The notation describes an address mapping, not an allocation. Two views may
share storage, and an implementation must state which object owns that storage.

## RMSNorm V0

Let `R` be the number of token rows and let `H = 896` be the Qwen hidden size.
The operation has these logical tensors:

```text
X[row, hidden]  BF16  input activations
W[hidden]       BF16  learned weights
Y[row, hidden]  BF16  output activations
```

The initial storage contract is:

| View | Layout | Meaning |
| --- | --- | --- |
| `X` | `(R, H) : (H, 1)` | Contiguous hidden values for each row |
| `Y` | `(R, H) : (H, 1)` | Same physical organization as `X` |
| `W` as seen by `Y` | `(R, H) : (0, 1)` | One weight row projected across all token rows |
| inverse RMS as seen by `Y` | `(R, H) : (1, 0)` | One scalar per row projected across the hidden axis |

The projected views are semantic descriptions; the reference implementation
need not construct them as objects or allocate their repeated values.

The reference work mapping is one explicit row traversal followed by one
explicit hidden-axis traversal.

The first Apple GPU mapping is equally concrete:

| Question | Mapping |
| --- | --- |
| Row owner | Metal threadgroup `block_idx.x` owns one logical row |
| Hidden owner | Thread `t` owns `h = t + k * 128` for nonnegative `k` |
| Threadgroup | 128 threads; one-dimensional |
| Reduction storage | FP32 shared scratch `(128) : (1)` |
| Reduction order | Per-thread serial sums, then a `128 -> 64 -> ... -> 1` shared-memory tree |
| Synchronization | One barrier after publishing partial sums, one per tree level, and one after thread 0 publishes inverse RMS |
| Output | The same strided hidden ownership writes `Y` after the BF16 cast and weight multiply |

The enqueue boundary is asynchronous: the kernel does not synchronize the
device context for its caller. Correctness tests explicitly order host-to-device
copies, the kernel, device-to-host copy, and final synchronization. The GPU
enqueue path rejects any device API other than `metal`, while the kernel
requires an Apple GPU compilation target. Together with a readback comparison,
those checks prevent silent CPU execution from being counted as Apple GPU
evidence.

## Affine linear projection V0

Let `R` be the number of token rows, `K` the input-feature width, and `N` the
output-feature width. The operation has these logical tensors:

```text
X[row, input_feature]             BF16  input activations
W[output_feature, input_feature]  BF16  learned weights
B[output_feature]                 BF16  learned bias
Y[row, output_feature]            BF16  output activations
```

The initial storage contract is source-compatible and output-major:

| View | Layout | Meaning |
| --- | --- | --- |
| `X` | `(R, K) : (K, 1)` | Contiguous input features for each token row |
| `W` | `(N, K) : (K, 1)` | One contiguous input-feature row per output feature |
| `B` | `(N) : (1)` | One bias value per output feature |
| `Y` | `(R, N) : (N, 1)` | Contiguous projected features for each token row |

The reference work mapping traverses row, output feature, then input feature.
Each dot product accumulates serially in FP32, adds the BF16 bias after
promotion to FP32, then casts once to BF16.

The first Apple GPU mapping is rowwise rather than a tiled matrix
multiplication:

| Question | Mapping |
| --- | --- |
| Dot-product owner | One 32-lane SIMD group owns one `(row, output_feature)` pair |
| Input owner | Lane `l` owns `k = l + i * 32` for nonnegative `i` |
| Threadgroup | 128 threads containing four independent SIMD groups |
| Reduction | Each lane accumulates FP32 in a register; `warp.sum` combines the 32 partials |
| Weight access | At each iteration, adjacent lanes read adjacent `W[output_feature, k]` values |
| Synchronization | No threadgroup barrier or shared allocation; the SIMD reduction is group-local |
| Output | Lane zero adds bias, casts once to BF16, and writes `Y` |
| Dispatch | `ceil(R * N / 4)` one-dimensional threadgroups |

For Qwen decode, `R = 1` and `K = 896`, so every lane owns exactly 28
input-weight products. The mapping remains numerically valid for any positive
`R`, including prefill, but it deliberately does not reuse weight tiles across
token rows. A later tiled prefill kernel may sit behind the same operation
boundary after a synchronized benchmark establishes its useful row-count
range. The V0 attention path enqueues Q, K, and V separately and does not pack
or materialize a combined weight tensor.

As with RMSNorm, enqueue is asynchronous, the enqueue wrapper rejects a
non-Metal device context, and the kernel requires an Apple GPU compilation
target. The committed oracle tests cover a non-SIMD-aligned decode width, more
than one input iteration, and multiple token rows. Separate host-versus-GPU
checks cover the Qwen query and key/value shapes. This is a correctness mapping,
not a performance claim.

The experimental direct-prefill control changes only output ownership. One
128-thread group owns an `8x16` rectangle of `Y`; each thread serially computes
one complete dot product and handles bounds at incomplete `M` or `N` tiles.
It still reads `X` and `W` directly from device memory, allocates no threadgroup
storage, and executes no threadgroup barrier. This control is not the public
dispatch path. Its purpose is to separate the cost of one-thread-per-output
ownership from the effect of explicitly staging `BK` operand tiles.

The first tiled-prefill candidate preserves that exact output ownership and
adds `BK=32` operand staging. In each K phase, all 128 threads cooperatively
fill an `8x32` input tile and a `16x32` weight tile in threadgroup memory,
zero-filling ragged edges. One barrier makes both tiles visible, each valid
thread accumulates its 32 products into an FP32 register, and a second barrier
protects the storage before the next phase overwrites it. The accumulator
survives all `ceil(K / 32)` phases; bias and the single BF16 cast happen only
after the full K reduction.

For BF16 operands, this is 1,536 bytes of threadgroup storage and two barriers
per K phase (56 barriers at `K=896`). Separate `M`, `N`, and `K` tail tests
cover the synchronization-safe zero-fill policy. This is an explicit learning
candidate, not the public dispatch path: timing must establish whether its
reuse pays for its shared-memory accesses and barriers.

The measured ownership comparisons now live in
[linear prefill](../studies/linear_prefill/README.md). That study covers direct,
shared BK16, register 2x2, rowwise, and Apple MMA mappings. The engine keeps
these explicit choices; a historical crossover is not an automatic selector.

## RoPE V0

Let `R` be the number of contiguous token rows, `N` the number of heads,
`D = 64` the Qwen head dimension, and `P` the number of positions represented
by the rotary table. Query application uses `N = 14`; key application uses
`N = 2`. The generic operation has these logical tensors:

```text
X[row, head, dimension]  BF16  input query or key values
C[position, dimension]  BF16  full cosine table
S[position, dimension]  BF16  full sine table
Y[row, head, dimension]  BF16  rotated output values
```

The initial storage contract is:

| View | Layout | Meaning |
| --- | --- | --- |
| `X` | `(R, N, D) : (N*D, D, 1)` | Contiguous dimensions within each token and head |
| `Y` | `(R, N, D) : (N*D, D, 1)` | Same physical organization as `X` |
| `C` | `(P, D) : (D, 1)` | One full duplicated cosine row per absolute position |
| `S` | `(P, D) : (D, 1)` | One full duplicated sine row per absolute position |

For input row `r`, the operation reads table row
`p = start_position + r`. Within a head, dimension `i` for
`0 <= i < D / 2` pairs with dimension `i + D / 2`. The reference work mapping
is an explicit row, head, and first-half traversal; each iteration reads and
writes one pair.

The first Apple GPU mapping is:

| Question | Mapping |
| --- | --- |
| Pair owner | Global thread `g` owns one `(row, head, first-half dimension)` tuple |
| Pair index | `i = g mod (D / 2)` |
| Head index | `n = (g / (D / 2)) mod N` |
| Row index | `r = g / ((D / 2) * N)` |
| Threadgroup | 128 threads; one-dimensional, with an explicit total-pair bound |
| Storage | Direct BF16 global loads and out-of-place BF16 global writes |
| Synchronization | None inside the kernel; pairs are independent |
| Dispatch | The same kernel is enqueued separately for query and key tensors |

As with RMSNorm, enqueue is asynchronous, the host path rejects a non-Metal
device context, and the kernel requires an Apple GPU compilation target. This
is a correctness mapping, not a claim that separate dispatches, out-of-place
storage, or the 128-thread block size are optimal.

## Grouped-query attention V0

Let `R` be the current query-row count, `T` the full active key/value-row
count, `Nq` the query-head count, `Nkv` the key/value-head count, and `D` the
head dimension. The initial operation has these logical tensors:

```text
Q[query_row, query_head, dimension]       BF16  rotated queries
K[key_position, key_value_head, dimension] BF16 rotated active keys
V[key_position, key_value_head, dimension] BF16 active values
S[query_row, query_head, key_position]    BF16  scores, then probabilities
O[query_row, query_head, dimension]       BF16  attention output
```

Its row-major storage contract is:

| View | Layout | Meaning |
| --- | --- | --- |
| `Q` | `(R, Nq, D) : (Nq*D, D, 1)` | One contiguous dimension vector per query head |
| `K` | `(T, Nkv, D) : (Nkv*D, D, 1)` | One contiguous dimension vector per active key head |
| `V` | `(T, Nkv, D) : (Nkv*D, D, 1)` | One contiguous dimension vector per active value head |
| `S` | `(R, Nq, T) : (Nq*T, T, 1)` | Caller-owned materialized scores, overwritten by probabilities |
| `O` | `(R, Nq, D) : (Nq*D, D, 1)` | One contiguous output vector per query head |

The contract requires `R <= T` and `Nq` divisible by `Nkv`. It maps
`query_head` to `key_value_head` with
`query_head / (Nq / Nkv)`, without creating repeated K or V storage. Query row
`r` sees the inclusive key range `0 .. T - R + r`; later slots in `S` are
written as zero.

The serial host reference traverses query row, query head, and key position for
QK, then each query row/head for stable softmax, then query row, query head,
and output dimension for probability-times-V. QK and probability-times-V
reductions accumulate in FP32 and materialize BF16 at the stage boundaries.

The first Apple GPU mapping preserves those three materialized stages:

| Stage | Global-thread owner | Serial work owned by that thread |
| --- | --- | --- |
| scaled QK | One `(query_row, query_head, key_position)` score | `D` BF16 operand pairs accumulated in FP32, scaled, then cast to BF16 |
| stable causal softmax | One `(query_row, query_head)` row | Maximum, exponential sum, and normalization over visible keys in FP32; probabilities cast to BF16 |
| probability times V | One `(query_row, query_head, dimension)` output | Visible key positions accumulated in FP32, then cast to BF16 |

Every stage uses a one-dimensional grid of 128-thread groups with a total-work
bound. No stage uses threadgroup storage or a barrier. The separate dispatches
are enqueued in dependency order on one `DeviceContext`; the enqueue function
does not synchronize for the caller.

`S` contains `R * Nq * T` BF16 values. For Qwen this is `28 * R * T` bytes:
linear in `T` for decode (`R = 1`) and quadratic in `T` for full prefill
(`R = T`). At the V0 `T = 4,096` limit, that is 112 KiB for decode or 448 MiB
for full prefill. It is a Metal device allocation from the runtime's
perspective. Apple Silicon provides unified physical memory, but this interface
does not make the device buffer a host tensor; tests use explicit
device-to-host copies.

The caller supplies five non-overlapping tensor views and must keep their
device-buffer handles alive through the last queued use. A later reuse enqueued
on the same ordered context is safe without a host synchronization.
Deallocation, host mutation, or reuse from an unordered execution context must
wait for an explicit completion boundary.

This is a deliberately inspectable correctness baseline, not a performance
claim. It establishes the oracle, cast points, masking, head mapping, and
asynchronous ownership boundary before an online-softmax, tiled, fused, or
explicit multi-query-reuse implementation is considered.

## Use in code and evidence

- Name semantic axes before reducing them to integer positions.
- Record logical shape, physical strides, dtype, and ownership at operation
  boundaries.
- Record work partition and reduction schedule separately from storage layout.
- Use Mojo's native layout and tensor-view facilities when they express the
  required mapping; do not build a project-specific layout algebra.
- Treat tiling, vectorization, projection, and composition as views until an
  implementation proves that data moved or was allocated.
- Do not infer performance from a layout. Performance evidence requires a
  synchronized benchmark and verified device/backend identity.
- Add vocabulary only when implemented code creates a distinction that needs a
  name.

This language is informed by the shape/stride layout algebra described in
[Categorical Foundations for CuTe Layouts](https://arxiv.org/abs/2601.05972),
while remaining tied to the operations and native Mojo facilities used here.
