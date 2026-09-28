# Batched decode plan

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives. The work landed on `main` as one squash
> commit, `132dc08` (#29). The commits it names are on #29's branch, which
> `git fetch origin pull/29/head` retrieves
> ([evidence commits](../experiments.md#evidence-commits)), except two from before
> a rebase that the 1a record explains.

Baseline: `3b23c65`, `main` with the [serving plan](../serving-plan.md), the
[September streamlining](streamlining-2026-09.md) (one-step setup,
explicit execution plans, current docs separated from history) and
[shared oracle fixtures](shared-fixtures-2026-09.md). The plan was first
written at `a0b01da`, and its work was rebased onto those changes. It
implements the serving plan's phase 1: several sequences decoding in one step,
with KV storage owned outside the model. The work proceeds in five steps, each
committed with its own gate:

1. **1a, step format and KV pool.** Route today's single-sequence calls
   through the step format and a KV pool, with no arithmetic change.
2. **1b, batched decode.** Add multi-sequence forms of the single-row decode
   kernels.
3. **1c, batch-size study.** Measure throughput and latency against batch size.
4. **1d, exact batched projections.** Cut the per-row work of multi-row
   projections without changing any row's arithmetic.
5. **1e, reordered batched projections.** Measure what a different summation
   order buys once batched rows still equal solo rows.

Approved on 2026-09-23 for local implementation of 1a: edits, builds, tests,
validation and incremental commits. Pushing, pull requests, toolchain upgrades
and numerical-contract changes need a separate decision. 1b was approved on
2026-09-25 and 1c on 2026-09-26, on the same terms. 1d was chosen on
2026-09-26 as path 1 of 1c's decision and approved the same day. 1e was
approved the same day, and adopting its reordered arrangement 8 was approved on
2026-09-27 after a single-sequence check.

Status: 1a–1e are complete; see the [validation record](#validation-record).
1c's throughput hypothesis failed with tile 4. 1d's exact arrangement 5 cut
batched step time by 31–65%. 1e's arrangement 8, which sums four adjacent
products per lane, cut batched steps by a further 17–36% from B = 16 and
single-sequence steps by about 10%, with arrangement 5's worst-case accuracy
and HF agreement. It is now the decode arrangement, which changes Fast's
arithmetic in the last bits. 1f, a diagnostic, found that none of that gain comes
from addressing: it is the four-wide loads.

## 1a. Step format and KV pool

### Ownership

| Owner | After 1a |
| --- | --- |
| `serving/batch.mojo` | `StepBatch`: one step's tokens, positions, sequence offsets, KV lengths, block table, write slots and logit rows |
| `serving/kv_pool.mojo` | `KVPool`: one BF16 allocation for all KV storage, cache views per block and layer, and their logical lengths; sized by a `KVGeometry` the model supplies |
| `QwenModel` | weights, workspaces, logits and selection buffers, validity and submission counters, and `kv_geometry()`; no KV storage or sequence length |
| `ChatSession`, CLIs, drivers | a one-block pool for their single sequence |

### Pool layout

The pool uses the serving plan's block-major layout:

```text
Pool[block, layer, kv, slot, head, dim]  BF16
(NB, 24, 2, BS, 2, 64) : (24*2*BS*128, 2*BS*128, BS*128, 128, 64, 1)
```

Each (block, layer) pair gets `AttentionCache` views of its K and V ranges,
created once with `DeviceBuffer.create_sub_buffer`. With one block of the full
context, where BS equals the model capacity, every view has exactly the shape of
today's per-layer caches. The attention and decoder code therefore run
unchanged. `AttentionCache` gains a constructor that adopts caller-provided
storage.

A probe on 2026-09-23 (Apple M4 Pro, Metal, MAX 26.5.0) established the
properties this relies on:
- Kernel writes through a sub-buffer view land at its offset in the parent.
- `enqueue_fill` and host mapping of a view touch only its range.

The decoder preflight's overlap check compares address ranges, so K and V views
of one allocation remain disjoint regions.

### Step format

`StepBatch.sequence(ids, past, block, block_size)` describes one sequence
writing `ids` at positions `past, past + 1, ...` in one block.
`validate(blocks, block_size, vocabulary)` checks:

- offsets start at zero, increase, and end at the token count;
- positions are contiguous within each sequence and end at its KV length;
- each write slot is `block_table[s, position / BS] * BS + position % BS`;
- block IDs are in range, and no block is written by two sequences;
- `decode_count` counts the leading one-token sequences;
- each logit row is the last row of a sequence;
- token IDs are within the vocabulary.

In 1a, `QwenModel.forward(ctx, batch, kv, plan)` and `forward_captured` accept
one sequence with one block; the `ExecutionPlan` still chooses the kernels. The
model rejects the batch in two cases:
- the pool geometry differs from the model (its layer count, 2 KV heads,
  64 dims, block size equal to the model capacity);
- the first position differs from the length of any of the block's layer
  views.

It then runs the existing layer code on those views. Benchmark rewinds use
`KVPool.truncate`, which only shortens logical lengths.

### Gate

- **Exact equality with the baseline.** Build the model and chat drivers from
  the baseline and from this change. On the pinned checkpoint, run fixed
  histories covering full prefill, chunked prefill with Fast selections,
  one-token decode, and the three-turn chat driver. Require byte-identical
  logits, all 48 KV buffers including untouched poisoned capacity, captured
  hidden states, selected tokens, cache lengths and submitted-row counts.
- **Regression suite.** `uv run --locked llm-mojo validate` passes: frozen
  oracles, Python tests, every native suite and benchmark smoke. The model
  lifecycle driver passes.
- **New native tests.** Step batch construction and every rejection rule; pool
  offsets, isolation between views, truncation and reset.
- **Sanity check, not a performance claim.** Alternating baseline and candidate
  generation runs show decode step time unchanged within run-to-run noise.

## 1b. Batched decode

Designed on 2026-09-25 from `e27ec90`.

A step decodes one token for each of S sequences. Each sequence owns one
full-context block of an S-block pool, as in 1a. 1b delivers:
- one launch per kernel for all S sequences, so every step issues the same 245
  launches whatever S is;
- each sequence's logits, selected token and appended K/V bytes equal to
  decoding it alone;
- unchanged bytes at S = 1.

Batched prefill, several blocks per sequence and scheduling belong to phases 2
and 3. The baseline and consistent research routes stay single-sequence.

### Today's step and its batched form

Configuration 26 issues 245 launches per token: the embedding, 11 launches in
layer 0 and 10 in each later layer, the vocabulary projection and two argmax
passes. Every launch has a batched form that keeps each row's arithmetic:

| Launch | Today | Batched |
| --- | --- | --- |
| Embedding | `_embedding`, one thread per element | unchanged, S rows |
| Input RMSNorm, layer 0 | `_rms_norm_apple_gpu_simdgroup_kernel`, one threadgroup per row | unchanged, S rows |
| QKV with bias, Wo, gate, up, down and vocabulary projections | `_linear_rowwise_apple_gpu_kernel`, one SIMD group per output | `_linear_rowwise_rows_apple_gpu_kernel`, with each row's lane-strided FP32 sum, `warp.sum`, bias and single rounding |
| Unpack, RoPE and K/V append | `_fused_decode_qkv`, position as a scalar argument | a sequence index; position and block from the step buffer; still elementwise |
| Decode attention | `_decode_kernel[32, 1, 1]`, one threadgroup per query head | `_decode_sequences_kernel`, route 4's arithmetic with each sequence's Q row, K/V base and length; keys still go to SIMD groups by position mod 32 and merge in a fixed order |
| Residual and RMSNorm, twice per layer | `_residual_norm`, one row | one threadgroup per row |
| SiLU × up | `_silu_multiply`, elementwise | S × 4,864 elements |
| Argmax | `_argmax` over 149 groups, then `_finish` | a row index in both passes, one record per sequence |

Buffer swapping between layers stays valid, because it exchanges buffer owners
rather than rows. The greedy readback maps all S records at once.

### Decisions

- **One kernel source for every S.** Configuration 26 runs the batched
  composition at S = 1 too, so a batched row and a solo row execute the same
  compiled kernels and differ only in the data they address. Projections are
  the one exception: the rows kernel hands one-row calls to the one-row kernel,
  and `tests/test_consistency.mojo` shows the two agree bit for bit. The
  attention kernel is shared with research routes 4 and 11, which already use
  its grid axes (`block_idx.y` is the split and `block_idx.z` a query row of one
  sequence), so the batched step gets its own kernel with route 4's arithmetic,
  tested against route 4 bit for bit; those routes keep their kernel.
- **A fixed composition.** Multi-row calls through today's layer dispatch
  switch to MMA projections at 16 rows, prefill attention above one row and MLP
  mapping 7, each a different reduction order. `enqueue_decode_batch_layer` in
  `layers/decoder_layer.mojo` therefore composes the ten launches directly.
  Configuration 26 moves to it at every S, and the single-row fusion branches
  it used in the generic layer code retire: fused QKV, fused activation and the
  deferred residuals and norms. `forward_captured` covers the composition, so
  the capture-based gates keep working.
- **Comptime model dimensions.** New and generalized kernels take the query and
  KV heads, head size and hidden width as comptime parameters that `QwenModel`
  supplies, so they name no model; see the
  [dependency direction](../cli.md#dependency-direction).
- **One upload per step.** Positions reach today's kernels as the scalar
  `cache.length`. A batched step needs them on the GPU, so one int32 step
  buffer holds the S token IDs, positions and block IDs, written through a
  single host mapping like today's token upload. The layer index stays a scalar
  argument, so one upload serves all 24 layers. Kernels compute K/V element
  addresses in 64 bits from the pool layout:
  `((block · L + layer) · 2 + kv) · BS · 128 + position · 128 + column`.
- **Row tile.** The rows kernel reads each weight once per tile of 4, 8 or 16
  rows, which changes weight traffic but not arithmetic. 1b uses 4, the only
  tile in use today; 1c measures the others.
- **Model state.** `QwenModel` gains `max_sequences`, at most `max_rows`, which
  sizes the per-sequence buffers: logits `[max_sequences, 151936]`, the final
  norm, argmax partials `[max_sequences, 149, 3]` and results
  `[max_sequences, 3]`. `ExecutionPlan.validate` takes the
  sequence count: configuration 26 needs one row per sequence, and every other
  configuration needs one sequence. `fast_plan` selects configuration 26 for a
  decode-only batch on the measured device and rejects batches elsewhere.
  `greedy_tokens` returns one token per sequence, and `greedy` stays for one. A
  nonfinite row invalidates the model, as a nonfinite logit does today;
  per-request handling is a phase 3 question.
- **Preflight.** A batched step requires every sequence to decode one token,
  S ≤ `max_sequences`, one block per sequence with the model's block size, and
  each block's 24 layer lengths equal to its sequence's position.
  `StepBatch.validate` already rejects two sequences writing one block. A
  rejection happens before the upload and changes nothing.
- **Route record.** `ForwardRoute` records S and counts launches where they are
  enqueued. A per-sequence loop would multiply those counts, so the route test
  proves that launches do not depend on S.

### Gate

- **Kernels, without the checkpoint.** Each generalized kernel at
  S ∈ {1, 2, 3, 8, 16, 32}, with mixed positions (1, 31, 32, 33 and 4,096 keys
  included) and scattered, unsorted blocks, equals the same kernel run once per
  sequence, and unwritten pool rows keep their poison. The rows kernel equals
  one-row launches at the Qwen widths 1,152 with bias, 896, 4,864 and 151,936,
  with tiles 4, 8 and 16. Today's test covers only 19 outputs, and the
  vocabulary head has never run through this kernel.
- **S = 1 unchanged.** The 1a equality gate against the previous head (4,170
  files) and the decode-route test's byte comparison of Fast with the baseline.
- **Batched equals solo, without the checkpoint.** On three fixture layers, for
  S ∈ {2, 3, 8, 16, 32}, sequences with different prefixes decode 12 steps
  together and alone. Logits, final norms, the K/V bytes of every block with
  unwritten rows poisoned, lengths and submitted rows agree, and the route's
  launch counts equal their S = 1 values.
- **Batched equals solo, with the checkpoint.** A model-driver mode decodes
  eight real conversations of different lengths together for 16 steps; tokens
  and logits match their solo runs.
- **Rejections.** A mixed batch, S above `max_sequences`, a non-Fast plan with
  S > 1, a position that disagrees with its block, and a block size other than
  the model's are each rejected with state unchanged.
- **Suite and timing.** `uv run --locked llm-mojo validate` passes. At S = 1,
  alternating baseline and candidate runs show decode step time unchanged
  within run-to-run noise; this is a sanity check, not a claim.

**Test data.** Each batched row's reference is the same sequence decoded alone,
so 1b adds no oracle family. Native tests build small models from the verified
decoder fixtures, as the decode-route test does, and run without the
checkpoint; model-level checks use the prepared checkpoint. Validation links
those fixtures read-only from the
[shared store](../development.md#shared-oracle-fixtures). Tests write captures
under `build/test_*` and records under `build/oracle_records/`, never into
`build/oracle_data/`. Leaving the generators' inputs untouched keeps validation's
oracle stage to seconds, so every 1b step can run the full suite; a new family
would need its own generator, anchors and store entry.

### Steps

Each step is one commit with its own gate:

1. Rows-kernel tests at the Qwen widths, with no production change.
2. Row-indexed `_residual_norm` and argmax, and batched forms of
   `_fused_decode_qkv` and route 4 attention, each with kernel tests.
3. `enqueue_decode_batch_layer` and the step buffer. Configuration 26 moves to
   them at S = 1, its generic-layer branches retire, and the S = 1 gate runs.
4. Batched steps: `max_sequences`, the plan and preflight rules,
   `greedy_tokens` and route counts, with the fixture gate and rejections.
5. The checkpoint driver mode, the suite and the validation record.

### Risks

- **Contraction.** Metal may contract `a · b + c` into a fused multiply-add
  differently in each compiled kernel; this once changed RoPE bits. Running one
  kernel source at every S removes the risk between batched and solo rows.
  Projections, the one pair of different kernels, multiply BF16 values whose
  products are exact in FP32, so contraction cannot change their sums outside
  subnormal and overflow cases. Between old and new kernels, the S = 1 gate
  catches any difference, and explicit `fma` control fixes it, as it did for
  RoPE.
- **Weight traffic.** With tile 4, S = 16 reads the weights four times per
  step and S = 64 sixteen times; see the 1c hypothesis.
- **Pool size.** 64 full-context blocks form one 3 GiB buffer, and the largest
  Metal buffer is an open question in the serving plan.
- **Host bookkeeping.** Each step makes 24·S length updates on the host; 1c's
  host breakdown shows what they cost.

## 1c. Batch-size study

Designed on 2026-09-26 from `68e0fa3` and approved the same day on the terms
of 1a and 1b.

**Question.** How do step latency, aggregate throughput and each sequence's
token latency scale with the number of sequences decoding together, at short,
medium and long context, and how much does the projection row tile change that?

### Matrix

- **Workloads.** B ∈ {1, 2, 4, 8, 16, 32, 64} at contexts 64, 1024 and 3968,
  plus one mixed batch of 32 sequences whose contexts spread evenly from 64 to
  3968.
- **Comparisons.** Each workload runs four pairs of arms against tile 4 in one
  process:
  - tile 4 against itself, for calibration;
  - tile 8 against tile 4;
  - tile 16 against tile 4;
  - tile 4 with host marks against tile 4 without, for observation.
- **Procedure.** The four-block paired procedure of the
  [experimental method](../experiments.md): ten warmups and ten samples per arm,
  with blocks 2 and 3 reversing the order of contexts, batch sizes,
  comparisons and arms. One process per block and context runs all of that
  context's batch sizes.
- **Timed interval.** From building the step batch and plan through
  `greedy_tokens` readback, as the token-profile study times one sequence.
  Rewinding each block to its context is outside the interval.

### Setup

`QwenModel(ctx, prepared, 4096, 256, 64)` serves a pool of 64 full-context
blocks: one 3 GiB allocation, which a probe on 2026-09-26 allocated and copied
between blocks. The frozen history is prefilled once into block 0 with Fast
chunks and copied into every other block, whose lengths are then set to their
sequences' contexts. Sequence b decodes the history token after its context,
offset by b, so rows differ. An untimed tile-4 step records every sequence's
token. Every sample must reproduce those tokens with 245 launches for B
sequences; the tile cannot change a token, because tiles are bit-identical.

### Changes

- **Model.** `forward` takes the row tile as a comptime parameter, defaulting
  to the plan's `DECODE_ROW_TILE`, so one binary holds tiles 4, 8 and 16.
  `greedy_tokens` records host marks under `OBSERVE`, as `greedy` does.
- **Benchmark.** `benchmarks/model.mojo` gains a `batch` mode for timing and a
  `-D MODEL_BATCH_PROFILE` build for traces; the token-profile modes are
  unchanged.
- **Contract.** `model_contract.py` declares the matrix (`BATCH_DECLARATION`)
  and the trace geometry of the implementation `qwen_model_batch`. Its command
  sequence is the current route's 249 commands per step at every B, so the
  existing capture and analysis tools apply unchanged.
- **Pipeline.** `model_profile.py` gains `batch-size-build`, `-collect`,
  `-capture`, `-archive`, `-replay` and `-plot` beside the token-profile
  commands, which keep replaying their own archive.
- **Traces.** Metal System Traces at context 1024 for B ∈ {1, 16, 64} with
  tile 4, two repeats each, give active GPU time, the enclosing GPU span and
  Metal submission intervals per step.
- **Evidence.** `studies/model_generation/batch-size.md` explains the results
  from one lossless archive, `batch-size.json.gz` with its manifest, a derived
  `batch-size-summary.json` and figures that `batch-size-plot` rebuilds without
  a GPU.

### Reported quantities

For each workload, as the median over blocks of block medians:
- step latency, which is also each sequence's token latency;
- aggregate tokens per second, B divided by step latency, and its ratio to
  B = 1 at the same context;
- tile 8 and tile 16 against tile 4 under the method's decision rule, with the
  calibration noise floor;
- host intervals from the observation arm: preflight, step upload, embedding,
  decoder-stack and head enqueue, the readback wait and selection.

A tile that meets the gain rule is evidence for changing Fast's tile; adopting
it is a separate decision.

### Gate and steps

Collection starts from a clean commit whose suite passes. Before and after each
block, AC power, Low Power Mode off and a nominal thermal state are required and
recorded. No run is discarded or repeated for a preferred outcome. Each step is
one commit:

1. The row-tile parameter and `greedy_tokens` marks, with a test that tiles 8
   and 16 reproduce tile 4's bytes for batched steps.
2. The benchmark mode, contract and pipeline, with parser, census and replay
   tests.
3. Collection and traces on the reference machine; the archive, the study and
   its record.

**Hypothesis, recorded before measurement.** In the
[runtime study](../../studies/model_generation/runtime-measurements.csv), 16-row
forward calls took about 21–29 ms and 64-row calls about 36–49 ms at contexts
1024–4096. Today's one-row step takes about 8–9 ms. If B decode rows cost about
as much as B prompt rows, B = 16 gives roughly 5–7× today's throughput and
B = 64 gives 12–16×. Those rows used other kernels and attended to a single
sequence, so this is only a prior. If B = 16 gives less than 3×, investigate
with traces before starting phase 2.

**Refinement from the 1b analysis, still before measurement.** The prior
assumes each weight is read about once per step. The rows kernel reads it once
per row tile, and at history 1024 projections take 6.3 ms of a one-row step's
7.4 ms of active GPU time
([projection arrangements](../../studies/model_generation/projection-arrangements.md)).
With tile 4, a step of B rows would then spend about ⌈B/4⌉ × 6.3 ms on
projections. That puts B = 4 near 4× today's throughput and both B = 16 and
B = 64 near 5×, unless tiles 8 and 16 keep their time per pass. In multi-row
prefill cells, tile 8 was 11–16% slower than tile 4 and tile 16 was 37–61%
slower ([decoder policies](../../studies/decoder_layer/policies.md)); those cells
do not predict decode-shaped batches.

## 1d. Exact batched projections

Planned on 2026-09-26 from `dfa9a56` as path 1 of the decision recorded under
1c in the [validation record](#validation-record), and approved the same day
on the terms of 1a–1c. No output element's arithmetic changes, so this is not a
numerical-contract change.

**Question.** How much of the batched projections' per-row work can be removed
without changing any row's arithmetic, and how much batched throughput does
that recover?

### What costs time today

1c's traces at 1,024 cached tokens put 90% of a B = 64 step's GPU time in the
121 projections. A one-row pass over them takes 6.4 ms, and a tile-4 pass over
four rows 13.1–14.2 ms. The tile-4 kernel gives one SIMD group four rows and one
output column. For each weight element a lane loads, it then does, for every
row: an input load, a conversion to FP32, a row guard and a multiply-add. Only
the weight load and its conversion are shared by the four rows. Tiles 8 and 16,
which share each weight read among more rows, were slower. The work to remove
is therefore the per-row load, conversion and guard, not the weight read.

### The exact contract

For output (r, c), lane l of a SIMD group accumulates in FP32, in increasing
order, the products x[r, f] · w[c, f] for every f ≡ l (mod 32). `warp.sum`
combines the 32 lane sums, the bias is added in FP32 and the result is rounded
to BF16. Every arrangement below keeps that computation for every output
element. What may change is how many output elements one SIMD group computes,
which SIMD groups share a threadgroup, when loads are issued and whether rows
beyond the batch are guarded inside the loop. One row still goes to the one-row
kernel, so single-sequence decode does not change.

### Levers

- **Column blocking.** One SIMD group computes four output columns for its four
  rows. Each input value is loaded and converted once for four weights, and
  each lane step has 16 independent multiply-adds instead of four.
- **Fixed widths with early loads.** The reduction width is a compile-time 896
  or 4,864, both multiples of 128. Each lane issues four iterations' loads before
  their four sequential updates. At one row this cut projection time by 22–23%
  with identical bytes
  ([projection arrangements](../../studies/model_generation/projection-arrangements.md)).
- **No guard in the loop.** Rows past the batch load the last valid row, and
  their sums are never stored. Every projection width (1,152, 896, 4,864 and
  151,936 outputs) is a multiple of four, so no column guard is needed.
- **Column-block order.** Consecutive SIMD groups take the row tiles of one
  column block. From B = 16, the four SIMD groups of a threadgroup then read the
  same weight rows at nearly the same time, so the cache can serve reads that
  tile 4 repeats in every pass.

### Arrangements

`forward` takes a compile-time projection arrangement in place of the row tile,
and the decode composition and vocabulary head pass it to one rows kernel. IDs
0–2 keep 1c's tiles 4, 8 and 16, so 1c's builds stay reproducible.

| ID | Rows × columns | Width and loads | Order | Isolates |
| ---: | --- | --- | --- | --- |
| 0 | 4 × 1 | runtime, one iteration | row tiles outer | today's tile 4, the control |
| 3 | 4 × 1 | fixed, four iterations | row tiles outer | early loads |
| 4 | 4 × 4 | runtime, one iteration | row tiles outer | column blocking |
| 5 | 4 × 4 | fixed, four iterations | row tiles outer | both |
| 6 | 4 × 4 | fixed, four iterations | column blocks outer | both, plus shared weight reads |

Arrangements 3–6 keep the row guard out of the loop. Each adds one lever to a
neighbour, so each lever's effect can be read from adjacent arms.

### Exactness gates

- **Kernel.** `tests/test_decode_batch.mojo` compares every arrangement with
  one-row launches, bit for bit, at the five decode shapes. Those are 1,152
  outputs with bias, 896 and 4,864 outputs from 896 inputs, 896 from 4,864, and
  151,936 from 896. It covers 2, 3, 5, 8, 13, 16, 31, 33 and 64 rows, poisoned
  guard rows and BF16 edge values: signed zeros, subnormals and values next to
  rounding boundaries. Unsupported widths are rejected before any launch.
- **Composition.** On the fixture layers, every arrangement's batched steps
  for S ∈ {2, 3, 8, 16, 32} reproduce arrangement 0's logits, final norms and
  every pool byte, with 245 launches per step.
- **Checkpoint.** `validation.model batch` runs with the selected arrangement:
  eight conversations, batched equal to solo in every token, logit and K/V byte.
- **One row.** Single-sequence decode is untouched, and the S = 1 route tests
  and decode parity run unchanged.

### Measurement

The batch-size matrix is reused under a new declaration. 1c's declaration
stays frozen, so its archive keeps replaying, and the batch-size commands take
the declaration's name.

- **Workloads.** 1c's 22: B from 1 to 64 at 64, 1,024 and 3,968 cached
  tokens, plus the mixed batch of 32. At B = 1 every arrangement runs the
  one-row kernel, so those comparisons measure identical code.
- **Comparisons.** Arrangement 0 against itself for calibration, and 3, 4, 5
  and 6 against 0, in the four-block paired procedure with ten warmups and ten
  samples per arm. That is 8,800 samples, about 30 minutes.
- **Traces.** Metal System Traces at B = 64 and 1,024 cached tokens for
  arrangements 0, 3, 4, 5 and 6, two repeats each, give each projection's
  active time. Attempts that fail the coverage check are replaced with the
  same binary and recorded, as in 1c.
- **Reported.** Step latency, tokens per second and paired ratios for every
  workload and arrangement, per-projection active time and the cost of a
  projection pass. The study `studies/model_generation/batch-projections.md`
  reports them from one lossless archive, replayed without a GPU.

### Decision

- **Per workload.** Each candidate is a gain, a regression or inconclusive
  under the method's rule, with the calibration noise floor.
- **Qualifying.** No regression in any workload with B ≥ 2, and a gain in
  every workload with B ≥ 4.
- **Selecting.** Among qualifiers, the lowest worst-case median ratio over
  workloads with B ≥ 4, then the lowest mean ratio, then the lower ID.
- **Confirming.** A fresh four-block run compares the selected arrangement with
  arrangement 0 over the same 22 workloads, with its own calibration and the same
  qualifying rule. If it fails, no other candidate is tried.
- **Promoting.** If confirmed, the selected arrangement becomes the default for
  batched decode. Otherwise tile 4 stays, and the study records why.

**Hypothesis, recorded before measurement.** If per-row work sets the cost,
column blocking (4) should save 25–50% of the projection time, about 20–45% of
a step at B ≥ 16. Early loads (3) should save about the 22% they saved at one
row. Together (5), they would take the B = 64 step at 1,024 cached tokens from
234 ms to about 110–150 ms, 1.5–2× today's 273 tokens/s. At that speed, sixteen
tile-4 passes request about 16 GB of weights per step. Column-block order (6)
should therefore beat 5 from B = 16 and match it at B ≤ 4. If 4 shortens steps
at B ≥ 16 by less than 10%, input work is not the main per-row cost. In that
case, stop adding arrangements and measure with GPU counters instead.

### Steps

Each step is one commit with its gate:

1. The rows kernel with arrangements 3–6, and the kernel exactness tests.
2. The arrangement parameter through `forward`, the decode composition and the
   head, with the composition test.
3. The declaration, the benchmark's arms and the pipeline, with census,
   decision, selection and replay tests. `llm-mojo validate` passes before
   collection.
4. Screen, traces, selection and confirmation on the reference machine; the
   archive, the study and its record.
5. If confirmed, the new default, the checkpoint gate and the full suite.

### Risks

- **Registers.** Four-by-four blocking with four-deep loads holds about 48 FP32
  values per lane. Lower occupancy may cancel part of the savings; arrangement 4
  against 5 shows whether it does.
- **Contraction.** A new loop shape could change whether the compiler fuses a
  multiply and an add. The kernel tests catch any difference. An inexact
  arrangement is fixed before the freeze or dropped, with the reason recorded.
- **Conditions.** Background load and trace attribution are handled as in 1c.
  Conditions are recorded, and the paired design absorbs slow drift.

**Out of scope.** The one-row kernel, split reductions and matrix-multiply
projections (path 2), a different arrangement per batch size, and merging gate
and up into one launch.

## 1e. Reordered batched projections

Planned on 2026-09-26 from `07ed15f` and approved the same day on the terms of
1a–1d. Arrangements 8–10 change the summation order of decode projections, which
is a numerical-contract change. The approval covers building and measuring them,
not adopting one: adopting a reordered arrangement waits for your decision on
the study's evidence.

**Question.** With batched rows still equal to solo rows, how much faster can
decode projections be if each output's summation order may change, and how
does the change move Fast's numbers?

### What limits arrangement 5

At 64 sequences and 1,024 cached tokens, arrangement 5's projections take
69.1 ms of 90.3 ms of active GPU time
([batched projections](../../studies/model_generation/batch-projections.md)).
Each four-row pass reads every weight, so 64 sequences take 16 passes and
request about 16 GB of weights per step. Across the five projection shapes those
requests arrive at 196–253 GB/s, close to the chip's nominal 273 GB/s memory
bandwidth. That suggests, without proving, that the passes stream weights from
memory. Column-block order, which let SIMD groups share reads through the cache,
did not help. The next lever is to use each loaded weight for more rows. In
today's order every lane holds one accumulator per row and column it serves, so
more rows cost registers. Tiles that change the order, above all the GPU's
matrix units, reuse a weight across many rows cheaply.

### Batch invariance under a new order

An arrangement keeps batched rows equal to solo rows when three conditions hold:
- it computes each output from that row's inputs alone;
- its summation order does not depend on how many rows there are;
- one row runs the same order.

Arrangement 5 meets them with the one-row kernel's order. A new order needs its
own one-row path: the same kernel with one row, padding its tile if necessary,
or a one-row kernel that sums in exactly the same order. Single-sequence decode
then adopts the new order too, which is what makes this a contract change.

### Arrangements

| ID | Rows × columns per SIMD group | Summation order | Rows per weight read | One row |
| ---: | --- | --- | ---: | --- |
| 5 | 4 × 4 | today's: each lane sums every 32nd product, then `warp.sum` | 4 | the one-row kernel (control) |
| 7 | 8 × 4 | today's | 8 | the one-row kernel |
| 8 | 4 × 4 | each lane sums four adjacent products in every 128, then `warp.sum` | 4 | a one-row kernel with the same order |
| 9 | 8 × 32 | matrix units: 8×8 fragments along K in steps of 8, FP32 accumulators | 8 | the same kernel, padded to 8 rows |
| 10 | 16 × 16 | matrix units, as 9 | 16 | the same kernel, padded to 16 rows |

Arrangement 7 is the exact frontier: if it matches the reordered arrangements,
no contract change is needed. Arrangement 8 keeps 5's passes, so it isolates the
cost of load instructions. Arrangements 9 and 10 run the existing matrix-unit
tiles `_linear_mma_tile` at 8×32 and 16×16, which have no shared storage,
barriers or K splitting. Fast's configuration 3 already uses the 16×16 tile for
prefill's QKV and output projections. Arrangements 7 and 8 keep 5's fixed widths
and early loads.

### Numerical gates and diagnostics

- **Batch invariance, exact.** For every arrangement, batched rows equal its own
  one-row results bit for bit at the five decode shapes. This holds for 1 to 64
  rows, with poisoned guard rows and edge values. Whole batched steps equal each
  sequence decoded alone under the same arrangement, with 245 launches.
- **Accuracy bound, a gate.** Each arrangement's error is measured in BF16 ulps
  against an FP64 sum of the same BF16 operands, rounded to BF16. This covers
  the five shapes with random and edge values. A reordered arrangement
  qualifies only if its worst error does not exceed arrangement 5's on the same
  cases; the report gives both distributions.
- **Teacher-forced comparison, a diagnostic.** Decode parity's schedule, 53
  prefix tokens and 32 fixed decode tokens, runs each reordered arrangement
  against arrangement 5 on the real model. It records per-call distances in
  hidden states, logits and K/V, and token agreement.
- **HF comparison, a diagnostic with a stop rule.** The runtime study's HF
  workflow runs for arrangement 5 and the selected arrangement in one session,
  on its decode cases and generation histories. The study stops before any
  adoption and investigates if either holds:
  - the selected arrangement agrees with HF on more than one fewer same-history
    choice than 5;
  - its largest KL divergence more than doubles.
- **Fusion exactness.** Decode parity and the route test keep checking the
  decode composition's fusions byte for byte against the baseline route. They
  pin arrangement 5, whose projections equal the baseline's.

### Measurement and decision

The batch-size matrix runs under a new declaration, with arrangement 5 as the
control.
- **Workloads.** The same 22. B = 1 now matters: it is single-sequence decode,
  and arrangements 8, 9 and 10 change it.
- **Comparisons.** 5 against itself, and 7, 8, 9 and 10 against 5: 8,800
  samples. Traces capture B = 64 at 1,024 cached tokens for all five
  arrangements, two repeats each.
- **Qualifying.** Both exactness gates and the accuracy bound; no regression in
  any workload, B = 1 included; a gain in every workload with B ≥ 16.
- **Selecting.** The lowest worst-case median ratio over workloads with
  B ≥ 16, then the lowest mean, then the lower ID.
- **Confirming.** A fresh four-block run of the selected arrangement against 5
  with the same rule, then the two model-level diagnostics.
- **Adopting.** If 7 is selected, it becomes the default on 1d's terms. If a
  reordered arrangement is selected, the study goes to you first. On your
  approval it becomes the default, the model contract documents its order, and
  tests that pin Fast decode bytes move to it. Otherwise arrangement 5 stays.

**Hypothesis, recorded before measurement.** If weight passes set the cost at
large B:
- Arrangements that halve them, 7 and 9, should cut projection time at B = 64 by
  30–45%, and 10, which quarters them, by 40–60%.
- Arrangement 8 keeps the passes, so it should gain less than 10% from B = 16,
  though it may gain more at B ≤ 4.
- Arrangement 7 may lose part of its gain to registers.
- Padding one row to a tile of 8 or 16 may slow single-sequence decode with 9
  and 10. The [linear prefill study](../../studies/linear_prefill/README.md) found
  every tiled mapping slower than row-wise at one row when weights stream from
  memory.
- All reordered arrangements should stay within one BF16 ulp of the FP64 sum.

If none of 7, 9 and 10 shortens the B = 64 step by 10%, weight passes are not
the limit, and the next step is GPU counters, not more arrangements.

### Steps

Each step is one commit with its gate:

1. Arrangements 7–10, with the batch-invariance and accuracy tests.
2. The arrangements through `forward`, and the composition test. Decode parity
   and the route test pin arrangement 5, and a build define selects the decode
   arrangement for validation builds.
3. The declaration, the benchmark's arms and the pipeline, with their tests.
   `llm-mojo validate` passes before collection.
4. Screen, traces, selection, confirmation and the model-level diagnostics on
   the reference machine; the archive, the study and its record.
5. Adoption as above, with the checkpoint gate and the full suite.

### Risks

- **One row.** Arrangements 9 and 10 may qualify at every B except 1, and the
  rule then rejects them. A tile that keeps single-sequence decode fast would be
  a separate study.
- **Matrix-unit order.** The hardware's accumulation inside a fragment is not
  documented. Batch invariance rests on the tests, not on reasoning about the
  hardware.
- **Registers.** Arrangement 7 holds 32 accumulators per lane before its early
  loads.
- **Reference cost.** The HF reference runs on CPU with pinned Torch. It is slow,
  but it runs once per compared arrangement.

**Out of scope.** Attention, prefill kernels, split reductions, a different
arrangement per batch size, and lower-precision accumulation.

## 1f. Addressing check

Requested on 2026-09-28, after 1e's adoption, as a diagnostic: it changes no
default.

**Question.** Arrangement 8 differs from arrangement 5 in two ways at once. Each
lane loads four adjacent values with one vector load, and every load is
addressed from a raw pointer offset instead of through the tensor layout. How
much of 8's gain does raw addressing give alone, in arrangement 5's order?

**Arrangement 11.** Arrangement 5 with each scalar load addressed as
`ptr[row * WIDTH + k]`, with `WIDTH` fixed at 896 or 4,864. It keeps 5's tile,
early loads and lane-strided order, so its outputs equal the one-row kernel's bit
for bit, and one row runs the one-row kernel. The kernel tests check that.

**Measurement.** The batch-size matrix under its own declaration: arrangement 5
against itself, 11 and 8 in four blocks, 5,280 samples, and traces of 5, 11 and
8 at B = 64 and 1,024 cached tokens, two repeats each. Every sample must
reproduce its arm's tokens, and arrangement 11's must equal arrangement 5's.
There is no selection and no confirmation.

**Analysis.** In each workload where arrangement 8 is a gain, the share of its
gain that arrangement 11 reaches, (1 − r11) / (1 − r8), from their median paired
ratios against arrangement 5; and the same share of traced projection time.

**Hypothesis, recorded before measurement.** Load width, not addressing,
explains most of arrangement 8's gain: arrangement 11 reaches less than a third
of it from B = 16.

**Consequence.** Arrangement 8 stays the default unless arrangement 11 is slower
in no workload and reaches at least 80% of 8's gain in every workload from
B = 16. Then an exact default goes back to you as a decision, with 1e's gates.

**Steps.** First, arrangement 11 with its exactness tests and the study's
declaration and arms. Then the screen and traces on a quiet machine, with the
analysis. An archive and a study follow if you ask for them or the consequence
applies.

**Result.** Raw-pointer addressing gives none of the gain. Arrangement 11 was
inconclusive against arrangement 5 in all 22 workloads, with median ratios of
0.977–1.023 from B = 16. There it reached between −12% and 9% of arrangement
8's gain, a median of 1%. In the traces, all projections took 69.9 ms with
arrangement 5, 70.3 ms with 11 and 43.2 ms with 8. The compiler already turns
arrangement 5's layout indexing into the same address arithmetic; what costs
time is the number of load instructions, which only the four-wide loads cut.
The hypothesis held, the consequence did not apply, and arrangement 8 stays the
default. At your request the result is kept as a compact record, not a study:
see the validation record below.

## Validation record

### 1f on `6b222ee`, 2026-09-28

Two commits implement 1f:

| Commit | Change |
| --- | --- |
| `d263f99` | arrangement 11 with its exactness tests, the addressing declaration and arms, and the reordered checks narrowed to 8–10 |
| `6b222ee` | layout checks before launch for arrangements 8 and 11 and for residual RMSNorm's normal output, from review on #29 |

- **Exactness.** `tests/test_decode_batch.mojo` passed on both commits:
  arrangement 11's outputs equal the one-row kernel's bit for bit at every
  decode width for 1 to 64 rows, and its batched steps equal arrangement 0's.
  The Python suite passed.
- **Screen.** The frozen build was rebuilt from a clean `6b222ee` after the
  layout checks. Four blocks from 13:21 to 13:29 produced 5,280 samples, each
  reproducing its arm's tokens with 245 launches; arrangement 11 changed no
  token. AC power, normal power mode and no thermal or performance warning were
  recorded before and after every block and trace. One-minute load averages
  were 4.2 at the start, just after the rebuild and its tests, and 1.8 at the
  end of the screen.
- **Traces.** Six accepted captures at B = 64 and 1,024 cached tokens cover
  11,952 measured commands. Arrangement 8's repeat 0 failed the coverage check:
  two measured commands' Compute slots were labelled as WindowServer's, the
  label swap 1c recorded. It was replaced with the same binary, which passed on
  the first recapture, and the rejected attempt's evidence is kept.
- **Analysis.** As declared; see the [result](#1f-addressing-check).
- **Record.** `studies/model_generation/batch-addressing.json.gz`, 59,720 bytes,
  keeps every screen sample, each accepted trace's per-stage totals rather than
  its intervals, and the rejected attempt. `batch-size-replay --study
  addressing` verifies it and recomputes the analysis into
  `batch-addressing-summary.json`; the retained-record test rejects eleven
  kinds of damage.

**Deviation from the plan.** None beyond the replaced trace.

### 1e adoption on `39fecba`, 2026-09-27

Two commits adopt arrangement 8, whose evidence `ff007a5` recorded:

| Commit | Change |
| --- | --- |
| `d5e38e1` | decode parity and the route test's byte checks stay on exact arrangement 5; the route test also checks the default arrangement's route, launches and untouched scratch; the whole-step batched-equals-solo test also covers 5; build receipts record the effective arrangement |
| `39fecba` | `DECODE_PROJECTION = 8`, the model contract's decode projection order, the docs and study statuses, and the single-sequence check's record |

- **Single-sequence check.** Before adoption, under a rule fixed before
  measuring, the Fast generator ran in arrangements 5 and 8 from clean
  `9d5a24f` builds. Sixteen runs each generated 128 tokens after a 1,176-token
  prompt, in four blocks ordered 5 8 8 5 and 8 5 5 8, from 12:16 to 12:17.
  Arrangement 8's median decode step was 7.87 ms against 8.70 ms, with block
  ratios of 0.893–0.915, and both generated the same text. AC power, normal
  power mode and no thermal or performance warning were recorded before and
  after; one-minute load averages were 2.9–3.4. The
  [record](../../studies/model_generation/batch-reordered-single-sequence.json)
  keeps every decode step.
- **Tests under 8 before the switch.** `tests/test_decode_route.mojo` and
  `tests/test_decode_batch.mojo` passed on `d5e38e1` with the source default and
  with `-D DECODE_PROJECTION=8`.
- **Checkpoint gate.** `validation.model batch` ran from a clean `39fecba`,
  whose default build reported arrangement 8 and whose receipt records it.
  Eight conversations of 11 to 3,301 prompt tokens decoded 16 steps batched and
  alone, and every token, logit and K/V byte was identical.
- **Decode parity.** A build with `--decode-projection 5` matched the baseline
  in all 32 teacher-forced calls and 4,059 captured files, so the fusions
  change no bytes on the full model.
- **Decode comparison.** The default build against that exact build agreed in
  31 of 32 tokens, with logit relative RMS up to 0.054, as in the 1e
  diagnostics.
- **Suite.** `uv run --locked llm-mojo validate` passed on `39fecba`: oracle
  anchors with the shared fixtures, 281 Python tests, all 27 native test files
  plus the Unicode tokenizer run on Metal, and every benchmark smoke.

**Deviations from the plan.** The plan moved decode parity and the route test
to the adopted arrangement. Neither can move: both compare the fused route
with the baseline route, whose one-row projections have no arrangement 8
order. By your choice, their byte checks stay on arrangement 5, where they
isolate the fusions, and the route test adds the default arrangement's route
checks. The single-sequence check was added before adoption at your request.

### 1e on `3185eea`, 2026-09-26 and 27

These commits implement 1e's first four steps, each with its gate:

| Commit | Change |
| --- | --- |
| `91a9908` | arrangements 7–10 with the batch-invariance and accuracy tests |
| `5c85f52` | the arrangement through `forward` and validation builds: the `DECODE_PROJECTION` define, `decode-comparison`, and decode parity pinned to exact arrangements |
| `3185eea` | the declaration, the benchmark's arms, the accuracy census, `batch-size-diagnose` and the pipeline tests |
| `9d5a24f` | the HF references run with `uv` from PATH and a three-hour limit |
| `02eecff` | the reordered replay accepts an attempt that failed during capture |
| this commit | the archive, the [study](../../studies/model_generation/batch-reordered.md) and this record |

- **Before collection.** `uv run --locked llm-mojo validate` passed on a clean
  `3185eea`: oracle anchors with the shared fixtures, 278 Python tests, the
  native suites on Metal with the new batch-invariance tests, and every
  benchmark smoke. Functional runs of the reordered batch mode and of the
  decode comparison passed their checks; their timings were not used.
- **Accuracy census.** Recorded before the screen's first block. Arrangements
  5, 7 and 8 share the same worst error in every shape; 9 and 10 exceed it in
  all five, so they could not qualify.
- **Screen.** Four blocks from 18:32 to 18:43 on 2026-09-26 produced 8,800
  samples. Every sample reproduced its arm's reference tokens with 245
  launches, and no exact arm changed a token. AC power, normal power mode and
  no thermal or performance warning were recorded before and after every
  block. The machine was quiet, with load averages of 1.5–2.5 around the screen.
- **Traces.** Ten accepted captures of 64 sequences at 1,024 cached tokens
  cover 19,920 measured commands. Two attempts were set aside and replaced with
  the same binaries:
  - Arrangement 5's repeat 0, captured at 18:43, failed the coverage check.
    The Compute interval in one measured command's slot was labelled as the
    Codex service's, and that command's ID sat on a Vertex interval 11.7 ms
    before the preceding command ended. Its replacement, at 09:58 the next
    day, passed.
  - Arrangement 9's repeat 0 ran its binary to completion at 18:45. Then
    xctrace stopped responding at "Stopping recording..." and saved no trace,
    holding its "Instruments is recording" sleep assertion. It was found and
    stopped at 09:21 the next morning: it ignored SIGINT, and SIGTERM ended it
    with exit code 1. `capture_trace` wrote an invalid receipt naming the
    failure. Its replacement, at 09:23, passed.

  The other six captures ran from 09:23 to 09:26. `verify_build` compares the
  whole source identity, branch name included, and by the arrangement 5
  replacement the branch had moved to `02eecff`. For that capture the branch
  was renamed, recreated at `3185eea` in this worktree's clean checkout, and
  restored afterwards.
- **Decision.** The frozen rule qualified only arrangement 8 and selected it,
  with a worst median ratio of 0.822 from B = 16 and a mean of 0.727.
  Arrangement 7 was slower at B = 2 and 4; 9 and 10 failed the accuracy gate
  and were slower from B = 1 to 8.
- **Confirmation.** A fresh four-block run from 09:26 to 09:31 on 2026-09-27
  produced 3,520 samples. Arrangement 8 was a gain in every workload from
  B = 8, with median ratios of 0.636–0.827 from B = 16, and slower in none. The
  machine was in use: one-minute load averages of 2.6–6.3, with file syncing at
  up to 113% of a core. Calibration deviations reached 31% at B ≤ 4; from
  B = 8 the control's steps stayed within 2.6% of the screen's.
- **Diagnostics.** `batch-size-diagnose` ran from `9d5a24f` between 09:40 and
  09:50. Arrangement 8 selected the same token as 5 in 31 of 32 teacher-forced
  calls. It agreed with HF on 242 of 244 same-history choices, as 5 did, and
  its largest KL divergence was 0.00774 nats against 0.00708. The stop rule
  did not trigger. A first run at 09:33 built all four binaries and wrote the
  decode comparison, then stopped at the HF reference, whose command looked for
  `uv` inside the environment. The rerun from `9d5a24f` reproduced the binaries
  byte for byte and every numerical field of the comparison.
- **Replay.** `batch-size-replay --study reordered` reapplies the accuracy
  gate, the frozen rule and the stop rule, and regenerates the summary without
  a GPU. The retained-archive test rejects twenty-one kinds of damage.
- **Suite.** The Python suite passed with the evidence: 279 tests, including
  the documentation links. No Mojo source changed after `3185eea`, whose full
  suite passed before collection.

**Outcome.** In the confirmation, arrangement 8 cut the batched step by 17–36%
from B = 16. On the quiet screen, at 64 sequences, it took a step from 74.4 to
47.0 ms at 64 cached tokens, from 91.3 to 64.1 ms at 1,024 and from 159.3 to
132.7 ms at 3,968. That is 1,362, 998 and 482 tokens/s. Traced projection time
at B = 64 fell from 71.4 to 43.3 ms. Against the recorded hypothesis:
- arrangement 8 gained far more than the predicted 10%: load instructions, not
  weight passes, set most of the per-layer projections' cost;
- the matrix-unit tiles cut projection time by 47% and 54%, as fewer passes
  predicted, but slowed one sequence 1.7–2.4× and failed the accuracy gate;
- arrangement 7 gained 6–12% at B = 64, well short of 30–45%;
- no arrangement stayed within one ulp in the vocabulary shape, where 5 and 8
  reach 1.21 ulps.

**Deviations from the plan.**
- The two set-aside capture attempts, and the branch procedure for the
  arrangement 5 replacement, as above.
- The confirmation and eight of the ten traces ran while the machine was in use.
- Two pipeline fixes after collection, both committed before the archive was
  built: the HF reference's `uv` and time limit (`9d5a24f`), and the replay's
  acceptance of an attempt that failed during capture (`02eecff`).
- The draft pull request's push, at 18:45:46 on 2026-09-26, came 38 seconds
  into the arrangement 9 capture that failed. Captures took 23–32 seconds, so
  the failure most likely began before the push.

**Decision.** Arrangement 8 met every gate. On 2026-09-27 you chose to adopt
it after a single-sequence check, and to keep decode parity and the route
test's byte comparisons on exact arrangement 5, since the baseline route has
no arrangement 8 order. The [study](../../studies/model_generation/batch-reordered.md#decision)
records the check, and the adoption record above records the gates.

### 1d on `f47fb8b`, 2026-09-26

Five commits implement 1d, each with its gate:

| Commit | Change |
| --- | --- |
| `6bfd837` | arrangements 3–6 and the kernel exactness tests |
| `2259144` | the arrangement parameter through `forward`, the decode composition and the head |
| `f47fb8b` | the projection declaration, the benchmark's arms, `batch-size-confirm` and the pipeline tests |
| `6cc28c1` | the archive, the [study](../../studies/model_generation/batch-projections.md) and this record |
| `5925268` | arrangement 5 as the batched default, with the docs and study indexes |

- **Before collection.** `uv run --locked llm-mojo validate` passed on a clean
  `f47fb8b`: oracle anchors with the shared fixtures, 274 Python tests, the
  native suites on Metal with the new exactness tests, and every benchmark
  smoke. A functional run of the projection batch mode at 64 cached tokens
  passed its token and launch checks; its timings were not used.
- **Screen.** Four blocks from 13:02 to 13:21 produced 8,800 samples. Every sample
  reproduced its reference tokens with 245 launches. AC power, normal power mode
  and no thermal or performance warning were recorded before and after every
  block. Every calibration deviation stayed within 5%.
- **Traces.** Ten captures, of arrangements 0, 3, 4, 5 and 6 with two repeats
  each, at B = 64 and 1,024 cached tokens, cover 19,920 measured commands. All
  passed the coverage check on the first attempt.
- **Decision.** All four candidates qualified. The frozen rule selected
  arrangement 5, with a worst median ratio of 0.561 from B = 4; 6, 4 and 3
  followed at 0.565, 0.609 and 0.832.
- **Confirmation.** A fresh four-block run from 13:25 to 13:34 produced 3,520
  samples. Arrangement 5 was a gain in every workload from B = 2, with median
  ratios of 0.351–0.692, all within 0.009 of the screen's.
- **Replay.** `batch-size-replay --study projections` reapplies the frozen rule
  and regenerates the summary without a GPU. The retained-archive test rejects
  twelve kinds of damage.
- **Promotion.** `validation.model batch` ran from a clean `5925268`, where
  `DECODE_PROJECTION` is 5. Eight conversations of 11 to 3,301 prompt tokens
  decoded 16 steps batched and alone, and every token, logit and K/V byte was
  identical. Single-sequence decode still runs the one-row kernel.
- **Suite.** `uv run --locked llm-mojo validate` passed on `5925268`: oracle
  anchors with the shared fixtures, 275 Python tests, the native suites on
  Metal and every benchmark smoke.

**Outcome.** At 64 sequences, arrangement 5 takes a step from 210.0 to 73.7 ms at
64 cached tokens, from 227.6 to 90.8 ms at 1,024 and from 292.2 to 155.5 ms at
3,968. That is 868, 705 and 412 tokens/s, 7.0, 5.7 and 3.8 times one sequence.
Traced projection time at B = 64 fell from 206.6 to 69.1 ms. Against the
recorded hypothesis:
- column blocking saved more than predicted, 45–63% of a step from B = 16;
- early loads saved about what they saved at one row;
- the composition beat the predicted 110–150 ms;
- column-block order never beat arrangement 5, so that prediction failed.

**Deviations from the plan.** None beyond the functional run above.

### 1c on `7b3b131`, 2026-09-26

Three commits implement 1c, each with its gate:

| Commit | Change |
| --- | --- |
| `f18b08f` | the row tile as a `forward` parameter and `greedy_tokens` marks, with a test that tiles 8 and 16 reproduce tile 4's batched steps |
| `7b3b131` | the benchmark mode, contract and pipeline, with parser, census and summary tests |
| `f1f89fb` | the archive, the [study](../../studies/model_generation/batch-size.md) and this record |

- **Before collection.** `uv run --locked llm-mojo validate` passed on a clean
  `7b3b131`: oracle anchors with the shared fixtures, 270 Python tests, the
  native suites on Metal and every benchmark smoke.
- **Timing.** Four blocks from 09:36 to 09:57 produced 7,040 samples. Every
  sample reproduced its reference tokens with 245 launches. AC power, normal
  power mode and no thermal or performance warning were recorded before and
  after every block.
- **Traces.** Six accepted Metal System Traces at 1,024 cached tokens, for
  B = 1, 16 and 64 with two repeats each, cover 11,952 measured commands. B = 16
  needed two attempts for repeat 0 and four for repeat 1. The four rejected
  attempts each lost one or two commands' Compute intervals: the trace labelled
  them as WindowServer or the wallpaper extension. Each was replaced with the
  same binary, and the archive keeps the rejected receipts and evidence.
- **Replay.** `batch-size-replay` regenerates the summary from the archive
  without a GPU. It now also checks every block's and trace's recorded power
  conditions, and that each rejected attempt used the frozen binary. The
  retained-archive test rejects eight kinds of damage.
- **After the evidence.** `uv run --locked llm-mojo validate` passed on
  `f1f89fb`: oracle anchors with the shared fixtures, 271 Python tests, the
  native suites on Metal and every benchmark smoke.

**Outcome.** The recorded hypothesis failed. B = 16 gave 2.9×, 2.2× and 2.2×
B = 1 at 64, 1,024 and 3,968 cached tokens, and B = 64 gave 3.2×, 2.3× and
2.2×. Each added sequence costs 3.2–4.6 ms of step time. The refinement failed
too: a tile-4 projection pass costs 13.1–14.2 ms, not the one-row 6.4 ms. Tiles
8 and 16 were slower than tile 4 in all 19 multi-row workloads, so Fast keeps
tile 4. Because B = 16 gave less than 3×, traces were examined before phase 2.
Projections take 90% of a B = 64 step's GPU time, and their per-row work, not
weight traffic, sets that cost.

**Deviations from the plan.**
- Trace replacements, as above.
- A Docker Desktop virtual machine started one minute into the third block and
  ran through the rest of the collection and every trace. For multi-row
  workloads, the last two blocks' medians were within −5% to +6% of the first
  two.
- The replay checks and one figure subtitle changed after collection. The
  archive was then rebuilt from the same raw timings and traces with the
  committed analysis code.

**Decision needed before phase 2.** Batched decode is exact and gives 2.2–3.2×
one sequence's throughput at B = 64. Raising that depends on the multi-row
projection kernel. There are three paths:

1. **Exact kernel work.** Keep each row's lane-strided FP32 order and cut the
   per-row work, for example by letting one SIMD group compute several output
   columns so that each input load and conversion serves several weights.
   Batched rows stay bit-identical to today's one-row decode, so the numerical
   contract is unchanged.
2. **A matrix-multiply projection.** SIMD-group matrix operations share weights
   across rows with little per-row work but change the K reduction order.
   Using the same kernel for one row keeps batched rows equal to solo rows but
   changes Fast's arithmetic. That numerical-contract change needs approval and
   the diagnostic comparisons.
3. **Phase 2 first.** Paged KV does not depend on the projection kernel, and its
   study question, the cost of block translation, can be answered at today's
   throughput.

### 1b on `f3abcd2`, 2026-09-25

Five commits implement 1b, each with its gate:

| Commit | Change |
| --- | --- |
| `c397c61` | tests the rows kernel at every decode width |
| `de069cc` | sequence-aware residual RMSNorm, argmax, fused QKV and decode attention |
| `0b73841` | Fast decode runs through the batched composition, still one sequence per step |
| `ca2b9fc` | a configuration 26 step decodes several sequences |
| `f3abcd2` | checks batched decode on the real model |

Same machine and toolchain as below.

- **Kernels.** `tests/test_decode_batch.mojo` checks, bit for bit and with
  guard rows and unwritten pool rows poisoned:
  - the rows kernel against one-row launches at the widths 1,152 with bias,
    896, 4,864 and 151,936, for 2 to 64 rows with tiles 4, 8 and 16;
  - residual RMSNorm and argmax against single-row launches, where a NaN flags
    only its row;
  - fused QKV/RoPE/append against the unfused path, and batched attention
    against route 4, for 1 to 32 sequences with 1 to 4,096 keys in scattered
    blocks.
- **S = 1 unchanged.** Executables built from `0b73841` and from the pre-1b
  head `ce05db5` produced byte-identical outputs on the 1a equality gate: 4,170
  files per side. The decode-route test finds the composition byte-identical to
  the baseline route and counts 35 launches for its three layers, which is 245
  for 24.
- **Batched equals solo, fixture.** For S ∈ {2, 3, 8, 16, 32}, 12 steps on
  three fixture layers agree with each sequence decoded alone in tokens, all
  logits, final norms, cache lengths, submitted rows and every pool byte. The
  launch count does not depend on S. Invalid batches change nothing: a prefill
  chunk beside a decode, too many sequences, a research plan with several
  sequences, a disagreeing position, another block size and a batched capture.
- **Batched equals solo, checkpoint.** `validation.model batch` ran from a clean
  `f3abcd2`. Eight conversations of 11 to 3,301 prompt tokens decoded 16 steps
  with every token, all 151,936 logits per row and every K/V byte of both pools
  identical, at 245 launches per batched step. The lifecycle study passed from
  the same build, in normal and device-sync mode.
- **Suite.** `uv run --locked llm-mojo validate` passed on `0b73841` and on
  `f3abcd2`:
  - frozen oracle anchors, with the three large families from the shared store;
  - 267 Python tests;
  - all 26 native test files plus the Unicode tokenizer run on Metal;
  - every benchmark smoke route.

**Deviations from the plan.** Decode attention got its own kernel with route
4's arithmetic rather than a mode of the shared kernel; routes 4 and 11 keep
theirs. `tests/test_qkv_fusion.mojo` retired with the single-row fused kernel,
and its checks moved to the batched kernel's test.

**Timing sanity check at S = 1, no claim.** On AC power, 16 runs of 128 tokens
after a 1,176-token prompt formed four alternating blocks, comparing the pre-1b
generate binary with `f3abcd2`'s. Median decode steps were 8.020 ms for the
baseline (runs 7.990–8.236 ms) and 8.081 ms for 1b (8.005–8.295 ms). 1b/baseline
block ratios were 0.999, 1.007, 1.001 and 1.020. The 0.8% difference is below the
5% floor and one block favors 1b, so the check is inconclusive: no regression is
visible. 1c measures batched throughput.

### Package boundaries before 1b, 2026-09-25

Three commits prepare 1b under the
[dependency direction](../cli.md#dependency-direction):
- `f10668d` sizes every pool from `QwenModel.kv_geometry()` instead of Qwen
  defaults inside `serving/`;
- `0cf5671` records the direction and adds its import test;
- `548f2ca` plans 1b's kernels with comptime model dimensions.

Same machine and toolchain as below.

- **Exact equality.** `f10668d` changes no arithmetic. The baseline
  executables were the 1a candidates, whose 79 Mojo sources match `ce05db5`
  byte for byte under the same `uv.lock`. Against them, `f10668d` produced
  byte-identical outputs on the 1a equality gate recorded below: 4,170 files
  per side.
- **Lifecycle.** The model lifecycle study passed on Metal from a clean
  `548f2ca`, and again in device-sync mode. It now also rejects a pool whose
  KV heads differ from the model's, leaving every length unchanged.
- **Suite.** `uv run --locked llm-mojo validate` passed on `f10668d`'s tree
  with the boundary test present:
  - frozen oracle anchors, with the three large families verified from the
    shared store;
  - 268 Python tests, including the four boundary tests;
  - all 26 native test files plus the Unicode tokenizer run on Metal;
  - every benchmark smoke route.

  The Python tests passed again with the documentation changes.

### 1a on `3b23c65`, 2026-09-25

The branch's commits rebased onto `3b23c65` without conflicts; the
implementation commits `b969c47` and `c1a8776` recorded below became `65d183c`
and `6036300`. That base commit changed validation tooling, the shared store and
documentation but no Mojo source, and the branch's Mojo sources are
byte-identical to the tree gated below, so its equality results stand for this
baseline. `uv run --locked llm-mojo validate` passed again:

- all oracles match the frozen anchors, with the attention-sublayer, MLP and
  decoder-layer families verified from the shared store in 1.0, 0.8 and 2.9 s
  instead of regenerated;
- 264 Python tests;
- all 26 native test files plus the Unicode tokenizer run on Metal;
- every benchmark smoke route.

This worktree's 7.9 GB of local fixture copies gave way to links.

### 1a on `a144066`, 2026-09-24

Validated after rebasing onto `a144066`. Implementation commits `b969c47` (step
format, pool and tests) and `c1a8776` (model and clients on the execution-plan
engine). Apple M4 Pro (Mac16,7, 24 GiB), Metal, macOS 26.6.2 (25G83), Xcode 26.6
(17F113), Mojo 1.0.0 (`ed45d567`) and MAX 26.5.0 from `uv.lock`, which the
rebase did not change. `uv run --locked llm-mojo setup --offline` verified the
shared store's checkpoint and prepared model (196 BF16 tensors) and reported the
worktree ready.

**Exact equality with the baseline.** Executables were built separately from
`a144066` and from the rebased branch. On identical inputs, every compared
output is byte-identical:

| Route | Coverage | Compared |
| --- | --- | --- |
| Model driver, Fast | 37-token prompt then 8 decodes; 4 × 256-row chunks then 4 decodes; 240 + 16 rows (configuration 21) then 2 decodes; 2048 + 1920 rows then 2 decodes | stdout and 3,075 captured files: hidden states, appended K/V, all 48 caches with poisoned capacity, logits |
| Model driver, explicit | configurations 3, 2, 26 and 0 over 64, 16, 1 and 1 rows | stdout and 492 captured files |
| Chat driver | three turns, full-history replay, reset, failure recovery, interrupted turn | non-timing stdout and 298 files of caches and logits |
| Generation | 27-token prompt, 48 new tokens, Fast | text and every non-timing report event |
| Model benchmark `verify` | prefixes 64 and 1024: rewind, poisoned outputs, plain and observed steps | stdout and 148 snapshot files each |
| Chat CLI, piped | two turns, `/reset`, a third turn reaching the reply limit | transcript and all 478 report rows without their timing column |

**Other checks.**
- The model lifecycle driver passed on Metal in device-sync mode with the
  execution plans, including the new rejections: a position that disagrees
  with the pool, a mismatched pool geometry and a two-sequence batch. Each left
  lengths and counters unchanged.
- The seven native serving tests passed. The decode-route test passed on pools:
  the fused decode route and the baseline route still produce identical bytes,
  and the fused kernels leave the unfused scratch buffers untouched.
- The model benchmark compiled in its default, batch-support and profile builds.
- `uv run --locked llm-mojo validate` passed: frozen oracle anchors, 234 Python
  tests including the documentation link test, all 26 native test files plus
  the Unicode tokenizer run on Metal, and every benchmark smoke route.

**Timing sanity check, no claim.** Each run generated 128 tokens after a
1,176-token prompt, with per-step synchronization from the diagnostic report.
Sixteen runs formed four alternating blocks on an otherwise quiet machine.
Median decode steps were 8.23 ms for the baseline (runs 8.11–8.32 ms) and
8.00 ms for the candidate (7.96–8.40 ms); candidate/base block ratios were
0.974, 0.976, 1.010 and 0.976. Under the repository's decision rule this is
inconclusive: one block favors the baseline, and the 2.8% difference is below
the 5% floor. No regression is visible; 1c measures under controlled conditions.

Before the rebase, the same equality gate and suite passed against `a0b01da` on
2026-09-23; that timing check was inconclusive under heavy concurrent host load.
