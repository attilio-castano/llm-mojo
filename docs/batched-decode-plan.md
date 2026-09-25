# Batched decode plan

Baseline: `3b23c65`, `main` with the [serving plan](serving-plan.md), the
[September streamlining](history/streamlining-2026-09.md) (one-step setup,
explicit execution plans, current docs separated from history) and
[shared oracle fixtures](history/shared-fixtures-2026-09.md). The plan was first
written at `a0b01da`, and its work was rebased onto those changes. It
implements the serving plan's phase 1: several sequences decoding in one step,
with KV storage owned outside the model. The work proceeds in three steps, each
committed with its own gate:

1. **1a, step format and KV pool.** Route today's single-sequence calls
   through the step format and a KV pool, with no arithmetic change.
2. **1b, batched decode.** Add multi-sequence forms of the single-row decode
   kernels.
3. **1c, batch-size study.** Measure throughput and latency against batch size.

Approved on 2026-09-23 for local implementation of 1a: edits, builds, tests,
validation and incremental commits. Pushing, pull requests, toolchain upgrades
and numerical-contract changes need a separate decision. 1b was approved on
2026-09-25 on the same terms; 1c gets a detailed plan before its
implementation.

Status: 1a and 1b are complete; see the [validation record](#validation-record).
1c is next and gets a detailed plan before implementation.

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
  [dependency direction](cli.md#dependency-direction).
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
[shared store](development.md#shared-oracle-fixtures). Tests write captures
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

## 1c. Batch-size study (outline)

Extend the existing model benchmark with a batch axis, B in
{1, 2, 4, 8, 16, 32, 64}, and a projection row-tile axis, 4, 8 and 16, at
contexts 64, 1024 and 3968, plus one mixed-length batch. A 64-block pool of
full-context blocks is 3 GiB. Time complete steps,
from upload through per-sequence readback, with the existing four-block paired
procedure. Report:

- step latency;
- aggregate tokens per second;
- per-sequence token latency;
- a host step breakdown.

Capture traces for a subset to separate submission from GPU execution.

**Hypothesis, recorded before measurement.** In the
[runtime study](../studies/model_generation/runtime-measurements.csv), 16-row
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
([projection arrangements](../studies/model_generation/projection-arrangements.md)).
With tile 4, a step of B rows would then spend about ⌈B/4⌉ × 6.3 ms on
projections. That puts B = 4 near 4× today's throughput and both B = 16 and
B = 64 near 5×, unless tiles 8 and 16 keep their time per pass. In multi-row
prefill cells, tile 8 was 11–16% slower than tile 4 and tile 16 was 37–61%
slower ([decoder policies](../studies/decoder_layer/policies.md)); those cells
do not predict decode-shaped batches.

## Validation record

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
[dependency direction](cli.md#dependency-direction):
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
