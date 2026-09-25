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
and numerical-contract changes need a separate decision. 1b and 1c get detailed
plans before implementation.

Status: 1a is complete; see the [validation record](#validation-record). 1b is
next.

## 1a. Step format and KV pool

### Ownership

| Owner | After 1a |
| --- | --- |
| `serving/batch.mojo` | `StepBatch`: one step's tokens, positions, sequence offsets, KV lengths, block table, write slots and logit rows |
| `serving/kv_pool.mojo` | `KVPool`: one BF16 allocation for all KV storage, cache views per block and layer, and their logical lengths |
| `QwenModel` | weights, workspaces, logits and selection buffers, validity and submission counters; no KV storage or sequence length |
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

## 1b. Batched decode (outline)

A step carries S one-token sequences, each owning one full-context block of a
pool with S blocks. Projections, including the vocabulary head, use the
multi-row rowwise kernel for all S rows. These kernels gain multi-sequence
forms:

- fused QKV unpack, RoPE and KV append, indexed by positions and write slots;
- decode attention on the unsplit 32-simdgroup route, with a sequence index,
  per-sequence lengths and block bases;
- residual RMSNorm fusion, keeping each row's reduction;
- GPU argmax, with a winner and nonfinite flag per row;
- SiLU/multiply, as a multi-row fused kernel or an unfused path with identical
  rounding.

Buffer swapping between layers stays valid, because it exchanges buffer
owners rather than rows.

The gate: for S in {2, 3, 8, 16, 32} with mixed context lengths and
nonuniform data, each batched row's logits and appended KV bytes equal the same
sequence decoded alone at the same position. Guard rows around each block stay
untouched, and invalid batches are rejected without changing state.

**Test data.** Each batched row's reference is the same sequence decoded alone
on the existing single-row route, so 1b adds no oracle family. Native tests
build small models from the verified decoder fixtures, as the decode-route test
does, and run without the checkpoint; model-level checks use the prepared
checkpoint. Validation links those fixtures read-only from the
[shared store](development.md#shared-oracle-fixtures). Tests write captures
under `build/test_*` and records under `build/oracle_records/`, never into
`build/oracle_data/`. Leaving the generators' inputs untouched keeps validation's
oracle stage to seconds, so every 1b step can run the full suite; a new family
would need its own generator, anchors and store entry.

## 1c. Batch-size study (outline)

Extend the existing model benchmark with a batch axis: B in
{1, 2, 4, 8, 16, 32, 64} at contexts 64, 1024 and 3968, plus one mixed-length
batch. A 64-block pool of full-context blocks is 3 GiB. Time complete steps,
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

## Validation record

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
