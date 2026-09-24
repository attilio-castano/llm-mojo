# Batched decode plan

Baseline: `a0b01da`, the merged [serving plan](serving-plan.md). This plan
implements its phase 1: several sequences decoding in one step, with KV storage
owned outside the model. The work proceeds in three steps, each committed with
its own gate:

1. **1a, step format and KV pool.** Route today's single-sequence calls
   through the step format and a KV pool, with no arithmetic change.
2. **1b, batched decode.** Add multi-sequence forms of the single-row decode
   kernels.
3. **1c, batch-size study.** Measure throughput and latency against batch size.

Approved on 2026-09-23 for local implementation of 1a: edits, builds, tests,
validation and incremental commits. Pushing, pull requests, toolchain upgrades
and numerical-contract changes need a separate decision. 1b and 1c get detailed
plans before implementation.

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
- the first position differs from the length of any of the block's 24 layer
  views.

It then runs the existing layer code on those views. Benchmark rewinds use
`KVPool.truncate`, which only shortens logical lengths.

### Gate

- **Exact equality with the baseline.** Build the model and chat drivers from
  `a0b01da` and from this change. On the pinned checkpoint, run fixed
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

Recorded as each step completes.
