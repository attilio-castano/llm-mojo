# Paged KV plan

Baseline: `6422f84`, `main` at `8fbe44f` (#30) with the documentation
re-baseline after serving phase 1. The plan implements phase 2 of the
[serving plan](serving-plan.md): KV blocks small enough to share and preempt,
a table of them for each sequence, attention that reads through the tables and
a block manager that owns them. The work proceeds in five steps, each
committed with its own gate:

1. **2a, paged kernels.** Every launch that reads or writes KV addresses the
   pool through a block table, with unchanged arithmetic.
2. **2b, block manager.** A host-side owner of blocks: states, allocation,
   release and invariant checks.
3. **2c, paged model.** The model steps sequences held in several blocks in
   every configuration, and its clients hold their sequences through the block
   manager.
4. **2d, translation-cost study.** Measure what translation costs at block
   sizes 32, 64 and 128, and whether head-major order within a block changes it.
5. **2e, adoption.** The selected block size becomes the default, and the
   documents that describe current behavior change.

Status: proposed on 2026-09-28 and approved on 2026-09-29 on phase 1's terms:
local implementation, builds, tests, validation and incremental commits, with
pushing, pull requests and toolchain upgrades needing a separate decision.
Paging changes addresses, not arithmetic, so no step changes the numerical
contract. The approval also moved
[keys and events to phase 4](#keys-and-events-move-to-phase-4). 2a and 2b
are complete, and 2c's gates have passed apart from its timing sanity check; see
the [validation record](#validation-record).

## What paging must preserve

KV memory is plentiful here (the serving plan's constraint 3): a full
4,096-token sequence takes 48 MiB, and phase 1 holds 64 of them in one 3 GiB
pool. Blocks serve the later phases. Phase 3 preempts and resumes sequences a
block at a time, and phase 4 shares full blocks between requests. Phase 2
therefore delivers two properties and measures a third:

- **Exactness.** A sequence's logits, tokens and K/V bytes are the same at every
  block size and in every configuration, alone or batched. Paging changes where
  a row lives, never the order in which a kernel visits or sums rows.
- **Ownership.** No step writes outside its own blocks, and the allocation
  invariants hold after every operation.
- **Cost.** Translation is measured against one block per sequence, today's
  layout. Phase 2 has no speed target, but translation must be close to free
  for the later phases to be worth their blocks.

## Where KV is touched today

Five launches read or write KV. Every other launch acts on `[rows, hidden]`
values and knows nothing about sequences.

| Launch | Configurations | Access |
| --- | --- | --- |
| `_fused_decode_qkv_batch` | 26 | writes each sequence's K and V row at its position in its one block |
| `_decode_sequences_kernel` | 26 | 32 SIMD groups per head read a sequence's keys, group g the rows t ≡ g (mod 32) |
| `_append` | 0, 2, 3, 20–22 | writes a chunk's K and V rows after the cached prefix of a contiguous per-layer view |
| `_mma_tuned`, routes 6 and 10 | 0; 2 and 3 | reads K/V in 32-row tiles starting at multiples of 32; split 8 partitions whole tiles |
| `_decode_kernel[32, 1, 1]`, routes 11 and 4 | 20–22; 0, 2 and 3 at one row | reads rows t ≡ g (mod 32) per query row |

Fast decodes with configuration 26 and prefills with configurations 0, 2, 3
and 21, so it runs all five launches. The baseline research mode runs
`_append` with routes 6 and 4, and the consistent mode `_append` with route 11.
The remaining attention routes appear only in the sublayer studies.

Three things tie each sequence to one block:
- `QwenModel.preflight` rejects tables of more than one block and pools whose
  block size differs from the model's capacity.
- The decode launches address `blocks[s]` alone, and the generic layer path
  hands every layer a contiguous `AttentionCache` view.
- `KVPool` creates a K and a V sub-buffer view for every block and layer. A
  3 GiB pool of 32-slot blocks would need 393,216 of them.

`StepBatch.validate` already accepts tables of several blocks: distinct blocks
per sequence, write slots that agree with the table, and no written block
shared between sequences.

## Design

### Ownership

| Owner | After phase 2 |
| --- | --- |
| `serving/kv_pool.mojo` | `KVPool`: one BF16 allocation, its geometry, block size and order within a block, and each block's count of written slots; no per-layer views |
| `serving/blocks.mojo`, new | `BlockManager`: block states, free blocks, each sequence's table and length, allocation at block boundaries, truncation, release and invariant checks; host only |
| `serving/batch.mojo` | `StepBatch.sequence` takes the sequence's table; validation is unchanged |
| `kernels/`, `layers/` | a paged form of each of the five launches, all using one address function; the contiguous forms stay for the sublayer and decoder studies |
| `QwenModel` | uploads positions and tables with the step's tokens, checks the tables against the pool's written counts, and launches the paged forms in every configuration |
| Chat, generate, drivers, benchmarks | a pool and a block manager; one block per sequence until 2e |

### Pool layout and translation

The pool keeps phase 1's block-major layout, in either of two orders within a
block:

```text
slot-major  Pool[block, layer, kv, slot, head, dim]   phase 1's order
head-major  Pool[block, layer, kv, head, slot, dim]
K_s[t, h, d] = Pool[T_s[t / BS], l, 0, t % BS, h, d]  slot-major, layer l
```

`T_s` is sequence s's row of the block table and `BS` the block size, a
multiple of 32. A pool can also hold each sequence in one block of any size
up to 4,096, as today. Every read visits K/V in 32-row tiles that start at
multiples of 32, or in residue classes modulo 32, so a tile never straddles two
blocks:
- prefill (`_mma_tuned`) translates once per tile;
- decode (`_decode_sequences_kernel`, `_decode_kernel`) walks a sequence block
  by block, translating once per block in each SIMD group, and visits each
  group's rows in today's increasing order;
- the two writes translate once per row.

One address function maps (block, layer, kv, slot, head) to an element offset
in either order, and every paged launch uses it. Each paged form keeps the
contiguous form's tiles, loads, sums and merges, so paged output equals
contiguous output by construction; 2a's tests check it byte for byte.

**Staging, considered and not adopted.** Copying a sequence's blocks into
contiguous scratch before each prefill attention would leave those kernels
untouched. It would also move up to 2 MiB per layer per chunk and add a launch
per layer, where translation reads one table entry per tile.

### Step upload and the model's checks

Phase 1 uploads each step through one host mapping of a step buffer. That
buffer now carries token IDs, positions and each sequence's table,
`[S, max_blocks]` int32, sized for 32-slot blocks: 64 sequences × 128 blocks ×
4 bytes = 32 KiB.

The pool counts written slots per block. Every step writes a row in all 24
layers, so one count per block replaces phase 1's 24 per-layer lengths. The
model accepts a sequence's step only if:
- every table entry is a block of the pool, and the table covers the
  sequence's new length;
- the blocks before the one holding its first position are full;
- that block holds exactly `position % BS` written slots, and later blocks in
  the table hold none;
- the block size is a multiple of 32, unless the table has one block.

It advances the counts after enqueuing the step. A rejected step changes
nothing, as in phase 1.

### Block manager

The block states follow the serving plan's, without Registered, which arrives
with sharing in phase 4:

| State | Meaning |
| --- | --- |
| Reset | free, no sequence |
| Partial | held by one sequence, not yet full |
| Complete | held by one sequence, every slot submitted |

The manager adds a sequence, reserves blocks to cover a new length, allocating
at each block boundary, commits the length after a forward, truncates by
releasing whole blocks past a new length, and releases a sequence tail first.
A table lists a sequence's blocks in position order, and the manager builds
each step's tables and write slots from them.

Invariants, checked after every operation in tests:
- free and held blocks partition the pool, and each held block appears in
  exactly one table entry, so every reference count is 0 or 1;
- a sequence's table has `ceil(length / BS)` blocks, all Complete except
  possibly the last;
- lengths agree with the pool's written counts;
- after every sequence is released, every block is Reset.

Phase 4 adds shared Registered blocks, reference counts above one and their
invariants.

### Captures and diagnostics

With one block per sequence, captures are unchanged. With smaller blocks, a
capture holds the sequence's K and V rows in position order through its last
block, so its first `length` rows compare byte for byte with a one-block
capture. Tests and diagnostics reach one block's K or V in one layer through a
sub-buffer made when needed.

### Keys and events move to phase 4

The serving plan put block keys, KV events and event replay in phase 2. With
this plan's approval on 2026-09-29 they moved to phase 4, with the Registered
state. In phase 2 no block outlives its sequence and none is reused. An event
log would record only allocation churn, which the invariants already check,
and nothing would read it. In phase 4 events describe reusable blocks, and
replaying them checks the prefix index they build. The serving plan's phase
table changed accordingly: phase 2's gate is paged equals one block, the
allocation invariants and write isolation, and phase 4 gains the logical event
replay.

## 2a. Paged kernels

The address function and the paged forms of the five launches, with no model
change. Each paged form's reference is its contiguous form on the same logical
inputs, so phase 2 adds no oracle family.

### Gate

- **Exactness.** Each paged form equals its contiguous form bit for bit on the
  same logical K/V:
  - block sizes 32, 64 and 128 and one block per sequence, in both orders;
  - scattered, unsorted tables;
  - lengths 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 4,095 and 4,096;
  - decode at S ∈ {1, 2, 3, 8, 16, 32} with mixed lengths;
  - prefill chunks of 1, 15, 16, 17, 64, 255 and 256 rows through routes 6,
    10 and 11, after prefixes that end inside a block and at its edge.
- **Isolation.** Every slot the launch does not address, including the unwritten
  slots of a sequence's last block, holds a NaN poison that stays in place. A
  stray read would reach an output; a stray write would replace poison.
- **Unchanged.** The contiguous forms and their tests.

## 2b. Block manager

`serving/blocks.mojo`, host only.

### Gate

- **Properties.** Seeded random sequences of add, reserve, commit, truncate and
  release, over pools of 1 to 512 blocks and block sizes from 32 to 4,096,
  check every invariant after every operation. A seeded permutation of the free
  list scatters the tables, as a busy pool would.
- **Rejections.** Reserving more blocks than are free or a length past the
  per-sequence limit, committing past a reservation, and naming an unknown
  sequence each raise with state unchanged.

## 2c. Paged model

The pool counts written slots instead of holding views. The step buffer
carries tables, the model checks them, and every configuration launches the
paged forms. `StepBatch.sequence` takes a table. Chat, generate, the drivers,
the model benchmark and the batch validation build their tables with the block
manager, one block per sequence by default; the drivers and the batch
validation take a block size and order.

### Gate

- **Unchanged at one block.** Executables built from `6422f84` and from 2c
  produce byte-identical outputs on phase 1's 1a equality gate, which its
  [validation record](history/batched-decode-plan.md#validation-record)
  describes: the model driver's Fast histories and explicit configurations,
  the chat driver, generation, the benchmark's `verify` and the piped chat CLI.
- **Paged equals one block.** The same model and chat driver histories run at
  block sizes 32, 64 and 128 in both orders: full and chunked prefill through
  configurations 0, 2, 3 and 21, one-token decode and three chat turns. Logits,
  tokens, hidden states and every written K/V row equal the one-block run's
  byte for byte, and unwritten slots and unused blocks keep their poison.
- **Batched equals solo across blocks.** On three fixture layers, 2, 3, 8, 16
  and 32 sequences in 32-slot blocks decode 12 steps together and alone,
  crossing the boundaries at 32 and 64. On the checkpoint,
  `validation.model batch` decodes its eight conversations in 64-slot blocks
  allocated in interleaved order, so every table is scattered.
- **Write isolation.** In both batched tests, a step changes only its own write
  slots; every other byte of the pool stays the same.
- **Rejections.** A step that breaks any of the model's checks leaves counts,
  lengths and counters unchanged.
- **Lifecycle on the checkpoint.** `validation.model lifecycle` passes from a
  clean build. `llm-mojo validate` runs no checkpoint driver, so it needs its
  own run.
- **Suite.** `uv run --locked llm-mojo validate` passes. Decode parity and the
  route test keep their checks.
- **Timing sanity check, no claim.** Alternating generation runs of `6422f84`'s
  and 2c's executables at one block per sequence, as in 1a and 1b.

## 2d. Translation-cost study

**Question.** What does address translation cost in decode and prefill at
block sizes 32, 64 and 128, against one block per sequence, and does head-major
order within a block change it?

### Matrix

- **Layouts.** The control is one block per sequence in slot-major order,
  today's layout through the paged forms. The six candidates are 32, 64 and
  128 slots in each order. One binary runs all seven; a layout is a property
  of the pool, not of the build.
- **Decode workloads.** The batch-size matrix's 22: B from 1 to 64 at 64, 1,024
  and 3,968 cached tokens, and the mixed batch of 32.
- **Prefill workloads.** Thirteen one-sequence chunks under Fast's plan: the
  runtime study's eleven cells, which run configurations 2, 3 and 21, and
  256-row chunks after 256 and 2,816 cached tokens, which run configuration 0
  like most chat chunks.
- **Procedure.** The four-block paired procedure: the control against itself
  for calibration and each candidate against the control, with ten warmups and
  ten samples per arm, and blocks 2 and 3 reversing the order. That is 12,320
  decode and 7,280 prefill samples.
- **Timed intervals.** Decode: from building the step batch, the manager's
  tables included, through `greedy_tokens` readback, as in phase 1. Prefill:
  the runtime study's boundary, a resident forward from the token upload to
  device synchronization.
- **Traces.** Metal System Traces at B = 64 and 3,968 cached tokens, where
  attention's share is largest, for the control and the three slot-major
  sizes, two repeats each. They give attention's and the KV writes' active time
  per step.

### Setup

One pool per layout would need 21 GiB at 3,968 cached tokens, on a 24 GiB
machine. Instead:
- One working pool of 262,144 slots, the 3 GiB of phase 1's pool, serves every
  decode layout: 64 blocks of 4,096 slots, 2,048 of 128, 4,096 of 64 or 8,192
  of 32.
- At setup the frozen history is prefilled once per layout into a
  one-sequence pool of 48 MiB. Every layout's written K/V must equal the
  control's byte for byte before measurement starts, which checks exactness on
  the real model.
- Before each arm, the control's included, the working pool is rebuilt in that
  arm's layout outside the timed interval: tables come from a seeded
  permutation of the pool's blocks, and each sequence's blocks are copied from
  the layout's one-sequence pool.
- Prefill workloads run in their own processes, one per block, on the
  one-sequence pools.
- Every decode sample must reproduce its reference tokens with 245 launches,
  and every layout the control's tokens.
- Before measurement, the measured commit passes 2c's lifecycle check.

### Reported quantities

For each workload and layout: step or forward latency, its paired ratio
against the control under the method's rule with the calibration noise floor,
and aggregate tokens per second for decode. From the traces: attention's and
the KV writes' active time per step against the control. The study
`studies/model_generation/paged-kv.md` explains them from one lossless
archive, `paged-kv.json.gz`, which its replay command rebuilds without a GPU.

### Decision

Frozen before measurement:
- **Qualifying.** A layout qualifies if it is a regression in none of the 35
  workloads.
- **Selecting.** The smallest qualifying block size. At that size, slot-major,
  unless head-major also qualifies and is a gain in at least one workload.
- **Confirming.** A fresh four-block run of the selected layout against the
  control over all 35 workloads, with its own calibration and the same rule.
  If it fails, no other layout is tried.
- **Single-sequence check.** Sixteen generation runs in four alternating
  blocks, 128 tokens after a 1,176-token prompt as in 1e, compare `6422f84`'s
  generate executable with one built to use the selected layout. If the
  selected layout is slower in all four blocks by a median of more than 5%,
  adoption stops; if it is slower in all four by less, the decision comes back
  to you.
- **Otherwise.** One block per sequence stays the default, the study records
  why, and phase 3 starts with that question open.

### Hypothesis, recorded before measurement

- **Prefill: no resolvable cost.** A thread reads one table entry per 32-row
  tile, beside its 32 K and V loads for that tile.
- **Decode: a cost only at 32 slots, and only where attention dominates.** For
  each key row, every lane loads two K and two V values. A SIMD group reads rows
  32 apart, so it enters a new block with every row at 32 slots, every second
  row at 64 and every fourth at 128, and each block costs one table read. If
  attention time grew with its loads, that would add at most 25%, 12.5% and
  6.25% to it. In 1e's traces at
  B = 64 and 1,024 cached tokens, attention took 20.7 of 66.0 ms of active GPU
  time; extrapolated to 3,968 cached tokens, it is about 80 of 125 ms. The
  bounds at B = 64 are then 8%, 4% and 2% of a step at 1,024 cached tokens, and
  16%, 8% and 4% at 3,968. The table read is the same address for the whole
  SIMD group and stays in cache, so the cost should be well below the bound.
  Prediction: 64 and 128 qualify everywhere, 32 may regress at the longest
  context and largest batches, and 64 is selected.
- **Head-major: no resolvable difference.** At B = 64 and 3,968 cached tokens,
  attention reads about 3.1 GB of distinct K/V per step, about a seventh of
  the nominal memory bandwidth over 80 ms. Its speed is not limited by where
  consecutive rows sit.
- **Host: negligible.** At B = 64 and 3,968 cached tokens with 32-slot blocks,
  the manager fills 8,000 table entries per step and the upload grows to
  32 KiB, beside B = 64 steps of 47 ms or more.

## 2e. Adoption

If 2d selects and confirms a layout and the single-sequence check passes:
- `models/qwen2/plan.mojo` gains the default block size and order beside the
  decode arrangement; chat, generate and the batch validation use them, and
  2c's gates run again at that layout.
- The documents that describe current behavior change: the block-table gather
  in [layouts.md](layouts.md), KV storage in the [model contract](model.md),
  the pool in the [runtime guide](generation.md), [chat](chat.md) and the
  [walkthrough](walkthrough.md), `serving/` in the
  [code ownership](cli.md#code-ownership) table, and phase 2's status in the
  serving plan.

## Risks

- **Contraction.** Each paged form is compiled separately, and a new loop shape
  could change whether the compiler fuses a multiply and an add. 2a's tests
  catch any difference; explicit `fma` control fixed the same problem in RoPE.
- **Registers.** Translation keeps a table row and a block base per thread.
  Lower occupancy would show in the traces as longer attention.
- **What the control measures.** The control runs the paged forms with one
  block, so the matrix measures translation, not phase 2 against phase 1. 2c's
  sanity check and 2d's single-sequence check compare with phase 1's
  executable.
- **Rebuilding the working pool.** Copying up to 3 GiB before each arm leaves
  cold caches; the ten warmups absorb it, and every arm, the control's
  included, starts the same way.
- **Conditions.** Background load and trace attribution are handled as in
  phase 1.

## Not in phase 2

- Block keys, KV events, Registered blocks and prefix sharing (phase 4).
- Preemption, scheduling, steps that mix decode and prefill, and prefill of
  several sequences in one step (phase 3).
- Blocks smaller than 32 slots, a block size per request, KV quantization and
  split-K decode.

## Validation record

### 2c review fix on `f87e989`, 2026-10-03

Codex's review of #32 found that the model driver's lifecycle check had failed
since `1cb1942`. It expected the model, whose context is 4 tokens, to reject a
pool of one 8-slot block. That was phase 1's rule that a block equals the
model's context, which 2c dropped on purpose (see 2c's deviations). Nothing
had run the check: `llm-mojo validate` starts no checkpoint driver, and 2c's
gate used phase 1's equality scenarios, which leave it out, although
`1cb1942` edited it for the new pool.

`f87e989` replaces that case with the rule that took its place: a pool's
blocks must hold the step. A three-token step laid out for 4-slot blocks, valid
on its own, goes to a pool of one 2-slot block. The model must reject it and
leave the block's written count at zero.

- **Lifecycle.** `validation.model lifecycle` passed from a clean `f87e989`,
  and the same source passed in device-sync mode. `0c438eb`'s driver fails at
  the old case.
- **Deliberate fault.** With the rule that a sequence must fit its table
  removed from `StepBatch.validate`, the driver aborts in the new case: the
  model's preflight reads a second table entry that the one-block sequence
  does not have. The fault was reverted.
- **Suite.** `uv run --locked llm-mojo validate` passed in 33 minutes, the
  first full run since the Metal toolchain was downloaded again; `metal`
  reports the same version, 32023.883.

2c's gate now includes the lifecycle check, so 2e runs it again with 2c's
other gates, and 2d's measured commit must pass it first.

### 2c on `1c88501`, 2026-09-29

Three commits implement 2c, each validated before it was committed:

| Commit | Change |
| --- | --- |
| `a8039dc` | the model launches 2a's paged kernels in every configuration, still one block per sequence; phase 1's one-block kernels are deleted |
| `1cb1942` | the pool counts written slots per block instead of holding per-layer views; the model steps tables of several blocks in either order within a block |
| `1c88501` | chat, generate and the model and chat drivers get their tables from the block manager; the drivers and the batch validation take a block size and an order |

- **Unchanged at one block.** Executables built from `6422f84` and from each
  commit gave byte-identical outputs on phase 1's equality gate, 5,283 files
  per side:
  - the model driver: four Fast histories, among them 240 + 16 rows through
    configuration 21 and 2,048 + 1,920 rows, configurations 3, 2, 26 and 0
    explicitly, and the baseline and consistent modes;
  - the chat driver, generation, the benchmark's `verify` at 64 and 1,024
    cached tokens, and the piped chat CLI.
- **Paged equals one block.** With the last commit's build, the model and
  chat driver scenarios ran at 32, 64 and 128 slots in both orders: 4,980
  files per layout, identical to the one-block run. In the 1,824 cache captures
  of each layout, the rows through each call's length are identical and every
  row after them still holds the fill.
- **Batched equals solo across blocks.** `tests/test_decode_batch.mojo`
  decodes eight fixture sequences in 32-, 64- and 128-slot blocks in both
  orders. They are prefilled through configurations 0, 2, 3 and 21 and cross
  the 32 and 64 boundaries. Every step and every written row equals one block
  per sequence decoded alone. On the checkpoint, the model driver's batch mode
  decoded eight conversations of 11 to 3,301 prompt tokens for 16 steps, with
  64-slot blocks allocated to the conversations in turn, slot-major and
  head-major, and with 32-slot blocks. Every token, all 151,936 logits per row
  and every written K/V row equal decoding alone. `validation.model batch --block-size 64`
  ran from a clean `1c88501` and wrote its receipt.
- **Write isolation.** In both batched tests, every slot that no step wrote
  keeps its poison or fill.
- **Rejections.** Each of these is rejected with written counts, lengths and
  counters unchanged:
  - a table of several blocks whose size is not a multiple of 32;
  - a table wider than the model's context in 32-slot blocks;
  - a position past the model's context;
  - a position that disagrees with its block's written slots.
- **Suite.** `uv run --locked llm-mojo validate` passed on each commit's tree,
  in 34, 37 and 33 minutes: frozen oracle anchors, 283 Python tests, all 28
  native test files plus the Unicode tokenizer run on Metal, and every benchmark
  smoke route.
- **Timing sanity check.** Not yet run: it is a measurement and waits for a
  quiet machine.

**Deviations from the plan.**
- Phase 1's one-block kernels, `_decode_sequences_kernel` and
  `_fused_decode_qkv_batch`, and the contiguous `validate_decoder_configuration`
  are deleted, since nothing calls them (your decision on 2026-09-29).
  `tests/test_paged_kv.mojo` now checks the paged fused write against the
  unfused path.
- `StepBatch.sequence` takes the sequence's table; a one-block overload was
  ambiguous with list literals, and its callers pass `[block]`.
- The model accepts any block size for one-block tables, not only the model's
  capacity, and a block may be larger than the context.
- The model benchmark keeps one block per sequence with direct tables; 2d
  gives it the study's layouts.

### 2b on `fdce177`, 2026-09-29

`fdce177` adds `serving/blocks.mojo`. `BlockManager` keeps each sequence's
table, its committed length and the length its blocks are reserved for, and
each block's state. A step reserves room for its new positions, which
allocates a block at each block boundary, and commits the new length once its
writes are enqueued. Truncation and release free whole blocks, last block
first. A seed permutes the free list, and `check()` verifies the invariants.

- **Properties.** `tests/test_block_manager.mojo` runs seeded random
  operations, 54,500 in all, with every invariant checked after each:
  - pools of 1, 2, 7, 64 and 512 blocks at block sizes 32, 64, 128 and 4,096,
    from scattered and in-order free lists;
  - a 40-block pool whose 100-position limit is not a whole number of blocks.

  The test counts each kind of operation and requires at least 500 of each. It
  saw 7,101 adds, 3,814 reservations that allocated blocks, 6,774 rejected
  reservations, 6,447 commits, 6,229 truncations, 6,666 releases, 3,920
  reserve-and-commit steps and 6,391 other rejections. After every run,
  releasing all sequences left every block free and Reset. Scenario tests
  cover allocation at a block's 33rd position, interleaved growth, truncation
  inside a Complete block and tail-first release. A seeded 64-block table is a
  permutation of the pool.
- **Rejections.** Each of these raises and leaves the manager's state
  unchanged:
  - invalid geometry;
  - a reservation that needs more blocks than are free, passes the
    per-sequence limit or falls below the sequence's length;
  - a commit past the reservation or below the length;
  - a truncation past the length or below zero;
  - a sequence never added, negative or released.
- **Sensitivity.** Three deliberate faults, each reverted, failed the tests:
  - truncation that left freed blocks Partial;
  - a commit that marked a partly written block Complete;
  - a reservation one block short. The scatter test hit it as an out-of-range
    index and now checks the table's length first.
- **Suite.** `uv run --locked llm-mojo validate` passed on 2b's tree before the
  commit, in 32 minutes: frozen oracle anchors, 283 Python tests, all 28 native
  test files and the Unicode tokenizer run on Metal, and every benchmark smoke
  route.

**Deviations from the plan.** The plan states that a sequence's table has
`ceil(length / BS)` blocks. That holds between steps. While a step's blocks are
reserved but not committed, the table covers the reserved length, and those
blocks are Partial; `check()` verifies the table against the reservation. The
invariant that lengths agree with the pool's written counts needs the pool, so
2c checks it.

### 2a on `fe9c081`, 2026-09-29

`fe9c081` adds `kernels/paged_kv.mojo`'s address function and the paged forms
of the five launches. Route 4's arithmetic serves batched decode and routes 11
and 4, so four kernels cover the five launches:

| Paged form | Equals, bit for bit |
| --- | --- |
| `enqueue_paged_attention_g32_apple_gpu` | route 4 for each batched decode row, and route 11 for a sequence's query rows |
| `enqueue_paged_attention_prefill_apple_gpu` | route 6 |
| `enqueue_paged_attention_prefill_split_apple_gpu[8]` | route 10 |
| `enqueue_fused_decode_qkv_paged` | phase 1's `enqueue_fused_decode_qkv_batch` |
| `enqueue_append_paged` | `_append` |

- **Exactness.** `tests/test_paged_kv.mojo` lays each sequence's rows through
  scattered tables in layer 1 of 2, with its own address formula, at block
  sizes 32, 64 and 128 and one block of 4,096 slots, in both orders:
  - decode: 1, 2, 3, 8, 16 and 32 sequences of 1 to 4,096 keys, including
    31–33, 63–65, 127–129, 4,095 and 4,096;
  - prefill: chunks of 1, 15, 16, 17, 64, 255 and 256 rows after a prefix
    ending at a block edge, inside a block, and filling the context.

  Every output matches its contiguous form. The test also pins the address
  function to both documented layouts and to phase 1's pool offsets.
- **Isolation.** Unwritten slots, those of each sequence's last block
  included, and unused blocks hold a NaN poison. After every write, exactly the
  written K/V rows differ from it, and each equals the contiguous write.
- **Sensitivity.** Three deliberate faults, each reverted, failed the test:
  - reading every key from a sequence's first block failed at 32-slot blocks;
  - starting every prefill tile at its block's first slot failed at 64;
  - writing each appended V row one slot late failed at 32.
- **Unchanged.** The contiguous forms' code is untouched.
- **Suite.** `uv run --locked llm-mojo validate` passed on 2a's tree before the
  commit, in 35 minutes:
  - all oracles match the frozen anchors, with the three large families
    verified from the shared store;
  - 283 Python tests;
  - all 27 native test files, `test_paged_kv.mojo` among them, plus the
    Unicode tokenizer run on Metal;
  - every benchmark smoke route.

**Deviation from the plan.** The plan listed the batched decode kernel and
routes 11 and 4 as separate launches to page. They share route 4's arithmetic,
so one paged kernel serves all three, given each query row's position and each
sequence's table row.

### Baseline on `6422f84`, 2026-09-28

Apple M4 Pro (Mac16,7, 24 GiB), Metal, macOS 26.6.2 (25G83), Xcode 26.6
(17F113), Mojo 1.0.0 and MAX 26.5.0 from `uv.lock`. `uv run --locked llm-mojo
validate` passed from a clean `6422f84` in 35 minutes:
- all oracles match the frozen anchors, with the attention-sublayer, MLP and
  decoder-layer families verified from the shared store in 0.7, 0.5 and 1.9 s;
- 283 Python tests;
- all 26 native test files plus the Unicode tokenizer run on Metal;
- every benchmark smoke route.

This is the first full validation since `39fecba`. Phase 1's last three
commits, `d263f99`, `6b222ee` and `ebe42fc` (#29), had passed the Python suite
and only the native suites they changed.
