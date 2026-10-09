# Serving engine plan

Proposed on 2026-09-23 from baseline `edb610a`. Phase 1, batched decode, is
complete and merged as `132dc08` (#29); its
[plan and validation record](history/batched-decode-plan.md) are history now.
Phase 2, a paged KV cache, followed the
[paged KV plan](paged-kv-plan.md), approved on 2026-09-29, which also moved
block keys and KV events to phase 4. Its paged kernels, block manager and paged
model are merged as `a84ad34` (#32). Its translation-cost study found that
small blocks made decode attention pay for every block; after a fix to that
kernel, a rerun selected and confirmed 32-slot blocks, the default since
2026-10-05. Phase 2 is complete.
Phase 3's synchronous core and bounded load studies are validated on
`codex/engine-core`, authorized from `cb2416a` on 2026-10-08. Its
[engine study](../studies/model_generation/engine-core.md) retains readiness,
correctness, fitted-budget evaluation and the separate asynchronous API probe.
The initial runner uses reference configuration 27 in every arm, so scheduling
has one numerical route. Existing chat and generation remain Fast; a measured
Fast engine route is a separate optimization decision.
The original measurements bind clean implementation `b18563b`. The optional
[lifetime reservation follow-up](../studies/model_generation/engine-core.md#lifetime-reservation-admission-bounded-successor-study)
binds `b81ea6c` and eliminates replay on the frozen pressure trace by delaying
admission until declared cache growth fits. Incremental admission remains the
default and its original pressure regression is retained.
The follow-up records queueing and token-latency tradeoffs separately.
The [144-run operating-range comparison](../studies/model_generation/engine-core.md#admission-operating-range-retained-bounded-results)
binds `2af0933`: reservation removes replay at 40 blocks, while larger declared
output limits tie up more unused capacity and can delay FIFO admission.
Adequate-capacity paired speed remains inconclusive; incremental stays default.
Fitted budgeting remains optional, without a client latency guarantee. The
[fixed-workspace budget and optional Fast contracts](../studies/model_generation/engine-core.md#fixed-workspace-budget-and-optional-fast-contracts)
now have separate retained measurements from clean `01d8be4`. The
[116-run budget comparison](../studies/model_generation/engine-core.md#fixed-workspace-budget-retained-bounded-results)
retains fixed-256: smaller chunks reduced long offline gaps but added steps and
TTFT, and the 25 ms prediction target was exceeded by measured executes. The
[qualified 12-run Fast comparison](../studies/model_generation/engine-core.md#optional-fast-engine-retained-bounded-results)
matched full histories and ordered work, but its speed verdict is inconclusive
within the 5% noise floor. Reference-27 remains the engine default. Both studies
retain independent local canonical retrieval/CPU replay and preserved originals.
The optional [`chat --engine` adapter](chat.md#optional-engine-terminal-chat) now
connects one terminal conversation to the synchronous reference core, recomputing
its full token history each turn. Its checkpoint lifecycle acceptance is retained.
Full asynchronous stepping, Fast default promotion, prefix caching and the
multi-request frontend/process transport remain separate work.
Each phase is approved separately and records its own validation, like the
existing plans. The [project direction](project.md) lists this as a follow-up
track.

## Why this track

The completed milestone serves one conversation at a time: one token history,
one set of 24 KV caches and greedy decoding in a persistent session. Its
kernels, numerics and measurements are the foundation for everything below.

Production inference engines spend most of their design effort elsewhere:
deciding which requests share each GPU step, who owns KV memory, what happens
under memory pressure and failure, and how latency distributions respond to
load. vLLM, SGLang and TensorRT-LLM solve these inside an engine; NVIDIA Dynamo
coordinates them across engines and machines. This track builds those
responsibilities on one Mac with the same standards as the kernel work:
explicit ownership, exact invariants, declared numerical policies and
reproducible measurements.

## Scope

In order: batched decode, a paged KV cache, continuous batching with chunked
prefill, prefix caching, a frontend/engine process split with an
OpenAI-compatible HTTP interface and crash recovery, and an SSD KV tier.
Several engine processes behind a KV-aware router are an optional last phase.

The target remains Qwen2.5-0.5B-Instruct in BF16 with greedy lowest-ID
selection, at most 4,096 tokens per request, on Apple M4 Pro / Metal. Sampling
parameters are part of the request schema but are rejected explicitly until a
separate track implements them. Quantization, speculative decoding, other
models, multiple GPUs and disaggregated prefill/decode are out of scope.

## Measured constraints

Existing results shape the design:

1. **Step cost is mostly per launch.** A decode token issues about 245 compute
   launches, each in its own Metal command buffer, and spends about 7 ms inside
   MAX's enqueue calls ([runtime enqueue](../studies/model_generation/runtime-enqueue.md),
   [batching feasibility](../studies/model_generation/batch-support.md)).
   Adding sequences to a step adds GPU work but not launches. Launches per step
   must therefore scale with layers, never with sequences.
2. **The host waits at the end of every step.** Greedy readback waited about
   1.4–2.4 ms per token with the default projections
   ([projection scheduling](../studies/model_generation/projection-scheduling.md)).
3. **KV memory is plentiful for this model.** BF16 KV costs 12,288 bytes per
   token, 48 MiB for a full 4,096-token sequence
   ([model contract](model.md#current-runtime-boundary)). Capacity alone does
   not motivate paging here; prefix sharing and granular preemption do.
   Studies set the KV pool size explicitly to create memory pressure.
4. **Memory is unified.** Moving KV to host memory moves nothing, so
   preemption means recomputation. Local NVMe is the only slower tier with
   distinct capacity.
5. **The host thread is scarce.** Host submission limits short-context decode.
   Tokenization, detokenization, HTTP and bookkeeping compete with kernel
   submission unless they run elsewhere, so every step records where its host
   time goes.

## Architecture

```text
client ── HTTP/SSE ──► frontend process (Mojo)
                         HTTP adapter → chat template → tokenizer → RequestTracker
                         RequestTracker: authoritative token histories,
                         text streaming, replay after an engine restart
                               │  token protocol over a Unix socket:
                               │  add · abort · tokens · finish · reject
                               ▼
                       engine process (Mojo, owns the GPU)
                         EngineCore: inbox → Scheduler → KVCacheManager
                                     → ModelRunner → outbox
                         event stream: KV events · step records
                               │  StepBatch
                               ▼
                       QwenModel.forward(StepBatch)
                         weights and workspaces, no sequence state between steps
                               │
                               ▼
                       Metal, one ordered stream
```

The Python launcher starts and supervises both processes, as it already
prepares assets and launches native executables.

| Component | Owns | Does not own |
| --- | --- | --- |
| HTTP adapter | connections, OpenAI JSON, SSE framing | token IDs, request state |
| RequestTracker | request IDs, token histories, stop strings, text streams, replay | KV, GPU work |
| EngineCore | step loop, inbox/outbox, rebuildable per-request engine state | text |
| Scheduler | which requests run this step, with how many tokens | memory, execution |
| KVCacheManager | KV pool allocation, block tables, block states, prefix index, events, SSD tier | scheduling decisions |
| ModelRunner | StepBatch upload, forward, token selection, readback, step timing | policy |
| QwenModel | weights, workspaces, kernel dispatch | sequence state between steps |

`QwenModel` once owned 24 caches and one `length`; phase 1a moved KV storage and
lengths into a caller-owned `KVPool`, and `ChatSession` still owns one history.
In this design, the model becomes stateless between steps, the
KVCacheManager owns all KV storage, each request owns its history, and chat
becomes an engine client whose turns reuse earlier turns through prefix hits.

### Failure domains decide the process split

The engine can fail: a Metal error already invalidates today's session. The
frontend therefore holds every in-flight request's authoritative token history
and never shares a process with the GPU. After an engine exit, the supervisor
restarts it and the frontend resubmits each in-flight request as its prompt
plus the tokens generated so far. The client stream continues with no lost or
duplicated tokens. Replay recomputes the generated tokens as prompt rows, so the
continuation can differ numerically from an uninterrupted run; that difference
is a recorded diagnostic, not a failure.

This requires that all per-request engine state can be rebuilt from
(parameters, token history). KV contents are a cache, never the source of
truth. A future feature whose state tokens cannot reconstruct, such as sampler
or grammar state, must define its own recovery first.

### Three planes

- **Request plane:** the token protocol, carrying requests, aborts and token
  streams.
- **Event plane:** KV events and per-step records, as versioned line-delimited
  records.
- **Control plane:** configuration, readiness, health and restart, owned by
  the launcher.

## Interfaces

### Token protocol

The engine accepts and returns token IDs only. Text, chat templates and stop
strings belong to the frontend. Generated token IDs are authoritative, as in
today's chat.

```text
frontend → engine
  Add    { request_id, prompt_ids, max_new_tokens, stop_ids, cache_salt,
           resumed_tokens, arrival_ns }
  Abort  { request_id }
engine → frontend
  Ready  { model_card }
  Tokens { request_id, token_ids, cached_prompt_tokens }
  Finish { request_id, reason: stop | length | abort | error, usage }
  Reject { request_id, reason }
```

Messages for one request are ordered. Abort is idempotent. Unsupported
parameters produce `Reject`, never silent defaults, and rejection leaves
engine state unchanged. `resumed_tokens` marks a replayed suffix so accounting
and latency records distinguish recovery from ordinary prefill.

Mojo 1.0's standard library provides `subprocess` and `os` but no socket
module. The transport uses POSIX sockets through `external_call`, as the
runtime already does for clocks and signals.

### Model card

At startup the engine publishes the pinned model revision and hashes,
tokenizer table hashes, vocabulary size, stop IDs, per-request token limit,
block size, pool capacity, numerical policy, source commit and executable
hash. The frontend refuses to serve if its tokenizer tables differ. Every
retained serving measurement records the card.

### StepBatch

`QwenModel.forward(ids)` becomes `QwenModel.forward(batch: StepBatch)`. A step
concatenates the scheduled tokens of all its sequences, decode sequences first:

| Field | Shape | Meaning |
| --- | --- | --- |
| `token_ids` | [N] | scheduled tokens, sequences concatenated |
| `positions` | [N] | absolute position of each token; drives RoPE |
| `query_start` | [S+1] | offset of each sequence's tokens |
| `decode_count` | scalar | leading sequences with exactly one token |
| `seq_lens` | [S] | KV length of each sequence after this step |
| `block_table` | [S, max_blocks] | physical KV blocks of each sequence |
| `slot_mapping` | [N] | physical KV slot written by each token |
| `logits_rows` | [S′] | rows whose logits are needed |

Embedding, normalization, projections, SwiGLU and residuals act per token on
`[N, 896]` without sequence knowledge. RoPE and KV writes use `positions` and
`slot_mapping`. In each layer, attention issues one decode launch for the
leading decode sequences and one prefill launch for the remaining chunk. Launch
count depends on the layer count and on whether a step contains decode or
prefill work, never on S. The vocabulary projection reads only `logits_rows`.

GPU workspaces are sized once from declared physical capacity and maximum
sequence count; the scheduler budget can use fewer rows within that capacity.
The synchronous core currently allocates host metadata and event lists per step;
its measurements include that bookkeeping. Avoiding those allocations is a
later optimization. A mixed call contains singleton decode sequences and at
most one multi-row prefill sequence.

### ModelRunner

The implemented `ModelRunner` interface has `execute(batch, kv) -> List[Int]`
and `now_ns()`. Execution returns tokens in `logits_rows` order and completes
before cache ownership can change. `EngineCore` records timings in `EngineStep`.
Two implementations share this interface:

- `QwenRunner` executes synchronously on Metal with reference configuration 27
  by default. The optional Fast study selects 26 for eligible steps and 27
  otherwise; its measured result leaves the default unchanged.
- `SimulatedRunner` advances a virtual clock using configured synthetic fixed,
  per-token and per-position costs, and returns a deterministic token script.
  It does not consume the fitted hardware policy.

The scheduler, KV manager and EngineCore are the same code in both. The
simulator explores policies quickly; hardware measurements confirm them.

### KV events and step records

KV events describe what is cached, not where. Each carries
`{event_id, step_id, kind, tier, block_hash, parent_hash}`, and `stored` events
also carry the block's token IDs. The three kinds, `stored`, `removed` and
`cleared`, each apply to one tier: writing a block to SSD is `stored` in the
disk tier, and dropping it from memory is `removed` in the memory tier. IDs
increase by one. A consumer that sees a gap discards its view and requests a
snapshot.

Replaying the log must reconstruct the logical index, the set of
(key, parent key, tier), exactly. Recomputing each stored key from its parent,
token IDs and salt must reproduce it. Physical placement, reference counts and
eviction order stay private; the step invariants check them.

Each step emits one versioned record:

```text
step_id, schema_version, begin_ns
schedule_ns, build_ns, upload_ns, submit_ns, wait_ns, postprocess_ns
decode_seqs, prefill_seqs, prefill_tokens, total_tokens, attended_positions
blocks_free, blocks_active, blocks_cached, blocks_loading
admitted, preempted, finished, aborted, waiting
cache_hit_tokens, predicted_step_ns
```

Step records explain results, fit the step-time model and expose drift between
predicted and measured step time.

## Engine core

### Request lifecycle

```text
WAITING → [LOADING_KV →] PREFILL → DECODE → FINISHED(stop | length | abort | error)
PREFILL or DECODE → WAITING on preemption: history kept, blocks released to the cache
any state → FINISHED(abort) at the next step boundary
```

`LOADING_KV` exists only with the SSD tier. A preempted request returns to the
front of the waiting queue. In phase 3 its blocks are released to Reset and its
retained prompt plus delivered tokens are recomputed; prefix hits require phase
4. Recomputed rows are charged to execution, never delivered again as output.

### Step loop

1. Drain the inbox: adds and aborts.
2. Schedule within the token budget and maximum sequence count:
   - running decodes first, one token each; incremental allocation extends KV
     at block boundaries;
   - then one continuing prefill chunk;
   - then first-come, first-served admission. The default incremental policy
     requires space for the next chunk plus a watermark. Optional lifetime
     reservation instead requires the request's full declared cache demand.

   Incremental allocation preempts the newest eligible unscheduled holder when
   a running request needs more blocks. Lifetime reservation keeps admitted
   requests' blocks until completion and waits instead of evicting.
3. Build the StepBatch and upload its metadata once.
4. Execute: forward, token selection and one readback for all sequences.
5. Append tokens, apply stop IDs and limits, release finished requests' blocks
   tail-first, and emit tokens and events.

Initially at most one sequence prefills per step, so the prefill kernel still
handles one sequence with a cached prefix. Multi-sequence prefill is a separate
measured extension.

For positive output, lifetime reservation owns enough blocks for
`prompt_length + max_new_tokens - 1` cached positions before prefill begins;
the final emitted token need not enter KV. Zero-output requests finish without
holding blocks. Reserved capacity stays distinct from committed cache length
and from the row budget: execution still uses the chosen fixed or fitted chunk.
The oldest waiting request cannot be bypassed. The watermark applies while any
request is resident, even if none is selected this step, and is ignored when
the pool has no residents so a request that fits alone can progress. Every
terminal path releases written and unused reserved blocks.

Capacity reserved for maximum output can sit unused until completion; early stop
makes actual demand lower than the reserved peak. FIFO can delay a smaller
request behind an older request that needs more blocks.
The admission study measures those choices with the same fixed row budget and
numerical route. It does not promote reservation to the default or establish a
client latency guarantee.

### Token budget

Start with a fixed budget equal to today's 256-row chunk limit. Then fit

```text
execute_ns ≈ c0 + c1 · total_tokens + c2 · attended_positions
             + c3 · attention_partitions + c4 · sampled_logit_rows
```

from step records. The first implementation fits five nonnegative coefficients
and freezes them before evaluating a separate trace; `predicted_ns` records
each estimate beside measured `execute_ns`. Attention partitions count actual
singleton/decode and multi-row prefill launches; sampled rows account for the
vocabulary head. The largest prefill chunk whose prediction fits is selected.
Mandatory decodes, or one token when otherwise nothing can progress, may exceed
the target and are reported as such. The provisional 25 ms research target is
not a promised request-latency SLO. Prediction error and observed latency remain
separate from the policy's target. This follows Sarathi-Serve's stall-free
batching question with a measured cost model rather than a hand-tuned constant.
The completed fixed-workspace comparison retains 256: offline at 128 blocks,
fixed-32, fixed-64 and adaptive were slower by the paired noise rule, while
fixed-128 was inconclusive; at 40 blocks all candidates were inconclusive.
Online distributions remain descriptive. Thirty of 2,019 positive adaptive
evaluation steps exceeded the 25 ms setting, despite predictions within it.
The [retained results](../studies/model_generation/engine-core.md#fixed-workspace-budget-retained-bounded-results)
report prediction errors and first-token/gap tradeoffs separately.

### Asynchronous stepping

GPU token selection writes each decode sequence's next token directly into the
next step's input buffer. The host submits step n+1 before reading step n's
tokens and finishes its bookkeeping one step behind. A sequence that stops is
detected one step late, and its extra token is discarded. The intended effect
is to hide the readback wait in constraint 2. It requires writing step
metadata without a synchronizing map; see the open questions.

### Failure semantics

- An execution error fails every in-flight request in the engine, which then
  exits. The supervisor restarts it, and the frontend replays in-flight
  requests up to a per-request migration limit.
- Aborts and client disconnects take effect at the next step boundary.
- The waiting queue is bounded; a full queue rejects new requests explicitly.
- A request exceeding the per-request limit is rejected before any state
  changes, as today's chat rejects a full conversation.

The supervisor and frontend above belong to phase 5. Phase 3 exposes the engine
failure and fails its in-flight requests; it claims no automatic restart or
client-stream recovery. Retained histories make that later recovery possible.

### Phase 3 acceptance order

1. **Readiness.** Freeze the baseline, assets, toolchain and device. Record the
   full suite plus separate checkpoint lifecycle, batched equality and generation
   checks in a readiness receipt. A historical suite result does not substitute
   for this baseline check. A missing prepared checkpoint is an unavailable
   check, not a pass.
2. **Synchronous core.** Implement mixed decode with at most one prefill sequence,
   a fixed total-token budget of 256, bounded admission, preemption, aborts,
   scripted simulation and the Metal runner. Check scheduler and allocator
   invariants after every simulated step and in Metal acceptance runs. Preserve
   exact request/token accounting, rejection atomicity and write isolation.
3. **Load evidence.** Compare sequential execution, static batching, continuous
   batching and chunked prefill on frozen offline and seeded online traces. Keep
   every request and step record. This is the first phase 3 performance boundary;
   a working scheduler or faster kernel alone establishes no serving speedup.
4. **Fitted budget.** Fit the declared step-time model on calibration traces, freeze
   its coefficients, then evaluate on separate traces. Declare the predicted
   execution-cost research target before fitting; the initial setting is 25 ms.
   Declare a client inter-token SLO and goodput formula before corresponding
   latency or target-capacity claims. No target or goodput threshold is inferred
   from the observed result. Report predictions and errors as well as request metrics.
5. **Asynchronous stepping.** First prove that the pinned Metal API permits
   preparing the next metadata without a synchronizing map. Compare delivered
   tokens against synchronous execution on an identical frozen step schedule,
   including stop, limit and abort boundaries and discarded extra tokens. Then
   measure it as its own arm. If the API cannot support it, retain the probe and
   report that part of phase 3 as incomplete.

Stages have separate receipts. Phase 3 is complete only after its declared gates
and retained load study pass; a synchronous milestone does not close the fitted
budget or asynchronous work. The [declaration](../studies/model_generation/engine-core.md)
defines schemas, replay requirements and the numerical boundary. Reference-route
mixed-versus-solo execution, untouched storage and request accounting are exact
checks. Comparisons against Fast are diagnostics.

## KV cache manager

### Pool layout

One allocation holds every layer of every block, block-major:

```text
Pool[block, layer, kv, slot, head, dim]  BF16
(NB, L, 2, BS, Nkv, D) : (L*2*BS*Nkv*D, 2*BS*Nkv*D, BS*Nkv*D, Nkv*D, D, 1)
```

For Qwen, `L = 24`, `Nkv = 2` and `D = 64`, so a block holds `12,288 · BS`
bytes: 384 KiB at `BS = 32`. Attention still reads contiguous per-layer tiles,
and each block is one contiguous region, so SSD transfer and snapshots need one
I/O per block. The logical key of sequence `s` in layer `l` is a gather:

```text
K_s[t, h, d] = Pool[block_table[s, t / BS], l, 0, t % BS, h, d]
```

This is the first non-affine mapping in the [layout notation](layouts.md),
which will need an indirection form. Phase 1 keeps today's `(slot, head, dim)`
order within a block. Phase 2 also measures head-major `(head, slot, dim)`,
which makes each head's rows contiguous for the one-head-per-threadgroup
decode kernel. Neither order changes arithmetic.

### Block size

Every multi-row attention route the model uses reads KV in 32-row tiles that
start at multiples of 32; split prefill rounds its boundaries to whole tiles.
A block size that is a multiple of 32 therefore keeps each tile inside one
block, so address translation happens once per tile. Phase 1 gives each
sequence one block of the maximum context, which reproduces today's contiguous
caches exactly.

Phase 2 measures 32, 64 and 128. Larger blocks mean fewer block-table lookups
in both prefill and decode. Smaller blocks would split tiles across blocks and
buy little here: KV memory is plentiful (constraint 3), and a partial prefix hit
recomputes at most one block inside a step dominated by fixed launch cost.
Revisit smaller blocks only if a larger model makes KV memory scarce.

### Block states

| State | Meaning |
| --- | --- |
| Reset | free, no sequence |
| Partial | receiving tokens for one sequence; private |
| Complete | every slot submitted; not yet reusable |
| Registered | hashed, indexed and reusable by other requests |

`Complete` does not mean written: cache lengths count submitted tokens, and
the GPU writes asynchronously. Readers on the same ordered stream are safe.
Host readers, including SSD offload, snapshots and diagnostics, must wait for
completion of the step that wrote the block.

### Invariants

Checked after every step in debug builds and in every scheduler simulation:

- Total reference counts equal total block-table entries.
- Free, referenced and cached blocks partition the pool.
- A Registered block's bytes never change while referenced or indexed.
- Only full blocks are shared; Partial blocks are private, so no
  copy-on-write is needed.
- After all requests finish, no block is referenced.

### Prefix index

A block key is `SHA-256(parent_key, block_token_ids, cache_salt)`, so a key
commits to the complete prefix and tenant. Each Registered block also stores
its token IDs, which are compared on every hit: the hash is an index, not
proof. In memory, one engine build writes every block, so the key needs no engine
identity; persisted blocks do (see the SSD tier below). On admission, the
manager finds the longest chain of Registered blocks.
At least the final prompt token must be computed, because its logits are
needed. When the whole prompt hits, the final block is recomputed into a
private block, so shared blocks are never written.

Eviction is least recently used among unreferenced Registered blocks. A
finished sequence releases its blocks tail-first, so descendants are evicted
before their parents. An interactive chat session may pin its chain for a
limited time.

A cache hit reveals through latency that another request sent the same
prefix. `cache_salt` isolates tenants when a server has more than one.

### SSD tier

Registered blocks leaving memory persist in the shared store, under the
conventions it already applies to
[shared oracle fixtures](development.md#shared-oracle-fixtures):

- **Key.** A persisted block's key adds everything that determines its bytes:
  the model revision, the pool layout version and the engine's numerical
  identity. Under Fast, KV bytes depend on the kernels and chunk schedule that
  wrote them, as a fixture depends on its generator, so a kernel change starts
  a new namespace instead of mixing blocks.
- **Publication.** A block is written to staging and published read-only by one
  rename, so a reader sees a complete block or none. One file per block, named
  by its key, lets the filesystem serve as the index. Engines sharing the tier
  (phase 7) publish each key once under a per-key lock.
- **Verification.** Each entry records its token IDs and the SHA-256 of its
  bytes, and every load compares both. A mismatch is a miss; the entry is set
  aside, never trusted or silently deleted.
- **Capacity.** Least recently loaded entries are evicted within a byte budget.
  Namespaces that no current engine identity selects are listed and pruned on
  request, like `fixtures prune`.

A request whose prefix is on disk enters `LOADING_KV`. The load is asynchronous,
and the scheduler keeps running other requests.

A 4,096-token prefix is 48 MiB: an estimated 10–20 ms of sequential SSD reads.
The [token profile](../studies/model_generation/token-profile.md) measured about
2.2 s to first text for a 3,839-token prompt on an earlier runtime. The tier's
study question is the prefix length at which restoring beats recomputing.
Restored bytes equal stored bytes, so a disk hit is as exact as a memory hit.

## Batched execution and numerics

### Kernel changes

- RoPE and KV append index by `positions` and `slot_mapping` instead of
  `start_position` and `past`, including the fused configuration-26 decode
  kernel.
- Fast single-row decode attention uses the unsplit 32-simdgroup route.
  It gains a sequence index, per-sequence lengths and block-table translation.
  A future split-K variant must fix the split size, not the split count, so
  each sequence's partitions still depend only on its own length and the exact
  batched-versus-solo check below stays valid.
- Decode projections use the existing multi-row rowwise kernel
  (`_linear_rowwise_rows_apple_gpu_kernel`), which keeps the one-row kernel's
  lane-strided FP32 accumulation and `warp.sum` order. Phase 1 later replaced
  it: decode now uses the decode projection kernel's arrangement 8 for one
  sequence and for many, whose order differs from the one-row kernel's but not
  between batched and solo rows
  ([decode projection order](model.md#decode-projection-order)). The other
  configuration-26 decode kernels need multi-row forms with unchanged per-row
  reductions: SiLU/multiply fusion, residual RMSNorm fusion and GPU argmax.

### Numerical policy

Fast remains the default application policy for chat and generation; optional
`chat --engine` uses reference 27. Phase 3's initial
runner instead uses reference configuration 27 for every scheduling arm, with
an exact same-route mixed-versus-solo gate. This is a correctness and scheduling
baseline, not a promoted Fast route or a timing comparison against the existing
Fast application. Comparisons against Fast follow the diagnostic split in the
[model contract](model.md#correctness-and-diagnostic-policy).

In Fast, a row's arithmetic can depend on its step. A decode row scheduled with
a prefill chunk may use a different projection kernel. A prefix-cache hit
reuses KV computed inside another request's chunk. Replay recomputes generated
tokens as prompt rows. As with today's cached chat turns versus full replay,
these can change logits and occasionally a greedy token.

### Exact gates

These test data movement and ownership with the same numerical route:

- Existing S = 1 Fast execution remains unchanged. Reference configuration 27
  through StepBatch equals its solo route: logits and all KV bytes.
- A decode-only batch equals decoding each sequence alone with the
  same kernels. Wrong positions, wrong blocks and
  cross-sequence writes break this equality even when outputs look plausible.
- Paged attention equals contiguous attention on identical inputs; paging
  changes addresses, not arithmetic.
- Asynchronous stepping produces the same token streams as synchronous
  stepping.
- Reused and restored blocks keep the exact bytes and token IDs they were
  stored with.
- No sequence writes outside its own Partial blocks, checked with guarded and
  poisoned inactive storage as today's cache checks do.
- Allocator invariants hold; replaying events reconstructs the logical index,
  and every recomputed key matches.

### Diagnostics

For Fast, these comparisons record token agreement, first-divergence position
and logit distances, never pass/fail thresholds. Configuration 27's own-route
mixed-versus-solo gate remains exact:

- mixed steps against solo execution;
- prefix-cache hits against recomputation;
- replayed continuations against uninterrupted runs;
- outputs across replicas.

Whether serving can be made batch-invariant, and at what cost, belongs to the
[schedule-determinism follow-up](project.md#follow-up-direction). That work
would extend the [decoder policy study](../studies/decoder_layer/policies.md)
and the [consistency investigation](../studies/model_generation/consistency.md)
to batch composition, cache history and replay. It is optional research, not a
gate for any phase.

## Evidence

A serving claim is a latency distribution at a declared load, not one paired
median. Phase 3 extends the [experimental method](experiments.md) with:

- **Offline runs:** all requests arrive at time zero, for throughput and
  makespan. The existing four-block paired procedure applies.
- **Online runs:** arrivals come from a seeded Poisson process or a recorded
  trace. Arms alternate order across four blocks, and every request and step
  record is retained.
- **Metrics:** time to first token, time per output token, the full
  inter-token latency distribution, end-to-end latency, throughput, and goodput
  against a declared latency target.
- **Workloads:** generated traces with declared length distributions and
  seeds, multi-turn chat scripts, shared system prompts and deliberately small
  KV pools. External datasets require recorded provenance and license.
- **Arms:** each adds one mechanism to the previous arm: one request at a time
  (today), static batching, continuous batching, chunked prefill, prefix
  caching, asynchronous stepping and the SSD tier.
- **Predictions:** simulator predictions are recorded before the hardware run
  they predict. Agreement and disagreement are both reported.

Provenance follows the existing contract (hardware, software, commit, binary
hash, actual Metal device and conditions) plus the model card, scheduler
configuration and trace identity.

## Phases

| Phase | Delivers | Exact gate | Study question |
| --- | --- | --- | --- |
| 1. Batched decode | StepBatch; multi-row configuration-26 decode kernels; one maximum-context block per sequence; a batch axis in the existing model benchmark | S = 1 equals today; batched rows equal solo rows | How do throughput and per-token latency scale for B = 1–64 at contexts 64, 1024 and 3968? |
| 2. Paged KV | small blocks in the block-major pool, a block manager with the Reset, Partial and Complete states, paged decode and prefill attention | paged equals one block per sequence; allocation invariants; no writes outside a sequence's blocks | What does translation cost at each block size, and does head-major order help? |
| 3. Engine core | EngineCore, Scheduler, both runners, chunked prefill, incremental and optional lifetime admission, preemption, aborts, step records, trace driver, fitted budget, asynchronous stepping | scheduler and allocator invariants in simulation and on Metal; exact token accounting; reserved requests drain without replay; asynchronous equals synchronous | How do latency percentiles respond to arrival rate and memory admission across the scheduling arms, and where does the simulator disagree? |
| 4. Prefix caching | block keys, Registered blocks, KV events, prefix index, eviction, pinning, chat as an engine client | reused blocks keep their bytes and token IDs; only the uncached suffix is computed; logical event replay; existing chat checks pass | How does time to first token depend on shared-prefix length, hit rate and pool size? |
| 5. Frontend and API | frontend process, token protocol, model card, HTTP/SSE, supervisor, replay, backpressure, HTTP load generator | replay loses and duplicates nothing and preserves delivered tokens | What do the edge and recovery cost end to end? |
| 6. SSD tier | store entries keyed by engine identity, publication by rename, asynchronous loading, verification, eviction | restored bytes equal stored bytes; disk and memory hits agree; a changed engine identity never hits older entries | At what prefix length does restoring beat recomputing? |
| 7. Replicas (optional) | several engines behind a KV-aware router in the frontend | routing preserves histories and token accounting | Do independent submission threads raise throughput, and what does KV-aware routing gain over round-robin? |

The [batched decode plan](history/batched-decode-plan.md) details phase 1, and
its [batch-size](../studies/model_generation/batch-size.md),
[batched projection](../studies/model_generation/batch-projections.md) and
[reordered projection](../studies/model_generation/batch-reordered.md) studies
answer its study question: 64 sequences decode 998 tokens/s in aggregate at
1,024 cached tokens, 7.8 times one sequence's 127. The
[paged KV plan](paged-kv-plan.md) details phase 2, and its
[translation-cost study](../studies/model_generation/paged-kv.md) answered its
question for 2a's kernels: prefill paid at most 1.5% for 32-slot blocks, but
decode attention paid for every block a sequence spans, so steps of 64
sequences at 3,968 cached tokens took 2.8, 1.8 and 1.4 times as long with 32-,
64- and 128-slot blocks. Head-major order did not help. With decode attention
walking each group's keys in one loop, the
[rerun](../studies/model_generation/paged-kv-loop.md) found no resolvable cost
at any block size and selected 32-slot slot-major blocks: 133 ms against 132 ms
for 64 sequences at 3,968 cached tokens. They are the default since 2e.
`src/llm_mojo/serving/` starts in phase 1 with StepBatch and grows only as each
phase lands. The Qwen template, stop IDs and card values stay in
`models/qwen2/`. The `serve` command belongs in `cli/`, and the trace driver
and load generator in `benchmarks/`.

Documents that describe current behavior change when the phase lands, not
before:

- **Phase 1:** the forward input and ownership in [generation.md](generation.md),
  and `serving/` in the [code ownership](cli.md#code-ownership) table.
- **Phase 2:** the block-table gather in [layouts.md](layouts.md), and KV storage
  in the [model contract](model.md).
- **Phase 3:** the serving evidence contract in [experiments.md](experiments.md),
  and the trace driver in the [measurement tools](../src/llm_mojo/benchmarks/README.md).
- **Phase 4:** [chat.md](chat.md), where cache reuse becomes prefix hits in the
  engine.
- **Phase 5:** the `serve` command and launcher supervision in [cli.md](cli.md),
  plus a new serving guide covering the HTTP API, token protocol and recovery.
- **Phase 6:** SSD tier configuration and integrity checks in the serving guide.

## Reference designs

| Concept | Source | Where here |
| --- | --- | --- |
| Iteration-level scheduling | Orca (OSDI 2022) | step loop |
| Paged KV cache | vLLM PagedAttention (SOSP 2023) | KV cache manager |
| Chunked prefill, stall-free batching | Sarathi-Serve (OSDI 2024) | token budget |
| Hash-chained prefix caching | vLLM automatic prefix caching | prefix index |
| Radix-tree prefix sharing | SGLang RadixAttention | later alternative |
| Overlapped scheduling | vLLM asynchronous scheduling, SGLang overlap scheduler | asynchronous stepping |
| Batch invariance | [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | optional research follow-up |
| Frontend tokenization, token-only engines | [Dynamo frontend](https://docs.nvidia.com/dynamo/v1.3.0/backends/sg-lang/reference-guide) | token protocol |
| Request migration | [Dynamo fault tolerance](https://docs.nvidia.com/dynamo/v1.3.0/user-guides/fault-tolerance/request-migration) | failure domains, replay |
| KV events with gap detection | [Dynamo router design](https://docs.nvidia.com/dynamo/v1.3.0/design-docs/component-design/router-design) | event plane |
| Block states, tiered KV | [Dynamo KVBM](https://docs.nvidia.com/dynamo/v1.3.0/design-docs/component-design/kvbm-design) | block states, SSD tier |
| Engine simulation | [Dynamo mocker](https://docs.nvidia.com/dynamo/v1.3.0/user-guides/dynosim/mocker) | SimulatedRunner |
| Per-iteration metrics | [Dynamo forward-pass metrics](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/docs/fern/pages/developer-guide/knowledge-base/concepts/observability/forward-pass-metrics-rfc.md) | step records |
| KV-aware routing | [Dynamo KV router](https://docs.nvidia.com/dynamo/v1.3.0/components/router/routing-concepts) | phase 7 |
| Content-addressed, verified store | [shared oracle fixtures](development.md#shared-oracle-fixtures) (#28) | SSD tier conventions |

## Not adopted

- **Disaggregated prefill and decode.** The benefit comes from dedicating
  separately sized GPUs to each phase. On one GPU, chunked prefill addresses
  the same interference. The block export/import interface built for the SSD
  tier is the part that would carry over.
- **A distributed runtime.** Service discovery, message brokers, cluster
  operators and replica autoscaling solve datacenter problems. Here a Unix
  socket, static configuration and the scheduler's budget cover those roles.

## Open questions

- Can a host-visible MAX buffer on Metal be written for the next step without
  synchronizing the stream? Asynchronous stepping and per-step uploads depend
  on the answer.
- What does host access to a region of a device buffer cost, and what does it
  synchronize? SSD offload depends on it.
- What is the largest single buffer the pool can use on this device, or should
  the pool be split into slabs?
- Is an HTTP/1.1 and SSE adapter over POSIX sockets practical in Mojo, or
  should a thin adapter that carries no request state run as interop tooling?
- Does Metal interleave two processes' command buffers well enough for
  replicas to help?
