# Serving engine plan

Proposed on 2026-09-23 from baseline `edb610a`. Nothing below is implemented.
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

Today `QwenModel` owns 24 caches and one `length`, and `ChatSession` owns one
history. In this design, the model becomes stateless between steps, the
KVCacheManager owns all KV storage, each request owns its history, and chat
becomes an engine client whose turns reuse earlier turns through prefix hits.

### Failure domains decide the process split

The engine can fail: a Metal error already invalidates today's session. The
frontend therefore holds every in-flight request's authoritative token history
and never shares a process with the GPU. After an engine exit, the supervisor
restarts it and the frontend resubmits each in-flight request as its prompt
plus the tokens generated so far. The client stream continues with no lost or
duplicated tokens.

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

Workspaces are sized once from the token budget and maximum sequence count.
The step loop allocates nothing. Today's call is S = 1.

### ModelRunner

`execute(batch) -> StepResult` returns one selected token per sampling
sequence plus the step's timings. Two implementations share the interface:

- `MetalRunner` executes on the GPU.
- `SimulatedRunner` advances a virtual clock with the fitted step-time model
  and returns tokens from a deterministic script.

The scheduler, KV manager and EngineCore are the same code in both. The
simulator explores policies quickly; hardware measurements confirm them.

### KV events and step records

KV events carry `{event_id, step_id, kind, tier, block_hash, parent_hash}` with
kinds `stored`, `removed`, `cleared`, `offloaded` and `onboarded`. IDs increase
by one. A consumer that sees a gap discards its view and requests a snapshot.
Replaying the log must reconstruct the manager's prefix index exactly.

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
front of the waiting queue and usually resumes by hitting its own released
blocks.

### Step loop

1. Drain the inbox: adds and aborts.
2. Schedule within the token budget and maximum sequence count:
   - running decodes first, one token each, allocating a block at each
     boundary;
   - then one continuing prefill chunk;
   - then first-come, first-served admission when free blocks cover the next
     chunk plus a watermark.

   If a decode needs a block and none is free, preempt the most recently
   admitted running request.
3. Build the StepBatch and upload its metadata once.
4. Execute: forward, token selection and one readback for all sequences.
5. Append tokens, apply stop IDs and limits, release finished requests' blocks
   tail-first, and emit tokens and events.

Initially at most one sequence prefills per step, so the prefill kernel still
handles one sequence with a cached prefix. Multi-sequence prefill is a separate
measured extension.

### Token budget

Start with a fixed budget equal to today's 256-row chunk limit. Then fit

```text
step_ns ≈ c0 + c1 · total_tokens + c2 · attended_positions
```

from step records, and size each prefill chunk so the predicted step stays
under a declared inter-token target while decodes are running. This is
Sarathi-Serve's stall-free batching with a calibrated model in place of a
hand-tuned constant. `predicted_step_ns` records each prediction beside its
measurement.

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

Block size is a multiple of the prefill attention KV tile, so address
translation happens once per tile. Phase 1 gives each sequence one block of
the maximum context, which reproduces today's contiguous caches exactly.
Phase 2 measures 16, 32 and 64. Larger blocks mean fewer lookups. Smaller
blocks waste less of each sequence's final block and share prefixes more
finely.

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
proof. On admission, the manager finds the longest chain of Registered blocks
and always leaves at least the final prompt token to compute, because its
logits are needed.

Eviction is least recently used among unreferenced Registered blocks. A
finished sequence releases its blocks tail-first, so descendants are evicted
before their parents. An interactive chat session may pin its chain for a
limited time.

A cache hit reveals through latency that another request sent the same
prefix. `cache_salt` isolates tenants when a server has more than one.

### SSD tier

Registered blocks leaving memory can be written to a slab file with one block
per fixed-size region, plus an index of key, token IDs and a checksum of the
block bytes. An index entry is written only after its block data. A request
whose prefix is on disk enters `LOADING_KV`. The load is asynchronous, and the
scheduler keeps running other requests. A checksum mismatch is a miss.

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
  that each sequence's partitions still depend only on its own length.
- Decode projections use the existing multi-row rowwise kernel
  (`_linear_rowwise_rows_apple_gpu_kernel`), which keeps the one-row kernel's
  lane-strided FP32 accumulation and `warp.sum` order. The other
  configuration-26 decode kernels need multi-row forms with unchanged per-row
  reductions: SiLU/multiply fusion, residual RMSNorm fusion and GPU argmax.

### Batch invariance

A row's reduction strategy depends only on its own sequence, never on batch
size or co-scheduled work. With the kernels above, a decode row computed in a
batch must be byte-identical to the same row decoded alone.

Mixed steps are the hazard: choosing an MMA projection because N grew would
change decode arithmetic whenever a prefill is co-scheduled. The two policies
differ here:

| Policy | Mixed-step projections | Equality across batches and schedules |
| --- | --- | --- |
| Fast (default) | fastest measured choice for the step shape | diagnostic |
| Deterministic | decode rows keep rowwise order; prefill uses its invariant route | required, exact |

Prefix caching is transparent only if prefill is schedule-invariant. Under
Fast, a reused block computed inside another request's chunk can differ from
recomputation, just as a cached chat turn can differ from full replay today.
Under Deterministic, a hit must equal recomputation byte for byte. This extends
the [decoder policy study](../studies/decoder_layer/policies.md) and the
[consistency investigation](../studies/model_generation/consistency.md) from
chunk schedules to batch composition and cache history.

### Exact gates in every policy

- S = 1 through StepBatch equals today's forward: logits and all KV bytes.
- Paged attention equals contiguous attention on identical inputs; paging
  changes addresses, not arithmetic.
- Asynchronous stepping produces the same token streams as synchronous
  stepping.
- No sequence writes outside its own Partial blocks, checked with guarded and
  poisoned inactive storage as today's cache checks do.
- Allocator invariants hold, and replaying events reconstructs the index.

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
| 1. Batched decode | StepBatch; multi-row configuration-26 decode kernels; one maximum-context block per sequence; a batch axis in the existing model benchmark | S = 1 equals today; batched rows equal solo rows | How do throughput and per-token latency scale for B = 1–32 at contexts 64, 1024 and 3968? |
| 2. Paged KV | block-major pool, block manager, block states, events, paged decode and prefill attention | paged equals contiguous; invariants; event replay | What does translation cost at each block size, and does head-major order help? |
| 3. Engine core | EngineCore, Scheduler, both runners, chunked prefill, preemption, aborts, step records, trace driver, fitted budget, asynchronous stepping | simulated invariants; per-request equality under Deterministic; asynchronous equals synchronous | How do latency percentiles respond to arrival rate across the scheduling arms, and where does the simulator disagree? |
| 4. Prefix caching | prefix index, eviction, pinning, chat as an engine client | Deterministic hits equal recomputation; existing chat checks pass | How does time to first token depend on shared-prefix length, hit rate and pool size? |
| 5. Frontend and API | frontend process, token protocol, model card, HTTP/SSE, supervisor, replay, backpressure, HTTP load generator | replay loses and duplicates nothing; Deterministic replay equals uninterrupted output | What do the edge and recovery cost end to end? |
| 6. SSD tier | slab file, index, asynchronous loading, integrity checks | restored bytes equal stored bytes; disk and memory hits agree | At what prefix length does restoring beat recomputing? |
| 7. Replicas (optional) | several engines behind a KV-aware router in the frontend | routing never changes Deterministic output | Do independent submission threads raise throughput, and what does KV-aware routing gain over round-robin? |

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
| Batch invariance | [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Deterministic policy |
| Frontend tokenization, token-only engines | [Dynamo frontend](https://docs.nvidia.com/dynamo/v1.3.0/backends/sg-lang/reference-guide) | token protocol |
| Request migration | [Dynamo fault tolerance](https://docs.nvidia.com/dynamo/user-guides/fault-tolerance/request-migration) | failure domains, replay |
| KV events with gap detection | [Dynamo router design](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/router-design) | event plane |
| Block states, tiered KV | [Dynamo KVBM](https://docs.nvidia.com/dynamo/v1.2.1/design-docs/component-design/kvbm-design) | block states, SSD tier |
| Engine simulation | [Dynamo mocker](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/simulation/simulation-model) | SimulatedRunner |
| Per-iteration metrics | [Dynamo forward-pass metrics](https://docs.nvidia.com/dynamo/reference/observability/forward-pass-metrics-traces) | step records |
| KV-aware routing | [Dynamo KV router](https://docs.nvidia.com/dynamo/latest/router/README.html) | phase 7 |

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
