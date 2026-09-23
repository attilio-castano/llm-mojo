# MLP sublayer plans

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives.

Moved from the [MLP contract](../mlp-sublayer.md) on 2026-09-23.

## Original reference-readiness plan (completed)

The following plan records the predeclared reference milestone. Its results and
reproduction commands follow below; it is not an outstanding implementation task.

This plan governs the reference-readiness work recorded below. The run ends with a reproducible
upstream reference package and frozen numerical acceptance rules, ready for
Mojo baseline implementation. It follows the request to establish the numerical
contract and fixture specification first.

Work locally on `codex/swiglu-correctness-baseline`. Implement the Python fixture
tooling, its necessary tests, and documentation; run the pinned CPU reference
and diagnostics using the existing locked environment. Routine implementation
and fixture-tooling repairs can proceed without intermediate approval. Existing
Mojo numerical contracts and dependency locks remain fixed. Commits, pushes,
PR creation, new checkpoint downloads, GPU MLP implementation, and profiling
are outside this run's scope.

### 1. Establish reproducible capture

Inspect local checkpoint assets read-only, validate their provenance, and check
whether complete layer-0 MLP tensors are available. Continue synthetic work if
they are absent; record checkpoint validation as pending rather than substituting
synthetic weights. Add an MLP generator to the existing fixture dispatcher.
Execute actual pinned upstream modules and capture the declared boundaries.

Exit evidence: tiny development cases have the expected shapes/dtypes, the
captured data flow is consistent, and adding observation leaves upstream output
unchanged. The manifest identifies sources, dependencies, and fixture inputs.

### 2. Characterize development numerics

Implement and test the independent diagnostic rounding and FP64 calculations.
Run the declared synthetic development matrix, structured stress cases,
activation/gating sweep, and upstream full/chunked comparisons. Add available
checkpoint development cases with post-attention inputs under the selected
attention policy. Record all discrepancies, including signed zero, subnormal,
and overflow cases. Establish small boundary regressions that distinguish the
specified computation from one that skips an intended BF16 rounding step.

Exit evidence: a compact report identifies the worst differences, their
operands, and the supported explanations; the capture and diagnostic tools have
independent checks. Holdout outputs remain unobserved.

### 3. Freeze the contract and acceptance rules

Choose and explain the initial stage budgets from development evidence and
numerical reasoning, preserving the specified BF16 boundaries. Define the
SiLU representable-step and zero/subnormal rules. Initial budget selection may
proceed autonomously; weakening existing gates, changing arithmetic, or narrowing
the supported input domain requires an explicit numerical-policy decision.
Record the evidence as a hypothesis for later GPU validation, not a GPU bound.

Freeze contract version, recipes, development array hashes, the holdout case
list, and available checkpoint token IDs. Verify that the proposed gates reject
the deliberately incorrect rounding/data-flow controls. Preserve boundary
regressions even when aggregate tolerances would accept a changed computation.

Exit evidence: every acceptance rule has a concrete value or exact predicate,
provenance, and rationale. No gate is chosen from Mojo results or holdout outputs.

### 4. Verify and hand off

Regenerate the development arrays once and verify their frozen hashes, capture
invariants, and diagnostic results. Add meaningful tests for the rounding oracle,
capture instrumentation, manifest validation, and refusal to overwrite frozen
anchors silently. Integrate reference preparation with the existing validation
workflow, then run `uv run --locked llm-mojo-validate` to check repository
regressions. Keep holdout execution explicit for the later baseline acceptance;
ordinary reference preparation must not silently consume it.

Exit evidence: reproducible commands, passing relevant checks, compact retained
results, and an updated contract identifying the exact completed matrix. The
handoff names checkpoint evidence still pending and makes no Mojo MLP
correctness or performance claim.

### Stop conditions and progress

Diagnose ordinary tooling defects and continue within the fixed contract.
If evidence requires changing arithmetic, relaxing an exact requirement,
narrowing the supported domain, or selecting budgets without a defensible
explanation, preserve a minimal reproducer and stop the dependent phase for a
decision. Continue independent checks where useful. A missing checkpoint asset
blocks checkpoint acceptance, not the synthetic reference work; acquiring new
assets requires separate authorization. Resource pressure calls for processing
smaller batches of the same declared cases, not dropping coverage silently.

Report progress at each exit criterion and any material failure. Finish with the
changed files, exact validation coverage, unresolved decisions, and the command
sequence for reproduction. Once these criteria are met, end the run; the next
milestone is the Mojo correctness baseline described below.

## Original implementation and baseline plan (completed)

The implementation and measurements below are complete. This section preserves
their original scope and stop conditions; the study records subsequent campaigns.

Initial fixture acceptance completed on 2026-09-07 at local source `afbe288`.
All 43 synthetic and three checkpoint development cases passed, followed by
the six declared synthetic holdouts and reserved checkpoint prompt on the
same frozen binary. All GPU full/chunked intermediate comparisons were
bit-exact. No numerical budget or supported input domain changed. See the
[MLP study](../../studies/mlp_sublayer/README.md) for the retained checks and
the subsequent residual cutoff repair and baseline measurement results.

The original run was authorized for local implementation, validation, local commits,
Metal measurement and trace capture, and study documentation using existing
assets. Preserve this frozen reference package; new acceptance tooling records
its own identity and does not rewrite the reference anchors.

Implement and test materialized SiLU and multiply first, then compose RMSNorm,
rowwise gate/up projections, SiLU, multiply, down projection and residual using
caller-owned buffers. Accept operation and composition gates on all development
cases, followed by the previously unopened holdouts. Include full/chunked and
normal-mode asynchronous reuse, invalid-call, poison and guard tests.

After full validation and benchmark-route checks, commit the source and build
once from that clean identity. Whole-block timing uses R=1,7,15,16,17,33,65,257,
1024,4096 in hot and ring24 modes with the existing four-block, ten-warmup,
ten-sample self-pair protocol. Isolated stages and whole-block traces use
R=1,17,1024,4096. Profile separately, within 5,000 measured dispatches per
capture. Record backend/device, source/binary/fixture hashes, power/thermal
conditions and exact allocation/synchronization boundaries. Preserve valid
noisy observations. Finish with reproducible numerical and performance evidence
and a ranked proposal for the next optimization.

Routine implementation defects may be repaired under the fixed contract. Any
required change to arithmetic, budgets, or supported inputs stops acceptance
for a numerical-policy decision. Holdout failures remain recorded. Invalid
measurement conditions pause collection rather than weaken its requirements.
