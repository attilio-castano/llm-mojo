# Streamlining: one-step setup, one production path, current docs

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives.

Baseline: `edb610a`, the unified CLI after native Fast chat. Carried out on
2026-09-23 as three stacked changes. None of them changes arithmetic,
numerical policy, retained evidence, frozen oracle hashes or `uv.lock`.

## Why

- **Setup** took several manual steps and had no toolchain checks. A missing
  model surfaced as a raw `[Errno 2]`, an interrupted preparation left a broken
  directory, and every checkout needed its own ~1.9 GB of assets.
- **The engine** carried the arms of finished experiments on its production
  path:
  - 25 policy strings, several of which selected the same route;
  - 8 study-only decoder configurations;
  - a 10-flag `QwenModel.forward`;
  - two "fast" lookup tables that disagreed.

  Default validation never ran the route that every chat token uses.
- **The docs** interleaved current contracts with completed plans. They repeated
  numbers in up to seven places and carried about twenty stale statements and
  two broken documented commands. Nothing explained how a token flows through
  the engine.

## What changed

**Setup and the shared store.**
- `llm-mojo setup` checks the toolchain with printed remedies.
- It provisions the pinned checkpoint and prepared model once per machine in a
  shared, read-only, hash-verified store, and links each checkout to it.
- It prepares the tokenizer tables and builds chat and generate.
- The prepared model's identity is a pinned digest of its 196 tensor hashes.
- Imports prefer verified local copies over downloads, and preparation publishes
  by atomic rename.

**One production path.**
- Every model call takes an `ExecutionPlan`: fast, baseline or consistent, or
  an explicit retained configuration for diagnostics.
- Configuration 26 always carries its three decode features, and no other
  configuration carries any, so the unpromoted compositions cannot be expressed.
- The model records the route it enqueued (`ForwardRoute`); generate reports
  and validation check it.
- The following exist through `edb610a`:
  - decoder configurations 1, 4, 8, 12, 14, 23, 24 and 25, with the mappings
    only they reached;
  - the projection arrangements and the fused head;
  - 22 policy strings;
  - the collectors of completed model-level experiments.

  Their retained evidence still replays.
- New default tests cover:
  - the production decode route against baseline, byte for byte, on a
    synthetic model;
  - every Fast lookup cell;
  - configuration 21.

  A real-model `decode-parity` check compares the full 24-layer decode.

**Current docs.**
- Completed plans and superseded narratives moved here, with the
  [history index](README.md).
- Study indexes give each result a status.
- Stale statements were corrected.
- A [documentation map](../README.md) and a [walkthrough](../walkthrough.md)
  were added.
- A link test checks that every relative link and anchor resolves, and that
  every page is reachable from the README.

## Validation

- **Behavior.** Every engine commit was compared byte for byte with the
  baseline: model-driver captures (hidden states, final norm, logits and all
  K/V for three modes, every measured prefill cell and mixed schedules),
  generate output and report events, chat driver files and a terminal
  transcript. All 87 records were identical after every commit.
- **Decode parity.** On the real model, 32 teacher-forced Fast decode calls
  matched baseline in all 4,059 captured files.
- **Evidence.** Every offline replay, `summarize.py` and `plot` produced the
  same output as at the baseline. `plot --policies`, broken since a path rename
  in #22, reproduces its committed report and figure again.
- **Speed.** The old and new generate binaries alternated five times each on
  an idle machine on AC power (1,089-token prompt, 256 tokens). Median decode
  times were 7.954 and 7.960 ms per token, and every run fell between 7.935 and
  7.988 ms. An earlier run on a loaded machine gave 9.66 and 9.76 ms. The
  route-checked generation studies show the fused decode route ran on every
  Fast decode call.
- **Full suite.** `uv run --locked llm-mojo validate` passed after setup and again
  at the end of the engine changes, 49 minutes each. The second run regenerated
  every oracle against its frozen anchor and passed 222 Python tests, 132 Mojo
  tests in all 25 suites on Metal, and every benchmark route smoke check. The
  documentation changes added the link, reachability, history-banner and
  walkthrough-symbol checks and a test for the plot selection, bringing the
  Python suite to 229 tests.

## Known differences found, not changed

Retained evidence is frozen, so these were recorded rather than regenerated:

- `plot` regenerates the nine MLP figures with different bytes from the
  committed PNGs; the tables it regenerates are identical.
- The token-profile replay labels one host phase "host winner processing",
  where the committed `token-profile-summary.json` says "CPU argmax scan". The
  values are identical.
