# Shared oracle fixtures

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives.

Baseline: `a144066`, after the September streamlining. Carried out on
2026-09-24. None of it changes arithmetic, numerical policy, retained evidence,
frozen oracle hashes, the generator scripts or `uv.lock`.

## Why

- **Space.** Validation regenerated every independent oracle into each
  worktree's `build/oracle_data/`. The attention-sublayer, MLP and decoder-layer
  families were 8.4 GB of the 8.5 GB there. Three worktrees on the development
  machine held identical copies, 25 GB in all, on a disk that was 88% full.
- **Time.** Generating the three families took most of the oracle stage of every
  validation, and much longer when two validations overlapped.
- **They could be shared.** The families are deterministic and frozen anchors
  pin their contents. Only `mlp/last_run.json` depends on the worktree, and every
  consumer reads them through `build/oracle_data/<family>`.
- **Their inputs rarely change.** Of the 27 commits on `main` at the baseline,
  15 touched `tests/fixtures/`, but only 5 changed a file these generators read,
  the last in #19.

## What changed

- The shared store from #24 keeps one read-only copy of each family per
  generator version, with a record of every file's size and SHA-256 and of the
  run that generated it. Validation links `build/oracle_data/<family>` to it.
- The key hashes the generator commands and every tracked or untracked file the
  three generators read, so other edits under `tests/fixtures/` regenerate
  nothing.
- A hit re-hashes every file and writes nothing to the store. A miss generates
  the family as before, under a per-key lock, and publishes it by rename; another
  worktree validating the same key waits and reuses it. A failed or interrupted
  generation publishes nothing and restores the checkout.
- A checkout's existing copy gives way to the link only when validation wrote
  everything in it; anything else is kept beside it.
- `validate --regenerate-fixtures` proves byte-for-byte reproduction, and
  `--no-fixture-cache` keeps the old behavior. `fixtures list`, `prune` and
  `detach` manage the entries. The Mojo MLP suites write their check records to
  `build/oracle_records/mlp/` instead of the linked directory.

The [development guide](../development.md#shared-oracle-fixtures) describes the
current behavior.

## Validation

Runs used a scratch store first, then the real one, on the Apple M4 Pro. Times
marked contended overlapped the full validation below.

| Check | Result |
| --- | --- |
| First run, empty store | 3 misses; the oracle stage took 504 s and published 8.36 GB, read-only at every depth |
| Rerun | 3 hits; the oracle stage took 12.4 s, of which verifying all 14,275 files took 2.3 s |
| Existing worktree | This worktree's 8.4 GB of copies gave way to links in 14.9 s, and free space rose by as much |
| Full `validate` through the links | Passed in 41 min (contended): 262 Python tests, 132 Mojo tests in all 25 suites on Metal, every route smoke check |
| `--regenerate-fixtures` | All three families reproduced byte for byte (577 s, contended) |
| Two processes, one empty store | One generated the decoder family; the other waited, then hit |
| Manual capture after `detach mlp` | The next validation relinked `mlp` and kept the capture's `tiny_manifest.json` beside it |
| `--no-fixture-cache` | Regenerated in the checkout (828 s, contended); all 14,378 store paths kept their size, time and mode |
| Real store | 3 misses published the same three keys as the scratch store (796 s, contended); the store grew from 1.99 to 10.36 GB |
| Damaged entry | A file added to the cached `mlp` tree was found on the next run; the tree and its record were set aside and `mlp` regenerated. `fixtures prune` listed the two set-aside items without changing anything, and `--yes` removed only them |

The Mojo suites read the large families through cwd-relative paths or paths
relative to their Python helpers, and the MLP check records appeared in the
checkout's `build/oracle_records/mlp/`, so the full run read the linked copies.

Apart from the capture kept by the detach check, this worktree now holds
0.08 GB of oracle data. Other worktrees keep their copies until they validate
with this change; each then links the shared copy, so the three copies that
held 25 GB become one of 8.4 GB in the store.
