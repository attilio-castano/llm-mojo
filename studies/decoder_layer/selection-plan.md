# Selecting the implemented decoder configurations

Approved for autonomous local execution on 2026-09-08. Starting source:
`13b32b4`, branch `codex/decoder-layer-baseline`, clean worktree. Use the
existing locked environment, oracle, numerical gates and study tools. Work
inline; no delegation. Local edits, sequential Metal measurements, Instruments
captures, fixture generation from verified local assets and local commits are
authorized. No push, PR, downloads, runtime/model implementation or new kernels.

## Question and frozen shortlist

Find the best demonstrated combinations of implemented kernels on a declared
workload grid. Hot and ring24 are separate execution boundaries. Preserve the
historical baseline and evidence. No numerical tolerance or oracle arithmetic
changes are permitted. Existing holdouts may be reused as regressions; new
reserved outputs are declared below before execution.

Decoder study IDs (not attention/MLP IDs):

| ID | Attention GQA mapping | Attention projection mapping | MLP mapping |
| ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 7 for multiple rows, 0 for one row |
| 1 | 0 | 5 (both 16x16) | as 0 |
| 2 | 4 (split8) | 0 | as 0 |
| 3 | 4 (split8) | 5 | as 0 |
| 4 | 0 | 0 | 0 (short-row challenger) |
| 8 | 0 | 0 | 8 on one row, 7 otherwise |
| 12 | 0 | 0 | 12 on one row, 7 otherwise |
| 14 | 0 | 0 | 14 on one row, 7 otherwise |

IDs 8/12/14 define complete schedules, so prefix preparation uses the accepted
multi-row path and one-row calls use the named decode challenger. Their prior
MLP evidence is inconclusive, not a previous promotion. Split8 requires explicit
caller-owned partial storage. All projection/GQA arithmetic already exists.

## Fixed measurements and selection

Four paired blocks, ten warmups and ten samples per arm, with matching control
self-pairs. Require all four block ratios below one and median improvement
greater than max(5%, the run's control self-pair deviation) for promotion.
Never repeat a valid observation because of noise. Select separately per exact
workload/mode: qualifying candidate with lowest median paired ratio, then lower
ID. Without qualification use 0. The shared default uses a candidate only when
both modes select it and independent confirmation passes in both.

| Study | (R,T) workloads | IDs |
| --- | --- | --- |
| Full prefill | (16,16),(64,64),(256,256),(1024,1024),(4096,4096) | 0,1 |
| Short full prefill | (7,7),(15,15),(16,16),(17,17) | 0,4 |
| Cached prefill | (16,256), R in {16,64,256} at T in {1024,4096} | 0,1,2,3 |
| Decode | (1,64),(1,256),(1,1024),(1,4096) | 0,8,12,14 |

Screen ceiling: 9,920 latency observations. The duplicate (16,16) studies are
separate mechanism comparisons; each retains its own calibration. A short-row
MLP winner takes precedence there only if confirmed against the baseline;
disagreeing candidate families remain explicit alternatives.

Before decode screening, execute one fixed control-only calibration at T=256
and 4096 in hot/ring24, plus a diagnostic hot batch of 16 ordered calls with
one synchronization. This adds 480 observations. The batch is explicitly a
different boundary; it diagnoses amortization and cannot promote a raw-hot
winner or replace raw-hot samples. Report unresolved noise honestly.

Freeze a selection JSON from the completed screens before confirmation. Run
one independent confirmation of the chosen per-mode candidates against 0,
including control self-pairs. Also check neighboring shapes:
full (257,257),(1023,1023); cached (15,256),(17,256),(65,4096),(255,4096);
decode (1,257),(1,4095). The proposed neighbor uses the nearest screened R
within the same phase and T, or nearest T for decode; tie goes to smaller R/T.
Full neighbors use the main full-prefill screen. Neighbor measurements do not
change a failed candidate or reopen selection. No extrapolated crossover is
claimed. At most two distinct proposed IDs plus control per workload; at most
28 workload entries (including separately labeled duplicate screens), two
modes, three candidates: 13,440 confirmation observations. Empty challenger
sets retain only control. Total timing ceiling: 23,840 observations.

After confirmation, retain an explicit lookup table of accepted workload/mode
configurations with baseline fallback outside tested cells. Report disagreements
and non-confirmations. Do not invent continuous ranges from a few points.

Profile only representative accepted configurations at (256,256), (64,4096),
(1,4096), retaining the control plus at most the two mode winners: at most nine
captures, ten warmups and respectively 25/25/100 measured calls. Dispatch counts
are derived from actual GQA split and combined gate/up routes, not fixed at 16.
Require complete target Metal coverage and join preempted intervals. One
malformed-capture retry after diagnosed tooling repair is allowed; keep failure
artifacts and identify changed tool/binary receipts. Profiles explain costs;
they do not replace paired latency.

## Numerical gates and sequencing

1. Extend the existing decoder entrypoint/preflight and tools for the IDs above.
   Preserve ID 0. Regression-test route identity, partial-buffer capacity,
   invalid mapping/overlap rejection before enqueue, exact cache/protected
   storage, operation-local checks and whole-layer boundaries. Exercise the
   asynchronous mixed prefill/decode schedule for every ID.
2. Run every ID on all existing Qwen synthetic/checkpoint cases and schedules;
   retain tiny reference tests. Decode challengers apply only to single-row
   calls; their multi-row schedule path is ID 0. All original BF16 gates remain.
3. Declare fresh reserved seeds 6011 and 6029 at T=1,17,257,4096, plus checkpoint
   prompt: "A train travels 60 kilometers in 45 minutes. Explain its average
   speed in kilometers per hour and why the units matter." Use the verified
   pinned tokenizer to record token IDs/hash before reserved model execution.
   Use the unchanged upstream generator. Freeze numerical/measurement source
   and binary receipts before opening these nine reserved outputs.
4. Execute and evaluate every ID on all reserved schedules with full coverage
   verification. On any numerical failure, stop dependent measurement; retain
   the exact candidate and failure. Do not adjust gates, consume more holdouts
   or revise a candidate in response to reserved outputs.
5. Run the fixed calibration, screens, frozen-selection confirmation and
   selected profiles, sequentially on proved Metal. Record conditions before
   and after. If required hardware conditions fail, pause dependent work.
6. Curate into this topic using selection-prefixed compact records and the
   existing readers/plotter. Retain all raw observations and failed attempts,
   excluding arrays, binaries and traces from Git. Explain allocation/traffic,
   ownership and synchronization separately. Validate reproduction and commit
   locally. No automatic follow-up optimization or full-model implementation.

## Progress

- [x] Inspect clean source, prior results, kernel mappings and authorization.
- [x] Freeze executable declaration and reserved token identity.
- [x] Implement route/preflight/measurement extensions and meaningful tests.
- [x] Pass development/checkpoint/asynchronous checks and full validation.
- [x] Freeze clean binaries and pass all fresh reserved acceptance gates.
- [x] Collect calibration/screens and freeze workload-specific proposals.
- [x] Confirm proposals/neighbors and collect selected diagnostic profiles.
- [x] Retain reproducible evidence, selection lookup and explanation; final
  checks and local commit, leaving a clean worktree.

Validation before candidate freeze: 111 Mojo tests in 16 suites; 98 Python
tests (one retained-selection test awaits collection); all frozen oracle checks
and benchmark smoke routes passed. All eight IDs passed 51,296 synthetic,
2,544 checkpoint and 10,352 previously opened holdout core checks, plus exact
storage and asynchronous checks. No numerical policy was changed.

Completed measurements: 16,800 observations (480 calibration, 9,920 screening,
6,400 confirmation); 21 accepted mode-specific cached cells and ten shared
choices. Full/short prefill and decode retain ID 0. Four successful diagnostic
captures retain 2,825 measured dispatches; no capture retries or valid-trial
reruns occurred. Final curation includes a reusable selected-window reader and
a clearer graph; all inference/measurement evidence remains bound to `b88ca50`.

Final curation verification: all 98 Python tests pass with no skips; all 19
generated PNG/CSV artifacts reproduce byte-for-byte, including the historical
baseline artifacts. Local documentation links and `git diff --check` pass.
