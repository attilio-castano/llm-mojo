# Fast and deterministic decoder policies

This campaign studies one Qwen2.5-0.5B decoder layer on Apple Metal. Its
[executable declaration](../../tests/fixtures/decoder_policies.json) freezes
the workloads, schedules, fresh confirmation inputs and search budget before
candidate measurements. The previous selection study and model-consistency
results retain their original meaning.

Both policies implement the same Qwen equations, causal visibility, absolute
RoPE positions, BF16 stored boundaries and FP32 reductions. Both retain the
existing layer and isolated-operation accuracy gates. Hugging Face supplies
the independent numerical reference; matching its exact rounded outputs is
not a policy requirement. Tolerances cannot be enlarged after seeing failures.

Fast permits schedule-dependent rounding and selects the fastest demonstrated
validated configuration at each measured workload and reuse mode. Deterministic
requires all 16 intermediates, output and active KV prefixes to match the same
candidate's full-call bytes across full, repeated, tokenwise and irregular
schedules on the same hardware and build. A new deterministic family may use
a different reduction order from configuration 20. Its prefill and decode
kernels must agree with one another. Cache policy is fixed for its lifetime;
changing it requires rebuilding the prefix.

Start with the previous shared fast lookup: configuration 0 for full prefill,
decode and cached (16,256), and configuration 3 for cached (64,4096).
Configuration 20 is the initial deterministic control. Validate the existing
Qwen synthetic stress cases and three real checkpoint cases. Repeat every
case with the declared schedules through 4096 tokens. Preserve all operation,
cache ownership, asynchronous reuse and negative-control checks.
Tokenwise replay uses one active scratch row and one fully checked guard row,
with the full cache and rotary tables retained. Fixture arrays are mapped read
only and sliced before comparison. Full and irregular calls continue checking
the complete larger workspace. This bounds host test traffic without reducing
the tokenwise schedule or the checked storage extent of any allocation.

Measure the seven declared shapes in hot and ring24 modes separately using
the existing paired protocol and self-pair calibration. Ring24 has 24 distinct
weight/input/cache allocations and shared scratch; it is not a 24-layer model.
Allocation and prefix preparation precede timing. Capture six initial traces,
one representative per phase and policy, and require the actual dispatch census.

Use those profiles to justify at most two rounds of two new configurations.
Freeze each round's mechanism and parameters before observing its results.
Possible mechanisms are fixed-arithmetic tiled projections, launch fusion that
preserves materialization rounding, and fixed attention partitions. There is
no obligation to use every candidate slot. Reject failed accuracy candidates;
non-invariant candidates remain eligible only for Fast. Confirm the final
choice once on the declared fresh inputs and an independent timing session.
Retain failures and noisy measurements, and keep a validated fallback.

Deliver the two layer policies, an explicit workload lookup, compact raw samples
and provenance, and the measured cost of the deterministic guarantee. This
campaign does not establish full-model accuracy, text-generation quality,
cross-device identity, or fidelity to the unknown training execution path.

## First optimization round

The six baseline captures passed their dispatch and runtime checks at
`96937e0`: 4,975 measured dispatches on Apple M4 Pro / Metal. At full
`(256,256)`, configuration 20's MLP accounts for 85.225% of captured GPU
active time. At cached `(64,4096)`, its MLP accounts for 68.538% and GQA for
20.592%. These measurements justify testing projection reuse first.

Freeze two new configurations before their first execution:

| ID | Attention | Projections | Intended arithmetic family |
| --- | --- | --- | --- |
| 21 | Existing consistent G32 | Existing 8x16 QKV/Wo and 16x16 MLP MMA at every row count, including one | New fixed-MMA family; exact agreement with 20 is not required |
| 22 | Existing consistent G32 | Packed QKV and four-row weight reuse for QKV, Wo, gate, up and down; original rowwise kernel for one row | Preserve 20's lane-strided FP32 sums and SIMD-group reduction |

The first screen uses the six declared cells other than `(4096,4096)`, in
hot and ring24 modes. Compare both candidates directly with 20 and its
same-run self-pairs, then directly with each cell's previous Fast control
(0 or 3) and its self-pairs. The executable declaration contains the exact
three matrices. The independent final comparison includes all seven cells;
the long-prefill extension cannot be called a demonstrated winner from this
smaller screen.

Every unchanged numerical gate must pass before timing. Classify each
candidate's own schedule invariance separately. Compare all 16 stored stages,
including Y, and both KV prefixes between 20 and 22. Only a byte-compatible
family may mix these configurations across calls. Retain 20 wherever 22 does
not pass the existing direct, noise-calibrated gain rule.

Configuration 21 is a global family candidate. It can replace the compatible
family only if every screened cell in both modes is demonstrably faster than
20, and its complete block-ratio range also beats each qualifying 22 range by
the calibrated margin. Otherwise record its tradeoff. This avoids choosing a
default deterministic family by an unstated weighting of prefill and decode.
The exact inequality and tie-breaking rules are in the declaration. Fast may
select any numerically qualified per-cell winner, whether invariant or not.

The final numerical suite executes the actual lookups in both modes, using
test-only IDs 100/101 for hot Fast/Deterministic and 102/103 for ring24. A
lookup schedule `[16,224,16,3776,64]`, clipped at the case length, reaches the
measured cached cells. A separate 256-row prefix schedule reaches full prefill;
tokenwise calls reach both measured decode cells. Together these schedules
exercise every lookup cell while preserving complete prefix ownership.
Deterministic must also agree between reuse modes before they can share a
family. These test IDs are not benchmark configuration IDs.

The fresh confirmation prompt has 46 pinned chat-template tokens, declared
without decoder execution, with little-endian int64 SHA-256
`fdc37a0a6bdf05ef4f7764425d99d3cbe491c5d8f0180bfd6fc97fa295213180`.
Seeds 7141/7151 and the previously declared lengths remain unobserved.

Baseline collection encountered a tuple/list comparison error after all six
captures had passed analysis. Canonicalizing the capture triples fixes that
metadata seam. Retain the original captures and failure note; no measurement
was repeated or removed.
