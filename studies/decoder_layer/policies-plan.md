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
