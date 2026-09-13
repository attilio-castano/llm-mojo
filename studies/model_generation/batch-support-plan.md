# Batching feasibility gate

Approved question: can grouping the same ordered Qwen kernels into per-layer or
per-token submissions reduce completed-token latency? Keep current Fast kernels,
buffers, arithmetic and dependencies unchanged. Before timing, require an actual
batching/replay interface in pinned MAX 26.5.0 / Mojo 1.0.0 on M4 Pro / Metal.

Extend the existing model benchmark with a build-only `MODEL_BATCH_SUPPORT`
route. Execute two ordered integer kernels, x=2*x+1 from x=3, and require 15.
Build the same two kernels with an explicit graph dependency; if supported,
replay twice and require 15 then 63. Record whether the builder callback was
entered. Recognize only the precise unsupported graph-builder error as an
unsupported capability; other failures are errors, not a negative capability
result. Run twice in fresh processes from one clean source/binary, without
performance measurements. Inspect installed DeviceContext/Stream/Graph exports
for alternative batching/flush/commit interfaces. Record source, binary/runtime
hashes, versions, hardware, full output and AC/thermal/power conditions.

If no usable interface is found, retain the capability result and describe the
smallest backend change needed. Do not substitute a native Metal demonstration
for a Qwen result or modify installed binaries. Stop the timing branch at this
explicit feasibility gate; this does not reject the batching hypothesis.

If supported, first verify actual command-buffer grouping in a separate trace.
Then compare current submission, per-layer batching and full-token batching at
64/1024/3968 cached tokens, with 16 warmups and 64 advancing steps, four balanced
blocks and matched self controls. Flush before required host visibility/waits;
include token upload through winning-token availability. Instrumented diagnosis
is separate from uninstrumented acceptance. Require full token/logit/KV parity,
cache boundary invariants, all-four-block gains above max(5%, self-noise), and no
reproducible regression at another context before proposing promotion.
