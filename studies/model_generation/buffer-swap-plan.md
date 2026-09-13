# Remove inter-layer copies by swapping buffer ownership

Control: current Fast single-row configuration 26, CPU greedy selection.
Candidate: identical kernels and selection, with the input and MLP-output
DeviceBuffer owners exchanged after layers 0..22. Both allocations remain
alive on the same ordered stream. Rebuild each next layer's input view from
its current owner. Final output remains in mlp.output. No arithmetic changes,
new allocations, relaxed alias checks or synchronization points. Multi-row
prefill retains the copy path. GPU argmax remains a separate parked experiment.

The candidate removes 23 compute launches: 314 to 291 per token, plus the same
four mapping blits. The hypothesis is reduced submission overhead; the prior
0.074 ms active copy duration does not predict complete-token savings.

Correctness: test queued GPU work across odd/even owner swaps; compare every
logit and all 48 KV buffers at histories 64, 1024 and 3968 with poisoned outputs;
verify all layer hidden states for one full-model step; exercise consecutive
steps, transitions back to multi-row prefill, reset, rejection and reuse.
Verify ownership identities reverse after each single-row call and that the
buffers remain distinct. Stream three fixed prompts, 128 tokens each, in four
paired blocks with exact emitted bytes, token IDs and history. Run full
repository validation before committing the measured candidate.

Freeze clean source and compile once. Reuse the existing fusion harness with
--copy-free: four timing blocks, three histories, CPU self-pairs and candidate
pairs, ten warmups and ten retained samples per arm (480 retained samples).
Reverse arm and workload/comparison order in blocks 2 and 3. Boundary: token
upload through greedy readback; exclude rewind, recording and initialization.
Capture control/candidate traces separately at history 1024, ten warmups plus
eight steps. Retain all samples, conditions, asset/source/binary hashes and
complete trace command coverage. No tuning after acceptance samples.

Promote only if all four candidate/control ratios are below one and median
paired reduction exceeds max(5%, largest absolute self-pair variation), at
all three histories. Otherwise retain the current Fast policy. Promotion, if
qualified, is limited to single-row Fast/auto on Apple M4 Pro / Metal. Keep
explicit combined (copy) and buffer-swap (candidate) controls.

Reproduction follows the existing model_profile commands: build --copy-free,
collect, fusion-capture, fusion-terminal, fusion-archive --copy-free,
fusion-replay --copy-free and fusion-plot --copy-free. Expanded artifacts
remain outside Git; the compact archive is named buffer-swap.json.gz.
