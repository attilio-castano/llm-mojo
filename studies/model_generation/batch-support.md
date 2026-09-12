# Metal batching feasibility: backend support is missing

The batching experiment stopped at its declared feasibility gate. On our pinned
MAX 26.5.0 / Mojo 1.0.0 Metal backend, the eager dependent-kernel control passes,
but the available graph API fails before recording any operation:

```text
createGraphBuilder() not supported on this device context
```

Both fresh processes gave this result from one clean build. No usable direct
batching scope was found in the reviewed DeviceContext/DeviceStream interfaces
or installed exported ABI. The per-layer/per-token Qwen timing comparison was
therefore not run. This is a backend capability result; it neither measures nor
rejects the potential speedup from batching. Fast remains unchanged.

## Executed check

The [approved gate and conditional timing plan](batch-support-plan.md) are
implemented in the existing model benchmark, using `MODEL_BATCH_SUPPORT`.
The [receipt](batch-support.json) retains clean source identity, build command,
binary/runtime hashes, installed API entrypoints, hardware/software, conditions,
full process output and parsed results.

The control starts an Int32 value at 3 and executes two ordered GPU kernels,
each calculating `x = 2*x + 1`. Reading 15 proves that real dependent work ran on
Apple M4 Pro / Metal. The graph attempt uses the same compiled kernel, buffer and
two nodes, with an explicit predecessor dependency. If creation succeeded, two
replays would have to produce 15 then 63. In both processes, creation instead
raised the unsupported-builder error before entering the callback. No graph
replay or batched GPU computation ran.

The collector recognizes this exact error and rejects unrelated errors, failed
control results, missing completion markers or incorrect replay values. It also
rejects a supposed unsupported-builder result if the callback was entered.
Successful replay would still require inspecting command-buffer grouping before
making a batching claim.

The installed library exports `AsyncRT_DeviceContext_createGraphBuilder`, its
recording-context and graph-replay functions, and individual context/stream
`enqueueFunctionDirect` entrypoints. Export presence does not imply support on
Metal; the executed probe distinguishes them. No exposed begin/end-batch or
flush/commit control was found in that interface review. This is a bounded API
finding, not a claim that no internal implementation or possible workaround
exists.

The documented [DeviceGraph](https://max.modular.com/stable/api/mojo/max/gpu/host/device_graph/DeviceGraph/)
and [recording context](https://max.modular.com/stable/api/mojo/max/gpu/host/device_graph/DeviceGraphBuilder/)
are relevant generic APIs, but our local execution supplies the backend-specific
answer. The [upstream Metal issue](https://github.com/modular/modular/issues/6899)
also reports missing batching and graph support for this version. Its status was
not used as proof of local capability.

## Existing command-buffer evidence

Reanalysis of the previously captured original-projection trace at prefix 1024
found, for each of eight measured tokens:

| Retained commands | Distinct command buffers | Encoders per buffer |
| --- | ---: | ---: |
| 245 compute + 4 blits | 249 | 1 |

Each retained command's submission start was matched to the original submission
XML and its command-buffer ID. All 249 IDs were distinct per token. The
[compact census](batch-support-prior-trace.json) records the original capture,
source commit, input hashes and all eight counts. This is reuse of the frozen
[projection scheduling evidence](projection-scheduling.md), not a new capture
or new timing measurement. It confirms the existing path's separate command
buffers and provides a concrete comparison target for a future batching path.

## Smallest proposed backend change

A scoped submission batch in MAX's Metal driver is the smallest design to
investigate before implementing full graph replay. This is a proposed interface,
not an existing API:

1. Begin a batch and retain one command buffer on the existing stream.
2. Encode each existing kernel with its own arguments into that buffer. Preserve
   all execution and memory dependencies using the required encoder boundaries
   or barriers, and retain referenced resources through completion.
3. Commit at the end of a layer or token. Flush pending work before host-visible
   copies, waits and synchronization; define failure cleanup so pending work and
   resources cannot escape their owners.
4. Keep ordinary eager submission as the default outside the scope.

The initial prototype could keep separate compute encoders and change only
command-buffer creation/commit frequency. It would still perform per-kernel
encoding and resource work, so reducing commit count does not guarantee a
speedup. Implementing a graph API that merely replays the same eager submissions
would likewise not establish the intended improvement.

This requires a change below the public Mojo enqueue wrapper. Replacing MAX's
submission/resource layer with a native Metal implementation would be a larger
project. Neither installed dependencies nor engine/kernel implementations were
modified here. Once a usable backend path exists, resume the declared comparison
with actual command-buffer verification, exact Qwen token/logit/KV checks and
uninstrumented completed-token latency at all three contexts.

## Verification and reproduction

Clean measured source: `df0ad03` (full identity in the receipt). Apple M4 Pro,
Metal, MAX 26.5.0 and Mojo 1.0.0; AC, normal power and nominal thermal state.
Both the diagnostic build and ordinary model benchmark build compile. The full
Python suite passed all 181 tests, including classification of unsupported
backend errors versus incorrect graph results. Receipt replay and
`git diff --check` passed. There is no new throughput or full-model numerical
claim from this small Int32 capability probe.

```sh
uv run --locked python -m llm_mojo.benchmarks.model_profile batch-support --output /private/tmp/new-batch-support-proof
uv run --locked python -m llm_mojo.benchmarks.model_profile batch-support-replay --output studies/model_generation/batch-support.json
```

The collector requires clean source and refuses an existing output directory.
The executable and complete export/build logs remain outside Git in
`/private/tmp/batch-support-proof`.
