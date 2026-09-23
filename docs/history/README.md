# History

These pages record plans and investigations that are complete or superseded.
They are kept as written, apart from link paths, so that the reasoning behind
the current engine stays inspectable. None of them describes current behavior;
each entry names the documents that do, and the
[documentation map](../README.md) lists all current guidance.

| Record | What it was | Outcome | Current guidance |
| --- | --- | --- | --- |
| [Qwen forward and generation plans](generation-plan.md) | The approved scope and stop gates for the first full-model forward pass and generation, with the qualification that stopped it | Replaced by the Fast plan below | [Model contract](../model.md), [runtime guide](../generation.md) |
| [Fast full-model implementation plan](fast-generation-plan.md) | The plan that moved full-model numerical comparisons from gates to diagnostics, and its earlier failed qualification | Fast runtime and native chat (#20) | [Runtime guide](../generation.md), [runtime study](../../studies/model_generation/runtime.md) |
| [Decoder layer implementation plan](decoder-layer-plan.md) | Task-by-task plan for one validated Qwen decoder layer and its baseline measurement | Composed decoder and selected baselines (#17) | [Decoder layer contract](../decoder-layer.md), [decoder study](../../studies/decoder_layer/README.md) |
| [GQA prefill resource study plan](gqa-prefill-resources.md) | The bounded plan and frozen selection for five compiler-resource ablations of GQA prefill | Completed with the prefill optimization (#13) | [GQA prefill results](../../studies/gqa_prefill/README.md#compiler-resources-and-synchronization) |
| [Numerical-policy studies before Fast](numerical-policy-studies.md) | The tolerance-gated qualification, the consistency candidate and the HF/PyTorch backend study, moved from the runtime guide | Superseded by the diagnostic policy | [Correctness and diagnostic policy](../model.md#correctness-and-diagnostic-policy) |
| [Decoder layer implementation record](decoder-layer-implementation.md) | The reference-package handoff, the Mojo implementation checkpoint and the completed baseline, moved from the decoder layer contract | Composed decoder and selected baselines (#17) | [Decoder layer contract](../decoder-layer.md), [decoder study](../../studies/decoder_layer/README.md) |
| [MLP sublayer plans](mlp-sublayer-plans.md) | The reference-readiness plan and the implementation and baseline plan, moved from the MLP contract | Qwen SwiGLU MLP and its optimization study (#16) | [MLP contract](../mlp-sublayer.md), [MLP study](../../studies/mlp_sublayer/README.md) |
| [Decode experiment summaries](decode-experiments.md) | Summaries of the GPU token selection, buffer ownership, residual/RMSNorm and projection scheduling experiments, moved from the model contract | The composed Fast decode route (#21) | [Workload policy](../generation.md#workload-policy), [studies index](../../studies/README.md) |
| [CLI and configuration cleanup plan](cli-cleanup-plan.md) | The plan that unified the command line and organized engine and validation ownership | Executed in #22 | [Commands and configuration](../cli.md) |
