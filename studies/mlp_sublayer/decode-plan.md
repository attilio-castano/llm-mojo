# MLP single-token decode campaign

Approved scope: local code, fixtures, Metal runs/profiles, documentation and
commits. Base 5d6a3ee. R=1, H=896, I=4864. No default change or publication.

Frozen matrix before candidate outputs:
- 0: separate gate/up, one output/group; rowwise down.
- 8: combined gate/up launch, one output/group; rowwise down.
- 9: separate gate/up, two outputs/group; rowwise down.
- 10: combined gate/up launch, two outputs/group; rowwise down.
- 11/12: original gate/up; down uses two/four cooperating groups per output.
- At most one combination of qualifying gate/up and down finalists. IDs
  13..18 encode (gate variant 8..10, down variant 11..12), in that order.

Combined launch addresses existing separate weights and outputs using a
second grid dimension. There is no repacking, duplicate weight allocation,
output copy or setup work. This is launch packing; arithmetic and G/U stores
remain separate. Two-output mapping reuses each input load with two FP32
accumulators. Down uses 128 threads/block, two or one outputs/block, four
FP32 shared partials, one unconditional barrier and deterministic merging.
No atomic reduction, extra dispatch or global partial buffer.

Numerics: frozen operation and composed gates in docs/mlp-sublayer.md.
N/G/U/D local atol=rtol=2^-7; composed D=2^-6, Y=2^-5; exact isolated
multiply/residual and frozen SiLU rules. All BF16 materialization retained.
Existing cases are regression data. Fresh seeds 4051/4057/4073, R=1,
and last post-attention row of the reserved checkpoint prompt are declared
in tests/fixtures/mlp_decode_holdout.json before candidate outputs. Freeze
final source/binary before generating these four holdouts. Failure disqualifies;
no holdout-driven tuning or gate changes.

Both screens measure whole MLP at R=1 in hot and ring24 modes, using the
existing frozen seed-1601 prefix and calibrated four-block 10-warmup/10-sample
protocol, reversed arm/workload order, and all samples including self-pairs.
Each family qualifies only when both modes are faster under the existing
>5% and matching self-pair noise rule. Select minimum worst-mode ratio, then
lower variant ID. Combine only qualified families; if neither qualifies,
finish with the control. Final fresh direct comparison has control self-pairs
and tests each qualifying component plus their combination (if both qualify).
No new candidates or repeated timing to chase significance. Final claims use
this confirmation run; failing confirmation leaves the control selected.

Validate all candidate R=1 development inputs, checkpoint-derived row inputs,
guards, unwritten outputs, invalid multi-row preflight, input preservation,
and asynchronous reuse with differing input rows. Validate unchanged existing
multi-row routes in the complete repository workflow before source commits.
GPU execution remains sequential; profile runs are separate from latency.
Profile control and final eligible design at R=1, 10 warmups/500 iterations;
if none qualifies, profile control and the fastest observed rejected design
as a diagnostic, clearly not a selected optimization. Prove dispatch identity
(6 combined launches or 7 separate), Metal device, source/binary/fixture hashes.

Retain compact timing/numerical/dispatch samples and provenance, regenerate
figures, and explain group ownership, requested bytes and merging costs.
Ring24 is not a decoder stack or guaranteed cold DRAM. End with a documented
positive, negative or inconclusive result and local commits. Hardware/tooling
failure that cannot be repaired within scope is reported as incomplete evidence.
