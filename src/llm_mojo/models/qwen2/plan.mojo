"""How one Qwen model call executes: a decoder configuration plus decode features.

`fast` is the measured Apple M4 Pro lookup: the runtime study's multi-row cells
(studies/model_generation/runtime-selection.csv) and, for single-token decode,
configuration 26 composed with GPU argmax, buffer swapping and residual/RMSNorm
fusion (studies/model_generation/residual-norm.md). Unmeasured shapes and every
other device use the baseline configuration. `baseline` and `consistent` are
fixed research routes kept for comparison; see docs/generation.md.
"""
from llm_mojo.layers.decoder_layer import (
    DECODER_BASELINE, DECODER_SPLIT8, DECODER_SPLIT8_TILED, DECODER_CONSISTENT,
    DECODER_CONSISTENT_MMA, DECODER_FUSED_DECODE, decoder_mappings,
)

comptime MEASURED_DEVICE = "Apple M4 Pro"
comptime MAX_CONTEXT = 4096


@fieldwise_init
struct ExecutionPlan(ImplicitlyCopyable, Movable):
    """One call's decoder configuration and single-row decode features.

    Configuration 26 always carries all three features and exactly one row;
    no other configuration carries any. The unpromoted compositions measured in
    the decode studies are not expressible.
    """
    var configuration: Int
    var gpu_argmax: Bool
    var swap_buffers: Bool
    var fuse_residual_norm: Bool

    def validate(self, rows: Int) raises:
        var fused = self.configuration == DECODER_FUSED_DECODE
        var features = self.gpu_argmax or self.swap_buffers or self.fuse_residual_norm
        if fused and not (rows == 1 and self.gpu_argmax and self.swap_buffers and self.fuse_residual_norm):
            raise Error("configuration 26 requires one row with GPU argmax, buffer swap and residual/RMSNorm fusion")
        if not fused and features:
            raise Error("decode features are measured only together with configuration 26")
        _ = decoder_mappings(self.configuration, rows)


def _check(rows: Int, total: Int) raises:
    if rows < 1 or total < rows or total > MAX_CONTEXT:
        raise Error("invalid model call: rows " + String(rows) + ", total " + String(total))


def fast_prefill_configuration(rows: Int, total: Int) -> Int:
    """Multi-row cells measured in the complete model on Apple M4 Pro."""
    if rows == 16 and total == 256:
        return DECODER_CONSISTENT_MMA
    if (rows == 16 and (total == 1024 or total == 4096)) or (total == 256 and (rows == 15 or rows == 17)):
        return DECODER_SPLIT8
    if ((rows == 64 or rows == 256) and (total == 1024 or total == 4096)) or (
        total == 4096 and (rows == 65 or rows == 255)
    ):
        return DECODER_SPLIT8_TILED
    return DECODER_BASELINE


def fast_plan(rows: Int, total: Int, device: String) raises -> ExecutionPlan:
    _check(rows, total)
    if device != MEASURED_DEVICE:
        return ExecutionPlan(DECODER_BASELINE, False, False, False)
    if rows == 1:
        return ExecutionPlan(DECODER_FUSED_DECODE, True, True, True)
    return ExecutionPlan(fast_prefill_configuration(rows, total), False, False, False)


def baseline_plan(rows: Int, total: Int) raises -> ExecutionPlan:
    _check(rows, total)
    return ExecutionPlan(DECODER_BASELINE, False, False, False)


def consistent_plan(rows: Int, total: Int) raises -> ExecutionPlan:
    """FP32 G32 attention and rowwise projections at every row count (research)."""
    _check(rows, total)
    return ExecutionPlan(DECODER_CONSISTENT, False, False, False)


def configured_plan(configuration: Int, rows: Int, total: Int) raises -> ExecutionPlan:
    """An explicit retained configuration for diagnostics; 26 carries its decode features."""
    _check(rows, total)
    var fused = configuration == DECODER_FUSED_DECODE
    var plan = ExecutionPlan(configuration, fused, fused, fused)
    plan.validate(rows)
    return plan^


def execution_plan(mode: String, rows: Int, total: Int, device: String) raises -> ExecutionPlan:
    if mode == "fast":
        return fast_plan(rows, total, device)
    if mode == "baseline":
        return baseline_plan(rows, total)
    if mode == "consistent":
        return consistent_plan(rows, total)
    raise Error("unknown execution mode '" + mode + "': use fast, baseline or consistent")
