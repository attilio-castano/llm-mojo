"""How one Qwen model call executes: a decoder configuration plus decode features.

`fast` is the measured Apple M4 Pro lookup: the runtime study's multi-row cells
(studies/model_generation/runtime-selection.csv) and, for single-token decode,
configuration 26 composed with GPU argmax, buffer swapping and residual/RMSNorm
fusion (studies/model_generation/residual-norm.md). Unmeasured shapes and every
other device use the baseline configuration. `baseline` and `consistent` are
fixed research routes kept for comparison; see docs/generation.md.
"""
from std.sys import get_defined_int
from llm_mojo.layers.decoder_layer import (
    DECODER_BASELINE, DECODER_SPLIT8, DECODER_SPLIT8_TILED, DECODER_CONSISTENT,
    DECODER_CONSISTENT_MMA, DECODER_FUSED_DECODE, decoder_mappings,
)

comptime MEASURED_DEVICE = "Apple M4 Pro"
comptime MAX_CONTEXT = 4096
# The decode composition's projection arrangement (kernels/linear.mojo), for one
# sequence and for many. 8 gives each SIMD group four rows by four columns with
# fixed widths, and each lane sums four adjacent products in every 128 inputs;
# 1e selected and confirmed it against arrangement 5, and it was adopted on
# 2026-09-27. Its one-row path sums in the same order, so batched rows equal
# solo rows, but the order differs from the one-row kernel's (docs/model.md,
# decode projection order). Exact arrangements 0-7 and 11 keep the one-row
# kernel's arithmetic; decode parity and the route test use 5. Validation
# builds may select another arrangement with -D DECODE_PROJECTION=N.
comptime DECODE_PROJECTION = get_defined_int["DECODE_PROJECTION", default=8]()


def _kv_block_size[size: Int]() -> Int:
    comptime assert size > 0 and size <= MAX_CONTEXT and size % 32 == 0, (
        "KV_BLOCK_SIZE must be a multiple of 32 up to MAX_CONTEXT")
    return size


# Chat, generation and the batch validation hold each sequence's K/V in blocks of
# KV_BLOCK_SIZE slots, slot-major unless KV_HEAD_MAJOR. The paged KV study selected
# and confirmed 32-slot slot-major blocks, adopted on 2026-10-05
# (studies/model_generation/paged-kv-loop.md); a layout changes where a row lives,
# never a result. Builds may select another multiple of 32 with -D KV_BLOCK_SIZE=N,
# MAX_CONTEXT holding a sequence in one block, and head-major order with
# -D KV_HEAD_MAJOR=1.
comptime KV_BLOCK_SIZE = _kv_block_size[get_defined_int["KV_BLOCK_SIZE", default=32]()]()
comptime KV_HEAD_MAJOR = get_defined_int["KV_HEAD_MAJOR", default=0]() == 1


@fieldwise_init
struct ExecutionPlan(ImplicitlyCopyable, Movable):
    """One call's decoder configuration and decode features.

    Configuration 26 always carries all three features and one row per
    sequence; no other configuration carries any or steps more than one
    sequence. The unpromoted compositions measured in the decode studies are not
    expressible.
    """
    var configuration: Int
    var gpu_argmax: Bool
    var swap_buffers: Bool
    var fuse_residual_norm: Bool

    def validate(self, rows: Int, sequences: Int = 1) raises:
        var fused = self.configuration == DECODER_FUSED_DECODE
        var features = self.gpu_argmax or self.swap_buffers or self.fuse_residual_norm
        if fused:
            if not (rows == sequences and self.gpu_argmax and self.swap_buffers and self.fuse_residual_norm):
                raise Error("configuration 26 requires one row per sequence with GPU argmax, buffer swap and residual/RMSNorm fusion")
            return
        if features:
            raise Error("decode features are measured only together with configuration 26")
        if sequences != 1:
            raise Error("only configuration 26 steps several sequences")
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


def _check_batch(rows: Int, total: Int, sequences: Int) raises:
    if rows != sequences or total < 1 or total > MAX_CONTEXT:
        raise Error("a batched decode step has one row per sequence and at most "
                    + String(MAX_CONTEXT) + " tokens in its longest sequence")


def fast_plan(rows: Int, total: Int, device: String, sequences: Int = 1) raises -> ExecutionPlan:
    """The measured lookup. A decode step of several sequences, one row each, takes
    configuration 26 on the measured device; total is then its longest sequence."""
    if sequences > 1:
        _check_batch(rows, total, sequences)
        if device != MEASURED_DEVICE:
            raise Error("batched decode runs only on the measured device")
        return ExecutionPlan(DECODER_FUSED_DECODE, True, True, True)
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


def configured_plan(configuration: Int, rows: Int, total: Int, sequences: Int = 1) raises -> ExecutionPlan:
    """An explicit retained configuration for diagnostics; 26 carries its decode features."""
    if sequences > 1:
        _check_batch(rows, total, sequences)
    else:
        _check(rows, total)
    var fused = configuration == DECODER_FUSED_DECODE
    var plan = ExecutionPlan(configuration, fused, fused, fused)
    plan.validate(rows, sequences)
    return plan^


def execution_plan(mode: String, rows: Int, total: Int, device: String) raises -> ExecutionPlan:
    if mode == "fast":
        return fast_plan(rows, total, device)
    if mode == "baseline":
        return baseline_plan(rows, total)
    if mode == "consistent":
        return consistent_plan(rows, total)
    raise Error("unknown execution mode '" + mode + "': use fast, baseline or consistent")
