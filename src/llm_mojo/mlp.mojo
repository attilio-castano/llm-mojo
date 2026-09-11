"""Materialized Qwen MLP. Caller owns storage and one ordered Metal stream."""
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.linear import (
    enqueue_linear_apple_gpu,
    enqueue_linear_rowwise_rows_apple_gpu,
    enqueue_linear_apple_gpu_two_output,
    enqueue_linear_pair_decode_apple_gpu,
    enqueue_linear_cooperative_decode_apple_gpu,
    enqueue_linear_prefill_mma_8x16_apple_gpu,
    enqueue_linear_prefill_mma_tile_apple_gpu,
)
from llm_mojo.swiglu import enqueue_silu_apple_gpu, enqueue_multiply_apple_gpu, enqueue_silu_multiply_apple_gpu
from llm_mojo.residual import enqueue_residual_apple_gpu


def mlp_projection_mapping(mapping: Int, stage: Int) -> Int:
    """0 rowwise; 1/2/3 tile gate/up; 4/5/6 tile down; 7 combines finalists.

    Tile IDs 1/2/3 mean 8x16, 16x16, 8x32. No row-count selector.
    Public entrypoints validate the configuration before enqueue.
    """
    if mapping >= 19:
        return mapping - 12 if stage == 1 or stage == 2 or stage == 5 else 0
    if mapping >= 8:
        var gate_mapping = mapping if mapping <= 10 else 0
        var down_mapping = mapping if mapping == 11 or mapping == 12 else 0
        if mapping >= 13:
            gate_mapping = 8 + (mapping - 13) // 2
            down_mapping = 11 + (mapping - 13) % 2
        if stage == 1 or stage == 2:
            return 4 if gate_mapping == 9 or gate_mapping == 10 else 0
        if stage == 5:
            return down_mapping - 6 if down_mapping else 0
        return 0
    if mapping == 7 and (stage == 1 or stage == 2 or stage == 5):
        return 2  # Both independent screens selected 16x16.
    if (stage == 1 or stage == 2) and mapping <= 3:
        return mapping
    if stage == 5 and mapping >= 4:
        return mapping - 3
    return 0


def _enqueue_projection[
    IL: TensorLayout, WL: TensorLayout, OL: TensorLayout
](
    ctx: DeviceContext,
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OL, MutAnyOrigin],
    mapping: Int,
) raises:
    if mapping == 0:
        enqueue_linear_apple_gpu(ctx, input, weight, output)
    elif mapping == 1:
        enqueue_linear_prefill_mma_8x16_apple_gpu(ctx, input, weight, output)
    elif mapping == 2:
        enqueue_linear_prefill_mma_tile_apple_gpu[16, 16](
            ctx, input, weight, output
        )
    elif mapping == 3:
        enqueue_linear_prefill_mma_tile_apple_gpu[8, 32](
            ctx, input, weight, output
        )

    elif mapping == 4:
        enqueue_linear_apple_gpu_two_output(ctx, input, weight, output)
    elif mapping == 5:
        enqueue_linear_cooperative_decode_apple_gpu[2](
            ctx, input, weight, output
        )
    elif mapping == 6:
        enqueue_linear_cooperative_decode_apple_gpu[4](
            ctx, input, weight, output
        )

    elif mapping == 7:
        enqueue_linear_rowwise_rows_apple_gpu[4](ctx, input, weight, output)
    elif mapping == 8:
        enqueue_linear_rowwise_rows_apple_gpu[8](ctx, input, weight, output)
    elif mapping == 9:
        enqueue_linear_rowwise_rows_apple_gpu[16](ctx, input, weight, output)


def mlp_combines_gate_up(mapping: Int) -> Bool:
    if mapping >= 19:
        return False
    var gate_mapping = 8 + (mapping - 13) // 2 if mapping >= 13 else mapping
    return gate_mapping == 8 or gate_mapping == 10


struct MLPWeights(Movable):
    var hidden: Int
    var intermediate: Int
    var norm: DeviceBuffer[DType.bfloat16]
    var gate: DeviceBuffer[DType.bfloat16]
    var up: DeviceBuffer[DType.bfloat16]
    var down: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        hidden: Int = 896,
        intermediate: Int = 4864,
    ) raises:
        if (
            hidden <= 0
            or intermediate <= 0
            or hidden > 2147483647 // intermediate
        ):
            raise Error("invalid MLP weight dimensions")
        self.hidden = hidden
        self.intermediate = intermediate
        self.norm = ctx.enqueue_create_buffer[DType.bfloat16](hidden)
        self.gate = ctx.enqueue_create_buffer[DType.bfloat16](
            hidden * intermediate
        )
        self.up = ctx.enqueue_create_buffer[DType.bfloat16](
            hidden * intermediate
        )
        self.down = ctx.enqueue_create_buffer[DType.bfloat16](
            hidden * intermediate
        )


struct MLPWorkspace(Movable):
    var max_rows: Int
    var hidden: Int
    var intermediate: Int
    var normalized: DeviceBuffer[DType.bfloat16]
    var gate: DeviceBuffer[DType.bfloat16]
    var up: DeviceBuffer[DType.bfloat16]
    var activated: DeviceBuffer[DType.bfloat16]
    var gated: DeviceBuffer[DType.bfloat16]
    var down: DeviceBuffer[DType.bfloat16]
    var output: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        max_rows: Int,
        hidden: Int = 896,
        intermediate: Int = 4864,
    ) raises:
        if (
            max_rows <= 0
            or max_rows > 4096
            or hidden <= 0
            or intermediate <= 0
            or hidden > 2147483647 // max_rows
            or intermediate > 2147483647 // max_rows
        ):
            raise Error("invalid MLP workspace dimensions")
        self.max_rows = max_rows
        self.hidden = hidden
        self.intermediate = intermediate
        self.normalized = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * hidden
        )
        self.gate = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * intermediate
        )
        self.up = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * intermediate
        )
        self.activated = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * intermediate
        )
        self.gated = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * intermediate
        )
        self.down = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * hidden)
        self.output = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * hidden
        )


def _validate_mlp[
    XL: TensorLayout
](
    ctx: DeviceContext,
    weights: MLPWeights,
    work: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    mapping: Int,
) raises:
    comptime assert x.flat_rank == 2
    if mapping < 0 or mapping > 21:
        raise Error("unknown MLP projection mapping")
    var r = Int(x.dim[0]())
    if mapping >= 8 and mapping <= 18 and r != 1:
        raise Error("MLP decode mapping requires one row")
    if (
        r <= 0
        or r > work.max_rows
        or Int(x.dim[1]()) != weights.hidden
        or work.hidden != weights.hidden
        or work.intermediate != weights.intermediate
    ):
        raise Error("MLP rows, weight dimensions and workspace must agree")
    if ctx.api() != "metal":
        raise Error("MLP requires Metal")


def _enqueue_mlp_stage[
    XL: TensorLayout
](
    ctx: DeviceContext,
    mut weights: MLPWeights,
    mut work: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    stage: Int,
    mapping: Int,
) raises:
    """Internal stage dispatch after preflight. Stage inputs are caller-visible.
    """
    var r = Int(x.dim[0]())
    var h = weights.hidden
    var i = weights.intermediate
    var normal = TileTensor(work.normalized, row_major(r, h))
    var gate = TileTensor(work.gate, row_major(r, i))
    var up = TileTensor(work.up, row_major(r, i))
    var activated = TileTensor(work.activated, row_major(r, i))
    var gated = TileTensor(work.gated, row_major(r, i))
    var down = TileTensor(work.down, row_major(r, h))
    if stage == 0:
        enqueue_rms_norm_apple_gpu(
            ctx, x, TileTensor(weights.norm, row_major(h)), normal
        )
    elif stage == 1:
        _enqueue_projection(
            ctx,
            normal,
            TileTensor(weights.gate, row_major(i, h)),
            gate,
            mlp_projection_mapping(mapping, stage),
        )
    elif stage == 2:
        _enqueue_projection(
            ctx,
            normal,
            TileTensor(weights.up, row_major(i, h)),
            up,
            mlp_projection_mapping(mapping, stage),
        )
    elif stage == 3:
        enqueue_silu_apple_gpu(ctx, gate, activated)
    elif stage == 4:
        enqueue_multiply_apple_gpu(ctx, activated, up, gated)
    elif stage == 5:
        _enqueue_projection(
            ctx,
            gated,
            TileTensor(weights.down, row_major(h, i)),
            down,
            mlp_projection_mapping(mapping, stage),
        )
    elif stage == 6:
        enqueue_residual_apple_gpu(
            ctx, x, down, TileTensor(work.output, row_major(r, h))
        )


def enqueue_mlp_stage_apple_gpu[
    XL: TensorLayout
](
    ctx: DeviceContext,
    mut weights: MLPWeights,
    mut work: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    stage: Int,
    mapping: Int = 0,
) raises:
    """Isolated operation on workspace inputs, for numerical checks and study.
    """
    if stage < 0 or stage > 6:
        raise Error("unknown MLP stage")
    _validate_mlp(ctx, weights, work, x, mapping)
    _enqueue_mlp_stage(ctx, weights, work, x, stage, mapping)


def enqueue_mlp_apple_gpu[
    XL: TensorLayout
](
    ctx: DeviceContext,
    mut weights: MLPWeights,
    mut work: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    mapping: Int = 0,
    fuse_activation: Bool = False,
) raises:
    """Six or seven ordered dispatches; no allocation, upload, or synchronization.

    X and weights must not overlap any writable workspace. All buffers live on
    this stream through completion. Consume output before workspace reuse.
    Mapping 0 retains rowwise projections. 1/2/3 change only gate/up and
    4/5/6 change only down to 8x16/16x16/8x32 MMA, at every row count.
    Mapping 7 uses the independently selected 16x16 tile for all projections.
    Decode-only 8/9/10 combine gate/up launches and/or use two outputs/group;
    11/12 cooperate with two/four groups per down output. 13..18 compose them.
    Mapping 19 reuses each weight across four independent rowwise reductions.
    Isolated stage APIs always enqueue exactly that stage, even for a combined
    mapping; the complete entrypoint combines stages 1/2 into one dispatch.
    Optional single-row mapping-zero activation fusion retains the BF16 SiLU
    bits in registers and leaves work.activated untouched.
    """
    _validate_mlp(ctx, weights, work, x, mapping)
    if fuse_activation and (Int(x.dim[0]()) != 1 or mapping != 0):
        raise Error("fused MLP activation requires one row and mapping zero")
    for stage in range(7):
        if stage == 3 and fuse_activation:
            var i = weights.intermediate
            enqueue_silu_multiply_apple_gpu(ctx,
                TileTensor(work.gate, row_major(1, i)),
                TileTensor(work.up, row_major(1, i)),
                TileTensor(work.gated, row_major(1, i)))
        elif stage == 4 and fuse_activation:
            continue
        elif stage == 1 and mlp_combines_gate_up(mapping):
            var h = weights.hidden
            var i = weights.intermediate
            var normal = TileTensor(work.normalized, row_major(1, h))
            var wg = TileTensor(weights.gate, row_major(i, h))
            var wu = TileTensor(weights.up, row_major(i, h))
            var g = TileTensor(work.gate, row_major(1, i))
            var u = TileTensor(work.up, row_major(1, i))
            if mlp_projection_mapping(mapping, 1) == 4:
                enqueue_linear_pair_decode_apple_gpu[True](
                    ctx, normal, wg, wu, g, u
                )
            else:
                enqueue_linear_pair_decode_apple_gpu[False](
                    ctx, normal, wg, wu, g, u
                )
        elif stage != 2 or not mlp_combines_gate_up(mapping):
            _enqueue_mlp_stage(ctx, weights, work, x, stage, mapping)
