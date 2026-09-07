"""Materialized Qwen MLP. Caller owns storage and one ordered Metal stream."""
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.linear import enqueue_linear_apple_gpu
from llm_mojo.swiglu import enqueue_silu_apple_gpu, enqueue_multiply_apple_gpu
from llm_mojo.residual import enqueue_residual_apple_gpu


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
) raises:
    comptime assert x.flat_rank == 2
    var r = Int(x.dim[0]())
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
) raises:
    """Internal stage dispatch after preflight. Stage inputs are caller-visible."""
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
        enqueue_linear_apple_gpu(
            ctx, normal, TileTensor(weights.gate, row_major(i, h)), gate
        )
    elif stage == 2:
        enqueue_linear_apple_gpu(
            ctx, normal, TileTensor(weights.up, row_major(i, h)), up
        )
    elif stage == 3:
        enqueue_silu_apple_gpu(ctx, gate, activated)
    elif stage == 4:
        enqueue_multiply_apple_gpu(ctx, activated, up, gated)
    elif stage == 5:
        enqueue_linear_apple_gpu(
            ctx, gated, TileTensor(weights.down, row_major(h, i)), down
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
) raises:
    """Isolated operation on workspace inputs, for numerical checks and study."""
    if stage < 0 or stage > 6:
        raise Error("unknown MLP stage")
    _validate_mlp(ctx, weights, work, x)
    _enqueue_mlp_stage(ctx, weights, work, x, stage)


def enqueue_mlp_apple_gpu[
    XL: TensorLayout
](
    ctx: DeviceContext,
    mut weights: MLPWeights,
    mut work: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
) raises:
    """Seven ordered dispatches; no allocation, upload, or synchronization.

    X and weights must not overlap any writable workspace. All buffers live on
    this stream through completion. Consume output before workspace reuse.
    """
    _validate_mlp(ctx, weights, work, x)
    for stage in range(7):
        _enqueue_mlp_stage(ctx, weights, work, x, stage)
