"""BF16 residual addition, with explicit input and output storage."""
from layout import TensorLayout, TileTensor
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import ceildiv
from llm_mojo.bf16_arithmetic import add_bits


def residual_reference[
    XL: TensorLayout, BL: TensorLayout, YL: TensorLayout
](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    branch: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
) raises:
    comptime assert (
        x.flat_rank == 2 and branch.flat_rank == 2 and y.flat_rank == 2
    )
    if (
        Int(x.dim[0]()) != Int(branch.dim[0]())
        or Int(x.dim[1]()) != Int(branch.dim[1]())
        or Int(x.dim[0]()) != Int(y.dim[0]())
        or Int(x.dim[1]()) != Int(y.dim[1]())
    ):
        raise Error("residual shapes must agree")
    for r in range(Int(x.dim[0]())):
        for d in range(Int(x.dim[1]())):
            var index = r * Int(x.dim[1]()) + d
            y.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=index] = add_bits(
                x.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=index],
                branch.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=index],
            )


def _residual[
    XL: TensorLayout, BL: TensorLayout, YL: TensorLayout
](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    branch: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
    rows: Int32,
    hidden: Int32,
):
    comptime assert (
        x.flat_rank == 2 and branch.flat_rank == 2 and y.flat_rank == 2
    )
    var i = global_idx.x
    if i < Int(rows) * Int(hidden):
        var r = i // Int(hidden)
        var d = i % Int(hidden)
        y.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i] = add_bits(
            x.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i],
            branch.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i],
        )


def enqueue_residual_apple_gpu[
    XL: TensorLayout, BL: TensorLayout, YL: TensorLayout
](
    ctx: DeviceContext,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    branch: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
) raises:
    """Borrow non-overlapping row-major views; enqueue without synchronization."""
    comptime assert (
        x.flat_rank == 2 and branch.flat_rank == 2 and y.flat_rank == 2
    )
    var r = Int(x.dim[0]())
    var h = Int(x.dim[1]())
    if (
        r <= 0
        or h <= 0
        or Int(x.dim[0]()) != Int(branch.dim[0]())
        or Int(x.dim[1]()) != Int(branch.dim[1]())
        or Int(x.dim[0]()) != Int(y.dim[0]())
        or Int(x.dim[1]()) != Int(y.dim[1]())
    ):
        raise Error("residual shapes must agree and be positive")
    if ctx.api() != "metal":
        raise Error("residual requires Metal")
    ctx.enqueue_function[_residual[XL, BL, YL]](
        x,
        branch,
        y,
        Int32(r),
        Int32(h),
        grid_dim=ceildiv(r * h, 128),
        block_dim=128,
    )
