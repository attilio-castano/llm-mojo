"""Residual addition and RMSNorm per row, with exact stored boundaries.

One 128-thread threadgroup owns one row, so every row keeps the reduction order
of the existing single-row norm whatever the row count.
"""
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.collections import InlineArray
from std.gpu import WARP_SIZE, block_idx, thread_idx, lane_id
from std.gpu.primitives import warp
from std.math import rsqrt
from std.memory import bitcast
from std.sys.info import is_apple_gpu
from llm_mojo.kernels.bf16_arithmetic import add_bits
from llm_mojo.kernels.rms_norm import RMS_NORM_EPSILON


def _residual_norm[HIDDEN: Int, XL: TensorLayout, BL: TensorLayout, WL: TensorLayout,
                   YL: TensorLayout, NL: TensorLayout](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    branch: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    residual: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
    normal: TileTensor[DType.bfloat16, NL, MutAnyOrigin],
    hidden_size: Int32,
):
    comptime assert is_apple_gpu()
    comptime assert x.flat_rank == 2 and branch.flat_rank == 2
    comptime assert residual.flat_rank == 2 and normal.flat_rank == 2 and weight.flat_rank == 1
    comptime CHUNKS = HIDDEN // 128
    comptime assert HIDDEN % 128 == 0 and CHUNKS <= 8, "a row is at most eight 128-element chunks"
    var row = block_idx.x
    var start = row * HIDDEN
    var thread = thread_idx.x
    var lane = lane_id()
    var values = SIMD[DType.bfloat16, 8](0)
    var sum_of_squares: Float32 = 0
    # Same values and accumulation order as the existing 128-thread norm.
    comptime for chunk in range(CHUNKS):
        var column = thread + chunk * 128
        var bits = add_bits(x.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=start+column],
                            branch.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=start+column])
        residual.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=start+column] = bits
        var value = bitcast[DType.bfloat16](bits)
        values[chunk] = value
        var fp32 = value.cast[DType.float32]()
        sum_of_squares += fp32 * fp32
    var simd_sum = warp.sum(sum_of_squares)
    var groups = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[128 // WARP_SIZE]())
    comptime assert groups.flat_rank == 1
    if lane == 0:
        groups[thread // WARP_SIZE] = rebind[groups.ElementType](simd_sum)
    barrier()
    var partial: Float32 = 0
    if thread < 128 // WARP_SIZE:
        partial = rebind[Float32](groups[thread])
    var sum = warp.sum(partial)
    if thread == 0:
        groups[0] = rebind[groups.ElementType](rsqrt(sum / Float32(Int(hidden_size)) + RMS_NORM_EPSILON))
    barrier()
    var inverse_rms = rebind[Float32](groups[0])
    comptime for chunk in range(CHUNKS):
        var column = thread + chunk * 128
        var normalized = (values[chunk].cast[DType.float32]() * inverse_rms).cast[DType.bfloat16]()
        var scale = rebind[Scalar[DType.bfloat16]](weight[column])
        normal[row,column] = rebind[normal.ElementType](normalized * scale)


def enqueue_residual_norm[HIDDEN: Int, XL: TensorLayout, BL: TensorLayout, WL: TensorLayout,
                          YL: TensorLayout, NL: TensorLayout](
    ctx: DeviceContext,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    branch: TileTensor[DType.bfloat16, BL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    residual: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
    normal: TileTensor[DType.bfloat16, NL, MutAnyOrigin],
) raises:
    """Write both disjoint outputs for every row; callers keep all five contiguous views live."""
    comptime assert x.flat_rank == 2 and branch.flat_rank == 2
    comptime assert residual.flat_rank == 2 and normal.flat_rank == 2 and weight.flat_rank == 1
    var rows = Int(x.dim[0]())
    if (ctx.api() != "metal" or rows < 1 or Int(x.dim[1]()) != HIDDEN
        or Int(branch.dim[0]()) != rows or Int(branch.dim[1]()) != HIDDEN
        or Int(residual.dim[0]()) != rows or Int(residual.dim[1]()) != HIDDEN
        or Int(normal.dim[0]()) != rows or Int(normal.dim[1]()) != HIDDEN or Int(weight.dim[0]()) != HIDDEN):
        raise Error("residual RMSNorm requires matching [rows, hidden] views on Metal")
    if (Int(x.layout.stride[0]().product()) != HIDDEN or Int(x.layout.stride[1]().product()) != 1
        or Int(branch.layout.stride[0]().product()) != HIDDEN or Int(branch.layout.stride[1]().product()) != 1
        or Int(residual.layout.stride[0]().product()) != HIDDEN or Int(residual.layout.stride[1]().product()) != 1
        or Int(normal.layout.stride[1]().product()) != 1 or Int(weight.layout.stride[0]().product()) != 1):
        raise Error("residual RMSNorm requires contiguous rows")
    var pointers = InlineArray[Int, 5](uninitialized=True)
    pointers[0] = Int(residual.ptr)
    pointers[1] = Int(normal.ptr)
    pointers[2] = Int(x.ptr)
    pointers[3] = Int(branch.ptr)
    pointers[4] = Int(weight.ptr)
    var extents = InlineArray[Int, 5](uninitialized=True)
    for i in range(4):
        extents[i] = rows * HIDDEN * 2
    extents[4] = HIDDEN * 2
    for left in range(2):
        for right in range(left + 1, 5):
            if pointers[left] < pointers[right] + extents[right] and pointers[right] < pointers[left] + extents[left]:
                raise Error("residual RMSNorm output overlaps live storage")
    ctx.enqueue_function[_residual_norm[HIDDEN,XL,BL,WL,YL,NL]](x,branch,weight,residual,normal,Int32(HIDDEN),
                                                               grid_dim=rows,block_dim=128)
