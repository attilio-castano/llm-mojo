"""Materialized BF16 SiLU and gating, with FP32 arithmetic between stores."""
from layout import TensorLayout, TileTensor
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.memory import bitcast
from std.sys.info import is_gpu
from std.ffi import external_call
from llm_mojo.bf16_arithmetic import multiply_bits
from std.math import ceildiv, exp


def silu_bits(bits: UInt16) -> UInt16:
    # In this interval exp(-x) is exactly 1 in FP32. Integer x/2 with
    # ties-to-even avoids both arithmetic flushing and BF16 lowering casts.
    var magnitude = bits & 0x7FFF
    if magnitude < 256:
        var half = magnitude >> 1
        half += (magnitude & 1) & (half & 1)
        return (bits & 0x8000) | half
    var x = bitcast[DType.float32](UInt32(bits) << 16)
    var exponential: Float32
    comptime if is_gpu():
        exponential = exp(-x)
    else:
        # The host std.math exp overflows early at 88.5 in this toolchain.
        # libm expf retains the pinned FP32 exponential overflow boundary.
        exponential = external_call["expf", Float32](-x)
    return bitcast[DType.uint16](
        (x / (1.0 + exponential)).cast[DType.bfloat16]()
    )


def silu_reference[
    XL: TensorLayout, YL: TensorLayout
](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
) raises:
    comptime assert x.flat_rank == 2 and y.flat_rank == 2
    _validate_shape(x, y)
    for r in range(Int(x.dim[0]())):
        for c in range(Int(x.dim[1]())):
            y.ptr.unsafe_bitcast[UInt16]()[
                unsafe_offset=r * Int(x.dim[1]()) + c
            ] = silu_bits(
                x.ptr.unsafe_bitcast[UInt16]()[
                    unsafe_offset=r * Int(x.dim[1]()) + c
                ]
            )


def multiply_reference[
    XL: TensorLayout, UL: TensorLayout, YL: TensorLayout
](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    u: TileTensor[DType.bfloat16, UL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
) raises:
    comptime assert x.flat_rank == 2 and u.flat_rank == 2 and y.flat_rank == 2
    _validate_shape(x, y)
    _validate_shape(x, u)
    for r in range(Int(x.dim[0]())):
        for c in range(Int(x.dim[1]())):
            var index = r * Int(x.dim[1]()) + c
            y.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=index] = multiply_bits(
                x.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=index],
                u.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=index],
            )


def _validate_shape[
    XL: TensorLayout, YL: TensorLayout
](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
) raises:
    comptime assert x.flat_rank == 2 and y.flat_rank == 2
    if (
        Int(x.dim[0]()) <= 0
        or Int(x.dim[1]()) <= 0
        or Int(x.dim[0]()) != Int(y.dim[0]())
        or Int(x.dim[1]()) != Int(y.dim[1]())
    ):
        raise Error("SwiGLU views must have equal positive shapes")


def _silu[
    XL: TensorLayout, YL: TensorLayout
](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
    rows: Int32,
    width: Int32,
):
    comptime assert x.flat_rank == 2 and y.flat_rank == 2
    var i = global_idx.x
    if i < Int(rows) * Int(width):
        var r = i // Int(width)
        var c = i % Int(width)
        y.ptr.unsafe_bitcast[UInt16]()[
            unsafe_offset=r * Int(x.dim[1]()) + c
        ] = silu_bits(
            x.ptr.unsafe_bitcast[UInt16]()[
                unsafe_offset=r * Int(x.dim[1]()) + c
            ]
        )


def _multiply[
    XL: TensorLayout, UL: TensorLayout, YL: TensorLayout
](
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    u: TileTensor[DType.bfloat16, UL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
    rows: Int32,
    width: Int32,
):
    comptime assert x.flat_rank == 2 and u.flat_rank == 2 and y.flat_rank == 2
    var i = global_idx.x
    if i < Int(rows) * Int(width):
        var r = i // Int(width)
        var c = i % Int(width)
        y.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i] = multiply_bits(
            x.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i],
            u.ptr.unsafe_bitcast[UInt16]()[unsafe_offset=i],
        )


def enqueue_silu_apple_gpu[
    XL: TensorLayout, YL: TensorLayout
](
    ctx: DeviceContext,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
) raises:
    _validate_shape(x, y)
    if ctx.api() != "metal":
        raise Error("SiLU requires Metal")
    var r = Int(x.dim[0]())
    var w = Int(x.dim[1]())
    ctx.enqueue_function[_silu[XL, YL]](
        x, y, Int32(r), Int32(w), grid_dim=ceildiv(r * w, 128), block_dim=128
    )


def enqueue_multiply_apple_gpu[
    XL: TensorLayout, UL: TensorLayout, YL: TensorLayout
](
    ctx: DeviceContext,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    u: TileTensor[DType.bfloat16, UL, MutAnyOrigin],
    y: TileTensor[DType.bfloat16, YL, MutAnyOrigin],
) raises:
    _validate_shape(x, y)
    _validate_shape(x, u)
    if ctx.api() != "metal":
        raise Error("gating multiply requires Metal")
    var r = Int(x.dim[0]())
    var w = Int(x.dim[1]())
    ctx.enqueue_function[_multiply[XL, UL, YL]](
        x, u, y, Int32(r), Int32(w), grid_dim=ceildiv(r * w, 128), block_dim=128
    )
