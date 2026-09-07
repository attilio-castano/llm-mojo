"""Finite BF16 arithmetic with exact handling of Metal subnormal boundaries."""
from std.memory import bitcast


def round_shift(value: UInt32, shift: UInt32) -> UInt32:
    if shift == 0:
        return value
    if shift >= 32:
        return 0
    var quotient = value >> shift
    var remainder = value & ((UInt32(1) << shift) - 1)
    var midpoint = UInt32(1) << (shift - 1)
    if remainder > midpoint or (remainder == midpoint and (quotient & 1) != 0):
        quotient += 1
    return quotient


def multiply_bits(a: UInt16, b: UInt16) -> UInt16:
    var sign = (UInt32(a) ^ UInt32(b)) & 0x8000
    var ea = (UInt32(a) >> 7) & 255
    var eb = (UInt32(b) >> 7) & 255
    if ea > 0 and eb > 0 and ea + eb > 128:
        var x = bitcast[DType.float32](UInt32(a) << 16)
        var y = bitcast[DType.float32](UInt32(b) << 16)
        return bitcast[DType.uint16]((x * y).cast[DType.bfloat16]())
    # A BF16 product has at most sixteen significand bits. For the narrow
    # underflow/input-subnormal path, compute that exact product as an integer
    # and round once, preserving the FP32-product/BF16-store contract.
    var ma = UInt32(a) & 127
    var mb = UInt32(b) & 127
    if ea != 0:
        ma |= 128
    else:
        ea = 1
    if eb != 0:
        mb |= 128
    else:
        eb = 1
    var product = ma * mb
    if product == 0:
        return UInt16(sign)
    var top: UInt32 = 0
    var remaining = product
    while remaining > 1:
        remaining >>= 1
        top += 1
    var exponent = Int32(ea + eb) - 141 + Int32(top)
    if exponent <= 0:
        var subnormal: UInt32
        if ea + eb >= 135:
            subnormal = product << (ea + eb - 135)
        else:
            subnormal = round_shift(product, 135 - ea - eb)
        return UInt16(sign | subnormal)
    var mantissa: UInt32
    if top >= 7:
        mantissa = round_shift(product, top - 7)
    else:
        mantissa = product << (7 - top)
    if mantissa == 256:
        mantissa = 128
        exponent += 1
    if exponent >= 255:
        return UInt16(sign | 0x7F80)
    return UInt16(sign | (UInt32(exponent) << 7) | (mantissa - 128))


def add_bits(a: UInt16, b: UInt16) -> UInt16:
    var ea = (UInt32(a) >> 7) & 255
    var eb = (UInt32(b) >> 7) & 255
    if ea > 9 or eb > 9:
        var x = bitcast[DType.float32](UInt32(a) << 16)
        var y = bitcast[DType.float32](UInt32(b) << 16)
        return bitcast[DType.uint16]((x + y).cast[DType.bfloat16]())
    # Below exponent 10 both operands are integer multiples of 2^-133 and
    # their sum fits in eighteen signed bits. Exponent 9 is needed because
    # spacing halves immediately below a power of two: 0x0480 + 0x807f must
    # round to 0x047f. At larger exponents even that half-binade boundary
    # cannot be crossed by an input subnormal; nonzero cancellation results
    # are also FP32-normal.
    var ma = UInt32(a) & 127
    var mb = UInt32(b) & 127
    if ea != 0:
        ma = (ma | 128) << (ea - 1)
    if eb != 0:
        mb = (mb | 128) << (eb - 1)
    var ia = -Int32(ma) if (a & 0x8000) != 0 else Int32(ma)
    var ib = -Int32(mb) if (b & 0x8000) != 0 else Int32(mb)
    var total = ia + ib
    if total == 0:
        return (a & b) & 0x8000
    var sign: UInt32 = 0x8000 if total < 0 else 0
    var magnitude = UInt32(-total if total < 0 else total)
    if magnitude < 128:
        return UInt16(sign | magnitude)
    var top: UInt32 = 0
    var remaining = magnitude
    while remaining > 1:
        remaining >>= 1
        top += 1
    var mantissa = round_shift(magnitude, top - 7)
    var exponent = top - 6
    if mantissa == 256:
        mantissa = 128
        exponent += 1
    return UInt16(sign | (exponent << 7) | (mantissa - 128))
