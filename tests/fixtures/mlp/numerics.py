"""Independent BF16 diagnostics. No Torch or Mojo arithmetic is reused here."""

import numpy as np


def bf16_bits(x):
    """Bit patterns of values already stored as BF16 (represented by FP32)."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    if np.any(bits & 0xffff):
        raise ValueError('expected exactly representable BF16 values')
    return (bits >> 16).astype(np.uint16)


def from_bits(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def round_bf16(x):
    """Direct FP64 -> BF16, ties to even; preserves signed zero.

    Binary scaling is exact, then rint rounds an at-most-8-bit significand.
    In particular, there is no intermediate FP32 rounding at a BF16 midpoint.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('diagnostic rounding expects finite FP64 inputs')
    magnitude = np.abs(x)
    _, exponent = np.frexp(magnitude)
    shift = np.maximum(exponent - 8, -133)
    with np.errstate(over='ignore', under='ignore'):
        rounded = np.ldexp(np.rint(np.ldexp(magnitude, -shift)), shift)
        rounded = np.where(rounded >= 2.0**128, np.inf, rounded)
        return np.copysign(rounded, x).astype(np.float32)


def silu64(x):
    """Stable real-formula diagnostic; not the FP32 exponential-overflow policy."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(under='ignore'):
        e = np.exp(-np.abs(x))
    return np.where(x >= 0, x / (1 + e), x * e / (1 + e))


def fp64_stages(inputs, captured=None):
    """Local diagnostics use upstream operands; composition propagates its own."""
    x = inputs['X'].astype(np.float64)
    normal = round_bf16(x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6))
    n = round_bf16(normal.astype(np.float64) * inputs['norm'])
    n_input = n if captured is None else captured['N']
    g = round_bf16(n_input.astype(np.float64) @ inputs['gate'].astype(np.float64).T)
    u = round_bf16(n_input.astype(np.float64) @ inputs['up'].astype(np.float64).T)
    g_input = g if captured is None else captured['G']
    a = round_bf16(silu64(g_input))
    a_input = a if captured is None else captured['A']
    u_input = u if captured is None else captured['U']
    s = round_bf16(a_input.astype(np.float64) * u_input)
    s_input = s if captured is None else captured['S']
    d = round_bf16(s_input.astype(np.float64) @ inputs['down'].astype(np.float64).T)
    d_input = d if captured is None else captured['D']
    y = round_bf16(x + d_input)
    return dict(N=n, G=g, U=u, A=a, S=s, D=d, Y=y)


def differences(actual, reference, budget=None):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError('comparison needs equal nonempty shapes')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('nonfinite comparison operand')
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    err = np.abs(a-r)
    scaled = err / (1+np.abs(r))
    worst = int(scaled.argmax())
    ab, rb = bf16_bits(actual), bf16_bits(reference)
    # Ordered integer coordinates collapse -0/+0, which are checked separately.
    def ordered(bits):
        magnitude = (bits & 0x7fff).astype(np.int32)
        return np.where(bits & 0x8000, -magnitude, magnitude)
    steps = np.abs(ordered(ab)-ordered(rb))
    record = dict(elements=a.size, bit_differences=int(np.count_nonzero(ab != rb)),
                  max_abs=float(err.max()), max_scaled=float(scaled.max()),
                  max_bf16_steps=int(steps.max()),
                  zero_sign_differences=int(np.count_nonzero((a == 0) & (r == 0) & (ab != rb))),
                  worst_index=list(np.unravel_index(worst, a.shape)),
                  actual=float(a.flat[worst]), reference=float(r.flat[worst]))
    record['worst_index'] = [int(i) for i in record['worst_index']]
    if budget is not None:
        failed = err > budget['atol'] + budget['rtol']*np.abs(r)
        if 'max_bf16_steps' in budget:
            failed |= steps > budget['max_bf16_steps']
        if budget.get('exact_bits'):
            failed |= ab != rb
        if budget.get('exact_reference_zero'):
            failed |= (r == 0) & (ab != rb)
        record['failed'] = int(np.count_nonzero(failed))
    return record
