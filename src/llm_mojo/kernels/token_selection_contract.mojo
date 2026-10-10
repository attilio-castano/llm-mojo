"""Integer selection primitives and finite compile-time contract checkers.

The GPU uses these same scalar primitives. The checks do not reflect or prove
the GPU body. See docs/token-selection-contract.md for scope and assumptions.
"""


@always_inline
def bf16_rank(bits: UInt16) -> UInt32:
    # Canonicalize signed zero; integer ordering retains subnormals.
    var b = UInt32(bits)
    if (b & 0x7FFF) == 0:
        b = 0
    return ((~b) & 0xFFFF) if (b & 0x8000) != 0 else (b ^ 0x8000)


@always_inline
def bf16_nonfinite(bits: UInt16) -> UInt32:
    return UInt32((bits & 0x7F80) == 0x7F80)


@fieldwise_init
struct ArgmaxRecord(ImplicitlyCopyable):
    var rank: UInt32
    var token: UInt32
    var invalid: UInt32


@always_inline
def empty_argmax() -> ArgmaxRecord:
    return ArgmaxRecord(0, 0xFFFFFFFF, 0)


@always_inline
def merge_argmax(a: ArgmaxRecord, b: ArgmaxRecord) -> ArgmaxRecord:
    # Keep this comparison/selection-only: the order-pattern argument depends
    # on it. invalid is restricted to {0, 1} at the GPU reduction boundary.
    if b.rank > a.rank or (b.rank == a.rank and b.token < a.token):
        return ArgmaxRecord(b.rank, b.token, a.invalid | b.invalid)
    return ArgmaxRecord(a.rank, a.token, a.invalid | b.invalid)


def check_bf16_ordering[ranker: def(UInt16) thin -> UInt32]() -> Bool:
    """Exhaust finite encodings against decoded exponent/significand ordinals.

    Finite magnitude is m * 2**(max(e, 1) - 134), m=f for subnormals,
    m=128+f otherwise. Adjacent exponent bins are strictly ordered; ordinals
    therefore specify numerical order without host floating-point arithmetic.
    """
    for exponent in range(255):
        for fraction in range(128):
            var significand = fraction if exponent == 0 else 128 + fraction
            var ordinal = significand
            if exponent != 0:
                ordinal += (exponent - 1) * 128
            var positive = UInt16(exponent * 128 + fraction)
            var negative = UInt16(0x8000 + exponent * 128 + fraction)
            var positive_rank = UInt32(32768 + ordinal)
            var negative_rank = UInt32(32768 if ordinal == 0 else 32767 - ordinal)
            if ranker(positive) != positive_rank or ranker(negative) != negative_rank:
                return False
    return True


def check_bf16_classification[classifier: def(UInt16) thin -> UInt32]() -> Bool:
    """Exhaust all 65,536 encodings, including every NaN and infinity."""
    for encoding in range(65536):
        var exponent = (encoding // 128) % 256
        if classifier(UInt16(encoding)) != UInt32(exponent == 255):
            return False
    return True


def _equal(a: ArgmaxRecord, b: ArgmaxRecord) -> Bool:
    return a.rank == b.rank and a.token == b.token and a.invalid == b.invalid


def _representative(index: Int) -> ArgmaxRecord:
    var r = index // 6
    var t = (index // 2) % 3
    return ArgmaxRecord(
        UInt32(0xFFFFFFFF if r == 2 else r),
        UInt32(0xFFFFFFFF if t == 2 else t),
        UInt32(index % 2),
    )


def _reference_merge(a: ArgmaxRecord, b: ArgmaxRecord) -> ArgmaxRecord:
    # Independent two-pass specification: find score, then eligible lowest ID.
    var top = a.rank if a.rank >= b.rank else b.rank
    var token: UInt32 = 0xFFFFFFFF
    if a.rank == top:
        token = a.token
    if b.rank == top and b.token < token:
        token = b.token
    return ArgmaxRecord(top, token, UInt32(a.invalid != 0 or b.invalid != 0))


def check_argmax_laws[
    combine: def(ArgmaxRecord, ArgmaxRecord) thin -> ArgmaxRecord
]() -> Bool:
    """Exhaust 18 representative records and 5,832 triples.

    This is a finite check, not arbitrary-body verification. Lifting it to
    UInt32 requires the documented comparison-only order-pattern argument.
    """
    var empty = empty_argmax()
    for i in range(18):
        var a = _representative(i)
        if not _equal(combine(a, a), a):
            return False
        if not _equal(combine(empty, a), a) or not _equal(combine(a, empty), a):
            return False
        for j in range(18):
            var b = _representative(j)
            var ab = combine(a, b)
            if not _equal(ab, _reference_merge(a, b)):
                return False
            if not _equal(ab, combine(b, a)):
                return False
            for k in range(18):
                var c = _representative(k)
                if not _equal(combine(ab, c), combine(a, combine(b, c))):
                    return False
    return True
