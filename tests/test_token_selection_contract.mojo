"""Force exhaustive encoding and finite argmax checks during compilation."""
from std.testing import assert_equal
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, bf16_rank, bf16_nonfinite, merge_argmax,
    check_bf16_ordering, check_bf16_classification, check_argmax_laws,
)


def main() raises:
    comptime assert check_bf16_ordering[bf16_rank](), "BF16 ordering contract failed"
    comptime assert check_bf16_classification[bf16_nonfinite](), "BF16 classification contract failed"
    comptime assert check_argmax_laws[merge_argmax](), "argmax representative laws failed"
    # Repeat the finite checks in compiled host code, so the test exercises
    # both compile-time interpretation and native scalar execution.
    assert_equal(check_bf16_ordering[bf16_rank](), True)
    assert_equal(check_bf16_classification[bf16_nonfinite](), True)
    assert_equal(check_argmax_laws[merge_argmax](), True)
    # Runtime execution uses the same shared primitives, independently of the
    # compile-time interpreter. Interior values supplement the order classes.
    var a = ArgmaxRecord(bf16_rank(UInt16(0x0001)), 1024, 0)
    var b = ArgmaxRecord(bf16_rank(UInt16(0x0001)), 31, 0)
    var winner = merge_argmax(a, b)
    assert_equal(winner.token, UInt32(31))
    assert_equal(winner.invalid, UInt32(0))
    var poisoned = merge_argmax(winner, ArgmaxRecord(0, 1, 1))
    assert_equal(poisoned.token, winner.token)
    assert_equal(poisoned.invalid, UInt32(1))
    assert_equal(bf16_rank(UInt16(0)), bf16_rank(UInt16(0x8000)))
    print("Compile-time BF16 ordering/classification and argmax representative checks passed")
