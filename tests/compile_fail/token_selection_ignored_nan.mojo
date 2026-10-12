"""The classifier checker must reject ignoring even one NaN encoding."""
from llm_mojo.kernels.token_selection_contract import bf16_nonfinite, check_bf16_classification


def ignored_nan_mutant(bits: UInt16) -> UInt32:
    if bits == UInt16(0x7F81):
        return UInt32(0)
    return bf16_nonfinite(bits)


def main():
    comptime assert check_bf16_classification[ignored_nan_mutant](), "BF16 classification contract failed"
