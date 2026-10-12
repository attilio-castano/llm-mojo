"""The ordering checker must reject a ranker that fails signed-zero canonicalization."""
from llm_mojo.kernels.token_selection_contract import bf16_rank, check_bf16_ordering


def signed_zero_mutant(bits: UInt16) -> UInt32:
    if bits == UInt16(0x8000):
        return UInt32(0)
    return bf16_rank(bits)


def main():
    comptime assert check_bf16_ordering[signed_zero_mutant](), "BF16 ordering contract failed"
