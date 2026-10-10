"""The merge checker must reject losing the right operand's invalid flag."""
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord, check_argmax_laws


def dropped_invalid_mutant(left: ArgmaxRecord, right: ArgmaxRecord) -> ArgmaxRecord:
    if right.rank > left.rank or (right.rank == left.rank and right.token < left.token):
        return ArgmaxRecord(right.rank, right.token, left.invalid)
    return ArgmaxRecord(left.rank, left.token, left.invalid)


def main():
    comptime assert check_argmax_laws[dropped_invalid_mutant](), "argmax representative laws failed"
