"""The merge checker must reject choosing the larger token ID on a tied score."""
from llm_mojo.kernels.token_selection_contract import ArgmaxRecord, check_argmax_laws


def larger_token_mutant(left: ArgmaxRecord, right: ArgmaxRecord) -> ArgmaxRecord:
    var rank = left.rank
    var token = left.token
    if right.rank > left.rank or (right.rank == left.rank and right.token > left.token):
        rank = right.rank
        token = right.token
    return ArgmaxRecord(rank, token, left.invalid | right.invalid)


def main():
    comptime assert check_argmax_laws[larger_token_mutant](), "argmax representative laws failed"
