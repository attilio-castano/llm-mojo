"""An ordinary host call must enforce the compile-time admission body."""
from llm_mojo.kernels.finishing_proof import FinishingProgram, make_finishing_program
from llm_mojo.kernels.finishing_checked_exports import require_finishing_traversal


def wrong_program() -> FinishingProgram:
    var program = make_finishing_program(64, 3, 192)
    program.item_scale += 1
    return program


def main():
    comptime program = wrong_program()
    require_finishing_traversal[program]()
    print("must not compile")
