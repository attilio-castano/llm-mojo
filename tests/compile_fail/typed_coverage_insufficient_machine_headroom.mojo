"""Insufficient machine headroom must fail at compile-time admission."""
from llm_mojo.kernels.finishing_proof import make_finishing_program
from test_typed_coverage import INSUFFICIENT_MACHINE_HEADROOM, typed_coverage_mutant_accepts


def main():
    comptime program = make_finishing_program(3, 5, 15)
    comptime assert typed_coverage_mutant_accepts(program, INSUFFICIENT_MACHINE_HEADROOM), "typed coverage rejected"
    print(INSUFFICIENT_MACHINE_HEADROOM)
