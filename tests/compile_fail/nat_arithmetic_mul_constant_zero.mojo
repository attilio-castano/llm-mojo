"""Reject malformed or weakened Nat arithmetic certification."""
from nat_arithmetic_cases import nat_arithmetic_mutant_accepts


def main():
    comptime accepted = nat_arithmetic_mutant_accepts(1)
    comptime assert accepted, "native Nat arithmetic rejected"
