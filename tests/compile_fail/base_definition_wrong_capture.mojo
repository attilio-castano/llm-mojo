"""Reject an invalid checked Base definition or equation proof."""
from nat_arithmetic_cases import nat_arithmetic_mutant_accepts


def main():
    comptime accepted = nat_arithmetic_mutant_accepts(26)
    comptime assert accepted, "checked Base definition rejected"
