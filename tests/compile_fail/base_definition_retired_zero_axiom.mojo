"""Reject an invalid checked Base definition or equation proof."""
from nat_arithmetic_cases import retired_arithmetic_axiom_accepts


def main():
    comptime accepted = retired_arithmetic_axiom_accepts(0)
    comptime assert accepted, "checked Base definition rejected"
