"""Reject caller interference with the immutable division definition."""
from nat_arithmetic_cases import local_shadow_case


def main():
    comptime accepted = local_shadow_case(3)
    comptime assert accepted, "native Nat arithmetic rejected"
