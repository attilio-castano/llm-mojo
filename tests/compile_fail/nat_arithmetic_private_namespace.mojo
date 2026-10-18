"""Reject caller interference with the immutable arithmetic library."""
from nat_arithmetic_cases import local_shadow_case


def main():
    comptime accepted = local_shadow_case(2)
    comptime assert accepted, "native Nat arithmetic rejected"
