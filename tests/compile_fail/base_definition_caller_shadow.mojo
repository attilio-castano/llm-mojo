"""Reject an invalid checked Base definition or equation proof."""
from nat_arithmetic_cases import local_shadow_case


def main():
    comptime accepted = local_shadow_case(4)
    comptime assert accepted, "checked Base definition rejected"
