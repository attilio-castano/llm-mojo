"""A named alias cannot hide a negative recursive constructor field."""
from nat_arithmetic_cases import aliased_constructor_case


def main():
    comptime accepted = aliased_constructor_case(True)
    comptime assert accepted, "native Nat arithmetic rejected"
