"""Reject a false raw operator representation recipe or its metadata."""
from operator_representation_cases import operator_representation_library_accepts


def main():
    comptime accepted = operator_representation_library_accepts(10)
    comptime assert accepted, "checked operator representation rejected"
