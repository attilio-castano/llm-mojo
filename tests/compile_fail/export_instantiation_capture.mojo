"""Reject a malformed immutable export instantiation or contract."""
from export_instantiation_cases import export_capture_accepts


def main():
    comptime accepted = export_capture_accepts(True)
    comptime assert accepted, "checked export instantiation rejected"
