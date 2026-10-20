"""Reject a malformed immutable export instantiation or contract."""
from export_instantiation_cases import export_open_root_accepts


def main():
    comptime accepted = export_open_root_accepts()
    comptime assert accepted, "checked export instantiation rejected"
