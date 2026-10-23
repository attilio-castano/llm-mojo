"""Reject a malformed immutable export instantiation or contract."""
from export_instantiation_cases import caller_export_spoof_accepts


def main():
    comptime accepted = caller_export_spoof_accepts()
    comptime assert accepted, "checked export instantiation rejected"
