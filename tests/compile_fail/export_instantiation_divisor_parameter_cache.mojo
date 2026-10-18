"""Reject a malformed immutable export instantiation or contract."""
from export_instantiation_cases import mixed_export_parameters_accept


def main():
    comptime accepted = mixed_export_parameters_accept(False, 1)
    comptime assert accepted, "checked export instantiation rejected"
