"""Reject a malformed immutable export instantiation or contract."""
from export_instantiation_cases import export_contracts_accept


def main():
    comptime accepted = export_contracts_accept(17, 17, 3)
    comptime assert accepted, "checked export instantiation rejected"
