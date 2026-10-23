"""Reject an invalid checked Base definition or equation proof."""
from base_definition_cases import base_driver_equations_accepts


def main():
    comptime accepted = base_driver_equations_accepts(17, 1)
    comptime assert accepted, "checked Base definition rejected"
