"""Monotonic host timestamp shared by native applications."""
from std.ffi import external_call


def now() -> UInt64:
    return external_call["clock_gettime_nsec_np", UInt64](UInt32(8))
