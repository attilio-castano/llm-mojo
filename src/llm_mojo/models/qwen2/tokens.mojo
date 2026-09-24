"""Qwen end-of-text and end-of-turn token handling."""

comptime END_OF_TEXT = 151643
comptime IM_END = 151645


def is_stop(token: Int) -> Bool:
    return token == IM_END or token == END_OF_TEXT
