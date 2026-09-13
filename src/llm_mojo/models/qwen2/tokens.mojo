"""Qwen end-of-text and end-of-turn token handling."""


def is_stop(token: Int) -> Bool:
    return token == 151645 or token == 151643
