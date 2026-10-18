from headroom_cases import headroom_mutant_accepts, HEAD_FALSE_TIGHT


def main():
    comptime assert headroom_mutant_accepts(HEAD_FALSE_TIGHT), "checked Nat headroom rejected"
