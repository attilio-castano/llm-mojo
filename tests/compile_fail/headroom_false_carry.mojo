from headroom_cases import headroom_mutant_accepts, HEAD_FALSE_CARRY


def main():
    comptime assert headroom_mutant_accepts(HEAD_FALSE_CARRY), "checked Nat headroom rejected"
