from headroom_cases import headroom_mutant_accepts, HEAD_UNUSED_MALFORMED


def main():
    comptime assert headroom_mutant_accepts(HEAD_UNUSED_MALFORMED), "checked Nat headroom rejected"
