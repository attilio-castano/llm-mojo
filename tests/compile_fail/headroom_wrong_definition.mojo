from headroom_cases import headroom_mutant_accepts, HEAD_WRONG_ADD_DEFINITION


def main():
    comptime assert headroom_mutant_accepts(HEAD_WRONG_ADD_DEFINITION), "checked Nat headroom rejected"
