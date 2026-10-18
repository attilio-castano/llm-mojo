from headroom_cases import headroom_mutant_accepts, HEAD_WRONG_CARRY_LEMMA


def main():
    comptime assert headroom_mutant_accepts(HEAD_WRONG_CARRY_LEMMA), "checked Nat headroom rejected"
