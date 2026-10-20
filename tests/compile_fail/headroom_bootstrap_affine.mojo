from headroom_cases import headroom_mutant_accepts, HEAD_BOOTSTRAP_AFFINE


def main():
    comptime assert headroom_mutant_accepts(HEAD_BOOTSTRAP_AFFINE), "checked Nat headroom rejected"
