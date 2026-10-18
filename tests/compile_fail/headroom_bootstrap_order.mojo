from headroom_cases import headroom_mutant_accepts, HEAD_BOOTSTRAP_ORDER


def main():
    comptime assert headroom_mutant_accepts(HEAD_BOOTSTRAP_ORDER), "checked Nat headroom rejected"
