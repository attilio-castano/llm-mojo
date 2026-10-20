from headroom_cases import headroom_mutant_accepts, HEAD_REQUEST_CAP


def main():
    comptime assert headroom_mutant_accepts(HEAD_REQUEST_CAP), "checked Nat headroom rejected"
