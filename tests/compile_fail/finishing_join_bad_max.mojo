from finishing_join_cases import finishing_join_mutant_accepts, JOIN_BAD_MAX


def main():
    comptime assert finishing_join_mutant_accepts(JOIN_BAD_MAX), "checked finishing Nat join rejected"
