from finishing_join_cases import finishing_join_mutant_accepts, JOIN_BAD_JOIN_FLAG


def main():
    comptime assert finishing_join_mutant_accepts(JOIN_BAD_JOIN_FLAG), "checked finishing Nat join rejected"
