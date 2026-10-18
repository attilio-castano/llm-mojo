from finishing_join_cases import finishing_join_mutant_accepts, JOIN_BAD_PACK


def main():
    comptime assert finishing_join_mutant_accepts(JOIN_BAD_PACK), "checked finishing Nat join rejected"
