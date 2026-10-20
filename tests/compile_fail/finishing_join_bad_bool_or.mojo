from finishing_join_cases import finishing_join_mutant_accepts, JOIN_BAD_BOOL_OR


def main():
    comptime assert finishing_join_mutant_accepts(JOIN_BAD_BOOL_OR), "checked finishing Nat join rejected"
