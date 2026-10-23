from finishing_join_cases import finishing_join_mutant_accepts, JOIN_UNUSED_MALFORMED


def main():
    comptime assert finishing_join_mutant_accepts(JOIN_UNUSED_MALFORMED), "checked finishing Nat join rejected"
