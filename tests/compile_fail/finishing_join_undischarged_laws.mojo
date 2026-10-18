from finishing_join_cases import finishing_join_mutant_accepts, JOIN_UNDISCHARGED_LAWS


def main():
    comptime assert finishing_join_mutant_accepts(JOIN_UNDISCHARGED_LAWS), "checked finishing Nat join rejected"
