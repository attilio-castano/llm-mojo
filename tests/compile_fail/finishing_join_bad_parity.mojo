from finishing_join_cases import finishing_join_mutant_accepts, JOIN_BAD_PARITY


def main():
    comptime assert finishing_join_mutant_accepts(JOIN_BAD_PARITY), "checked finishing Nat join rejected"
