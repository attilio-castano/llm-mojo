from compact_numeral_cases import compact_numeral_mutant_accepts, NUM_DOUBLE_IDENTITY


def main():
    comptime accepted = compact_numeral_mutant_accepts(NUM_DOUBLE_IDENTITY)
    comptime assert accepted, "checked compact numeral bridge rejected"
