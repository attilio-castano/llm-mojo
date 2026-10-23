from compact_numeral_cases import compact_numeral_mutant_accepts, NUM_UNKNOWN_RULE


def main():
    comptime accepted = compact_numeral_mutant_accepts(NUM_UNKNOWN_RULE)
    comptime assert accepted, "checked compact numeral bridge rejected"
