from compact_numeral_cases import compact_numeral_mutant_accepts, NUM_CONGRUENCE_ARGUMENT


def main():
    comptime accepted = compact_numeral_mutant_accepts(NUM_CONGRUENCE_ARGUMENT)
    comptime assert accepted, "checked compact numeral bridge rejected"
