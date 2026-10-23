from compact_numeral_cases import compact_numeral_mutant_accepts, NUM_VALUE_BINDING


def main():
    comptime accepted = compact_numeral_mutant_accepts(NUM_VALUE_BINDING)
    comptime assert accepted, "checked compact numeral bridge rejected"
