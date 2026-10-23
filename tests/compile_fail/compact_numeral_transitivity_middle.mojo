from compact_numeral_cases import compact_numeral_mutant_accepts, NUM_TRANS_MIDDLE


def main():
    comptime accepted = compact_numeral_mutant_accepts(NUM_TRANS_MIDDLE)
    comptime assert accepted, "checked compact numeral bridge rejected"
