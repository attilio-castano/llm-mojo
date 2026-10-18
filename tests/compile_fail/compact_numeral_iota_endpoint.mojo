from compact_numeral_cases import compact_numeral_mutant_accepts, NUM_IOTA_ENDPOINT


def main():
    comptime accepted = compact_numeral_mutant_accepts(NUM_IOTA_ENDPOINT)
    comptime assert accepted, "checked compact numeral bridge rejected"
