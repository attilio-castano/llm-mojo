from literal_arithmetic_cases import literal_arithmetic_mutant_accepts, LIT_UNUSED_MALFORMED


def main():
    comptime assert literal_arithmetic_mutant_accepts(LIT_UNUSED_MALFORMED), "checked literal arithmetic conversion rejected"
