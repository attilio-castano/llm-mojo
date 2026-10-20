from literal_arithmetic_cases import literal_arithmetic_mutant_accepts, LIT_FALSE_QUOTIENT


def main():
    comptime assert literal_arithmetic_mutant_accepts(LIT_FALSE_QUOTIENT), "checked literal arithmetic conversion rejected"
