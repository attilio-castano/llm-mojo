from literal_arithmetic_cases import literal_arithmetic_mutant_accepts, LIT_FALSE_SUM


def main():
    comptime assert literal_arithmetic_mutant_accepts(LIT_FALSE_SUM), "checked literal arithmetic conversion rejected"
