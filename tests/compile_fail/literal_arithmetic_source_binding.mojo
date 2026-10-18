from literal_arithmetic_cases import literal_arithmetic_mutant_accepts, LIT_SOURCE_ENDPOINT


def main():
    comptime assert literal_arithmetic_mutant_accepts(LIT_SOURCE_ENDPOINT), "checked literal arithmetic conversion rejected"
