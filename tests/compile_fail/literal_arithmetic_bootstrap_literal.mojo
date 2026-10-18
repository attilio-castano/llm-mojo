from literal_arithmetic_cases import literal_arithmetic_mutant_accepts, LIT_BOOTSTRAP_LITERAL


def main():
    comptime assert literal_arithmetic_mutant_accepts(LIT_BOOTSTRAP_LITERAL), "checked literal arithmetic conversion rejected"
