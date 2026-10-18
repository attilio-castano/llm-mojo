from literal_arithmetic_cases import literal_arithmetic_mutant_accepts, LIT_WRONG_ADD_DEFINITION


def main():
    comptime assert literal_arithmetic_mutant_accepts(LIT_WRONG_ADD_DEFINITION), "checked literal arithmetic conversion rejected"
