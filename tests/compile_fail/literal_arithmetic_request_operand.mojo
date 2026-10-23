from literal_arithmetic_cases import literal_arithmetic_mutant_accepts, LIT_REQUEST_RIGHT


def main():
    comptime assert literal_arithmetic_mutant_accepts(LIT_REQUEST_RIGHT), "checked literal arithmetic conversion rejected"
