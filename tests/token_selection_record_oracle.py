"""Finite native record reference; Python integers supply independent scores.

The companion Mojo probe calls the maintained UInt32 merge and emits record
words. This oracle never mirrors its rank/branch/tie predicate. Finite agreement
is not an arbitrary-input proof, source correspondence, or GPU qualification.
"""
from __future__ import annotations

from collections import Counter
import itertools
from pathlib import Path
import random

RADIX = 1 << 32
MAXIMUM = RADIX - 1
EDGES = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF)
SEED = 0x4C4C4D
SEEDED_PAIRS = 8192
SEEDED_TRIPLES = 8192
MARKER = "Native UInt32 record diagnostic completed"
EXPECTED_CASES = 25145
EXPECTED_OUTPUT_RECORDS = 36937
EXPECTED_CATEGORY_COUNTS = {
    "empty": 1, "edge_empty_idempotence": 72, "edge_pairs": 5184,
    "edge_rank_triples": 1728, "edge_token_triples": 1728,
    "adjacent_rank_carry": 48, "seeded_pairs": 8192, "seeded_triples": 8192,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cases():
    rows = []

    def add(kind, records, category):
        rows.append({"id": len(rows), "kind": kind, "records": records, "category": category})

    add("E", [], "empty")
    edge_records = list(itertools.product(EDGES, EDGES, (0, 1)))
    for a in edge_records:
        add("U", [a], "edge_empty_idempotence")
    for a, b in itertools.product(edge_records, repeat=2):
        add("P", [a, b], "edge_pairs")
    # Isolate rank order across all edge triples, with every bad-flag pattern.
    for ranks in itertools.product(EDGES, repeat=3):
        for flags in itertools.product((0, 1), repeat=3):
            add("T", [(ranks[i], MAXIMUM, flags[i]) for i in range(3)], "edge_rank_triples")
    # Isolate reversed token order at a rank on the signed-word boundary.
    for tokens in itertools.product(EDGES, repeat=3):
        for flags in itertools.product((0, 1), repeat=3):
            add("T", [(0x80000000, tokens[i], flags[i]) for i in range(3)], "edge_token_triples")
    # Adjacent ranks must beat even the greatest token-complement gap by one.
    for rank in (0, 1, 0x7FFFFFFE, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE):
        for ba, bb in itertools.product((0, 1), repeat=2):
            a, b = (rank, 0, ba), (rank + 1, MAXIMUM, bb)
            add("P", [a, b], "adjacent_rank_carry")
            add("P", [b, a], "adjacent_rank_carry")
    rng = random.Random(SEED)

    def record():
        def word():
            return rng.choice(EDGES) if rng.randrange(2) else rng.getrandbits(32)
        return (word(), word(), rng.randrange(2))

    for _ in range(SEEDED_PAIRS):
        a, b = record(), record()
        add("P", [a, b], "seeded_pairs")
    for _ in range(SEEDED_TRIPLES):
        a, b, c = record(), record(), record()
        add("T", [a, b, c], "seeded_triples")
    return rows


def flat_oracle(records):
    # The sole winner mechanism is Python's arbitrary-precision integer max.
    # We do not mirror merge_argmax's rank/branch/tie predicate.
    if not records:
        return (0, MAXIMUM, 0)
    maximum_score = max(rank * RADIX + (MAXIMUM - token) for rank, token, _ in records)
    rank, complement = divmod(maximum_score, RADIX)
    token = MAXIMUM - complement
    bad = int(any(flag for _, _, flag in records))
    return (rank, token, bad)


def check_output(text, rows):
    lines = text.splitlines()
    require(len(lines) == len(rows) + 1, "Native output row count differs from the input")
    require(lines[-1] == f"{MARKER} {len(rows)}", "Missing exact native completion count")
    checked_records = 0
    for case, line in zip(rows, lines[:-1], strict=True):
        fields = line.split()
        require(fields[:2] == [case["kind"], str(case["id"])], "Native case order/identity changed")
        count = {"E": 1, "U": 3, "P": 1, "T": 2}[case["kind"]]
        require(len(fields) == 2 + count * 3, "Native output width differs")
        expected = flat_oracle(case["records"])
        for offset in range(count):
            actual = tuple(map(int, fields[2 + 3 * offset:5 + 3 * offset]))
            require(actual == expected, f"Oracle mismatch at {case['category']} case {case['id']}: {actual} != {expected}")
            checked_records += 1
    return checked_records


def write_cases(path, rows):
    """Write the strict TSV protocol consumed by token_selection_record_probe."""
    Path(path).write_text("".join("\t".join(map(str, [case["kind"], case["id"],
        *(word for record in case["records"] for word in record)])) + "\n" for case in rows))


def check_case_inventory(rows):
    """Check the fixed reproducible finite domain, including its 65-bit edge."""
    require(len(rows) == EXPECTED_CASES, "Unexpected finite case count")
    require(dict(Counter(row["category"] for row in rows)) == EXPECTED_CATEGORY_COUNTS,
            "Unexpected finite category counts")
    for index, case in enumerate(rows):
        require(case["id"] == index, "Nonconsecutive finite case identity")
        for rank, token, bad in case["records"]:
            require(0 <= rank <= MAXIMUM and 0 <= token <= MAXIMUM and bad in (0, 1),
                    "Case lies outside UInt32/Boolean domain")
    maximum_bits = max((2 * (rank * RADIX + MAXIMUM - token) + bad).bit_length()
        for case in rows for rank, token, bad in case["records"])
    require(maximum_bits == 65, "Missing actual 65-bit Python encoding edge")
    return maximum_bits
