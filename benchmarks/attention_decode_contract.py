"""Shared identity checks for the bounded attention profile protocol."""

VARIANTS = {
    0: (0, 0, 0),
    1: (1, 1, 1),
    2: (2, 1, 1),
    3: (8, 1, 1),
    4: (32, 1, 1),
    5: (1, 1, 4),
    6: (1, 1, 16),
    7: (1, 1, 64),
    8: (1, 2, 64),
    9: (1, 4, 64),
    10: (1, 7, 64),
}
OPERATION = "grouped_query_attention_decode"
ENTRYPOINTS = {
    "gqa_decode_"
    + str(v): (
        "enqueue_grouped_query_attention_apple_gpu" if v
        == 0 else "enqueue_grouped_query_attention_decode_apple_gpu"
    )
    for v in VARIANTS
}
TARGET_FIELDS = (
    "profile_workload",
    "dispatches_per_iteration",
    "key_value_rows",
    "query_heads",
    "key_value_heads",
    "groups",
    "heads",
    "splits",
)


def configuration(data):
    implementation = data.get("implementation", "")
    if (
        implementation not in ENTRYPOINTS
        or data.get("entrypoint") != ENTRYPOINTS[implementation]
    ):
        raise ValueError("invalid attention implementation identity")
    variant = int(implementation.removeprefix("gqa_decode_"))
    groups, heads, splits = VARIANTS[variant]
    rows = data.get("key_value_rows")
    if type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError("invalid attention context length")
    expected = {
        "profile_rows": 1,
        "hidden_size": 64,
        "query_heads": 14,
        "key_value_heads": 2,
        "groups": groups,
        "heads": heads,
        "splits": splits,
        "key_value_rows": rows,
        "profile_workload": f'decode-t{rows}-v{variant}',
        "dispatches_per_iteration": 3 if variant
        == 0 else (2 if splits > 1 else 1),
    }
    if any(
        data.get(k) != v or (type(v) is int and type(data.get(k)) is not int)
        for k, v in expected.items()
    ):
        raise ValueError(
            "attention shape, parameters or dispatch count disagrees with implementation"
        )
    iterations = data.get("profile_iterations")
    warmup = data.get("profile_warmup_iterations")
    if (
        type(iterations) is not int
        or iterations <= 0
        or iterations * expected["dispatches_per_iteration"] > 5000
    ):
        raise ValueError("attention profile exceeds dispatch budget")
    if type(warmup) is not int or not 0 <= warmup <= 100:
        raise ValueError("attention profile warmup is outside bounds")
    return expected
