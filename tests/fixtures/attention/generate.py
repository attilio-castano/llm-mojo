#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Generate deterministic Qwen2 grouped-query attention oracle fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.models.qwen2.modeling_qwen2 import repeat_kv


FIXTURE_DIR = Path(__file__).resolve().parents[3] / "build/oracle_data" / Path(__file__).parent.name
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = FIXTURE_DIR / "reference_data.mojo"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ATOL = 0.015625
RTOL = 0.015625


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16(values: list[float], shape: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(shape).to(torch.bfloat16)


def deterministic_values(count: int, seed: int, scale: int) -> list[float]:
    return [
        ((((index * 37 + seed * 19) % 257) - 128) / scale)
        for index in range(count)
    ]


def run_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_rows, query_heads, head_dim = query.shape
    key_value_rows, key_value_heads, key_head_dim = key.shape
    if value.shape != key.shape:
        raise ValueError("key and value shapes must match")
    if key_head_dim != head_dim:
        raise ValueError("query and key head dimensions must match")
    if query_rows > key_value_rows:
        raise ValueError("query rows must be an active K/V suffix")
    if query_heads % key_value_heads != 0:
        raise ValueError("query heads must divide evenly across K/V heads")

    query_states = query.permute(1, 0, 2).unsqueeze(0).contiguous()
    key_states = key.permute(1, 0, 2).unsqueeze(0).contiguous()
    value_states = value.permute(1, 0, 2).unsqueeze(0).contiguous()
    groups = query_heads // key_value_heads
    key_states = repeat_kv(key_states, groups)
    value_states = repeat_kv(value_states, groups)

    with torch.no_grad():
        scores = torch.matmul(query_states, key_states.transpose(2, 3))
        scores = scores / math.sqrt(head_dim)

        past = key_value_rows - query_rows
        query_positions = torch.arange(past, key_value_rows).reshape(query_rows, 1)
        key_positions = torch.arange(key_value_rows).reshape(1, key_value_rows)
        future = key_positions > query_positions
        scores = scores.masked_fill(
            future.reshape(1, 1, query_rows, key_value_rows),
            torch.finfo(scores.dtype).min,
        )

        probabilities = torch.nn.functional.softmax(
            scores, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        output = torch.matmul(probabilities, value_states)

    output_rows = output.squeeze(0).transpose(0, 1).contiguous()
    probability_rows = probabilities.squeeze(0).transpose(0, 1).contiguous()
    return output_rows, probability_rows


def as_float32_list(tensor: torch.Tensor) -> list[float]:
    if not torch.isfinite(tensor).all():
        raise ValueError("attention fixtures require finite values")
    return tensor.to(torch.float32).reshape(-1).tolist()


def render_float(value: float) -> str:
    return format(value, ".9g")


def render_list(name: str, values: list[float]) -> str:
    lines = [f"def {name}() raises -> List[Float32]:", "    return ["]
    for offset in range(0, len(values), 8):
        chunk = ", ".join(
            render_float(value) for value in values[offset : offset + 8]
        )
        lines.append(f"        {chunk},")
    lines.extend(["    ]", ""])
    return "\n".join(lines)


def build_cases() -> list[dict[str, Any]]:
    specs = [
        {
            "name": "tiny_full_prefill",
            "query_rows": 3,
            "key_value_rows": 3,
            "query_heads": 4,
            "key_value_heads": 2,
            "head_dim": 4,
            "query": deterministic_values(3 * 4 * 4, 1, 32),
            "key": deterministic_values(3 * 2 * 4, 2, 32),
            "value": deterministic_values(3 * 2 * 4, 3, 16),
        },
        {
            "name": "tiny_incremental_prefill",
            "query_rows": 2,
            "key_value_rows": 5,
            "query_heads": 4,
            "key_value_heads": 2,
            "head_dim": 4,
            "query": deterministic_values(2 * 4 * 4, 4, 32),
            "key": deterministic_values(5 * 2 * 4, 5, 32),
            "value": deterministic_values(5 * 2 * 4, 6, 16),
        },
        {
            "name": "stable_softmax_decode",
            "query_rows": 1,
            "key_value_rows": 3,
            "query_heads": 2,
            "key_value_heads": 1,
            "head_dim": 4,
            "query": [
                16.0,
                16.0,
                16.0,
                16.0,
                -16.0,
                16.0,
                -16.0,
                16.0,
            ],
            "key": [
                16.0,
                16.0,
                16.0,
                16.0,
                15.875,
                15.875,
                15.875,
                15.875,
                -16.0,
                -16.0,
                -16.0,
                -16.0,
            ],
            "value": [
                1.0,
                2.0,
                3.0,
                4.0,
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                0.5,
                -0.5,
                1.5,
                -1.5,
            ],
        },
        {
            "name": "qwen_decode",
            "query_rows": 1,
            "key_value_rows": 7,
            "query_heads": 14,
            "key_value_heads": 2,
            "head_dim": 64,
            "query": deterministic_values(14 * 64, 7, 128),
            "key": deterministic_values(7 * 2 * 64, 8, 128),
            "value": deterministic_values(7 * 2 * 64, 9, 64),
        },
    ]

    cases: list[dict[str, Any]] = []
    for spec in specs:
        query_shape = (
            spec["query_rows"],
            spec["query_heads"],
            spec["head_dim"],
        )
        key_value_shape = (
            spec["key_value_rows"],
            spec["key_value_heads"],
            spec["head_dim"],
        )
        query = bf16(spec["query"], query_shape)
        key = bf16(spec["key"], key_value_shape)
        value = bf16(spec["value"], key_value_shape)
        expected, probabilities = run_oracle(query, key, value)
        cases.append(
            {
                **spec,
                "query": as_float32_list(query),
                "key": as_float32_list(key),
                "value": as_float32_list(value),
                "expected": as_float32_list(expected),
                "probabilities": as_float32_list(probabilities),
            }
        )
    return cases


def render_data(cases: list[dict[str, Any]]) -> str:
    sections = [
        '"""Generated Qwen2 grouped-query attention oracle data. Do not edit by hand."""',
        "",
        f"comptime ATTENTION_ATOL: Float32 = {ATOL}",
        f"comptime ATTENTION_RTOL: Float32 = {RTOL}",
        "",
    ]
    for case in cases:
        prefix = case["name"].upper()
        sections.extend(
            [
                f"comptime {prefix}_QUERY_ROWS = {case['query_rows']}",
                f"comptime {prefix}_KEY_VALUE_ROWS = {case['key_value_rows']}",
                f"comptime {prefix}_QUERY_HEADS = {case['query_heads']}",
                f"comptime {prefix}_KEY_VALUE_HEADS = {case['key_value_heads']}",
                f"comptime {prefix}_HEAD_DIM = {case['head_dim']}",
                "",
                render_list(f"{case['name']}_query", case["query"]),
                render_list(f"{case['name']}_key", case["key"]),
                render_list(f"{case['name']}_value", case["value"]),
                render_list(f"{case['name']}_expected", case["expected"]),
                render_list(
                    f"{case['name']}_probabilities", case["probabilities"]
                ),
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def build_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "Numerical parity for the Mojo Qwen2 BF16 grouped-query attention reference path.",
        "target_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifact_used": False,
            "note": "Synthetic tensors cover causal full prefill, incremental prefill, stable softmax, and Qwen-shaped decode; no model weights were downloaded or extracted.",
        },
        "source_tensors": {
            "query": "deterministic synthetic BF16 values defined in generate.py",
            "key": "deterministic synthetic BF16 values defined in generate.py",
            "value": "deterministic synthetic BF16 values defined in generate.py",
        },
        "oracle": {
            "implementation": "transformers.models.qwen2.modeling_qwen2 repeat_kv plus the Qwen2Attention eager core operations",
            "source": "https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "contract": {
            "input_dtype": "bfloat16",
            "score_dtype": "bfloat16",
            "softmax_dtype": "float32",
            "probability_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "mojo_qk_accumulation_dtype": "float32",
            "mojo_pv_accumulation_dtype": "float32",
            "scale": "1 / sqrt(head_dim)",
            "causal_visibility": "query row r sees key positions 0 through key_value_rows - query_rows + r inclusive",
            "group_mapping": "key_value_head = query_head // (query_heads // key_value_heads)",
            "query_layout": "(query_rows, query_heads, head_dim):(query_heads*head_dim, head_dim, 1)",
            "key_value_layout": "(key_value_rows, key_value_heads, head_dim):(key_value_heads*head_dim, head_dim, 1)",
            "scratch_layout": "(query_rows, query_heads, key_value_rows):(query_heads*key_value_rows, key_value_rows, 1)",
            "output_layout": "(query_rows, query_heads, head_dim):(query_heads*head_dim, head_dim, 1)",
        },
        "comparison": {
            "formula": "abs(actual - expected) <= atol + rtol * abs(expected)",
            "atol": ATOL,
            "rtol": RTOL,
            "declared_before_first_mojo_comparison": True,
            "rationale": "Two BF16 machine epsilons in absolute and relative terms allow backend QK and PV reduction-order variation across two BF16 materialization boundaries without masking head-mapping, causal-mask, or softmax errors.",
        },
        "cases": [
            {
                "name": case["name"],
                "query_shape": [
                    case["query_rows"],
                    case["query_heads"],
                    case["head_dim"],
                ],
                "key_shape": [
                    case["key_value_rows"],
                    case["key_value_heads"],
                    case["head_dim"],
                ],
                "value_shape": [
                    case["key_value_rows"],
                    case["key_value_heads"],
                    case["head_dim"],
                ],
                "probability_shape": [
                    case["query_rows"],
                    case["query_heads"],
                    case["key_value_rows"],
                ],
                "output_shape": [
                    case["query_rows"],
                    case["query_heads"],
                    case["head_dim"],
                ],
            }
            for case in cases
        ],
        "generation": {
            "command": "uv run --script tests/fixtures/attention/generate.py",
            "generator_sha256": sha256(Path(__file__)),
            "data_sha256": sha256(DATA_PATH),
        },
    }


def main() -> None:
    cases = build_cases()
    DATA_PATH.write_text(render_data(cases), encoding="utf-8")
    manifest = build_manifest(cases)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
