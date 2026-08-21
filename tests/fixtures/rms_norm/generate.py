#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Generate deterministic Qwen2 RMSNorm oracle fixtures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


FIXTURE_DIR = Path(__file__).resolve().parent
DATA_PATH = FIXTURE_DIR / "reference_data.mojo"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EPSILON = 1.0e-6
ATOL = 0.0078125
RTOL = 0.0078125


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16(values: list[float], shape: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(shape).to(torch.bfloat16)


def run_oracle(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    hidden_size = hidden_states.shape[-1]
    module = Qwen2RMSNorm(hidden_size, eps=EPSILON).to(dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(weight)
        return module(hidden_states).contiguous()


def as_float32_list(tensor: torch.Tensor) -> list[float]:
    return tensor.to(torch.float32).reshape(-1).tolist()


def render_float(value: float) -> str:
    return format(value, ".9g")


def render_list(name: str, values: list[float]) -> str:
    lines = [f"def {name}() raises -> List[Float32]:", "    return ["]
    for offset in range(0, len(values), 8):
        chunk = ", ".join(render_float(value) for value in values[offset : offset + 8])
        lines.append(f"        {chunk},")
    lines.extend(["    ]", ""])
    return "\n".join(lines)


def build_cases() -> list[dict[str, Any]]:
    small_rows = 2
    small_hidden = 8
    small_input_values = [
        0.0,
        0.25,
        -0.5,
        1.0,
        -1.5,
        2.0,
        -3.0,
        4.0,
        -0.125,
        0.375,
        -0.75,
        1.25,
        -1.75,
        2.5,
        -3.5,
        4.5,
    ]
    small_weight_values = [0.75 + hidden / 16 for hidden in range(small_hidden)]

    full_rows = 1
    full_hidden = 896
    full_input_values = [
        (((index * 37) % 257) - 128) / 64 for index in range(full_hidden)
    ]
    full_weight_values = [
        0.875 + ((((index * 13) % 31) - 15) / 128)
        for index in range(full_hidden)
    ]

    case_specs = [
        ("small", small_rows, small_hidden, small_input_values, small_weight_values),
        (
            "qwen_hidden",
            full_rows,
            full_hidden,
            full_input_values,
            full_weight_values,
        ),
    ]
    cases: list[dict[str, Any]] = []
    for name, rows, hidden_size, input_values, weight_values in case_specs:
        hidden_states = bf16(input_values, (rows, hidden_size))
        weight = bf16(weight_values, (hidden_size,))
        expected = run_oracle(hidden_states, weight)
        cases.append(
            {
                "name": name,
                "rows": rows,
                "hidden_size": hidden_size,
                "input": as_float32_list(hidden_states),
                "weight": as_float32_list(weight),
                "expected": as_float32_list(expected),
            }
        )
    return cases


def render_data(cases: list[dict[str, Any]]) -> str:
    sections = [
        '"""Generated Qwen2 RMSNorm oracle data. Do not edit by hand."""',
        "",
        f"comptime RMS_NORM_ATOL: Float32 = {ATOL}",
        f"comptime RMS_NORM_RTOL: Float32 = {RTOL}",
        "",
    ]
    for case in cases:
        prefix = case["name"].upper()
        sections.extend(
            [
                f"comptime {prefix}_ROWS = {case['rows']}",
                f"comptime {prefix}_HIDDEN_SIZE = {case['hidden_size']}",
                "",
                render_list(f"{case['name']}_input", case["input"]),
                render_list(f"{case['name']}_weight", case["weight"]),
                render_list(f"{case['name']}_expected", case["expected"]),
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def build_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "Numerical parity for the Mojo Qwen2 BF16 RMSNorm reference path.",
        "target_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifact_used": False,
            "note": "Synthetic tensors exercise the pinned model's operation and hidden width; no model weights were downloaded or extracted.",
        },
        "source_tensors": {
            "hidden_states": "deterministic synthetic values defined in generate.py",
            "weight": "deterministic synthetic values defined in generate.py",
        },
        "oracle": {
            "implementation": "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
            "source": "https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "contract": {
            "input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "accumulation_dtype": "float32",
            "output_dtype": "bfloat16",
            "epsilon": EPSILON,
            "cast_order": "FP32 normalization, cast normalized values to BF16, then multiply by BF16 weight",
            "input_layout": "(rows, hidden_size):(hidden_size, 1)",
            "weight_layout": "(hidden_size):(1)",
            "output_layout": "(rows, hidden_size):(hidden_size, 1)",
        },
        "comparison": {
            "formula": "abs(actual - expected) <= atol + rtol * abs(expected)",
            "atol": ATOL,
            "rtol": RTOL,
            "declared_before_first_mojo_comparison": True,
            "rationale": "One BF16 machine epsilon in both absolute and relative terms permits reduction-order variation without masking large errors.",
        },
        "cases": [
            {
                "name": case["name"],
                "input_shape": [case["rows"], case["hidden_size"]],
                "weight_shape": [case["hidden_size"]],
                "output_shape": [case["rows"], case["hidden_size"]],
            }
            for case in cases
        ],
        "generation": {
            "command": "uv run --script tests/fixtures/rms_norm/generate.py",
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
