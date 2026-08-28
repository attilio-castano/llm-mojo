#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Generate deterministic Qwen2 affine linear projection oracle fixtures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers


FIXTURE_DIR = Path(__file__).resolve().parent
DATA_PATH = FIXTURE_DIR / "reference_data.mojo"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ATOL = 0.0078125
RTOL = 0.0078125


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
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    output_features, input_features = weight.shape
    module = torch.nn.Linear(
        input_features,
        output_features,
        bias=True,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        module.weight.copy_(weight)
        module.bias.copy_(bias)
        return module(input_tensor).contiguous()


def as_float32_list(tensor: torch.Tensor) -> list[float]:
    if not torch.isfinite(tensor).all():
        raise ValueError("linear fixtures require finite values")
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
            "name": "tiny_decode",
            "rows": 1,
            "input_features": 7,
            "output_features": 5,
            "input": [1.0, -0.5, 0.25, 2.0, -1.5, 0.75, -0.125],
            "weight": [
                ((index * 11) % 29 - 14) / 16
                for index in range(5 * 7)
            ],
            "bias": [-0.25, 0.125, 0.5, -0.375, 0.0625],
        },
        {
            "name": "short_prefill",
            "rows": 3,
            "input_features": 33,
            "output_features": 5,
            "input": deterministic_values(3 * 33, 2, 64),
            "weight": deterministic_values(5 * 33, 3, 128),
            "bias": deterministic_values(5, 4, 64),
        },
    ]
    cases: list[dict[str, Any]] = []
    for spec in specs:
        rows = spec["rows"]
        input_features = spec["input_features"]
        output_features = spec["output_features"]
        input_tensor = bf16(spec["input"], (rows, input_features))
        weight = bf16(spec["weight"], (output_features, input_features))
        bias = bf16(spec["bias"], (output_features,))
        expected = run_oracle(input_tensor, weight, bias)
        cases.append(
            {
                **spec,
                "input": as_float32_list(input_tensor),
                "weight": as_float32_list(weight),
                "bias": as_float32_list(bias),
                "expected": as_float32_list(expected),
            }
        )
    return cases


def render_data(cases: list[dict[str, Any]]) -> str:
    sections = [
        '"""Generated Qwen2 affine linear oracle data. Do not edit by hand."""',
        "",
        f"comptime LINEAR_ATOL: Float32 = {ATOL}",
        f"comptime LINEAR_RTOL: Float32 = {RTOL}",
        "",
    ]
    for case in cases:
        prefix = case["name"].upper()
        sections.extend(
            [
                f"comptime {prefix}_ROWS = {case['rows']}",
                f"comptime {prefix}_INPUT_FEATURES = {case['input_features']}",
                f"comptime {prefix}_OUTPUT_FEATURES = {case['output_features']}",
                "",
                render_list(f"{case['name']}_input", case["input"]),
                render_list(f"{case['name']}_weight", case["weight"]),
                render_list(f"{case['name']}_bias", case["bias"]),
                render_list(f"{case['name']}_expected", case["expected"]),
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def build_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "Numerical parity for the Mojo Qwen2 BF16 affine linear reference path.",
        "target_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifact_used": False,
            "note": "Synthetic tensors exercise decode, short-prefill, bias, and non-SIMD-aligned input widths; no model weights were downloaded or extracted.",
        },
        "source_tensors": {
            "input": "deterministic synthetic BF16 values defined in generate.py",
            "weight": "deterministic synthetic BF16 values in source-compatible (output_features, input_features) order",
            "bias": "deterministic synthetic BF16 values defined in generate.py",
        },
        "oracle": {
            "implementation": "torch.nn.Linear",
            "model_call_site": "transformers.models.qwen2.modeling_qwen2.Qwen2Attention q_proj, k_proj, and v_proj",
            "source": "https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "contract": {
            "input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "bias_dtype": "bfloat16",
            "accumulation_dtype": "float32",
            "output_dtype": "bfloat16",
            "arithmetic_order": "FP32 products accumulate over input_features, BF16 bias is promoted and added in FP32, then the result is cast once to BF16",
            "input_layout": "(rows, input_features):(input_features, 1)",
            "weight_layout": "(output_features, input_features):(input_features, 1)",
            "bias_layout": "(output_features):(1)",
            "output_layout": "(rows, output_features):(output_features, 1)",
        },
        "comparison": {
            "formula": "abs(actual - expected) <= atol + rtol * abs(expected)",
            "atol": ATOL,
            "rtol": RTOL,
            "declared_before_first_mojo_comparison": True,
            "rationale": "One BF16 machine epsilon in absolute and relative terms permits backend reduction-order variation without masking transpose, bias, or indexing errors.",
        },
        "cases": [
            {
                "name": case["name"],
                "input_shape": [case["rows"], case["input_features"]],
                "weight_shape": [
                    case["output_features"],
                    case["input_features"],
                ],
                "bias_shape": [case["output_features"]],
                "output_shape": [case["rows"], case["output_features"]],
            }
            for case in cases
        ],
        "generation": {
            "command": "uv run --script tests/fixtures/linear/generate.py",
            "generator_sha256": sha256(Path(__file__)),
            "data_sha256": sha256(DATA_PATH),
        },
    }


def main() -> None:
    cases = build_cases()
    DATA_PATH.write_text(render_data(cases), encoding="utf-8")
    manifest = build_manifest(cases)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
