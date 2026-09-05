#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Generate deterministic Qwen2 RoPE oracle fixtures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)


FIXTURE_DIR = Path(__file__).resolve().parents[3] / "build/oracle_data" / Path(__file__).parent.name
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = FIXTURE_DIR / "reference_data.mojo"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ROPE_THETA = 1_000_000.0
ATOL = 0.0078125
RTOL = 0.0078125


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16(values: list[float], shape: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(shape).to(torch.bfloat16)


def deterministic_values(count: int, seed: int) -> list[float]:
    return [((((index * 37 + seed * 19) % 257) - 128) / 32) for index in range(count)]


def run_oracle(
    input_tensor: torch.Tensor,
    start_position: int,
    table_positions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, _, head_dim = input_tensor.shape
    attention_layout = input_tensor.permute(1, 0, 2).unsqueeze(0).contiguous()
    rotary = Qwen2RotaryEmbedding(
        head_dim,
        max_position_embeddings=table_positions,
        base=ROPE_THETA,
    )
    with torch.no_grad():
        cosine, sine = rotary(attention_layout, seq_len=table_positions)
        position_ids = torch.arange(
            start_position,
            start_position + rows,
            dtype=torch.int64,
        ).unsqueeze(0)
        rotated, _ = apply_rotary_pos_emb(
            attention_layout,
            attention_layout,
            cosine,
            sine,
            position_ids,
        )
    output = rotated.squeeze(0).transpose(0, 1).contiguous()
    selected_cosine = cosine[start_position : start_position + rows].contiguous()
    selected_sine = sine[start_position : start_position + rows].contiguous()
    return output, selected_cosine, selected_sine


def as_float32_list(tensor: torch.Tensor) -> list[float]:
    if not torch.isfinite(tensor).all():
        raise ValueError("RoPE fixtures require finite values")
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
    specs = [
        {
            "name": "tiny",
            "rows": 2,
            "heads": 1,
            "head_dim": 4,
            "start_position": 3,
            "table_positions": 5,
            "values": [1.0, 2.0, 3.0, 4.0, -1.5, 0.5, 2.5, -3.5],
        },
        {
            "name": "qwen_query_decode",
            "rows": 1,
            "heads": 14,
            "head_dim": 64,
            "start_position": 127,
            "table_positions": 128,
            "values": None,
        },
        {
            "name": "qwen_key_incremental",
            "rows": 3,
            "heads": 2,
            "head_dim": 64,
            "start_position": 4093,
            "table_positions": 4096,
            "values": None,
        },
    ]
    cases: list[dict[str, Any]] = []
    for case_index, spec in enumerate(specs):
        rows = spec["rows"]
        heads = spec["heads"]
        head_dim = spec["head_dim"]
        count = rows * heads * head_dim
        source_values = spec["values"]
        if source_values is None:
            source_values = deterministic_values(count, case_index + 1)
        input_tensor = bf16(source_values, (rows, heads, head_dim))
        expected, cosine_rows, sine_rows = run_oracle(
            input_tensor,
            spec["start_position"],
            spec["table_positions"],
        )
        cases.append(
            {
                **spec,
                "input": as_float32_list(input_tensor),
                "cosine_rows": as_float32_list(cosine_rows),
                "sine_rows": as_float32_list(sine_rows),
                "expected": as_float32_list(expected),
            }
        )
    return cases


def render_data(cases: list[dict[str, Any]]) -> str:
    sections = [
        '"""Generated Qwen2 RoPE oracle data. Do not edit by hand."""',
        "",
        f"comptime ROPE_ATOL: Float32 = {ATOL}",
        f"comptime ROPE_RTOL: Float32 = {RTOL}",
        "",
    ]
    for case in cases:
        prefix = case["name"].upper()
        sections.extend(
            [
                f"comptime {prefix}_ROWS = {case['rows']}",
                f"comptime {prefix}_HEADS = {case['heads']}",
                f"comptime {prefix}_HEAD_DIM = {case['head_dim']}",
                f"comptime {prefix}_START_POSITION = {case['start_position']}",
                f"comptime {prefix}_TABLE_POSITIONS = {case['table_positions']}",
                "",
                render_list(f"{case['name']}_input", case["input"]),
                render_list(f"{case['name']}_cosine_rows", case["cosine_rows"]),
                render_list(f"{case['name']}_sine_rows", case["sine_rows"]),
                render_list(f"{case['name']}_expected", case["expected"]),
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def build_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "Numerical parity for the Mojo Qwen2 BF16 RoPE reference path.",
        "target_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifact_used": False,
            "note": "Synthetic tensors exercise the pinned model's Q/K head shapes and V0 position boundary; no model weights were downloaded or extracted.",
        },
        "source_tensors": {
            "input": "deterministic synthetic BF16 values defined in generate.py",
            "cosine_and_sine": "Qwen2RotaryEmbedding tables generated for each absolute position range",
        },
        "oracle": {
            "implementation": "transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb",
            "rotary_table": "transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding",
            "source": "https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "contract": {
            "input_dtype": "bfloat16",
            "table_generation_dtype": "float32",
            "table_application_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "rope_theta": ROPE_THETA,
            "head_pairing": "dimension i pairs with i + head_dim / 2",
            "arithmetic_order": "BF16 input*cosine and BF16 rotate_half(input)*sine products are materialized before the BF16 add",
            "input_layout": "(rows, heads, head_dim):(heads*head_dim, head_dim, 1)",
            "table_layout": "(table_positions, head_dim):(head_dim, 1)",
            "output_layout": "(rows, heads, head_dim):(heads*head_dim, head_dim, 1)",
            "position_mapping": "table row = start_position + input row",
        },
        "comparison": {
            "formula": "abs(actual - expected) <= atol + rtol * abs(expected)",
            "atol": ATOL,
            "rtol": RTOL,
            "declared_before_first_mojo_comparison": True,
            "rationale": "One BF16 machine epsilon in absolute and relative terms permits backend arithmetic variation without masking position or half-split pairing errors.",
        },
        "cases": [
            {
                "name": case["name"],
                "input_shape": [case["rows"], case["heads"], case["head_dim"]],
                "stored_table_rows_shape": [case["rows"], case["head_dim"]],
                "runtime_table_shape": [
                    case["table_positions"],
                    case["head_dim"],
                ],
                "output_shape": [case["rows"], case["heads"], case["head_dim"]],
                "start_position": case["start_position"],
                "last_position": case["start_position"] + case["rows"] - 1,
            }
            for case in cases
        ],
        "generation": {
            "command": "uv run --script tests/fixtures/rope/generate.py",
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
