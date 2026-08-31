"""Run and record the Apple GPU prefill BK sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from benchmarks.run_linear import (
        BENCHMARK_RESULTS_BEGIN,
        BENCHMARK_RESULTS_END,
        MATERIAL_IMPROVEMENT_RATIO,
        MATERIAL_REGRESSION_RATIO,
        REQUIRED_DIRECTION_BLOCKS,
        conditions_snapshot,
        ensure_record_location,
        percentile,
        repository_state,
        require_ac,
        require_nominal_thermal_state,
        sha256_bytes,
        stable_environment,
        utc_now,
        write_run_artifacts,
    )
except ModuleNotFoundError:
    from run_linear import (  # type: ignore[no-redef]
        BENCHMARK_RESULTS_BEGIN,
        BENCHMARK_RESULTS_END,
        MATERIAL_IMPROVEMENT_RATIO,
        MATERIAL_REGRESSION_RATIO,
        REQUIRED_DIRECTION_BLOCKS,
        conditions_snapshot,
        ensure_record_location,
        percentile,
        repository_state,
        require_ac,
        require_nominal_thermal_state,
        sha256_bytes,
        stable_environment,
        utc_now,
        write_run_artifacts,
    )


REPOSITORY = Path(__file__).resolve().parents[1]
INPUT_FEATURES = 896
OUTPUT_FEATURES = 1_152
RING_LAYERS = 24
WORKLOAD_ROWS = (1, 4, 8, 16, 32, 64, 128, 256)
BK_VALUES = (16, 32, 64, 128)
CONTROL_BK = 32
EXPECTED_REPETITIONS = 10
BLOCK_ORDERS = ("ascending", "descending", "descending", "ascending")
BLOCK_BK_ORDERS = (
    (16, 32, 128, 64),
    (32, 64, 16, 128),
    (64, 128, 32, 16),
    (128, 16, 64, 32),
)
ADVANCE_REQUIRED_ROWS = (64, 128, 256)
ADVANCE_GUARD_ROWS = (8, 16, 32, 64, 128, 256)

IMPLEMENTATIONS = {
    bk: {
        "id": f"apple_gpu_prefill_tiled_8x16x{bk}_v0",
        "entrypoint": "enqueue_linear_prefill_tiled_apple_gpu_bk",
        "bk": bk,
    }
    for bk in BK_VALUES
}
BENCHMARK_NAME = re.compile(
    r"^linear_prefill_tiled_bk(16|32|64|128)_apple_gpu/input_id:"
    r"m(\d+)-k(\d+)-n(\d+)-layers(\d+)$"
)


def workload_id(rows: int) -> str:
    return (
        f"m{rows}-k{INPUT_FEATURES}-n{OUTPUT_FEATURES}-layers{RING_LAYERS}"
    )


WORKLOAD_ORDER = tuple(workload_id(rows) for rows in WORKLOAD_ROWS)
WORKLOADS = {
    workload_id(rows): {
        "rows": rows,
        "input_features": INPUT_FEATURES,
        "output_features": OUTPUT_FEATURES,
        "layers": RING_LAYERS,
        "semantic_role": "24-layer rotating packed-QKV cache-pressure proxy",
        "macs": RING_LAYERS * rows * OUTPUT_FEATURES * INPUT_FEATURES,
        "allocated_footprint_bytes": 2
        * (
            rows * INPUT_FEATURES
            + RING_LAYERS * OUTPUT_FEATURES * INPUT_FEATURES
            + OUTPUT_FEATURES
            + rows * OUTPUT_FEATURES
        ),
    }
    for rows in WORKLOAD_ROWS
}


def requested_traffic_bytes(specification: dict[str, Any]) -> int:
    rows = int(specification["rows"])
    input_loads = rows * math.ceil(OUTPUT_FEATURES / 16) * INPUT_FEATURES
    weight_loads = math.ceil(rows / 8) * OUTPUT_FEATURES * INPUT_FEATURES
    bias_loads_and_output_stores = 2 * rows * OUTPUT_FEATURES
    return 2 * RING_LAYERS * (
        input_loads + weight_loads + bias_loads_and_output_stores
    )


def parse_identity(output: str, *, bk_order: tuple[int, ...]) -> dict[str, str]:
    implementation = re.search(
        r"^implementation:\s*(.+)$", output, re.MULTILINE
    )
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    emitted_order = re.search(r"^BK order:\s*(.+)$", output, re.MULTILINE)
    if not implementation or not device or not api or not emitted_order:
        raise ValueError("BK benchmark output omitted runtime identity")
    identity = {
        "implementation": implementation.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
        "bk_order": emitted_order.group(1).strip(),
    }
    if identity["implementation"] != "enqueue_linear_prefill_tiled_apple_gpu_bk":
        raise ValueError("BK benchmark identified an unexpected implementation")
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError(
            "BK benchmark did not prove an Apple device using the Metal API"
        )
    expected_order = ",".join(str(bk) for bk in bk_order)
    if identity["bk_order"].replace(" ", "") != expected_order:
        raise ValueError("BK benchmark emitted an unexpected BK order")
    return identity


def table_lines(output: str) -> list[str]:
    try:
        start = output.index(BENCHMARK_RESULTS_BEGIN)
        end = output.index(BENCHMARK_RESULTS_END, start)
    except ValueError as error:
        raise ValueError("benchmark result markers were not found") from error
    payload = output[start + len(BENCHMARK_RESULTS_BEGIN) : end].strip()
    lines = [line for line in payload.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("benchmark result table is empty")
    return lines


def parse_samples(
    output: str,
    *,
    experiment_id: str,
    run_id: str,
    block_id: str,
    block_order: str,
    bk_order: tuple[int, ...],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if block_order not in ("ascending", "descending"):
        raise ValueError("BK benchmark has an invalid workload order")
    if tuple(sorted(bk_order)) != BK_VALUES:
        raise ValueError("BK benchmark order must contain each BK exactly once")
    identity = parse_identity(output, bk_order=bk_order)
    reader = csv.DictReader(table_lines(output), skipinitialspace=True)
    counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    observed_order: list[tuple[str, int]] = []
    samples: list[dict[str, Any]] = []
    for raw_row in reader:
        row = {
            str(key).strip(): str(value).strip()
            for key, value in raw_row.items()
            if key is not None and value is not None
        }
        match = BENCHMARK_NAME.fullmatch(row.get("name", ""))
        if not match:
            raise ValueError(f"unrecognized BK benchmark row: {row!r}")
        bk = int(match.group(1))
        rows = int(match.group(2))
        input_features = int(match.group(3))
        output_features = int(match.group(4))
        layers = int(match.group(5))
        current_workload = workload_id(rows)
        if (
            input_features != INPUT_FEATURES
            or output_features != OUTPUT_FEATURES
            or layers != RING_LAYERS
            or current_workload not in WORKLOADS
        ):
            raise ValueError(f"unexpected BK workload {current_workload}")
        order_key = (current_workload, bk)
        if not observed_order or observed_order[-1] != order_key:
            observed_order.append(order_key)

        value_text = row.get("met (ms)", "")
        iterations_text = row.get("iters", "")
        value = float(value_text)
        iterations = int(iterations_text)
        valid = math.isfinite(value) and value > 0 and iterations > 0
        counts[order_key] += 1
        repetition = counts[order_key]
        specification = WORKLOADS[current_workload]
        implementation = IMPLEMENTATIONS[bk]
        phases = math.ceil(input_features / bk)
        samples.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "block_id": block_id,
                "block_order": block_order,
                "bk_order": list(bk_order),
                "sample_id": (
                    f"{run_id}-{block_id}-{implementation['id']}-"
                    f"{current_workload}-rep{repetition:02d}"
                ),
                "implementation": implementation["id"],
                "implementation_entrypoint": implementation["entrypoint"],
                "workload": current_workload,
                "semantic_role": specification["semantic_role"],
                "rows": rows,
                "input_features": input_features,
                "output_features": output_features,
                "layers": layers,
                "dispatches_per_iteration": layers,
                "bk": bk,
                "k_phases_per_dispatch": phases,
                "barriers_per_k_phase": 2,
                "barriers_per_dispatch": 2 * phases,
                "threadgroup_operand_scratch_bytes": 2 * (8 + 16) * bk,
                "repetition": repetition,
                "value": value,
                "source_value": value_text,
                "unit": "ms_per_workload_iteration",
                "ms_per_layer": value / layers,
                "iterations": iterations,
                "valid": valid,
                "macs_per_iteration": specification["macs"],
                "allocated_footprint_bytes": specification[
                    "allocated_footprint_bytes"
                ],
                "program_requested_traffic_bytes": requested_traffic_bytes(
                    specification
                ),
                "traffic_note": (
                    "Source-derived logical loads; not observed hardware traffic."
                ),
            }
        )

    workload_order = list(
        WORKLOAD_ORDER if block_order == "ascending" else reversed(WORKLOAD_ORDER)
    )
    expected_order = [
        (current_workload, bk)
        for current_workload in workload_order
        for bk in bk_order
    ]
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got "
            f"{observed_order}"
        )
    expected_counts = {
        (current_workload, bk): EXPECTED_REPETITIONS
        for current_workload in WORKLOAD_ORDER
        for bk in BK_VALUES
    }
    if dict(counts) != expected_counts:
        raise ValueError(
            f"repetition count mismatch: expected {expected_counts}, got "
            f"{dict(counts)}"
        )
    if any(not sample["valid"] for sample in samples):
        raise ValueError("benchmark emitted a non-positive or non-finite sample")
    return identity, samples


def classify_ratios(ratios: list[float]) -> tuple[str, int, int]:
    median_ratio = statistics.median(ratios)
    faster_blocks = sum(ratio < 1.0 for ratio in ratios)
    slower_blocks = sum(ratio > 1.0 for ratio in ratios)
    if (
        median_ratio <= MATERIAL_IMPROVEMENT_RATIO
        and faster_blocks >= REQUIRED_DIRECTION_BLOCKS
    ):
        classification = "material_improvement"
    elif (
        median_ratio >= MATERIAL_REGRESSION_RATIO
        and slower_blocks >= REQUIRED_DIRECTION_BLOCKS
    ):
        classification = "material_regression"
    else:
        classification = "inconclusive"
    return classification, faster_blocks, slower_blocks


def summarize_implementation(
    samples: Iterable[dict[str, Any]], *, bk: int
) -> dict[str, Any]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample["valid"] and int(sample["bk"]) == bk:
            grouped[sample["workload"]].append(float(sample["value"]))

    workloads = []
    for current_workload in WORKLOAD_ORDER:
        values = grouped[current_workload]
        if not values:
            continue
        specification = WORKLOADS[current_workload]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        rows = int(specification["rows"])
        macs = int(specification["macs"])
        workloads.append(
            {
                "workload": current_workload,
                "rows": rows,
                "count": len(values),
                "median_ms_per_workload_iteration": median,
                "median_ms_per_layer": median / RING_LAYERS,
                "median_us_per_row_per_layer": (
                    median * 1_000.0 / (rows * RING_LAYERS)
                ),
                "median_absolute_deviation_ms": statistics.median(deviations),
                "p25_ms": percentile(values, 0.25),
                "p75_ms": percentile(values, 0.75),
                "min_ms": min(values),
                "max_ms": max(values),
                "effective_gflop_per_second": (
                    2 * macs / (median * 1_000_000.0)
                ),
            }
        )
    return {
        "implementation": IMPLEMENTATIONS[bk],
        "threadgroup_operand_scratch_bytes": 2 * (8 + 16) * bk,
        "k_phases_per_dispatch": math.ceil(INPUT_FEATURES / bk),
        "barriers_per_dispatch": 2 * math.ceil(INPUT_FEATURES / bk),
        "workloads": workloads,
    }


def summarize_candidate(
    samples: Iterable[dict[str, Any]], *, candidate_bk: int
) -> dict[str, Any]:
    retained = list(samples)
    grouped: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
    for sample in retained:
        if sample["valid"]:
            grouped[
                (sample["workload"], sample["block_id"], int(sample["bk"]))
            ].append(float(sample["value"]))

    workloads = []
    for current_workload in WORKLOAD_ORDER:
        blocks = []
        for block_number in range(1, 5):
            block_id = f"block-{block_number:02d}"
            control = grouped[(current_workload, block_id, CONTROL_BK)]
            candidate = grouped[(current_workload, block_id, candidate_bk)]
            if (
                len(control) != EXPECTED_REPETITIONS
                or len(candidate) != EXPECTED_REPETITIONS
            ):
                raise ValueError(
                    "paired BK summary requires complete samples for "
                    f"{current_workload} {block_id}"
                )
            control_median = statistics.median(control)
            candidate_median = statistics.median(candidate)
            ratio = candidate_median / control_median
            blocks.append(
                {
                    "block_id": block_id,
                    "bk32_median_ms": control_median,
                    "candidate_median_ms": candidate_median,
                    "candidate_to_bk32_ratio": ratio,
                    "candidate_faster": ratio < 1.0,
                }
            )
        ratios = [float(block["candidate_to_bk32_ratio"]) for block in blocks]
        classification, faster_blocks, slower_blocks = classify_ratios(ratios)
        specification = WORKLOADS[current_workload]
        workloads.append(
            {
                "workload": current_workload,
                "rows": specification["rows"],
                "candidate_bk": candidate_bk,
                "median_candidate_to_bk32_ratio": statistics.median(ratios),
                "median_change_percent": (statistics.median(ratios) - 1.0)
                * 100.0,
                "candidate_faster_blocks": faster_blocks,
                "candidate_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": blocks,
            }
        )
    return {
        "candidate_bk": candidate_bk,
        "control_bk": CONTROL_BK,
        "ratio_definition": "candidate block median / BK32 block median",
        "workloads": workloads,
    }


def summarize(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    retained = list(samples)
    candidates = [
        summarize_candidate(retained, candidate_bk=bk)
        for bk in BK_VALUES
        if bk != CONTROL_BK
    ]
    eligibility: dict[int, bool] = {}
    for candidate in candidates:
        by_rows = {int(item["rows"]): item for item in candidate["workloads"]}
        eligibility[int(candidate["candidate_bk"])] = all(
            by_rows[rows]["classification"] == "material_improvement"
            for rows in ADVANCE_REQUIRED_ROWS
        ) and all(
            by_rows[rows]["classification"] != "material_regression"
            for rows in ADVANCE_GUARD_ROWS
        )
    eligible = [bk for bk in BK_VALUES if eligibility.get(bk, False)]
    if eligible:
        candidate_lookup = {
            int(candidate["candidate_bk"]): candidate for candidate in candidates
        }
        winner = min(
            eligible,
            key=lambda bk: statistics.median(
                float(item["median_candidate_to_bk32_ratio"])
                for item in candidate_lookup[bk]["workloads"]
                if int(item["rows"]) in ADVANCE_REQUIRED_ROWS
            ),
        )
        decision = f"advance_bk{winner}_to_direct_control_comparison"
    else:
        winner = None
        decision = "no_bk_qualifies_for_direct_control_comparison"
    return {
        "schema_version": 1,
        "implementations": [
            summarize_implementation(retained, bk=bk) for bk in BK_VALUES
        ],
        "paired_against_bk32": candidates,
        "classification_rule": {
            "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
            "material_regression_ratio": MATERIAL_REGRESSION_RATIO,
            "required_direction_blocks": REQUIRED_DIRECTION_BLOCKS,
        },
        "advance_rule": {
            "required_material_improvement_rows": list(ADVANCE_REQUIRED_ROWS),
            "no_material_regression_rows": list(ADVANCE_GUARD_ROWS),
            "eligible_candidates": eligible,
            "selected_candidate_bk": winner,
        },
        "timing_decision": decision,
        "production_dispatch_decision": "none",
    }


def benchmark_command(*, block_number: int) -> list[str]:
    if block_number not in range(1, 5):
        raise ValueError("block number must be in 1...4")
    args = ["uv", "run", "--locked", "mojo", "run", "-I", "src"]
    if BLOCK_ORDERS[block_number - 1] == "descending":
        args.extend(["-D", "LINEAR_PREFILL_BK_SWEEP_REVERSE=true"])
    if block_number > 1:
        args.extend(
            [
                "-D",
                f"LINEAR_PREFILL_BK_SWEEP_SEQUENCE_{block_number}=true",
            ]
        )
    args.append("benchmarks/linear_prefill_bk_sweep.mojo")
    return args


def run_block(
    *,
    experiment_id: str,
    run_id: str,
    block_number: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    block_id = f"block-{block_number:02d}"
    block_order = BLOCK_ORDERS[block_number - 1]
    bk_order = BLOCK_BK_ORDERS[block_number - 1]
    before = conditions_snapshot()
    command_args = benchmark_command(block_number=block_number)
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    print(
        f"Running {block_id} ({block_order}; BK {bk_order})...", flush=True
    )
    started = utc_now()
    result = subprocess.run(
        command_args,
        cwd=REPOSITORY,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command_args, output=result.stdout
        )
    identity, samples = parse_samples(
        result.stdout,
        experiment_id=experiment_id,
        run_id=run_id,
        block_id=block_id,
        block_order=block_order,
        bk_order=bk_order,
    )
    stdout_bytes = result.stdout.encode()
    block = {
        "block_id": block_id,
        "order": block_order,
        "bk_order": list(bk_order),
        "started_utc": started,
        "completed_utc": utc_now(),
        "command": shlex.join(command_args),
        "environment": {"MODULAR_DEBUG": "unset"},
        "runtime_identity": identity,
        "conditions_before": before,
        "conditions_after": conditions_snapshot(),
        "sample_count": len(samples),
        "stdout_bytes": len(stdout_bytes),
        "stdout_sha256": sha256_bytes(stdout_bytes),
    }
    return block, samples, result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="exploration")
    parser.add_argument("--run-id")
    parser.add_argument("--blocks", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recorded", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recorded and (
        args.blocks != 4
        or args.output_dir is None
        or args.experiment_id == "exploration"
        or args.run_id is None
    ):
        raise RuntimeError(
            "--recorded requires four blocks, an external output directory, "
            "and explicit IDs"
        )

    run_id = args.run_id or datetime.now(UTC).strftime(
        "exploration-%Y%m%dT%H%M%SZ"
    )
    initial_repository = repository_state()
    initial_conditions = conditions_snapshot()
    require_clean = args.recorded or args.require_clean
    if require_clean and initial_repository["dirty"]:
        raise RuntimeError("recorded run requires a clean repository")
    if args.recorded:
        require_ac(initial_conditions)
        require_nominal_thermal_state(initial_conditions)
        assert args.output_dir is not None
        ensure_record_location(args.output_dir)
        if args.output_dir.exists():
            raise RuntimeError("refusing to overwrite an existing run directory")

    blocks: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    outputs: list[str] = []
    for index in range(args.blocks):
        block, block_samples, output = run_block(
            experiment_id=args.experiment_id,
            run_id=run_id,
            block_number=index + 1,
        )
        blocks.append(block)
        samples.extend(block_samples)
        outputs.append(output)
        current_repository = repository_state()
        if require_clean and current_repository != initial_repository:
            raise RuntimeError("repository changed during the recorded run")
        if args.recorded:
            require_ac(block["conditions_after"])
            require_nominal_thermal_state(block["conditions_after"])

    result_summary = summarize(samples) if args.blocks == 4 else {
        "schema_version": 1,
        "status": "incomplete_exploration",
        "completed_blocks": args.blocks,
        "implementations": [
            summarize_implementation(samples, bk=bk) for bk in BK_VALUES
        ],
        "production_dispatch_decision": "none",
    }
    metadata = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_utc": utc_now(),
        "operation": "linear_projection",
        "scope": (
            "M=1..256 BF16 rotating packed-QKV prefill projection with FP32 "
            "accumulation; BM=8, BN=16, BK in 16,32,64,128"
        ),
        "repository": initial_repository,
        "recorded": args.recorded,
        "blocks": blocks,
        "sample_count": len(samples),
        "environment": stable_environment(),
    }
    if args.output_dir is not None:
        write_run_artifacts(
            args.output_dir, metadata, samples, result_summary, outputs
        )
        print(f"artifacts: {args.output_dir.resolve()}")
    print(json.dumps(result_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
