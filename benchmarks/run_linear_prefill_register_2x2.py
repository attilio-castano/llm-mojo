"""Run and record the Apple GPU direct versus 2x2-register prefill experiment."""

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
    from benchmarks.run_linear_prefill_bk_sweep import (
        EXPECTED_REPETITIONS,
        INPUT_FEATURES,
        OUTPUT_FEATURES,
        RING_LAYERS,
        WORKLOAD_ORDER,
        WORKLOADS,
        table_lines,
        workload_id,
    )
except ModuleNotFoundError:
    from run_linear import (  # type: ignore[no-redef]
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
    from run_linear_prefill_bk_sweep import (  # type: ignore[no-redef]
        EXPECTED_REPETITIONS,
        INPUT_FEATURES,
        OUTPUT_FEATURES,
        RING_LAYERS,
        WORKLOAD_ORDER,
        WORKLOADS,
        table_lines,
        workload_id,
    )


REPOSITORY = Path(__file__).resolve().parents[1]
BLOCK_ORDERS = ("ascending", "descending", "descending", "ascending")
BLOCK_IMPLEMENTATION_ORDERS = (
    "direct_then_register_2x2",
    "register_2x2_then_direct",
    "register_2x2_then_direct",
    "direct_then_register_2x2",
)
ADVANCE_REQUIRED_ROWS = (64, 128, 256)
ADVANCE_GUARD_ROWS = (8, 16, 32, 64, 128, 256)

DIRECT_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_direct_8x16_v0",
    "entrypoint": "enqueue_linear_prefill_direct_apple_gpu",
    "kind": "direct",
    "threads_per_threadgroup": 128,
    "outputs_per_thread": 1,
    "fp32_accumulators_per_thread": 1,
}
REGISTER_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_register_2x2_8x16_v0",
    "entrypoint": "enqueue_linear_prefill_register_2x2_apple_gpu",
    "kind": "register_2x2",
    "threads_per_threadgroup": 32,
    "outputs_per_thread": 4,
    "fp32_accumulators_per_thread": 4,
}
IMPLEMENTATIONS = {
    "linear_prefill_direct_apple_gpu": DIRECT_IMPLEMENTATION,
    "linear_prefill_register_2x2_apple_gpu": REGISTER_IMPLEMENTATION,
}
BENCHMARK_NAME = re.compile(
    r"^(linear_prefill_(?:direct|register_2x2)_apple_gpu)/input_id:"
    r"m(\d+)-k(\d+)-n(\d+)-layers(\d+)$"
)


def direct_requested_traffic_bytes(specification: dict[str, Any]) -> int:
    """Count BF16 loads and stores explicitly requested by the direct source."""

    rows = int(specification["rows"])
    operand_loads = 2 * rows * OUTPUT_FEATURES * INPUT_FEATURES
    bias_loads_and_output_stores = 2 * rows * OUTPUT_FEATURES
    return 2 * RING_LAYERS * (operand_loads + bias_loads_and_output_stores)


def register_requested_traffic_bytes(specification: dict[str, Any]) -> int:
    """Count BF16 loads and stores explicitly requested by the 2x2 source."""

    rows = int(specification["rows"])
    row_pairs = math.ceil(rows / 2)
    output_feature_pairs = math.ceil(OUTPUT_FEATURES / 2)
    input_loads = rows * output_feature_pairs * INPUT_FEATURES
    weight_loads = row_pairs * OUTPUT_FEATURES * INPUT_FEATURES
    bias_loads = row_pairs * OUTPUT_FEATURES
    output_stores = rows * OUTPUT_FEATURES
    return 2 * RING_LAYERS * (
        input_loads + weight_loads + bias_loads + output_stores
    )


def parse_identity(
    output: str, *, implementation_order: str
) -> dict[str, str]:
    implementation = re.search(
        r"^implementation:\s*(.+)$", output, re.MULTILINE
    )
    comparison = re.search(
        r"^comparison implementation:\s*(.+)$", output, re.MULTILINE
    )
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    emitted_order = re.search(
        r"^implementation order:\s*(.+)$", output, re.MULTILINE
    )
    if not all((implementation, comparison, device, api, emitted_order)):
        raise ValueError("direct/register benchmark omitted runtime identity")
    assert implementation is not None
    assert comparison is not None
    assert device is not None
    assert api is not None
    assert emitted_order is not None
    identity = {
        "implementation": implementation.group(1).strip(),
        "comparison_implementation": comparison.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
        "implementation_order": emitted_order.group(1).strip(),
    }
    if identity["implementation"] != DIRECT_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the direct control")
    if (
        identity["comparison_implementation"]
        != REGISTER_IMPLEMENTATION["entrypoint"]
    ):
        raise ValueError("benchmark did not identify the register candidate")
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError(
            "benchmark did not prove an Apple device using the Metal API"
        )
    expected_order = (
        "register_2x2,direct"
        if implementation_order == "register_2x2_then_direct"
        else "direct,register_2x2"
    )
    if identity["implementation_order"].replace(" ", "") != expected_order:
        raise ValueError("benchmark emitted an unexpected implementation order")
    return identity


def parse_samples(
    output: str,
    *,
    experiment_id: str,
    run_id: str,
    block_id: str,
    block_order: str,
    implementation_order: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if block_order not in ("ascending", "descending"):
        raise ValueError("benchmark has an invalid workload order")
    if implementation_order not in (
        "direct_then_register_2x2",
        "register_2x2_then_direct",
    ):
        raise ValueError("benchmark has an invalid implementation order")
    identity = parse_identity(
        output, implementation_order=implementation_order
    )
    reader = csv.DictReader(table_lines(output), skipinitialspace=True)
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    observed_order: list[tuple[str, str]] = []
    samples: list[dict[str, Any]] = []
    for raw_row in reader:
        row = {
            str(key).strip(): str(value).strip()
            for key, value in raw_row.items()
            if key is not None and value is not None
        }
        match = BENCHMARK_NAME.fullmatch(row.get("name", ""))
        if not match:
            raise ValueError(f"unrecognized direct/register row: {row!r}")
        implementation = IMPLEMENTATIONS[match.group(1)]
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
            raise ValueError(
                f"unexpected direct/register workload {current_workload}"
            )
        order_key = (current_workload, str(implementation["id"]))
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
        is_register = implementation == REGISTER_IMPLEMENTATION
        samples.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "block_id": block_id,
                "block_order": block_order,
                "implementation_order": implementation_order,
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
                "bk": None,
                "k_phases_per_dispatch": 0,
                "barriers_per_dispatch": 0,
                "threadgroup_operand_scratch_bytes": 0,
                "threads_per_threadgroup": implementation[
                    "threads_per_threadgroup"
                ],
                "outputs_per_thread": implementation["outputs_per_thread"],
                "fp32_accumulators_per_thread": implementation[
                    "fp32_accumulators_per_thread"
                ],
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
                "program_requested_traffic_bytes": (
                    register_requested_traffic_bytes(specification)
                    if is_register
                    else direct_requested_traffic_bytes(specification)
                ),
                "traffic_note": (
                    "Source-derived logical loads and stores; not observed "
                    "hardware traffic."
                ),
            }
        )

    workload_order = list(
        WORKLOAD_ORDER if block_order == "ascending" else reversed(WORKLOAD_ORDER)
    )
    implementation_ids = (
        [REGISTER_IMPLEMENTATION["id"], DIRECT_IMPLEMENTATION["id"]]
        if implementation_order == "register_2x2_then_direct"
        else [DIRECT_IMPLEMENTATION["id"], REGISTER_IMPLEMENTATION["id"]]
    )
    expected_order = [
        (current_workload, str(implementation_id))
        for current_workload in workload_order
        for implementation_id in implementation_ids
    ]
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got "
            f"{observed_order}"
        )
    expected_counts = {
        (current_workload, str(implementation_id)): EXPECTED_REPETITIONS
        for current_workload in WORKLOAD_ORDER
        for implementation_id in implementation_ids
    }
    if dict(counts) != expected_counts:
        raise ValueError(
            f"repetition count mismatch: expected {expected_counts}, got "
            f"{dict(counts)}"
        )
    if any(not sample["valid"] for sample in samples):
        raise ValueError("benchmark emitted a non-positive or non-finite sample")
    return identity, samples


def summarize_implementation(
    samples: Iterable[dict[str, Any]], *, implementation: dict[str, Any]
) -> dict[str, Any]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample["valid"] and sample["implementation"] == implementation["id"]:
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
    return {"implementation": implementation, "workloads": workloads}


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


def summarize_paired(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[
                (sample["workload"], sample["block_id"], sample["implementation"])
            ].append(float(sample["value"]))
    workloads = []
    for current_workload in WORKLOAD_ORDER:
        blocks = []
        for block_number in range(1, 5):
            block_id = f"block-{block_number:02d}"
            direct = grouped[
                (current_workload, block_id, str(DIRECT_IMPLEMENTATION["id"]))
            ]
            register = grouped[
                (current_workload, block_id, str(REGISTER_IMPLEMENTATION["id"]))
            ]
            if (
                len(direct) != EXPECTED_REPETITIONS
                or len(register) != EXPECTED_REPETITIONS
            ):
                raise ValueError(
                    "paired summary requires complete samples for "
                    f"{current_workload} {block_id}"
                )
            direct_median = statistics.median(direct)
            register_median = statistics.median(register)
            ratio = register_median / direct_median
            blocks.append(
                {
                    "block_id": block_id,
                    "direct_median_ms": direct_median,
                    "register_2x2_median_ms": register_median,
                    "register_2x2_to_direct_ratio": ratio,
                    "register_2x2_faster": ratio < 1.0,
                }
            )
        ratios = [
            float(block["register_2x2_to_direct_ratio"]) for block in blocks
        ]
        classification, faster_blocks, slower_blocks = classify_ratios(ratios)
        workloads.append(
            {
                "workload": current_workload,
                "rows": WORKLOADS[current_workload]["rows"],
                "median_register_2x2_to_direct_ratio": statistics.median(ratios),
                "median_change_percent": (statistics.median(ratios) - 1.0)
                * 100.0,
                "register_2x2_faster_blocks": faster_blocks,
                "register_2x2_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": blocks,
            }
        )
    return {
        "ratio_definition": "2x2-register block median / direct block median",
        "workloads": workloads,
    }


def summarize(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    retained = list(samples)
    paired = summarize_paired(retained)
    by_rows = {int(item["rows"]): item for item in paired["workloads"]}
    qualifies = all(
        by_rows[rows]["classification"] == "material_improvement"
        for rows in ADVANCE_REQUIRED_ROWS
    ) and all(
        by_rows[rows]["classification"] != "material_regression"
        for rows in ADVANCE_GUARD_ROWS
    )
    rejected = all(
        by_rows[rows]["classification"] == "material_regression"
        for rows in ADVANCE_REQUIRED_ROWS
    )
    if qualifies:
        decision = "advance_register_2x2_to_public_rowwise_comparison"
    elif rejected:
        decision = "reject_register_2x2_for_prefill_direct_mapping"
    else:
        decision = "no_advance_mixed_or_inconclusive"
    return {
        "schema_version": 1,
        "implementations": [
            summarize_implementation(retained, implementation=implementation)
            for implementation in (DIRECT_IMPLEMENTATION, REGISTER_IMPLEMENTATION)
        ],
        "paired_comparison": paired,
        "classification_rule": {
            "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
            "material_regression_ratio": MATERIAL_REGRESSION_RATIO,
            "required_direction_blocks": REQUIRED_DIRECTION_BLOCKS,
        },
        "advance_rule": {
            "required_material_improvement_rows": list(ADVANCE_REQUIRED_ROWS),
            "no_material_regression_rows": list(ADVANCE_GUARD_ROWS),
            "qualified": qualifies,
            "rejected_at_required_rows": rejected,
        },
        "timing_decision": decision,
        "production_dispatch_decision": "none",
    }


def benchmark_command(*, block_number: int) -> list[str]:
    if block_number not in range(1, 5):
        raise ValueError("block number must be in 1...4")
    args = [
        "uv",
        "run",
        "--locked",
        "mojo",
        "run",
        "-I",
        "src",
        "-D",
        "LINEAR_PREFILL_REGISTER_DIRECT_COMPARISON=true",
    ]
    if BLOCK_ORDERS[block_number - 1] == "descending":
        args.extend(["-D", "LINEAR_PREFILL_BK_SWEEP_REVERSE=true"])
    if (
        BLOCK_IMPLEMENTATION_ORDERS[block_number - 1]
        == "register_2x2_then_direct"
    ):
        args.extend(["-D", "LINEAR_PREFILL_REGISTER_FIRST=true"])
    args.append("benchmarks/linear_prefill_bk_sweep.mojo")
    return args


def run_block(
    *, experiment_id: str, run_id: str, block_number: int
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    block_id = f"block-{block_number:02d}"
    block_order = BLOCK_ORDERS[block_number - 1]
    implementation_order = BLOCK_IMPLEMENTATION_ORDERS[block_number - 1]
    before = conditions_snapshot()
    command_args = benchmark_command(block_number=block_number)
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    print(
        f"Running {block_id} ({block_order}; {implementation_order})...",
        flush=True,
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
        implementation_order=implementation_order,
    )
    stdout_bytes = result.stdout.encode()
    block = {
        "block_id": block_id,
        "order": block_order,
        "implementation_order": implementation_order,
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

    result_summary = (
        summarize(samples)
        if args.blocks == 4
        else {
            "schema_version": 1,
            "status": "incomplete_exploration",
            "completed_blocks": args.blocks,
            "implementations": [
                summarize_implementation(samples, implementation=implementation)
                for implementation in (
                    DIRECT_IMPLEMENTATION,
                    REGISTER_IMPLEMENTATION,
                )
            ],
            "production_dispatch_decision": "none",
        }
    )
    metadata = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_utc": utc_now(),
        "operation": "linear_projection",
        "scope": (
            "M=1..256 BF16 rotating packed-QKV prefill projection with FP32 "
            "accumulation; scalar 8x16 direct control versus direct 2x2 "
            "register ownership"
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
