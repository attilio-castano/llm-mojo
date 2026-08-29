"""Run and record the Apple GPU BK16 versus direct prefill experiment."""

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
    from benchmarks.run_linear_prefill_bk_sweep import (
        EXPECTED_REPETITIONS,
        INPUT_FEATURES,
        OUTPUT_FEATURES,
        RING_LAYERS,
        WORKLOAD_ORDER,
        WORKLOAD_ROWS,
        WORKLOADS,
        requested_traffic_bytes,
        table_lines,
        workload_id,
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
    from run_linear_prefill_bk_sweep import (  # type: ignore[no-redef]
        EXPECTED_REPETITIONS,
        INPUT_FEATURES,
        OUTPUT_FEATURES,
        RING_LAYERS,
        WORKLOAD_ORDER,
        WORKLOAD_ROWS,
        WORKLOADS,
        requested_traffic_bytes,
        table_lines,
        workload_id,
    )


REPOSITORY = Path(__file__).resolve().parents[1]
BK = 16
BLOCK_ORDERS = ("ascending", "descending", "descending", "ascending")
BLOCK_IMPLEMENTATION_ORDERS = (
    "direct_then_bk16",
    "bk16_then_direct",
    "bk16_then_direct",
    "direct_then_bk16",
)
ADVANCE_REQUIRED_ROWS = (64, 128, 256)
ADVANCE_GUARD_ROWS = (8, 16, 32, 64, 128, 256)

DIRECT_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_direct_8x16_v0",
    "entrypoint": "enqueue_linear_prefill_direct_apple_gpu",
    "kind": "direct",
}
BK16_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_tiled_8x16x16_v0",
    "entrypoint": "enqueue_linear_prefill_tiled_apple_gpu_bk",
    "kind": "bk16",
}
IMPLEMENTATIONS = {
    "linear_prefill_direct_apple_gpu": DIRECT_IMPLEMENTATION,
    "linear_prefill_tiled_bk16_apple_gpu": BK16_IMPLEMENTATION,
}
BENCHMARK_NAME = re.compile(
    r"^(linear_prefill_(?:direct|tiled_bk16)_apple_gpu)/input_id:"
    r"m(\d+)-k(\d+)-n(\d+)-layers(\d+)$"
)


def direct_requested_traffic_bytes(specification: dict[str, Any]) -> int:
    rows = int(specification["rows"])
    return 2 * RING_LAYERS * (
        2 * rows * OUTPUT_FEATURES * INPUT_FEATURES
        + 2 * rows * OUTPUT_FEATURES
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
    comparison_bk = re.search(
        r"^comparison BK:\s*(.+)$", output, re.MULTILINE
    )
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    emitted_order = re.search(
        r"^implementation order:\s*(.+)$", output, re.MULTILINE
    )
    if not all(
        (implementation, comparison, comparison_bk, device, api, emitted_order)
    ):
        raise ValueError("direct/BK16 benchmark omitted runtime identity")
    assert implementation is not None
    assert comparison is not None
    assert comparison_bk is not None
    assert device is not None
    assert api is not None
    assert emitted_order is not None
    identity = {
        "implementation": implementation.group(1).strip(),
        "comparison_implementation": comparison.group(1).strip(),
        "comparison_bk": comparison_bk.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
        "implementation_order": emitted_order.group(1).strip(),
    }
    if identity["implementation"] != DIRECT_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the direct control")
    if identity["comparison_implementation"] != BK16_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the BK16 candidate")
    if identity["comparison_bk"] != str(BK):
        raise ValueError("benchmark identified an unexpected candidate BK")
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError(
            "benchmark did not prove an Apple device using the Metal API"
        )
    expected_order = (
        "bk16,direct"
        if implementation_order == "bk16_then_direct"
        else "direct,bk16"
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
        "direct_then_bk16",
        "bk16_then_direct",
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
            raise ValueError(f"unrecognized direct/BK16 row: {row!r}")
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
            raise ValueError(f"unexpected direct/BK16 workload {current_workload}")
        order_key = (current_workload, implementation["id"])
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
        is_bk16 = implementation == BK16_IMPLEMENTATION
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
                "bk": BK if is_bk16 else None,
                "k_phases_per_dispatch": 56 if is_bk16 else 0,
                "barriers_per_dispatch": 112 if is_bk16 else 0,
                "threadgroup_operand_scratch_bytes": 768 if is_bk16 else 0,
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
                    requested_traffic_bytes(specification)
                    if is_bk16
                    else direct_requested_traffic_bytes(specification)
                ),
                "traffic_note": (
                    "Source-derived logical loads; not observed hardware traffic."
                ),
            }
        )

    workload_order = list(
        WORKLOAD_ORDER if block_order == "ascending" else reversed(WORKLOAD_ORDER)
    )
    implementation_ids = (
        [BK16_IMPLEMENTATION["id"], DIRECT_IMPLEMENTATION["id"]]
        if implementation_order == "bk16_then_direct"
        else [DIRECT_IMPLEMENTATION["id"], BK16_IMPLEMENTATION["id"]]
    )
    expected_order = [
        (current_workload, implementation_id)
        for current_workload in workload_order
        for implementation_id in implementation_ids
    ]
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got "
            f"{observed_order}"
        )
    expected_counts = {
        (current_workload, implementation_id): EXPECTED_REPETITIONS
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
    samples: Iterable[dict[str, Any]], *, implementation: dict[str, str]
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
                (current_workload, block_id, DIRECT_IMPLEMENTATION["id"])
            ]
            bk16 = grouped[
                (current_workload, block_id, BK16_IMPLEMENTATION["id"])
            ]
            if (
                len(direct) != EXPECTED_REPETITIONS
                or len(bk16) != EXPECTED_REPETITIONS
            ):
                raise ValueError(
                    "paired summary requires complete samples for "
                    f"{current_workload} {block_id}"
                )
            direct_median = statistics.median(direct)
            bk16_median = statistics.median(bk16)
            ratio = bk16_median / direct_median
            blocks.append(
                {
                    "block_id": block_id,
                    "direct_median_ms": direct_median,
                    "bk16_median_ms": bk16_median,
                    "bk16_to_direct_ratio": ratio,
                    "bk16_faster": ratio < 1.0,
                }
            )
        ratios = [float(block["bk16_to_direct_ratio"]) for block in blocks]
        classification, faster_blocks, slower_blocks = classify_ratios(ratios)
        workloads.append(
            {
                "workload": current_workload,
                "rows": WORKLOADS[current_workload]["rows"],
                "median_bk16_to_direct_ratio": statistics.median(ratios),
                "median_change_percent": (statistics.median(ratios) - 1.0)
                * 100.0,
                "bk16_faster_blocks": faster_blocks,
                "bk16_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": blocks,
            }
        )
    return {
        "ratio_definition": "BK16 block median / direct block median",
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
        decision = "advance_bk16_to_public_rowwise_comparison"
    elif rejected:
        decision = "reject_bk16_shared_staging_for_scalar_output_mapping"
    else:
        decision = "no_advance_mixed_or_inconclusive"
    return {
        "schema_version": 1,
        "implementations": [
            summarize_implementation(retained, implementation=implementation)
            for implementation in (DIRECT_IMPLEMENTATION, BK16_IMPLEMENTATION)
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
        "LINEAR_PREFILL_BK16_DIRECT_COMPARISON=true",
    ]
    if BLOCK_ORDERS[block_number - 1] == "descending":
        args.extend(["-D", "LINEAR_PREFILL_BK_SWEEP_REVERSE=true"])
    if BLOCK_IMPLEMENTATION_ORDERS[block_number - 1] == "bk16_then_direct":
        args.extend(["-D", "LINEAR_PREFILL_BK16_FIRST=true"])
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

    result_summary = summarize(samples) if args.blocks == 4 else {
        "schema_version": 1,
        "status": "incomplete_exploration",
        "completed_blocks": args.blocks,
        "implementations": [
            summarize_implementation(samples, implementation=implementation)
            for implementation in (DIRECT_IMPLEMENTATION, BK16_IMPLEMENTATION)
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
            "accumulation; direct 8x16 control versus shared 8x16x16 candidate"
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
