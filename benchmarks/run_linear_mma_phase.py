"""Run the Apple 8x8-MMA linear experiment against phase-best controls."""

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
    from benchmarks.run_linear_prefill_register_2x2 import (
        classify_ratios,
        direct_requested_traffic_bytes,
        register_requested_traffic_bytes,
        summarize_implementation,
    )
except ModuleNotFoundError:
    from run_linear import (  # type: ignore[no-redef]
        MATERIAL_IMPROVEMENT_RATIO,
        MATERIAL_REGRESSION_RATIO,
        REQUIRED_DIRECTION_BLOCKS,
        conditions_snapshot,
        ensure_record_location,
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
    from run_linear_prefill_register_2x2 import (  # type: ignore[no-redef]
        classify_ratios,
        direct_requested_traffic_bytes,
        register_requested_traffic_bytes,
        summarize_implementation,
    )


REPOSITORY = Path(__file__).resolve().parents[1]
BLOCK_ORDERS = ("ascending", "descending", "descending", "ascending")
BLOCK_IMPLEMENTATION_ORDERS = (
    "control_then_mma",
    "mma_then_control",
    "mma_then_control",
    "control_then_mma",
)
ROWWISE_CONTROL_MAX_ROWS = 8
LARGE_PREFILL_ROWS = (64, 128, 256)
CROSSOVER_CANDIDATE_ROWS = (8, 16, 32, 64)

ROWWISE_IMPLEMENTATION = {
    "id": "apple_gpu_one_output_simdgroup_v0",
    "entrypoint": "enqueue_linear_apple_gpu",
    "kind": "rowwise",
    "threads_per_threadgroup": 128,
    "simd_groups_per_threadgroup": 4,
    "outputs_per_simd_group": 1,
    "outputs_per_thread": None,
    "k_elements_per_lane": math.ceil(INPUT_FEATURES / 32),
    "fp32_accumulators_per_thread": 1,
    "simd_group_reductions_per_output": 1,
}
REGISTER_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_register_2x2_8x16_v0",
    "entrypoint": "enqueue_linear_prefill_register_2x2_apple_gpu",
    "kind": "register_2x2",
    "threads_per_threadgroup": 32,
    "simd_groups_per_threadgroup": 1,
    "outputs_per_simd_group": 128,
    "outputs_per_thread": 4,
    "k_elements_per_lane": INPUT_FEATURES,
    "fp32_accumulators_per_thread": 4,
    "simd_group_reductions_per_output": 0,
}
MMA_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_mma_8x16_v0",
    "entrypoint": "enqueue_linear_prefill_mma_8x16_apple_gpu",
    "kind": "apple_simdgroup_mma_8x8",
    "threads_per_threadgroup": 32,
    "simd_groups_per_threadgroup": 1,
    "outputs_per_simd_group": 128,
    "outputs_per_thread": 4,
    "k_elements_per_lane": None,
    "mma_k": 8,
    "k_phases_per_dispatch": math.ceil(INPUT_FEATURES / 8),
    "mma_operations_per_k_phase": 2,
    "fp32_accumulators_per_thread": 4,
    "simd_group_reductions_per_output": 0,
}
IMPLEMENTATIONS = {
    "linear_prefill_rowwise_apple_gpu": ROWWISE_IMPLEMENTATION,
    "linear_prefill_register_2x2_apple_gpu": REGISTER_IMPLEMENTATION,
    "linear_prefill_mma_8x16_apple_gpu": MMA_IMPLEMENTATION,
}
BENCHMARK_NAME = re.compile(
    r"^(linear_prefill_(?:rowwise|register_2x2|mma_8x16)_apple_gpu)/input_id:"
    r"m(\d+)-k(\d+)-n(\d+)-layers(\d+)$"
)


def control_for_rows(rows: int) -> dict[str, Any]:
    if rows <= ROWWISE_CONTROL_MAX_ROWS:
        return ROWWISE_IMPLEMENTATION
    return REGISTER_IMPLEMENTATION


def mma_requested_traffic_bytes(specification: dict[str, Any]) -> int:
    """Count BF16 loads and stores explicitly requested by the MMA source."""

    rows = int(specification["rows"])
    row_tiles = math.ceil(rows / 8)
    output_tiles = math.ceil(OUTPUT_FEATURES / 16)
    input_loads = rows * INPUT_FEATURES * output_tiles
    weight_loads = row_tiles * OUTPUT_FEATURES * INPUT_FEATURES
    bias_loads = rows * OUTPUT_FEATURES
    output_stores = rows * OUTPUT_FEATURES
    return 2 * RING_LAYERS * (
        input_loads + weight_loads + bias_loads + output_stores
    )


def requested_traffic_bytes(
    specification: dict[str, Any], implementation: dict[str, Any]
) -> int:
    if implementation == MMA_IMPLEMENTATION:
        return mma_requested_traffic_bytes(specification)
    if implementation == REGISTER_IMPLEMENTATION:
        return register_requested_traffic_bytes(specification)
    return direct_requested_traffic_bytes(specification)


def parse_identity(
    output: str, *, block_order: str, implementation_order: str
) -> dict[str, str]:
    implementation = re.search(
        r"^implementation:\s*(.+)$", output, re.MULTILINE
    )
    small_control = re.search(
        r"^M=1,8 control:\s*(.+)$", output, re.MULTILINE
    )
    large_control = re.search(r"^M>=16 control:\s*(.+)$", output, re.MULTILINE)
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    emitted_workload_order = re.search(
        r"^workload order:\s*(.+)$", output, re.MULTILINE
    )
    emitted_implementation_order = re.search(
        r"^implementation order:\s*(.+)$", output, re.MULTILINE
    )
    if not all(
        (
            implementation,
            small_control,
            large_control,
            device,
            api,
            emitted_workload_order,
            emitted_implementation_order,
        )
    ):
        raise ValueError("MMA phase benchmark omitted runtime identity")
    assert implementation is not None
    assert small_control is not None
    assert large_control is not None
    assert device is not None
    assert api is not None
    assert emitted_workload_order is not None
    assert emitted_implementation_order is not None
    identity = {
        "implementation": implementation.group(1).strip(),
        "small_control": small_control.group(1).strip(),
        "large_control": large_control.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
        "workload_order": emitted_workload_order.group(1).strip(),
        "implementation_order": emitted_implementation_order.group(1).strip(),
    }
    if identity["implementation"] != MMA_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the MMA candidate")
    if identity["small_control"] != ROWWISE_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the small-M control")
    if identity["large_control"] != REGISTER_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the large-M control")
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError(
            "benchmark did not prove an Apple device using the Metal API"
        )
    if identity["workload_order"] != block_order:
        raise ValueError("benchmark emitted an unexpected workload order")
    expected_order = (
        "mma,control"
        if implementation_order == "mma_then_control"
        else "control,mma"
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
    if implementation_order not in ("control_then_mma", "mma_then_control"):
        raise ValueError("benchmark has an invalid implementation order")
    identity = parse_identity(
        output,
        block_order=block_order,
        implementation_order=implementation_order,
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
            raise ValueError(f"unrecognized MMA phase row: {row!r}")
        implementation = IMPLEMENTATIONS[match.group(1)]
        rows = int(match.group(2))
        input_features = int(match.group(3))
        output_features = int(match.group(4))
        layers = int(match.group(5))
        current_workload = workload_id(rows)
        control = control_for_rows(rows)
        if implementation not in (MMA_IMPLEMENTATION, control):
            raise ValueError(
                f"unexpected implementation for {current_workload}: "
                f"{implementation['id']}"
            )
        if (
            input_features != INPUT_FEATURES
            or output_features != OUTPUT_FEATURES
            or layers != RING_LAYERS
            or current_workload not in WORKLOADS
        ):
            raise ValueError(f"unexpected MMA phase workload {current_workload}")
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
        is_mma = implementation == MMA_IMPLEMENTATION
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
                "role": "candidate" if is_mma else "control",
                "implementation": implementation["id"],
                "implementation_entrypoint": implementation["entrypoint"],
                "control_implementation": control["id"],
                "workload": current_workload,
                "semantic_role": (
                    "batch-1 decode" if rows == 1 else "packed-QKV prefill"
                ),
                "rows": rows,
                "input_features": input_features,
                "output_features": output_features,
                "layers": layers,
                "dispatches_per_iteration": layers,
                "shared_bk": None,
                "mma_k": implementation.get("mma_k"),
                "k_phases_per_dispatch": implementation.get(
                    "k_phases_per_dispatch", 0
                ),
                "mma_operations_per_k_phase": implementation.get(
                    "mma_operations_per_k_phase", 0
                ),
                "barriers_per_dispatch": 0,
                "threadgroup_operand_scratch_bytes": 0,
                "threads_per_threadgroup": implementation[
                    "threads_per_threadgroup"
                ],
                "simd_groups_per_threadgroup": implementation[
                    "simd_groups_per_threadgroup"
                ],
                "outputs_per_simd_group": implementation[
                    "outputs_per_simd_group"
                ],
                "outputs_per_thread": implementation["outputs_per_thread"],
                "k_elements_per_lane": implementation["k_elements_per_lane"],
                "fp32_accumulators_per_thread": implementation[
                    "fp32_accumulators_per_thread"
                ],
                "simd_group_reductions_per_output": implementation[
                    "simd_group_reductions_per_output"
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
                "program_requested_traffic_bytes": requested_traffic_bytes(
                    specification, implementation
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
    expected_order: list[tuple[str, str]] = []
    for current_workload in workload_order:
        rows = int(WORKLOADS[current_workload]["rows"])
        control = control_for_rows(rows)
        implementation_ids = (
            [MMA_IMPLEMENTATION["id"], control["id"]]
            if implementation_order == "mma_then_control"
            else [control["id"], MMA_IMPLEMENTATION["id"]]
        )
        expected_order.extend(
            (current_workload, str(implementation_id))
            for implementation_id in implementation_ids
        )
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got "
            f"{observed_order}"
        )
    expected_counts: dict[tuple[str, str], int] = {}
    for current_workload in WORKLOAD_ORDER:
        rows = int(WORKLOADS[current_workload]["rows"])
        control = control_for_rows(rows)
        expected_counts[(current_workload, str(control["id"]))] = (
            EXPECTED_REPETITIONS
        )
        expected_counts[(current_workload, str(MMA_IMPLEMENTATION["id"]))] = (
            EXPECTED_REPETITIONS
        )
    if dict(counts) != expected_counts:
        raise ValueError(
            f"repetition count mismatch: expected {expected_counts}, got "
            f"{dict(counts)}"
        )
    if any(not sample["valid"] for sample in samples):
        raise ValueError("benchmark emitted a non-positive or non-finite sample")
    return identity, samples


def summarize_paired(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[
                (sample["workload"], sample["block_id"], sample["implementation"])
            ].append(float(sample["value"]))
    workloads = []
    for current_workload in WORKLOAD_ORDER:
        rows = int(WORKLOADS[current_workload]["rows"])
        control = control_for_rows(rows)
        blocks = []
        for block_number in range(1, 5):
            block_id = f"block-{block_number:02d}"
            control_values = grouped[
                (current_workload, block_id, str(control["id"]))
            ]
            mma_values = grouped[
                (current_workload, block_id, str(MMA_IMPLEMENTATION["id"]))
            ]
            if (
                len(control_values) != EXPECTED_REPETITIONS
                or len(mma_values) != EXPECTED_REPETITIONS
            ):
                raise ValueError(
                    "paired summary requires complete samples for "
                    f"{current_workload} {block_id}"
                )
            control_median = statistics.median(control_values)
            mma_median = statistics.median(mma_values)
            ratio = mma_median / control_median
            blocks.append(
                {
                    "block_id": block_id,
                    "control_implementation": control["id"],
                    "control_median_ms": control_median,
                    "mma_median_ms": mma_median,
                    "mma_to_control_ratio": ratio,
                    "mma_faster": ratio < 1.0,
                }
            )
        ratios = [float(block["mma_to_control_ratio"]) for block in blocks]
        classification, faster_blocks, slower_blocks = classify_ratios(ratios)
        median_ratio = statistics.median(ratios)
        workloads.append(
            {
                "workload": current_workload,
                "rows": rows,
                "control_implementation": control["id"],
                "median_mma_to_control_ratio": median_ratio,
                "median_change_percent": (median_ratio - 1.0) * 100.0,
                "mma_faster_blocks": faster_blocks,
                "mma_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": blocks,
            }
        )
    return {
        "ratio_definition": "MMA block median / phase-control block median",
        "workloads": workloads,
    }


def crossover_row(
    by_rows: dict[int, dict[str, Any]], *, require_improvement: bool
) -> int | None:
    allowed = (
        {"material_improvement"}
        if require_improvement
        else {"material_improvement", "inconclusive"}
    )
    for threshold in CROSSOVER_CANDIDATE_ROWS:
        larger_rows = [
            int(WORKLOADS[item]["rows"])
            for item in WORKLOAD_ORDER
            if int(WORKLOADS[item]["rows"]) >= threshold
        ]
        if all(by_rows[rows]["classification"] in allowed for rows in larger_rows):
            return threshold
    return None


def summarize(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    retained = list(samples)
    paired = summarize_paired(retained)
    by_rows = {int(item["rows"]): item for item in paired["workloads"]}
    prefill_qualifies = all(
        by_rows[rows]["classification"] == "material_improvement"
        for rows in LARGE_PREFILL_ROWS
    )
    prefill_rejected = all(
        by_rows[rows]["classification"] == "material_regression"
        for rows in LARGE_PREFILL_ROWS
    )
    decode_classification = str(by_rows[1]["classification"])
    if prefill_qualifies:
        timing_decision = "advance_mma_as_large_prefill_candidate"
    elif prefill_rejected:
        timing_decision = "reject_mma_for_large_prefill"
    else:
        timing_decision = "no_advance_mixed_or_inconclusive"
    return {
        "schema_version": 1,
        "implementations": [
            summarize_implementation(retained, implementation=implementation)
            for implementation in (
                ROWWISE_IMPLEMENTATION,
                REGISTER_IMPLEMENTATION,
                MMA_IMPLEMENTATION,
            )
        ],
        "paired_comparison": paired,
        "classification_rule": {
            "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
            "material_regression_ratio": MATERIAL_REGRESSION_RATIO,
            "required_direction_blocks": REQUIRED_DIRECTION_BLOCKS,
        },
        "large_prefill_rule": {
            "required_material_improvement_rows": list(LARGE_PREFILL_ROWS),
            "qualified": prefill_qualifies,
            "rejected_at_required_rows": prefill_rejected,
        },
        "decode": {
            "rows": 1,
            "control": ROWWISE_IMPLEMENTATION["id"],
            "classification": decode_classification,
            "disposition": (
                "reject_mma_for_batch_1_decode"
                if decode_classification == "material_regression"
                else "no_batch_1_decode_decision"
            ),
        },
        "crossover": {
            "candidate_rows": list(CROSSOVER_CANDIDATE_ROWS),
            "smallest_no_larger_material_regression_row": crossover_row(
                by_rows, require_improvement=False
            ),
            "smallest_all_larger_material_improvement_row": crossover_row(
                by_rows, require_improvement=True
            ),
        },
        "timing_decision": timing_decision,
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
    ]
    if BLOCK_ORDERS[block_number - 1] == "descending":
        args.extend(["-D", "LINEAR_MMA_DIAGNOSTIC_REVERSE=true"])
    if BLOCK_IMPLEMENTATION_ORDERS[block_number - 1] == "mma_then_control":
        args.extend(["-D", "LINEAR_MMA_DIAGNOSTIC_MMA_FIRST=true"])
    args.append("benchmarks/linear_mma_diagnostic.mojo")
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
                    ROWWISE_IMPLEMENTATION,
                    REGISTER_IMPLEMENTATION,
                    MMA_IMPLEMENTATION,
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
            "M=1..256 BF16 rotating packed-QKV projection with FP32 "
            "accumulation; Apple 8x8 MMA candidate versus the strongest "
            "previously measured control for each M regime"
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
