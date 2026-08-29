"""Run and record Apple GPU prefill linear-projection experiments."""

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
        BLOCK_IMPLEMENTATION_ORDERS,
        BLOCK_ORDERS,
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
    from run_linear import (
        BENCHMARK_RESULTS_BEGIN,
        BENCHMARK_RESULTS_END,
        BLOCK_IMPLEMENTATION_ORDERS,
        BLOCK_ORDERS,
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
WORKLOAD_ROWS = (1, 4, 8, 16, 32, 64, 128, 256)
WORKLOAD_SHAPES = (
    (128, 1, "single KV-width projection"),
    (896, 1, "single query-width projection"),
    (1_152, 1, "one hot packed-QKV projection"),
    (1_152, 24, "24-layer rotating packed-QKV cache-pressure proxy"),
)
EXPECTED_REPETITIONS = 10
BASELINE_IMPLEMENTATION = {
    "id": "apple_gpu_one_output_simdgroup_v0",
    "entrypoint": "enqueue_linear_apple_gpu",
}
DIRECT_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_direct_8x16_v0",
    "entrypoint": "enqueue_linear_prefill_direct_apple_gpu",
}
TILED_IMPLEMENTATION = {
    "id": "apple_gpu_prefill_tiled_8x16x32_v0",
    "entrypoint": "enqueue_linear_prefill_tiled_apple_gpu",
}
IMPLEMENTATIONS = {
    "linear_prefill_rowwise_apple_gpu": BASELINE_IMPLEMENTATION,
    "linear_prefill_direct_apple_gpu": DIRECT_IMPLEMENTATION,
    "linear_prefill_tiled_apple_gpu": TILED_IMPLEMENTATION,
}
BENCHMARK_NAME = re.compile(
    r"^(linear_prefill_(?:rowwise|direct|tiled)_apple_gpu)/input_id:"
    r"m(\d+)-k(\d+)-n(\d+)-layers(\d+)$"
)


def workload_id(rows: int, output_features: int, layers: int) -> str:
    return (
        f"m{rows}-k{INPUT_FEATURES}-n{output_features}-layers{layers}"
    )


WORKLOAD_ORDER = tuple(
    workload_id(rows, output_features, layers)
    for rows in WORKLOAD_ROWS
    for output_features, layers, _ in WORKLOAD_SHAPES
)
WORKLOADS = {
    workload_id(rows, output_features, layers): {
        "rows": rows,
        "input_features": INPUT_FEATURES,
        "output_features": output_features,
        "layers": layers,
        "semantic_role": semantic_role,
        "macs": layers * rows * output_features * INPUT_FEATURES,
        "allocated_footprint_bytes": 2
        * (
            rows * INPUT_FEATURES
            + layers * output_features * INPUT_FEATURES
            + output_features
            + rows * output_features
        ),
        "program_requested_traffic_bytes": 2
        * layers
        * (2 * rows * output_features * INPUT_FEATURES + 2 * rows * output_features),
    }
    for rows in WORKLOAD_ROWS
    for output_features, layers, semantic_role in WORKLOAD_SHAPES
}


def comparison_implementations(
    *, direct_comparison: bool, tiled_comparison: bool
) -> tuple[dict[str, str], dict[str, str]] | None:
    if direct_comparison and tiled_comparison:
        raise ValueError("comparison modes are mutually exclusive")
    if direct_comparison:
        return BASELINE_IMPLEMENTATION, DIRECT_IMPLEMENTATION
    if tiled_comparison:
        return DIRECT_IMPLEMENTATION, TILED_IMPLEMENTATION
    return None


def requested_traffic_bytes(
    specification: dict[str, Any], implementation: dict[str, str]
) -> int:
    if implementation != TILED_IMPLEMENTATION:
        return int(specification["program_requested_traffic_bytes"])
    rows = int(specification["rows"])
    input_features = int(specification["input_features"])
    output_features = int(specification["output_features"])
    layers = int(specification["layers"])
    input_loads = rows * math.ceil(output_features / 16) * input_features
    weight_loads = math.ceil(rows / 8) * output_features * input_features
    bias_loads_and_output_stores = 2 * rows * output_features
    return 2 * layers * (
        input_loads + weight_loads + bias_loads_and_output_stores
    )


def parse_identity(
    output: str,
    *,
    direct_comparison: bool = False,
    tiled_comparison: bool = False,
) -> dict[str, str]:
    implementation = re.search(
        r"^implementation:\s*(.+)$", output, re.MULTILINE
    )
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    comparison = re.search(
        r"^comparison implementation:\s*(.+)$", output, re.MULTILINE
    )
    if not implementation or not device or not api:
        raise ValueError("benchmark output omitted runtime identity")
    identity = {
        "implementation": implementation.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
    }
    comparison_pair = comparison_implementations(
        direct_comparison=direct_comparison,
        tiled_comparison=tiled_comparison,
    )
    primary = (
        comparison_pair[0] if comparison_pair else BASELINE_IMPLEMENTATION
    )
    if identity["implementation"] != primary["entrypoint"]:
        raise ValueError("benchmark identified an unexpected implementation")
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError(
            "benchmark did not prove an Apple device using the Metal API"
        )
    if comparison_pair:
        if comparison is None:
            raise ValueError("comparison benchmark omitted candidate identity")
        identity["comparison_implementation"] = comparison.group(1).strip()
        if (
            identity["comparison_implementation"]
            != comparison_pair[1]["entrypoint"]
        ):
            raise ValueError("benchmark identified an unexpected candidate")
    elif comparison is not None:
        raise ValueError("baseline benchmark emitted a candidate identity")
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
    implementation_order: str = "baseline_only",
    direct_comparison: bool = False,
    tiled_comparison: bool = False,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    comparison_pair = comparison_implementations(
        direct_comparison=direct_comparison,
        tiled_comparison=tiled_comparison,
    )
    if comparison_pair and implementation_order not in (
        "baseline_then_variant",
        "variant_then_baseline",
    ):
        raise ValueError("comparison benchmark has an invalid pair order")
    if comparison_pair is None and implementation_order != "baseline_only":
        raise ValueError("baseline benchmark has an invalid implementation order")
    identity = parse_identity(
        output,
        direct_comparison=direct_comparison,
        tiled_comparison=tiled_comparison,
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
            raise ValueError(f"unrecognized benchmark row: {row!r}")
        implementation = IMPLEMENTATIONS[match.group(1)]
        implementation_id = implementation["id"]
        rows = int(match.group(2))
        input_features = int(match.group(3))
        output_features = int(match.group(4))
        layers = int(match.group(5))
        current_workload = workload_id(rows, output_features, layers)
        if input_features != INPUT_FEATURES or current_workload not in WORKLOADS:
            raise ValueError(f"unexpected prefill workload {current_workload}")
        order_key = (current_workload, implementation_id)
        if not observed_order or observed_order[-1] != order_key:
            observed_order.append(order_key)
        expected_implementations = (
            comparison_pair if comparison_pair else (BASELINE_IMPLEMENTATION,)
        )
        if implementation not in expected_implementations:
            raise ValueError("baseline benchmark emitted a candidate row")

        value_text = row.get("met (ms)", "")
        iterations_text = row.get("iters", "")
        value = float(value_text)
        iterations = int(iterations_text)
        valid = math.isfinite(value) and value > 0 and iterations > 0
        counts[order_key] += 1
        repetition = counts[order_key]
        specification = WORKLOADS[current_workload]
        samples.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "block_id": block_id,
                "block_order": block_order,
                "sample_id": (
                    f"{run_id}-{block_id}-{implementation_id}-"
                    f"{current_workload}-rep{repetition:02d}"
                ),
                "implementation": implementation_id,
                "implementation_entrypoint": implementation["entrypoint"],
                "workload": current_workload,
                "semantic_role": specification["semantic_role"],
                "rows": rows,
                "input_features": input_features,
                "output_features": output_features,
                "layers": layers,
                "dispatches_per_iteration": layers,
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
                "threadgroup_operand_scratch_bytes": (
                    2 * (8 * 32 + 16 * 32)
                    if implementation == TILED_IMPLEMENTATION
                    else 0
                ),
                "barriers_per_dispatch": (
                    2 * math.ceil(input_features / 32)
                    if implementation == TILED_IMPLEMENTATION
                    else 0
                ),
            }
        )

    workload_order = list(
        WORKLOAD_ORDER
        if block_order == "ascending"
        else reversed(WORKLOAD_ORDER)
    )
    if comparison_pair:
        baseline_implementation, candidate_implementation = comparison_pair
        implementation_ids = (
            [candidate_implementation["id"], baseline_implementation["id"]]
            if implementation_order == "variant_then_baseline"
            else [baseline_implementation["id"], candidate_implementation["id"]]
        )
    else:
        implementation_ids = [BASELINE_IMPLEMENTATION["id"]]
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
    samples: Iterable[dict[str, Any]],
    *,
    implementation: dict[str, str],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[sample["workload"]].append(sample)

    workloads: list[dict[str, Any]] = []
    for current_workload in WORKLOAD_ORDER:
        group = grouped[current_workload]
        if not group:
            continue
        specification = WORKLOADS[current_workload]
        values = [float(sample["value"]) for sample in group]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        rows = int(specification["rows"])
        layers = int(specification["layers"])
        macs = int(specification["macs"])
        requested_byte_values = {
            int(sample["program_requested_traffic_bytes"]) for sample in group
        }
        if len(requested_byte_values) != 1:
            raise ValueError(
                f"inconsistent requested traffic for {current_workload}"
            )
        requested_bytes = requested_byte_values.pop()
        workloads.append(
            {
                "workload": current_workload,
                "semantic_role": specification["semantic_role"],
                "rows": rows,
                "input_features": specification["input_features"],
                "output_features": specification["output_features"],
                "layers": layers,
                "count": len(values),
                "median_ms_per_workload_iteration": median,
                "median_ms_per_layer": median / layers,
                "median_us_per_row_per_layer": median * 1_000.0 / (rows * layers),
                "median_absolute_deviation_ms": statistics.median(deviations),
                "p25_ms": percentile(values, 0.25),
                "p75_ms": percentile(values, 0.75),
                "min_ms": min(values),
                "max_ms": max(values),
                "workload_rows_per_second": rows * 1_000.0 / median,
                "layer_rows_per_second": layers * rows * 1_000.0 / median,
                "effective_gflop_per_second": 2 * macs / (median * 1_000_000.0),
                "program_requested_gb_per_second": requested_bytes
                / (median * 1_000_000.0),
                "traffic_note": (
                    "Requested bytes are source-derived logical loads and "
                    "are not observed hardware traffic."
                ),
            }
        )
    return {
        "schema_version": 1,
        "statistics": {
            "percentile_method": "linear interpolation at (n - 1) * p",
            "spread": "median absolute deviation and interquartile range",
        },
        "implementation": implementation,
        "workloads": workloads,
    }


def summarize_paired_workloads(
    samples: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[
                (
                    sample["workload"],
                    sample["block_id"],
                    sample["implementation"],
                )
            ].append(float(sample["value"]))

    workloads: list[dict[str, Any]] = []
    for current_workload in WORKLOAD_ORDER:
        blocks: list[dict[str, Any]] = []
        for block_number in range(1, 5):
            block_id = f"block-{block_number:02d}"
            baseline = grouped[
                (
                    current_workload,
                    block_id,
                    BASELINE_IMPLEMENTATION["id"],
                )
            ]
            direct = grouped[
                (
                    current_workload,
                    block_id,
                    DIRECT_IMPLEMENTATION["id"],
                )
            ]
            if (
                len(baseline) != EXPECTED_REPETITIONS
                or len(direct) != EXPECTED_REPETITIONS
            ):
                raise ValueError(
                    "paired summary requires complete baseline and direct "
                    f"samples for {current_workload} {block_id}"
                )
            baseline_median = statistics.median(baseline)
            direct_median = statistics.median(direct)
            ratio = direct_median / baseline_median
            blocks.append(
                {
                    "block_id": block_id,
                    "baseline_median_ms": baseline_median,
                    "direct_median_ms": direct_median,
                    "direct_to_baseline_ratio": ratio,
                    "direct_faster": ratio < 1.0,
                }
            )

        ratios = [float(block["direct_to_baseline_ratio"]) for block in blocks]
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
        specification = WORKLOADS[current_workload]
        workloads.append(
            {
                "workload": current_workload,
                "semantic_role": specification["semantic_role"],
                "rows": specification["rows"],
                "output_features": specification["output_features"],
                "layers": specification["layers"],
                "median_direct_to_baseline_ratio": median_ratio,
                "median_change_percent": (median_ratio - 1.0) * 100.0,
                "direct_faster_blocks": faster_blocks,
                "direct_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": blocks,
            }
        )
    return {
        "ratio_definition": "direct block median / rowwise block median",
        "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
        "material_regression_ratio": MATERIAL_REGRESSION_RATIO,
        "required_direction_blocks": REQUIRED_DIRECTION_BLOCKS,
        "workloads": workloads,
        "timing_decision": "control_only_no_dispatch_decision",
    }


def summarize_tiled_paired_workloads(
    samples: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[
                (
                    sample["workload"],
                    sample["block_id"],
                    sample["implementation"],
                )
            ].append(float(sample["value"]))

    workloads: list[dict[str, Any]] = []
    for current_workload in WORKLOAD_ORDER:
        blocks: list[dict[str, Any]] = []
        for block_number in range(1, 5):
            block_id = f"block-{block_number:02d}"
            direct = grouped[
                (
                    current_workload,
                    block_id,
                    DIRECT_IMPLEMENTATION["id"],
                )
            ]
            tiled = grouped[
                (
                    current_workload,
                    block_id,
                    TILED_IMPLEMENTATION["id"],
                )
            ]
            if (
                len(direct) != EXPECTED_REPETITIONS
                or len(tiled) != EXPECTED_REPETITIONS
            ):
                raise ValueError(
                    "paired summary requires complete direct and tiled "
                    f"samples for {current_workload} {block_id}"
                )
            direct_median = statistics.median(direct)
            tiled_median = statistics.median(tiled)
            ratio = tiled_median / direct_median
            blocks.append(
                {
                    "block_id": block_id,
                    "direct_median_ms": direct_median,
                    "tiled_median_ms": tiled_median,
                    "tiled_to_direct_ratio": ratio,
                    "tiled_faster": ratio < 1.0,
                }
            )

        ratios = [float(block["tiled_to_direct_ratio"]) for block in blocks]
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
        specification = WORKLOADS[current_workload]
        workloads.append(
            {
                "workload": current_workload,
                "semantic_role": specification["semantic_role"],
                "rows": specification["rows"],
                "output_features": specification["output_features"],
                "layers": specification["layers"],
                "median_tiled_to_direct_ratio": median_ratio,
                "median_change_percent": (median_ratio - 1.0) * 100.0,
                "tiled_faster_blocks": faster_blocks,
                "tiled_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": blocks,
            }
        )
    return {
        "ratio_definition": "tiled block median / direct block median",
        "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
        "material_regression_ratio": MATERIAL_REGRESSION_RATIO,
        "required_direction_blocks": REQUIRED_DIRECTION_BLOCKS,
        "workloads": workloads,
        "timing_decision": "shared_staging_control_only_no_dispatch_decision",
    }


def summarize(
    samples: Iterable[dict[str, Any]],
    *,
    direct_comparison: bool = False,
    tiled_comparison: bool = False,
) -> dict[str, Any]:
    retained = list(samples)
    comparison_pair = comparison_implementations(
        direct_comparison=direct_comparison,
        tiled_comparison=tiled_comparison,
    )
    if comparison_pair is None:
        return summarize_implementation(
            retained, implementation=BASELINE_IMPLEMENTATION
        )
    implementations = []
    for implementation in comparison_pair:
        implementation_samples = [
            sample
            for sample in retained
            if sample["implementation"] == implementation["id"]
        ]
        implementations.append(
            summarize_implementation(
                implementation_samples,
                implementation=implementation,
            )
        )
    return {
        "schema_version": 2,
        "implementations": implementations,
        "paired_comparison": (
            summarize_tiled_paired_workloads(retained)
            if tiled_comparison
            else summarize_paired_workloads(retained)
        ),
    }


def benchmark_command(
    *,
    reverse: bool,
    direct_comparison: bool = False,
    direct_first: bool = False,
    tiled_comparison: bool = False,
    tiled_first: bool = False,
) -> list[str]:
    comparison_implementations(
        direct_comparison=direct_comparison,
        tiled_comparison=tiled_comparison,
    )
    if direct_first and not direct_comparison:
        raise ValueError("direct-first order requires direct comparison")
    if tiled_first and not tiled_comparison:
        raise ValueError("tiled-first order requires tiled comparison")
    args = ["uv", "run", "--locked", "mojo", "run", "-I", "src"]
    if reverse:
        args.extend(["-D", "LINEAR_PREFILL_BENCH_REVERSE=true"])
    if direct_comparison:
        args.extend(
            ["-D", "LINEAR_PREFILL_BENCH_DIRECT_COMPARISON=true"]
        )
    if direct_first:
        args.extend(["-D", "LINEAR_PREFILL_BENCH_DIRECT_FIRST=true"])
    if tiled_comparison:
        args.extend(
            ["-D", "LINEAR_PREFILL_BENCH_TILED_COMPARISON=true"]
        )
    if tiled_first:
        args.extend(["-D", "LINEAR_PREFILL_BENCH_TILED_FIRST=true"])
    args.append("benchmarks/linear_prefill.mojo")
    return args


def run_block(
    *,
    experiment_id: str,
    run_id: str,
    block_number: int,
    block_order: str,
    implementation_order: str,
    direct_comparison: bool,
    tiled_comparison: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    block_id = f"block-{block_number:02d}"
    before = conditions_snapshot()
    command_args = benchmark_command(
        reverse=block_order == "descending",
        direct_comparison=direct_comparison,
        direct_first=(
            direct_comparison
            and implementation_order == "variant_then_baseline"
        ),
        tiled_comparison=tiled_comparison,
        tiled_first=(
            tiled_comparison
            and implementation_order == "variant_then_baseline"
        ),
    )
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    print(f"Running {block_id} ({block_order})...", flush=True)
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
        direct_comparison=direct_comparison,
        tiled_comparison=tiled_comparison,
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
    parser.add_argument("--direct-comparison", action="store_true")
    parser.add_argument("--tiled-comparison", action="store_true")
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
    if args.direct_comparison and args.tiled_comparison:
        raise RuntimeError("comparison modes are mutually exclusive")
    if (args.direct_comparison or args.tiled_comparison) and args.blocks != 4:
        raise RuntimeError("comparison requires all four ABBA blocks")

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
        block_order = BLOCK_ORDERS[index]
        implementation_order = (
            BLOCK_IMPLEMENTATION_ORDERS[index]
            if args.direct_comparison or args.tiled_comparison
            else "baseline_only"
        )
        block, block_samples, output = run_block(
            experiment_id=args.experiment_id,
            run_id=run_id,
            block_number=index + 1,
            block_order=block_order,
            implementation_order=implementation_order,
            direct_comparison=args.direct_comparison,
            tiled_comparison=args.tiled_comparison,
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

    result_summary = summarize(
        samples,
        direct_comparison=args.direct_comparison,
        tiled_comparison=args.tiled_comparison,
    )
    metadata = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_utc": utc_now(),
        "operation": "linear_projection",
        "scope": (
            "M=1..256 BF16 prefill projection with FP32 accumulation; "
            + (
                "direct 8x16 control versus shared 8x16x32 candidate"
                if args.tiled_comparison
                else (
                    "rowwise Apple GPU baseline versus direct 8x16 control"
                    if args.direct_comparison
                    else "rowwise Apple GPU baseline"
                )
            )
        ),
        "repository": initial_repository,
        "recorded": args.recorded,
        "direct_comparison": args.direct_comparison,
        "tiled_comparison": args.tiled_comparison,
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
