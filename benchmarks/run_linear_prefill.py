"""Run and record the Apple GPU prefill linear-projection baseline."""

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
        BLOCK_ORDERS,
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
        BLOCK_ORDERS,
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
IMPLEMENTATION = {
    "id": "apple_gpu_one_output_simdgroup_v0",
    "entrypoint": "enqueue_linear_apple_gpu",
}
BENCHMARK_NAME = re.compile(
    r"^linear_prefill_rowwise_apple_gpu/input_id:"
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


def parse_identity(output: str) -> dict[str, str]:
    implementation = re.search(
        r"^implementation:\s*(.+)$", output, re.MULTILINE
    )
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    if not implementation or not device or not api:
        raise ValueError("benchmark output omitted runtime identity")
    identity = {
        "implementation": implementation.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
    }
    if identity["implementation"] != IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark identified an unexpected implementation")
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError(
            "benchmark did not prove an Apple device using the Metal API"
        )
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
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    identity = parse_identity(output)
    reader = csv.DictReader(table_lines(output), skipinitialspace=True)
    counts: defaultdict[str, int] = defaultdict(int)
    observed_order: list[str] = []
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
        rows = int(match.group(1))
        input_features = int(match.group(2))
        output_features = int(match.group(3))
        layers = int(match.group(4))
        current_workload = workload_id(rows, output_features, layers)
        if input_features != INPUT_FEATURES or current_workload not in WORKLOADS:
            raise ValueError(f"unexpected prefill workload {current_workload}")
        if not observed_order or observed_order[-1] != current_workload:
            observed_order.append(current_workload)

        value_text = row.get("met (ms)", "")
        iterations_text = row.get("iters", "")
        value = float(value_text)
        iterations = int(iterations_text)
        valid = math.isfinite(value) and value > 0 and iterations > 0
        counts[current_workload] += 1
        repetition = counts[current_workload]
        specification = WORKLOADS[current_workload]
        samples.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "block_id": block_id,
                "block_order": block_order,
                "sample_id": (
                    f"{run_id}-{block_id}-{IMPLEMENTATION['id']}-"
                    f"{current_workload}-rep{repetition:02d}"
                ),
                "implementation": IMPLEMENTATION["id"],
                "implementation_entrypoint": IMPLEMENTATION["entrypoint"],
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
                "program_requested_traffic_bytes": specification[
                    "program_requested_traffic_bytes"
                ],
            }
        )

    expected_order = list(
        WORKLOAD_ORDER
        if block_order == "ascending"
        else reversed(WORKLOAD_ORDER)
    )
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got "
            f"{observed_order}"
        )
    expected_counts = {
        current_workload: EXPECTED_REPETITIONS
        for current_workload in WORKLOAD_ORDER
    }
    if dict(counts) != expected_counts:
        raise ValueError(
            f"repetition count mismatch: expected {expected_counts}, got "
            f"{dict(counts)}"
        )
    if any(not sample["valid"] for sample in samples):
        raise ValueError("benchmark emitted a non-positive or non-finite sample")
    return identity, samples


def summarize(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
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
        requested_bytes = int(specification["program_requested_traffic_bytes"])
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
                    "Requested bytes are source-derived from the rowwise "
                    "kernel and are not observed hardware traffic."
                ),
            }
        )
    return {
        "schema_version": 1,
        "statistics": {
            "percentile_method": "linear interpolation at (n - 1) * p",
            "spread": "median absolute deviation and interquartile range",
        },
        "implementation": IMPLEMENTATION,
        "workloads": workloads,
    }


def benchmark_command(*, reverse: bool) -> list[str]:
    args = ["uv", "run", "--locked", "mojo", "run", "-I", "src"]
    if reverse:
        args.extend(["-D", "LINEAR_PREFILL_BENCH_REVERSE=true"])
    args.append("benchmarks/linear_prefill.mojo")
    return args


def run_block(
    *,
    experiment_id: str,
    run_id: str,
    block_number: int,
    block_order: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    block_id = f"block-{block_number:02d}"
    before = conditions_snapshot()
    command_args = benchmark_command(reverse=block_order == "descending")
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
    )
    stdout_bytes = result.stdout.encode()
    block = {
        "block_id": block_id,
        "order": block_order,
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
        block_order = BLOCK_ORDERS[index]
        block, block_samples, output = run_block(
            experiment_id=args.experiment_id,
            run_id=run_id,
            block_number=index + 1,
            block_order=block_order,
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

    result_summary = summarize(samples)
    metadata = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_utc": utc_now(),
        "operation": "linear_projection",
        "scope": (
            "M=1..256 BF16 prefill projection with FP32 accumulation; "
            "rowwise Apple GPU baseline"
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
