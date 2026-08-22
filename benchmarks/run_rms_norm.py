"""Run and record the Apple GPU RMSNorm benchmark or build a profile binary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


REPOSITORY = Path(__file__).resolve().parents[1]
HIDDEN_SIZE = 896
WORKLOAD_ROWS = (1, 4, 16, 128, 512, 2048, 4096)
BLOCK_ORDERS = ("ascending", "descending", "descending", "ascending")
BLOCK_IMPLEMENTATION_ORDERS = (
    "baseline_then_variant",
    "variant_then_baseline",
    "variant_then_baseline",
    "baseline_then_variant",
)
EXPECTED_REPETITIONS = 10
PROFILE_ITERATIONS_LIMIT = 5_000
MATERIAL_IMPROVEMENT_RATIO = 0.95
MATERIAL_REGRESSION_RATIO = 1.05
REQUIRED_DIRECTION_BLOCKS = 3
RELEVANT_PRODUCTION_ROWS = (1, 128, 512, 2048, 4096)
BENCHMARK_RESULTS_BEGIN = "BENCHMARK_RESULTS_BEGIN"
BENCHMARK_RESULTS_END = "BENCHMARK_RESULTS_END"
BENCHMARK_NAME = re.compile(
    r"^(rms_norm_apple_gpu(?:_simdgroup)?)/input_id:"
    r"rows=(\d+) hidden=(\d+)$"
)
IMPLEMENTATIONS = {
    "rms_norm_apple_gpu": {
        "id": "apple_gpu_shared_tree_v0",
        "entrypoint": "enqueue_rms_norm_apple_gpu_shared_tree",
    },
    "rms_norm_apple_gpu_simdgroup": {
        "id": "apple_gpu_simdgroup_v1",
        "entrypoint": "enqueue_rms_norm_apple_gpu",
    },
}
BASELINE_IMPLEMENTATION = IMPLEMENTATIONS["rms_norm_apple_gpu"]
VARIANT_IMPLEMENTATION = IMPLEMENTATIONS["rms_norm_apple_gpu_simdgroup"]


def command(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=REPOSITORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def optional_command(*args: str) -> str:
    try:
        return command(*args)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def displays() -> list[dict[str, str]]:
    raw = optional_command("system_profiler", "SPDisplaysDataType", "-json")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [{"status": raw}]

    result: list[dict[str, str]] = []
    for gpu in payload.get("SPDisplaysDataType", []):
        for display in gpu.get("spdisplays_ndrvs", []):
            result.append(
                {
                    "name": display.get("_name", "unknown"),
                    "connection": display.get(
                        "spdisplays_connection_type", "unknown"
                    ),
                    "resolution": display.get(
                        "_spdisplays_resolution", "unknown"
                    ),
                    "online": display.get("spdisplays_online", "unknown"),
                }
            )
    return result


def battery() -> dict[str, str]:
    raw = optional_command("pmset", "-g", "batt")
    source_match = re.search(r"Now drawing from '([^']+)'", raw)
    charge_match = re.search(r"(\d+)%;\s*([^;]+);\s*([^\n]+)", raw)
    return {
        "power_source": source_match.group(1) if source_match else "unknown",
        "charge": charge_match.group(1) + "%" if charge_match else "unknown",
        "state": charge_match.group(2).strip() if charge_match else "unknown",
        "estimate": charge_match.group(
            3
        ).strip() if charge_match else "unknown",
    }


def power_mode() -> str:
    raw = optional_command("pmset", "-g")
    match = re.search(r"^\s*powermode\s+(\d+)\s*$", raw, re.MULTILINE)
    return match.group(1) if match else "unavailable"


def memory() -> dict[str, Any]:
    raw = optional_command("memory_pressure", "-Q")
    total_match = re.search(r"system has (\d+)", raw)
    free_match = re.search(r"free percentage:\s*(\d+)%", raw)
    return {
        "physical_bytes": int(total_match.group(1)) if total_match else None,
        "free_percent": int(free_match.group(1)) if free_match else None,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def repository_state() -> dict[str, Any]:
    status = command("git", "status", "--porcelain")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "dirty": bool(status),
    }


def stable_environment() -> dict[str, Any]:
    xcode = optional_command("xcodebuild", "-version").splitlines()
    metal = optional_command("xcrun", "metal", "--version").splitlines()
    return {
        "hardware": {
            "model": optional_command("sysctl", "-n", "hw.model"),
            "chip": optional_command(
                "sysctl", "-n", "machdep.cpu.brand_string"
            ),
            "logical_cpu_count": optional_command("sysctl", "-n", "hw.ncpu"),
            "gpu_api": optional_command(
                "uv", "run", "--locked", "gpu-query", "--api"
            ),
            "gpu_target": optional_command(
                "uv",
                "run",
                "--locked",
                "gpu-query",
                "--target-accelerator",
            ),
            "physical_memory_bytes": memory()["physical_bytes"],
        },
        "software": {
            "macos": optional_command("sw_vers", "-productVersion"),
            "macos_build": optional_command("sw_vers", "-buildVersion"),
            "darwin": optional_command("uname", "-r"),
            "xcode": xcode,
            "metal": metal[:3],
            "uv": optional_command("uv", "--version"),
            "mojo": package_version("mojo"),
            "max": package_version("max"),
        },
    }


def conditions_snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "battery": battery(),
        "power_mode_raw": power_mode(),
        "thermal": optional_command("pmset", "-g", "therm").splitlines(),
        "memory": memory(),
        "displays": displays(),
    }


def require_ac(conditions: dict[str, Any]) -> None:
    source = conditions["battery"]["power_source"]
    if source != "AC Power":
        raise RuntimeError(
            f"recorded run requires AC Power; observed {source!r}"
        )


def require_nominal_thermal_state(conditions: dict[str, Any]) -> None:
    thermal = "\n".join(conditions["thermal"])
    required_lines = (
        "No thermal warning level has been recorded",
        "No performance warning level has been recorded",
    )
    if not all(line in thermal for line in required_lines):
        raise RuntimeError(
            "recorded run requires no thermal or performance warning; "
            f"observed {conditions['thermal']!r}"
        )


def ensure_record_location(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == REPOSITORY or resolved.is_relative_to(REPOSITORY):
        raise RuntimeError(
            "recorded run output must be outside the repository; promote only "
            "the compact reviewed evidence afterward"
        )


def parse_identity(
    output: str, *, variant_comparison: bool
) -> dict[str, str]:
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    implementation = re.search(
        r"^implementation:\s*(.+)$", output, re.MULTILINE
    )
    comparison_implementation = re.search(
        r"^comparison implementation:\s*(.+)$", output, re.MULTILINE
    )
    if not device or not api or not implementation:
        raise ValueError("benchmark output omitted runtime identity")
    result = {
        "implementation": implementation.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
    }
    if not result["device"].startswith("Apple ") or result["api"] != "metal":
        raise ValueError(
            "benchmark did not prove an Apple device using the Metal API"
        )
    if result["implementation"] != BASELINE_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the frozen baseline")
    if variant_comparison:
        if comparison_implementation is None:
            raise ValueError("comparison benchmark omitted variant identity")
        result["comparison_implementation"] = (
            comparison_implementation.group(1).strip()
        )
        if (
            result["comparison_implementation"]
            != VARIANT_IMPLEMENTATION["entrypoint"]
        ):
            raise ValueError("benchmark did not identify the frozen variant")
    elif comparison_implementation is not None:
        raise ValueError("baseline-only benchmark emitted a variant identity")
    return result


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
    implementation_order: str,
    variant_comparison: bool,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if variant_comparison and implementation_order not in (
        "baseline_then_variant",
        "variant_then_baseline",
    ):
        raise ValueError("comparison benchmark has an invalid pair order")
    if not variant_comparison and implementation_order != "baseline_only":
        raise ValueError("baseline-only benchmark has an invalid pair order")
    identity = parse_identity(output, variant_comparison=variant_comparison)
    reader = csv.DictReader(table_lines(output), skipinitialspace=True)
    repetition_by_key: defaultdict[tuple[str, int], int] = defaultdict(int)
    observed_order: list[tuple[int, str]] = []
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
        benchmark_name = match.group(1)
        implementation = IMPLEMENTATIONS[benchmark_name]
        implementation_id = implementation["id"]
        rows = int(match.group(2))
        hidden_size = int(match.group(3))
        if rows not in WORKLOAD_ROWS or hidden_size != HIDDEN_SIZE:
            raise ValueError(
                f"unexpected RMSNorm workload rows={rows}, hidden={hidden_size}"
            )
        if not variant_comparison and implementation_id != (
            BASELINE_IMPLEMENTATION["id"]
        ):
            raise ValueError("baseline-only benchmark emitted a variant row")
        order_key = (rows, implementation_id)
        if not observed_order or observed_order[-1] != order_key:
            observed_order.append(order_key)

        value_text = row.get("met (ms)", "")
        iterations_text = row.get("iters", "")
        value = float(value_text)
        iterations = int(iterations_text)
        valid = math.isfinite(value) and value > 0 and iterations > 0
        repetition_by_key[(implementation_id, rows)] += 1
        repetition = repetition_by_key[(implementation_id, rows)]
        workload_id = f"r{rows}-h{hidden_size}"
        samples.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "block_id": block_id,
                "block_order": block_order,
                "block_implementation_order": implementation_order,
                "sample_id": (
                    f"{run_id}-{block_id}-{implementation_id}-"
                    f"{workload_id}-rep{repetition:02d}"
                ),
                "implementation": implementation_id,
                "implementation_entrypoint": implementation["entrypoint"],
                "workload": workload_id,
                "rows": rows,
                "hidden_size": hidden_size,
                "repetition": repetition,
                "value": value,
                "source_value": value_text,
                "unit": "ms_per_dispatch",
                "iterations": iterations,
                "valid": valid,
                "logical_elements": rows * hidden_size,
                "allocated_footprint_bytes": hidden_size * (4 * rows + 2),
                "logical_tensor_traffic_bytes": rows * hidden_size * 6,
                "program_requested_traffic_bytes": rows * hidden_size * 8,
            }
        )

    row_order = list(
        WORKLOAD_ROWS if block_order == "ascending" else reversed(WORKLOAD_ROWS)
    )
    if variant_comparison:
        implementation_ids = (
            [VARIANT_IMPLEMENTATION["id"], BASELINE_IMPLEMENTATION["id"]]
            if implementation_order == "variant_then_baseline"
            else [BASELINE_IMPLEMENTATION["id"], VARIANT_IMPLEMENTATION["id"]]
        )
    else:
        implementation_ids = [BASELINE_IMPLEMENTATION["id"]]
    expected_order = [
        (rows, implementation_id)
        for rows in row_order
        for implementation_id in implementation_ids
    ]
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got "
            f"{observed_order}"
        )
    expected_counts = {
        (implementation_id, rows): EXPECTED_REPETITIONS
        for implementation_id in implementation_ids
        for rows in WORKLOAD_ROWS
    }
    if dict(repetition_by_key) != expected_counts:
        raise ValueError(
            f"repetition count mismatch: expected {expected_counts}, got "
            f"{dict(repetition_by_key)}"
        )
    if any(not sample["valid"] for sample in samples):
        raise ValueError(
            "benchmark emitted a non-finite or non-positive sample"
        )
    return identity, samples


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_implementation(
    samples: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[sample["workload"]].append(sample)

    workloads: list[dict[str, Any]] = []
    for rows in WORKLOAD_ROWS:
        workload_id = f"r{rows}-h{HIDDEN_SIZE}"
        group = grouped[workload_id]
        values = [float(sample["value"]) for sample in group]
        if not values:
            continue
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        block_values: defaultdict[str, list[float]] = defaultdict(list)
        for sample in group:
            block_values[sample["block_id"]].append(float(sample["value"]))
        block_summaries = []
        for block_id, values_in_block in block_values.items():
            first_half = values_in_block[: len(values_in_block) // 2]
            second_half = values_in_block[len(values_in_block) // 2 :]
            first_median = statistics.median(first_half)
            second_median = statistics.median(second_half)
            block_summaries.append(
                {
                    "block_id": block_id,
                    "median_ms_per_dispatch": statistics.median(
                        values_in_block
                    ),
                    "first_half_median_ms": first_median,
                    "second_half_median_ms": second_median,
                    "half_shift_fraction": (
                        (second_median - first_median) / first_median
                    ),
                }
            )
        logical_bytes = rows * HIDDEN_SIZE * 6
        requested_bytes = rows * HIDDEN_SIZE * 8
        workloads.append(
            {
                "workload": workload_id,
                "rows": rows,
                "hidden_size": HIDDEN_SIZE,
                "count": len(values),
                "median_ms_per_dispatch": median,
                "median_us_per_row": median * 1000.0 / rows,
                "median_absolute_deviation_ms": statistics.median(deviations),
                "p25_ms": percentile(values, 0.25),
                "p75_ms": percentile(values, 0.75),
                "min_ms": min(values),
                "max_ms": max(values),
                "logical_elements_per_second": (
                    rows * HIDDEN_SIZE * 1000.0 / median
                ),
                "logical_tensor_gb_per_second": logical_bytes
                / (median * 1_000_000.0),
                "program_requested_gb_per_second": requested_bytes
                / (median * 1_000_000.0),
                "traffic_note": (
                    "Both byte rates are source-derived and are not observed "
                    "hardware traffic."
                ),
                "blocks": block_summaries,
            }
        )
    return {
        "schema_version": 1,
        "statistics": {
            "percentile_method": "linear interpolation at (n - 1) * p",
            "spread": "median absolute deviation and interquartile range",
        },
        "workloads": workloads,
    }


def summarize_paired_workloads(
    samples: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[
        tuple[str, str, str], list[float]
    ] = defaultdict(list)
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
    for rows in WORKLOAD_ROWS:
        workload_id = f"r{rows}-h{HIDDEN_SIZE}"
        block_ids = sorted(
            {
                block_id
                for workload, block_id, _ in grouped
                if workload == workload_id
            }
        )
        block_ratios: list[dict[str, Any]] = []
        for block_id in block_ids:
            baseline_values = grouped[
                (
                    workload_id,
                    block_id,
                    BASELINE_IMPLEMENTATION["id"],
                )
            ]
            variant_values = grouped[
                (
                    workload_id,
                    block_id,
                    VARIANT_IMPLEMENTATION["id"],
                )
            ]
            if (
                len(baseline_values) != EXPECTED_REPETITIONS
                or len(variant_values) != EXPECTED_REPETITIONS
            ):
                raise ValueError(
                    f"paired summary requires {EXPECTED_REPETITIONS} samples "
                    f"per implementation for {workload_id} {block_id}"
                )
            baseline_median = statistics.median(baseline_values)
            variant_median = statistics.median(variant_values)
            ratio = variant_median / baseline_median
            block_ratios.append(
                {
                    "block_id": block_id,
                    "baseline_median_ms_per_dispatch": baseline_median,
                    "variant_median_ms_per_dispatch": variant_median,
                    "variant_to_baseline_ratio": ratio,
                    "variant_faster": ratio < 1.0,
                }
            )

        if len(block_ratios) != len(BLOCK_ORDERS):
            raise ValueError(
                f"paired summary requires {len(BLOCK_ORDERS)} blocks for "
                f"{workload_id}; observed {len(block_ratios)}"
            )
        ratios = [
            float(block["variant_to_baseline_ratio"])
            for block in block_ratios
        ]
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
        workloads.append(
            {
                "workload": workload_id,
                "rows": rows,
                "hidden_size": HIDDEN_SIZE,
                "median_variant_to_baseline_ratio": median_ratio,
                "median_change_percent": (median_ratio - 1.0) * 100.0,
                "variant_faster_blocks": faster_blocks,
                "variant_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": block_ratios,
            }
        )

    relevant = [
        workload
        for workload in workloads
        if workload["rows"] in RELEVANT_PRODUCTION_ROWS
    ]
    relevant_improvements = [
        workload["workload"]
        for workload in relevant
        if workload["classification"] == "material_improvement"
    ]
    relevant_regressions = [
        workload["workload"]
        for workload in relevant
        if workload["classification"] == "material_regression"
    ]
    timing_decision = (
        "replace_baseline"
        if relevant_improvements and not relevant_regressions
        else "retain_baseline"
    )
    return {
        "ratio_definition": "variant_block_median / baseline_block_median",
        "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
        "material_regression_ratio": MATERIAL_REGRESSION_RATIO,
        "required_direction_blocks": REQUIRED_DIRECTION_BLOCKS,
        "relevant_production_rows": list(RELEVANT_PRODUCTION_ROWS),
        "workloads": workloads,
        "relevant_material_improvements": relevant_improvements,
        "relevant_material_regressions": relevant_regressions,
        "timing_decision": timing_decision,
    }


def summarize(
    samples: Iterable[dict[str, Any]], *, variant_comparison: bool = False
) -> dict[str, Any]:
    retained = list(samples)
    if not variant_comparison:
        return summarize_implementation(retained)

    implementations = []
    for implementation in (BASELINE_IMPLEMENTATION, VARIANT_IMPLEMENTATION):
        implementation_samples = [
            sample
            for sample in retained
            if sample["implementation"] == implementation["id"]
        ]
        implementation_summary = summarize_implementation(
            implementation_samples
        )
        implementations.append(
            {
                "implementation": implementation["id"],
                "entrypoint": implementation["entrypoint"],
                "workloads": implementation_summary["workloads"],
            }
        )
    return {
        "schema_version": 2,
        "statistics": {
            "implementation_summaries": (
                "median, median absolute deviation, and interquartile range"
            ),
            "paired_statistic": (
                "median of four within-block variant/baseline median ratios"
            ),
        },
        "implementations": implementations,
        "paired_comparison": summarize_paired_workloads(retained),
    }


def benchmark_command(
    *, reverse: bool, variant_comparison: bool, variant_first: bool
) -> list[str]:
    args = [
        "uv",
        "run",
        "--locked",
        "mojo",
        "run",
        "-I",
        "src",
    ]
    if reverse:
        args.extend(["-D", "RMS_NORM_BENCH_REVERSE=true"])
    if variant_comparison:
        args.extend(["-D", "RMS_NORM_BENCH_VARIANT_COMPARISON=true"])
    if variant_first:
        args.extend(["-D", "RMS_NORM_BENCH_VARIANT_FIRST=true"])
    args.append("benchmarks/rms_norm.mojo")
    return args


def run_block(
    *,
    experiment_id: str,
    run_id: str,
    block_number: int,
    block_order: str,
    implementation_order: str,
    variant_comparison: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    block_id = f"block-{block_number:02d}"
    before = conditions_snapshot()
    variant_first = implementation_order == "variant_then_baseline"
    command_args = benchmark_command(
        reverse=block_order == "descending",
        variant_comparison=variant_comparison,
        variant_first=variant_comparison and variant_first,
    )
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    print(
        f"Running {block_id} ({block_order}, {implementation_order}) "
        "with MODULAR_DEBUG unset...",
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
        variant_comparison=variant_comparison,
    )
    after = conditions_snapshot()
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
        "conditions_after": after,
        "sample_count": len(samples),
        "stdout_bytes": len(stdout_bytes),
        "stdout_sha256": sha256_bytes(stdout_bytes),
    }
    return block, samples, result.stdout


def write_run_artifacts(
    output_dir: Path,
    metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    summary: dict[str, Any],
    block_outputs: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, output in enumerate(block_outputs, start=1):
        (output_dir / f"block-{index:02d}.stdout.txt").write_text(output)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "samples.jsonl").write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def build_profile_binary(args: argparse.Namespace) -> None:
    if args.profile_rows not in WORKLOAD_ROWS:
        raise RuntimeError(
            f"profile rows must be one of {', '.join(map(str, WORKLOAD_ROWS))}"
        )
    if args.profile_iterations <= 0:
        raise RuntimeError("profile iterations must be positive")
    if args.profile_iterations > PROFILE_ITERATIONS_LIMIT:
        raise RuntimeError(
            "profile iterations must not exceed "
            f"{PROFILE_ITERATIONS_LIMIT}; use multiple short captures instead"
        )
    output = args.profile_binary.resolve()
    if output.exists():
        raise RuntimeError(
            f"refusing to overwrite existing profile binary: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    state = repository_state()
    if args.require_clean and state["dirty"]:
        raise RuntimeError("profile build requires a clean repository")
    if args.require_clean:
        ensure_record_location(output)
    command_args = [
        "uv",
        "run",
        "--locked",
        "mojo",
        "build",
        "-I",
        "src",
        "-D",
        f"RMS_NORM_PROFILE_ROWS={args.profile_rows}",
        "-D",
        f"RMS_NORM_PROFILE_ITERATIONS={args.profile_iterations}",
    ]
    profile_implementation = (
        VARIANT_IMPLEMENTATION
        if args.profile_implementation == "simdgroup"
        else BASELINE_IMPLEMENTATION
    )
    if args.profile_implementation == "simdgroup":
        command_args.extend(["-D", "RMS_NORM_PROFILE_SIMDGROUP=true"])
    command_args.extend(["benchmarks/rms_norm.mojo", "-o", str(output)])
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    subprocess.run(command_args, cwd=REPOSITORY, check=True, env=environment)
    recorded_command = command_args.copy()
    recorded_command[-1] = "<external-profile-binary>"
    provenance = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "repository": state,
        "command": shlex.join(recorded_command),
        "environment": {"MODULAR_DEBUG": "unset"},
        "profile_rows": args.profile_rows,
        "hidden_size": HIDDEN_SIZE,
        "profile_iterations": args.profile_iterations,
        "implementation": profile_implementation["id"],
        "entrypoint": profile_implementation["entrypoint"],
        "binary": {
            "path_note": "External local artifact; path is intentionally omitted.",
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        **stable_environment(),
    }
    provenance_path = output.with_name(output.name + ".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"profile binary: {output}")
    print(f"provenance: {provenance_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="exploration")
    parser.add_argument("--run-id")
    parser.add_argument("--blocks", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recorded", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--variant-comparison", action="store_true")
    parser.add_argument("--profile-binary", type=Path)
    parser.add_argument("--profile-rows", type=int)
    parser.add_argument(
        "--profile-implementation",
        choices=("baseline", "simdgroup"),
        default="baseline",
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=PROFILE_ITERATIONS_LIMIT,
        help=(
            "profile dispatch count "
            f"(default and maximum: {PROFILE_ITERATIONS_LIMIT})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile_binary is not None:
        if args.profile_rows is None:
            raise RuntimeError("--profile-binary requires --profile-rows")
        build_profile_binary(args)
        return
    if args.profile_rows is not None:
        raise RuntimeError("--profile-rows requires --profile-binary")
    if args.profile_implementation != "baseline":
        raise RuntimeError(
            "--profile-implementation requires --profile-binary"
        )
    if args.recorded and (
        args.output_dir is None
        or args.experiment_id == "exploration"
        or args.run_id is None
    ):
        raise RuntimeError(
            "--recorded requires --output-dir, an explicit --experiment-id, "
            "and an explicit --run-id"
        )
    if args.variant_comparison and args.blocks != 4:
        raise RuntimeError(
            "variant comparison requires all four paired blocks"
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
    if args.output_dir is not None:
        if require_clean:
            ensure_record_location(args.output_dir)
        if args.output_dir.exists():
            raise RuntimeError(
                f"refusing to overwrite existing output directory: "
                f"{args.output_dir}"
            )

    environment = stable_environment()
    metadata: dict[str, Any] = {
        "schema_version": 3 if args.variant_comparison else 2,
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "recorded": args.recorded,
        "started_utc": utc_now(),
        "repository": initial_repository,
        **environment,
        "conditions_at_start": initial_conditions,
        "benchmark": {
            "implementations": (
                [BASELINE_IMPLEMENTATION, VARIANT_IMPLEMENTATION]
                if args.variant_comparison
                else [BASELINE_IMPLEMENTATION]
            ),
            "variant_comparison": args.variant_comparison,
            "workload_rows": list(WORKLOAD_ROWS),
            "hidden_size": HIDDEN_SIZE,
            "block_orders": list(BLOCK_ORDERS[: args.blocks]),
            "block_implementation_orders": list(
                BLOCK_IMPLEMENTATION_ORDERS[: args.blocks]
                if args.variant_comparison
                else ("baseline_only",) * args.blocks
            ),
            "num_warmup_iters": 1000,
            "max_iters": 1000,
            "num_repetitions": EXPECTED_REPETITIONS,
            "primary_metric": "synchronized_device_execution_ms_per_dispatch",
            "MODULAR_DEBUG": "unset",
        },
        "blocks": [],
    }
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)

    samples: list[dict[str, Any]] = []
    block_outputs: list[str] = []
    for block_number, block_order in enumerate(
        BLOCK_ORDERS[: args.blocks], start=1
    ):
        implementation_order = (
            BLOCK_IMPLEMENTATION_ORDERS[block_number - 1]
            if args.variant_comparison
            else "baseline_only"
        )
        current_repository = repository_state()
        if current_repository["commit"] != initial_repository["commit"]:
            raise RuntimeError("repository commit changed during the run")
        if require_clean and current_repository["dirty"]:
            raise RuntimeError("repository became dirty during the run")
        conditions = conditions_snapshot()
        if args.recorded:
            require_ac(conditions)
            require_nominal_thermal_state(conditions)
        block, block_samples, output = run_block(
            experiment_id=args.experiment_id,
            run_id=run_id,
            block_number=block_number,
            block_order=block_order,
            implementation_order=implementation_order,
            variant_comparison=args.variant_comparison,
        )
        if args.recorded:
            require_ac(block["conditions_after"])
            require_nominal_thermal_state(block["conditions_after"])
        metadata["blocks"].append(block)
        samples.extend(block_samples)
        block_outputs.append(output)

    metadata["completed_utc"] = utc_now()
    metadata["conditions_at_end"] = conditions_snapshot()
    if args.recorded:
        require_ac(metadata["conditions_at_end"])
        require_nominal_thermal_state(metadata["conditions_at_end"])
    summary = summarize(samples, variant_comparison=args.variant_comparison)
    implementation_count = 2 if args.variant_comparison else 1
    expected_samples = (
        args.blocks
        * len(WORKLOAD_ROWS)
        * EXPECTED_REPETITIONS
        * implementation_count
    )
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} samples, collected {len(samples)}"
        )
    if args.output_dir is not None:
        write_run_artifacts(
            args.output_dir, metadata, samples, summary, block_outputs
        )
        print(f"recorded run artifacts: {args.output_dir.resolve()}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
