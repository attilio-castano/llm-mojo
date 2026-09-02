"""Run and optionally record the materialized Apple GPU GQA baseline."""

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
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
QUERY_HEADS = 14
KEY_VALUE_HEADS = 2
HEAD_DIM = 64
DISPATCHES_PER_ITERATION = 3
EXPECTED_REPETITIONS = 10
BLOCK_ORDERS = ("ascending", "descending", "descending", "ascending")
BENCHMARK_RESULTS_BEGIN = "BENCHMARK_RESULTS_BEGIN"
BENCHMARK_RESULTS_END = "BENCHMARK_RESULTS_END"
IMPLEMENTATION = {
    "id": "apple_gpu_materialized_three_stage_v0",
    "entrypoint": "enqueue_grouped_query_attention_apple_gpu",
    "benchmark_name": "grouped_query_attention_apple_gpu_materialized",
}
WORKLOAD_ORDER = (
    "decode-r1-t1-qh14-kvh2-d64",
    "decode-r1-t16-qh14-kvh2-d64",
    "decode-r1-t64-qh14-kvh2-d64",
    "decode-r1-t256-qh14-kvh2-d64",
    "decode-r1-t1024-qh14-kvh2-d64",
    "decode-r1-t4096-qh14-kvh2-d64",
    "incremental-prefill-r4-t128-qh14-kvh2-d64",
    "incremental-prefill-r16-t512-qh14-kvh2-d64",
    "incremental-prefill-r16-t4096-qh14-kvh2-d64",
    "full-prefill-r4-t4-qh14-kvh2-d64",
    "full-prefill-r32-t32-qh14-kvh2-d64",
    "full-prefill-r128-t128-qh14-kvh2-d64",
    "full-prefill-r256-t256-qh14-kvh2-d64",
)
BENCHMARK_NAME = re.compile(
    r"^grouped_query_attention_apple_gpu_materialized/input_id:"
    r"((decode|incremental-prefill|full-prefill)-r(\d+)-t(\d+)"
    r"-qh(\d+)-kvh(\d+)-d(\d+))$"
)


def workload_metrics(query_rows: int, key_value_rows: int) -> dict[str, int]:
    if query_rows <= 0 or query_rows > key_value_rows:
        raise ValueError("invalid attention workload dimensions")
    visible_positions_per_head = query_rows * (2 * key_value_rows - query_rows + 1) // 2
    visible_scores = QUERY_HEADS * visible_positions_per_head
    materialized_scores = query_rows * QUERY_HEADS * key_value_rows
    output_elements = query_rows * QUERY_HEADS * HEAD_DIM
    query_elements = output_elements
    key_value_elements = key_value_rows * KEY_VALUE_HEADS * HEAD_DIM
    allocated_elements = (
        query_elements + 2 * key_value_elements + materialized_scores + output_elements
    )
    qk_macs = visible_scores * HEAD_DIM
    probability_value_macs = visible_scores * HEAD_DIM
    program_requested_bytes = (
        8 * visible_scores * HEAD_DIM
        + 6 * visible_scores
        + 4 * materialized_scores
        + 2 * output_elements
    )
    return {
        "visible_scores": visible_scores,
        "materialized_scores": materialized_scores,
        "output_elements": output_elements,
        "qk_macs": qk_macs,
        "probability_value_macs": probability_value_macs,
        "total_macs": qk_macs + probability_value_macs,
        "scratch_bytes": 2 * materialized_scores,
        "allocated_footprint_bytes": 2 * allocated_elements,
        "program_requested_traffic_bytes": program_requested_bytes,
    }


def build_workloads() -> dict[str, dict[str, int | str]]:
    workloads: dict[str, dict[str, int | str]] = {}
    for workload_id in WORKLOAD_ORDER:
        match = BENCHMARK_NAME.fullmatch(
            IMPLEMENTATION["benchmark_name"] + "/input_id:" + workload_id
        )
        if match is None:
            raise ValueError(f"invalid configured workload {workload_id!r}")
        regime = match.group(2)
        query_rows = int(match.group(3))
        key_value_rows = int(match.group(4))
        metrics = workload_metrics(query_rows, key_value_rows)
        workloads[workload_id] = {
            "regime": regime,
            "query_rows": query_rows,
            "key_value_rows": key_value_rows,
            **metrics,
        }
    return workloads


WORKLOADS = build_workloads()


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
                    "connection": display.get("spdisplays_connection_type", "unknown"),
                    "resolution": display.get("_spdisplays_resolution", "unknown"),
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
        "estimate": (charge_match.group(3).strip() if charge_match else "unknown"),
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
        "branch": command("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def stable_environment() -> dict[str, Any]:
    xcode = optional_command("xcodebuild", "-version").splitlines()
    metal = optional_command("xcrun", "metal", "--version").splitlines()
    return {
        "hardware": {
            "model": optional_command("sysctl", "-n", "hw.model"),
            "chip": optional_command("sysctl", "-n", "machdep.cpu.brand_string"),
            "logical_cpu_count": optional_command("sysctl", "-n", "hw.ncpu"),
            "gpu_api": optional_command("uv", "run", "--locked", "gpu-query", "--api"),
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
        raise RuntimeError(f"recorded run requires AC Power; observed {source!r}")


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
            "compact reviewed evidence afterward"
        )


def parse_identity(output: str) -> dict[str, str]:
    implementation = re.search(r"^implementation:\s*(.+)$", output, re.MULTILINE)
    device = re.search(r"^device:\s*(.+)$", output, re.MULTILINE)
    api = re.search(r"^api:\s*(.+)$", output, re.MULTILINE)
    if implementation is None or device is None or api is None:
        raise ValueError("benchmark output omitted runtime identity")
    identity = {
        "implementation": implementation.group(1).strip(),
        "device": device.group(1).strip(),
        "api": api.group(1).strip(),
    }
    if identity["implementation"] != IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the frozen baseline")
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError("benchmark did not prove Apple GPU Metal execution")
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
    if block_order not in ("ascending", "descending"):
        raise ValueError("invalid attention workload order")
    identity = parse_identity(output)
    reader = csv.DictReader(table_lines(output), skipinitialspace=True)
    repetition_by_workload: defaultdict[str, int] = defaultdict(int)
    observed_order: list[str] = []
    samples: list[dict[str, Any]] = []

    for raw_row in reader:
        row = {
            str(key).strip(): str(value).strip()
            for key, value in raw_row.items()
            if key is not None and value is not None
        }
        match = BENCHMARK_NAME.fullmatch(row.get("name", ""))
        if match is None:
            raise ValueError(f"unrecognized benchmark row: {row!r}")
        workload_id = match.group(1)
        if workload_id not in WORKLOADS:
            raise ValueError(f"unexpected attention workload: {workload_id}")
        workload = WORKLOADS[workload_id]
        if (
            int(match.group(3)) != workload["query_rows"]
            or int(match.group(4)) != workload["key_value_rows"]
            or int(match.group(5)) != QUERY_HEADS
            or int(match.group(6)) != KEY_VALUE_HEADS
            or int(match.group(7)) != HEAD_DIM
        ):
            raise ValueError("attention benchmark row disagrees with its workload")
        if not observed_order or observed_order[-1] != workload_id:
            observed_order.append(workload_id)

        value_text = row.get("met (ms)", "")
        iterations_text = row.get("iters", "")
        value = float(value_text)
        iterations = int(iterations_text)
        valid = math.isfinite(value) and value > 0.0 and iterations > 0
        repetition_by_workload[workload_id] += 1
        repetition = repetition_by_workload[workload_id]
        samples.append(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "block_id": block_id,
                "block_order": block_order,
                "sample_id": (f"{run_id}-{block_id}-{workload_id}-rep{repetition:02d}"),
                "implementation": IMPLEMENTATION["id"],
                "implementation_entrypoint": IMPLEMENTATION["entrypoint"],
                "workload": workload_id,
                "regime": workload["regime"],
                "query_rows": workload["query_rows"],
                "key_value_rows": workload["key_value_rows"],
                "query_heads": QUERY_HEADS,
                "key_value_heads": KEY_VALUE_HEADS,
                "head_dim": HEAD_DIM,
                "dispatches_per_iteration": DISPATCHES_PER_ITERATION,
                "repetition": repetition,
                "value": value,
                "source_value": value_text,
                "unit": "ms_per_attention_call",
                "iterations": iterations,
                "valid": valid,
                **{
                    key: metric
                    for key, metric in workload.items()
                    if key not in ("regime", "query_rows", "key_value_rows")
                },
            }
        )

    expected_order = list(
        WORKLOAD_ORDER if block_order == "ascending" else reversed(WORKLOAD_ORDER)
    )
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got {observed_order}"
        )
    expected_counts = {
        workload_id: EXPECTED_REPETITIONS for workload_id in WORKLOAD_ORDER
    }
    if dict(repetition_by_workload) != expected_counts:
        raise ValueError(
            f"repetition count mismatch: expected {expected_counts}, got "
            f"{dict(repetition_by_workload)}"
        )
    if any(not sample["valid"] for sample in samples):
        raise ValueError("benchmark emitted a non-finite or non-positive sample")
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


def summarize(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[str(sample["workload"])].append(sample)

    workloads: list[dict[str, Any]] = []
    for workload_id in WORKLOAD_ORDER:
        group = grouped[workload_id]
        values = [float(sample["value"]) for sample in group]
        if not values:
            continue
        workload = WORKLOADS[workload_id]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        block_values: defaultdict[str, list[float]] = defaultdict(list)
        for sample in group:
            block_values[str(sample["block_id"])].append(float(sample["value"]))
        block_summaries: list[dict[str, Any]] = []
        for block_id, values_in_block in block_values.items():
            midpoint = len(values_in_block) // 2
            first_median = statistics.median(values_in_block[:midpoint])
            second_median = statistics.median(values_in_block[midpoint:])
            block_summaries.append(
                {
                    "block_id": block_id,
                    "median_ms_per_attention_call": statistics.median(values_in_block),
                    "first_half_median_ms": first_median,
                    "second_half_median_ms": second_median,
                    "half_shift_fraction": (
                        (second_median - first_median) / first_median
                    ),
                }
            )
        total_macs = int(workload["total_macs"])
        requested_bytes = int(workload["program_requested_traffic_bytes"])
        output_elements = int(workload["output_elements"])
        workloads.append(
            {
                "workload": workload_id,
                "regime": workload["regime"],
                "query_rows": workload["query_rows"],
                "key_value_rows": workload["key_value_rows"],
                "query_heads": QUERY_HEADS,
                "key_value_heads": KEY_VALUE_HEADS,
                "head_dim": HEAD_DIM,
                "dispatches_per_attention_call": DISPATCHES_PER_ITERATION,
                "count": len(values),
                "median_ms_per_attention_call": median,
                "median_absolute_deviation_ms": statistics.median(deviations),
                "p25_ms": percentile(values, 0.25),
                "p75_ms": percentile(values, 0.75),
                "min_ms": min(values),
                "max_ms": max(values),
                "output_elements_per_second": output_elements * 1000.0 / median,
                "effective_gmac_per_second": total_macs / (median * 1_000_000.0),
                "program_requested_gb_per_second": requested_bytes
                / (median * 1_000_000.0),
                "visible_scores": workload["visible_scores"],
                "materialized_scores": workload["materialized_scores"],
                "total_macs": total_macs,
                "scratch_bytes": workload["scratch_bytes"],
                "allocated_footprint_bytes": workload["allocated_footprint_bytes"],
                "program_requested_traffic_bytes": requested_bytes,
                "traffic_note": (
                    "Requested bytes are derived from source-level tensor "
                    "accesses and are not observed hardware traffic."
                ),
                "blocks": block_summaries,
            }
        )
    return {
        "schema_version": 1,
        "implementation": IMPLEMENTATION,
        "statistics": {
            "primary_metric": "synchronized milliseconds per attention call",
            "percentile_method": "linear interpolation at (n - 1) * p",
            "spread": "median absolute deviation and interquartile range",
        },
        "scope": (
            "One hot materialized attention operation; no 24-layer cache "
            "pressure or end-to-end model claim."
        ),
        "workloads": workloads,
    }


def benchmark_command(*, reverse: bool) -> list[str]:
    args = ["uv", "run", "--locked", "mojo", "run", "-I", "src"]
    if reverse:
        args.extend(["-D", "ATTENTION_BENCH_REVERSE=true"])
    args.append("benchmarks/attention.mojo")
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
            "and explicit experiment and run IDs"
        )

    run_id = args.run_id or datetime.now(UTC).strftime("exploration-%Y%m%dT%H%M%SZ")
    initial_repository = repository_state()
    initial_conditions = conditions_snapshot()
    require_clean = args.recorded or args.require_clean
    if require_clean and initial_repository["dirty"]:
        raise RuntimeError("recorded run requires a clean repository")
    if args.output_dir is not None:
        ensure_record_location(args.output_dir)
        if args.output_dir.exists():
            raise RuntimeError("refusing to overwrite an existing run directory")
    if args.recorded:
        require_ac(initial_conditions)
        require_nominal_thermal_state(initial_conditions)

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
        "operation": "grouped_query_attention",
        "scope": ("BF16 Qwen-shaped materialized three-stage Apple GPU baseline"),
        "repository": initial_repository,
        "recorded": args.recorded,
        "blocks": blocks,
        "sample_count": len(samples),
        "environment": stable_environment(),
    }
    if args.output_dir is not None:
        write_run_artifacts(args.output_dir, metadata, samples, result_summary, outputs)
        print(f"artifacts: {args.output_dir.resolve()}")
    print(json.dumps(result_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
