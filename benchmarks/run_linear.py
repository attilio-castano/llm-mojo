"""Run and record the Apple GPU M=1 projection benchmark."""

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
    from benchmarks.run_rms_norm import (
        conditions_snapshot,
        ensure_record_location,
        percentile,
        repository_state,
        require_ac,
        require_nominal_thermal_state,
        sha256_bytes,
        sha256_file,
        stable_environment,
        utc_now,
    )
except ModuleNotFoundError:
    from run_rms_norm import (  # type: ignore[no-redef]
        conditions_snapshot,
        ensure_record_location,
        percentile,
        repository_state,
        require_ac,
        require_nominal_thermal_state,
        sha256_bytes,
        sha256_file,
        stable_environment,
        utc_now,
    )


REPOSITORY = Path(__file__).resolve().parents[1]
ROWS = 1
INPUT_FEATURES = 896
QUERY_OUTPUT_FEATURES = 896
KV_OUTPUT_FEATURES = 128
QKV_OUTPUT_FEATURES = 1_152
RING_LAYERS = 24
EXPECTED_REPETITIONS = 10
BLOCK_ORDERS = ("ascending", "descending", "descending", "ascending")
BLOCK_IMPLEMENTATION_ORDERS = (
    "baseline_then_variant",
    "variant_then_baseline",
    "variant_then_baseline",
    "baseline_then_variant",
)
MATERIAL_IMPROVEMENT_RATIO = 0.95
MATERIAL_REGRESSION_RATIO = 1.05
REQUIRED_DIRECTION_BLOCKS = 3
PRIMARY_WORKLOAD = "qkv3-ring24-m1-k896-n1152-layers24"
SECONDARY_WORKLOADS = (
    "q-m1-k896-n896",
    "kv-m1-k896-n128",
    "qkv3-hot-m1-k896-n1152",
)
WORKLOAD_ORDER = (
    "kv-m1-k896-n128",
    "q-m1-k896-n896",
    "qkv3-hot-m1-k896-n1152",
    PRIMARY_WORKLOAD,
)
BENCHMARK_RESULTS_BEGIN = "BENCHMARK_RESULTS_BEGIN"
BENCHMARK_RESULTS_END = "BENCHMARK_RESULTS_END"
PROFILE_ITERATIONS_LIMIT = 5_000
DEFAULT_PROFILE_WARMUP_ITERATIONS = 100
PROFILE_WARMUP_ITERATIONS_LIMIT = 1_000
DEFAULT_PROFILE_POST_IDLE_MILLISECONDS = 0
PROFILE_POST_IDLE_MILLISECONDS_LIMIT = 1_000
PROFILE_WORKLOADS = {"q": 0, "kv": 1, "qkv-hot": 2, "qkv-ring24": 3}

BASELINE_IMPLEMENTATION = {
    "id": "apple_gpu_one_output_simdgroup_v0",
    "entrypoint": "enqueue_linear_apple_gpu",
}
VARIANT_IMPLEMENTATION = {
    "id": "apple_gpu_two_output_simdgroup_v1",
    "entrypoint": "enqueue_linear_apple_gpu_two_output",
}

BENCHMARK_NAMES = {
    "linear_decode_apple_gpu": BASELINE_IMPLEMENTATION,
    "linear_decode_qkv3_apple_gpu": BASELINE_IMPLEMENTATION,
    "linear_decode_qkv3_ring24_apple_gpu": BASELINE_IMPLEMENTATION,
    "linear_decode_apple_gpu_two_output": VARIANT_IMPLEMENTATION,
    "linear_decode_qkv3_apple_gpu_two_output": VARIANT_IMPLEMENTATION,
    "linear_decode_qkv3_ring24_apple_gpu_two_output": VARIANT_IMPLEMENTATION,
}
BENCHMARK_NAME = re.compile(
    r"^(linear_decode(?:_qkv3(?:_ring24)?)?_apple_gpu(?:_two_output)?)/"
    r"input_id:(.+)$"
)

WORKLOADS: dict[str, dict[str, int | str]] = {
    "kv-m1-k896-n128": {
        "semantic_role": "single KV projection",
        "output_features": KV_OUTPUT_FEATURES,
        "layers": 1,
        "dispatches": 1,
        "macs": KV_OUTPUT_FEATURES * INPUT_FEATURES,
        "allocated_bytes": 2
        * (
            INPUT_FEATURES
            + KV_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * KV_OUTPUT_FEATURES
        ),
        "program_requested_bytes": 2
        * (
            INPUT_FEATURES
            + KV_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * KV_OUTPUT_FEATURES
        ),
    },
    "q-m1-k896-n896": {
        "semantic_role": "single Q projection",
        "output_features": QUERY_OUTPUT_FEATURES,
        "layers": 1,
        "dispatches": 1,
        "macs": QUERY_OUTPUT_FEATURES * INPUT_FEATURES,
        "allocated_bytes": 2
        * (
            INPUT_FEATURES
            + QUERY_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * QUERY_OUTPUT_FEATURES
        ),
        "program_requested_bytes": 2
        * (
            INPUT_FEATURES
            + QUERY_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * QUERY_OUTPUT_FEATURES
        ),
    },
    "qkv3-hot-m1-k896-n1152": {
        "semantic_role": "one hot QKV layer",
        "output_features": QKV_OUTPUT_FEATURES,
        "layers": 1,
        "dispatches": 3,
        "macs": QKV_OUTPUT_FEATURES * INPUT_FEATURES,
        "allocated_bytes": 2
        * (
            INPUT_FEATURES
            + QKV_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * QKV_OUTPUT_FEATURES
        ),
        "program_requested_bytes": 2
        * (
            3 * INPUT_FEATURES
            + QKV_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * QKV_OUTPUT_FEATURES
        ),
    },
    PRIMARY_WORKLOAD: {
        "semantic_role": "24-layer rotating QKV cache-pressure proxy",
        "output_features": QKV_OUTPUT_FEATURES,
        "layers": RING_LAYERS,
        "dispatches": 3 * RING_LAYERS,
        "macs": RING_LAYERS * QKV_OUTPUT_FEATURES * INPUT_FEATURES,
        "allocated_bytes": 2
        * (
            INPUT_FEATURES
            + RING_LAYERS * QKV_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * QKV_OUTPUT_FEATURES
        ),
        "program_requested_bytes": 2
        * RING_LAYERS
        * (
            3 * INPUT_FEATURES
            + QKV_OUTPUT_FEATURES * INPUT_FEATURES
            + 2 * QKV_OUTPUT_FEATURES
        ),
    },
}


def parse_identity(
    output: str, *, variant_comparison: bool
) -> dict[str, str]:
    implementation = re.search(
        r"^implementation:\s*(.+)$", output, re.MULTILINE
    )
    comparison = re.search(
        r"^comparison implementation:\s*(.+)$", output, re.MULTILINE
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
    if not identity["device"].startswith("Apple ") or identity["api"] != "metal":
        raise ValueError("benchmark did not prove Apple GPU Metal execution")
    if identity["implementation"] != BASELINE_IMPLEMENTATION["entrypoint"]:
        raise ValueError("benchmark did not identify the frozen baseline")
    if variant_comparison:
        if comparison is None:
            raise ValueError("comparison benchmark omitted variant identity")
        identity["comparison_implementation"] = comparison.group(1).strip()
        if (
            identity["comparison_implementation"]
            != VARIANT_IMPLEMENTATION["entrypoint"]
        ):
            raise ValueError("benchmark did not identify the frozen candidate")
    elif comparison is not None:
        raise ValueError("baseline benchmark unexpectedly emitted a candidate")
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
    implementation_order: str,
    variant_comparison: bool,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    identity = parse_identity(output, variant_comparison=variant_comparison)
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
        if match is None or match.group(1) not in BENCHMARK_NAMES:
            raise ValueError(f"unrecognized benchmark row: {row!r}")
        implementation = BENCHMARK_NAMES[match.group(1)]
        implementation_id = str(implementation["id"])
        workload_id = match.group(2)
        if workload_id not in WORKLOADS:
            raise ValueError(f"unexpected projection workload: {workload_id}")
        if not variant_comparison and implementation != BASELINE_IMPLEMENTATION:
            raise ValueError("baseline run emitted a candidate sample")
        order_key = (workload_id, implementation_id)
        if not observed_order or observed_order[-1] != order_key:
            observed_order.append(order_key)
        value_text = row.get("met (ms)", "")
        iterations_text = row.get("iters", "")
        value = float(value_text)
        iterations = int(iterations_text)
        valid = math.isfinite(value) and value > 0 and iterations > 0
        counts[order_key] += 1
        repetition = counts[order_key]
        workload = WORKLOADS[workload_id]
        dispatches = int(workload["dispatches"])
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
                "rows": ROWS,
                "input_features": INPUT_FEATURES,
                "output_features": workload["output_features"],
                "layers": workload["layers"],
                "dispatches_per_iteration": dispatches,
                "repetition": repetition,
                "value": value,
                "source_value": value_text,
                "unit": "ms_per_workload_iteration",
                "ms_per_dispatch": value / dispatches,
                "iterations": iterations,
                "valid": valid,
                "macs_per_iteration": workload["macs"],
                "allocated_footprint_bytes": workload["allocated_bytes"],
                "program_requested_traffic_bytes": workload[
                    "program_requested_bytes"
                ],
            }
        )

    workload_order = list(
        WORKLOAD_ORDER
        if block_order == "ascending"
        else reversed(WORKLOAD_ORDER)
    )
    if variant_comparison:
        implementation_ids = (
            [VARIANT_IMPLEMENTATION["id"], BASELINE_IMPLEMENTATION["id"]]
            if implementation_order == "variant_then_baseline"
            else [BASELINE_IMPLEMENTATION["id"], VARIANT_IMPLEMENTATION["id"]]
        )
    else:
        if implementation_order != "baseline_only":
            raise ValueError("baseline run has an invalid implementation order")
        implementation_ids = [BASELINE_IMPLEMENTATION["id"]]
    expected_order = [
        (workload_id, str(implementation_id))
        for workload_id in workload_order
        for implementation_id in implementation_ids
    ]
    if observed_order != expected_order:
        raise ValueError(
            f"workload order mismatch: expected {expected_order}, got "
            f"{observed_order}"
        )
    expected_counts = {
        (workload_id, str(implementation_id)): EXPECTED_REPETITIONS
        for workload_id in WORKLOAD_ORDER
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
) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["valid"]:
            grouped[sample["workload"]].append(sample)
    result: list[dict[str, Any]] = []
    for workload_id in WORKLOAD_ORDER:
        group = grouped[workload_id]
        if not group:
            continue
        values = [float(sample["value"]) for sample in group]
        median = statistics.median(values)
        workload = WORKLOADS[workload_id]
        deviations = [abs(value - median) for value in values]
        result.append(
            {
                "workload": workload_id,
                "semantic_role": workload["semantic_role"],
                "count": len(values),
                "median_ms_per_workload_iteration": median,
                "median_ms_per_dispatch": median
                / int(workload["dispatches"]),
                "median_absolute_deviation_ms": statistics.median(deviations),
                "p25_ms": percentile(values, 0.25),
                "p75_ms": percentile(values, 0.75),
                "min_ms": min(values),
                "max_ms": max(values),
                "macs_per_second": int(workload["macs"]) * 1_000.0 / median,
                "program_requested_gb_per_second": int(
                    workload["program_requested_bytes"]
                )
                / (median * 1_000_000.0),
                "traffic_note": (
                    "Source-derived requested bytes, not measured hardware "
                    "traffic."
                ),
            }
        )
    return {"schema_version": 1, "workloads": result}


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
    for workload_id in WORKLOAD_ORDER:
        blocks: list[dict[str, Any]] = []
        for block_number in range(1, 5):
            block_id = f"block-{block_number:02d}"
            baseline = grouped[
                (workload_id, block_id, BASELINE_IMPLEMENTATION["id"])
            ]
            variant = grouped[
                (workload_id, block_id, VARIANT_IMPLEMENTATION["id"])
            ]
            if not baseline or not variant:
                raise ValueError(f"paired samples missing for {workload_id}")
            baseline_median = statistics.median(baseline)
            variant_median = statistics.median(variant)
            ratio = variant_median / baseline_median
            blocks.append(
                {
                    "block_id": block_id,
                    "baseline_median_ms": baseline_median,
                    "variant_median_ms": variant_median,
                    "variant_to_baseline_ratio": ratio,
                    "variant_faster": ratio < 1.0,
                }
            )
        ratios = [float(block["variant_to_baseline_ratio"]) for block in blocks]
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
                "median_variant_to_baseline_ratio": median_ratio,
                "median_change_percent": (median_ratio - 1.0) * 100.0,
                "variant_faster_blocks": faster_blocks,
                "variant_slower_blocks": slower_blocks,
                "classification": classification,
                "blocks": blocks,
            }
        )
    by_workload = {item["workload"]: item for item in workloads}
    primary_pass = (
        by_workload[PRIMARY_WORKLOAD]["classification"]
        == "material_improvement"
    )
    secondary_regressions = [
        workload_id
        for workload_id in SECONDARY_WORKLOADS
        if by_workload[workload_id]["classification"]
        == "material_regression"
    ]
    return {
        "ratio_definition": "candidate block median / baseline block median",
        "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
        "material_regression_ratio": MATERIAL_REGRESSION_RATIO,
        "required_direction_blocks": REQUIRED_DIRECTION_BLOCKS,
        "primary_workload": PRIMARY_WORKLOAD,
        "workloads": workloads,
        "primary_rule_passed": primary_pass,
        "secondary_material_regressions": secondary_regressions,
        "timing_decision": (
            "promote_two_output_m1"
            if primary_pass and not secondary_regressions
            else "retain_one_output_m1"
        ),
    }


def summarize(
    samples: Iterable[dict[str, Any]], *, variant_comparison: bool
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
        implementations.append(
            {
                "implementation": implementation["id"],
                "entrypoint": implementation["entrypoint"],
                **summarize_implementation(implementation_samples),
            }
        )
    return {
        "schema_version": 2,
        "implementations": implementations,
        "paired_comparison": summarize_paired_workloads(retained),
    }


def benchmark_command(
    *, reverse: bool, variant_comparison: bool, variant_first: bool
) -> list[str]:
    args = ["uv", "run", "--locked", "mojo", "run", "-I", "src"]
    if reverse:
        args.extend(["-D", "LINEAR_BENCH_REVERSE=true"])
    if variant_comparison:
        args.extend(["-D", "LINEAR_BENCH_VARIANT_COMPARISON=true"])
    if variant_first:
        args.extend(["-D", "LINEAR_BENCH_VARIANT_FIRST=true"])
    args.append("benchmarks/linear.mojo")
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
    command_args = benchmark_command(
        reverse=block_order == "descending",
        variant_comparison=variant_comparison,
        variant_first=implementation_order == "variant_then_baseline",
    )
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    print(
        f"Running {block_id} ({block_order}, {implementation_order})...",
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
    if args.profile_workload not in PROFILE_WORKLOADS:
        raise RuntimeError("profile workload is required")
    if not 0 < args.profile_iterations <= PROFILE_ITERATIONS_LIMIT:
        raise RuntimeError("profile iterations are outside the bounded range")
    warmup = (
        DEFAULT_PROFILE_WARMUP_ITERATIONS
        if args.profile_warmup_iterations is None
        else args.profile_warmup_iterations
    )
    if not 0 <= warmup <= PROFILE_WARMUP_ITERATIONS_LIMIT:
        raise RuntimeError("profile warmup iterations are outside the range")
    post_idle = (
        DEFAULT_PROFILE_POST_IDLE_MILLISECONDS
        if args.profile_post_idle_milliseconds is None
        else args.profile_post_idle_milliseconds
    )
    if not 0 <= post_idle <= PROFILE_POST_IDLE_MILLISECONDS_LIMIT:
        raise RuntimeError("profile post-idle milliseconds are outside the range")
    output = args.profile_binary.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite profile binary: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = repository_state()
    if args.require_clean and state["dirty"]:
        raise RuntimeError("profile build requires a clean repository")
    if args.require_clean:
        ensure_record_location(output)
    implementation = (
        VARIANT_IMPLEMENTATION
        if args.profile_implementation == "two-output"
        else BASELINE_IMPLEMENTATION
    )
    command_args = [
        "uv",
        "run",
        "--locked",
        "mojo",
        "build",
        "-I",
        "src",
        "-D",
        f"LINEAR_PROFILE_WORKLOAD={PROFILE_WORKLOADS[args.profile_workload]}",
        "-D",
        f"LINEAR_PROFILE_WARMUP_ITERATIONS={warmup}",
        "-D",
        f"LINEAR_PROFILE_ITERATIONS={args.profile_iterations}",
        "-D",
        f"LINEAR_PROFILE_POST_IDLE_MILLISECONDS={post_idle}",
    ]
    if args.profile_implementation == "two-output":
        command_args.extend(["-D", "LINEAR_PROFILE_TWO_OUTPUT=true"])
    command_args.extend(["benchmarks/linear.mojo", "-o", str(output)])
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    subprocess.run(command_args, cwd=REPOSITORY, check=True, env=environment)
    recorded_command = command_args.copy()
    recorded_command[-1] = "<external-profile-binary>"
    workload = WORKLOADS[
        {
            "q": "q-m1-k896-n896",
            "kv": "kv-m1-k896-n128",
            "qkv-hot": "qkv3-hot-m1-k896-n1152",
            "qkv-ring24": PRIMARY_WORKLOAD,
        }[args.profile_workload]
    ]
    provenance = {
        "schema_version": 1,
        "operation": "linear_projection",
        "created_utc": utc_now(),
        "repository": state,
        "command": shlex.join(recorded_command),
        "environment": {"MODULAR_DEBUG": "unset"},
        "profile_workload": args.profile_workload,
        "rows": ROWS,
        "input_features": INPUT_FEATURES,
        "output_features": workload["output_features"],
        "layers": workload["layers"],
        "dispatches_per_iteration": workload["dispatches"],
        "profile_warmup_iterations": warmup,
        "profile_iterations": args.profile_iterations,
        "profile_post_idle_milliseconds": post_idle,
        "implementation": implementation["id"],
        "entrypoint": implementation["entrypoint"],
        "binary": {
            "path_note": "External local artifact; path intentionally omitted.",
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
    parser.add_argument("--profile-workload", choices=tuple(PROFILE_WORKLOADS))
    parser.add_argument(
        "--profile-implementation",
        choices=("baseline", "two-output"),
        default="baseline",
    )
    parser.add_argument(
        "--profile-iterations", type=int, default=PROFILE_ITERATIONS_LIMIT
    )
    parser.add_argument("--profile-warmup-iterations", type=int)
    parser.add_argument("--profile-post-idle-milliseconds", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile_binary is not None:
        build_profile_binary(args)
        return
    if args.profile_workload is not None:
        raise RuntimeError("--profile-workload requires --profile-binary")
    if args.profile_post_idle_milliseconds is not None:
        raise RuntimeError(
            "--profile-post-idle-milliseconds requires --profile-binary"
        )
    if args.recorded and (
        args.output_dir is None
        or args.experiment_id == "exploration"
        or args.run_id is None
    ):
        raise RuntimeError(
            "--recorded requires an external output directory and explicit IDs"
        )
    if args.variant_comparison and args.blocks != 4:
        raise RuntimeError("candidate comparison requires all four ABBA blocks")

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
            if args.variant_comparison
            else "baseline_only"
        )
        block, block_samples, output = run_block(
            experiment_id=args.experiment_id,
            run_id=run_id,
            block_number=index + 1,
            block_order=block_order,
            implementation_order=implementation_order,
            variant_comparison=args.variant_comparison,
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
        samples, variant_comparison=args.variant_comparison
    )
    metadata = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "created_utc": utc_now(),
        "operation": "linear_projection",
        "scope": "M=1 BF16 decode projection with FP32 accumulation",
        "repository": initial_repository,
        "recorded": args.recorded,
        "variant_comparison": args.variant_comparison,
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
