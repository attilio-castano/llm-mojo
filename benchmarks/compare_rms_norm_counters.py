"""Compare four scrubbed RMSNorm counter summaries in ABBA order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


EXPECTED_TEMPLATE = "LLM_Mojo_Metal_Limiters"
EXPECTED_GPU_SETTINGS = {
    "Counter Set: Performance Limiters",
    "Shader Timeline: Disabled",
    "Induced GPU Performance State: Default",
}
DEFAULT_WARMUP_ITERATIONS = 100
DEFAULT_PROFILE_ITERATIONS = 500
DEFAULT_MINIMUM_TIMESTAMPS = 10
DEFAULT_MINIMUM_SAMPLE_SPAN = 0.80
DEFAULT_MATERIAL_CHANGE = 0.05
EXPECTED_CAPTURE_ORDER = ("baseline", "variant", "variant", "baseline")
EXPECTED_WORKLOAD = {
    "rows": 512,
    "hidden_size": 896,
    "warmup_iterations": DEFAULT_WARMUP_ITERATIONS,
    "profile_iterations": DEFAULT_PROFILE_ITERATIONS,
    "post_profile_idle_milliseconds": 250,
}
IMPLEMENTATION_ROLES = {
    (
        "apple_gpu_shared_tree_v0",
        "enqueue_rms_norm_apple_gpu_shared_tree",
    ): "baseline",
    (
        "apple_gpu_simdgroup_v1",
        "enqueue_rms_norm_apple_gpu",
    ): "variant",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, capture_id: str) -> dict[str, Any]:
    return {
        "capture_id": capture_id,
        "kind": "scrubbed_trace_summary_json",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "path_note": "External local artifact; path intentionally omitted.",
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity_hash(value: Any, label: str, capture_label: str) -> dict[str, Any]:
    require(
        isinstance(value, dict),
        f"{capture_label} has no {label} identity",
    )
    size = value.get("bytes")
    digest = value.get("sha256")
    require(
        isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{capture_label} has an invalid {label} identity",
    )
    return {"bytes": size, "sha256": digest}


def verified_identity(
    summary: dict[str, Any], capture_index: int
) -> tuple[dict[str, Any], str, str]:
    capture_label = f"capture {capture_index}"
    require(
        summary.get("schema_version") == 3,
        f"{capture_label} does not use the verified trace-summary schema",
    )
    identity = summary.get("capture_identity")
    require(
        isinstance(identity, dict)
        and identity.get("verification")
        == "capture_receipt_v2_and_trace_target",
        f"{capture_label} has no verified capture identity",
    )
    capture_id = identity.get("capture_id")
    require(
        isinstance(capture_id, str)
        and capture_id.startswith("rmsnorm-")
        and len(capture_id) == 40
        and all(
            character in "0123456789abcdef" for character in capture_id[8:]
        ),
        f"{capture_label} has an invalid capture ID",
    )
    capture_label = f"capture {capture_index} ({capture_id})"

    implementation = identity.get("implementation")
    entrypoint = identity.get("entrypoint")
    role = IMPLEMENTATION_ROLES.get((implementation, entrypoint))
    require(
        role is not None,
        f"{capture_label} has an unknown RMSNorm implementation",
    )
    binary = identity_hash(identity.get("binary"), "binary", capture_label)
    provenance = identity_hash(
        identity.get("provenance"), "provenance", capture_label
    )
    receipt = identity_hash(
        identity.get("capture_receipt"), "capture receipt", capture_label
    )
    inputs = summary.get("inputs")
    require(
        isinstance(inputs, list),
        f"{capture_label} has no analyzer input identities",
    )
    receipt_inputs = [
        value
        for value in inputs
        if isinstance(value, dict)
        and value.get("kind") == "capture_receipt_json"
    ]
    require(
        len(receipt_inputs) == 1
        and identity_hash(
            receipt_inputs[0], "analyzer capture receipt", capture_label
        )
        == receipt,
        f"{capture_label} is not bound to its analyzer receipt input",
    )

    repository = identity.get("repository")
    require(
        isinstance(repository, dict)
        and isinstance(repository.get("commit"), str)
        and len(repository["commit"]) == 40
        and all(
            character in "0123456789abcdef"
            for character in repository["commit"]
        )
        and repository.get("dirty") is False,
        f"{capture_label} was not built from a clean immutable commit",
    )
    runtime = identity.get("runtime")
    require(
        isinstance(runtime, dict)
        and isinstance(runtime.get("device"), str)
        and runtime["device"].startswith("Apple ")
        and runtime.get("backend") == "metal",
        f"{capture_label} did not verify an Apple GPU through Metal",
    )
    workload = identity.get("workload")
    require(
        workload == EXPECTED_WORKLOAD,
        f"{capture_label} does not match the frozen RMSNorm workload",
    )
    trace = summary.get("trace")
    require(isinstance(trace, dict), f"{capture_label} has no trace metadata")
    trace_target = trace.get("target")
    require(
        isinstance(trace_target, dict)
        and trace_target.get("capture_id_verified") is True,
        f"{capture_label} is not bound to the trace target process",
    )
    return (
        {
            "capture_id": capture_id,
            "role": role,
            "implementation": implementation,
            "entrypoint": entrypoint,
            "binary": binary,
            "provenance": provenance,
            "capture_receipt": receipt,
            "repository": repository,
            "runtime": runtime,
            "workload": workload,
        },
        capture_id,
        role,
    )


def validate_capture(
    summary: dict[str, Any],
    *,
    capture_index: int,
    minimum_timestamps: int,
    minimum_sample_span: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    identity, capture_id, role = verified_identity(summary, capture_index)
    require(
        summary.get("analysis") == "rmsnorm_metal_trace",
        f"{capture_id} is not an RMSNorm Metal trace summary",
    )
    trace = summary.get("trace", {})
    require(
        trace.get("end_reason") == "Target app exited",
        f"{capture_id} target did not exit normally",
    )
    require(
        trace.get("template") == EXPECTED_TEMPLATE,
        f"{capture_id} used an unexpected Instruments template",
    )
    settings = set(trace.get("metal_application_gpu_settings", []))
    require(
        EXPECTED_GPU_SETTINGS <= settings,
        f"{capture_id} does not have the frozen Metal settings",
    )

    sequence = summary.get("validated_sequence", {})
    expected_compute = (
        1
        + EXPECTED_WORKLOAD["warmup_iterations"]
        + EXPECTED_WORKLOAD["profile_iterations"]
    )
    require(
        sequence.get("setup_compute_commands") == 0,
        f"{capture_id} contains unexpected setup compute commands",
    )
    require(
        sequence.get("correctness_dispatches") == 1
        and sequence.get("warmup_dispatches")
        == EXPECTED_WORKLOAD["warmup_iterations"]
        and sequence.get("profile_dispatches")
        == EXPECTED_WORKLOAD["profile_iterations"]
        and sequence.get("compute_channel_command_kinds", {}).get("compute")
        == expected_compute,
        f"{capture_id} does not contain the exact declared compute sequence",
    )

    profile_counters = summary.get("profile_gpu_counters", {})
    samples = profile_counters.get("samples", {})
    timestamp_count = samples.get("timestamp_count", 0)
    sample_span = samples.get("sample_span_fraction", 0.0)
    require(
        timestamp_count >= minimum_timestamps,
        f"{capture_id} has too few profile-window counter timestamps",
    )
    require(
        math.isfinite(sample_span) and sample_span >= minimum_sample_span,
        f"{capture_id} counter samples do not span enough of the profile",
    )
    defined_count = profile_counters.get("defined_counter_count", 0)
    sampled_count = profile_counters.get("sampled_counter_count", 0)
    require(
        defined_count > 0 and sampled_count == defined_count,
        f"{capture_id} does not sample every named counter",
    )

    counters_by_name: dict[str, dict[str, Any]] = {}
    for counter in profile_counters.get("counters", []):
        name = counter.get("name")
        median = counter.get("median")
        require(
            isinstance(name, str) and name,
            f"{capture_id} contains an unnamed counter",
        )
        require(
            name not in counters_by_name,
            f"{capture_id} contains duplicate counter {name!r}",
        )
        require(
            isinstance(median, (int, float)) and math.isfinite(median),
            f"{capture_id} counter {name!r} has a non-finite median",
        )
        counters_by_name[name] = counter
    require(
        len(counters_by_name) == sampled_count,
        f"{capture_id} counter metadata is incomplete",
    )

    profile_window = profile_counters.get("profile_window", {})
    return (
        {
            **identity,
            "trace_duration_seconds": trace.get(
                "recording_duration_seconds"
            ),
            "profile_window_duration_nanoseconds": profile_window.get(
                "duration_nanoseconds"
            ),
            "target_gpu_busy_fraction": profile_window.get(
                "target_gpu_busy_fraction"
            ),
            "counter_timestamp_count": timestamp_count,
            "counter_sample_span_fraction": sample_span,
            "gpu_performance_state": summary.get("gpu_performance_state"),
            "target_spill_event_count": summary.get(
                "compiler_spills", {}
            ).get("target_event_count"),
            "instrumented_profile_duration": summary.get(
                "instrumented_gpu_interval_duration", {}
            ).get("profile"),
        },
        counters_by_name,
    )


def compare(
    summaries: list[dict[str, Any]],
    *,
    minimum_timestamps: int = DEFAULT_MINIMUM_TIMESTAMPS,
    minimum_sample_span: float = DEFAULT_MINIMUM_SAMPLE_SPAN,
    material_change: float = DEFAULT_MATERIAL_CHANGE,
) -> dict[str, Any]:
    require(len(summaries) == 4, "comparison requires exactly four captures")
    require(minimum_timestamps > 0, "minimum timestamps must be positive")
    require(
        0.0 <= minimum_sample_span <= 1.0,
        "minimum sample span must be between zero and one",
    )
    require(material_change >= 0.0, "material change must be non-negative")

    captures = []
    counter_maps = []
    for capture_index, summary in enumerate(summaries, start=1):
        capture, counters = validate_capture(
            summary,
            capture_index=capture_index,
            minimum_timestamps=minimum_timestamps,
            minimum_sample_span=minimum_sample_span,
        )
        captures.append(capture)
        counter_maps.append(counters)

    capture_ids = [capture["capture_id"] for capture in captures]
    roles = [capture["role"] for capture in captures]
    require(
        len(set(capture_ids)) == len(capture_ids),
        "comparison contains a repeated capture ID",
    )
    require(
        roles == list(EXPECTED_CAPTURE_ORDER),
        "verified capture roles are not in the frozen ABBA order",
    )
    receipt_hashes = [
        capture["capture_receipt"]["sha256"] for capture in captures
    ]
    require(
        len(set(receipt_hashes)) == len(receipt_hashes),
        "comparison contains a repeated capture receipt",
    )
    commits = {capture["repository"]["commit"] for capture in captures}
    require(
        len(commits) == 1,
        "captures were not built from the same repository commit",
    )
    devices = {capture["runtime"]["device"] for capture in captures}
    backends = {capture["runtime"]["backend"] for capture in captures}
    require(len(devices) == 1, "captures do not identify the same GPU device")
    require(backends == {"metal"}, "captures do not share the Metal backend")

    role_binary_hashes: dict[str, set[str]] = {}
    role_provenance_hashes: dict[str, set[str]] = {}
    for role in ("baseline", "variant"):
        role_captures = [capture for capture in captures if capture["role"] == role]
        role_binary_hashes[role] = {
            capture["binary"]["sha256"] for capture in role_captures
        }
        role_provenance_hashes[role] = {
            capture["provenance"]["sha256"] for capture in role_captures
        }
        require(
            len(role_binary_hashes[role]) == 1,
            f"{role} captures do not use the same binary",
        )
        require(
            len(role_provenance_hashes[role]) == 1,
            f"{role} captures do not use the same provenance",
        )
    require(
        role_binary_hashes["baseline"] != role_binary_hashes["variant"],
        "baseline and variant captures use the same binary",
    )

    reference_names = set(counter_maps[0])
    for capture, counters in zip(captures[1:], counter_maps[1:], strict=True):
        require(
            set(counters) == reference_names,
            f"{capture['capture_id']} has a different named-counter set",
        )

    comparisons = []
    for name in sorted(reference_names):
        observed = [counter_map[name] for counter_map in counter_maps]
        counter_ids = {counter.get("counter_id") for counter in observed}
        counter_types = {counter.get("type") for counter in observed}
        counter_descriptions = {
            counter.get("description") for counter in observed
        }
        counter_units = {counter.get("unit") for counter in observed}
        require(
            len(counter_ids) == 1
            and len(counter_types) == 1
            and len(counter_descriptions) == 1
            and len(counter_units) == 1,
            f"counter metadata changed across captures for {name!r}",
        )
        medians = [float(counter["median"]) for counter in observed]
        pair_ratios = None
        median_ratio = None
        relative_change_percent = None
        direction = None
        classification = "ineligible_nonpositive_median"
        if all(median > 0.0 for median in medians):
            pair_ratios = [medians[1] / medians[0], medians[2] / medians[3]]
            median_ratio = statistics.median(pair_ratios)
            relative_change_percent = (median_ratio - 1.0) * 100.0
            if pair_ratios[0] > 1.0 and pair_ratios[1] > 1.0:
                direction = "variant_higher"
            elif pair_ratios[0] < 1.0 and pair_ratios[1] < 1.0:
                direction = "variant_lower"
            if direction is None:
                classification = "directionally_inconsistent"
            elif abs(median_ratio - 1.0) < material_change:
                classification = "below_material_threshold"
            else:
                classification = "repeatable_difference"
        comparisons.append(
            {
                "counter_id": next(iter(counter_ids)),
                "name": name,
                "type": next(iter(counter_types)),
                "description": next(iter(counter_descriptions)),
                "unit": next(iter(counter_units)),
                "capture_medians": {
                    capture["capture_id"]: median
                    for capture, median in zip(captures, medians, strict=True)
                },
                "pair_variant_over_baseline_ratios": pair_ratios,
                "median_variant_over_baseline_ratio": median_ratio,
                "relative_change_percent": relative_change_percent,
                "direction": direction,
                "classification": classification,
            }
        )

    repeatable = [
        counter
        for counter in comparisons
        if counter["classification"] == "repeatable_difference"
    ]
    return {
        "schema_version": 2,
        "analysis": "rmsnorm_limiter_counter_comparison",
        "protocol": {
            "capture_order": roles,
            "pairs": [
                f"{capture_ids[1]} / {capture_ids[0]}",
                f"{capture_ids[2]} / {capture_ids[3]}",
            ],
            "workload": EXPECTED_WORKLOAD,
            "minimum_counter_timestamps": minimum_timestamps,
            "minimum_counter_sample_span_fraction": minimum_sample_span,
            "material_relative_change_threshold": material_change,
            "repeatable_rule": (
                "Positive medians in all captures, both pair ratios in the "
                "same direction, and median absolute relative change at or "
                "above the material threshold."
            ),
            "identity_rule": (
                "Roles are derived from receipt-verified implementation and "
                "entrypoint identity. All captures must have unique capture "
                "receipts, the same clean commit, device, backend, and frozen "
                "workload, with one stable binary and provenance per role."
            ),
        },
        "captures": captures,
        "counter_summary": {
            "compared_count": len(comparisons),
            "eligible_positive_median_count": sum(
                counter["classification"]
                != "ineligible_nonpositive_median"
                for counter in comparisons
            ),
            "repeatable_difference_count": len(repeatable),
            "repeatable_difference_names": [
                counter["name"] for counter in repeatable
            ],
        },
        "comparisons": comparisons,
        "caveats": [
            "Counters are device-wide samples inside the target profile "
            "window, not command-buffer-exclusive measurements.",
            "The named counter set has no direct workgroup-barrier stall counter.",
            "Instrumented command durations are diagnostic and do not replace "
            "ordinary paired timing.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captures",
        type=Path,
        nargs=4,
        required=True,
        metavar=("CAPTURE_1", "CAPTURE_2", "CAPTURE_3", "CAPTURE_4"),
        help=(
            "four verified trace summaries in intended ABBA sequence; roles "
            "are derived from capture identity"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.captures
    require(
        len({path.resolve() for path in paths}) == len(paths),
        "comparison input paths must be unique",
    )
    summaries = [json.loads(path.read_text()) for path in paths]
    result = compare(summaries)
    result["inputs"] = [
        artifact(path, capture["capture_id"])
        for path, capture in zip(paths, result["captures"], strict=True)
    ]
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"counter comparison: {args.output.resolve()}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    try:
        main()
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
