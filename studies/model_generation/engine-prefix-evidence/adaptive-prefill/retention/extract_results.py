#!/usr/bin/env python3
"""Extract a compact derived report from a completed adaptive-prefill archive.

Standard library only. This reads retained summaries; it does not reconstruct
native events or replace the source-bound guarded CPU replay. Run after timing.
The output path must be fresh and its parent must already exist.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


KIND = "qwen-engine-prefix-adaptive-v1"
CELLS = ('prefix-1024-reuse-4-blocks-384-suffix-64-out-16-first', 'prefix-1024-reuse-8-blocks-384-suffix-64-out-16-first', 'prefix-1024-reuse-4-blocks-40-suffix-64-out-16-first', 'prefix-1024-reuse-8-blocks-40-suffix-64-out-16-first', 'prefix-1024-reuse-0-blocks-384-suffix-64-out-16-first', 'prefix-1024-reuse-8-blocks-41-suffix-64-out-16-first', 'prefix-1024-reuse-8-blocks-56-suffix-64-out-16-first', 'prefix-1024-reuse-8-blocks-36-suffix-33-out-16-first', 'prefix-1024-reuse-8-blocks-40-suffix-63-out-16-first', 'prefix-1024-reuse-8-blocks-40-suffix-65-out-16-first', 'prefix-1024-reuse-8-blocks-40-suffix-64-out-2-first', 'prefix-1024-reuse-4-blocks-40-suffix-64-out-16-even')
ARMS = {("current", False): "current", ("current", True): "self_current",
        ("fixed32", False): "fixed32", ("adaptive", False): "adaptive"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "invalid nonnegative metric: " + name)
    return value


def count(value, name):
    require(type(value) is int and value >= 0, "invalid count: " + name)
    return value


def milliseconds(value, name):
    return None if value is None else number(value, name) / 1_000_000


def distribution(run, name, percentile):
    value = run[name]
    return None if value is None else milliseconds(value[percentile + "_ns"], name + percentile)


def run_metrics(run):
    require(type(run["block"]) is int and run["block"] in range(4), "invalid block")
    mixed = run["step_classes"]["mixed"]
    gaps = run["gap_tail"]
    return dict(
        block=run["block"],
        duration_ms=milliseconds(run["duration_ns"], "duration"),
        episode_ms=milliseconds(run["episode_ns"], "episode"),
        seed_ms=milliseconds(run["seed_ns"], "seed"),
        clear_ms=milliseconds(run["clear_ns"], "clear"),
        ttft_p50_ms=distribution(run, "ttft", "p50"),
        ttft_p95_ms=distribution(run, "ttft", "p95"),
        gap_p95_ms=distribution(run, "inter_token", "p95"),
        gap_p99_ms=distribution(run, "inter_token", "p99"),
        gap_maximum_ms=milliseconds(gaps["maximum_ns"], "maximum gap"),
        steps=count(run["steps"], "steps"),
        mixed_steps=count(mixed["count"], "mixed steps"),
        mixed_prefill_rows=count(mixed["prefill_rows"], "mixed prefill rows"),
        maximum_mixed_prefill_rows=count(mixed["prefill_rows_max"], "maximum mixed prefill rows"),
        gaps=count(gaps["count"], "gaps"),
        gaps_above_20ms=count(gaps["above_20ms"], "gaps above 20ms"),
        gaps_above_50ms=count(gaps["above_50ms"], "gaps above 50ms"),
        computed_rows=count(run["total_tokens"], "computed rows"),
        reused_prefix_tokens=count(run["cache_hit_tokens"], "cache hit tokens"),
        seed_rows=count(run["seed_rows"], "seed rows"),
        attended_positions=count(run["attended_positions"], "attended positions"),
        delivered_tokens=count(run["delivered_tokens"], "delivered tokens"),
        preemptions=count(run["preemptions"], "preemptions"),
    )


def extract(archive, manifest_path):
    packed = archive.read_bytes()
    manifest = decode(manifest_path.read_bytes())
    require(manifest.get("kind") == KIND, "wrong manifest kind")
    require(type(manifest.get("bytes")) is int and len(packed) == manifest["bytes"],
            "compressed byte count differs")
    require(digest(packed) == manifest.get("sha256"), "compressed SHA-256 differs")
    raw = gzip.decompress(packed)
    require(digest(raw) == manifest.get("uncompressed_sha256"), "uncompressed SHA-256 differs")
    if "uncompressed_bytes" in manifest:
        require(type(manifest["uncompressed_bytes"]) is int and len(raw) == manifest["uncompressed_bytes"],
                "uncompressed byte count differs")
    record = decode(raw)
    require(record.get("kind") == KIND, "wrong archive kind")
    summary = record["summary"]
    require(type(summary.get("measured_runs")) is int and summary["measured_runs"] == 192
            and type(summary.get("untimed_preflights")) is int and summary["untimed_preflights"] == 36,
            "incomplete 192-run/36-preflight summary")
    require(tuple(c["cell_id"] for c in record["cells"]) == CELLS
            and tuple(c["cell_id"] for c in summary["cells"]) == CELLS,
            "cell identity/order differs")
    commit = record["build"]["source"]["repository"]["commit"]
    require(type(commit) is str and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            "invalid source commit")
    cells = []
    for original, cell in zip(record["cells"], summary["cells"]):
        require(len(original["runs"]) == len(cell["runs"]) == 16, "incomplete cell")
        grouped = {arm: [] for arm in ARMS.values()}
        for run in cell["runs"]:
            require(type(run.get("prefill_policy")) is str and type(run.get("calibration")) is bool
                    and run.get("cache") == "enabled", "invalid arm identity")
            key = run["prefill_policy"], run["calibration"]
            require(key in ARMS, "unexpected arm")
            grouped[ARMS[key]].append(run_metrics(run))
        arms = {}
        for name, runs in grouped.items():
            runs.sort(key=lambda r: r["block"])
            require([r["block"] for r in runs] == list(range(4)), "missing/duplicate arm block")
            medians = {}
            for metric in runs[0]:
                if metric == "block":
                    continue
                values = [run[metric] for run in runs]
                medians[metric] = None if any(v is None for v in values) else statistics.median(values)
            arms[name] = dict(runs=runs, median_per_run_metrics=medians)
        cells.append(dict(cell_id=cell["cell_id"], arms=arms,
                          comparisons=cell["comparisons"], decisions=cell["decisions"],
                          logical_schedule_matches=cell["logical_schedule_matches"]))
    return dict(
        kind="engine-prefix-adaptive-derived-report-v1", schema_version=1,
        authority="Derived retained-summary report; guarded CPU replay remains authoritative.",
        inputs=dict(archive=str(archive), manifest=str(manifest_path), bytes=len(packed),
                    sha256=digest(packed), uncompressed_sha256=digest(raw)),
        source_commit=commit, measured_runs=192, untimed_preflights=36,
        aggregation="Median of four per-run metrics for each arm; no pooling of gaps across runs.",
        arms=dict(current=dict(prefill_policy="current", calibration=False),
                  self_current=dict(prefill_policy="current", calibration=True),
                  fixed32=dict(prefill_policy="fixed32", calibration=False),
                  adaptive=dict(prefill_policy="adaptive", calibration=False)),
        usefulness=summary["usefulness"], same_delivered_histories=summary["same_delivered_histories"],
        same_reuse_work=summary["same_reuse_work"], promotion=summary["promotion"], cells=cells,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Completed engine-prefix-adaptive.json.gz")
    parser.add_argument("output", type=Path, help="Fresh compact JSON output path")
    parser.add_argument("--manifest", type=Path, help="Defaults to archive path with .gz removed")
    args = parser.parse_args()
    archive = args.archive.resolve()
    manifest = (args.manifest or archive.with_suffix("")).resolve()
    output = args.output.absolute()
    require(not output.exists(), "refusing to replace output: " + str(output))
    result = extract(archive, manifest)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps(dict(output=str(output), source_commit=result["source_commit"],
                          measured_runs=192, cells=len(result["cells"]))))


if __name__ == "__main__":
    main()
