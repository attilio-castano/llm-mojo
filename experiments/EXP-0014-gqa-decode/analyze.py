"""Verify retained EXP-0014 samples; optionally regenerate tables and plots.

Verification and CSV generation use only Python's standard library plus the
repository benchmark summarizer. Plotting is optional and needs matplotlib.
"""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from benchmarks.run_attention_decode import summarize


def read_json(path):
    return json.loads(path.read_text())


def verify():
    manifest = read_json(HERE / "manifest.json")
    for name, identity in manifest["retained_artifacts"].items():
        data = (HERE / name).read_bytes()
        assert len(data) == identity["bytes"], name
        assert hashlib.sha256(data).hexdigest() == identity["sha256"], name
    reports, observations = {}, {}
    for run in manifest["runs"]:
        name = run["id"]
        meta = read_json(HERE / run["metadata"])
        recorded = read_json(HERE / run["summary"])
        with gzip.open(HERE / run["samples"], "rt") as stream:
            samples = [json.loads(line) for line in stream]
        assert len(samples) == run["sample_count"], name
        assert meta["recorded"] and not meta["repository"]["dirty"], name
        assert meta["repository"]["commit"] == run["source_commit"], name
        counts = Counter()
        for s in samples:
            assert s["candidate"] in meta["candidates"]
            assert s["control"] == meta["control"]
            assert s["rows"] in meta["lengths"] and s["layers"] in (1, 24)
            assert s["block"] in (1, 2, 3, 4)
            assert s["candidate_first"] == (s["block"] in (2, 3))
            assert s["arm"] in ("candidate", "control")
            assert s["variant"] == s[s["arm"]]
            assert math.isfinite(s["us"]) and s["us"] > 0
            counts[s["candidate"], s["rows"], s["layers"], s["block"],
                   s["arm"], s["repetition"]] += 1
        expected = Counter({(c, t, l, b, a, r): 1
                            for c in meta["candidates"]
                            for t in meta["lengths"] for l in (1, 24)
                            for b in (1, 2, 3, 4)
                            for a in ("candidate", "control") for r in range(10)})
        assert counts == expected, name
        noise = {}
        if run["noise_run"]:
            for row in reports[run["noise_run"]]:
                noise[f'{row["rows"]}/{row["layers"]}'] = max(
                    abs(r - 1) for r in row["block_ratios"])
        calculated = summarize(samples, noise)
        assert len(calculated) == len(recorded), name
        for actual, original in zip(calculated, recorded):
            # Earlier summaries predate the optional conditional-rescale field.
            for key, value in original.items():
                assert actual[key] == value, (name, key, actual[key], value)
        reports[name], observations[name] = recorded, samples
    print(f'Verified {len(manifest["retained_artifacts"])} artifacts, '
          f'{len(reports)} valid runs and '
          f'{sum(map(len, observations.values()))} raw observations.')
    return reports, observations


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def write_csv(reports, observations):
    rows = []
    for run, report in reports.items():
        for item in report:
            group = [s for s in observations[run]
                     if (s["candidate"], s["rows"], s["layers"]) ==
                     (item["candidate"], item["rows"], item["layers"])]
            row = {"run": run, "candidate": item["candidate"],
                   "control": group[0]["control"], "T": item["rows"],
                   "mode": "hot" if item["layers"] == 1 else "ring24",
                   "control_median_us": item["control_us"],
                   "candidate_median_us": item["candidate_us"]}
            for arm in ("control", "candidate"):
                values = [s["us"] for s in group if s["arm"] == arm]
                row[f"{arm}_samples"] = len(values)
                row[f"{arm}_pooled_p10_us"] = percentile(values, 0.1)
                row[f"{arm}_pooled_p90_us"] = percentile(values, 0.9)
            ratios = item["block_ratios"]
            row.update(ratio_min=min(ratios), ratio_median=statistics.median(ratios),
                       ratio_max=max(ratios), improvement=item["improvement_fraction"],
                       noise_floor=item["noise_floor"], accepted=item["accepted"],
                       regression=item["regression"])
            rows.append(row)
    with (HERE / "statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "EXP-0014"})
    colors = {4: "#176596", 9: "#bf5a21"}
    lengths = sorted({s["rows"] for s in reports["confirmation-baseline"]})
    xs = list(range(len(lengths)))
    directory = HERE / "plots"
    directory.mkdir(exist_ok=True)

    def finish(fig, name, note):
        fig.text(0.08, 0.025, note, fontsize=9, color="#454545")
        fig.tight_layout(rect=(0.02, 0.085, 1, 0.94))
        fig.savefig(directory / f"{name}.png", dpi=180)
        svg = directory / f"{name}.svg"
        fig.savefig(svg, metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("GQA decode: gains against our materialized baseline", fontsize=15)
    for ax, layers, title in zip(axes, (1, 24),
                                 ("One call, then synchronize", "24-buffer sweep, then synchronize")):
        for candidate, label in ((4, "One kernel: 32 SIMD groups/head"),
                                 (9, "Two kernels: 64 splits, up to 4 heads/SIMD group")):
            rows = [s for s in reports["confirmation-baseline"]
                    if s["layers"] == layers and s["candidate"] == candidate]
            medians = [1 / statistics.median(s["block_ratios"]) for s in rows]
            low = [m - 1 / max(s["block_ratios"]) for m, s in zip(medians, rows)]
            high = [1 / min(s["block_ratios"]) - m for m, s in zip(medians, rows)]
            ax.errorbar(xs, medians, yerr=[low, high], color=colors[candidate],
                        linewidth=1.7, capsize=3, label=label)
            for x, m, s in zip(xs, medians, rows):
                ax.plot(x, m, "o", color=colors[candidate], markersize=5,
                        markerfacecolor=colors[candidate] if s["accepted"] else "white")
        ax.set_title(title, loc="left", fontsize=11)
        ax.axhline(1, color="#777777", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylabel("Paired speedup (×)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    axes[-1].set_xticks(xs, lengths)
    axes[-1].set_xlabel("KV context length T (categorical spacing)")
    finish(fig, "baseline-speedup",
           "Fresh seed 101 · Apple M4 Pro / Metal · medians and full four-block ranges\n"
           "Filled points pass the frozen gate. Each curve uses its own paired baseline; compare finalists directly below.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Where does splitting and sharing K/V help?", fontsize=15)
    for ax, layers, title in zip(axes, (1, 24),
                                 ("One call, then synchronize", "24-buffer sweep, then synchronize")):
        rows = [s for s in reports["confirmation-crossover"] if s["layers"] == layers]
        medians = [statistics.median(s["block_ratios"]) for s in rows]
        ax.plot(xs, medians, color=colors[9], linewidth=1.7, label="Median paired ratio")
        for b, offset in enumerate((-0.12, -0.04, 0.04, 0.12)):
            ax.scatter([x + offset for x in xs], [s["block_ratios"][b] for s in rows],
                       s=18, color=colors[9], alpha=0.4, label="Four blocks" if b == 0 else None)
        ax.axhline(1, color="#444444", linewidth=1, label="Equal latency")
        thresholds = [1 - max(0.05, s["noise_floor"]) for s in rows]
        ax.plot(xs, thresholds, linestyle="--", color="#608368", linewidth=1,
                label="Gain/noise threshold (also requires 4/4 faster blocks)")
        for x, m, s in zip(xs, medians, rows):
            if s["accepted"]:
                ax.plot(x, m, "D", color="#176596", markersize=6)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Two-kernel / one-kernel latency")
        ax.set_ylim(0.25, 3.4)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[-1].set_xticks(xs, lengths)
    axes[-1].set_xlabel("KV context length T (categorical spacing)")
    finish(fig, "crossover",
           "Lower is faster for splitting. Blue diamonds pass the frozen gate. T=2048 ring24 has only 3/4 faster blocks.\n"
           "Direct paired confirmation · seed 101 · host enqueue through completion · no automatic dispatch rule")
    (directory / "environment.json").write_text(json.dumps({
        "python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
        "data": ["confirmation-baseline", "confirmation-crossover"],
        "scope": "Derived figures; no new GPU measurements"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-derived", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    reports, observations = verify()
    if args.write_derived:
        write_csv(reports, observations)
    if args.plots:
        plot(reports)
