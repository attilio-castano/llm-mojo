"""Recompute tables and figures from retained raw samples; no GPU needed.

Run with: uv run --no-project --with matplotlib==3.10.8 python benchmarks/plot.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from study import load_run

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#6b7280', '#167d9a', '#c75b39', '#8064a2', '#579059']


def render(directory):
    record, samples, summary = load_run(directory)
    spec = record['specification']
    names = {int(k): v for k, v in spec['names'].items()}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': '#fcfcfa', 'figure.facecolor': '#fcfcfa'})
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), squeeze=False)
    for col, layers in enumerate((1, 24)):
        ax, relative = axes[0, col], axes[1, col]
        ax.set_title('Hot call · one synchronization' if layers == 1 else '24 buffers · one synchronization per sweep')
        for color, candidate in zip(COLORS, spec['candidates']):
            data = [s for s in summary if s['layers'] == layers and s['candidate'] == candidate]
            x = [s['rows'] for s in data]
            ax.plot(x, [s['candidate_us'] for s in data], '-o', color=color, label=names[candidate], markersize=4)
            if candidate != spec['control']:
                relative.plot(x, [s['ratio'] for s in data], '-o', color=color, markersize=4)
                relative.fill_between(x, [s['ratio_min'] for s in data], [s['ratio_max'] for s in data], color=color, alpha=.13)
                uncertain = [s for s in data if s['decision'] == 'inconclusive']
                relative.scatter([s['rows'] for s in uncertain], [s['ratio'] for s in uncertain],
                                 s=65, facecolors='none', edgecolors=color, linewidths=1.3)
        # The baseline self-pair supplies a workload-specific conservative noise floor.
        calibration = [s for s in summary if s['layers'] == layers and s['candidate'] == spec['control']]
        relative.plot([s['rows'] for s in calibration], [max(.001, 1 - s['noise_floor']) for s in calibration],
                      ':', color='#777777', label='gain threshold')
        relative.axhline(1, color='#333333', linewidth=.8)
        ax.set_yscale('log')
        ax.set_ylabel('µs / operation · median of block medians')
        relative.set_ylabel('Candidate / paired control · lower is faster')
        if len(spec['rows']) > 1:
            ax.set_xscale('log', base=2)
            relative.set_xscale('log', base=2)
        for panel in (ax, relative):
            panel.set_xticks(spec['rows'], labels=list(map(str, spec['rows'])))
            panel.grid(axis='y', alpha=.18)
        relative.set_xlabel('KV context length T' if record['study'] == 'gqa_decode' else 'Token rows M')
        if len(spec['candidates']) == 1:
            relative.text(.5, .55, 'Baseline characterization only\nNo optimized RoPE comparison',
                          ha='center', va='center', transform=relative.transAxes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=min(5, len(labels)), frameon=False)
    fig.suptitle(record['study'].replace('_', ' ').upper() + f"  /  {record['runtime']['device']} · Metal · BF16", fontsize=17, fontweight='bold', y=.995)
    fig.text(.05, .02, 'Shading: range of four paired block ratios. Open rings: inconclusive under the noise/direction rule.\n'
             f"{len(samples):,} retained observations · source {record['repository']['commit'][:7]} · enqueue through completion; not model throughput.", fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .075, 1, .905))
    fig.savefig(directory / 'latency.png', dpi=160)
    plt.close(fig)
    print(directory.name, len(samples), 'observations verified; summary and figure regenerated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='*', type=Path)
    args = parser.parse_args()
    directories = args.directories or sorted((ROOT / 'studies').glob('*/'))
    for directory in directories:
        render(directory)


if __name__ == '__main__':
    main()
