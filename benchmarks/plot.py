"""Recompute tables and figures from retained raw samples; no GPU needed.

Run with: uv run --no-project --with matplotlib==3.10.8 python benchmarks/plot.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from study import load_run, load_profile

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
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), squeeze=False)
    for col, layers in enumerate((1, 24)):
        ax, relative = axes[0, col], axes[1, col]
        ax.set_title('Hot call · one synchronization' if layers == 1 else '24 buffers · one synchronization per sweep')
        for color, candidate in zip(COLORS, spec['candidates']):
            data = [s for s in summary if s['layers'] == layers and s['candidate'] == candidate]
            single = len(spec['rows']) == 1
            x = [spec['candidates'].index(candidate)] if single else [s['rows'] for s in data]
            if single:
                ax.bar(x, [s['candidate_us'] for s in data], color=color, label=names[candidate], width=.55)
            else:
                ax.plot(x, [s['candidate_us'] for s in data], '-o', color=color, label=names[candidate], markersize=4)
            if candidate != spec['control'] or len(spec['candidates']) == 1:
                relative.plot(x, [s['ratio'] for s in data], '-o', color=color, markersize=4)
                if single:
                    relative.errorbar(x, [s['ratio'] for s in data],
                                      yerr=[[s['ratio']-s['ratio_min'] for s in data], [s['ratio_max']-s['ratio'] for s in data]],
                                      fmt='none', color=color, capsize=5, alpha=.6)
                else:
                    relative.fill_between(x, [s['ratio_min'] for s in data], [s['ratio_max'] for s in data], color=color, alpha=.13)
                uncertain = [s for s in data if s['decision'] == 'inconclusive']
                relative.scatter([spec['candidates'].index(candidate) if single else s['rows'] for s in uncertain], [s['ratio'] for s in uncertain],
                                 s=65, facecolors='none', edgecolors=color, linewidths=1.3)
        # The baseline self-pair supplies a workload-specific conservative noise floor.
        calibration = [s for s in summary if s['layers'] == layers and s['candidate'] == spec['control']]
        if len(spec['rows']) == 1:
            relative.axhline(1 - calibration[0]['noise_floor'], linestyle=':', color='#777777')
        else:
            relative.plot([s['rows'] for s in calibration], [1 - s['noise_floor'] for s in calibration], ':', color='#777777')
        relative.axhline(1, color='#333333', linewidth=.8)
        if len(spec['rows']) > 1:
            ax.set_yscale('log')
        ax.set_ylabel('Latency (µs / operation)')
        relative.set_ylabel('Self-pair time ratio' if len(spec['candidates']) == 1 else 'Time / paired control')
        if len(spec['rows']) > 1:
            ax.set_xscale('log', base=2)
            relative.set_xscale('log', base=2)
        for panel in (ax, relative):
            if len(spec['rows']) == 1:
                panel.set_xticks(range(len(spec['candidates'])), labels=[names[c].replace(' ', '\n', 1) for c in spec['candidates']])
                panel.set_xlim(-.5, len(spec['candidates'])-.5)
            else:
                panel.set_xticks(spec['rows'], labels=list(map(str, spec['rows'])))
            panel.grid(axis='y', alpha=.18)
        relative.set_xlabel('KV context length T' if record['study'] == 'gqa_decode' else ('One token row' if len(spec['rows']) == 1 else 'Token rows M'))
        if len(spec['candidates']) == 1:
            relative.text(.5, .97, 'Self-pair calibration · no optimized comparison',
                          ha='center', va='top', fontsize=9, transform=relative.transAxes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=min(5, len(labels)), frameon=False)
    fig.suptitle(record['study'].replace('_', ' ').upper() + f"  /  {record['runtime']['device']} · Metal · BF16", fontsize=17, fontweight='bold', y=.995)
    fig.text(.05, .018, 'Latency: median of block medians. Shading/whiskers: range of four paired ratios. Open rings: inconclusive.\n'
             'Dotted line: gain threshold (negative means noise prevents any gain claim). Lower ratios are faster.\n'
             'Gray latency: self-pair control. Ratios use their own paired controls; these can differ in a noisy run.\n'
             f"{len(samples):,} retained observations · source {record['repository']['commit'][:7]} · enqueue through completion; not model throughput.", fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .14, 1, .905))
    fig.savefig(directory / 'latency.png', dpi=160)
    plt.close(fig)
    if (directory / 'profiles.json').exists():
        profile_rows = load_profile(directory)
        with (directory / 'profile_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(profile_rows)
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
