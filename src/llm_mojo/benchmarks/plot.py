"""Recompute tables and figures from retained raw samples; no GPU needed.

Run with: uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .study import load_run, load_profile

from .._repository import repository_root
COLORS = ['#6b7280', '#167d9a', '#c75b39', '#8064a2', '#579059']


def table(directory, name, rows):
    with (directory / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prefill_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': '#fcfcfa', 'figure.facecolor': '#fcfcfa'})


def prefill_screen(directory, record, samples, summary):
    # Contract: eleven routes x six workload/mode cells. A ratio matrix shows
    # stage selection without connecting different rectangular workloads.
    # Static report PNG; blue/orange plus numbers and ? for inconclusive cells.
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    spec = record['specification']
    cases = [(w['query_rows'],w['rows'],l) for l in (1,24) for w in spec['workloads']]
    candidates = spec['candidates']
    lookup = {(s['query_rows'],s['rows'],s['layers'],s['candidate']):s for s in summary}
    values = np.array([[lookup[(*case,c)]['ratio'] for case in cases] for c in candidates])
    scale = max(1,float(np.max(np.abs(np.log2(values)))))
    cmap = LinearSegmentedColormap.from_list('paired',['#a9c8dc','#fcfcfa','#e8b88a'])
    fig, ax = plt.subplots(figsize=(10.5,7.4))
    ax.imshow(np.log2(values),cmap=cmap,norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),aspect='auto')
    names = {int(k):v for k,v in spec['names'].items()}
    ax.set_yticks(range(len(candidates)),[names[c] for c in candidates])
    ax.set_xticks(range(len(cases)),[f'R={r}, T={t}\n'+('Hot' if l==1 else 'Ring24') for r,t,l in cases])
    for i,c in enumerate(candidates):
        for j,case in enumerate(cases):
            s = lookup[(*case,c)]
            text = f'{s["ratio"]:.2f}×' + (' ?' if s['decision']=='inconclusive' else '')
            ax.text(j,i,text,ha='center',va='center',color='#222222',fontsize=10)
    ax.axvline(2.5,color='#777777',linewidth=1)
    ax.tick_params(length=0,pad=10)
    fig.suptitle('GQA prefill · bounded candidate screen',fontsize=17,fontweight='bold',y=.97)
    fig.text(.5,.918,f'Time / paired materialized control · lower is faster · {record["runtime"]["device"]} / Metal / BF16',ha='center',fontsize=10)
    fig.text(.04,.035,f'{len(samples):,} observations · four paired blocks · ? = inconclusive under the measured noise rule.\n'
             'Each cell uses its own paired control. First row is self-pair calibration; colors are centered on equal time.\n'
             f'Source {record["repository"]["commit"][:7]}. Exact ratios, block ranges and decisions: screen_summary.csv.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.12,1,.88))
    fig.savefig(directory/'screen.png',dpi=160)
    plt.close(fig)


def render_prefill(directory, record, samples, summary):
    # Contract: full prefill is a five-point ordered length curve; incremental
    # prefill is six discrete (R,T) categories. Both show per-call microseconds.
    # Paired dot/range panels retain uncertainty independently of absolute time.
    # Static PNGs, explicit series colors plus distinct markers and open fills.
    prefill_style()
    spec = record['specification']
    names = {int(k):v for k,v in spec['names'].items()}
    candidates = spec['candidates']
    colors = dict(zip(candidates,['#6b7280','#c75b39','#167d9a','#579059','#c17b9a']))
    markers = dict(zip(candidates,['o','s','D','^','v']))
    table(directory,'summary.csv',summary)
    fig, axes = plt.subplots(2,2,figsize=(12,8.4),sharey='row')
    for col,layers in enumerate((1,24)):
        for row,full in enumerate((True,False)):
            ax = axes[row,col]
            workloads = [w for w in spec['workloads'] if (w['query_rows']==w['rows'])==full]
            for i,c in enumerate(candidates):
                data = [s for s in summary if s['layers']==layers and s['candidate']==c
                        and (s['query_rows']==s['rows'])==full]
                x = [s['rows'] for s in data] if full else [j+(i-(len(candidates)-1)/2)*.12 for j in range(len(data))]
                ax.plot(x,[s['candidate_us'] for s in data],linestyle='-' if full else 'none',
                        marker=markers[c],markersize=5,color=colors[c],label=names[c])
            ax.set_yscale('log')
            ax.set_ylabel('Latency (µs / attention) · log scale')
            ax.grid(axis='y',alpha=.18)
            if full:
                ax.set_xscale('log',base=2)
                ax.set_xticks([w['rows'] for w in workloads],[str(w['rows']) for w in workloads])
                ax.set_xlabel('Full prefill · R = T')
            else:
                ax.set_xticks(range(len(workloads)),[f'R={w["query_rows"]}\nT={w["rows"]}' for w in workloads])
                ax.set_xlabel('Incremental prefill · query rows R, total KV rows T')
        axes[0,col].set_title('Hot · one call through completion' if layers==1 else 'Ring24 · one synchronization per sweep')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=len(candidates),frameon=False)
    fig.suptitle(f'GQA prefill latency · {record["runtime"]["device"]} / Metal / BF16',fontsize=17,fontweight='bold',y=.995)
    fig.text(.04,.025,f'{len(samples):,} retained observations · median of four block medians · source {record["repository"]["commit"][:7]}.\n'
             'Enqueue through completion; allocation, checks and compilation excluded. Ring24 uses distinct Q/K/V and reuses O/scratch.\n'
             'Absolute curves do not establish paired gains; see comparisons.png and summary.csv. This is not model throughput.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.115,1,.91))
    fig.savefig(directory/'latency.png',dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(12,8.3),sharey=True,sharex=True)
    rivals = [c for c in candidates if c != spec['control']]
    paired = [s for s in summary if s['candidate'] in rivals]
    ratio_limits = (min(1,*(s['ratio_min'] for s in paired))/1.15,
                    max(1,*(s['ratio_max'] for s in paired))*1.15)
    for ax,layers in zip(axes,(1,24)):
        for i,c in enumerate(rivals):
            data=[s for s in summary if s['layers']==layers and s['candidate']==c]
            for j,s in enumerate(data):
                y=j+(i-(len(rivals)-1)/2)*.2
                ax.errorbar(s['ratio'],y,xerr=[[s['ratio']-s['ratio_min']],[s['ratio_max']-s['ratio']]],
                            color=colors[c],fmt=markers[c],capsize=3,markersize=5,
                            markerfacecolor='white' if s['decision']=='inconclusive' else colors[c],
                            label=names[c] if j==0 else None)
        ax.axvline(1,color='#333333',linewidth=1)
        ax.set_xscale('log',base=2)
        ax.set_xlim(*ratio_limits)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:g}×'))
        ax.grid(axis='x',alpha=.18)
        ax.set_xlabel(f'Time / paired {names[spec["control"]]} · lower is faster')
        ax.set_title('Hot call' if layers==1 else 'Ring24 per call')
    labels=[f'Full {w["rows"]}' if w['query_rows']==w['rows'] else f'R={w["query_rows"]}, T={w["rows"]}' for w in spec['workloads']]
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=len(rivals),frameon=False)
    fig.suptitle('GQA prefill · direct paired comparisons',fontsize=17,fontweight='bold',y=.995)
    fig.text(.04,.025,'Whiskers: range of four paired block ratios, not confidence intervals. Open marks: inconclusive.\n'
             'A gain needs all blocks faster and a median reduction exceeding both 5% and the matching control self-pair deviation.\n'
             f'{len(samples):,} observations · {record["runtime"]["device"]} / Metal / BF16 · source {record["repository"]["commit"][:7]}.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.12,1,.91))
    fig.savefig(directory/'comparisons.png',dpi=160)
    plt.close(fig)
    if (directory/'screen_run.json').exists():
        screen_record,screen_samples,screen_summary=load_run(directory,'screen_')
        table(directory,'screen_summary.csv',screen_summary)
        prefill_screen(directory,screen_record,screen_samples,screen_summary)
    if (directory/'profiles.json').exists():
        rows=load_profile(directory)
        table(directory,'profile_summary.csv',rows)
        fig,axes=plt.subplots(1,3,figsize=(12,4.9))
        workloads=list(dict.fromkeys((s['query_rows'],s['rows']) for s in rows))
        for ax,(r,t) in zip(axes,workloads):
            data=[s for s in rows if s['query_rows']==r and s['rows']==t]
            ax.barh(range(len(data)),[s['median_us'] for s in data],color=['#6b7280']*3+['#167d9a'],height=.55)
            ax.set_yticks(range(len(data)),[s['stage'] for s in data])
            ax.invert_yaxis()
            ax.set_title(f'R={r}, T={t}')
            ax.set_xlabel('Median active GPU time (µs)')
            ax.set_xlim(0,max(s['median_us'] for s in data)*1.25)
            for j,s in enumerate(data):
                ax.text(s['median_us'],j,f' {s["median_us"]:.1f}',va='center',fontsize=9)
            ax.grid(axis='x',alpha=.18)
        fig.suptitle('GQA prefill · active time per GPU dispatch',fontsize=16,fontweight='bold',y=.98)
        fig.text(.04,.032,'Gray: materialized QK, softmax and PV. Blue: fused finalist. Separate captures from paired latency runs.\n'
                 'Preempted segments are joined by dispatch identity; durations exclude preemption gaps and host gaps.\n'
                 f'{sum(s["count"] for s in rows):,} target dispatch durations retained; see profiles.json and profile_summary.csv.',fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.2,1,.9))
        fig.savefig(directory/'profile.png',dpi=160)
        plt.close(fig)
    print(directory.name,len(samples),'observations verified; prefill figures regenerated')


def render(directory):
    record, samples, summary = load_run(directory)
    spec = record['specification']
    if record['study'] == 'gqa_prefill_screen':
        prefill_style()
        table(directory,'summary.csv',summary)
        return prefill_screen(directory,record,samples,summary)
    if spec['operation'] == 'gqa_prefill':
        return render_prefill(directory,record,samples,summary)
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
    directories = args.directories or sorted((repository_root() / 'studies').glob('*/'))
    for directory in directories:
        render(directory)


if __name__ == '__main__':
    main()
