"""Recompute tables and figures from retained raw samples; no GPU needed.

Run with: uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

from .study import load_run, load_profile

from .._repository import repository_root
COLORS = ['#6b7280', '#167d9a', '#c75b39', '#8064a2', '#579059']


def figure_directory(directory):
    figures = directory.parent / 'figures'
    return figures if directory.name == 'data' and figures.is_dir() else directory


def table(directory, name, rows):
    with (directory / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prefill_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': '#fcfcfa', 'figure.facecolor': '#fcfcfa'})


def prefill_screen(directory, record, samples, summary, prefix='screen_', resources=False):
    # A ratio matrix shows the frozen screen without connecting different
    # rectangular workloads. Each cell has its own paired control.
    # Static report PNG; blue/orange plus numbers and ? for inconclusive cells.
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    spec = record['specification']
    cases = [(w['query_rows'],w['rows'],l) for l in (1,24) for w in spec['workloads']]
    candidates = spec['candidates']
    lookup = {(s['query_rows'],s['rows'],s['layers'],s['candidate']):s for s in summary}
    values = np.array([[lookup[(*case,c)]['ratio'] for case in cases] for c in candidates])
    scale = max(.15 if resources else 1,float(np.max(np.abs(np.log2(values)))))
    cmap = LinearSegmentedColormap.from_list('paired',['#a9c8dc','#fcfcfa','#e8b88a'])
    fig, ax = plt.subplots(figsize=(14,6.3) if resources else (10.5,7.4))
    ax.imshow(np.log2(values),cmap=cmap,norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),aspect='auto')
    names = {int(k):v for k,v in spec['names'].items()}
    ax.set_yticks(range(len(candidates)),[names[c] for c in candidates])
    ax.set_xticks(range(len(cases)),[(f'R={r}\nT={t}\n' if resources else f'R={r}, T={t}\n')+('Hot' if l==1 else 'Ring24') for r,t,l in cases])
    for i,c in enumerate(candidates):
        for j,case in enumerate(cases):
            s = lookup[(*case,c)]
            text = f'{s["ratio"]:.2f}×' + (' ?' if s['decision']=='inconclusive' else '')
            ax.text(j,i,text,ha='center',va='center',color='#222222',fontsize=10)
    ax.axvline(len(cases)/2-.5,color='#777777',linewidth=1)
    ax.tick_params(length=0,pad=10)
    fig.suptitle('GQA prefill · compiler and synchronization ablations' if resources else 'GQA prefill · bounded candidate screen',fontsize=17,fontweight='bold',y=.97)
    fig.text(.5,.918,f'Time / paired {names[spec["control"]]} control · lower is faster · {record["runtime"]["device"]} / Metal / BF16',ha='center',fontsize=10)
    fig.text(.04,.035,f'{len(samples):,} observations · four paired blocks · ? = inconclusive under the measured noise rule.\n'
             'Each cell uses its own paired control. First row is self-pair calibration; colors are centered on equal time.\n'
             f'Source {record["repository"]["commit"][:7]}. Exact ratios, block ranges and decisions: {prefix}summary.csv.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.12,1,.88))
    fig.savefig(figure_directory(directory)/(prefix.rstrip('_')+'.png'),dpi=160)
    plt.close(fig)


def prefill_comparisons(directory, record, samples, summary, prefix='', resources=False):
    spec = record['specification']
    names = {int(k):v for k,v in spec['names'].items()}
    candidates = spec['candidates']
    colors = dict(zip(candidates,['#6b7280','#c75b39','#167d9a','#579059','#c17b9a']))
    markers = dict(zip(candidates,['o','s','D','^','v']))
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
        if not resources:
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
    fig.suptitle('GQA prefill · rolled QK against original MMA' if resources else 'GQA prefill · direct paired comparisons',fontsize=17,fontweight='bold',y=.995)
    fig.text(.04,.025,'Whiskers: range of four paired block ratios, not confidence intervals. Open marks: inconclusive.\n'
             'A gain needs all blocks faster and a median reduction exceeding both 5% and the matching control self-pair deviation.\n'
             f'{len(samples):,} observations · {record["runtime"]["device"]} / Metal / BF16 · source {record["repository"]["commit"][:7]}.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.12,1,.91))
    fig.savefig(figure_directory(directory)/(prefix+'comparisons.png'),dpi=160)
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
    fig.savefig(figure_directory(directory)/'latency.png',dpi=160)
    plt.close(fig)

    prefill_comparisons(directory,record,samples,summary)
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
        fig.savefig(figure_directory(directory)/'profile.png',dpi=160)
        plt.close(fig)
    if (directory/'resources_screen_run.json').exists():
        resource_record,resource_samples,resource_summary=load_run(directory,'resources_screen_')
        table(directory,'resources_screen_summary.csv',resource_summary)
        prefill_screen(directory,resource_record,resource_samples,resource_summary,'resources_screen_',True)
    if (directory/'resources_run.json').exists():
        resource_record,resource_samples,resource_summary=load_run(directory,'resources_')
        table(directory,'resources_summary.csv',resource_summary)
        prefill_comparisons(directory,resource_record,resource_samples,resource_summary,'resources_',True)
    if (directory/'resources_profiles.json').exists():
        table(directory,'resources_profile_summary.csv',load_profile(directory,'resources_'))
    print(directory.name,len(samples),'primary observations verified; prefill figures and retained follow-ups regenerated')


def render_sublayer_wo(directory, prefix, *, fp32_prefill=False, integration=False):
    """Whole-block paired ratios for the single Wo mapping experiment."""
    record, samples, summary = load_run(directory, prefix)
    table(directory, prefix+'summary.csv', summary)
    spec = record['specification']
    data = [s for s in summary if s['candidate'] == (9 if integration else (7 if fp32_prefill else 4))]
    labels = [f'Decode T={w["rows"]}' if w['query_rows'] == 1 else
              f'Full R=T={w["rows"]}' if w['query_rows'] == w['rows'] else
              f'Chunk R={w["query_rows"]}, T={w["rows"]}' for w in spec['workloads']]
    fig, axes = plt.subplots(1,2,figsize=(12, max(5.5,.37*len(labels)+2.7)),sharex=True,sharey=True)
    for ax,layers in zip(axes,(1,24)):
        values = [s for s in data if s['layers'] == layers]
        for i,s in enumerate(values):
            color = '#167d9a' if s['decision']=='faster' else '#c75b39' if s['decision']=='slower' else '#777777'
            ax.errorbar(s['ratio'],i,xerr=[[s['ratio']-s['ratio_min']],[s['ratio_max']-s['ratio']]],
                        fmt='o',color=color,capsize=3,markersize=6,
                        markerfacecolor='white' if s['decision']=='inconclusive' else color)
        ax.axvline(1,color='#333333',linewidth=1)
        ax.grid(axis='x',alpha=.18)
        ax.set_title('Hot call' if layers==1 else 'Ring24 per call')
        ax.set_xlabel('Whole-block time / paired control' if integration else
                      ('Whole-block time / paired control (MMA Wo fixed)' if fp32_prefill
                       else 'Whole-block time / paired rowwise Wo control'))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}×'))
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis()
    screen = record['study'].endswith('_screen')
    title = ('Qwen attention · integrating QKV projections' if prefix == 'projections_' else
             'Qwen attention · all studied mappings together') if integration else (
             'Qwen attention · FP32 prefill tiling' if fp32_prefill else 'Qwen attention · changing only Wo')
    fig.suptitle(title
                 +(' · screen' if screen else ''),fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Left of 1× is faster. Whiskers: four-block ratio range; open marks: inconclusive.\n'
             'A gain requires all four blocks faster and > max(5%, matching self-pair deviation).\n'
             f'{len(samples):,} retained observations · {record["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention.\n'
             f'Source {record["repository"]["commit"][:7]}. Fixed cache prefix; allocation and correctness checks excluded.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.18,1,.94))
    fig.savefig(figure_directory(directory)/(prefix.rstrip('_')+'.png'),dpi=160)
    plt.close(fig)


def render_sublayer_parallelism(directory, prefix):
    record,samples,summary = load_run(directory,prefix)
    table(directory,prefix+'summary.csv',summary)
    spec = record['specification']
    variants = [v for v in spec['candidates'] if v != spec['control']]
    labels = [f'Decode T={w["rows"]}' if w['query_rows'] == 1 else
              f'Full R=T={w["rows"]}' if w['query_rows'] == w['rows'] else
              f'Chunk R={w["query_rows"]}, T={w["rows"]}' for w in spec['workloads']]
    colors = {10:'#167d9a',11:'#3c9e79',12:'#8064a2',13:'#ba7437',
              14:'#167d9a',15:'#8064a2',16:'#167d9a',17:'#8064a2',18:'#167d9a',19:'#167d9a'}
    isolated = spec.get('measurement') == 'isolated_wo'
    split_combined = prefix.startswith('split_combined_')
    tiles = prefix.startswith('tiles') or prefix == 'combined_'
    boundary = 'Isolated Wo' if isolated else 'Whole-block'
    fig,axes = plt.subplots(1,2,figsize=(13,max(6,.6*len(labels)+2.4)),sharex=True,sharey=True)
    for ax,layers in zip(axes,(1,24)):
        for index,variant in enumerate(variants):
            offset = (index-(len(variants)-1)/2)*.17
            values = [s for s in summary if s['candidate']==variant and s['layers']==layers]
            for i,s in enumerate(values):
                color = colors[variant]
                ax.errorbar(s['ratio'],i+offset,
                            xerr=[[s['ratio']-s['ratio_min']],[s['ratio_max']-s['ratio']]],
                            fmt='o',color=color,capsize=3,markersize=5,
                            markerfacecolor='white' if s['decision']=='inconclusive' else color,
                            label=spec['names'][str(variant)] if i==0 else None)
        ax.axvline(1,color='#333333',linewidth=1)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}×'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda value,_:''))
        ax.grid(axis='x',alpha=.18)
        ax.set_title('Hot call' if layers==1 else 'Ring24 per call')
        ax.set_xlabel(boundary+' time / paired '+('8x16 MMA control' if isolated else (spec['names'][str(spec['control'])] if split_combined else 'integrated control'))+' · log scale')
    data = [s for s in summary if s['candidate'] in variants]
    axes[0].set_xlim(min(s['ratio_min'] for s in data)/1.2,max(1,max(s['ratio_max'] for s in data))*1.2)
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis()
    handles,names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.93),ncol=len(variants),frameon=False)
    title = ('Qwen Wo · MMA tile ownership' if isolated else
             'Qwen attention · '+('QKV + Wo' if prefix == 'combined_' else ('QKV' if prefix == 'tiles_qkv_' else 'Wo'))+' tile ownership') if tiles else (
             'Qwen attention · KV split domain' if prefix == 'split_domain_' else 'Qwen attention · GQA work distribution')
    if split_combined:
        mechanism = ('Split8 fixed; both projection tiles change.\n' if spec['control'] == 13
                     else 'Both projections fixed; split8 and merge added.\n')
    elif isolated:
        mechanism = 'Same frozen attention input; Wo only.\n'
    elif tiles:
        mechanism = ('GQA fixed; both projections use 16x16.\n' if prefix == 'combined_'
                     else 'GQA fixed; one projection changes.\n')
    else:
        mechanism = 'QKV/Wo fixed; merge included.\n'
    if split_combined:
        title = 'Qwen attention · '+('projection gain with split8' if spec['control'] == 13 else 'split8 gain with 16x16 projections')
    fig.suptitle(title+(' · screen' if record['study'].endswith('_screen') else ''),
                 fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Left of 1× is faster. Whiskers: four-block ratio range; open marks: inconclusive.\n'
             'Gain rule: all four blocks faster and > max(5%, matching self-pair deviation). '
             +mechanism+
             f'{len(samples):,} retained observations · {record["runtime"]["device"]} / Metal · BF16 I/O, FP32 '+('accumulation.\n' if isolated else 'attention.\n')+
             f'Source {record["repository"]["commit"][:7]}. Fixed cache prefix; allocation and correctness checks excluded.',
             fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.18,1,.86))
    fig.savefig(figure_directory(directory)/(prefix.rstrip('_')+'.png'),dpi=160)
    plt.close(fig)


def stage_panels(axes, rows, grid, stages, variants, *, height=.34,
                 stage_order=False, ticks=None, invert=False, decode=False):
    """Draw the same measured stages for each declared shape and variant."""
    for ax, (r,t) in zip(axes, grid):
        for variant, color, offset, label in variants:
            values = [s for s in rows if (s['query_rows'],s['rows'],s['variant']) == (r,t,variant)]
            if stage_order:
                values.sort(key=lambda s: stages.index(s['stage']))
            ax.barh([stages.index(s['stage'])+offset for s in values],
                    [s['median_us'] for s in values], height=height, color=color, label=label)
        ax.set_xscale('log')
        if ticks is not None:
            ax.set_yticks(range(len(stages)), stages, fontsize=ticks)
        if invert:
            ax.invert_yaxis()
        ax.set_title(f'Decode T={t}' if decode else f'R={r}, T={t}')
        ax.set_xlabel('Median active GPU time (µs) · log scale')
        ax.grid(axis='x',alpha=.15)


def render_sublayer_combined_profile(directory):
    rows=load_profile(directory,'combined_')
    table(directory,'combined_profile_summary.csv',rows)
    record=json.loads((directory/'combined_profiles.json').read_text())
    from .attention_sublayer_contract import STAGES_BY_VARIANT
    stages=STAGES_BY_VARIANT[9]
    fig,axes=plt.subplots(2,2,figsize=(14,11),sharex=True,sharey=True)
    stage_panels(axes.flat, rows, record['specification']['workloads'], stages,
                 ((9,'#777777',-.18,'8x16 projections'),(18,'#167d9a',.18,'16x16 QKV + Wo')))
    axes[0,0].set_yticks(range(len(stages)),stages,fontsize=9)
    axes[0,0].invert_yaxis()
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.suptitle('Combined projections · where does attention time remain?',fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Separate single captures; active durations exclude host and preemption gaps. Stage medians are not whole-block latency.\n'
             'GQA is fixed; use paired unprofiled measurements for speed claims. Optional counter analysis is absent.\n'
             f'{sum(s["count"] for s in rows):,} retained durations · source {record["common"]["repository"]["commit"][:7]} · '
             f'{record["captures"][0]["capture"]["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.9))
    fig.savefig(figure_directory(directory)/'combined_profile.png',dpi=160)
    plt.close(fig)


def render_sublayer_timing(directory):
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharex=True,sharey=True)
    count=0
    for prefix,offset,color,label in (('timing_',-.12,'#777777','Print each sample'),
                                      ('timing_buffered_',.12,'#167d9a','Print after both arms')):
        record,samples,summary=load_run(directory,prefix)
        table(directory,prefix+'summary.csv',summary)
        count+=len(samples)
        labels=[f'R={w["query_rows"]}, T={w["rows"]}' for w in record['specification']['workloads']]
        for ax,layers in zip(axes,(1,24)):
            for i,row in enumerate(s for s in summary if s['layers']==layers):
                ax.errorbar(row['ratio'],i+offset,
                    xerr=[[row['ratio']-row['ratio_min']],[row['ratio_max']-row['ratio']]],
                    fmt='o',capsize=3,color=color,label=label if i==0 else None)
    for ax,layers in zip(axes,(1,24)):
        ax.axvline(1,color='#333333',linewidth=1)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}×'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v,_:''))
        ax.set_title('Hot call' if layers==1 else 'Ring24 per call')
        ax.set_xlabel('Identical-kernel paired time ratio · log scale')
        ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis()
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
    fig.suptitle('Attention timing · sensitivity to sample printing',fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Control self-pairs only; 1× means agreement. Whiskers show the four-block range, not a confidence interval.\n'
             'Samples retain enqueue-through-completion timing; only emission changes. These are calibration diagnostics, not gains.\n'
             f'{count:,} retained observations · {record["runtime"]["device"]} / Metal · source {record["repository"]["commit"][:7]}.',fontsize=9)
    fig.tight_layout(rect=(0,.18,1,.84))
    fig.savefig(figure_directory(directory)/'timing.png',dpi=160)
    plt.close(fig)


def render_sublayer_parallelism_profile(directory):
    rows = load_profile(directory,'parallelism_')
    record = json.loads((directory/'parallelism_profiles.json').read_text())
    table(directory,'parallelism_profile_summary.csv',rows)
    from .attention_sublayer_contract import STAGES_BY_VARIANT
    from .study import PARALLELISM_NAMES
    variants = record['specification']['variants']
    stages = STAGES_BY_VARIANT[9][:-2]+['FP32 GQA split','FP32 GQA merge']+STAGES_BY_VARIANT[9][-2:]
    colors = {9:'#777777',10:'#167d9a',11:'#3c9e79',12:'#8064a2',13:'#ba7437'}
    fig,axes = plt.subplots(1,2,figsize=(13,7),sharey=True)
    stage_panels(axes, rows, record['specification']['workloads'], stages,
                 [(v,colors[v],(i-(len(variants)-1)/2)*.23,PARALLELISM_NAMES[v])
                  for i,v in enumerate(variants)], height=.22, stage_order=True, ticks=9)
    axes[0].invert_yaxis()
    handles,names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.93),ncol=len(variants),frameon=False)
    fig.suptitle('Integrated attention · GQA work and merge cost',fontsize=16,fontweight='bold')
    fig.text(.04,.025,'QKV/Wo fixed; only GQA work distribution changes. Missing bars are stages that do not execute.\n'
             'Separate instrumented captures; active durations exclude preemption and host gaps. Use paired latency for gains.\n'
             f'{sum(s["count"] for s in rows):,} measured dispatch durations · '
             f'{record["captures"][0]["capture"]["runtime"]["device"]} / Metal · FP32 attention · '
             f'source {record["common"]["repository"]["commit"][:7]}.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.14,1,.86))
    fig.savefig(figure_directory(directory)/'parallelism_profile.png',dpi=160)
    plt.close(fig)


def render_sublayer_integrated_profile(directory):
    rows = load_profile(directory, 'integrated_')
    record = json.loads((directory/'integrated_profiles.json').read_text())
    table(directory, 'integrated_profile_summary.csv', rows)
    from .attention_sublayer_contract import STAGES_BY_VARIANT, PROFILE_WORKLOADS
    stages = STAGES_BY_VARIANT[8][:4] + STAGES_BY_VARIANT[9][1:]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharey=True)
    stage_panels(axes.flat, rows, PROFILE_WORKLOADS, stages,
                 ((8,'#777777',-.18,'Separate QKV'),(9,'#167d9a',.18,'Integrated packed QKV')),
                 stage_order=True, ticks=9)
    axes[0,0].invert_yaxis()
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Integrated attention · where does the time go now?',fontsize=16,fontweight='bold')
    fig.text(.04,.025,'GQA and Wo policy fixed; QKV packing/tiling and the layout copy change. Missing bars are stages that do not execute.\n'
             'Separate single captures; active durations exclude preemption and host gaps. Use paired latency for speed claims.\n'
             f'{sum(s["count"] for s in rows):,} measured dispatch durations · optional counter analysis absent.\n'
             f'{record["captures"][0]["capture"]["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention · '
             f'source {record["common"]["repository"]["commit"][:7]}.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.12,1,.95))
    fig.savefig(figure_directory(directory)/'integrated_profile.png',dpi=160)
    plt.close(fig)


def render_sublayer_wo_profile(directory):
    rows = load_profile(directory,'wo_')
    record = json.loads((directory/'wo_profiles.json').read_text())
    table(directory,'wo_profile_summary.csv',rows)
    from .attention_sublayer_contract import PROFILE_WORKLOADS, STAGES
    fig, axes = plt.subplots(2,2,figsize=(13,10))
    stage_panels(axes.flat, rows, PROFILE_WORKLOADS, STAGES,
                 ((3,'#777777',-.18,'Rowwise Wo'),(4,'#167d9a',.18,'MMA Wo')),
                 stage_order=True, ticks=9, invert=True)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Wo experiment · which stage changed?',fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Separate single captures; active dispatch durations exclude preemption and host gaps.\n'
             'These stage medians are diagnostic and are not added to construct whole-block latency.\n'
             f'{sum(s["count"] for s in rows):,} measured dispatch durations. Counter tables were not analyzed; absence is not zero.\n'
             f'{record["captures"][0]["capture"]["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention. '
             f'Source {record["common"]["repository"]["commit"][:7]}.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.12,1,.94))
    fig.savefig(figure_directory(directory)/'wo_profile.png',dpi=160)
    plt.close(fig)


def render_sublayer_decode(directory, prefix):
    """Paired whole-block comparison with rowwise Wo fixed in every arm."""
    record,samples,summary = load_run(directory,prefix)
    table(directory,prefix+'summary.csv',summary)
    lengths = [w['rows'] for w in record['specification']['workloads']]
    fig,axes = plt.subplots(1,2,figsize=(12,max(5.5,.65*len(lengths)+2.4)),sharex=True,sharey=True)
    for ax,layers in zip(axes,(1,24)):
        for variant,color,offset,label in ((5,'#167d9a',-.15,'FP32 G32'),
                                          (6,'#8064a2',.15,'FP32 split64 H4')):
            values = [s for s in summary if s['candidate']==variant and s['layers']==layers]
            for i,s in enumerate(values):
                ax.errorbar(s['ratio'],i+offset,
                            xerr=[[s['ratio']-s['ratio_min']],[s['ratio_max']-s['ratio']]],
                            fmt='o',color=color,capsize=3,markersize=6,
                            markerfacecolor='white' if s['decision']=='inconclusive' else color,
                            label=label if i==0 else None)
        ax.axvline(1,color='#333333',linewidth=1)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}×'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda value,_:''))
        ax.grid(axis='x',alpha=.18)
        ax.set_title('Hot call' if layers==1 else 'Ring24 per call')
        ax.set_xlabel('Whole-block time / paired materialized control · log scale')
    candidates = [s for s in summary if s['candidate'] in (5,6)]
    axes[0].set_xlim(min(s['ratio_min'] for s in candidates)/1.2,
                     max(1,max(s['ratio_max'] for s in candidates))*1.2)
    axes[0].set_yticks(range(len(lengths)),[f'Decode T={t}' for t in lengths])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False,fontsize=10)
    screen = record['study'].endswith('_screen')
    fig.suptitle('Qwen attention · FP32 decode ownership'+(' · screen' if screen else ''),
                 fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Left of 1× is faster. Whiskers: four-block ratio range; open marks: inconclusive.\n'
             'A gain requires all four blocks faster and > max(5%, matching self-pair deviation).\n'
             f'{len(samples):,} retained observations · {record["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention.\n'
             f'Source {record["repository"]["commit"][:7]}. Rowwise Wo fixed; cache prefix fixed; allocation excluded.',
             fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.2,1,.94))
    fig.savefig(figure_directory(directory)/(prefix.rstrip('_')+'.png'),dpi=160)
    plt.close(fig)


def render_sublayer_decode_profile(directory):
    rows = load_profile(directory,'decode_')
    table(directory,'decode_profile_summary.csv',rows)
    record = json.loads((directory/'decode_profiles.json').read_text())
    from .attention_sublayer_contract import STAGES, DECODE_PROFILE_WORKLOADS
    stages = STAGES[:10]+['GQA G32','GQA split','GQA merge']+STAGES[-2:]
    fig,axes = plt.subplots(1,2,figsize=(13,9),sharey=True)
    stage_panels(axes, rows, DECODE_PROFILE_WORKLOADS, stages,
                 ((3,'#777777',-.23,'Materialized FP32'),(5,'#167d9a',0,'FP32 G32'),
                  (6,'#8064a2',.23,'FP32 split64 H4')), height=.21, decode=True)
    axes[0].set_yticks(range(len(stages)),stages,fontsize=10)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('FP32 decode · where does attention time move?',fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Separate single captures; active dispatch durations exclude preemption and host gaps.\n'
             'A missing bar means that variant does not execute that stage. Stage medians are not whole-block latency.\n'
             f'{sum(s["count"] for s in rows):,} durations. Unchanged stages vary across captures; use paired latency for speed claims.\n'
             f'{record["captures"][0]["capture"]["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention, rowwise Wo. '
             f'Source {record["common"]["repository"]["commit"][:7]}.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.15,1,.94))
    fig.savefig(figure_directory(directory)/'decode_profile.png',dpi=160)
    plt.close(fig)


def render_sublayer_prefill_profile(directory):
    rows = load_profile(directory,'prefill_')
    table(directory,'prefill_profile_summary.csv',rows)
    record = json.loads((directory/'prefill_profiles.json').read_text())
    from .attention_sublayer_contract import STAGES, PREFILL_PROFILE_WORKLOADS
    stages = STAGES[:10]+['GQA FP32 MMA']+STAGES[-2:]
    fig,axes = plt.subplots(1,3,figsize=(15,8),sharey=True)
    stage_panels(axes, rows, PREFILL_PROFILE_WORKLOADS, stages,
                 ((4,'#777777',-.18,'Materialized FP32'),(7,'#167d9a',.18,'FP32 rolled MMA')))
    axes[0].set_yticks(range(len(stages)),stages,fontsize=10)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('FP32 prefill · which work remains after tiling?',fontsize=16,fontweight='bold')
    fig.text(.04,.025,'Separate single captures; active durations exclude preemption and host gaps. Use paired latency for speed claims.\n'
             'A missing bar means that variant does not execute that stage. Stage medians are not whole-block latency.\n'
             f'{sum(s["count"] for s in rows):,} measured dispatch durations. Optional counter analysis is absent.\n'
             f'{record["captures"][0]["capture"]["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention, MMA Wo fixed. '
             f'Source {record["common"]["repository"]["commit"][:7]}.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.16,1,.94))
    fig.savefig(figure_directory(directory)/'prefill_profile.png',dpi=160)
    plt.close(fig)


def render_sublayer(directory, record, samples, summary):
    """Separate latency scaling from diagnostic per-dispatch active time."""
    prefill_style()
    table(directory, 'summary.csv', summary)
    def regime(s):
        return 0 if s['query_rows'] == 1 else (1 if s['query_rows'] == s['rows'] else 2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.3))
    for group, ax in enumerate(axes):
        for layers, color, label in ((1, '#167d9a', 'Hot'), (24, '#c75b39', 'Ring24 per call')):
            data = [s for s in summary if s['layers'] == layers and regime(s) == group]
            x = [s['rows'] for s in data] if group != 2 else list(range(len(data)))
            ax.plot(x, [s['control_us'] for s in data], '-o', color=color, label=label, markersize=4)
            noisy = [(position, s) for position, s in zip(x, data) if s['noise_floor'] > .05]
            ax.scatter([position for position, _ in noisy], [s['control_us'] for _, s in noisy],
                       s=22, facecolors='white', edgecolors=color, zorder=3)
        if group != 2:
            ax.set_xscale('log', base=2)
            ax.set_xticks(x, [str(v) for v in x], rotation=35)
        else:
            ax.set_xticks(x, [f'R={s["query_rows"]}\nT={s["rows"]}' for s in data])
        ax.set_yscale('log')
        ax.set_ylabel('Latency (µs / sublayer) · log scale')
        ax.set_title(('Decode · R=1', 'Full prefill · R=T', 'Chunked prefill')[group])
        ax.set_xlabel('Total KV positions T' if group != 2 else 'New rows R and total positions T')
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False)
    fig.suptitle('Qwen attention sublayer · complete enqueue through completion', fontsize=16, fontweight='bold')
    fig.text(.04,.025, f'{record["runtime"]["device"]} / Metal · BF16 I/O, FP32 attention intermediates · source {record["repository"]["commit"][:7]}.\n'
             f'{len(samples):,} retained samples, four self-paired blocks; control-arm medians shown. No optimized comparison.\n'
             'Ring24 uses distinct weights, inputs and caches with shared scratch, and one synchronization per sweep.\n'
             'Open marks: self-pair deviation exceeds 5%; see the complete calibration ranges in summary.csv.', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,.18,1,.91))
    fig.savefig(figure_directory(directory) / 'latency.png', dpi=160)
    plt.close(fig)
    if (directory / 'profiles.json').exists():
        rows = load_profile(directory)
        table(directory, 'profile_summary.csv', rows)
        from .attention_sublayer_contract import PROFILE_WORKLOADS, STAGES
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
        for ax, (r,t) in zip(axes.flat, PROFILE_WORKLOADS):
            lookup = {s['stage']:s for s in rows if s['query_rows']==r and s['rows']==t}
            data = [lookup[stage] for stage in STAGES]
            values = [s['median_us'] for s in data]
            ax.barh(range(len(data)), values,
                    color=['#c75b39' if s['stage'] in ('QK','softmax','PV') else '#167d9a' for s in data])
            ax.set_yticks(range(len(data)), STAGES, fontsize=9)
            ax.invert_yaxis()
            ax.set_xscale('log')
            ax.set_xlim(min(values)/2, max(values)*3)
            ax.set_title(f'R={r}, T={t}')
            ax.set_xlabel('Median active GPU time (µs) · log scale')
            for i, value in enumerate(values):
                ax.text(value*1.05,i,f'{value:.2f}',va='center',fontsize=8)
            ax.grid(axis='x', alpha=.15)
        fig.suptitle('Attention sublayer · time spent in each kernel',fontsize=16,fontweight='bold')
        fig.text(.04,.023,'Orange: GQA stages. Blue: surrounding stages. Separate instrumented captures; no CPU or inter-dispatch gaps.\n'
                 'Stage labels follow validated enqueue order. Segments of a preempted dispatch are joined before labeling.\n'
                 f'{sum(s["count"] for s in rows):,} dispatch durations retained. These stage medians are not added to construct whole-sublayer latency.',fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.13,1,.94))
        fig.savefig(figure_directory(directory) / 'profile.png',dpi=160)
        plt.close(fig)
    for prefix in ('wo_screen_','wo_'):
        if (directory/(prefix+'run.json')).exists():
            render_sublayer_wo(directory,prefix)
    if (directory/'wo_profiles.json').exists():
        render_sublayer_wo_profile(directory)
    for prefix in ('decode_screen_','decode_'):
        if (directory/(prefix+'run.json')).exists():
            render_sublayer_decode(directory,prefix)
    if (directory/'decode_profiles.json').exists():
        render_sublayer_decode_profile(directory)
    for prefix in ('prefill_screen_','prefill_'):
        if (directory/(prefix+'run.json')).exists():
            render_sublayer_wo(directory,prefix,fp32_prefill=True)
    if (directory/'prefill_profiles.json').exists():
        render_sublayer_prefill_profile(directory)
    for prefix in ('projections_','integrated_'):
        if (directory/(prefix+'run.json')).exists():
            render_sublayer_wo(directory,prefix,integration=True)
    if (directory/'integrated_profiles.json').exists():
        render_sublayer_integrated_profile(directory)
    for prefix in ('parallelism_screen_','parallelism_','tiles_screen_','tiles_kernel_screen_','tiles_','tiles_qkv_','split_domain_','combined_','split_combined_projections_','split_combined_gqa_'):
        if (directory/(prefix+'run.json')).exists():
            render_sublayer_parallelism(directory,prefix)
    if (directory/'timing_run.json').exists() and (directory/'timing_buffered_run.json').exists():
        render_sublayer_timing(directory)
    if (directory/'parallelism_profiles.json').exists():
        render_sublayer_parallelism_profile(directory)
    if (directory/'combined_profiles.json').exists():
        render_sublayer_combined_profile(directory)
    print(directory.name, len(samples), 'baseline observations verified; attention figures and retained comparisons regenerated')


def render(directory):
    from .study import evidence_directory
    directory = evidence_directory(directory)
    record, samples, summary = load_run(directory)
    spec = record['specification']
    if spec['operation'] == 'attention_sublayer':
        return render_sublayer(directory,record,samples,summary)
    if record['study'] in ('gqa_prefill_screen','gqa_prefill_resources_screen'):
        prefill_style()
        table(directory,'screen_summary.csv',summary)
        return prefill_screen(directory,record,samples,summary,
                              resources=record['study']=='gqa_prefill_resources_screen')
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
    fig.savefig(figure_directory(directory) / 'latency.png', dpi=160)
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
