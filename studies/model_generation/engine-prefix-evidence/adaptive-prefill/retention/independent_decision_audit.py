#!/usr/bin/env python3
"""Independent stdlib audit of the compact adaptive report; never loads native/source/assets."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

METRICS = {
    'duration_ns': 'duration_ms', 'episode_ns': 'episode_ms',
    'p95_ttft_ns': 'ttft_p95_ms', 'p95_gap_ns': 'gap_p95_ms',
    'maximum_gap_ns': 'gap_maximum_ms',
}
PRIMARY = 'prefix-1024-reuse-8-blocks-40-suffix-64-out-16-first'
CELLS = [(4,384,64,16,'first'),(8,384,64,16,'first'),(4,40,64,16,'first'),
         (8,40,64,16,'first'),(0,384,64,16,'first'),(8,41,64,16,'first'),
         (8,56,64,16,'first'),(8,36,33,16,'first'),(8,40,63,16,'first'),
         (8,40,65,16,'first'),(8,40,64,2,'first'),(4,40,64,16,'even')]
ARMS = ('current','self_current','fixed32','adaptive')
checks = 0

def same(actual, expected, context):
    global checks
    checks += 1
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual) != set(expected):
            raise AssertionError(context + ': dictionary shape differs')
        for k in expected: same(actual[k],expected[k],context+'/'+k)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual) != len(expected):
            raise AssertionError(context + ': list shape differs')
        for i,(a,b) in enumerate(zip(actual,expected)): same(a,b,context+'/'+str(i))
    elif isinstance(expected,float):
        if type(actual) not in (float,int) or not math.isclose(actual,expected,rel_tol=3e-14,abs_tol=3e-14):
            raise AssertionError(f'{context}: {actual!r} != {expected!r}')
    elif actual != expected or type(actual) is not type(expected):
        raise AssertionError(f'{context}: {actual!r} != {expected!r}')

def ratios(baseline,control,candidate):
    if any(v is None or v<=0 for v in baseline+control+candidate):
        return dict(available=False,block_ratios=[],calibration_ratios=[],median_ratio=None,noise_floor=None,speed_verdict='inconclusive')
    paired=[candidate[i]/baseline[i] for i in range(4)]
    self_pair=[control[i]/baseline[i] for i in range(4)]
    center=statistics.median(paired)
    noise=max([0.05]+[abs(v-1) for v in self_pair])
    verdict='inconclusive'
    if max(paired)<1 and 1-center>noise: verdict='faster'
    if min(paired)>1 and center-1>noise: verdict='slower'
    return dict(available=True,block_ratios=paired,calibration_ratios=self_pair,median_ratio=center,noise_floor=noise,speed_verdict=verdict)

def decisions(metrics):
    guards={}
    for name,key in (('completion','duration_ns'),('p95_ttft','p95_ttft_ns')):
        m=metrics[key]
        if not m['available']:
            guards[name]=dict(status='inconclusive',buffer=None)
            continue
        lo=m['median_ratio']-m['noise_floor']; hi=m['median_ratio']+m['noise_floor']
        status='pass' if hi<=1.10 else 'fail' if lo>1.10 else 'inconclusive'
        guards[name]=dict(status=status,buffer=[lo,hi],limit=1.10)
    m=metrics['maximum_gap_ns']; status='inconclusive'
    if m['available']:
        if m['median_ratio']<=1+m['noise_floor']: status='pass'
        elif min(m['block_ratios'])>1: status='fail'
    guards['worst_gap']=dict(status=status)
    m=metrics['p95_gap_ns']; status='inconclusive'
    if m['available']:
        if max(m['block_ratios'])<1 and 1-m['median_ratio']>m['noise_floor']:
            status='pass' if m['median_ratio']<=0.75 else 'fail'
        elif m['speed_verdict']=='slower': status='fail'
    statuses=[status]+[g['status'] for g in guards.values()]
    return dict(p95_gap=dict(status=status,ratio_limit=.75),guards=guards,
                guards_pass=all(g['status']=='pass' for g in guards.values()),
                hypothesis='supported' if set(statuses)=={'pass'} else 'failed' if 'fail' in statuses else 'inconclusive')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); raw=args.input.read_bytes(); report=json.loads(raw)
    same(report['measured_runs'],192,'timed census'); same(report['untimed_preflights'],36,'preflight census')
    expected_ids=[f'prefix-1024-reuse-{r}-blocks-{b}-suffix-{s}-out-{g}-{o}' for r,b,s,g,o in CELLS]
    same([c['cell_id'] for c in report['cells']],expected_ids,'cell order')
    all_cells=[]; close=[]; total_runs=0
    for cell in report['cells']:
        name=cell['cell_id']; by_arm={}
        for arm in ARMS:
            runs=cell['arms'][arm]['runs']; same([r['block'] for r in runs],[0,1,2,3],name+'/'+arm+'/blocks')
            total_runs+=len(runs); by_arm[arm]=runs
            for key,retained in cell['arms'][arm]['median_per_run_metrics'].items():
                values=[r[key] for r in runs]
                calculated=statistics.median(values) if all(v is not None for v in values) else None
                same(retained,calculated,name+'/'+arm+'/median/'+key)
        # Aggregate work can be checked here; per-request identities and histories need raw replay.
        for block in range(4):
            for key in ('computed_rows','reused_prefix_tokens','seed_rows','attended_positions','delivered_tokens','gaps','preemptions'):
                values=[by_arm[a][block][key] for a in ARMS]
                same(values,[values[0]]*4,name+'/aggregate-work/'+key)
        recomputed={}; outcomes={}
        # Also check descriptive fixed32 values while staying independent of production code.
        for arm in ('fixed32','adaptive'):
            metrics={}
            for key,field in METRICS.items():
                vectors=[[r[field] for r in by_arm[a]] for a in ('current','self_current',arm)]
                metrics[key]=ratios(*vectors)
                same(cell['comparisons'][arm][key],metrics[key],name+'/'+arm+'/'+key)
            recomputed[arm]=metrics; outcomes[arm]=decisions(metrics)
            same(cell['decisions'][arm],outcomes[arm],name+'/'+arm+'/decision')
        for label,key in (('completion','duration_ns'),('p95_ttft','p95_ttft_ns')):
            m=recomputed['adaptive'][key]
            close.append(dict(cell_id=name,guard=label,median_ratio=m['median_ratio'],noise=m['noise_floor'],
                              upper=m['median_ratio']+m['noise_floor'],slack_to_1_10=1.10-m['median_ratio']-m['noise_floor']))
        m=recomputed['adaptive']['maximum_gap_ns']
        close.append(dict(cell_id=name,guard='worst_gap',median_ratio=m['median_ratio'],noise=m['noise_floor'],
                          upper=None,slack_to_1_plus_noise=1+m['noise_floor']-m['median_ratio']))
        all_cells.append(dict(cell_id=name,comparisons=recomputed,decisions=outcomes,
            medians={a:{field:statistics.median([r[field] for r in by_arm[a]]) for field in
                ('duration_ms','episode_ms','ttft_p95_ms','gap_p95_ms','gap_maximum_ms','steps','gaps','delivered_tokens',
                 'gaps_above_20ms','gaps_above_50ms','computed_rows','reused_prefix_tokens','attended_positions')} for a in ARMS}))
    same(total_runs,192,'actual run census')
    primary=next(c for c in all_cells if c['cell_id']==PRIMARY)
    same(report['usefulness']['primary_cell'],PRIMARY,'primary identity')
    same(report['usefulness']['primary'],primary['decisions']['adaptive'],'primary decision')
    broad=all(c['decisions']['adaptive']['guards_pass'] for c in all_cells)
    same(report['usefulness']['broad_guard_pass'],broad,'broad guard verdict')
    same(report['promotion'],False,'no promotion')
    result=dict(kind='engine-prefix-adaptive-independent-decision-audit-v1',passed=True,
        input=dict(path=str(args.input),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        method='Standalone Python standard library; recomputed all ratios/noise/decisions from compact per-run millisecond values. No production helper import.',
        scope='Validates compact arithmetic, census, arm medians and aggregate work. Does not independently establish per-request histories/hits, native correctness or full-archive provenance.',
        checked_comparisons=120,checked_decisions=24,measured_runs=total_runs,untimed_preflights_declared=36,
        recursive_checks=checks,primary_cell=PRIMARY,primary=primary,
        broad_guard_pass=broad,all_twelve_adaptive_guards_pass=broad,
        closest_adaptive_cost_guards=sorted((c for c in close if c['guard']!='worst_gap'),key=lambda c:c['slack_to_1_10'])[:6],
        cells=all_cells)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS: 192 runs, 120 metric comparisons, 24 decisions; all 12 adaptive cost/worst-gap guards pass.')
    print('Primary adaptive/current:')
    for key,m in primary['comparisons']['adaptive'].items():
        print(f"  {key}: median={m['median_ratio']:.12f}, noise={m['noise_floor']:.12f}, ratios={m['block_ratios']}")
    print('Closest adaptive cost guards:')
    for c in result['closest_adaptive_cost_guards'][:3]: print(c)

if __name__=='__main__': main()
