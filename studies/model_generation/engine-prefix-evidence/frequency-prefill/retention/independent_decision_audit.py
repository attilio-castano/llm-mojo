#!/usr/bin/env python3
"""Independent stdlib audit of the compact frequency report; never loads native/source/assets."""
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
PRIMARY = 'prefix-1024-reuse-8-blocks-40-suffix-64-out-16-requests-8-first'
CELLS = [(4,384,64,16,'first',8),(8,384,64,16,'first',8),(4,40,64,16,'first',8),
         (8,40,64,16,'first',8),(0,384,64,16,'first',8),(8,41,64,16,'first',8),
         (8,56,64,16,'first',8),(8,36,33,16,'first',8),(8,40,63,16,'first',8),
         (8,40,65,16,'first',8),(8,40,64,2,'first',8),(4,40,64,16,'even',8),
         (5,40,64,16,'first',8),(6,40,64,16,'first',8),(7,40,64,16,'first',8),
         (7,40,64,16,'first',7),(9,40,64,16,'first',9),(4,40,64,3,'first',8),
         (8,40,64,8,'first',8),(8,40,64,32,'first',8)]
ARMS = ('current','self_current','adaptive','frequency')
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

def no_harm(metric):
    status='inconclusive'; buffer=None
    if metric['available']:
        median=metric['median_ratio']; noise=metric['noise_floor']
        buffer=[median-noise,median+noise]
        if median<=1.05 and median+noise<=1.10: status='pass'
        elif min(metric['block_ratios'])>1 and median-1>noise: status='fail'
    return dict(status=status,median_limit=1.05,buffer_limit=1.10,buffer=buffer)


def audit_observations(cell,geometry,by_arm):
    """Check retained within-run group/count arithmetic, without claiming raw proof."""
    reuse,pool,suffix,maximum,order,count=geometry
    expected={(b,p,c) for b in range(4) for p,c in [('current',False),('current',True),('adaptive',False),('frequency',False)]}
    observations=cell['observations']
    same(len(observations),16,cell['cell_id']+'/observation census')
    observed={(o['block'],o['prefill_policy'],o['calibration']) for o in observations}
    if observed!=expected: raise AssertionError('observation policy/block census changed')
    lookup={}
    for observation in observations:
        block=observation['block']; policy=observation['prefill_policy']; calibration=observation['calibration']
        arm='self_current' if calibration else policy; run=by_arm[arm][block]
        requests=observation['requests']; ids={r['request_id'] for r in requests}
        same(sorted(ids),list(range(count)),cell['cell_id']+'/request IDs')
        same(sum(r['gap_count'] for r in requests),run['gaps'],cell['cell_id']+'/request gap counts')
        same(sum(r['delivered_tokens'] for r in requests),run['delivered_tokens'],cell['cell_id']+'/request deliveries')
        for request in requests:
            same(request['gap_count'],max(request['delivered_tokens']-1,0),cell['cell_id']+'/request consecutive gaps')
            if request['gap_count'] and not 0<=request['gap_p95_ns']<=request['gap_maximum_ns']:
                raise AssertionError('request tail bounds changed')
        actual={r['request_id'] for r in requests if r['cached_prompt_tokens']>0}
        intended={i for i in ids if i%2==0} if order=='even' else set(range(reuse))
        for name,members in [('actual_hit',actual),('actual_miss',ids-actual),('intended_match',intended),('intended_miss',ids-intended)]:
            group=observation['groups'][name]; selected=[r for r in requests if r['request_id'] in members]
            same(group['request_ids'],sorted(members),cell['cell_id']+'/'+name+'/IDs')
            same(group['gap_count'],sum(r['gap_count'] for r in selected),cell['cell_id']+'/'+name+'/gap count')
            same(group['gaps']['count'],group['gap_count'],cell['cell_id']+'/'+name+'/distribution count')
            same(group['delivered_tokens'],sum(r['delivered_tokens'] for r in selected),cell['cell_id']+'/'+name+'/delivered')
            maxima=[r['gap_maximum_ns'] for r in selected if r['gap_maximum_ns'] is not None]
            percentiles=[r['gap_p95_ns'] for r in selected if r['gap_p95_ns'] is not None]
            same(group['maximum_request_gap_ns'],max(maxima,default=None),cell['cell_id']+'/'+name+'/maximum request')
            same(group['maximum_request_p95_ns'],max(percentiles,default=None),cell['cell_id']+'/'+name+'/worst request p95')
            same(group['gaps']['maximum_ns'],max(maxima,default=None),cell['cell_id']+'/'+name+'/group maximum')
        forecast=observation['frequency_summary']
        if policy!='frequency':
            same(forecast['snapshots'],[],cell['cell_id']+'/control no forecast')
            same(forecast['forecast_ns'],0,cell['cell_id']+'/control no forecast work')
        else:
            same(len(forecast['snapshots']),1,cell['cell_id']+'/one frozen busy period')
            snapshot=forecast['snapshots'][0]
            same(snapshot['request_ids'],list(range(count)),cell['cell_id']+'/frozen arrived IDs')
            same(snapshot['tickets'],list(range(1,count+1)),cell['cell_id']+'/immutable tickets')
            same(snapshot['maximum_outputs'],[maximum]*count,cell['cell_id']+'/declared limits')
            same(snapshot['prompt_lengths'],[1024+suffix]*count,cell['cell_id']+'/prompt lengths')
            n=sum(max(g-1,0) for g in snapshot['maximum_outputs'])
            m=max(len(snapshot['run_request_ids'])-1,0); threshold=n-19*(n-1)//20 if n else 0
            same(snapshot['gap_budget'],n,cell['cell_id']+'/frozen N')
            same(snapshot['predicted_interruptions'],m,cell['cell_id']+'/frozen M')
            same(snapshot['threshold'],threshold,cell['cell_id']+'/exact integer T')
            same(snapshot['dense'],snapshot['valid'] and m>=threshold,cell['cell_id']+'/density')
            same(snapshot['enabled'],snapshot['valid'] and snapshot['dense'],cell['cell_id']+'/forecast permission')
        lookup[block,policy,calibration]=observation
    for comparison in cell['forecast_vs_current']:
        block=comparison['block']; current=lookup[block,'current',False]; candidate=lookup[block,'frequency',False]
        snapshot=candidate['frequency_summary']['snapshots'][0]
        for key in ('cohort_id','gap_budget','predicted_interruptions','threshold','valid','dense','reason'):
            same(comparison[key],snapshot[key],cell['cell_id']+'/forecast comparison/'+key)
        same(comparison['actual_current_gaps'],current['gap_tail']['count'],cell['cell_id']+'/current N')
        same(comparison['actual_current_mixed_gaps'],current['gap_tail']['attribution']['mixed']['count'],cell['cell_id']+'/current mixed gaps')
        same(comparison['actual_frequency_gaps'],candidate['gap_tail']['count'],cell['cell_id']+'/candidate N')
        same(comparison['actual_frequency_mixed_gaps'],candidate['gap_tail']['attribution']['mixed']['count'],cell['cell_id']+'/candidate mixed gaps')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',type=Path); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); raw=args.input.read_bytes(); report=json.loads(raw)
    same(report['measured_runs'],320,'timed census'); same(report['untimed_preflights'],60,'preflight census')
    expected_ids=[f'prefix-1024-reuse-{r}-blocks-{b}-suffix-{s}-out-{g}-requests-{q}-{o}' for r,b,s,g,o,q in CELLS]
    same([c['cell_id'] for c in report['cells']],expected_ids,'cell order')
    all_cells=[]; close=[]; p95_close=[]; total_runs=0
    for cell,geometry in zip(report['cells'],CELLS):
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
        audit_observations(cell,geometry,by_arm)
        recomputed={}; outcomes={}
        # Also check descriptive old-adaptive values while staying independent of production code.
        for arm in ('adaptive','frequency'):
            metrics={}
            for key,field in METRICS.items():
                vectors=[[r[field] for r in by_arm[a]] for a in ('current','self_current',arm)]
                metrics[key]=ratios(*vectors)
                same(cell['comparisons'][arm][key],metrics[key],name+'/'+arm+'/'+key)
            recomputed[arm]=metrics; outcomes[arm]=decisions(metrics)
            outcomes[arm]['p95_no_harm']=no_harm(metrics['p95_gap_ns'])
            same(cell['decisions'][arm],outcomes[arm],name+'/'+arm+'/decision')
        for label,key in (('completion','duration_ns'),('p95_ttft','p95_ttft_ns')):
            m=recomputed['frequency'][key]
            if not m['available']: continue
            close.append(dict(cell_id=name,guard=label,median_ratio=m['median_ratio'],noise=m['noise_floor'],
                              upper=m['median_ratio']+m['noise_floor'],slack_to_1_10=1.10-m['median_ratio']-m['noise_floor']))
        m=recomputed['frequency']['maximum_gap_ns']
        if m['available']:
            close.append(dict(cell_id=name,guard='worst_gap',median_ratio=m['median_ratio'],noise=m['noise_floor'],
                              upper=None,slack_to_1_plus_noise=1+m['noise_floor']-m['median_ratio']))
        m=recomputed['frequency']['p95_gap_ns']
        if m['available']:
            p95_close.append(dict(cell_id=name,median_ratio=m['median_ratio'],noise=m['noise_floor'],
                median_slack=1.05-m['median_ratio'],buffer_slack=1.10-m['median_ratio']-m['noise_floor']))
        all_cells.append(dict(cell_id=name,comparisons=recomputed,decisions=outcomes,
            medians={a:{field:(statistics.median([r[field] for r in by_arm[a]]) if all(r[field] is not None for r in by_arm[a]) else None) for field in
                ('duration_ms','episode_ms','ttft_p95_ms','gap_p95_ms','gap_maximum_ms','steps','gaps','delivered_tokens',
                 'gaps_above_20ms','gaps_above_50ms','computed_rows','reused_prefix_tokens','attended_positions')} for a in ARMS}))
    same(total_runs,320,'actual run census')
    primary=next(c for c in all_cells if c['cell_id']==PRIMARY)
    same(report['usefulness']['primary_cell'],PRIMARY,'primary identity')
    same(report['usefulness']['primary'],primary['decisions']['frequency'],'primary decision')
    broad=all(c['decisions']['frequency']['guards_pass'] for c in all_cells)
    same(report['usefulness']['broad_guard_pass'],broad,'broad guard verdict')
    broad_p95=all(c['decisions']['frequency']['p95_no_harm']['status']=='pass' for c in all_cells)
    study_success=primary['decisions']['frequency']['hypothesis']=='supported' and broad and broad_p95
    same(report['usefulness']['broad_p95_guard_pass'],broad_p95,'broad p95 verdict')
    same(report['usefulness']['study_success'],study_success,'overall study success')
    same(report['usefulness']['primary_policy'],'frequency','primary policy')
    same(report['promotion'],False,'no promotion')
    result=dict(kind='engine-prefix-frequency-independent-decision-audit-v1',passed=True,
        input=dict(path=str(args.input),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        method='Standalone Python standard library; recomputed all ratios/noise/decisions from compact per-run millisecond values. No production helper import.',
        scope='Validates compact arithmetic, census, medians, aggregate work, within-run group/count consistency and recorded forecast arithmetic. Group quantiles and raw per-request identity/publication/native/source provenance remain the separate raw audit.',
        checked_comparisons=200,checked_decisions=40,checked_p95_no_harm=40,measured_runs=total_runs,untimed_preflights_declared=60,
        recursive_checks=checks,primary_cell=PRIMARY,primary=primary,
        broad_guard_pass=broad,broad_p95_guard_pass=broad_p95,study_success=study_success,all_twenty_frequency_guards_pass=broad,
        closest_frequency_cost_guards=sorted((c for c in close if c['guard']!='worst_gap'),key=lambda c:c['slack_to_1_10'])[:6],
        closest_frequency_p95_guards=sorted(p95_close,key=lambda c:min(c['median_slack'],c['buffer_slack']))[:6],
        cells=all_cells)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS: compact arithmetic agrees for 320 runs, 200 metric comparisons, 40 decisions and 40 p95-no-harm guards.')
    print('broad guards:',broad,'broad p95:',broad_p95,'study success:',study_success)
    print('Primary frequency/current:')
    for key,m in primary['comparisons']['frequency'].items():
        print(f"  {key}: median={m['median_ratio']}, noise={m['noise_floor']}, ratios={m['block_ratios']}")
    print('Closest frequency cost guards:')
    for c in result['closest_frequency_cost_guards'][:3]: print(c)

if __name__=='__main__': main()
