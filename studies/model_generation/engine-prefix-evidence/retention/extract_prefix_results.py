#!/usr/bin/env python3
"""CPU-only compact extraction from the original completed prefix record."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from unittest import mock

from llm_mojo.benchmarks import engine_prefix as prefix
from llm_mojo.benchmarks import model_profile as profile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    record=json.loads(gzip.decompress(args.measurements.read_bytes()))
    with mock.patch.object(profile,'execute',side_effect=AssertionError('CPU extraction cannot execute native work')), \
         mock.patch.object(profile,'assets',side_effect=AssertionError('CPU extraction cannot read model assets')):
        summary=prefix.summary(record)
    assert summary==record['summary']
    result=dict(kind='engine-prefix-compact-results-v1',source_commit=record['build']['source']['repository']['commit'],
        source_dirty=record['build']['source']['repository']['dirty'],measurement=dict(path=str(args.measurements),sha256=sha(args.measurements)),
        environment=record['build']['environment'],declaration=record['declaration'],
        numerical_checks=summary['qualification']['checks'],measured_runs=summary['measured_runs'],untimed_preflights=summary['untimed_preflights'],
        same_delivered_histories=summary['same_delivered_histories'],cpu_summary_verified=True,
        definitions=dict(primary_arms='Four blocks per arm, excluding calibration self-controls; all twelve native calls per cell remain in the raw record.',
            comparisons='Exact original helper comparisons: enabled/disabled ratio within each block, median of four ratios; calibration determines noise floor and speed verdict.',
            displayed_medians='Median of four primary per-run statistics. TTFT and gap quantiles are medians of per-run quantiles, not quantiles of pooled requests.',
            gaps='Consecutive delivered token timestamps within each request; TTFT excluded. Maximum is computed from raw native delivery events.',
            duration_ns='Host phase-relative measure drain clock, including trace emission; excludes seeded preparation and final explicit cache clear.',
            episode_ns='Absolute host seed-begin to measure-end; includes seed and handoff, excludes ten-step warmup and final clear.',
            through_clear_ns='Descriptive only: absolute clear-end minus seed-begin, including the measure-to-clear gap and explicit cleanup. No new speed verdict.',
            seed_rows='Same cache-enabled seed in every arm; controls discard its index during handoff before measurement.',
            evictions='Removed index entries during measurement only; explicit handoff/final-clear removals counted separately.',
            admission_ns='Combined cache attach/reserve/evict/touch interval; not pure eviction time.',
            scope='One frozen bounded offline matrix on the qualified reference27 synchronous path; no capacity, SLO, async, Fast, or default-promotion claim.'),cells=[])
    for cell,rawcell in zip(summary['cells'],record['cells']):
        params=rawcell['trace']['prefix']
        out=dict(cell_id=cell['cell_id'],prefix_tokens=params['tokens'],reuse_requests=params['reuse_requests'],blocks=params['blocks'],
                 comparisons=cell['comparisons'],arms={})
        for cache in ('disabled','enabled'):
            runs=sorted([r for r in cell['runs'] if r['cache']==cache and not r['calibration']],key=lambda r:r['block'])
            rawruns=sorted([r for r in rawcell['runs'] if r['cache']==cache and not r['calibration']],key=lambda r:r['block'])
            assert len(runs)==len(rawruns)==4
            scalars=('duration_ns','episode_ns','seed_ns','handoff_ns','clear_ns','tokens_per_second','lookup_ns','publish_ns','cache_admission_ns')
            arm=dict(medians={key:statistics.median(r[key] for r in runs) for key in scalars},
                work={key:sorted({r[key] for r in runs}) for key in ('total_tokens','cache_hit_tokens','seed_rows','delivered_tokens','attended_positions','steps','evictions','peak_cached_blocks','peak_private_blocks','peak_references')},
                quantiles={key:{q:statistics.median(r[key][q] for r in runs) for q in ('p50_ns','p95_ns','p99_ns')} for key in ('ttft','tpot','inter_token')},
                clear_ns_range=[min(r['clear_ns'] for r in runs),max(r['clear_ns'] for r in runs)],cache_cleared=all(r['cache_cleared'] for r in runs))
            through=[]; gap_max=[]; cleanup_gap=[]; removals=[]; discarded=[]
            for run in rawruns:
                parsed=run['parsed']; times=parsed['phase_times']
                through.append(times['clear']['end_ns']-times['seed']['begin_ns'])
                cleanup_gap.append(times['clear']['begin_ns']-times['measure']['end_ns'])
                prior={}; gaps=[]
                for event in parsed['phases']['measure']['events']:
                    if event['kind']!='token': continue
                    request=event['request_id']
                    if request in prior: gaps.append(event['emitted_ns']-prior[request])
                    prior[request]=event['emitted_ns']
                gap_max.append(max(gaps,default=0))
                removals.append(sum(e['phase']=='clear' and e['kind']=='removed' for e in parsed['cache_events']))
                discarded.append(sum(e['phase']=='handoff' and e['kind']=='removed' for e in parsed['cache_events']))
            arm.update(through_clear_ns=through,through_clear_ns_median=statistics.median(through),
                cleanup_gap_ns_median=statistics.median(cleanup_gap),maximum_inter_token_gap_ns=gap_max,
                maximum_inter_token_gap_ns_median=statistics.median(gap_max),final_clear_removed_entries=sorted(set(removals)),handoff_removed_entries=sorted(set(discarded)))
            out['arms'][cache]=arm
        ratios=[a/b for a,b in zip(out['arms']['enabled']['through_clear_ns'],out['arms']['disabled']['through_clear_ns'])]
        out['through_clear_descriptive_comparison']=dict(block_ratios=ratios,median_ratio=statistics.median(ratios),speed_verdict=None)
        result['cells'].append(out)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('cell measured-ratio episode-ratio rows-disabled/enabled hits evictions clear-enabled-us verdict')
    for cell in result['cells']:
        disabled,enabled=cell['arms']['disabled'],cell['arms']['enabled']
        print(cell['cell_id'],f"{cell['comparisons']['duration_ns']['median_ratio']:.9f}",
            f"{cell['comparisons']['episode_ns']['median_ratio']:.9f}",str(disabled['work']['total_tokens'])+'/'+str(enabled['work']['total_tokens']),
            enabled['work']['cache_hit_tokens'],enabled['work']['evictions'],enabled['medians']['clear_ns']/1000,
            cell['comparisons']['duration_ns']['speed_verdict'])


if __name__=='__main__': main()
