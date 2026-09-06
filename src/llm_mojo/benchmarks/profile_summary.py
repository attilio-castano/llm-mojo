"""Curate validated GQA captures into one profile record and raw dispatch table."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path

from .analyze_trace import (integer, read_table, segment_compute_commands, duration_summary, coalesce_compute_commands)
from .study import sha, write_json, PREFILL_PROFILE_WORKLOADS, prefill_profile_grid
from . import attention_sublayer_contract as sublayer

STAGES = {0: ['QK', 'softmax', 'PV'], 4: ['fused'], 9: ['decode', 'merge']}
COUNTERS = {'Kernel Occupancy', 'Instruction Throughput Limiter', 'Last Level Cache Limiter'}


def collect(source, output, prefill_variant=None, *, prefill_variants=None, prefix='', attention_sublayer=False, wo_comparison=False, decode_comparison=False, prefill_comparison=False):
    records, samples = [], []
    common = None
    if prefill_variant is not None and prefill_variants is not None:
        raise ValueError('choose one prefill comparison')
    if attention_sublayer and (prefill_variant is not None or prefill_variants is not None):
        raise ValueError('choose one attention profile study')
    if (wo_comparison or decode_comparison or prefill_comparison) and not attention_sublayer:
        raise ValueError('contained comparison requires the attention sublayer')
    if sum((wo_comparison, decode_comparison, prefill_comparison)) > 1:
        raise ValueError('choose one attention sublayer comparison')
    variants = ([3,4] if wo_comparison else ([3,5,6] if decode_comparison else [3])) if attention_sublayer else ([0,prefill_variant] if prefill_variant is not None else prefill_variants)
    if prefill_comparison:
        variants = [4,7]
    prefill = variants is not None
    grid = sublayer.PROFILE_WORKLOADS if attention_sublayer else PREFILL_PROFILE_WORKLOADS
    if decode_comparison:
        grid = sublayer.DECODE_PROFILE_WORKLOADS
    if prefill_comparison:
        grid = sublayer.PREFILL_PROFILE_WORKLOADS
    spec = dict(workloads=grid,variants=list(variants)) if prefill else {}
    stage_map, _ = (sublayer.profile_grid(spec) if attention_sublayer else prefill_profile_grid(spec)) if prefill else (STAGES, None)
    captures = [(r,t,v,f'r{r}-t{t}-v{v}') for r,t in grid for v in variants] if prefill else [
        (None,None,v,str(v)) for v in STAGES]
    for r,t,variant,folder in captures:
        stages = stage_map[variant]
        directory = source / folder
        report = json.loads((directory / 'summary.json').read_text())
        provenance = json.loads((directory / 'profile.provenance.json').read_text())
        identity = report['capture_identity']
        if identity['capture_receipt']['sha256'] != sha(directory / 'capture.json'):
            raise ValueError('capture receipt changed')
        if identity['provenance']['sha256'] != sha(directory / 'profile.provenance.json'):
            raise ValueError('profile provenance changed')
        current = {k: provenance[k] for k in ('repository', 'hardware', 'software', 'source_sha256')}
        if 'analysis_source_sha256' not in report:
            raise ValueError('profile must be reanalyzed with dispatch coalescing')
        current['analysis_source_sha256'] = report['analysis_source_sha256']
        current['curation_source_sha256'] = sha(Path(__file__))
        if common is not None and current != common:
            raise ValueError('captures must share source and environment')
        common = current
        # Bind the exported timing observations to the validated analyzer inputs.
        for name, kind in [('submissions.xml', 'command_buffer_submissions_xml'), ('gpu-intervals.xml', 'gpu_intervals_xml')]:
            expected = next(x['sha256'] for x in report['inputs'] if x['kind'] == kind)
            if sha(directory / name) != expected:
                raise ValueError('timing export changed')
        submissions = [r for r in read_table(directory / 'submissions.xml') if integer(r, 'num-encoders') > 0]
        ids = {integer(r, 'cmdbuffer-id') for r in submissions}
        processes = {r['process'][1] for r in submissions if r['process'][1]}
        if len(processes) != 1:
            raise ValueError('ambiguous target')
        process = next(iter(processes))
        intervals = [r for r in read_table(directory / 'gpu-intervals.xml')
                     if integer(r, 'cmdbuffer-id') in ids and process in r['event-label'][1]
                     and r['channel-name'][0] == 'Compute' and ':Compute Command' in r['event-label'][1]]
        intervals.sort(key=lambda r: integer(r, 'start'))
        workload = identity['workload']
        shape = dict(query_rows=r,rows=t) if prefill else {}
        if prefill and (
            workload['profile_rows'] != r or workload['key_value_rows'] != t or
            identity['implementation'] != (f'attention_sublayer_{variant}' if attention_sublayer else f'gqa_prefill_{variant}')
        ):
            raise ValueError('prefill capture differs from requested comparison')
        intervals, coalescing = coalesce_compute_commands(intervals,submissions,
            (workload['warmup_iterations']+workload['profile_iterations'])*len(stages))
        if coalescing != report['validated_sequence']['interval_coalescing']:
            raise ValueError('dispatch coalescing differs from validated analysis')
        *_, profile = segment_compute_commands(intervals, workload['warmup_iterations'],
                                               workload['profile_iterations'], len(stages), False)
        if duration_summary(profile) != report['instrumented_gpu_interval_duration']['profile']:
            raise ValueError('profile sequence differs from validated analysis')
        for i, row in enumerate(profile):
            samples.append(dict(**shape,variant=variant, iteration=i // len(stages), stage=stages[i % len(stages)],
                                duration_ns=integer(row, 'duration')))
        counter_report = report.get('profile_gpu_counters')
        counter_note = {}
        if counter_report is None:
            note_path = directory / 'counter_analysis_note.json'
            note = json.loads(note_path.read_text()) if note_path.exists() else {}
            counter_note = dict(counter_analysis={**note, 'status': 'not_analyzed'})
        records.append(dict(**shape,variant=variant, capture=identity, trace=report['trace'],
                            interval_coalescing=coalescing,fragmented_profile_dispatches=report['validated_sequence']['fragmented_profile_dispatches'],
                            conditions=json.loads((directory / 'conditions.json').read_text()),
                            counters_scope=counter_report['scope'] if counter_report is not None else
                                'No counter analysis was supplied for this capture; absence is not zero.',
                            counters=[c for c in counter_report['counters'] if c['name'] in COUNTERS] if counter_report is not None else [],
                            **counter_note,
                            spills=report['compiler_spills']))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(samples[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(samples)
    raw = output / (prefix+'profile_samples.csv.gz')
    raw.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
    write_json(output / (prefix+'profiles.json'), dict(schema=3 if attention_sublayer else (2 if prefill else 1),
                **(dict(specification=spec) if prefill else {}), common=common, captures=records,
                samples_sha256=sha(raw),
                boundary=f'Instrumented GPU active dispatch durations (non-overlapping segments summed, preemption gaps excluded); {len(captures)} single captures, not paired latency trials. Counter statistics are device-wide within each target window. Stage labels follow the validated source enqueue order.',
                retention='All target dispatch durations retained. Three named counter summaries retained; full trace/XML exports and other counters remain external.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prefill-variant',type=int)
    group.add_argument('--prefill-variants',type=int,nargs='+')
    group.add_argument('--attention-sublayer',action='store_true')
    parser.add_argument('--prefix',default='')
    parser.add_argument('--wo-comparison',action='store_true')
    parser.add_argument('--decode-comparison',action='store_true')
    parser.add_argument('--prefill-comparison',action='store_true')
    args = parser.parse_args()
    collect(args.source, args.output, args.prefill_variant,
            prefill_variants=args.prefill_variants, prefix=args.prefix, attention_sublayer=args.attention_sublayer,
            wo_comparison=args.wo_comparison, decode_comparison=args.decode_comparison,
            prefill_comparison=args.prefill_comparison)
