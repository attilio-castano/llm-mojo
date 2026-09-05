"""Curate validated GQA captures into one profile record and raw dispatch table."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path

from .analyze_trace import (integer, read_table, segment_compute_commands, duration_summary)
from .study import sha, write_json

STAGES = {0: ['QK', 'softmax', 'PV'], 4: ['fused'], 9: ['decode', 'merge']}
COUNTERS = {'Kernel Occupancy', 'Instruction Throughput Limiter', 'Last Level Cache Limiter'}


def collect(source, output):
    records, samples = [], []
    common = None
    for variant, stages in STAGES.items():
        directory = source / str(variant)
        report = json.loads((directory / 'summary.json').read_text())
        provenance = json.loads((directory / 'profile.provenance.json').read_text())
        identity = report['capture_identity']
        if identity['capture_receipt']['sha256'] != sha(directory / 'capture.json'):
            raise ValueError('capture receipt changed')
        if identity['provenance']['sha256'] != sha(directory / 'profile.provenance.json'):
            raise ValueError('profile provenance changed')
        current = {k: provenance[k] for k in ('repository', 'hardware', 'software', 'source_sha256')}
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
        *_, profile = segment_compute_commands(intervals, workload['warmup_iterations'],
                                               workload['profile_iterations'], len(stages), False)
        if duration_summary(profile) != report['instrumented_gpu_interval_duration']['profile']:
            raise ValueError('profile sequence differs from validated analysis')
        for i, row in enumerate(profile):
            samples.append(dict(variant=variant, iteration=i // len(stages), stage=stages[i % len(stages)],
                                duration_ns=integer(row, 'duration')))
        records.append(dict(variant=variant, capture=identity, trace=report['trace'],
                            conditions=json.loads((directory / 'conditions.json').read_text()),
                            counters_scope=report['profile_gpu_counters']['scope'],
                            counters=[c for c in report['profile_gpu_counters']['counters'] if c['name'] in COUNTERS],
                            spills=report['compiler_spills']))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['variant', 'iteration', 'stage', 'duration_ns'], lineterminator='\n')
    writer.writeheader(); writer.writerows(samples)
    raw = output / 'profile_samples.csv.gz'
    raw.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
    write_json(output / 'profiles.json', dict(schema=1, common=common, captures=records,
                samples_sha256=sha(raw),
                boundary='Instrumented GPU dispatch durations; three single captures, not paired latency trials. Counter statistics are device-wide within each target window. Stage labels follow the validated source enqueue order.',
                retention='All target dispatch durations retained. Three named counter summaries retained; full trace/XML exports and other counters remain external.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    collect(args.source, args.output)
