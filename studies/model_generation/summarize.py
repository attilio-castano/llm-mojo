"""Verify retained reference evidence and regenerate the two study tables.

Uses only the Python standard library; never executes a model.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'reference-study.json').read_text())
    data = {}
    for name, record in manifest['files'].items():
        encoded = (ROOT / name).read_bytes()
        if sha(encoded) != record['sha256']:
            raise ValueError('retained evidence hash mismatch: ' + name)
        raw = gzip.decompress(encoded) if name.endswith('.gz') else encoded
        if sha(raw) != record['uncompressed_sha256'] or len(raw) != record['uncompressed_bytes']:
            raise ValueError('uncompressed evidence identity mismatch: ' + name)
        data[name] = raw
    frozen = json.loads(data['frozen-budgets.json'])
    result = json.loads(data['calibration-result.json'])
    rows = [json.loads(line) for line in data['calibration-observations.jsonl.gz'].splitlines()]
    if (sha(data['frozen-budgets.json']) != result['frozen_budgets_sha256']
            or sha(data['calibration-observations.jsonl.gz']) != result['observations_sha256']
            or frozen['gates'] != result['gates'] or len(rows) != result['checks']):
        raise ValueError('calibration result is not bound to these observations and budgets')
    with (ROOT / 'summary.csv').open('w') as stream:
        fields = ['phase', 'length', 'checks', 'pointwise_failures', 'relative_rms_failures',
                  'max_required_atol_ratio', 'max_relative_rms']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for phase in ('calibration', 'confirmation'):
            for length in frozen['declaration'][phase]['lengths']:
                group = [r for r in rows if r['phase'] == phase and r['length'] == length]
                expected_count = (length if length <= 17 else 3) * 75
                if len(group) != expected_count:
                    raise ValueError('incomplete declared case: ' + str((phase, length)))
                writer.writerow(dict(phase=phase, length=length, checks=len(group),
                    pointwise_failures=sum(r['required_atol'] > result['gates'][r['stage']]['atol'] for r in group),
                    relative_rms_failures=sum(r['relative_rms'] > result['gates'][r['stage']]['relative_rms'] for r in group),
                    max_required_atol_ratio=max(r['required_atol'] / result['gates'][r['stage']]['atol'] for r in group),
                    max_relative_rms=max(r['relative_rms'] for r in group)))
    diagnosis = json.loads(data['diagnosis.json.gz'])
    with (ROOT / 'diagnosis-summary.csv').open('w') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['mode', 'changed_boundaries', 'repeat_bitwise_equal',
                         'last_hidden_relative_rms', 'logits_relative_rms', 'logits_max_abs'])
        for mode, case in diagnosis['modes'].items():
            stages = {r['stage']: r for r in case['stages']}
            if len(stages) != 75:
                raise ValueError('incomplete diagnosis boundary census')
            writer.writerow([mode, sum(r['different'] > 0 for r in stages.values()),
                             all(r['repeat_bitwise_equal'] for r in stages.values()),
                             stages['hidden_24']['relative_rms'], stages['logits']['relative_rms'],
                             stages['logits']['max_abs']])
    print('Verified 5 evidence files; regenerated summary.csv and diagnosis-summary.csv.')


if __name__ == '__main__':
    main()
