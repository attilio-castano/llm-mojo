"""Verify retained reference evidence and regenerate the two study tables.

Uses only the Python standard library; never executes a model.
"""
import csv
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def table(name, rows):
    with (ROOT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def rounding(plot=False):
    manifest = json.loads((ROOT / 'rounding-study.json').read_text())
    encoded = (ROOT / 'rounding-detail.json.gz').read_bytes()
    record = manifest['files']['rounding-detail.json.gz']
    raw = gzip.decompress(encoded)
    if (sha(encoded) != record['sha256'] or sha(raw) != record['uncompressed_sha256']
            or len(raw) != record['uncompressed_bytes']):
        raise ValueError('rounding evidence checksum mismatch')
    result = json.loads(raw)
    detail = result['detail']
    if (result['source_sha256'] != manifest['source_sha256']
            or detail['declaration_sha256'] != manifest['declaration_sha256']
            or result['candidate_outputs_observed'] is not False
            or result['reserved_outputs_observed'] is not False):
        raise ValueError('rounding source or scope mismatch')
    if ([dict(length=c['length'], seed=c['seed']) for c in detail['cases']] != detail['declaration']['random_cases']
            or [c['text'] for c in detail['texts']] != detail['declaration']['text_cases']):
        raise ValueError('declared diagnostic case census mismatch')
    attention, propagation, predictions, trajectories, norms = [], [], [], [], []
    norm_gate = json.loads((ROOT / 'frozen-budgets.json').read_text())['gates']['final_norm']
    for case in detail['cases']:
        length = case['length']
        if (len(case['attention']) != 24 * length or len(case['hidden']) != 25 * length
                or len(case['predictions']) != length or case['observer_bitwise_equal'] is not True):
            raise ValueError('incomplete trace or changed instrumented output')
        for row in case['attention']:
            if row['layer'] == 0:
                attention.append(dict(length=length, position=row['position'],
                    qkv_equal=row['query_equal'] and row['key_equal'] and row['value_equal'],
                    fp32_max_abs=row['before_bf16']['max_abs'], bf16_max_abs=row['after_bf16']['max_abs'],
                    bf16_flips=row['after_bf16']['different']))
        for layer in range(25):
            rows = [r for r in case['hidden'] if r['layer'] == layer]
            propagation.append(dict(length=length, layer=layer,
                max_row_relative_rms=max(r['relative_rms'] for r in rows),
                max_abs=max(r['max_abs'] for r in rows)))
        for row in case['predictions']:
            predictions.append(dict(case='random_' + str(length), mode='tokenwise', step=row['position'],
                same_token=row['same_token'], full_token=row['full_token'], cached_token=row['cached_token'],
                full_margin=row['full_margin'], max_abs=row['max_abs'], relative_rms=row['relative_rms'],
                margin_certified=row['margin_certified']))
        for row in case['normalization']:
            point = row['pointwise_coordinate']
            norms.append(dict(length=length, position=row['position'],
                max_abs=row['max_abs'], max_abs_index=row['index'], relative_rms=row['relative_rms'],
                gate_index=point['index'], gamma=point['gamma'], full_input=point['full_input'],
                cached_input=point['cached_input'], full_output=point['full_output'],
                cached_output=point['cached_output'], required_atol=point['required_atol'],
                frozen_atol=norm_gate['atol'], pointwise_gate_passed=point['required_atol'] <= norm_gate['atol']))
    for index, case in enumerate(detail['texts']):
        for row in case['comparisons']:
            for mode in ('cached_full_prompt', 'cached_tokenwise_prompt'):
                values = row[mode]
                predictions.append(dict(case='text_' + str(index), mode=mode, step=row['step'],
                    same_token=values['same_token'], full_token=values['full_token'], cached_token=values['cached_token'],
                    full_margin=values['full_margin'], max_abs=values['max_abs'], relative_rms=values['relative_rms'],
                    margin_certified=values['margin_certified']))
        for mode, values in case['trajectories'].items():
            trajectories.append(dict(case='text_' + str(index), mode=mode, matches_full=values['matches_full'],
                full_tokens=json.dumps(case['full_tokens']), cached_tokens=json.dumps(values['tokens']),
                full_text=case['full_text'], cached_text=values['text']))
    table('rounding-first-layer.csv', attention)
    table('rounding-propagation.csv', propagation)
    table('rounding-predictions.csv', predictions)
    table('rounding-trajectories.csv', trajectories)
    table('rounding-normalization.csv', norms)
    print('Verified detailed reference evidence; regenerated five rounding tables.')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        rows = [r for r in attention if r['length'] == 15]
        # Zero-error token 0 is explicitly excluded from the logarithmic plot.
        for field, label, color in [('fp32_max_abs', 'Before BF16 rounding', '#2166ac'),
                                    ('bf16_max_abs', 'After BF16 rounding', '#b35806')]:
            shown = [r for r in rows if r[field] > 0]
            axes[0].plot([r['position'] for r in shown], [r[field] for r in shown],
                         marker='o', markersize=3, label=label, color=color)
        axes[0].set(yscale='log', xlabel='Token position (zero-based)', ylabel='Maximum absolute attention difference',
                    title='First attention layer: 15-token case\nIdentical Q/K/V inputs; token 0 has zero error')
        for length, color in [(15, '#b35806'), (17, '#2166ac')]:
            rows = [r for r in propagation if r['length'] == length]
            axes[1].plot([r['layer'] for r in rows], [100*r['max_row_relative_rms'] for r in rows],
                         label=str(length) + ' tokens', marker='o', markersize=3, color=color)
        axes[1].set(xlabel='Completed decoder layers (0 = embedding)', ylabel='Largest token-row relative L2 error (%)',
                    title='Differences propagate through the model\nFull prefill versus tokenwise cached execution')
        axes[1].set_ylim(bottom=0)
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(frameon=False, fontsize=9)
            axis.spines[['top', 'right']].set_visible(False)
        figure.savefig(ROOT / 'rounding.png', dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true', help='also regenerate the rounding figure with matplotlib')
    args = parser.parse_args()
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
    if (ROOT / 'rounding-study.json').exists():
        rounding(args.plot)


if __name__ == '__main__':
    main()
