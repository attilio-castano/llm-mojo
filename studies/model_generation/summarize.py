"""Verify retained reference evidence and regenerate study tables and figures.

Uses only the Python standard library; never executes a model.
"""
import csv
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
from fractions import Fraction

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


def aten(plot=False):
    manifest = json.loads((ROOT / 'aten-study.json').read_text())
    encoded = (ROOT / 'aten-detail.json.gz').read_bytes()
    record = manifest['files']['aten-detail.json.gz']
    raw = gzip.decompress(encoded)
    if (sha(encoded) != record['sha256'] or sha(raw) != record['uncompressed_sha256']
            or len(raw) != record['uncompressed_bytes']):
        raise ValueError('ATen evidence checksum mismatch')
    result = json.loads(raw)
    backend = result['backend']
    declaration = backend['declaration']
    if (result['source_sha256'] != manifest['source_sha256']
            or backend['source_sha256'] != manifest['helper_source_sha256']
            or backend['declaration_sha256'] != manifest['declaration_sha256']
            or backend['environment']['torch_git_version'] != declaration['torch_git_version']
            or result['candidate_outputs_observed'] is not False
            or result['reserved_outputs_observed'] is not False):
        raise ValueError('ATen source or scope mismatch')
    for field, declared in [('localized', 'trace_cases'), ('canonical', 'canonical_cases')]:
        if [dict(length=c['length'], seed=c['seed']) for c in backend[field]] != declaration[declared]:
            raise ValueError('ATen declared case census mismatch')
    stages, isolation, dispatch, canonical = [], [], [], []
    for case in backend['localized']:
        if (case['observer_and_repeat_bitwise_equal'] is not True
                or [r['position'] for r in case['records']] != list(range(case['length']))):
            raise ValueError('ATen observation or position census mismatch')
        for row in case['records']:
            if [s['stage'] for s in row['stages']] != declaration['operations']:
                raise ValueError('ATen operation census mismatch')
            for stage in row['stages']:
                stages.append(dict(length=case['length'], position=row['position'], **stage))
            isolation.append(dict(length=case['length'], position=row['position'],
                full_deterministic_equal=case['full_deterministic_equal'],
                cached_deterministic_equal=row['deterministic_equal'],
                fixed_scores_softmax_max_abs=row['fixed_scores_softmax']['max_abs'],
                fixed_probabilities_pv_max_abs=row['fixed_probabilities_pv']['max_abs']))
        expected = [(p, m) for p in declaration['threshold_positions'] for m in declaration['duplicate_query_rows']]
        if [(r['position'], r['query_rows']) for r in case['threshold_probes']] != expected:
            raise ValueError('ATen dispatch probe census mismatch')
        for row in case['threshold_probes']:
            duplicates = row['duplicate_row_comparisons']
            if [r['row'] for r in duplicates] != list(range(row['query_rows'])):
                raise ValueError('ATen duplicate query census mismatch')
            dispatch.append(dict(length=case['length'], position=row['position'], keys=row['keys'],
                query_rows=row['query_rows'], contraction=row['contraction'], work=row['work'],
                source_predicted_path=row['source_predicted_path'],
                forced_per_head_mm_equal=row['forced_per_head_mm_equal'],
                first_row_max_abs=row['versus_one_row']['max_abs'],
                all_rows_max_abs=max(r['max_abs'] for r in duplicates),
                different_rows=json.dumps([r['row'] for r in duplicates if r['different'] > 0])))
    boundaries = ({'hidden_' + str(i) for i in range(25)} | {'final_norm', 'logits'}
                  | {'cache_' + kind + '_' + str(i) for kind in ('key', 'value') for i in range(24)})
    for case in backend['canonical']:
        observed = {(r['position'], r['stage']) for r in case['checks']}
        expected = {(p, name) for p in range(case['length']) for name in boundaries}
        if len(case['checks']) != len(expected) or observed != expected:
            raise ValueError('canonical boundary census mismatch')
        exact = all(r['exact'] for r in case['checks'])
        if exact != case['schedule_bitwise_equal']:
            raise ValueError('canonical equality summary mismatch')
        original = next((c for c in backend['localized'] if (c['length'], c['seed']) ==
                         (case['length'], case['seed'])), None)
        cached_equal = (case['cached_boundary_sha256'] == original['original_cached_boundary_sha256']
                        if original else None)
        if cached_equal != case['original_cached_comparison']:
            raise ValueError('canonical cached digest comparison mismatch')
        canonical.append(dict(length=case['length'], seed=case['seed'], checks=len(case['checks']),
            exact_checks=sum(r['exact'] for r in case['checks']),
            repeat_bitwise_equal=case['repeat_bitwise_equal'], schedule_bitwise_equal=exact,
            original_cached_comparison=cached_equal))
    table('aten-stages.csv', stages)
    table('aten-isolation.csv', isolation)
    table('aten-dispatch.csv', dispatch)
    table('aten-canonical.csv', canonical)
    print('Verified ATen evidence; regenerated four tables:', sum(r['exact_checks'] for r in canonical),
          'exact canonical comparisons.')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
        rows = [r for r in dispatch if r['length'] == 15 and r['keys'] == 6]
        axes[0].plot([r['query_rows'] for r in rows], [1e6*r['all_rows_max_abs'] for r in rows],
                     marker='o', color='#2166ac')
        axes[0].axvline(1.5, color='#b35806', linestyle='--', label='Small bmm → per-head mm')
        axes[0].set(xlabel='Identical duplicated query rows (M)', ylabel='Largest score difference × 10⁶',
                    title='Six keys: changing M changes rounding\nWork M × 64 × 6 crosses 400 at M = 2',
                    xticks=range(1, 9), ylim=(0, 105))
        axes[0].legend(frameon=False, fontsize=9)
        axes[0].grid(alpha=.2)
        matrix = np.full((8, 8), np.nan)
        case = backend['localized'][0]
        for row in case['threshold_probes']:
            if row['keys'] == 2:
                for value in row['duplicate_row_comparisons']:
                    matrix[row['query_rows'] - 1, value['row']] = 1e6*value['max_abs']
        cmap = plt.get_cmap('YlOrRd').copy()
        cmap.set_bad('#dddddd')
        shown = axes[1].imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        axes[1].set(xlabel='Output row position (zero-based)', ylabel='Identical duplicated query rows (M)',
                    title='Two keys: row position also matters\nEach cell compared with the one-query result',
                    xticks=range(8), yticks=range(8), yticklabels=range(1, 9))
        figure.colorbar(shown, ax=axes[1], label='Largest score difference × 10⁶')
        figure.savefig(ROOT / 'aten-dispatch.png', dpi=180)
        plt.close(figure)


def consistency():
    manifest = json.loads((ROOT/'consistency-study.json').read_text())
    data = {}
    for name, record in manifest['files'].items():
        encoded = (ROOT/name).read_bytes()
        raw = gzip.decompress(encoded)
        if (sha(encoded) != record['sha256'] or sha(raw) != record['uncompressed_sha256']
                or len(raw) != record['uncompressed_bytes']):
            raise ValueError('consistency evidence checksum mismatch')
        data[name] = raw
    qualified = json.loads(data['consistency-reference.json.gz'])
    declaration = manifest['declaration']
    if (qualified['passed'] is not True or qualified['candidate_outputs_observed'] is not False
            or qualified['reserved_outputs_observed'] is not False
            or qualified['source']['tests/fixtures/model_consistency.json'] != manifest['declaration_sha256']
            or qualified['observations_sha256'] != sha(data['consistency-observations.jsonl.gz'])
            or [{k:c[k] for k in ('length','seed')} for c in qualified['cases']] != declaration['development_cases']):
        raise ValueError('consistency reference identity mismatch')
    boundaries = set(declaration['accuracy']['gates'])
    required, observed, summary = Counter(), Counter(), []
    for case in qualified['cases']:
        if set(case['arrays']) != boundaries:
            raise ValueError('incomplete consistency boundary census')
        seen, checks = set(), 0
        for schedule in case['schedules']:
            rows = tuple(schedule['rows'])
            if rows in seen or sum(rows) != case['length'] or min(rows) < 1 or schedule['failures']:
                raise ValueError('invalid consistency schedule')
            seen.add(rows)
            if schedule['checks'] != len(rows)*75:
                raise ValueError('incomplete schedule checks')
            checks += schedule['checks']
            start = 0
            for count in rows:
                for stage in boundaries:
                    required[(case['length'],case['seed'],rows,start,count,stage)] += 1
                start += count
        if (case['length'],) not in seen:
            raise ValueError('missing full repeat')
        summary.append(dict(length=case['length'],seed=case['seed'],schedules=len(seen),checks=checks,failures=0))
    for line in data['consistency-observations.jsonl.gz'].splitlines():
        row = json.loads(line)
        if row['exact'] is not True or row['max_abs'] != 0:
            raise ValueError('failed reference consistency observation')
        observed[(row['length'],row['seed'],tuple(row['schedule']),row['start'],row['rows'],row['stage'])] += 1
    if observed != required:
        raise ValueError('missing or duplicate reference observations')
    detail = json.loads(data['consistency-native.json.gz'])
    accuracy, ops = detail['accuracy'], detail['operations']
    if (accuracy['qualification_sha256'] != sha(data['consistency-reference.json.gz'])
            or accuracy['reserved_outputs_observed'] is not False or accuracy['passed'] is not False
            or set(r['stage'] for r in accuracy['checks']) != boundaries |
                {s+'_storage' for s in boundaries if s.startswith('cache_')}
            or len(accuracy['checks']) != 123 or sum(not r['passed'] for r in accuracy['checks']) != 7):
        raise ValueError('native accuracy evidence mismatch')
    operation_stages = ('N_att','Q_raw','K_raw','V_raw','O','B_att','Z','N_mlp','G','U','A','S','B_mlp','Y')
    for attempt in (detail['operations_initial'], ops):
        if (attempt['passed'] is not True or attempt['reserved_outputs_observed'] is not False
                or len(attempt['checks']) != 336 or any(r['failed'] for r in attempt['checks'])
                or {(r['layer'],r['stage']) for r in attempt['checks']} !=
                    {(i,s) for i in range(24) for s in operation_stages}):
            raise ValueError('incomplete identical-operand diagnosis')
    if len(ops['rounding']) != sum(r['bit_differences'] for r in ops['checks']):
        raise ValueError('incomplete exact-sum diagnosis')
    for row in ops['rounding']:
        total = Fraction(row['exact_numerator'],row['exact_denominator'])
        a, r = Fraction(row['native']), Fraction(row['upstream'])
        closer = 'native' if abs(total-a)<abs(total-r) else 'upstream' if abs(total-r)<abs(total-a) else 'tie'
        if (row['closer_to_exact'] != closer or row['exact_sum'] != float(total)
                or row['midpoint_distance'] != float(total-(a+r)/2)):
            raise ValueError('inconsistent exact-sum interpretation')
    layer = [json.loads(line) for line in data['consistency-layer-records.jsonl.gz'].splitlines()]
    if (sha(data['consistency-layer-records.jsonl.gz']) != detail['layer_receipt']['records_sha256']
            or len(layer) != 13165 or any(r.get('failed',0) for r in layer)
            or any(r['exact'] is not True for r in layer if r['kind']=='schedule_exact')
            or not any(r['kind']=='schedule_exact' for r in layer)):
        raise ValueError('failed or incomplete decoder consistency evidence')
    table('consistency-reference.csv',summary)
    table('consistency-accuracy.csv',[dict(stage=r['stage'],passed=r['passed'],
        max_abs=r.get('max_abs',''),max_scaled=r.get('max_scaled',''),relative_rms=r.get('relative_rms',''))
        for r in accuracy['checks']])
    table('consistency-operations.csv',[dict(stage=s,checks=24,
        changed_elements=sum(r['bit_differences'] for r in ops['checks'] if r['stage']==s),
        max_abs=max(r['max_abs'] for r in ops['checks'] if r['stage']==s),
        failed_elements=sum(r['failed'] for r in ops['checks'] if r['stage']==s)) for s in operation_stages])
    table('consistency-rounding.csv',ops['rounding'])
    print('Verified 71,250 HF comparisons, 13,165 decoder records, failed model accuracy and both operation diagnoses; regenerated four tables.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true', help='also regenerate study figures with matplotlib')
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
    if (ROOT / 'aten-study.json').exists():
        aten(args.plot)
    if (ROOT / 'consistency-study.json').exists():
        consistency()


if __name__ == '__main__':
    main()
