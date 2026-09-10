"""Verify retained reference evidence and regenerate study tables and figures.

Uses project validation helpers for schedule coverage; never executes a model.
"""
import csv
import argparse
import gzip
import hashlib
import json
import math
from statistics import median
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
    from llm_mojo.model_validation import required_consistency_schedules

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
    if (sha((json.dumps(declaration, indent=2) + '\n').encode()) != manifest['declaration_sha256']
            or qualified['passed'] is not True or qualified['candidate_outputs_observed'] is not False
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
            if not rows or rows in seen or sum(rows) != case['length'] or min(rows) < 1 or schedule['failures']:
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
        if seen != required_consistency_schedules(case['length'], declaration):
            raise ValueError('incomplete declared reference schedule census')
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
    print(f'Verified {sum(observed.values()):,} HF comparisons, {len(layer):,} decoder records, '
          'failed model accuracy and both operation diagnoses; regenerated four tables.')


def fast_reference():
    """Replay the complete failed Fast qualification without Torch or a model."""
    manifest = json.loads((ROOT/'fast-reference-study.json').read_text())
    if (manifest.get('kind') != 'fast-reference-study-v1'
            or any(manifest.get(k) is not False for k in (
                'calibration_passed','confirmation_executed','native_acceptance_executed'))
            or set(manifest['files']) != {'fast-reference-result.json.gz',
                'fast-reference-budgets.json.gz','fast-reference-observations.jsonl.gz',
                'fast-reference-diagnosis.json.gz','fast-reference-contract.json.gz'}):
        raise ValueError('Fast study scope or file census mismatch')
    data = {}
    for name, record in manifest['files'].items():
        encoded = (ROOT/name).read_bytes()
        raw = gzip.decompress(encoded)
        if (sha(encoded) != record['sha256'] or sha(raw) != record['uncompressed_sha256']
                or len(raw) != record['uncompressed_bytes']):
            raise ValueError('Fast evidence hash mismatch')
        data[name] = raw
    report = json.loads(data['fast-reference-result.json.gz'])
    frozen = json.loads(data['fast-reference-budgets.json.gz'])
    declaration = json.loads(data['fast-reference-contract.json.gz'])
    rows = [json.loads(line) for line in data['fast-reference-observations.jsonl.gz'].splitlines()]
    diagnosis = json.loads(data['fast-reference-diagnosis.json.gz'])
    if (report['source_commit'] != manifest['source_commit']
            or report['source']['tests/fixtures/model_fast.json'] != sha(data['fast-reference-contract.json.gz'])
            or report['declaration'] != declaration or frozen['declaration'] != declaration
            or report['observations_sha256'] != sha(data['fast-reference-observations.jsonl.gz'])
            or report['frozen_budgets_sha256'] != sha(data['fast-reference-budgets.json.gz'])
            or report['gates'] != frozen['gates'] or report['checks'] != len(rows)
            or not report['started_at'] <= report['frozen_at'] <= report['finished_at']):
        raise ValueError('Fast result/budget/source binding mismatch')
    boundaries = ({f'hidden_{i}' for i in range(25)} | {'final_norm','logits'} |
                  {f'cache_{kind}_{i}' for kind in ('key','value') for i in range(24)})
    def role(name):
        if name == 'hidden_0': return 'embedding'
        if name.startswith('hidden_'): return 'hidden'
        if name.startswith('cache_key_'): return 'key'
        if name.startswith('cache_value_'): return 'value'
        if name in ('final_norm','logits'): return name
        raise ValueError('unknown Fast boundary')
    cases = report['cases']['calibration']
    required_cases = ({f'random-{n}' for n in declaration['calibration']['lengths']} |
        {f'text-{i}' for i in range(len(declaration['calibration']['texts']))})
    if set(name for name, ids in cases) != required_cases or len(cases) != len(required_cases):
        raise ValueError('incomplete Fast calibration cases')
    expected = Counter()
    for case, ids in cases:
        length = len(ids)
        if (not 1 <= length <= 4096 or any(type(i) is not int or not 0 <= i < 151936 for i in ids)
                or (case.startswith('random-') and length != int(case.split('-')[1]))):
            raise ValueError('invalid Fast calibration input')
        schedules = {(length,), tuple([1]*length if length <= 17 else [length-17,16,1])}
        for schedule in schedules:
            for arm in declaration['reference_arms']:
                start = 0
                for count in schedule:
                    for stage in boundaries:
                        expected[('calibration',case,length,arm,schedule,start,count,stage)] += 1
                    start += count
    observed = Counter((r['phase'],r['case'],r['length'],r['arm'],tuple(r['schedule']),r['start'],r['rows'],r['stage']) for r in rows)
    if observed != expected:
        raise ValueError('incomplete or duplicated Fast reference schedule census')
    derived, summary = {}, []
    for group in declaration['maximum_atol']:
        records = [r for r in rows if role(r['stage']) == group]
        gate = dict(rtol=declaration['rtol'], exact=False)
        for metric, field in [('required_atol','atol'),('relative_rms','relative_rms')]:
            gate[field] = max(declaration[field+'_floor'], math.ceil(
                declaration['margin']*max(r[metric] for r in records)/declaration[field+'_quantum'])*declaration[field+'_quantum'])
        derived[group] = gate
        summary.append(dict(role=group, max_observed_relative_rms=max(r['relative_rms'] for r in records),
            derived_relative_rms=gate['relative_rms'], relative_rms_ceiling=declaration['maximum_qualified_relative_rms'],
            derived_atol=gate['atol'], atol_ceiling=declaration['maximum_atol'][group]))
    derived['embedding'] = dict(rtol=0.,atol=0.,relative_rms=0.,exact=True)
    gates = {name: derived[role(name)] for name in boundaries}
    ceiling_failures = sorted(n for n,g in gates.items() if role(n) != 'embedding' and (
        g['atol'] > declaration['maximum_atol'][role(n)] or g['relative_rms'] > declaration['maximum_qualified_relative_rms']))
    if gates != report['gates'] or ceiling_failures != report['ceiling_failures'] or not ceiling_failures:
        raise ValueError('Fast budget derivation or ceiling decision mismatch')
    for record in (report, frozen):
        if (record['calibration_passed'] is not False
                or record['new_candidate_outputs_used_for_calibration'] is not False
                or record['reserved_outputs_observed'] is not False):
            raise ValueError('Fast qualification scope/status mismatch')
    if report['passed'] is not False or report['confirmation_executed'] is not False:
        raise ValueError('failed Fast qualification was promoted')
    prediction_rows, failures = [], []
    for r in rows:
        g = gates[r['stage']]
        if any(not math.isfinite(r[k]) or r[k] < 0 for k in ('max_abs','required_atol','relative_rms')):
            raise ValueError('invalid Fast numerical observation')
        passed = (r['required_atol'] <= g['atol'] and r['relative_rms'] <= g['relative_rms']
                  and (not g['exact'] or r['exact']))
        if r['stage'] == 'logits':
            p = declaration['prediction']
            passed &= (0 <= r['kl_nats'] <= p['maximum_kl_nats']
                and 0 <= r['total_variation'] <= p['maximum_total_variation']
                and (r['reference_margin'] <= 2*(g['atol']+g['rtol']*r['reference_max_abs']) or r['same_token']))
            prediction_rows.append({k:r[k] for k in ('case','length','arm','start','rows','max_abs','relative_rms','kl_nats','total_variation','same_token','reference_margin')})
        if not passed: failures.append(r)
    if failures != report['failures']:
        raise ValueError('Fast numerical decision mismatch')
    detail = diagnosis['fast_affine']
    expected_modules = {f'model.layers.{i}.{name}' for i in range(24) for name in (
        'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
        'mlp.gate_proj','mlp.up_proj','mlp.down_proj')} | {'lm_head'}
    if (diagnosis['source_sha256'] != manifest['diagnosis_source_sha256']
            or detail['qualification_sha256'] != sha(data['fast-reference-result.json.gz'])
            or detail['observation_exact'] is not True or detail['calibration_source'] != report['source']
            or diagnosis['reserved_outputs_observed'] is not False
            or diagnosis['new_candidate_outputs_observed'] is not False
            or len(detail['reproduced_boundaries']) != 75
            or {r['stage'] for r in detail['reproduced_boundaries']} != boundaries
            or len(detail['operations']) != 169 or {r['module'] for r in detail['operations']} != expected_modules
            or len(detail['exact_dot_witnesses']) != 16):
        raise ValueError('incomplete Fast affine diagnosis')
    worst = max(rows, key=lambda r:r['relative_rms'])
    if worst != detail['worst_record']:
        raise ValueError('Fast diagnosis did not target the exposed maximum')
    for r in detail['reproduced_boundaries']:
        original = next(x for x in rows if x['case'] == worst['case'] and x['arm'] == worst['arm']
                        and x['schedule'] == worst['schedule'] and x['stage'] == r['stage'])
        if any(r[k] != original[k] for k in ('max_abs','required_atol','relative_rms')):
            raise ValueError('Fast failure did not reproduce')
    for r in detail['exact_dot_witnesses']:
        exact = Fraction(int(r['exact_numerator']),int(r['exact_denominator']))
        a,b = abs(Fraction(r['hf'])-exact), abs(Fraction(r['split'])-exact)
        if (r['nearer'] != ('hf' if a < b else 'split' if b < a else 'tie')
                or r['hf_absolute_error'] != float(a) or r['split_absolute_error'] != float(b)):
            raise ValueError('invalid Fast exact-dot interpretation')
    table('fast-reference-budgets.csv', summary)
    table('fast-reference-predictions.csv', prediction_rows)
    table('fast-reference-operations.csv', detail['operations'])
    print(f'Verified {len(rows):,} Fast reference checks, failed calibration ceilings, '
          '169 local affine comparisons and 16 exact witnesses; confirmation/native acceptance remain unexecuted.')


def runtime_ratios(report):
    samples=report['samples']
    expected=Counter()
    for w in report['specification']['measurements']:
        for block in range(4):
            for arm,config in enumerate([0,0,*w['candidates']]):
                for sample in range(10):
                    expected[(block,arm,w['total']-w['rows'],w['rows'],config,sample)]+=1
    observed=Counter(tuple(r[k] for k in ('block','arm','prefix','rows','configuration','sample')) for r in samples)
    if expected!=observed or any(r['nanoseconds']<=0 for r in samples):
        raise ValueError('incomplete runtime measurement census')
    results=[]
    for w in report['specification']['measurements']:
        def med(block,arm):
            return median(r['nanoseconds'] for r in samples if r['block']==block and r['arm']==arm
                and r['rows']==w['rows'] and r['prefix']+r['rows']==w['total'])
        noise=max(abs(med(b,1)/med(b,0)-1) for b in range(4))
        for arm,config in enumerate(w['candidates'],start=2):
            ratios=[med(b,arm)/med(b,0) for b in range(4)]
            gain=all(r<1 for r in ratios) and 1-median(ratios)>max(.05,noise)
            regression=all(r>1 for r in ratios) and median(ratios)-1>max(.05,noise)
            results.append(dict(rows=w['rows'],total=w['total'],configuration=config,
                baseline_ms=median(med(b,0) for b in range(4))/1e6,
                candidate_ms=median(med(b,arm) for b in range(4))/1e6,
                median_ratio=median(ratios),min_ratio=min(ratios),max_ratio=max(ratios),control_noise=noise,
                outcome='gain' if gain else 'regression' if regression else 'inconclusive'))
    return results


def runtime_diagnostic_census(diagnostics):
    wanted=Counter();stored=Counter()
    for case in diagnostics['specification']['cases']:
        for mode in ('full','scheduled'):
            schedule=[len(case['ids'])] if mode=='full' else case['schedule']
            variants=case['full_configurations'] if mode=='full' else [None]
            for variant in variants:
                start=0
                for call,rows in enumerate(schedule):
                    config=variant if mode=='full' else case['configurations'][call]
                    for stage in ({f'hidden_{i}' for i in range(25)} | {'logits','final_norm'} |
                                  {f'cache_{kind}_{i}' for kind in ('key','value') for i in range(24)}):
                        tag=(case['name'],mode,config,call,start,rows,stage)
                        wanted[tag+('hf_same_history',)]+=1
                        if stage.startswith('cache_'): stored[tag]+=1
                        if mode=='scheduled' and (stage not in ('logits','final_norm') or start+rows==len(case['ids'])):
                            wanted[tag+('native_full',)]+=1
                    start+=rows
    keys=('case','mode','configuration','call','start','rows','stage')
    if Counter(tuple(r[k] for k in keys+('comparison',)) for r in diagnostics['diagnostics'])!=wanted:
        raise ValueError('incomplete runtime diagnostic census')
    if Counter(tuple(r[k] for k in keys) for r in diagnostics['storage'])!=stored:
        raise ValueError('incomplete runtime storage census')
    if diagnostics['invariants_passed'] is not True or any(not all(r[k] for k in ('prefix','append','inactive')) for r in diagnostics['storage']):
        raise ValueError('runtime cache invariant failed')
    if any(not r['exact'] for r in diagnostics['diagnostics'] if r['stage']=='hidden_0'):
        raise ValueError('runtime embedding invariant failed')
    return len(wanted),len(stored)


def runtime():
    manifest=json.loads((ROOT/'runtime-study.json').read_text())
    payload={}
    for name,record in manifest['files'].items():
        encoded=(ROOT/name).read_bytes();raw=gzip.decompress(encoded)
        if sha(encoded)!=record['sha256'] or sha(raw)!=record['raw_sha256']:
            raise ValueError('runtime evidence hash mismatch')
        payload[name]=json.loads(raw)
    diagnostics=payload['runtime-diagnostics.json.gz']
    measurements=payload['runtime-measurements.json.gz']
    generation=payload['runtime-generations.json.gz']
    reference=payload['runtime-reference.json.gz']
    if (diagnostics['reference_sha256']!=manifest['files']['runtime-reference.json.gz']['raw_sha256']
            or diagnostics['specification']!=reference['specification']):
        raise ValueError('runtime reference binding mismatch')
    checks,storage_checks=runtime_diagnostic_census(diagnostics)
    predictions=[r for r in diagnostics['diagnostics'] if r['stage']=='logits']
    if 'runtime-history-diagnostics.json.gz' in payload:
        history=payload['runtime-history-diagnostics.json.gz']
        history_reference=payload['runtime-history-reference.json.gz']
        if (history['reference_sha256']!=manifest['files']['runtime-history-reference.json.gz']['raw_sha256']
                or history['specification']!=history_reference['specification']
                or history['specification']['history_source_sha256']!=manifest['files']['runtime-generations.json.gz']['raw_sha256']):
            raise ValueError('runtime history binding mismatch')
        extra_checks,extra_storage=runtime_diagnostic_census(history)
        checks+=extra_checks;storage_checks+=extra_storage
        predictions.extend(r for r in history['diagnostics'] if r['stage']=='logits')
        for i,(case,g) in enumerate(zip(history['specification']['cases'],generation['records'],strict=True)):
            remaining=len(g['prompt_ids']);schedule=[]
            while remaining:
                rows=min(remaining,g['chunk_rows'] or remaining)
                schedule.append(rows);remaining-=rows
            schedule += [1]*(len(g['tokens'])-1)
            if case['ids']!=g['prompt_ids']+g['tokens'][:-1] or case['schedule']!=schedule:
                raise ValueError('diagnostics did not follow actual native generation')
            logits=sorted((r for r in history['diagnostics'] if r['case']==case['name']
                and r['comparison']=='hf_same_history' and r['stage']=='logits' and r['mode']=='scheduled'
                and r['start']+r['rows']>=len(g['prompt_ids'])),key=lambda r:r['call'])
            if [r['token'] for r in logits]!=g['tokens']:
                raise ValueError('captured token choices differ from actual generation')
    summary=runtime_ratios(measurements)
    selected={}
    for r in sorted(summary,key=lambda r:(r['median_ratio'],r['configuration'])):
        if r['outcome']=='gain': selected.setdefault((r['rows'],r['total']),r['configuration'])
    for final_name,ref_name in (('runtime-selected-diagnostics.json.gz','runtime-reference.json.gz'),
                                ('runtime-mixed-diagnostics.json.gz','runtime-mixed-reference.json.gz')):
        if final_name not in payload: continue
        final=payload[final_name];selected_reference=payload[ref_name]
        if final['configuration_policy']!='fast' or final['reference_sha256']!=manifest['files'][ref_name]['raw_sha256']:
            raise ValueError('selected runtime reference/policy mismatch')
        for actual,original in zip(final['specification']['cases'],selected_reference['specification']['cases'],strict=True):
            if any(actual[k]!=original[k] for k in ('name','ids','schedule')):
                raise ValueError('selected runtime case mismatch')
            total=0;expected=[]
            for rows in actual['schedule']:
                total+=rows;expected.append(selected.get((rows,total),0))
            if (actual['configurations']!=expected or
                    actual['full_configurations']!=[selected.get((len(actual['ids']),len(actual['ids'])),0)]):
                raise ValueError('automatic dispatch differs from measured selection')
        extra_checks,extra_storage=runtime_diagnostic_census(final)
        checks+=extra_checks;storage_checks+=extra_storage
    propagation=None
    if 'runtime-propagation.json.gz' in payload:
        prop=payload['runtime-propagation.json.gz']['result']
        if (prop['native_result_sha256']!=manifest['files']['runtime-diagnostics.json.gz']['raw_sha256']
                or prop['reference_manifest_sha256']!=manifest['files']['runtime-reference.json.gz']['raw_sha256']
                or prop['baseline_reproduced_boundaries']!=75 or [r['layer'] for r in prop['layers']]!=[0,1,2,3]):
            raise ValueError('runtime propagation binding or coverage mismatch')
        propagation=[]
        for r in prop['layers']:
            for k in ('input_difference','observed_output_difference','hf_propagated_difference','identical_input_residual'):
                values=r[k]['row_relative_l2']
                if (len(values)!=len(prop['ids']) or any(not math.isfinite(v) or v<0 for v in values)
                        or max(values)!=r[k]['max_row_relative_l2']):
                    raise ValueError('incomplete propagation rows')
            propagation.append(dict(layer=r['layer'],**{k:r[k]['max_row_relative_l2'] for k in (
                'input_difference','observed_output_difference','hf_propagated_difference','identical_input_residual')}))
    generations=[]
    from llm_mojo.model_validation import validate_generation_events
    declaration=reference['specification']['declaration']
    if Counter((r['prompt'],r['chunk_rows']) for r in generation['records'])!=Counter(
            (prompt,chunk) for prompt in declaration['prompts'] for chunk in (0,4)):
        raise ValueError('incomplete generation prompt/chunk census')
    if generation['empty_prompt_rejected'] is not True or generation['zero_budget']['tokens']:
        raise ValueError('missing generation limit/input checks')
    for r in generation['records']:
        verified=validate_generation_events(r['events'],declaration['max_new_tokens'])
        if any(verified[k]!=r[k] for k in verified) or r['unobserved_output_exact'] is not True:
            raise ValueError('generation event binding or observation mismatch')
        oracle=next(x for x in reference['generations'] if x['prompt']==r['prompt'])
        if r['prompt_ids']!=oracle['prompt_ids']: raise ValueError('native/reference tokenizer mismatch')
        first=next((i for i,(a,b) in enumerate(zip(r['tokens'],oracle['tokens'])) if a!=b),None)
        if first is None and len(r['tokens'])!=len(oracle['tokens']):
            first=min(len(r['tokens']),len(oracle['tokens']))
        generations.append(dict(prompt=r['prompt'],chunk_rows=r['chunk_rows'],tokens=len(r['tokens']),
            reference_tokens=len(oracle['tokens']),first_difference=first,text=r['text'],reference_text=oracle['text']))
    table('runtime-measurements.csv',summary)
    table('runtime-selection.csv',[dict(rows=r,total=t,configuration=c) for (r,t),c in sorted(selected.items())])
    table('runtime-predictions.csv',predictions)
    table('runtime-generations.csv',generations)
    if propagation is not None: table('runtime-propagation.csv',propagation)
    print(f"Verified {checks:,} runtime diagnostics, {storage_checks:,} storage checks and {len(measurements['samples']):,} timing samples.")


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
    if (ROOT / 'fast-reference-study.json').exists():
        fast_reference()
    if (ROOT / 'runtime-study.json').exists():
        runtime()


if __name__ == '__main__':
    main()
