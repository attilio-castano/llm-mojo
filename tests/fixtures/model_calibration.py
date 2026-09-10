# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==1.26.4", "torch==2.4.0", "transformers==4.43.1"]
# ///
"""Bounded upstream-only schedule calibration and independent confirmation.

No native executable is launched and no Mojo output is read. A passing report
is evidence for reviewing the full-model contract, not model acceptance.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

import model_reference as reference

DECLARATION_PATH = Path(__file__).with_suffix('.json')
FAST_DECLARATION_PATH = Path(__file__).with_name('model_fast.json')


def boundary_role(name):
    if name == 'hidden_0':
        return 'embedding'
    if name.startswith('hidden_'):
        return 'hidden'
    if name.startswith('cache_key_'):
        return 'key'
    if name.startswith('cache_value_'):
        return 'value'
    if name in ('final_norm', 'logits'):
        return name
    raise ValueError('unknown model boundary: ' + name)


@contextmanager
def split_linear():
    """Independent FP32 association; retain one BF16 output and fused bias.

    No native implementation is imported. All seven affine operations per
    decoder and the tied head are intercepted through actual HF modules.
    """
    counts = [0]
    def forward(module, inputs):
        if inputs.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16:
            raise ValueError('split reference requires BF16 affine operands')
        middle = inputs.shape[-1] // 2
        result = torch.nn.functional.linear(inputs[..., :middle].float(), module.weight[:, :middle].float())
        result += torch.nn.functional.linear(inputs[..., middle:].float(), module.weight[:, middle:].float())
        if module.bias is not None:
            result += module.bias.float()
        counts[0] += 1
        return result.bfloat16()
    with patch.object(torch.nn.Linear, 'forward', forward):
        yield counts


def fast_schedules(length):
    return [list(v) for v in dict.fromkeys((
        (length,), tuple([1]*length if length <= 17 else [length-17, 16, 1])))]


def prediction_metrics(actual, expected):
    """KL(reference || actual), total variation and greedy-margin evidence."""
    a, b = actual.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    if a.shape != b.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid prediction logits')
    def log_softmax(x):
        shifted = x-x.max()
        return shifted-np.log(np.exp(shifted).sum())
    la, lb = log_softmax(a), log_softmax(b)
    pa, pb = np.exp(la), np.exp(lb)
    return dict(kl_nats=max(0., float(np.sum(pb*(lb-la)))),
        total_variation=float(np.abs(pa-pb).sum()/2),
        same_token=bool(np.argmax(a) == np.argmax(b)),
        reference_margin=float(b.max()-np.partition(b, -2)[-2]),
        reference_max_abs=float(np.abs(b).max()))


def fast_cases(phase, declaration, tokenizer):
    settings = declaration[phase]
    for length in settings['lengths']:
        seed = settings['seed']+length
        yield f'random-{length}', np.random.default_rng(seed).integers(0, 151643, size=length).tolist()
    for index, text in enumerate(settings['texts']):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not 1 <= len(ids) <= 4096:
            raise ValueError('text case exceeds declared model capacity')
        yield f'text-{index}', ids


def verify_fast_census(records, cases, declaration, phases):
    from model_consistency import BOUNDARIES
    expected = Counter()
    for phase in phases:
        for case, ids in cases[phase]:
            for schedule in fast_schedules(len(ids)):
                for arm in declaration['reference_arms']:
                    start = 0
                    for rows in schedule:
                        for stage in BOUNDARIES:
                            expected[(phase, case, len(ids), arm, tuple(schedule), start, rows, stage)] += 1
                        start += rows
    actual = Counter((r['phase'], r['case'], r['length'], r['arm'], tuple(r['schedule']),
                      r['start'], r['rows'], r['stage']) for r in records)
    if actual != expected:
        raise ValueError('incomplete or duplicated Fast reference observation census')


@torch.no_grad()
def fast_observations(model, phase, declaration, tokenizer):
    from model_consistency import calls, BOUNDARIES
    from model_attention_diagnosis import exact_array
    for case, ids in fast_cases(phase, declaration, tokenizer):
        for schedule in fast_schedules(len(ids)):
            control = list(calls(model, ids, schedule, canonical=False))
            for arm in declaration['reference_arms']:
                with split_linear() if arm == 'fp32_split2' else nullcontext() as counts:
                    for expected_call, actual_call in zip(control, calls(model, ids, schedule,
                            canonical=arm == 'hf_canonical'), strict=True):
                        start, rows, actual = actual_call
                        if expected_call[:2] != actual_call[:2] or set(actual) != BOUNDARIES:
                            raise ValueError('incomplete arithmetic reference coverage')
                        for name in sorted(BOUNDARIES):
                            expected = expected_call[2][name]
                            row = dict(phase=phase, case=case, length=len(ids), arm=arm,
                                schedule=schedule, start=start, rows=rows, stage=name,
                                **metrics(actual[name], expected, declaration['rtol']))
                            row['exact'] = exact_array(actual[name], expected)
                            if name == 'logits':
                                row.update(prediction_metrics(actual[name], expected))
                            yield row
                    if counts is not None and counts[0] != len(schedule)*169:
                        raise ValueError('missing affine reference coverage')
                print('Fast reference', phase, case, arm, 'calls', len(schedule), flush=True)


def fast_budgets(records, declaration):
    from model_consistency import BOUNDARIES
    if {r['stage'] for r in records} != BOUNDARIES:
        raise ValueError('incomplete calibration boundary census')
    grouped = {}
    for role in declaration['maximum_atol']:
        selected = [r for r in records if boundary_role(r['stage']) == role]
        gate = dict(rtol=declaration['rtol'], exact=False)
        for metric, field in [('required_atol', 'atol'), ('relative_rms', 'relative_rms')]:
            value = declaration['margin']*max(r[metric] for r in selected)
            quantum = declaration[field+'_quantum']
            gate[field] = max(declaration[field+'_floor'], math.ceil(value/quantum)*quantum)
        grouped[role] = gate
    grouped['embedding'] = dict(rtol=0., atol=0., relative_rms=0., exact=True)
    return {name: dict(grouped[boundary_role(name)]) for name in sorted(BOUNDARIES)}


def fast_accepted(row, gate, declaration):
    if not accepted(row, gate):
        return False
    if row['stage'] != 'logits':
        return True
    criteria = declaration['prediction']
    required_margin = 2*(gate['atol']+gate['rtol']*row['reference_max_abs'])
    return (row['kl_nats'] <= criteria['maximum_kl_nats']
        and row['total_variation'] <= criteria['maximum_total_variation']
        and (row['reference_margin'] <= required_margin or row['same_token']))


def fast_identity():
    import inspect
    import model_consistency
    import model_attention_diagnosis
    paths = [Path(__file__), FAST_DECLARATION_PATH, Path(reference.__file__),
        Path(model_consistency.__file__), Path(model_attention_diagnosis.__file__),
        Path(__file__).with_suffix('.py.lock'), model_attention_diagnosis.CONTRACT,
        Path(inspect.getfile(torch.nn.Linear))]
    return {str(p.relative_to(reference.ROOT)) if p.is_relative_to(reference.ROOT) else p.name:
            reference.sha(p) for p in paths}


def qualify_fast(output):
    from transformers import AutoTokenizer
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=reference.ROOT, text=True):
        raise ValueError('freeze and commit clean reference source before qualification')
    declaration = json.loads(FAST_DECLARATION_PATH.read_text())
    if declaration['reference_arms'] != ['hf_canonical', 'fp32_split2']:
        raise ValueError('unsupported reference arithmetic declaration')
    source = fast_identity()
    reference.verify_assets()
    from model_attention_diagnosis import CONTRACT as backend_contract
    backend = json.loads(backend_contract.read_text())
    if (reference.provenance()['upstream_sha256'] != backend['source_files']['modeling_qwen2.py']['sha256']
            or torch.version.git_version != backend['torch_git_version']):
        raise ValueError('upstream reference source differs from pinned backend contract')
    tokenizer_hashes = {name: reference.sha(reference.ASSETS/name) for name in
        ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}
    if tokenizer_hashes != {
        'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
        'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
        'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
        'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'}:
        raise ValueError('tokenizer reference artifact identity mismatch')
    model = reference.load_model()
    tokenizer = AutoTokenizer.from_pretrained(reference.ASSETS, local_files_only=True)
    cases = {phase: list(fast_cases(phase, declaration, tokenizer)) for phase in ('calibration', 'confirmation')}
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(kind='fast-reference-qualification', declaration=declaration,
        source=source, reference=reference.provenance(), tokenizer_sha256=tokenizer_hashes,
        cases=cases,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=reference.ROOT, text=True).strip(),
        new_candidate_outputs_used_for_calibration=False, reserved_outputs_observed=False)
    started_at = time.time()
    records = []
    with (output/'observations.jsonl').open('x') as stream:
        def observe(phase):
            for row in fast_observations(model, phase, declaration, tokenizer):
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush()
                records.append(row)
        observe('calibration')
        verify_fast_census(records, cases, declaration, ['calibration'])
        gates = fast_budgets(records, declaration)
        ceiling_failures = [name for name, gate in gates.items() if boundary_role(name) != 'embedding'
            and (gate['atol'] > declaration['maximum_atol'][boundary_role(name)]
                 or gate['relative_rms'] > declaration['maximum_qualified_relative_rms'])]
        calibration_passed = not ceiling_failures and all(fast_accepted(r, gates[r['stage']], declaration) for r in records)
        frozen_path = output/'frozen-budgets.json'
        frozen_path.write_text(json.dumps(dict(**identity, gates=gates,
            calibration_passed=calibration_passed), indent=2, allow_nan=False)+'\n')
        frozen_sha = reference.sha(frozen_path)
        frozen_at = time.time()
        if calibration_passed:
            observe('confirmation')
            verify_fast_census(records, cases, declaration, ['calibration', 'confirmation'])
    if reference.sha(frozen_path) != frozen_sha or source != fast_identity():
        raise ValueError('Fast reference source/declaration/budgets changed during collection')
    failures = [r for r in records if not fast_accepted(r, gates[r['stage']], declaration)]
    passed = calibration_passed and not failures
    report = dict(**identity, gates=gates, ceiling_failures=ceiling_failures,
        started_at=started_at, frozen_at=frozen_at, finished_at=time.time(),
        calibration_passed=calibration_passed, confirmation_executed=calibration_passed,
        frozen_budgets_sha256=frozen_sha, checks=len(records), failures=failures, passed=passed,
        observations_sha256=reference.sha(output/'observations.jsonl'))
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not passed:
        raise ValueError('Fast reference qualification failed; stop dependent native acceptance')
    print('Fast reference calibration and independent confirmation passed.', flush=True)


def capture_fast(output, qualification, ids_path, schedule_text):
    """Capture corresponding-schedule HF arrays only after full qualification."""
    from model_consistency import calls
    from transformers import AutoTokenizer
    report = json.loads(qualification.read_text())
    declaration = json.loads(FAST_DECLARATION_PATH.read_text())
    if (report.get('kind') != 'fast-reference-qualification' or report.get('passed') is not True
            or report.get('calibration_passed') is not True or report.get('confirmation_executed') is not True
            or report.get('new_candidate_outputs_used_for_calibration') is not False
            or report.get('reserved_outputs_observed') is not False
            or report.get('source') != fast_identity() or report.get('declaration') != declaration
            or report.get('reference') != reference.provenance()):
        raise ValueError('missing compatible passing Fast reference qualification')
    for name, field in [('observations.jsonl', 'observations_sha256'),
                        ('frozen-budgets.json', 'frozen_budgets_sha256')]:
        if reference.sha(qualification.parent/name) != report[field]:
            raise ValueError('Fast qualification evidence hash mismatch')
    records = [json.loads(line) for line in (qualification.parent/'observations.jsonl').read_text().splitlines()]
    reference.verify_assets()
    for name, digest in report['tokenizer_sha256'].items():
        if reference.sha(reference.ASSETS/name) != digest:
            raise ValueError('tokenizer changed since qualification')
    tokenizer = AutoTokenizer.from_pretrained(reference.ASSETS, local_files_only=True)
    cases = {phase: list(fast_cases(phase, declaration, tokenizer)) for phase in ('calibration', 'confirmation')}
    verify_fast_census(records, cases, declaration, ['calibration', 'confirmation'])
    gates = fast_budgets([r for r in records if r['phase'] == 'calibration'], declaration)
    frozen = json.loads((qualification.parent/'frozen-budgets.json').read_text())
    if (gates != report['gates'] or gates != frozen['gates'] or len(records) != report['checks']
            or any(not fast_accepted(r, gates[r['stage']], declaration) for r in records)
            or any(g['relative_rms'] > declaration['maximum_qualified_relative_rms']
                or (boundary_role(name) != 'embedding'
                    and g['atol'] > declaration['maximum_atol'][boundary_role(name)])
                for name, g in gates.items())):
        raise ValueError('Fast qualification does not replay')
    ids = json.loads(ids_path.read_text())
    if not isinstance(ids, list) or not 1 <= len(ids) <= 4096 or any(
            type(i) is not int or not 0 <= i < 151936 for i in ids):
        raise ValueError('invalid reference input IDs')
    schedule = [int(x) for x in schedule_text.split(',')]
    if not schedule or min(schedule) < 1 or sum(schedule) != len(ids):
        raise ValueError('invalid reference input schedule')
    source = fast_identity()
    model = reference.load_model()
    output.mkdir(parents=True, exist_ok=False)
    captured = []
    for index, (start, rows, values) in enumerate(calls(model, ids, schedule, canonical=False)):
        directory = output/f'call_{index}'
        directory.mkdir()
        arrays = {}
        for name, value in values.items():
            path = directory/(name+'.npy')
            np.save(path, value, allow_pickle=False)
            arrays[name] = dict(path=str(path.relative_to(output)), shape=list(value.shape), sha256=reference.sha(path))
        captured.append(dict(start=start, rows=rows, arrays=arrays))
    if source != fast_identity():
        raise ValueError('reference source changed during capture')
    manifest = dict(kind='fast-model-reference', source=source, reference=reference.provenance(),
        qualification_sha256=reference.sha(qualification), gates=gates,
        prediction=declaration['prediction'], ids=ids, schedule=schedule, calls=captured)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


def metrics(actual, expected, rtol):
    if (actual.shape != expected.shape or actual.ndim < 2
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError('nonfinite or mismatched reference boundary')
    error = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(error)
    delta = np.linalg.norm(error.reshape(len(error), -1), axis=1)
    signal = np.linalg.norm(expected.astype(np.float64).reshape(len(error), -1), axis=1)
    relative = np.divide(delta, signal, out=np.zeros_like(delta), where=signal != 0)
    if np.any((signal == 0) & (delta != 0)):
        raise ValueError('nonzero error at zero-signal reference row')
    return dict(max_abs=float(absolute.max()),
                required_atol=max(0., float((absolute - rtol * np.abs(expected)).max())),
                relative_rms=float(relative.max()), exact=bool(np.array_equal(actual, expected)))


@torch.no_grad()
def observations(model, phase, declaration):
    for length in declaration[phase]['lengths']:
        seed = declaration[phase]['seed'] + length
        ids = np.random.default_rng(seed).integers(0, 151643, size=length).tolist()
        full = reference.forward(model, ids, [length], all_logits=False)[0][2]
        schedule = [1] * length if length <= 17 else [length - 17, 16, 1]
        for start, rows, values in reference.forward(model, ids, schedule, all_logits=False):
            if len(values) != 75 or set(values) != set(full):
                raise ValueError('incomplete reference boundary census')
            for name, actual in values.items():
                if name.startswith('cache_'):
                    expected = full[name][:start + rows]
                elif name == 'logits':
                    # Only project the endpoint needed for next-token inference.
                    # This avoids a 4096 x vocabulary diagnostic allocation.
                    endpoint = torch.from_numpy(full['final_norm'][start + rows - 1:start + rows]).to(torch.bfloat16)
                    expected = model.lm_head(endpoint).float().numpy()
                else:
                    expected = full[name][start:start + rows]
                yield dict(phase=phase, length=length, seed=seed, start=start,
                           rows=rows, stage=name, **metrics(actual, expected, declaration['rtol']))
        print(phase, 'reference schedules complete:', length, flush=True)


def budgets(records, declaration):
    result = {}
    for name in sorted({r['stage'] for r in records}):
        stage = [r for r in records if r['stage'] == name]
        gate = dict(rtol=declaration['rtol'])
        for metric, field in [('required_atol', 'atol'), ('relative_rms', 'relative_rms')]:
            value = declaration['margin'] * max(r[metric] for r in stage)
            quantum = declaration[field + '_quantum']
            gate[field] = max(declaration[field + '_floor'], math.ceil(value / quantum) * quantum)
        gate['exact'] = name == 'hidden_0'
        result[name] = gate
    if len(result) != 75:
        raise ValueError('incomplete calibration census')
    return result


def accepted(record, gate):
    return (record['required_atol'] <= gate['atol']
            and record['relative_rms'] <= gate['relative_rms']
            and (not gate['exact'] or record['exact']))


def self_test():
    """Check rejection independently of the checkpoint and model code."""
    import unittest

    class MetricsTest(unittest.TestCase):
        def test_failed_qualification_prevents_reference_capture(self):
            import tempfile
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = root/'result.json'
                result.write_text(json.dumps(dict(kind='fast-reference-qualification', passed=False)))
                with patch.object(reference, 'load_model') as load:
                    with self.assertRaisesRegex(ValueError, 'passing Fast reference'):
                        capture_fast(root/'capture', result, root/'ids.json', '1')
                    load.assert_not_called()
                self.assertFalse((root/'capture').exists())

        def test_split_affine_preserves_equation_bias_and_bf16_boundary(self):
            module = torch.nn.Linear(5, 2, bias=True).bfloat16()
            with torch.no_grad():
                module.weight.copy_(torch.tensor([[1,2,-3,4,5],[-1,3,2,0,-4]]))
                module.bias.copy_(torch.tensor([0.5,-0.25]))
            x = torch.tensor([[[2.,-1.,3.,0.5,1.]]]).bfloat16()
            expected = (x.double() @ module.weight.double().T + module.bias.double()).bfloat16()
            original = torch.nn.Linear.forward
            with split_linear() as count:
                self.assertTrue(torch.equal(module(x), expected))
                self.assertEqual(count[0], 1)
            self.assertIs(torch.nn.Linear.forward, original)

        def test_prediction_checks_detect_changed_distribution_and_winner(self):
            declaration = json.loads(FAST_DECLARATION_PATH.read_text())
            expected = np.array([[8.,0.,-2.]])
            actual = np.array([[0.,8.,-2.]])
            row = dict(stage='logits', **metrics(actual, expected, 0.),
                **prediction_metrics(actual, expected))
            self.assertFalse(fast_accepted(row,
                dict(atol=100., relative_rms=100., exact=False, rtol=0.), declaration))
            # A constant shift leaves probabilities intact but remains subject
            # to the independently required numerical check.
            shifted = expected+100.
            row = dict(stage='logits', **metrics(shifted, expected, 0.),
                **prediction_metrics(shifted, expected))
            self.assertLess(row['total_variation'], 1e-12)
            self.assertFalse(fast_accepted(row,
                dict(atol=1., relative_rms=1., exact=False, rtol=0.), declaration))

        def test_role_budgets_preserve_embeddings_and_reject_omitted_stage(self):
            from model_consistency import BOUNDARIES
            declaration = json.loads(FAST_DECLARATION_PATH.read_text())
            records = [dict(stage=name, required_atol=0.125 if name == 'hidden_24' else 0.,
                relative_rms=0.01) for name in BOUNDARIES]
            gates = fast_budgets(records, declaration)
            self.assertEqual(gates['hidden_1'], gates['hidden_24'])
            self.assertTrue(gates['hidden_0']['exact'])
            self.assertEqual(gates['hidden_0']['atol'], 0.)
            with self.assertRaisesRegex(ValueError, 'census'):
                fast_budgets(records[:-1], declaration)

        def test_census_rejects_missing_schedule_and_duplicate_records(self):
            from model_consistency import BOUNDARIES
            declaration = json.loads(FAST_DECLARATION_PATH.read_text())
            cases = {'calibration': [('small', [1,2])]}
            records = []
            for schedule in fast_schedules(2):
                for arm in declaration['reference_arms']:
                    start = 0
                    for rows in schedule:
                        records.extend(dict(phase='calibration', case='small', length=2,
                            arm=arm, schedule=schedule, start=start, rows=rows, stage=stage)
                            for stage in BOUNDARIES)
                        start += rows
            verify_fast_census(records, cases, declaration, ['calibration'])
            for damaged in (records[:-1], records+[records[0]],
                    [r for r in records if r['schedule'] == [2]]):
                with self.assertRaisesRegex(ValueError, 'census'):
                    verify_fast_census(damaged, cases, declaration, ['calibration'])

        def test_split_patch_covers_actual_hf_decoder_and_head(self):
            from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
            from model_consistency import calls
            torch.set_num_threads(1)
            torch.manual_seed(410103)
            config = Qwen2Config(hidden_size=16, intermediate_size=37,
                num_hidden_layers=24, num_attention_heads=4, num_key_value_heads=2,
                vocab_size=64, max_position_embeddings=64, attention_dropout=0.)
            config._attn_implementation = 'sdpa'
            model = reference.Qwen2ForCausalLM(config).bfloat16().eval()
            with split_linear() as count:
                captured = list(calls(model, [1,2,3,4], [3,1], canonical=False))
            self.assertEqual(count[0], 338)
            self.assertEqual(len(captured), 2)
            self.assertEqual(captured[1][2]['cache_key_23'].shape, (4,2,4))

        def test_small_rows_are_not_hidden_by_large_rows(self):
            expected = np.array([[100., 100.], [1., 1.]])
            actual = expected + np.array([[0., 0.], [0.125, 0.125]])
            row = metrics(actual, expected, 0.03125)
            self.assertEqual(row['relative_rms'], 0.125)
            self.assertFalse(accepted(row, dict(atol=1., relative_rms=0.0625, exact=False)))

        def test_exact_and_invalid_boundaries(self):
            expected = np.array([[0., 1.]])
            row = metrics(expected, expected, 0.03125)
            self.assertTrue(accepted(row, dict(atol=0., relative_rms=0., exact=True)))
            different = metrics(expected + 0.01, expected, 0.03125)
            self.assertFalse(accepted(different, dict(atol=1., relative_rms=1., exact=True)))
            for actual in (np.zeros((1, 1)), expected * float('nan')):
                with self.assertRaises(ValueError):
                    metrics(actual, expected, 0.03125)
            with self.assertRaises(ValueError):
                metrics(np.ones((1, 2)), np.zeros((1, 2)), 0.03125)

    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(MetricsTest))
    if not result.wasSuccessful():
        raise ValueError('calibration metric self-test failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--fast', action='store_true', help='qualify the frozen Fast arithmetic contract')
    parser.add_argument('--fast-capture', action='store_true')
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--capture-ids', type=Path)
    parser.add_argument('--capture-schedule')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error('--output is required')
    if args.fast_capture:
        if args.fast or not args.qualification or not args.capture_ids or not args.capture_schedule:
            parser.error('Fast capture requires qualification, input IDs and schedule, without --fast')
        capture_fast(args.output, args.qualification, args.capture_ids, args.capture_schedule)
        return
    if args.fast:
        qualify_fast(args.output)
        return
    declaration = json.loads(DECLARATION_PATH.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(reference=reference.provenance(), source_sha256=reference.sha(__file__),
                    declaration_sha256=reference.sha(DECLARATION_PATH), declaration=declaration,
                    candidate_outputs_observed=False, reserved_outputs_observed=False)
    reference.verify_assets()
    model = reference.load_model()
    records = []
    with (args.output / 'observations.jsonl').open('x') as stream:
        def observe(phase):
            for row in observations(model, phase, declaration):
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                records.append(row)
        observe('calibration')
        gates = budgets(records, declaration)
        ceiling_passed = all(g['relative_rms'] <= declaration['maximum_qualified_relative_rms']
                             for g in gates.values())
        ceiling_passed &= all(accepted(r, gates[r['stage']]) for r in records)
        frozen = dict(**identity, gates=gates, calibration_passed=ceiling_passed)
        freeze_path = args.output / 'frozen-budgets.json'
        freeze_path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + '\n')
        freeze_sha = reference.sha(freeze_path)
        if ceiling_passed:
            observe('confirmation')
        if reference.sha(freeze_path) != freeze_sha:
            raise ValueError('budgets changed during confirmation')
    if (identity['source_sha256'] != reference.sha(__file__)
            or identity['declaration_sha256'] != reference.sha(DECLARATION_PATH)
            or identity['reference'] != reference.provenance()):
        raise ValueError('reference source or declaration changed during execution')
    passed = ceiling_passed and all(accepted(r, gates[r['stage']]) for r in records)
    result = dict(**identity, gates=gates, calibration_passed=ceiling_passed,
                  confirmation_executed=ceiling_passed, passed=passed,
                  frozen_budgets_sha256=freeze_sha,
                  observations_sha256=reference.sha(args.output / 'observations.jsonl'),
                  ceiling_failures=[name for name, gate in gates.items()
                                    if gate['relative_rms'] > declaration['maximum_qualified_relative_rms']],
                  checks=len(records), failures=[r for r in records if not accepted(r, gates[r['stage']])])
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if not passed:
        raise ValueError('bounded reference calibration/confirmation failed; stop dependent acceptance')
    print('Reference-only calibration and independent confirmation passed.', flush=True)


if __name__ == '__main__':
    main()
