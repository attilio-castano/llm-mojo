# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==1.26.4", "torch==2.4.0", "transformers==4.43.1"]
# ///
"""Bounded upstream-only schedule calibration and independent confirmation.

No native executable is launched and no Mojo output is read. A passing report
is evidence for reviewing the full-model contract, not model acceptance.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import model_reference as reference

DECLARATION_PATH = Path(__file__).with_suffix('.json')


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
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error('--output is required')
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
