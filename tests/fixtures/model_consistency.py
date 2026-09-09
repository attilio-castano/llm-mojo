# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==1.26.4", "torch==2.4.0", "transformers==4.43.1"]
# ///
"""Stream exact canonical HF schedule qualification; no native outputs read."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

import model_reference as reference
from model_attention_diagnosis import canonical_queries, exact_array, verify_sources, CONTRACT as BACKEND_CONTRACT

CONTRACT = Path(__file__).with_suffix('.json')
BOUNDARIES = ({f'hidden_{i}' for i in range(25)} | {'final_norm', 'logits'} |
              {f'cache_{kind}_{i}' for kind in ('key', 'value') for i in range(24)})


def schedules(length, declaration):
    settings = declaration['schedules']
    values = [[length]]
    if length <= settings['exhaustive_through']:
        for mask in range(1 << (length - 1)):
            cuts = [0] + [i for i in range(1, length) if mask & (1 << (i - 1))] + [length]
            values.append([b-a for a, b in zip(cuts, cuts[1:])])
    elif length <= settings['tokenwise_through']:
        values.append([1] * length)
    if length > 17:
        values.append([length - sum(settings['long_suffix']), *settings['long_suffix']])
    if length > 2:
        values.append([1, length-2, 1])
    remaining, ragged, index = length, [], 0
    while remaining:
        rows = min(remaining, settings['ragged_cycle'][index % len(settings['ragged_cycle'])])
        ragged.append(rows)
        remaining -= rows
        index += 1
    values.append(ragged)
    return [list(v) for v in dict.fromkeys(tuple(v) for v in values)]


def source_identity():
    paths = [Path(__file__), CONTRACT, Path(reference.__file__),
             Path(__file__).with_name('model_attention_diagnosis.py'), BACKEND_CONTRACT,
             Path(__file__).with_name('generate.py.lock')]
    return {str(p.relative_to(reference.ROOT)): reference.sha(p) for p in paths}


def calls(model, ids, schedule, canonical=True):
    """Yield one copied boundary set at a time; cache itself persists in HF."""
    if not schedule or min(schedule) < 1 or sum(schedule) != len(ids) or len(ids) > 4096:
        raise ValueError('invalid schedule')
    offset, cache = 0, None
    with torch.no_grad(), canonical_queries() if canonical else nullcontext():
        for rows in schedule:
            positions = torch.arange(offset, offset + rows)
            mask = torch.zeros((1, 1, rows, offset + rows), dtype=torch.bfloat16)
            mask.masked_fill_(torch.arange(offset + rows)[None] > positions[:, None], float('-inf'))
            last = []
            handle = model.model.layers[23].register_forward_hook(
                lambda module, args, result: last.append(result[0][0].float().numpy().copy()))
            try:
                with reference.precision_policy() as count:
                    result = model.model(input_ids=torch.tensor([ids[offset:offset+rows]]),
                        attention_mask=mask, position_ids=positions[None], past_key_values=cache,
                        use_cache=True, output_hidden_states=True, return_dict=True)
            finally:
                handle.remove()
            if count[0] != 24 or len(last) != 1:
                raise ValueError('incomplete upstream layer coverage')
            cache = result.past_key_values
            values = {f'hidden_{i}': h[0].float().numpy().copy()
                      for i, h in enumerate(result.hidden_states)}
            values['final_norm'] = values.pop('hidden_24')
            values['hidden_24'] = last[0]
            values['logits'] = model.lm_head(result.last_hidden_state[:, -1:])[0].float().numpy().copy()
            for layer, (key, value) in enumerate(cache):
                values[f'cache_key_{layer}'] = key[0].transpose(0, 1).float().numpy().copy()
                values[f'cache_value_{layer}'] = value[0].transpose(0, 1).float().numpy().copy()
            if set(values) != BOUNDARIES or any(not np.isfinite(v).all() for v in values.values()):
                raise ValueError('nonfinite or incomplete upstream capture')
            yield offset, rows, values
            offset += rows


def expected_rows(model, full, name, start, rows):
    if name.startswith('cache_'):
        return full[name][:start+rows]
    if name == 'logits':
        with torch.no_grad():
            endpoint = torch.from_numpy(full['final_norm'][start+rows-1:start+rows]).bfloat16()
            return model.lm_head(endpoint).float().numpy()
    return full[name][start:start+rows]


def qualify(model, output, declaration):
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=reference.ROOT, text=True).strip()
    results = []
    with (output/'observations.jsonl').open('w') as stream:
        for spec in declaration['development_cases']:
            length = spec['length']
            ids = np.random.default_rng(spec['seed']).integers(0, 151643, size=length).tolist()
            full = next(calls(model, ids, [length]))[2]
            directory = output/f'length_{length}'
            directory.mkdir()
            arrays = {}
            for name, value in full.items():
                path = directory/(name+'.npy')
                np.save(path, value, allow_pickle=False)
                arrays[name] = dict(path=str(path.relative_to(output)), shape=list(value.shape), sha256=reference.sha(path))
            summary = dict(**spec, ids=ids, arrays=arrays, schedules=[])
            for schedule in schedules(length, declaration):
                failures, checks = [], 0
                for start, rows, actual in calls(model, ids, schedule):
                    for name in sorted(BOUNDARIES):
                        expected = expected_rows(model, full, name, start, rows)
                        exact = exact_array(expected, actual[name])
                        row = dict(length=length, seed=spec['seed'], schedule=schedule,
                                   start=start, rows=rows, stage=name, exact=exact,
                                   max_abs=float(np.abs(expected-actual[name]).max()))
                        stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                        checks += 1
                        if not exact:
                            failures.append(dict(start=start, stage=name, max_abs=row['max_abs']))
                stream.flush()
                summary['schedules'].append(dict(rows=schedule, checks=checks, failures=failures))
                print('canonical reference', length, 'calls', len(schedule), 'checks', checks,
                      'failures', len(failures), flush=True)
            results.append(summary)
            del full
    if identity != source_identity():
        raise ValueError('reference source changed during qualification')
    passed = all(not s['failures'] for c in results for s in c['schedules'])
    report = dict(kind='model_consistency_reference', version=declaration['version'],
        source_commit=commit, source=identity, reference=reference.provenance(),
        candidate_outputs_observed=False, reserved_outputs_observed=False,
        observations_sha256=reference.sha(output/'observations.jsonl'), cases=results, passed=passed)
    (output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise ValueError('canonical reference schedule qualification failed')


def self_test():
    import unittest
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    declaration = json.loads(CONTRACT.read_text())
    class StreamingTests(unittest.TestCase):
        def test_schedules(self):
            self.assertEqual(len(schedules(4, declaration)), 8)
            for spec in declaration['development_cases']:
                for schedule in schedules(spec['length'], declaration):
                    self.assertEqual(sum(schedule), spec['length'])
                    self.assertGreater(min(schedule), 0)
        def test_stream_matches_original_observer(self):
            torch.set_num_threads(1)
            torch.manual_seed(18803)
            config = Qwen2Config(hidden_size=16, intermediate_size=37, num_hidden_layers=24,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                max_position_embeddings=64, attention_dropout=0.)
            config._attn_implementation = 'sdpa'
            model = reference.Qwen2ForCausalLM(config).bfloat16().eval()
            ids = [1, 2, 3, 2]
            old = reference.forward(model, ids, [1, 2, 1], all_logits=False)
            for a, b in zip(old, calls(model, ids, [1, 2, 1], canonical=False), strict=True):
                self.assertEqual(a[:2], b[:2])
                self.assertEqual(set(a[2]), BOUNDARIES)
                for name in BOUNDARIES:
                    self.assertTrue(exact_array(a[2][name], b[2][name]), name)
            full = next(calls(model, ids, [4]))[2]
            for schedule in schedules(4, declaration):
                for start, rows, actual in calls(model, ids, schedule):
                    for name in BOUNDARIES:
                        self.assertTrue(exact_array(expected_rows(model, full, name, start, rows), actual[name]), name)
        def test_signed_zero_is_not_byte_equal(self):
            self.assertFalse(exact_array(np.array([0.], dtype='f'), np.array([-0.], dtype='f')))
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(StreamingTests))
    if not result.wasSuccessful():
        raise ValueError('streaming reference self-test failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None:
        parser.error('--output is required')
    declaration = json.loads(CONTRACT.read_text())
    if reference.sha(reference.ROOT/declaration['accuracy']['source']) != declaration['accuracy']['sha256']:
        raise ValueError('frozen numerical gate source changed')
    reference.verify_assets()
    verify_sources(json.loads(BACKEND_CONTRACT.read_text()))
    qualify(reference.load_model(), args.output, declaration)


if __name__ == '__main__':
    main()
