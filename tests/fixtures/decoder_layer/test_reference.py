"""Executed only in the pinned script environment by --self-test."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from contract import QWEN, TINY, STAGES, GATES, schedules, case_spec
from reference import UpstreamDecoder, inputs, check_capture, same_bits, differences
import generate


class ContractTests(unittest.TestCase):
    def test_schedules_partition_inputs_and_keep_positions(self):
        for t in (1, 7, 15, 16, 17, 33, 65, 257, 1024, 4096):
            for chunks in schedules(t).values():
                self.assertEqual(sum(chunks), t)
                self.assertTrue(all(r > 0 for r in chunks))
        self.assertEqual(schedules(33)['threshold'], [16, 1, 15, 1])
        self.assertEqual(schedules(65)['reuse'], [53]+[1]*12)

    def test_rng_prefix_and_weight_identity(self):
        small = inputs(case_spec(TINY[0], 7, 4001))
        large = inputs(case_spec(TINY[0], 17, 4001))
        self.assertTrue(same_bits(small['X'], large['X'][:7]))
        for k in small.keys()-{'X'}:
            self.assertTrue(same_bits(small[k], large[k]))
        self.assertFalse(same_bits(small['input_norm'], small['post_norm']))
        self.assertFalse(same_bits(small['gate'], small['up']))

    def test_numerical_gate_rejects_invalid_values_and_zero_sign(self):
        zero = np.array([0], np.float32)
        for other in (np.array([np.nan]), np.zeros(2)):
            with self.assertRaises(ValueError):
                differences(zero, other)
        self.assertEqual(differences(zero, -zero, {'atol':0, 'rtol':0, 'exact_bits':True})['failed'], 1)
        self.assertEqual(set(GATES), {'B_att', 'Z', 'B_mlp', 'Y'})


class CaptureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.spec = case_spec(TINY[0], 7, 4001)
        self.data = inputs(self.spec)

    def test_observation_does_not_change_layer_or_cache(self):
        runner = UpstreamDecoder(self.spec, self.data)
        actual = runner.run(self.data['X'])
        plain = UpstreamDecoder(self.spec, self.data).run(self.data['X'], observe=False)
        self.assertEqual(set(actual), set(STAGES))
        for k in ('Y', 'cache_key', 'cache_value'):
            self.assertTrue(same_bits(actual[k], plain[k]), k)
        self.assertEqual(runner.sdpa_calls, 1)
        for module in runner.module.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_reconstruction_and_shapes_detect_corruption(self):
        captured = UpstreamDecoder(self.spec, self.data).run(self.data['X'])
        for key in ('Z', 'S', 'Y'):
            bad = {k:v.copy() for k,v in captured.items()}
            bad[key].flat[0] += 1
            with self.assertRaises(ValueError):
                check_capture(bad, self.spec, 0, 7)
        for change in ('missing', 'shape', 'nonfinite'):
            bad = {k:v.copy() for k,v in captured.items()}
            if change == 'missing':
                del bad['A']
            elif change == 'shape':
                bad['A'] = bad['A'][:1]
            else:
                bad['A'].flat[0] = np.nan
            with self.assertRaises(ValueError):
                check_capture(bad, self.spec, 0, 7)

    def test_failure_removes_observation_hooks(self):
        runner = UpstreamDecoder(self.spec, self.data)
        def corrupt(module, args):
            return (args[0]+1,)
        handle = runner.module.mlp.down_proj.register_forward_pre_hook(corrupt)
        try:
            with self.assertRaises(ValueError):
                runner.run(self.data['X'])
        finally:
            handle.remove()
        for module in runner.module.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_cached_schedule_uses_original_input_and_exact_append(self):
        full = UpstreamDecoder(self.spec, self.data).run(self.data['X'])
        runner = UpstreamDecoder(self.spec, self.data)
        for p in range(7):
            old = None if p == 0 else result['cache_key'].copy()
            result = runner.run(self.data['X'][p:p+1])
            if old is not None:
                self.assertTrue(same_bits(old, result['cache_key'][:p]))
            for name, gate in GATES.items():
                self.assertEqual(differences(result[name], full[name][p:p+1], gate)['failed'], 0)

    def test_zero_branches_and_weight_swaps_discriminate(self):
        data = copy.deepcopy(self.data)
        data['wo'].fill(0)
        data['down'].fill(0)
        out = UpstreamDecoder(self.spec, data).run(data['X'])
        self.assertTrue(same_bits(out['Y'], data['X']))
        full = UpstreamDecoder(self.spec, self.data).run(self.data['X'])
        for a,b in [('gate','up'), ('input_norm','post_norm')]:
            data = copy.deepcopy(self.data)
            data[a],data[b] = data[b],data[a]
            changed = UpstreamDecoder(self.spec, data).run(data['X'])
            self.assertFalse(same_bits(full['Y'], changed['Y']))


class EvidenceTests(unittest.TestCase):
    def test_compact_freeze_round_trip_and_corruption_rejection(self):
        record=dict(status='complete',cases={'tiny':{}},specification={},upstream={},sources={},
                    checkpoint={'tensors':{'norm':{'shape':[8]}}})
        with tempfile.TemporaryDirectory() as tmp:
            anchor=Path(tmp)/'checksums.json'; evidence=Path(tmp)/'development.json.gz'
            with patch.object(generate,'ANCHOR',anchor),patch.object(generate,'EVIDENCE',evidence):
                generate.freeze(record)
                self.assertEqual(generate.load_frozen(),record)
                with self.assertRaises(ValueError):
                    generate.freeze(record)
                summary=json.loads(anchor.read_text()); summary['cases']=2
                generate.write_json(anchor,summary)
                with self.assertRaises(ValueError):
                    generate.load_frozen()
                summary['cases']=1; generate.write_json(anchor,summary)
                evidence.write_bytes(evidence.read_bytes()+b'changed')
                with self.assertRaises(ValueError):
                    generate.load_frozen()

    def test_array_verification_rejects_mutated_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'tiny').mkdir()
            p=root/'tiny'/'X.npy'; np.save(p,np.ones(1,np.float32))
            record={'cases':{'tiny':{'arrays':{'X':{'sha256':generate.sha(p)}}}}}
            generate.verify_arrays(root,record)
            np.save(p,np.zeros(1,np.float32))
            with self.assertRaises(ValueError):
                generate.verify_arrays(root,record)


if __name__ == '__main__':
    unittest.main()
