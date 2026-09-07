"""Run in the pinned oracle environment via generate.py mlp -- --self-test."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from numerics import round_bf16, bf16_bits, from_bits, differences
from reference import inputs, mutate, UpstreamMLP, same_bits, activation_probe, projection_tail_probe
from generate import validate_anchors, write_json, sha


class RoundingTests(unittest.TestCase):
    def test_all_finite_bf16_patterns_round_trip_including_signed_zero(self):
        bits = np.arange(65536, dtype=np.uint16)
        bits = bits[(bits & 0x7f80) != 0x7f80]
        np.testing.assert_array_equal(bf16_bits(round_bf16(from_bits(bits).astype(np.float64))), bits)

    def test_midpoints_and_double_rounding_regression(self):
        mid = np.float64(1 + 2**-8)
        values = np.array([np.nextafter(mid, 0), mid, np.nextafter(mid, np.inf),
                           1 + 3*2**-8], dtype=np.float64)
        np.testing.assert_array_equal(round_bf16(values), [1, 1, 1+2**-7, 1+2**-6])
        # Rounding through FP32 would incorrectly turn the third input into 1.
        self.assertNotEqual(round_bf16(values)[2], round_bf16(values.astype(np.float32))[2])

    def test_subnormal_ties_and_overflow(self):
        values = np.array([2.0**-134, 3*2.0**-134, -2.0**-134,
                           2.0**128 - 2.0**119], dtype=np.float64)
        result = round_bf16(values)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 2*2.0**-133)
        self.assertTrue(np.signbit(result[2]))
        self.assertTrue(np.isposinf(result[3]))

    def test_gate_catches_nonfinite_shape_and_zero_sign(self):
        z = np.array([0], dtype=np.float32)
        with self.assertRaises(ValueError):
            differences(z, np.array([np.nan]))
        with self.assertRaises(ValueError):
            differences(z, np.zeros(2, dtype=np.float32))
        report = differences(z, -z, dict(atol=0, rtol=0, exact_bits=True))
        self.assertEqual(report['failed'], 1)
        self.assertEqual(report['zero_sign_differences'], 1)
        report = differences(from_bits([1]), z, dict(atol=1, rtol=1, exact_reference_zero=True))
        self.assertEqual(report['failed'], 1)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_capture_is_observation_and_cleans_hooks(self):
        data = inputs(7, 11, 7, 1601)
        runner = UpstreamMLP(data)
        before = runner.run(data['X'], observe=False)['Y']
        captured = runner.run(data['X'])
        self.assertEqual(set(captured), set(('X', 'N', 'G', 'U', 'A', 'S', 'D', 'Y')))
        self.assertTrue(same_bits(before, captured['Y']))
        self.assertTrue(same_bits(before, runner.run(data['X'], observe=False)['Y']))
        for module in runner.module.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_capture_checks_detect_corrupted_gating_and_release_hooks(self):
        data = inputs(8, 12, 1, 1601)
        runner = UpstreamMLP(data)
        # Change the stored down input without changing captured gate/up.
        def corrupt(module, args):
            return (args[0]+1,)
        handle = runner.module.down_proj.register_forward_pre_hook(corrupt)
        try:
            with self.assertRaisesRegex(ValueError, 'gating operands'):
                runner.run(data['X'])
        finally:
            handle.remove()
        self.assertTrue(same_bits(runner.run(data['X'])['Y'], runner.run(data['X'], False)['Y']))

    def test_prefix_and_distinct_gate_up_weights(self):
        small, large = inputs(7, 11, 7, 1601), inputs(7, 11, 17, 1601)
        self.assertTrue(same_bits(small['X'], large['X'][:7]))
        for k in ('gate', 'up', 'norm', 'down'):
            self.assertTrue(same_bits(small[k], large[k]))
        swapped = mutate(small, 'swap_gate_up')
        self.assertFalse(same_bits(UpstreamMLP(small).run(small['X'])['D'],
                                  UpstreamMLP(swapped).run(swapped['X'])['D']))

    def test_omitted_rounding_controls_are_distinguishable(self):
        _, report = activation_probe()
        for regression in report['rounding_regressions'].values():
            self.assertNotEqual(regression['correct'], regression['skipped'])

    def test_pinned_scalar_tail_discrepancy_has_an_independent_reproducer(self):
        report = projection_tail_probe()
        self.assertEqual(report['upstream'], 1.25)
        self.assertEqual(report['rounded_fp64'], 1.2421875)


class AnchorTests(unittest.TestCase):
    def test_manifest_and_file_tampering_fail_without_updating(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root/'x.npy', np.array([1], dtype=np.float32))
            expected = dict(arrays={'x.npy': dict(sha256=sha(root/'x.npy'))})
            validate_anchors(copy.deepcopy(expected), expected, root)
            with self.assertRaisesRegex(RuntimeError, 'frozen contract'):
                validate_anchors(dict(expected, changed=True), expected, root)
            np.save(root/'x.npy', np.array([2], dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                validate_anchors(expected, expected, root)

    def test_freeze_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'anchor.json'
            write_json(p, {'value': 1}, exclusive=True)
            with self.assertRaises(FileExistsError):
                write_json(p, {'value': 2}, exclusive=True)
            self.assertIn('1', p.read_text())


if __name__ == '__main__':
    unittest.main()
