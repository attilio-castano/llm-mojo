import tempfile
import json
from pathlib import Path
import unittest
import numpy as np
from llm_mojo.model_validation import (bf16, compare, consistency_accuracy,
    verify_consistency_observations, CONSISTENCY_BOUNDARIES,
    numerical_diagnostic, prediction_diagnostic, storage_diagnostic)
from llm_mojo.model_validation import generation_events


class ModelComparisonTests(unittest.TestCase):
    def test_generation_event_contract_rejects_truncation_and_bad_cache_accounting(self):
        good=('event\tindex\tvalue\tnanoseconds\n'
              'prompt\t0\t42\t0\n'
              'device\t0\tApple M4 Pro/metal\t0\n'
              'token\t0\t151645\t10\n'
              'cache\t0\t1\t0\n'
              'submitted\t0\t24\t0\n'
              'finish\t1\tstop\t11\n')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'events.tsv';p.write_text(good)
            self.assertEqual(generation_events(p,32)['tokens'],[151645])
            for bad in (good.replace('finish\t1\tstop\t11\n',''),
                        good.replace('cache\t0\t1','cache\t0\t2'),
                        good.replace('submitted\t0\t24','submitted\t0\t48'),
                        good.replace('151645','10')):
                p.write_text(bad)
                with self.assertRaises(ValueError): generation_events(p,32)

    def test_diagnostics_report_distance_without_an_accuracy_gate(self):
        expected=np.array([[1.,2.,3.]],dtype=np.float32)
        actual=expected+100
        metrics=numerical_diagnostic(actual,expected)
        self.assertEqual(metrics['max_abs'],100)
        self.assertNotIn('passed',metrics)
        predictions=prediction_diagnostic(actual,expected)
        self.assertAlmostEqual(predictions['kl_nats'],0)
        self.assertAlmostEqual(predictions['total_variation'],0)
        with self.assertRaises(ValueError):
            numerical_diagnostic(np.array([[np.nan,0,1]]),expected)

    def test_storage_checks_preserve_bits_and_cover_inactive_capacity(self):
        prior=np.array([[0.],[-0.],[123.],[123.]],dtype=np.float32)
        appended=np.array([[7.]],dtype=np.float32)
        actual=prior.copy(); actual[2]=appended[0]
        self.assertTrue(all(storage_diagnostic(actual,appended,prior,2,1).values()))
        actual[1]=0.
        self.assertFalse(storage_diagnostic(actual,appended,prior,2,1)['prefix'])
        actual=prior.copy(); actual[2]=8.
        self.assertFalse(storage_diagnostic(actual,appended,prior,2,1)['append'])
        actual[3]=1.
        self.assertFalse(storage_diagnostic(actual,appended,prior,2,1)['inactive'])

    def test_reference_rejects_omitted_schedule_even_with_complete_reported_records(self):
        # All records for the reported full call are present, but the declared
        # four-token partitions are missing. Counts alone must not qualify it.
        report=dict(cases=[dict(length=4,seed=99,arrays={s:{} for s in CONSISTENCY_BOUNDARIES},
                    schedules=[dict(rows=[4],checks=75,failures=[])])])
        rows=[dict(length=4,seed=99,schedule=[4],start=0,rows=4,stage=s,exact=True,max_abs=0.)
              for s in sorted(CONSISTENCY_BOUNDARIES)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'schedule census'):
                verify_consistency_observations(report,path)

    def test_reference_requires_complete_passing_observations(self):
        report=dict(cases=[dict(length=1,seed=99,arrays={s:{} for s in CONSISTENCY_BOUNDARIES},
                    schedules=[dict(rows=[1],checks=75,failures=[])])])
        rows=[dict(length=1,seed=99,schedule=[1],start=0,rows=1,stage=s,exact=True,max_abs=0.)
              for s in sorted(CONSISTENCY_BOUNDARIES)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.jsonl'
            def save(values):
                path.write_text(''.join(json.dumps(r)+'\n' for r in values))
            save(rows)
            verify_consistency_observations(report,path)
            for values in (rows[:-1],rows+[rows[0]],rows[:-1]+[dict(rows[-1],exact=False)]):
                save(values)
                with self.assertRaises(ValueError):verify_consistency_observations(report,path)

    def test_consistency_accuracy_requires_each_independent_gate(self):
        expected=np.ones((2,4),dtype=np.float32)
        gate=dict(atol=1.,rtol=0.,relative_rms=.1,exact=False)
        self.assertTrue(consistency_accuracy(expected,expected,gate)['passed'])
        self.assertFalse(consistency_accuracy(expected+.2,expected,gate)['passed'])
        gate['relative_rms']=1.
        gate['exact']=True
        self.assertFalse(consistency_accuracy(expected+.01,expected,gate)['passed'])
        positive=np.zeros((1,4),dtype=np.float32)
        negative=-positive
        self.assertFalse(consistency_accuracy(negative,positive,gate)['passed'])
        gate['exact']=False
        self.assertFalse(consistency_accuracy(expected[:1],positive,gate)['passed'])

    def test_error_gate_and_nonfinite(self):
        gate=dict(atol=0.0625,rtol=0.03125)
        expected=np.array([0.,1.,-1.],dtype=np.float32)
        self.assertTrue(compare(expected,expected,gate)['passed'])
        self.assertFalse(compare(expected+1,expected,gate)['passed'])
        with self.assertRaises(ValueError):compare(expected*float('nan'),expected,gate)
        with self.assertRaises(ValueError):compare(expected[:1],expected,gate)

    def test_bf16_signed_values_and_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'values.bin'
            np.array([0x3f80,0xbf80,0x0000,0x8000],dtype='<u2').tofile(path)
            result=bf16(path,(2,2))
            np.testing.assert_array_equal(result,np.array([[1,-1],[0,-0.]],dtype=np.float32))
            self.assertTrue(np.signbit(result[1,1]))
            with self.assertRaises(ValueError):bf16(path,(3,2))


if __name__=='__main__':unittest.main()
