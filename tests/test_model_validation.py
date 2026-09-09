import tempfile
import json
from pathlib import Path
import unittest
import numpy as np
from llm_mojo.model_validation import (bf16, compare, consistency_accuracy,
    verify_consistency_observations, CONSISTENCY_BOUNDARIES)


class ModelComparisonTests(unittest.TestCase):
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
