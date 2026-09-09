import tempfile
from pathlib import Path
import unittest
import numpy as np
from llm_mojo.model_validation import bf16, compare


class ModelComparisonTests(unittest.TestCase):
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
