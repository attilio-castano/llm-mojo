"""The fitted cost policy must retain a numerical and evaluation boundary."""
import copy
import unittest

import numpy as np

from llm_mojo.benchmarks.engine_budget import (
    COEFFICIENTS, features, nonnegative_fit, summary, validate_policy, workload_identity,
)


class EngineBudgetTests(unittest.TestCase):
    def test_recovers_independent_nonnegative_cost_model(self):
        rng = np.random.default_rng(19)
        x = rng.integers(1,100,size=(100,5))
        x[:,0] = 1
        expected = [7000,110,3,900,270]
        actual=nonnegative_fit(x,x@expected)
        self.assertTrue(all(0<=a-b<=1 for a,b in zip(actual,expected)))
        self.assertTrue(np.all(x@actual>=x@expected))

    def test_nonnegative_fit_accepts_rank_deficiency_without_negative_cost(self):
        x = np.tile([1,1,1,1,1],(10,1))
        actual = nonnegative_fit(x,np.full(10,10))
        self.assertTrue(all(v>=0 for v in actual))
        self.assertEqual(sum(actual),10)
        with self.assertRaises(ValueError):
            nonnegative_fit(x,np.zeros(10))

    def test_features_follow_physical_attention_partitions_and_step_ownership(self):
        step = dict(step_id=1,decode_seqs=2,prefill_tokens=1,total_tokens=3,attended_positions=42)
        events = [dict(kind='token',step_id=0),dict(kind='token',step_id=1),dict(kind='finish',step_id=1)]
        self.assertEqual(features(step,events),[1,3,42,1,1])
        step['prefill_tokens']=128
        step['total_tokens']=130
        self.assertEqual(features(step,events),[1,130,42,2,1])

    def test_policy_rejects_missing_negative_and_unbounded_coefficients(self):
        policy = dict(kind='engine-step-cost-v1',schema_version=1,
                      cost=dict(zip(COEFFICIENTS,[1,2,3,4,5]),target_ns=25_000_000),
                      calibration_sha256='a'*64,calibration_trace_sha256='b'*64,
                      calibration_workload_sha256='c'*64)
        self.assertEqual(validate_policy(policy),policy['cost'])
        for value in [-1,1_000_000_000_001,True]:
            broken=copy.deepcopy(policy)
            broken['cost']['per_row_ns']=value
            with self.assertRaises(ValueError): validate_policy(broken)
        with self.assertRaises(ValueError):
            summary(dict(policy=policy,trace={},trace_sha256='b'*64))

    def test_workload_identity_ignores_json_metadata_and_formatting(self):
        from tests.test_engine_trace import trace_fixture
        first=trace_fixture()
        second=copy.deepcopy(first)
        second['seed']=123
        second['scripted_tokens']=[999]
        self.assertEqual(workload_identity(first),workload_identity(second))
        second['requests'][0]['prompt_ids']=[99]
        self.assertNotEqual(workload_identity(first),workload_identity(second))


if __name__=='__main__':
    unittest.main()
