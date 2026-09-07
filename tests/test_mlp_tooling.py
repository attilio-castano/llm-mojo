"""MLP measurement and capture boundaries must fail closed."""
import copy
import unittest
from unittest.mock import patch
from llm_mojo.benchmarks import mlp_contract as mlp
from llm_mojo.benchmarks import capture_trace, study


class MLPToolingTests(unittest.TestCase):
    def identity(self):
        return dict(operation='mlp',implementation='mlp_0',entrypoint='enqueue_mlp_apple_gpu',
            profile_iterations=25,profile_warmup_iterations=10,
            profile_post_idle_milliseconds=250,**mlp.specification(0,17))

    def test_profile_shape_and_dispatch_identity(self):
        good=self.identity()
        self.assertEqual(mlp.configuration(good)['dispatches_per_iteration'],7)
        for key,value in [('implementation','mlp_1'),('entrypoint','wrong'),
                          ('profile_rows',True),('intermediate_size',896),
                          ('dispatches_per_iteration',6),('profile_iterations',715)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                mlp.configuration({**good,key:value})

    def test_runtime_receipt_requires_mlp_width_and_seven_dispatches(self):
        identity=self.identity()
        identity.update(implementation='mlp_0')
        output='''profile implementation: enqueue_mlp_apple_gpu
device: Apple Test GPU
api: metal
rows: 17
hidden: 896
intermediate size: 4864
profile workload: mlp-r17-v0
profile dispatches per iteration: 7
warmup iterations: 10
profile iterations: 25
post-profile idle milliseconds: 250
PROFILE_REGION_BEGIN
PROFILE_REGION_END
'''
        parsed=capture_trace.validate_target_identity(capture_trace.parse_target_identity(output),identity,dict(chip='Apple Test GPU',gpu_api='metal'))
        self.assertEqual(parsed['intermediate_size'],4864)
        with self.assertRaises(ValueError):
            capture_trace.validate_target_identity(capture_trace.parse_target_identity(output.replace('iteration: 7','iteration: 6')),identity,dict(chip='Apple Test GPU',gpu_api='metal'))
        self.assertRegex(capture_trace.new_capture_id('mlp'),r'^mlp-[0-9a-f]{32}$')

    def test_stage_grid_is_hot_only_and_requires_every_sample(self):
        spec=copy.deepcopy(study.STUDIES['mlp_stage_0'])
        self.assertEqual(spec['layers'],[1])
        samples=[dict(block=b,rows=r,layers=1,candidate=0,arm=arm,
                      variant=0,repetition=n,us=10.0)
                 for b in range(1,5) for r in mlp.PROFILE_ROWS
                 for arm in ('control','candidate') for n in range(10)]
        self.assertEqual(len(study.summarize(samples,spec)),4)
        with self.assertRaises(ValueError):
            study.summarize(samples[:-1],spec)
        with self.assertRaises(ValueError):
            study.summarize(samples+[dict(samples[0],layers=24)],spec)

    def test_fixture_identity_rejects_mutated_array_hash(self):
        with patch.object(mlp,'sha',return_value='changed'),self.assertRaises(ValueError):
            mlp.fixture_identity()


if __name__=='__main__':
    unittest.main()
