"""MLP measurement and capture boundaries must fail closed."""
import copy
import gzip
import json
from contextlib import ExitStack
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from llm_mojo.benchmarks import mlp_contract as mlp
from llm_mojo.benchmarks import analyze_trace, capture_trace, study, run as runner


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

    def test_profile_variant_must_match_recorded_workload(self):
        for variant in sorted(mlp.VARIANTS):
            identity = {**self.identity(), **mlp.specification(variant,17),
                        'implementation':f'mlp_{variant}'}
            self.assertEqual(mlp.configuration(identity)['profile_workload'], f'mlp-r17-v{variant}')
            with self.assertRaises(ValueError):
                mlp.configuration({**identity,'profile_workload':'mlp-r17-v999'})

    def test_retained_mlp_profile_rejects_false_capture_identity(self):
        source=Path(__file__).resolve().parents[1]/'studies/mlp_sublayer/data'
        original=json.loads((source/'profiles.json').read_text())
        changes=[('runtime','backend','cpu'),('workload','rows',17),
                 ('workload','profile_workload','mlp-r1-v1'),
                 ('workload','dispatches_per_iteration',6)]
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            (directory/'profile_samples.csv.gz').write_bytes((source/'profile_samples.csv.gz').read_bytes())
            (directory/'profiles.json').write_text(json.dumps(original))
            self.assertEqual(sum(s['count'] for s in study.load_profile(directory)),4445)
            for group,key,value in changes:
                record=copy.deepcopy(original)
                record['captures'][0]['capture'][group][key]=value
                (directory/'profiles.json').write_text(json.dumps(record))
                with self.subTest(group=group,key=key),self.assertRaises(ValueError):
                    study.load_profile(directory)
            record=copy.deepcopy(original)
            record['captures'][-1]=record['captures'][0]
            (directory/'profiles.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                study.load_profile(directory)

    def test_projection_finalist_needs_both_modes_and_uses_weaker_mode(self):
        variants=[0,1,2,3]
        rows=[dict(rows=1024,layers=l,candidate=v,decision='faster',ratio=r)
              for v,ratios in enumerate(((1,1),(.2,.8),(.4,.5),(.3,.4)))
              for l,r in zip((1,24),ratios)]
        rows[-1]['decision']='inconclusive'
        self.assertEqual(study.select_mlp_projection(rows,variants),2)
        rows[-1]['decision']='faster'
        self.assertEqual(study.select_mlp_projection(rows,variants),3)
        for row in rows:
            row['decision']='inconclusive'
        self.assertEqual(study.select_mlp_projection(rows,variants),0)
        with self.assertRaises(ValueError):
            study.select_mlp_projection(rows[:-1],variants)

    def test_numerical_stream_retains_multiple_mappings_without_rewrite(self):
        import mlp_support
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',MLP_RECORD_DIR=directory):
            for mapping in (0,2,0):
                mlp_support.set_mapping(mapping)
                mlp_support.record(dict(case='fixture',stage='D',failed=0))
            paths=list(Path(directory).glob('*.jsonl'))
            self.assertEqual(len(paths),1)
            records=[json.loads(line) for line in paths[0].read_text().splitlines()]
            self.assertEqual([r['mapping'] for r in records],[0,2,0])

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
        capture_id=capture_trace.new_capture_id('mlp')
        self.assertRegex(capture_id,r'^mlp-[0-9a-f]{32}$')
        self.assertIsNotNone(analyze_trace.CAPTURE_ID.fullmatch(capture_id))

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

    def test_completed_cases_survive_a_later_timeout_without_becoming_a_run(self):
        repo=dict(commit='a'*40,branch='test',dirty=False)
        spec={**study.STUDIES['mlp'], 'layers':[1]}
        observation=dict(arm='control',variant=0,repetition=0,us=10.0)
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root=Path(temporary); build=root/'build'; build.mkdir()
            (build/'build.json').write_text(json.dumps(dict(repository=repo,sources={},
                environment={},binaries={'mlp':'digest'},mlp_fixtures={})))
            replacements=dict(ensure_record_location=None,repository_state=repo,
                source_hashes={},stable_environment={},mlp_fixture_identity={},
                checked_conditions={'power':'AC'},sha='digest',
                workloads=[dict(rows=1),dict(rows=17)],
                parse_output=(dict(device='Apple Test GPU',api='metal'),[observation]))
            for name,value in replacements.items():
                stack.enter_context(patch.object(runner,name,return_value=value))
            stack.enter_context(patch.object(runner,'STUDIES',{'mlp':spec}))
            process=stack.enter_context(patch.object(runner.subprocess,'run',side_effect=[
                subprocess.CompletedProcess([],0,stdout='completed case',stderr=''),
                subprocess.TimeoutExpired(['mlp'],1200)]))
            output=root/'output'
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.run(build,output,['mlp'])
            saved=json.loads((output/'mlp/run.json').read_text())
            self.assertNotIn('completed_utc',saved)
            self.assertNotIn('after',saved['conditions'][0])
            raw=gzip.decompress((output/'mlp/samples.csv.gz').read_bytes()).decode()
            self.assertEqual(len(raw.splitlines()),2)
            self.assertIn('10.0',raw)
            self.assertEqual(process.call_args.kwargs['timeout'],1200)
            with self.assertRaises(ValueError):
                study.load_run(output/'mlp')


if __name__=='__main__':
    unittest.main()
