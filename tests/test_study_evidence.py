import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo.benchmarks import run as runner
from llm_mojo.benchmarks.study import load_run, load_profile, sha, write_json
from llm_mojo.benchmarks import attention_prefill_contract as prefill

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_prefill_profiles_require_each_shape_and_exact_dispatch_sequence(self):
        for variants,prefix in (([0,8],''),([8,12],'resources_'),([8,11,12],'resources_')):
            with self.subTest(variants=variants):
                workloads = [(16,16),(1024,1024),(64,4096)]
                repo = dict(commit='a'*40,dirty=False)
                records, samples = [], []
                for r,t in workloads:
                    for v in variants:
                        workload = dict(**prefill.specification(v,r,t),rows=r,warmup_iterations=1,profile_iterations=2)
                        identity = dict(operation=prefill.OPERATION,implementation=f'gqa_prefill_{v}',
                                        entrypoint=prefill.ENTRYPOINTS[f'gqa_prefill_{v}'],
                                        repository=repo,runtime=dict(device='Apple Test GPU',backend='metal'),workload=workload)
                        records.append(dict(query_rows=r,rows=t,variant=v,capture=identity))
                        for i in range(2):
                            for stage in (['QK','softmax','PV'] if v==0 else ['fused']):
                                samples.append(dict(query_rows=r,rows=t,variant=v,iteration=i,stage=stage,duration_ns=1000))
                with tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    path = directory/(prefix+'profile_samples.csv.gz')
                    stream=io.StringIO(newline='')
                    writer=csv.DictWriter(stream,fieldnames=list(samples[0]),lineterminator='\n')
                    writer.writeheader();writer.writerows(samples)
                    path.write_bytes(gzip.compress(stream.getvalue().encode()))
                    record = dict(schema=2,specification=dict(workloads=workloads,variants=variants),
                                  common=dict(repository=repo),captures=records,samples_sha256=sha(path))
                    write_json(directory/(prefix+'profiles.json'),record)
                    self.assertEqual(sum(s['count'] for s in load_profile(directory,prefix)),
                                     6 * sum(3 if v==0 else 1 for v in variants))
                    bad=copy.deepcopy(record)
                    bad['captures'][-1]['capture']['workload']['profile_rows']=16
                    write_json(directory/(prefix+'profiles.json'),bad)
                    with self.assertRaises(ValueError):
                        load_profile(directory,prefix)
                    bad=copy.deepcopy(record)
                    bad['captures'].pop()
                    write_json(directory/(prefix+'profiles.json'),bad)
                    with self.assertRaisesRegex(ValueError,'capture'):
                        load_profile(directory,prefix)
                    bad=copy.deepcopy(record)
                    bad['specification']['variants'][-1]=variants[0]
                    write_json(directory/(prefix+'profiles.json'),bad)
                    with self.assertRaisesRegex(ValueError,'variants'):
                        load_profile(directory,prefix)
                    bad=copy.deepcopy(record)
                    bad['specification']['workloads'][-1]=(64,1024)
                    write_json(directory/(prefix+'profiles.json'),bad)
                    with self.assertRaisesRegex(ValueError,'grid'):
                        load_profile(directory,prefix)
                    path.write_bytes(gzip.compress(('\n'.join(stream.getvalue().splitlines()[:-1])+'\n').encode()))
                    record['samples_sha256']=sha(path)
                    write_json(directory/(prefix+'profiles.json'),record)
                    with self.assertRaisesRegex(ValueError,'incomplete'):
                        load_profile(directory,prefix)

    def test_retained_studies_are_complete(self):
        count = 0
        for directory in (ROOT / 'studies').glob('*/'):
            for record in directory.glob('*run.json'):
                _, samples, _ = load_run(directory,record.name.removesuffix('run.json'))
                count += len(samples)
        self.assertEqual(count, 33600)
        profile = load_profile(ROOT / 'studies/gqa_decode')
        self.assertEqual(sum(row['count'] for row in profile), 3000)
        profile = load_profile(ROOT / 'studies/gqa_prefill')
        self.assertEqual(sum(row['count'] for row in profile), 4800)
        profile = load_profile(ROOT / 'studies/gqa_prefill','resources_')
        self.assertEqual(sum(row['count'] for row in profile), 2400)
        profile = load_profile(ROOT / 'studies/attention_sublayer')
        self.assertEqual(sum(row['count'] for row in profile), 1920)
        record = json.loads((ROOT / 'studies/attention_sublayer/profiles.json').read_text())
        full = next(c for c in record['captures'] if c['query_rows'] == 4096)
        self.assertEqual(full['counter_analysis']['status'], 'not_analyzed')
        self.assertEqual(full['counters'], [])
        self.assertIn('absence is not zero', full['counters_scope'])

    def test_profile_corruption_and_duplicate_dispatch_rejected(self):
        for topic, groups in (('gqa_decode', 6), ('attention_sublayer', 48)):
            source = ROOT / 'studies' / topic
            with self.subTest(topic=topic), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                record = json.loads((source / 'profiles.json').read_text())
                path = directory / 'profile_samples.csv.gz'
                path.write_bytes((source / path.name).read_bytes())
                write_json(directory / 'profiles.json', record)
                self.assertEqual(len(load_profile(directory)), groups)
                raw = gzip.decompress(path.read_bytes()).decode()
                path.write_bytes(gzip.compress((raw + raw.splitlines()[1] + '\n').encode()))
                with self.assertRaisesRegex(ValueError, 'hash'):
                    load_profile(directory)
                record['samples_sha256'] = sha(path)
                write_json(directory / 'profiles.json', record)
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    load_profile(directory)
                path.write_bytes(gzip.compress(('\n'.join(raw.splitlines()[:-1]) + '\n').encode()))
                record['samples_sha256'] = sha(path)
                write_json(directory / 'profiles.json', record)
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    load_profile(directory)

    def test_runner_rejects_foreign_source_environment_and_binary_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            binary = directory / 'operations'
            binary.write_bytes(b'fixture binary')
            repo = dict(commit='a' * 40, branch='codex/test', dirty=False)
            sources, environment = {'source': 'hash'}, {'hardware': 'test'}
            base = dict(repository=repo, sources=sources, environment=environment,
                        binaries={'operations': sha(binary)})
            bad = []
            for key, value in [('repository', {**repo, 'commit': 'b' * 40}),
                               ('repository', {**repo, 'dirty': True}),
                               ('sources', {}), ('environment', {}),
                               ('binaries', {'operations': 'wrong'})]:
                item = copy.deepcopy(base); item[key] = value; bad.append(item)
            for record in bad:
                write_json(directory / 'build.json', record)
                with patch.object(runner, 'repository_state', return_value=repo), \
                     patch.object(runner, 'source_hashes', return_value=sources), \
                     patch.object(runner, 'stable_environment', return_value=environment), \
                     patch.object(runner, 'checked_conditions', side_effect=AssertionError('reached GPU phase')), \
                     self.assertRaises(RuntimeError):
                    runner.run(directory, directory / 'output', ['gqa_decode'])
