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
from llm_mojo.benchmarks.study import load_run, load_profile, evidence_directory, load_numerical_record, select_parallelism_finalists, select_projection_tile, sha, write_json
from llm_mojo.benchmarks import attention_prefill_contract as prefill

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_compact_numerical_records_preserve_original_and_reject_corruption(self):
        directory = ROOT / 'studies/attention_sublayer/data'
        for name in ('decode_validation', 'prefill_validation', 'wo_validation', 'precision_numerics'):
            path = directory / (name + '.json')
            summary = json.loads(path.read_text())
            record = load_numerical_record(path)
            self.assertTrue(record)
            for key, value in summary['metadata'].items():
                self.assertEqual(record[key], value)
            for dotted, count in summary['list_lengths'].items():
                value = record
                for key in dotted.split('.'):
                    value = value[key]
                self.assertEqual(len(value), count)
        path = directory / 'prefill_validation.json'
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / path.name
            destination.write_bytes(path.read_bytes())
            with self.assertRaises(FileNotFoundError):
                load_numerical_record(destination)
            summary = json.loads(path.read_text())
            raw = Path(tmp) / summary['record']
            raw.write_bytes((directory / summary['record']).read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'compressed'):
                load_numerical_record(destination)
            raw.write_bytes((directory / summary['record']).read_bytes())
            summary['uncompressed_sha256'] = '0' * 64
            write_json(destination, summary)
            with self.assertRaisesRegex(ValueError, 'original'):
                load_numerical_record(destination)

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

    def test_split_combined_evidence_binds_both_comparisons_to_validated_source(self):
        directory=ROOT / 'studies/attention_sublayer/data'
        validation=json.loads((directory/'split_combined_validation.json').read_text())
        builds=[]
        expected=[(16,256),*[(r,t) for r in (16,64,256) for t in (1024,4096)]]
        for suffix,control in (('projections',13),('gqa',18)):
            record,samples,_=load_run(directory,'split_combined_'+suffix+'_')
            self.assertEqual(len(samples),2240)
            self.assertEqual(record['study'],'attention_sublayer_split_combined_'+suffix)
            self.assertEqual(record['specification']['control'],control)
            self.assertEqual(record['specification']['candidates'],[control,19])
            self.assertEqual([(w['query_rows'],w['rows']) for w in record['specification']['workloads']],expected)
            self.assertEqual(record['runtime']['measurement'],'whole_attention')
            self.assertEqual(record['repository']['commit'],validation['source_commit'])
            self.assertEqual(record['runtime']['device'],validation['device'])
            for name,digest in validation['source_sha256'].items():
                if name in record['build']['sources']:
                    self.assertEqual(digest,record['build']['sources'][name])
            builds.append(record['build'])
        self.assertEqual(builds[0],builds[1])
        self.assertEqual(validation['tests']['asynchronous_configurations'],20)
        self.assertEqual(validation['tests']['asynchronous_sequences_per_configuration'],12)
        self.assertEqual(validation['limits']['projected_and_final'],0.03125)
        self.assertEqual(validation['limits']['cache'],'exact BF16 bits')
        self.assertTrue(all(c['returncode']==0 and c['sources_unchanged'] for c in validation['commands']))
        for dataset in validation['numerics'].values():
            self.assertEqual(set(dataset),{'13','18','19'})
            for fields in dataset.values():
                self.assertEqual(set(fields),{'projected','output'})
                for value in fields.values():
                    self.assertGreater(value['checks'],0)
                    self.assertLessEqual(value['maximum_scaled_error'],0.03125)

    def test_retained_studies_are_complete(self):
        count = 0
        for directory in (ROOT / 'studies').glob('*/'):
            for record in evidence_directory(directory).glob('*run.json'):
                _, samples, _ = load_run(directory,record.name.removesuffix('run.json'))
                count += len(samples)
        self.assertEqual(count, 87200)
        profile = load_profile(ROOT / 'studies/gqa_decode')
        self.assertEqual(sum(row['count'] for row in profile), 3000)
        profile = load_profile(ROOT / 'studies/gqa_prefill')
        self.assertEqual(sum(row['count'] for row in profile), 4800)
        profile = load_profile(ROOT / 'studies/gqa_prefill','resources_')
        self.assertEqual(sum(row['count'] for row in profile), 2400)
        profile = load_profile(ROOT / 'studies/attention_sublayer/data')
        self.assertEqual(sum(row['count'] for row in profile), 1920)
        profile = load_profile(ROOT / 'studies/attention_sublayer/data', 'wo_')
        self.assertEqual(sum(row['count'] for row in profile), 1200)
        wo = json.loads((ROOT / 'studies/attention_sublayer/data/wo_profiles.json').read_text())
        run = json.loads((ROOT / 'studies/attention_sublayer/data/wo_run.json').read_text())
        self.assertEqual(wo['common']['repository'], run['repository'])
        self.assertEqual(wo['common']['source_sha256'], run['build']['sources'])
        for capture in wo['captures']:
            self.assertEqual(capture['counter_analysis']['status'], 'not_analyzed')
            self.assertEqual(capture['counters'], [])
        profile = load_profile(ROOT / 'studies/attention_sublayer/data', 'decode_')
        self.assertEqual(sum(row['count'] for row in profile), 3300)
        decode = json.loads((ROOT / 'studies/attention_sublayer/data/decode_profiles.json').read_text())
        run = json.loads((ROOT / 'studies/attention_sublayer/data/decode_run.json').read_text())
        self.assertEqual(decode['common']['repository'], run['repository'])
        self.assertEqual(decode['common']['source_sha256'], run['build']['sources'])
        profile = load_profile(ROOT / 'studies/attention_sublayer/data', 'prefill_')
        self.assertEqual(sum(row['count'] for row in profile), 1320)
        prefill_record = json.loads((ROOT / 'studies/attention_sublayer/data/prefill_profiles.json').read_text())
        run = json.loads((ROOT / 'studies/attention_sublayer/data/prefill_run.json').read_text())
        validation = load_numerical_record(ROOT / 'studies/attention_sublayer/data/prefill_validation.json')
        self.assertEqual(prefill_record['common']['repository'], run['repository'])
        self.assertEqual(prefill_record['common']['source_sha256'], run['build']['sources'])
        self.assertEqual(validation['source_commit'], run['repository']['commit'])
        for name, digest in validation['source_sha256'].items():
            if name in run['build']['sources']:
                self.assertEqual(digest, run['build']['sources'][name])
        profile = load_profile(ROOT / 'studies/attention_sublayer/data', 'integrated_')
        self.assertEqual(sum(row['count'] for row in profile), 2090)
        integrated = json.loads((ROOT / 'studies/attention_sublayer/data/integrated_profiles.json').read_text())
        validation = json.loads((ROOT / 'studies/attention_sublayer/data/integrated_validation.json').read_text())
        for prefix,control in (('projections_',8),('integrated_',3)):
            run,samples,_ = load_run(ROOT / 'studies/attention_sublayer/data',prefix)
            self.assertEqual(len(samples),4800)
            self.assertEqual(run['specification']['control'],control)
            self.assertEqual(integrated['common']['repository'],run['repository'])
            self.assertEqual(integrated['common']['source_sha256'],run['build']['sources'])
            self.assertEqual(validation['source_commit'],run['repository']['commit'])
            for name,digest in validation['source_sha256'].items():
                if name in run['build']['sources']:
                    self.assertEqual(digest,run['build']['sources'][name])
        for capture in integrated['captures']:
            self.assertEqual(capture['counter_analysis']['status'],'not_analyzed')
            self.assertEqual(capture['counters'],[])
        directory = ROOT / 'studies/attention_sublayer/data'
        screen, samples, summary = load_run(directory, 'parallelism_screen_')
        self.assertEqual(len(samples), 2400)
        finalists = select_parallelism_finalists(summary)
        self.assertEqual(finalists, [13])
        run, samples, _ = load_run(directory, 'parallelism_')
        self.assertEqual(len(samples), 4800)
        self.assertEqual(screen['specification']['control'], 9)
        self.assertEqual(run['specification']['control'], 9)
        self.assertEqual(run['specification']['candidates'], [9, *finalists])
        self.assertEqual(run['selection']['finalists'], finalists)
        self.assertEqual(run['selection']['screen_run_sha256'], sha(directory / 'parallelism_screen_run.json'))
        self.assertEqual(run['selection']['screen_samples_sha256'], screen['samples_sha256'])
        self.assertEqual(screen['build'], run['build'])
        profile = load_profile(directory, 'parallelism_')
        self.assertEqual(sum(row['count'] for row in profile), 950)
        parallelism = json.loads((directory / 'parallelism_profiles.json').read_text())
        validation = json.loads((directory / 'parallelism_validation.json').read_text())
        self.assertEqual(parallelism['specification']['variants'], [9, *finalists])
        self.assertEqual(parallelism['common']['repository'], run['repository'])
        self.assertEqual(parallelism['common']['source_sha256'], run['build']['sources'])
        self.assertEqual(validation['source_commit'], run['repository']['commit'])
        for name, digest in validation['source_sha256'].items():
            if name in run['build']['sources']:
                self.assertEqual(digest, run['build']['sources'][name])
        for capture in parallelism['captures']:
            self.assertEqual(capture['counter_analysis']['status'], 'not_analyzed')
            self.assertEqual(capture['counters'], [])
        tile_screen, _, block_summary = load_run(directory, 'tiles_screen_')
        kernel_screen, _, kernel_summary = load_run(directory, 'tiles_kernel_screen_')
        winner = select_projection_tile(block_summary, kernel_summary)
        self.assertEqual(winner, 14)
        validation = json.loads((directory / 'tiles_validation.json').read_text())
        cases = [('tiles_screen_', 1920, 'whole_attention', [9,14,15]),
                 ('tiles_kernel_screen_', 1920, 'isolated_wo', [9,14,15]),
                 ('timing_', 480, 'whole_attention', [9]),
                 ('timing_buffered_', 480, 'whole_attention_buffered', [9]),
                 ('split_domain_', 1920, 'whole_attention', [9,13])]
        for prefix, variant, size in (('tiles_',winner,4800),
                                      ('tiles_qkv_',None if winner is None else winner+2,1920)):
            self.assertEqual((directory / (prefix+'run.json')).exists(), winner is not None)
            if winner is not None:
                cases.append((prefix,size,'whole_attention',[9,variant]))
                record,_,_ = load_run(directory,prefix)
                self.assertEqual(record['selection']['wo_finalist'],winner)
                self.assertEqual(record['selection']['variant'],variant)
                for selection, screen_prefix in zip(record['selection']['screens'],
                                                     ('tiles_screen_','tiles_kernel_screen_')):
                    screen,_,_ = load_run(directory,screen_prefix)
                    self.assertEqual(selection['study'],screen['study'])
                    self.assertEqual(selection['run_sha256'],sha(directory / (screen_prefix+'run.json')))
                    self.assertEqual(selection['samples_sha256'],screen['samples_sha256'])
                self.assertEqual(len(record['selection']['screens']),2)
        for prefix,size,measurement,variants in cases:
            record,samples,_ = load_run(directory,prefix)
            self.assertEqual(len(samples),size)
            self.assertEqual(record['build'],tile_screen['build'])
            self.assertEqual(record['repository']['commit'],validation['source_commit'])
            self.assertEqual(record['specification']['control'],9)
            self.assertEqual(record['specification']['candidates'],variants)
            self.assertEqual(record['specification']['measurement'],measurement)
            self.assertEqual(record['runtime']['measurement'],measurement)
            for name,digest in validation['source_sha256'].items():
                if name in record['build']['sources']:
                    self.assertEqual(digest,record['build']['sources'][name])
        self.assertTrue(all(c['returncode']==0 and c['sources_unchanged'] for c in validation['commands']))
        self.assertEqual(validation['tests']['mojo'],93)
        self.assertEqual(validation['tests']['asynchronous_configurations'],18)
        self.assertEqual(validation['tests']['asynchronous_sequences_per_configuration'],12)
        for dataset in validation['numerics'].values():
            for family, mappings in dataset.items():
                gate = validation['limits']['isolated_qkv' if family=='isolated_qkv' else 'isolated_wo_and_composed']
                self.assertEqual(set(mappings),set(map(str,range(5 if family=='composition' else 3))))
                for fields in mappings.values():
                    for value in fields.values():
                        self.assertGreater(value['checks'],0)
                        self.assertLessEqual(value['maximum_scaled_error'],gate)
        run,samples,_ = load_run(directory,'combined_')
        profile = load_profile(directory,'combined_')
        record = json.loads((directory/'combined_profiles.json').read_text())
        validation = json.loads((directory/'combined_validation.json').read_text())
        self.assertEqual(len(samples),4800)
        self.assertEqual(sum(row['count'] for row in profile),1530)
        self.assertEqual(run['specification']['candidates'],[9,18])
        self.assertEqual(run['specification']['control'],9)
        self.assertEqual(run['runtime']['measurement'],'whole_attention')
        self.assertEqual(record['common']['repository'],run['repository'])
        self.assertEqual(record['common']['source_sha256'],run['build']['sources'])
        self.assertEqual(validation['source_commit'],run['repository']['commit'])
        for name,digest in validation['source_sha256'].items():
            if name in run['build']['sources']:
                self.assertEqual(digest,run['build']['sources'][name])
        self.assertEqual(len(record['captures']),8)
        for capture in record['captures']:
            self.assertEqual(capture['counter_analysis']['status'],'not_analyzed')
            self.assertEqual(capture['counters'],[])
        self.assertEqual(validation['tests']['asynchronous_configurations'],19)
        self.assertTrue(all(c['returncode']==0 and c['sources_unchanged'] for c in validation['commands']))
        for dataset in validation['numerics'].values():
            self.assertEqual(set(dataset['composition']),{'0','5'})
            for fields in dataset['composition'].values():
                self.assertEqual(set(fields),{'projected','output'})
                for value in fields.values():
                    self.assertGreater(value['checks'],0)
                    self.assertLessEqual(value['maximum_scaled_error'],validation['limits']['projected_and_final'])
        record = json.loads((ROOT / 'studies/attention_sublayer/data/profiles.json').read_text())
        full = next(c for c in record['captures'] if c['query_rows'] == 4096)
        self.assertEqual(full['counter_analysis']['status'], 'not_analyzed')
        self.assertEqual(full['counters'], [])
        self.assertIn('absence is not zero', full['counters_scope'])

    def test_profile_corruption_and_duplicate_dispatch_rejected(self):
        for topic, prefix, groups in (('gqa_decode', '', 6),
                                      ('attention_sublayer', '', 48),
                                      ('attention_sublayer', 'wo_', 96),
                                      ('attention_sublayer', 'decode_', 66),
                                      ('attention_sublayer', 'prefill_', 66),
                                      ('attention_sublayer', 'integrated_', 76),
                                      ('attention_sublayer', 'parallelism_', 38),
                                      ('attention_sublayer', 'combined_', 72)):
            source = evidence_directory(ROOT / 'studies' / topic)
            with self.subTest(topic=topic, prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                record = json.loads((source / (prefix+'profiles.json')).read_text())
                path = directory / 'profile_samples.csv.gz'
                path.write_bytes((source / (prefix+path.name)).read_bytes())
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
