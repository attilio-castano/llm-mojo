"""Decoder route, coverage and provenance regressions; no GPU execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo import decoder_validation as validation
from llm_mojo.benchmarks import decoder_layer_contract as contract
from llm_mojo.benchmarks.study import STUDIES,summarize,parse_output
from llm_mojo.benchmarks.capture_trace import parse_target_identity


class DecoderToolingTests(unittest.TestCase):
    def test_compact_holdout_manifest_preserves_original_fixture_identity(self):
        import gzip
        from llm_mojo._repository import repository_root
        path=repository_root()/'studies/decoder_layer/policies_holdout_manifest.json'
        wrapper=json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'manifest.json'
            original.write_bytes(gzip.decompress((path.parent/wrapper['record']).read_bytes()))
            expected=validation.holdout_manifest(original,verify_arrays=False)
            self.assertEqual(validation.holdout_manifest(path,verify_arrays=False),expected)
            self.assertEqual(expected[1]['manifest.json'],wrapper['uncompressed_sha256'])

    def test_column_archive_roundtrip_preserves_bytes_and_block_order(self):
        import gzip,hashlib,lzma
        records=[dict(case='escaped\n\"é',value=-0.0,items=[None,True,2**60,1e-30]),
                 dict(kind='negative',failed=1,expected_failure=True)]
        records=(records*4097)+[dict(value=1.0000000000000002)]
        raw=''.join(json.dumps(r)+'\n' for r in records).encode()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'checks.jsonl.gz';source.write_bytes(gzip.compress(raw))
            target=root/'checks.columns.jsonl.xz'
            spec=validation.compact_checks(source,target)
            restored=''.join(json.dumps(r)+'\n' for r in validation.check_records(target)).encode()
            self.assertEqual(restored,raw)
            self.assertEqual(spec['uncompressed_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(spec['records'],len(records))
            self.assertEqual(spec['uncompressed_bytes'],len(raw))
            original=target.read_bytes()
            with self.assertRaises(FileExistsError):validation.compact_checks(source,target)
            self.assertEqual(target.read_bytes(),original)
            # Missing columns, reordered schema IDs, or unused rows must not be
            # silently dropped by zip/index reconstruction.
            blocks=lzma.decompress(original).splitlines(keepends=True)
            for mutation in ('short_column','unused_row','negative_id','duplicate_key'):
                order,tables=json.loads(blocks[1])
                if mutation=='short_column':tables[0][1][0].pop()
                elif mutation=='unused_row':order.pop()
                elif mutation=='negative_id':order[0]=-1
                else:tables[0][0][1]=tables[0][0][0]
                target.write_bytes(lzma.compress(blocks[0]+(json.dumps([order,tables])+'\n').encode()))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    list(validation.check_records(target))
            target.write_bytes(original[:-8])
            with self.assertRaises((EOFError,lzma.LZMAError)):
                list(validation.check_records(target))
            target.unlink();source.write_bytes(gzip.compress(b'{"value":1}\n'))
            with self.assertRaisesRegex(ValueError,'canonical'):validation.compact_checks(source,target)
            self.assertFalse(target.exists())

    def test_column_receipt_replays_gates_and_rejects_rehashed_omissions(self):
        import gzip,hashlib
        import test_decoder_validation as baseline_tests
        fixture=baseline_tests.NumericalReceiptTests();fixture.setUp()
        raw=''.join(json.dumps(r)+'\n' for r in fixture.records).encode()
        native=b'TestSuite summary: 1 passed , 0 failed , 0 skipped\n'
        digest=lambda data:hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'checks.jsonl';source.write_bytes(raw)
            target=root/'checks.columns.jsonl.xz';spec=validation.compact_checks(source,target)
            output=gzip.compress(native);(root/'output.gz').write_bytes(output)
            record=dict(status='passed',checks_sha256=digest(raw),output_sha256=digest(native))
            wrapper=dict(format='decoder-policy-numerics-columns-v1',evaluation=record,
                original_evaluation_sha256='a'*64,checks=spec,
                output=dict(file='output.gz',sha256=digest(output),uncompressed_sha256=digest(native),original_sha256=digest(native)))
            receipt=root/'archive.json';receipt.write_text(json.dumps(wrapper))
            observed,path,identity=validation.read_policy_evidence(receipt)
            self.assertEqual((observed,identity),(record,'a'*64))
            self.assertEqual(validation.validate_results(path,fixture.cases),validation.validate_results(source,fixture.cases))
            # The existing compact JSON wrapper also works for large receipts.
            receipt_raw=receipt.read_bytes();packed=gzip.compress(receipt_raw)
            (root/'archive.json.gz').write_bytes(packed)
            receipt.write_text(json.dumps(dict(format='lossless-json-gzip-v1',record='archive.json.gz',
                sha256=digest(packed),uncompressed_sha256=digest(receipt_raw))))
            self.assertEqual(validation.read_policy_evidence(receipt),(observed,path,identity))
            for key,value in (('sha256','0'*64),('uncompressed_sha256','0'*64),('records',spec['records']-1),('uncompressed_bytes',0)):
                altered=copy.deepcopy(wrapper);altered['checks'][key]=value
                receipt.write_text(json.dumps(altered))
                with self.subTest(key=key),self.assertRaises(ValueError):validation.read_policy_evidence(receipt)
            # Even updating the receipt to match a valid archive cannot waive a
            # missing protected-storage check in the numerical validator.
            partial=[r for r in fixture.records if r.get('label')!='aw_qkv']
            source.write_text(''.join(json.dumps(r)+'\n' for r in partial))
            target.unlink();wrapper['checks']=validation.compact_checks(source,target)
            wrapper['evaluation']['checks_sha256']=wrapper['checks']['uncompressed_sha256']
            receipt.write_text(json.dumps(wrapper))
            _,path,_=validation.read_policy_evidence(receipt)
            with self.assertRaises(ValueError):validation.validate_results(path,fixture.cases)

    def test_recorded_policy_lookup_is_independent_of_live_dispatch(self):
        frozen=copy.deepcopy(contract._POLICY_LOOKUP)
        for cell in frozen['cells']:
            if (cell['query_rows'],cell['rows'],cell['layers'])==(16,16,1):
                cell.update(deterministic=22,fast=21)
        different=dict(cells=[],fallbacks=dict(fast=0,deterministic=20))
        with patch.object(contract,'_POLICY_LOOKUP',different):
            self.assertEqual(contract.policy_configuration(True,16,16),20)
            self.assertEqual(contract.policy_configuration(True,16,16,lookup=frozen),22)
            self.assertEqual(contract.execution_mappings(101,16,16,frozen),(5,7,19))
            self.assertEqual(contract.execution_mappings(100,16,16,frozen),(5,6,7))
            self.assertEqual(contract.policy_configuration(True,17,17,lookup=frozen),20)
            self.assertEqual(contract.policy_configuration(False,64,4095,lookup=frozen),0)
        for cell in contract._POLICY_LOOKUP['cells']:
            for deterministic,policy in ((False,'fast'),(True,'deterministic')):
                self.assertEqual(contract.policy_configuration(deterministic,cell['query_rows'],
                    cell['rows'],cell['layers']),cell[policy])

    def test_compressed_numerical_archive_preserves_and_rechecks_all_records(self):
        import gzip,hashlib
        import test_decoder_validation as baseline_tests
        fixture=baseline_tests.NumericalReceiptTests();fixture.setUp()
        raw=''.join(json.dumps(r)+'\n' for r in fixture.records).encode()
        native=b'TestSuite summary: 1 passed , 0 failed , 0 skipped\n'
        digest=lambda data:hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checks=root/'checks.jsonl';output=root/'output.log'
            checks.write_bytes(raw);output.write_bytes(native)
            record=dict(status='passed',exit_code=0,checks_sha256=digest(raw),output_sha256=digest(native))
            receipt=root/'evaluation.json';receipt.write_text(json.dumps(record))
            original_sha=validation.sha(receipt)
            wrapper=dict(format='decoder-policy-numerics-gzip-v1',evaluation=record,
                original_evaluation_sha256=original_sha)
            for field,data in (('checks',raw),('output',native)):
                name=field+'.gz';compressed=gzip.compress(data,mtime=0)
                (root/name).write_bytes(compressed)
                wrapper[field]=dict(file=name,sha256=digest(compressed),uncompressed_sha256=digest(data))
            wrapper['output']['original_sha256']=digest(native)
            archive=root/'archive.json'
            archive.write_text(json.dumps(wrapper))
            observed,path,identity=validation.read_policy_evidence(archive)
            self.assertEqual((observed,identity),(record,original_sha))
            self.assertEqual(validation.validate_results(path,fixture.cases),validation.validate_results(checks,fixture.cases))
            self.assertEqual(validation.read_policy_evidence(root),(record,checks,original_sha))
            # A syntactically valid, freshly hashed archive still needs every gate.
            partial=''.join(json.dumps(r)+'\n' for r in fixture.records[1:]).encode()
            bad=root/'partial.jsonl.gz';bad.write_bytes(gzip.compress(partial,mtime=0))
            with self.assertRaises(ValueError):validation.validate_results(bad,fixture.cases)
            for field in ('checks','output'):
                altered=copy.deepcopy(wrapper);altered[field]['uncompressed_sha256']='0'*64
                archive.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):validation.read_policy_evidence(archive)
            altered=copy.deepcopy(wrapper);altered['checks']['file']='../checks.gz'
            archive.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):validation.read_policy_evidence(archive)
            archive.write_text(json.dumps(wrapper))
            (root/'checks.gz').write_bytes((root/'checks.gz').read_bytes()[:-8])
            with self.assertRaises(ValueError):validation.read_policy_evidence(archive)

    def test_final_confirmation_is_frozen_and_cost_uses_the_accepted_lookup(self):
        from llm_mojo.benchmarks import study
        declaration=copy.deepcopy(contract.policy_declaration())
        basis=copy.deepcopy(declaration['rounds'][1]['incumbent_decision'])
        basis.update(round=2,compatible_row_reuse=True,deterministic_family=[20,22,23,24])
        for cell in basis['proposals']:
            if cell['query_rows']>1:cell['deterministic']=24
        declaration['final_confirmation']=json.loads(json.dumps(contract.propose_policy_confirmation(basis,declaration)))
        before=copy.deepcopy(declaration)
        build=dict(repository=dict(dirty=False),sources={contract.POLICY_PATH:contract.sha_json(declaration)})
        specs=contract.policy_confirmation_specs(declaration)
        self.assertEqual(len(declaration['final_confirmation']['cells']),14)
        def data(spec):
            return [dict(**w,layers=l,candidate=c,block=b,arm=a,
                variant=spec['control'] if a=='control' else c,repetition=n,
                us=100. if a=='control' or c==spec['control'] else
                    120. if (w['query_rows'],w['rows'],l,b)==(256,256,24,1) else 80.)
                for w,l,c in study.comparisons(spec) for b in range(1,5)
                for a in ('control','candidate') for n in range(10)]
        self.assertEqual(sum(len(data(s)) for s in specs.values()),2080)
        def load(path,prefix=''):
            name=prefix.removesuffix('_') if prefix else Path(path).name;spec=specs[name]
            samples=data(spec)
            return dict(study=name,build=build,specification=json.loads(json.dumps(spec)),samples_sha256='samples'),samples,study.summarize(samples,spec)
        with patch.object(study,'load_run',side_effect=load),patch.object(contract,'sha',return_value='0'*64):
            accepted=contract.policy_confirmed_selection(Path('/unused'),build,declaration)
        self.assertEqual(declaration,before)
        bad=next(c for c in accepted['cells'] if (c['query_rows'],c['rows'],c['layers'])==(256,256,24))
        self.assertEqual(bad['deterministic'],20)
        self.assertEqual(next(c for c in accepted['cells'] if (c['query_rows'],c['rows'],c['layers'])==(16,256,1))['fast'],21)
        # Equal policies need one complete self comparison and still appear in cost.
        accepted['cells'][0]['deterministic']=accepted['cells'][0]['fast']
        costs=contract.policy_cost_specs(accepted)
        def cost(path,prefix=''):
            name=prefix.removesuffix('_') if prefix else Path(path).name;spec=costs[name]
            samples=data(spec)
            return dict(study=name,build=build,selection=accepted,
                specification=json.loads(json.dumps(spec)),samples_sha256='samples'),samples,study.summarize(samples,spec)
        with patch.object(study,'load_run',side_effect=cost),patch.object(contract,'sha',return_value='0'*64):
            report=contract.policy_cost_report(Path('/unused'),accepted,build)
        self.assertEqual(len(report['rows']),14)
        first=next(r for r in report['rows'] if (r['query_rows'],r['rows'],r['layers'])==(16,16,1))
        self.assertEqual(first['ratio'],1.)
        altered=copy.deepcopy(declaration)
        altered['final_confirmation']['cells'][0]['deterministic']=23
        with self.assertRaises(ValueError):contract.policy_confirmation_specs(altered)

    def test_second_round_uses_actual_incumbents_and_complete_mode_census(self):
        declaration=contract.policy_declaration()
        declared=declaration['rounds'][1]
        summaries={screen['name']:[dict(query_rows=r,rows=t,layers=l,candidate=v,
            decision='calibration' if v==screen['control'] else 'faster',
            ratio=1.0 if v==screen['control'] else .8 if v==23 else .7)
            for r,t in screen['workloads'] for l in screen['layers'] for v in screen['candidates']]
            for screen in declared['screens']}
        choose=lambda compatible:contract.select_policy_round2(summaries,{20,22,23,24},compatible)
        selected=choose(True)
        self.assertTrue(all(x['deterministic']==24 and x['fast']==24
                            for x in selected['proposals'] if x['query_rows']>1))
        self.assertTrue(all(x['deterministic']==20 and x['fast']==0
                            for x in selected['proposals'] if x['query_rows']==1))
        self.assertTrue(all(x['deterministic']==22 for x in choose(False)['proposals'] if x['query_rows']>1))
        hot=summaries['decoder_policies_round2_fast_21_hot']
        for row in hot:row['decision']='inconclusive'
        cell=next(x for x in choose(True)['proposals'] if (x['query_rows'],x['rows'],x['layers'])==(16,256,1))
        self.assertEqual(cell['fast'],21)
        hot.pop()
        with self.assertRaises(ValueError):choose(True)

    def test_policy_selection_requires_complete_screens_and_compatible_arithmetic(self):
        summaries={screen['name']:[dict(query_rows=r,rows=t,layers=l,candidate=v,
            decision='calibration' if v==screen['control'] else 'faster',
            ratio=1.0 if v==screen['control'] else 0.4 if v==21 else 0.6,
            ratio_min=0.39 if v==21 else 0.59,ratio_max=0.41 if v==21 else 0.61,noise_floor=.05)
            for r,t in screen['workloads'] for l in (1,24) for v in screen['candidates']]
            for screen in contract.policy_declaration()['rounds'][0]['screens']}
        choose=lambda invariant,compatible:contract.select_policy_round1(summaries,invariant,compatible)
        # A globally faster fixed-MMA family is valid even with different bytes.
        selected=choose({20,21,22},True)
        self.assertEqual(selected['deterministic_family'],[21])
        self.assertTrue(all(x['deterministic']==21 for x in selected['proposals']))
        frozen=contract.policy_declaration()
        expected_schedules=contract.policy_schedules(4096,frozen)
        with patch.object(contract,'policy_declaration',return_value={}):
            self.assertEqual(contract.select_policy_round1(summaries,{20,21,22},True,frozen),selected)
            self.assertEqual(contract.policy_schedules(4096,frozen),expected_schedules)
        # One bad mode forbids mixing distinct arithmetic families by workload.
        row=next(x for x in summaries['decoder_policies_round1_det'] if x['candidate']==21)
        row['decision']='slower'
        selected=choose({20,21,22},True)
        self.assertEqual(selected['deterministic_family'],[20,22])
        self.assertTrue(all(x['deterministic']==22 for x in selected['proposals']))
        self.assertTrue(all(x['fast']==21 for x in selected['proposals']))
        # Own schedule invariance does not establish compatibility with ID20.
        self.assertTrue(all(x['deterministic']==20 for x in choose({20,22},False)['proposals']))
        row['decision']='faster';row['ratio_max']=.58
        self.assertFalse(choose({20,21,22},True)['fixed_mma_global_qualified'])
        name=next(iter(summaries));saved=summaries[name].pop()
        with self.assertRaises(ValueError):choose({20,21,22},True)
        summaries[name].extend([saved,saved])
        with self.assertRaises(ValueError):choose({20,21,22},True)

    def test_policy_registry_preserves_history_and_separate_qkv_dispatches(self):
        self.assertNotIn(20,contract.VARIANTS)
        self.assertEqual(contract.mappings(20,1),contract.mappings(20,4096))
        for rows in (1,17,4096):
            self.assertEqual(contract.stages(20,rows)[1:4],['Q projection','K projection','V projection'])
            self.assertEqual(contract.specification(20,rows,4096)['dispatches_per_iteration'],17)
        for rows in (1,15,16,17,63,64,65,4096):
            schedules=contract.policy_schedules(rows)
            self.assertEqual(schedules['policy_tokenwise'],[(p,1) for p in range(rows)])
            for calls in schedules.values():
                self.assertEqual([p for start,size in calls for p in range(start,start+size)],list(range(rows)))
        declared=contract.policy_declaration()
        spec=dict(policies=declared,policies_sha256=contract.sha_json(declared),
                  captures=[list(row[:3]) for row in declared['profiles']])
        self.assertEqual(contract.sha(contract.repository_root()/contract.POLICY_PATH),spec['policies_sha256'])
        self.assertEqual(len(contract.policy_profile_grid(spec)),6)
        # The live collector constructs tuples; JSON restoration returns lists.
        live={**spec,'captures':[tuple(row) for row in spec['captures']]}
        self.assertEqual(contract.policy_profile_grid(live),contract.policy_profile_grid(spec))
        for bad in ({**spec,'policies_sha256':'changed'},{**spec,'captures':spec['captures'][:-1]}):
            with self.assertRaises(ValueError):contract.policy_profile_grid(bad)

    def test_policy_schedules_hit_every_lookup_cell_and_keep_test_ids_separate(self):
        cells={(r,p+r) for calls in contract.policy_schedules(4096).values() for p,r in calls}
        self.assertTrue({(r,t) for r,t,_ in contract.policy_declaration()['workloads']}<=cells)
        self.assertFalse(contract.MEASUREMENT_VARIANTS & contract.POLICY_EXECUTIONS.keys())
        self.assertEqual(contract.execution_mappings(100,64,4096),contract.mappings(3,64))
        for variant in contract.POLICY_EXECUTIONS:
            with self.assertRaises(ValueError):contract.specification(variant,1,4096)
        for args in ((False,0,1,1),(True,2,1,1),(True,1,4097,1),(False,1,1,2)):
            with self.assertRaises(ValueError):contract.policy_configuration(*args)

    def test_frozen_workloads_and_sample_census(self):
        spec=STUDIES['decoder_layer']
        samples=[dict(query_rows=r,rows=t,layers=l,block=b,candidate=0,arm=a,variant=0,repetition=n,us=1.)
                 for r,t in contract.WORKLOADS for l in (1,24) for b in range(1,5)
                 for a in ('control','candidate') for n in range(10)]
        self.assertEqual(len(samples),960)
        result=summarize(samples,spec)
        self.assertEqual(len(result),12)
        self.assertTrue(all(r['decision']=='calibration' for r in result))
        for bad in (samples[:-1],samples+[samples[0]], [{**r,'variant':7} for r in samples]):
            with self.assertRaises(ValueError):summarize(bad,spec)

    def test_decoder_profile_identity_and_budget(self):
        self.assertEqual(len(contract.STAGES),16)
        self.assertEqual(sum(n*16 for _,_,n in contract.PROFILES),2400)
        for r,t,n in contract.PROFILES:
            data=dict(operation='decoder_layer',implementation='decoder_layer_0',entrypoint='enqueue_decoder_layer',
                profile_iterations=n,profile_warmup_iterations=10,**contract.specification(0,r,t))
            contract.configuration(data)
            for field,value in [('mlp_mapping',14),('key_value_rows',0),('dispatches_per_iteration',15),('profile_iterations',313)]:
                with self.assertRaises(ValueError):contract.configuration({**data,field:value})

    def test_capture_accepts_zero_decode_mapping_but_rejects_zero_dimensions(self):
        from llm_mojo.benchmarks.capture_trace import validate_target_identity
        output='\n'.join(['profile implementation: enqueue_decoder_layer',
            'device: Apple M4 Pro','api: metal','rows: 1','hidden: 896',
            'warmup iterations: 10','profile iterations: 100',
            'post-profile idle milliseconds: 250','profile workload: decoder-r1-t4096-v0',
            'key value rows: 4096','query heads: 14','key value heads: 2',
            'intermediate size: 4864','mlp mapping: 0','profile dispatches per iteration: 16'])
        identity=parse_target_identity(output)
        configuration=dict(operation='decoder_layer',implementation='decoder_layer_0',entrypoint='enqueue_decoder_layer',
            profile_warmup_iterations=10,profile_iterations=100,profile_post_idle_milliseconds=250,
            **contract.specification(0,1,4096))
        hardware=dict(chip='Apple M4 Pro',gpu_api='metal')
        self.assertEqual(validate_target_identity(identity,configuration,hardware)['mlp_mapping'],0)
        for bad in (output.replace('mlp mapping: 0','mlp mapping: -1'),output.replace('query heads: 14','query heads: 0')):
            with self.assertRaises(ValueError):parse_target_identity(bad)
        with self.assertRaises(ValueError):validate_target_identity({**identity,'mlp_mapping':7},configuration,hardware)

    def test_benchmark_runtime_rejects_wrong_rows_backend_and_truncation(self):
        output='\n'.join(['device: Apple M4 Pro','api: metal','operation: decoder_layer',
            'measurement: whole_decoder','query rows: 16','shape: 256 24 seed: 4001',
            'variants: 0 0 candidate-first: 0','correctness: passed']+
            [f'SAMPLE {a} 0 {n} 1.0' for a in ('control','candidate') for n in range(10)]+['BENCHMARK_COMPLETE'])
        def parse(text):return parse_output(text,0,0,False,rows=256,layers=24,seed=4001,operation='decoder_layer',query_rows=16,measurement='whole_decoder')
        parse(output)
        for bad in (output.replace('api: metal','api: cpu'),output.replace('query rows: 16','query rows: 17'),output[:-20]):
            with self.assertRaises(ValueError):parse(bad)

    def test_environment_drops_inherited_filters(self):
        with patch.dict('os.environ',{'DECODER_CASE':'wrong','DECODER_SPLIT':'development','MODULAR_DEBUG':'device-sync-mode'}):
            self.assertFalse(any(k.startswith('DECODER_') or k=='MODULAR_DEBUG' for k in validation.environment()))

    def test_binary_receipt_rejects_modified_or_stale_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'candidate';binary.write_bytes(b'original')
            source=dict(repository=dict(dirty=False),sources={'test':'one'})
            record=dict(schema=1,kind='decoder_numerical_build',source=source,binary_sha256=validation.sha(binary))
            Path(str(binary)+'.provenance.json').write_text(json.dumps(record))
            with patch.object(validation,'source_identity',return_value=source):
                validation.verify_build(binary)
                binary.write_bytes(b'changed')
                with self.assertRaises(ValueError):validation.verify_build(binary)
                binary.write_bytes(b'original')
            with patch.object(validation,'source_identity',return_value={**source,'sources':{'test':'two'}}):
                with self.assertRaises(ValueError):validation.verify_build(binary)

    def test_distinct_adversarial_patterns(self):
        import numpy as np
        signs=[contract.signs(i) for i in range(24)]
        self.assertEqual(len({s.tobytes() for s in signs}),24)
        self.assertTrue(all(set(np.unique(s))=={-1,1} for s in signs))


class ProfileEvidenceTests(unittest.TestCase):
    def test_curator_accepts_decoder_identity_and_preserves_all_dispatches(self):
        from llm_mojo.benchmarks import profile_summary as curator
        from llm_mojo.benchmarks.study import load_decoder_profile, load_decoder_windows
        from llm_mojo.benchmarks.analyze_trace import coalesce_compute_commands, duration_summary
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';output.mkdir();tables={}
            def cell(value):return (str(value),str(value))
            for r,t,n in contract.PROFILES:
                folder=root/f'r{r}-t{t}-v0';folder.mkdir()
                provenance=dict(repository=dict(commit='frozen',dirty=False),hardware={},software={},source_sha256={})
                (folder/'profile.provenance.json').write_text(json.dumps(provenance))
                (folder/'capture.json').write_text('{}')
                (folder/'conditions.json').write_text('{}')
                submissions=[];intervals=[]
                for j in range((n+10)*16):
                    submissions.append(dict(start=cell(j*3),**{'cmdbuffer-id':cell(j),'num-encoders':cell(1),'process':cell('target')}))
                    intervals.append(dict(start=cell(j*3),duration=cell(2),**{'cmdbuffer-id':cell(j),
                        'encoder-id':cell(0),'gpu-submission-id':cell(j),'event-label':cell('target:Compute Command'),'channel-name':cell('Compute')}))
                inputs=[]
                for name,kind,rows in [('submissions.xml','command_buffer_submissions_xml',submissions),('gpu-intervals.xml','gpu_intervals_xml',intervals)]:
                    path=folder/name;path.write_text(name);tables[path]=rows
                    inputs.append(dict(kind=kind,sha256=curator.sha(path)))
                joined,coalescing=coalesce_compute_commands(intervals,submissions,len(intervals))
                identity=dict(operation='decoder_layer',implementation='decoder_layer_0',entrypoint='enqueue_decoder_layer',
                    repository=provenance['repository'],runtime=dict(device='Apple Test GPU',backend='metal'),
                    workload=dict(**contract.specification(0,r,t),warmup_iterations=10,profile_iterations=n),
                    capture_receipt=dict(sha256=curator.sha(folder/'capture.json')),
                    provenance=dict(sha256=curator.sha(folder/'profile.provenance.json')))
                report=dict(capture_identity=identity,analysis_source_sha256={},inputs=inputs,trace={},
                    validated_sequence=dict(interval_coalescing=coalescing,fragmented_profile_dispatches=0),
                    instrumented_gpu_interval_duration=dict(profile=duration_summary(joined[160:])))
                (folder/'summary.json').write_text(json.dumps(report))
            with patch.object(curator,'read_table',side_effect=lambda path:tables[path]):
                curator.collect(root,output,decoder_layer=True)
            self.assertEqual(len(load_decoder_profile(output)),48)
            windows=load_decoder_windows(output)
            self.assertEqual(sum(w['dispatches'] for w in windows),2400)
            for window,(_,_,n) in zip(windows,contract.PROFILES):
                self.assertEqual(window['active_us'],n*16*2/1000)
                self.assertEqual(window['enclosing_us'],(n*16*3-1)/1000)
            result=json.loads((output/'profiles.json').read_text())
            self.assertTrue(all(c['spills']['status']=='not_analyzed' for c in result['captures']))

    def test_complete_profile_and_corrupted_census(self):
        import csv,gzip,hashlib,io
        from llm_mojo.benchmarks.study import load_decoder_profile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=dict(commit='frozen',dirty=False)
            captures=[];samples=[]
            for r,t,n in contract.PROFILES:
                w=dict(**contract.specification(0,r,t),rows=r,warmup_iterations=10,profile_iterations=n)
                captures.append(dict(query_rows=r,rows=t,variant=0,capture=dict(repository=source,
                    operation='decoder_layer',implementation='decoder_layer_0',entrypoint='enqueue_decoder_layer',
                    runtime=dict(backend='metal',device='Apple Test GPU'),workload=w)))
                for j in range(n):
                    for k,stage in enumerate(contract.STAGES):
                        samples.append(dict(query_rows=r,rows=t,variant=0,iteration=j,stage=stage,
                            duration_ns=1,start_ns=j*32+k*2,end_ns=j*32+k*2+1))
            record=dict(schema=5,common=dict(repository=source),captures=captures,
                specification=dict(workloads=[[r,t] for r,t,_ in contract.PROFILES],variants=[0]))
            def write(rows):
                out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(rows)
                raw=gzip.compress(out.getvalue().encode());(root/'profile_samples.csv.gz').write_bytes(raw)
                record['samples_sha256']=hashlib.sha256(raw).hexdigest();(root/'profiles.json').write_text(json.dumps(record))
            write(samples);self.assertEqual(len(load_decoder_profile(root)),48)
            for bad in (samples[:-1],samples+[samples[0]]):
                write(bad)
                with self.assertRaises(ValueError):load_decoder_profile(root)
            write(samples)
            (root/'profile_samples.csv.gz').write_bytes(b'changed')
            with self.assertRaises(ValueError):load_decoder_profile(root)
            write(samples);record['captures'][0]['capture']['runtime']['backend']='cpu'
            (root/'profiles.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):load_decoder_profile(root)


if __name__=='__main__':unittest.main()
