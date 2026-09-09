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
