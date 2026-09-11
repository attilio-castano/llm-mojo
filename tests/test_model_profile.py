import unittest
import json
import gzip
import hashlib
import copy
import subprocess
import tempfile
from pathlib import Path
from llm_mojo.benchmarks import model_contract as contract
from llm_mojo.benchmarks.model_profile import parse_samples, summarize
from llm_mojo.benchmarks.capture_trace import parse_target_identity


class ModelProfileTests(unittest.TestCase):
    def test_retained_archive_rejects_rehashed_missing_evidence(self):
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import replay
        path = repository_root()/'studies/model_generation/token-profile.json.gz'
        if not path.exists():
            self.skipTest('profiling evidence has not yet been collected')
        original = json.loads(gzip.decompress(path.read_bytes()))
        for damage in ('sample','cache','dispatch','provenance'):
            record = copy.deepcopy(original)
            if damage == 'sample': record['timing']['samples'].pop()
            elif damage == 'cache': record['timing']['numerical'][0]['observations'][0]['exact'] = False
            elif damage == 'dispatch': record['captures'][0]['samples'].pop()
            else: record['captures'][0]['provenance']['binary']['sha256'] = '0'*64
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                raw = json.dumps(record).encode()
                packed = gzip.compress(raw,mtime=0)
                (directory/'token-profile.json.gz').write_bytes(packed)
                (directory/'token-profile.json').write_text(json.dumps(dict(
                    sha256=hashlib.sha256(packed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with self.assertRaises(ValueError):
                    replay(directory)

    def test_whole_model_dispatch_contract(self):
        stages = contract.stages()
        self.assertEqual(len(stages), 410)
        self.assertEqual(sum(s == 'inter-layer copy' for _, s in stages), 23)
        self.assertEqual(sum(s == 'FP32 GQA' for _, s in stages), 24)
        self.assertEqual(stages[-1], (-1, 'vocabulary projection'))
        data = dict(operation='qwen_model', implementation='qwen_model_fast',
                    entrypoint='QwenModel.forward+greedy', **contract.specification(64),
                    profile_iterations=8, profile_warmup_iterations=10)
        contract.configuration(data)
        for key, value in [('dispatches_per_iteration',409), ('key_value_rows',4097), ('profile_iterations',13)]:
            with self.assertRaises(ValueError):
                contract.configuration({**data,key:value})

    def test_mixed_transfer_sequence_keeps_strict_coverage(self):
        stages = contract.command_stages()
        self.assertEqual(len(stages),414)
        self.assertEqual([k for _,_,k in stages[:2]+stages[-2:]],['blit']*4)
        rows = [{'event-label':('',f'Command Buffer 0:{kind.title()} Command 0')}
                for _,_,kind in stages]
        contract.validate_command_sequence(rows*2)
        with self.assertRaises(ValueError):
            contract.validate_command_sequence(rows[:-1])
        changed = copy.deepcopy(rows)
        changed[2] = changed[0]
        with self.assertRaises(ValueError):
            contract.validate_command_sequence(changed)

    def test_resubmitted_encoder_preserves_active_fragments_and_rejects_overlap(self):
        from llm_mojo.benchmarks.analyze_trace import coalesce_compute_commands
        def row(start, duration, submission):
            result = {k:(str(v),str(v)) for k,v in dict(start=start,duration=duration,
                      **{'cmdbuffer-id':1,'encoder-id':2,'gpu-submission-id':submission}).items()}
            result['event-label'] = ('','Command Buffer 0:Compute Command 0     ( target )')
            return result
        submitted = [{'start':('0','0'),'cmdbuffer-id':('1','1'),'num-encoders':('1','1')}]
        parts = [row(10,4,3),row(20,6,4)]
        joined,_ = coalesce_compute_commands(parts,submitted,1,join_resubmissions=True)
        self.assertEqual(joined[0]['duration'][0],'10')
        self.assertEqual(joined[0]['end'][0],'26')
        self.assertEqual(joined[0]['active-segments'][0],'2')
        with self.assertRaises(ValueError):
            coalesce_compute_commands(parts,submitted,1)
        with self.assertRaises(ValueError):
            coalesce_compute_commands([row(10,15,3),row(20,6,4)],submitted,1,join_resubmissions=True)
        changed = row(20,6,4)
        changed['event-label'] = ('','Command Buffer 0:Blit Command 0     ( target )')
        with self.assertRaises(ValueError):
            coalesce_compute_commands([parts[0],changed],submitted,1,join_resubmissions=True)

    def test_missing_duplicate_or_misordered_observations_rejected(self):
        header = 'device: Apple M4 Pro\napi: metal\n'
        lines = [f'SAMPLE {a} {s} 100'+(' 1 2 3 4 5 6 7 8 9 10' if a else '')
                 for a in range(2) for s in range(10)]
        output = header+'\n'.join(lines)+'\nBENCHMARK_COMPLETE\n'
        self.assertEqual(len(parse_samples(output,64,0,1)),20)
        for damaged in [output.replace(lines[0]+'\n',''),output+lines[0]+'\n',
                        output.replace('7 8 9 10','7 9 8 10'),output.replace('api: metal','api: cpu')]:
            with self.assertRaises(ValueError):
                parse_samples(damaged,64,0,1)

    def test_summary_requires_own_complete_calibration(self):
        samples = [dict(prefix=p,block=b,comparison=c,arm=a,sample=s,elapsed_ns=100,
                        marks=list(range(10)) if c==1 and a==1 else [])
                   for p in contract.PREFIXES for b in range(4) for c in range(2)
                   for a in range(2) for s in range(10)]
        self.assertEqual(len(summarize(samples)),3)
        with self.assertRaises(ValueError):
            summarize(samples[1:])

    def test_capture_parser_recognizes_model_geometry(self):
        stdout = '''device: Apple M4 Pro
api: metal
profile implementation: QwenModel.forward+greedy
rows: 1
hidden: 896
key value rows: 65
profile workload: model-p64
profile dispatches per iteration: 410
warmup iterations: 10
profile iterations: 8
post-profile idle milliseconds: 250
'''
        result = parse_target_identity(stdout)
        self.assertEqual(result['dispatches_per_iteration'],410)
        self.assertEqual(result['key_value_rows'],65)

    def test_full_model_receipt_round_trip(self):
        from test_trace_capture import write_profile
        from llm_mojo.benchmarks.capture_trace import capture_trace
        from llm_mojo.benchmarks.analyze_trace import capture_identity
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = write_profile(root/'profiles')
            path = binary.with_name(binary.name+'.provenance.json')
            provenance = json.loads(path.read_text())
            provenance.update(operation='qwen_model',implementation='qwen_model_fast',
                              entrypoint='QwenModel.forward+greedy',**contract.specification(64),
                              profile_iterations=8,profile_warmup_iterations=10)
            path.write_text(json.dumps(provenance)+'\n')
            trace = root/'model.trace'
            output = '''device: Apple Test GPU
api: metal
correctness: passed
profile implementation: QwenModel.forward+greedy
rows: 1
hidden: 896
key value rows: 65
profile workload: model-p64
profile dispatches per iteration: 410
warmup iterations: 10
profile iterations: 8
post-profile idle milliseconds: 0
PROFILE_REGION_BEGIN
PROFILE_REGION_END
'''
            def runner(command, **kwargs):
                if command[-1]=='version':
                    return subprocess.CompletedProcess(command,0,'xctrace version test\n')
                trace.mkdir()
                return subprocess.CompletedProcess(command,0,output)
            receipt_path = root/'capture.json'
            capture_trace(profile_binary=binary,output_trace=trace,receipt_path=receipt_path,
                          staging_root=root,runner=runner)
            identity,_ = capture_identity(receipt_path)
            self.assertEqual(identity['operation'],'qwen_model')
            self.assertEqual(identity['workload']['dispatches_per_iteration'],410)


if __name__ == '__main__':
    unittest.main()
