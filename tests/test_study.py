import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from llm_mojo.benchmarks.study import (parse_output, summarize, encode_samples, read_samples,
                              load_run, sha, write_json, REPETITIONS,
                              select_parallelism_finalists, select_projection_tile)


class StudyTests(unittest.TestCase):
    def test_projection_selection_requires_both_boundaries_and_modes(self):
        block = [dict(query_rows=1024,rows=1024,candidate=v,layers=l,
                      decision='calibration' if v==9 else 'faster',ratio=.8)
                 for v in (9,14,15) for l in (1,24)]
        kernel = copy.deepcopy(block)
        self.assertEqual(select_projection_tile(block,kernel),14)
        kernel[2]['decision']='inconclusive'
        self.assertEqual(select_projection_tile(block,kernel),15)
        block[-1]['decision']='inconclusive'
        self.assertIsNone(select_projection_tile(block,kernel))
        with self.assertRaises(ValueError):
            select_projection_tile(block[:-1],kernel)

    def test_measurement_boundary_cannot_be_silently_swapped(self):
        output=self.output()+'\nmeasurement: isolated_wo\n'
        # Keep the completion sentinel last, as the real instrument does.
        output=output.replace('BENCHMARK_COMPLETE\n','')+'BENCHMARK_COMPLETE\n'
        args=dict(rows=16,layers=24,seed=53,operation='linear')
        identity,_=parse_output(output,0,1,False,measurement='isolated_wo',**args)
        self.assertEqual(identity['measurement'],'isolated_wo')
        for boundary in ('whole_attention','whole_attention_buffered'):
            with self.assertRaises(ValueError):
                parse_output(output,0,1,False,measurement=boundary,**args)

    def test_parallelism_selection_requires_both_modes_and_respects_family_budget(self):
        rows = [dict(query_rows=64,rows=4096,candidate=v,layers=l,
                     decision='calibration' if v == 9 else 'faster',ratio=.8)
                for v in range(9,14) for l in (1,24)]
        self.assertEqual(select_parallelism_finalists(rows),[10,12])
        # A spectacular hot result cannot conceal a weaker ring result.
        for row in rows:
            if row['candidate'] == 11:
                row['ratio'] = .5 if row['layers'] == 1 else .9
            if row['candidate'] == 12 and row['layers'] == 24:
                row['decision'] = 'inconclusive'
        self.assertEqual(select_parallelism_finalists(rows),[10,13])
        for row in rows:
            if row['candidate'] != 9:
                row['decision'] = 'inconclusive'
        self.assertEqual(select_parallelism_finalists(rows),[])
        with self.assertRaises(ValueError):
            select_parallelism_finalists(rows[:-1])

    spec = dict(control=0, candidates=[0, 1], rows=[16])

    def samples(self, ratio=0.8):
        return [dict(block=b, rows=16, layers=l, candidate=c, arm=a,
                     variant=c if a == 'candidate' else 0, repetition=n,
                     us=100 * (ratio if c == 1 and a == 'candidate' else 1))
                for b in range(1, 5) for l in (1, 24) for c in (0, 1)
                for a in ('control', 'candidate') for n in range(REPETITIONS)]

    def output(self, first=False):
        arms = [('control', 0), ('candidate', 1)]
        if first:
            arms.reverse()
        return '\n'.join(['device: Apple Test GPU', 'api: metal', 'operation: linear',
                          'shape: 16 24 seed: 53', f'variants: 0 1 candidate-first: {int(first)}',
                          'correctness: passed'] +
                         [f'SAMPLE {a} {v} {n} 100' for a, v in arms for n in range(REPETITIONS)] +
                         ['BENCHMARK_COMPLETE'])

    def parse(self, output, first=False):
        return parse_output(output, 0, 1, first, rows=16, layers=24, seed=53, operation='linear')

    def test_parser_requires_exact_runtime_workload_order_and_completion(self):
        for first in (False, True):
            self.assertEqual(len(self.parse(self.output(first), first)[1]), 20)
        good = self.output()
        bad = [good.replace(a, b) for a, b in [
            ('Apple Test GPU', 'CPU'), ('api: metal', 'api: cuda'),
            ('shape: 16 24 seed: 53', 'shape: 16 24 seed: 54'),
            ('operation: linear', 'operation: rope'), ('correctness: passed', 'correctness: failed'),
            ('SAMPLE candidate 1 0', 'SAMPLE candidate 0 0'), (' 100', ' nan'),
            ('BENCHMARK_COMPLETE', ''), ('SAMPLE control 0 0 100\n', ''),
            ('device: Apple Test GPU', 'device: Apple Test GPU\ndevice: Apple Test GPU')]]
        bad.append(self.output(True))
        for output in bad:
            with self.subTest(output=output[:100]), self.assertRaises(ValueError):
                self.parse(output)

    def test_complete_calibration_required(self):
        good = self.samples()
        for samples in [good[:-1], good + good[:1], [s for s in good if s['candidate'] != 0]]:
            with self.assertRaises(ValueError):
                summarize(samples, self.spec)
        for key, value in [('variant', 9), ('us', float('nan')), ('us', -1), ('block', 5)]:
            bad = copy.deepcopy(good)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                summarize(bad, self.spec)

    def test_decisions_use_all_blocks_and_observed_noise(self):
        good = self.samples()
        self.assertEqual(summarize(good, self.spec)[1]['decision'], 'faster')
        for s in good:
            if s['candidate'] == 1 and s['block'] == 4 and s['arm'] == 'candidate':
                s['us'] = 101
        self.assertEqual(summarize(good, self.spec)[1]['decision'], 'inconclusive')
        good = self.samples()
        for s in good:
            if s['candidate'] == 0 and s['arm'] == 'candidate' and s['block'] == 1:
                s['us'] = 130
        self.assertEqual(summarize(good, self.spec)[1]['decision'], 'inconclusive')
        self.assertEqual(summarize(self.samples(1.2), self.spec)[1]['decision'], 'slower')

    def test_compressed_samples_roundtrip_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / 'samples.csv.gz'
            raw = self.samples()
            path.write_bytes(encode_samples(raw))
            self.assertEqual(read_samples(path), raw)
            record = dict(schema=1, completed_utc='test', samples_sha256=sha(path),
                          repository={'dirty': False}, runtime={'api': 'metal', 'device': 'Apple Test GPU'},
                          specification=self.spec, build={'repository': {'dirty': False}},
                          blocks=4, repetitions=10, warmup=10, conditions=[{'block': b} for b in range(1, 5)])
            write_json(directory / 'run.json', record)
            self.assertEqual(load_run(directory)[1], raw)
            record['build']['repository'] = {'dirty': False, 'commit': 'different'}
            write_json(directory / 'run.json', record)
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_run(directory)
            path.write_bytes(gzip.compress(b'corrupted'))
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_run(directory)

    def test_prefill_query_rows_are_part_of_runtime_and_calibration_identity(self):
        output = self.output().replace('operation: linear','operation: gqa_prefill')
        output = 'query rows: 7\n' + output
        kwargs = dict(rows=16,layers=24,seed=53,operation='gqa_prefill')
        self.assertEqual(len(parse_output(output,0,1,False,query_rows=7,**kwargs)[1]),20)
        for wrong in (None,8,0,17,True):
            with self.assertRaises(ValueError):
                parse_output(output,0,1,False,query_rows=wrong,**kwargs)
        spec = dict(control=0,candidates=[0,1],workloads=[dict(query_rows=r,rows=16) for r in (7,16)])
        samples = [dict(**s,query_rows=r) for r in (7,16) for s in self.samples()]
        result = summarize(samples,spec)
        self.assertEqual({s['query_rows'] for s in result},{7,16})
        # A full-prefill calibration must not stand in for an incremental one.
        missing = [s for s in samples if not (s['query_rows']==7 and s['candidate']==0)]
        with self.assertRaises(ValueError):
            summarize(missing,spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'samples.csv.gz'
            path.write_bytes(encode_samples(samples))
            self.assertEqual(read_samples(path),samples)

    def test_sublayer_parser_requires_the_query_suffix_and_fp32_route(self):
        output = self.output().replace('operation: linear', 'operation: attention_sublayer')
        output = output.replace('variants: 0 1', 'variants: 3 3')
        output = output.replace('SAMPLE control 0', 'SAMPLE control 3').replace('SAMPLE candidate 1', 'SAMPLE candidate 3')
        output = 'query rows: 7\n' + output
        kwargs = dict(rows=16, layers=24, seed=53, operation='attention_sublayer')
        self.assertEqual(len(parse_output(output,3,3,False,query_rows=7,**kwargs)[1]),20)
        for wrong in (None,8,0,17,True):
            with self.assertRaises(ValueError):
                parse_output(output,3,3,False,query_rows=wrong,**kwargs)
        with self.assertRaises(ValueError):
            parse_output(output.replace('SAMPLE candidate 3','SAMPLE candidate 0'),3,3,False,query_rows=7,**kwargs)
