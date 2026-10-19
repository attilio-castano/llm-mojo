import copy
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo.benchmarks.study import (parse_output, summarize, encode_samples, read_samples,
                              load_run, sha, write_json, REPETITIONS,
                              select_parallelism_finalists, select_projection_tile)


class StudyTests(unittest.TestCase):
    def test_split_combined_dispatch_contract_includes_merge_and_decode_fallback(self):
        from llm_mojo.benchmarks.attention_sublayer_contract import (
            STAGES_BY_VARIANT, specification)
        from llm_mojo.benchmarks.profile import argument_parser
        self.assertEqual(STAGES_BY_VARIANT[19], STAGES_BY_VARIANT[13])
        self.assertEqual(specification(19,17,64)['dispatches_per_iteration'],10)
        self.assertEqual(specification(19,1,64)['dispatches_per_iteration'],9)
        args=argument_parser().parse_args(['--operation','attention_sublayer',
            '--build-profile-binary','/private/tmp/not-built-by-this-test',
            '--profile-variant','19','--profile-query-rows','64',
            '--profile-rows','4096','--profile-iterations','25'])
        self.assertEqual(args.profile_variant,19)

    def test_profile_cli_accepts_the_combined_attention_variant(self):
        from llm_mojo.benchmarks.profile import argument_parser
        args=argument_parser().parse_args(['--operation','attention_sublayer',
            '--build-profile-binary','/private/tmp/not-built-by-this-test',
            '--profile-variant','18','--profile-query-rows','1024',
            '--profile-rows','1024','--profile-iterations','25'])
        self.assertEqual(args.profile_variant,18)
        self.assertEqual(args.operation,'attention_sublayer')

    def test_combined_projection_profile_contract(self):
        from llm_mojo.benchmarks.attention_sublayer_contract import (
            profile_grid, COMBINED_PROFILE_WORKLOADS, STAGES_BY_VARIANT, specification)
        spec=dict(workloads=COMBINED_PROFILE_WORKLOADS,variants=[9,18])
        stages,grid=profile_grid(spec)
        self.assertEqual(len(grid),8)
        self.assertEqual(stages[18],STAGES_BY_VARIANT[9])
        self.assertEqual(specification(18,17,64)['dispatches_per_iteration'],9)
        for bad in (dict(spec,variants=[9,17]),dict(spec,workloads=[(1,4096),*COMBINED_PROFILE_WORKLOADS[1:]])):
            with self.assertRaises(ValueError):
                profile_grid(bad)

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


class TokenSelectionStudyTests(unittest.TestCase):
    """Exercise the Python measurement protocol with simulated process output.

    These tests do not compile Mojo, execute kernels, or provide GPU evidence.
    """

    @staticmethod
    def process_output(command):
        _, operation, rows, layers, candidate, control, first, seed, mode, repetitions, _ = map(str, command)
        arms = [('control', control), ('candidate', candidate)]
        if first == '1':
            arms.reverse()
        return '\n'.join([
            'device: Apple Test GPU', 'api: metal', f'operation: {operation}',
            f'shape: {rows} {layers} seed: {seed}',
            f'variants: {control} {candidate} candidate-first: {first}',
            'correctness: passed',
            *[f'SAMPLE {arm} {variant} {rep} 100'
              for arm, variant in arms for rep in range(int(repetitions))],
            'BENCHMARK_COMPLETE',
        ]) + '\n'

    def test_token_selection_specification_declares_its_actual_measurement(self):
        from llm_mojo.benchmarks.study import STUDIES, comparisons
        spec = STUDIES['token_selection']
        self.assertEqual(spec['operation'], 'token_selection')
        self.assertIs(spec['opt_in'], True)
        self.assertEqual(spec['control'], 0)
        self.assertEqual(spec['candidates'], [0, 1, 2])
        self.assertEqual(spec['rows'], [1, 16])
        self.assertEqual(spec['layers'], [1, 24])
        self.assertEqual(set(spec['names']), {0, 1, 2})
        self.assertEqual(len(comparisons(spec)), 12)
        self.assertRegex(spec['arithmetic'].lower(), r'bf16|bfloat16')
        self.assertIn('uint32', spec['arithmetic'].lower())
        self.assertIn('151936', spec['layout'])
        for field in ('inputs', 'timing'):
            with self.subTest(field=field):
                self.assertIn('resident', spec[field].lower())

    def test_token_selection_runner_preserves_argv_metadata_and_both_arm_orders(self):
        from llm_mojo.benchmarks import run as runner
        from llm_mojo.benchmarks.study import STUDIES
        spec = STUDIES['token_selection']
        repo = dict(commit='a' * 40, branch='test', dirty=False)
        sources = {'fixture-source': 'fixture-digest'}
        environment = {'fixture-environment': True}
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            build = root / 'build'
            build.mkdir()
            binary = build / 'operations'
            binary.write_bytes(b'Simulated benchmark binary; never executed.\n')
            write_json(build / 'build.json', dict(
                repository=repo, sources=sources, environment=environment,
                binaries={'operations': sha(binary)},
            ))
            replacements = dict(
                ensure_record_location=None, repository_state=repo,
                source_hashes=sources, stable_environment=environment,
                checked_conditions={'power': 'AC'},
            )
            for name, value in replacements.items():
                stack.enter_context(patch.object(runner, name, return_value=value))
            process = stack.enter_context(patch.object(
                runner.subprocess, 'run', side_effect=lambda command, **kwargs:
                subprocess.CompletedProcess(command, 0, self.process_output(command), ''),
            ))
            parser = stack.enter_context(patch.object(runner, 'parse_output', wraps=parse_output))
            output = root / 'output'
            runner.run(build, output, ['token_selection'])
            record, samples, summary = load_run(output / 'token_selection')
            self.assertEqual(len(samples), 960)
            self.assertEqual(len(summary), 12)
            self.assertEqual(record['specification'], json.loads(json.dumps(spec)))
            self.assertEqual(record['dtype'], spec['arithmetic'])
            self.assertEqual(record['inputs'], spec['inputs'])
            self.assertEqual(record['timing'], spec['timing'])
            self.assertEqual(process.call_count, 48)
            self.assertEqual(parser.call_count, 48)
            commands = [list(map(str, call.args[0])) for call in process.call_args_list]
            self.assertEqual({command[0] for command in commands}, {str(binary)})
            self.assertEqual({command[1] for command in commands}, {'token_selection'})
            self.assertEqual({tuple(map(int, command[2:5])) for command in commands},
                             {(rows, layers, variant) for rows in (1, 16)
                              for layers in (1, 24) for variant in (0, 1, 2)})
            self.assertEqual({command[5] for command in commands}, {'0'})
            self.assertEqual({command[6] for command in commands}, {'0', '1'})
            self.assertEqual({tuple(command[7:]) for command in commands},
                             {('53', 'bench', str(REPETITIONS), '10')})
            self.assertEqual([bool(call.args[3]) for call in parser.call_args_list],
                             [first for first in (False, True, True, False) for _ in range(12)])
            for call in parser.call_args_list:
                self.assertEqual(call.kwargs['operation'], 'token_selection')

    def smoke_process(self, command, **kwargs):
        command = list(map(str, command))
        if '99' in command[4:6]:
            diagnostic = ('unknown token selection route'
                          if command[1] == 'token_selection' else 'invalid variant')
            return subprocess.CompletedProcess(command, 1, '', diagnostic)
        return subprocess.CompletedProcess(command, 0, self.process_output(command), '')

    def test_token_selection_smoke_covers_all_registered_routes_and_invalid_arms(self):
        from llm_mojo.benchmarks import smoke
        target = Path('/not-executed/operations')
        env = {'SIMULATED_BENCHMARK': '1'}
        with patch.object(smoke.subprocess, 'run', side_effect=self.smoke_process) as process:
            smoke.operations_smoke(target, env)
        calls = [call for call in process.call_args_list
                 if str(call.args[0][1]) == 'token_selection']
        valid = [call for call in calls if '99' not in list(map(str, call.args[0][4:6]))]
        invalid = [call for call in calls if call not in valid]
        self.assertEqual(len(valid), 24)
        self.assertEqual(len(invalid), 2)
        self.assertEqual({tuple(map(int, call.args[0][2:7])) for call in valid},
                         {(rows, layers, variant, 0, first)
                          for rows in (1, 16) for layers in (1, 24)
                          for variant in (0, 1, 2) for first in (0, 1)})
        self.assertEqual({tuple(map(int, call.args[0][4:6])) for call in invalid},
                         {(99, 0), (0, 99)})
        for call in valid:
            self.assertEqual(int(call.args[0][9]), REPETITIONS)
            self.assertEqual(call.kwargs['env'], env)

    def test_token_selection_smoke_rejects_wrong_identity_and_incomplete_output(self):
        from llm_mojo.benchmarks import smoke
        mutations = {
            'device': lambda text: text.replace('Apple Test GPU', 'CPU'),
            'operation': lambda text: text.replace('operation: token_selection', 'operation: rope'),
            'sample': lambda text: '\n'.join(line for line in text.splitlines()
                                           if not line.startswith('SAMPLE control 0 0 ')) + '\n',
            'completion': lambda text: text.replace('BENCHMARK_COMPLETE', ''),
        }
        for name, mutate in mutations.items():
            def simulated(command, **kwargs):
                result = self.smoke_process(command, **kwargs)
                if str(command[1]) == 'token_selection' and result.returncode == 0:
                    result.stdout = mutate(result.stdout)
                return result
            with self.subTest(mutation=name), \
                    patch.object(smoke.subprocess, 'run', side_effect=simulated), \
                    self.assertRaises((ValueError, RuntimeError)):
                smoke.operations_smoke(Path('/not-executed/operations'), {})

    def test_token_selection_smoke_rejects_an_accepted_invalid_arm(self):
        from llm_mojo.benchmarks import smoke
        for invalid_arm in ('candidate', 'control'):
            def simulated(command, **kwargs):
                result = self.smoke_process(command, **kwargs)
                index = 4 if invalid_arm == 'candidate' else 5
                if str(command[1]) == 'token_selection' and str(command[index]) == '99':
                    result.returncode = 0
                return result
            with self.subTest(arm=invalid_arm), \
                    patch.object(smoke.subprocess, 'run', side_effect=simulated), \
                    self.assertRaises(RuntimeError):
                smoke.operations_smoke(Path('/not-executed/operations'), {})

    def test_token_selection_smoke_does_not_count_unrelated_failures_as_invalid_routes(self):
        from llm_mojo.benchmarks import smoke
        for invalid_arm in ('candidate', 'control'):
            def simulated(command, **kwargs):
                result = self.smoke_process(command, **kwargs)
                index = 4 if invalid_arm == 'candidate' else 5
                if str(command[1]) == 'token_selection' and str(command[index]) == '99':
                    result.stderr = 'device unavailable'
                return result
            with self.subTest(arm=invalid_arm), \
                    patch.object(smoke.subprocess, 'run', side_effect=simulated), \
                    self.assertRaisesRegex(RuntimeError, 'expected diagnostic'):
                smoke.operations_smoke(Path('/not-executed/operations'), {})
