import unittest
import json
import gzip
import hashlib
import copy
import statistics
import subprocess
import tempfile
from pathlib import Path
from llm_mojo.benchmarks import model_contract as contract
from llm_mojo.benchmarks.model_profile import parse_samples, summarize
from llm_mojo.benchmarks.capture_trace import parse_target_identity


def _batch_stdout(context, drop=None, marked=(3, 1), unsorted=False, comparisons=4, study=None):
    from llm_mojo.benchmarks.model_profile import batch_sizes
    lines = ['device: Apple M4 Pro', 'api: metal'] + ([f'study: {study}'] if study else [])
    for sequences in batch_sizes(context):
        for comparison in range(comparisons):
            for arm in range(2):
                for sample in range(10):
                    if (sequences, comparison, arm, sample) == drop:
                        continue
                    elapsed = 1000*sequences + 100*comparison + 10*arm + sample + 500
                    record = f'BATCH {sequences} {comparison} {arm} {sample} {elapsed}'
                    if (comparison, arm) == marked:
                        marks = [10*i for i in range(10)]
                        if unsorted:
                            marks[3], marks[4] = marks[4], marks[3]
                        record += ' ' + ' '.join(map(str, marks))
                    lines.append(record)
    return '\n'.join(lines + ['BATCH_COMPLETE']) + '\n'


class ModelProfileTests(unittest.TestCase):
    def test_batch_contract_declares_its_matrix_and_trace_geometry(self):
        batch = contract.BATCH_IMPLEMENTATION
        self.assertEqual(contract.options(batch), contract.options('qwen_model_all_three'))
        self.assertEqual(len(contract.command_stages(*contract.options(batch))), 249)
        self.assertEqual(len(contract.batch_workloads()), 22)
        mixed = contract.mixed_contexts()
        self.assertEqual((len(mixed), mixed[0], mixed[-1]), (32, 64, 3968))
        for context, sequences, tile in contract.BATCH_TRACES:
            spec = contract.batch_specification(context, sequences, tile)
            self.assertEqual((spec['profile_rows'], spec['key_value_rows'], spec['dispatches_per_iteration']),
                             (sequences, context+1, 245))
            data = dict(implementation=batch, entrypoint=contract.ENTRYPOINTS[batch], row_tile=tile,
                        profile_iterations=8, profile_warmup_iterations=10, **spec)
            self.assertEqual(contract.configuration(data), dict(spec, row_tile=tile))
            for key, value in (('profile_rows', sequences+1), ('row_tile', 8), ('profile_iterations', 7)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    contract.configuration(dict(data, **{key: value}))
        with self.assertRaises(ValueError):
            contract.batch_specification(1024, 8, 4)

    def test_batch_samples_require_complete_census_and_observed_arm(self):
        from llm_mojo.benchmarks.model_profile import parse_batch_samples
        for context in (64, 0):
            records = parse_batch_samples(_batch_stdout(context), context, 2)
            self.assertEqual(len(records), (7 if context else 1)*4*2*10)
            self.assertTrue(all(bool(r['marks']) == ((r['comparison'], r['arm']) == (3, 1)) for r in records))
        for invalid in (_batch_stdout(64, drop=(16, 2, 1, 4)), _batch_stdout(64, marked=(2, 1)),
                        _batch_stdout(64, unsorted=True), _batch_stdout(64).replace('api: metal', 'api: cpu'),
                        _batch_stdout(64).replace('BATCH_COMPLETE\n', '')):
            with self.subTest(), self.assertRaises(ValueError):
                parse_batch_samples(invalid, 64, 0)

    def test_batch_summary_applies_the_decision_rule_per_tile(self):
        from llm_mojo.benchmarks.model_profile import batch_summarize, HOST_PHASES
        samples = []
        for context, sequences in contract.batch_workloads():
            for block in range(4):
                for comparison in range(4):
                    for arm in range(2):
                        for sample in range(10):
                            base = 1_000_000*(sequences+1)
                            scale = {1: .8, 2: 1.2}.get(comparison, 1) if arm else 1
                            elapsed = int(base*scale) + sample
                            marks = [elapsed*i//10 for i in range(10)] if (comparison, arm) == (3, 1) else []
                            samples.append(dict(context=context, sequences=sequences, block=block, comparison=comparison,
                                                arm=arm, sample=sample, elapsed_ns=elapsed, marks=marks))
        summary = batch_summarize(samples)
        self.assertEqual(len(summary), 22)
        for row in summary:
            self.assertEqual((row['tiles']['8']['outcome'], row['tiles']['16']['outcome']), ('faster', 'slower'))
            self.assertAlmostEqual(row['tokens_per_second'], row['sequences']*1000/row['step_ms'])
            self.assertEqual(set(row['host_phase_ms']), set(HOST_PHASES))
            if row['context']:
                self.assertAlmostEqual(row['throughput_vs_one'], row['tokens_per_second']/next(
                    r['tokens_per_second'] for r in summary if (r['context'], r['sequences']) == (row['context'], 1)))
            else:
                self.assertIsNone(row['throughput_vs_one'])
        with self.assertRaises(ValueError):
            batch_summarize(samples[1:])

    def test_projection_contract_declares_arrangements_and_trace_geometry(self):
        from llm_mojo.benchmarks.model_profile import batch_comparisons, batch_study
        declaration = contract.BATCH_PROJECTION_DECLARATION
        self.assertEqual(json.loads(json.dumps(declaration)), declaration)
        self.assertNotIn('row_tiles', declaration)
        self.assertEqual(len(declaration['comparisons']), 5)
        self.assertEqual([int(a) for a in declaration['arrangements']], [0, 3, 4, 5, 6])
        self.assertEqual(batch_study('projections')['traces'][0][3:], ('batch-profile-1024-64-a0', dict(arrangement=0)))
        self.assertEqual(batch_study('size')['traces'][1][2:4], (0, 'batch-profile-1024-16-4'))
        for context, sequences, arrangement in contract.BATCH_PROJECTION_TRACES:
            spec = contract.batch_projection_specification(context, sequences, arrangement)
            self.assertEqual(spec['profile_workload'], f'model-p1024-b64-a{arrangement}')
            data = dict(implementation=contract.BATCH_IMPLEMENTATION, arrangement=arrangement,
                        entrypoint=contract.ENTRYPOINTS[contract.BATCH_IMPLEMENTATION],
                        profile_iterations=8, profile_warmup_iterations=10, **spec)
            self.assertEqual(contract.configuration(data), dict(spec, arrangement=arrangement))
            for change in (dict(arrangement=1), dict(profile_rows=16), dict(row_tile=4)):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    contract.configuration(dict(data, **change))
        self.assertEqual(batch_comparisons('size'), (4, 3))
        self.assertEqual(batch_comparisons('projections'), (5, None))
        self.assertEqual(batch_comparisons('confirm:6'), (2, None))
        for invalid in ('confirm:1', 'confirm:', 'confirm:x', 'tiles'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                batch_comparisons(invalid)

    def test_projection_samples_carry_no_marks_and_name_their_study(self):
        from llm_mojo.benchmarks.model_profile import parse_batch_samples
        stdout = _batch_stdout(64, marked=None, comparisons=5, study='projections')
        self.assertEqual(len(parse_batch_samples(stdout, 64, 1, 5, None, 'projections')), 7*5*2*10)
        confirm = _batch_stdout(0, marked=None, comparisons=2, study='confirm:5')
        self.assertEqual(len(parse_batch_samples(confirm, 0, 0, 2, None, 'confirm:5')), 2*2*10)
        for invalid, arguments in ((_batch_stdout(64, marked=(4, 1), comparisons=5, study='projections'), (5, None, 'projections')),
                                   (stdout, (5, None, 'confirm:5')), (stdout, (4, 3, 'projections')),
                                   (_batch_stdout(64, marked=None, comparisons=5), (5, None, 'projections'))):
            with self.subTest(), self.assertRaises(ValueError):
                parse_batch_samples(invalid, 64, 0, *arguments)

    def test_projection_decision_qualifies_then_selects_one_arrangement(self):
        from llm_mojo.benchmarks.model_profile import projection_decision, projection_summarize
        def samples(ratio, arrangements=contract.BATCH_PROJECTION_ARRANGEMENTS):
            rows = []
            for context, sequences in contract.batch_workloads():
                for block in range(4):
                    for comparison in range(1+len(arrangements)):
                        for arm in range(2):
                            scale = ratio(context, sequences, arrangements[comparison-1]) if arm and comparison else 1
                            for sample in range(10):
                                rows.append(dict(context=context, sequences=sequences, block=block, comparison=comparison,
                                                 arm=arm, sample=sample, elapsed_ns=int(1_000_000*sequences*scale)+sample,
                                                 marks=[]))
            return rows
        # 4 regresses at B = 2 and 5 is inconclusive in one cell, so only 3 and 6 can qualify.
        def rule(three, six, six_at_eight=None):
            def ratio(context, sequences, arrangement):
                if arrangement == 4:
                    return 1.2 if sequences == 2 else .8
                if arrangement == 5:
                    return .97 if (context, sequences) == (64, 4) else .7
                if arrangement == 6 and sequences == 8 and six_at_eight:
                    return six_at_eight
                return three if arrangement == 3 else six
            return ratio
        def decide(ratio):
            return projection_decision(projection_summarize(samples(ratio), contract.BATCH_PROJECTION_ARRANGEMENTS))
        summary = projection_summarize(samples(rule(.9, .85)), contract.BATCH_PROJECTION_ARRANGEMENTS)
        self.assertEqual(len(summary), 22)
        cell = next(r for r in summary if (r['context'], r['sequences']) == (1024, 16))
        self.assertEqual({a: x['outcome'] for a, x in cell['arrangements'].items()},
                         {'3': 'faster', '4': 'faster', '5': 'faster', '6': 'faster'})
        decision = projection_decision(summary)
        self.assertEqual([q['arrangement'] for q in decision['qualified']], [6, 3])
        self.assertEqual(decision['selected'], 6)
        # Equal worst ratios fall to the mean, then to the lower ID.
        self.assertEqual(decide(rule(.85, .85, six_at_eight=.8))['selected'], 6)
        self.assertEqual(decide(rule(.85, .85))['selected'], 3)
        ratio = rule(.9, .85)
        confirmation = projection_summarize(samples(ratio, (5,)), (5,))
        self.assertIsNone(projection_decision(confirmation)['selected'])
        self.assertEqual(projection_decision(projection_summarize(samples(lambda c, b, a: .9, (5,)), (5,)))['selected'], 5)
        broken = samples(ratio)
        with self.assertRaises(ValueError):
            projection_summarize(broken[1:], contract.BATCH_PROJECTION_ARRANGEMENTS)
        broken[0]['marks'] = [1]
        with self.assertRaises(ValueError):
            projection_summarize(broken, contract.BATCH_PROJECTION_ARRANGEMENTS)

    def test_reordered_contract_accuracy_and_tokens(self):
        from llm_mojo.benchmarks.model_profile import (accuracy_gate, batch_comparisons, batch_study, parse_accuracy,
                                                       parse_batch_tokens)
        declaration = contract.BATCH_REORDERED_DECLARATION
        self.assertEqual(json.loads(json.dumps(declaration)), declaration)
        self.assertEqual([int(a) for a in declaration['arrangements']], [5, 7, 8, 9, 10])
        self.assertEqual(batch_study('reordered')['traces'][-1][3:], ('batch-profile-1024-64-a10', dict(arrangement=10)))
        spec = contract.batch_projection_specification(1024, 64, 9)
        data = dict(implementation=contract.BATCH_IMPLEMENTATION, arrangement=9, profile_iterations=8,
                    entrypoint=contract.ENTRYPOINTS[contract.BATCH_IMPLEMENTATION], profile_warmup_iterations=10, **spec)
        self.assertEqual(contract.configuration(data), dict(spec, arrangement=9))
        self.assertEqual(batch_comparisons('reordered'), (5, None))
        self.assertEqual(batch_comparisons('reordered-confirm:9'), (2, None))
        for invalid in ('reordered-confirm:5', 'reordered-confirm:3', 'confirm:9'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                batch_comparisons(invalid)
        def census(worst):
            lines = ['device: Apple M4 Pro', 'api: metal']
            for rows, n, k in contract.BATCH_ACCURACY_SHAPES:
                for a in (5, 7, 8, 9, 10):
                    lines.append(f'ACCURACY {a} {rows} {n} {k} {rows*n} 3 1 0 {worst(a, n)}')
            return '\n'.join(lines + ['ACCURACY_COMPLETE']) + '\n'
        records = parse_accuracy(census(lambda a, n: 2.5 if a >= 9 and n == 151936 else 1.25))
        self.assertEqual(len(records), 25)
        self.assertEqual(accuracy_gate(records), {7: True, 8: True, 9: False, 10: False})
        for bad in (census(lambda a, n: 1).replace('ACCURACY 10 2', 'ACCURACY 11 2'),
                    census(lambda a, n: 1).replace(' 3 1 0 ', ' 1 3 0 '),
                    census(lambda a, n: 1).replace('ACCURACY_COMPLETE\n', '')):
            with self.subTest(), self.assertRaises(ValueError):
                parse_accuracy(bad)
        tokens = parse_batch_tokens('tokens: 16 5 0\ntokens: 16 9 2\nBATCH 16 0 0 0 5\n', 1024, 3)
        self.assertEqual([(t['arrangement'], t['differing']) for t in tokens], [(5, 0), (9, 2)])
        with self.assertRaises(ValueError):
            parse_batch_tokens('tokens: 16 9 17\n', 1024, 3)

    def test_reordered_decision_needs_accuracy_one_row_and_large_batches(self):
        from llm_mojo.benchmarks.model_profile import diagnostic_stop, hf_summary, projection_decision, projection_summarize
        arrangements = contract.BATCH_REORDERED_ARRANGEMENTS
        def samples(ratio):
            rows = []
            for context, sequences in contract.batch_workloads():
                for block in range(4):
                    for comparison in range(1+len(arrangements)):
                        for arm in range(2):
                            scale = ratio(sequences, arrangements[comparison-1]) if arm and comparison else 1
                            for sample in range(10):
                                rows.append(dict(context=context, sequences=sequences, block=block, comparison=comparison,
                                                 arm=arm, sample=sample, elapsed_ns=int(1_000_000*sequences*scale)+sample,
                                                 marks=[]))
            return rows
        # 7 gains from B = 16 only; 8 gains everywhere; 9 is fastest but fails accuracy; 10 is slower at one row.
        table = {7: lambda b: .9 if b >= 16 else 1, 8: lambda b: .8, 9: lambda b: .5, 10: lambda b: 1.2 if b == 1 else .6}
        summary = projection_summarize(samples(lambda b, a: table[a](b)), arrangements)
        eligible = {7: True, 8: True, 9: False, 10: True}
        decision = projection_decision(summary, 1, 16, eligible)
        self.assertEqual([q['arrangement'] for q in decision['qualified']], [8, 7])
        self.assertEqual(decision['selected'], 8)
        self.assertEqual(projection_decision(summary, 1, 16, {**eligible, 8: False})['selected'], 7)
        choices = [dict(case='c', call=i, token=1, reference_token=1 if i < 9 else 2, kl_nats=0.01*(i+1),
                        total_variation=0.1, reference_margin=1.0) for i in range(10)]
        exact = hf_summary(choices)
        self.assertEqual((exact['decode_choices'], exact['agree'], exact['max_kl_nats']), (10, 9, 0.1))
        for candidate, stop in ((choices, False), ([dict(c, token=3) if c['call'] == 0 else c for c in choices], False),
                                ([dict(c, token=3) if c['call'] < 2 else c for c in choices], True),
                                ([dict(c, kl_nats=0.25) if c['call'] == 3 else c for c in choices], True)):
            with self.subTest(stop=stop):
                self.assertEqual(diagnostic_stop(dict(selected=8, hf={'5': exact, '8': hf_summary(candidate)})), stop)

    def test_retained_batch_size_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import batch_replay
        source = repository_root()/'studies/model_generation'
        original = json.loads(gzip.decompress((source/'batch-size.json.gz').read_bytes()))
        retained = json.loads((source/'batch-size-summary.json').read_text())
        for damage in (None, 'sample', 'block', 'capture', 'dispatch', 'provenance', 'conditions',
                       'trace-conditions', 'rejection'):
            record = copy.deepcopy(original)
            if damage == 'sample': record['timing']['samples'].pop()
            elif damage == 'block': record['timing']['blocks'].pop()
            elif damage == 'capture': record['captures'].pop()
            elif damage == 'dispatch': record['captures'][2]['samples'].pop()
            elif damage == 'provenance': record['captures'][2]['provenance']['binary']['sha256'] = '0'*64
            elif damage == 'conditions': record['timing']['blocks'][3]['after']['power_mode_raw'] = '1'
            elif damage == 'trace-conditions': record['captures'][0]['conditions']['before']['battery']['power_source'] = 'Battery Power'
            elif damage == 'rejection': record['rejected_captures'][0]['receipt']['profile']['binary']['sha256'] = '0'*64
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                raw = json.dumps(record).encode()
                packed = gzip.compress(raw, mtime=0)
                (directory/'batch-size.json.gz').write_bytes(packed)
                (directory/'batch-size.json').write_text(json.dumps(dict(kind=record['kind'],
                    sha256=hashlib.sha256(packed).hexdigest(), uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with self.subTest(damage=damage), redirect_stdout(StringIO()):
                    if damage is None:
                        batch_replay(directory)
                        self.assertEqual(json.loads((directory/'batch-size-summary.json').read_text()), retained)
                        self.assertEqual(len(record['rejected_captures']), 4)
                    else:
                        with self.assertRaises((ValueError, RuntimeError)): batch_replay(directory)

    def test_retained_batch_projections_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import projection_replay
        source = repository_root()/'studies/model_generation'
        original = json.loads(gzip.decompress((source/'batch-projections.json.gz').read_bytes()))
        retained = json.loads((source/'batch-projections-summary.json').read_text())
        self.assertEqual((retained['decision']['selected'], retained['confirmed']), (5, True))
        for damage in (None, 'sample', 'confirmation-sample', 'block', 'capture', 'dispatch', 'provenance',
                       'conditions', 'confirmation-conditions', 'trace-conditions', 'no-confirmation',
                       'other-arrangement', 'argument'):
            record = copy.deepcopy(original)
            if damage == 'sample': record['timing']['samples'].pop()
            elif damage == 'confirmation-sample': record['confirmation']['samples'].pop()
            elif damage == 'block': record['timing']['blocks'].pop()
            elif damage == 'capture': record['captures'].pop()
            elif damage == 'dispatch': record['captures'][2]['samples'].pop()
            elif damage == 'provenance': record['captures'][2]['provenance']['binary']['sha256'] = '0'*64
            elif damage == 'conditions': record['timing']['blocks'][3]['after']['power_mode_raw'] = '1'
            elif damage == 'confirmation-conditions': record['confirmation']['blocks'][0]['before']['power_mode_raw'] = '1'
            elif damage == 'trace-conditions': record['captures'][0]['conditions']['before']['battery']['power_source'] = 'Battery Power'
            elif damage == 'no-confirmation': record['confirmation'] = None
            elif damage == 'other-arrangement': record['confirmation']['argument'] = 'confirm:6'
            elif damage == 'argument': record['timing']['argument'] = 'size'
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                raw = json.dumps(record).encode()
                packed = gzip.compress(raw, mtime=0)
                (directory/'batch-projections.json.gz').write_bytes(packed)
                (directory/'batch-projections.json').write_text(json.dumps(dict(kind=record['kind'],
                    sha256=hashlib.sha256(packed).hexdigest(), uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with self.subTest(damage=damage), redirect_stdout(StringIO()):
                    if damage is None:
                        projection_replay(directory)
                        self.assertEqual(json.loads((directory/'batch-projections-summary.json').read_text()), retained)
                    else:
                        with self.assertRaises((ValueError, RuntimeError)): projection_replay(directory)

    def test_retained_batch_reordered_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import reordered_replay
        source = repository_root()/'studies/model_generation'
        original = json.loads(gzip.decompress((source/'batch-reordered.json.gz').read_bytes()))
        retained = json.loads((source/'batch-reordered-summary.json').read_text())
        self.assertEqual((retained['decision']['selected'], retained['confirmed'], retained['diagnostics']['stop']),
                         (8, True, False))
        self.assertEqual(sorted(r['status'] for r in original['rejected_captures']),
                         ['failed during capture', 'rejected by analysis'])
        for damage in (None, 'sample', 'confirmation-sample', 'block', 'capture', 'dispatch', 'provenance',
                       'conditions', 'confirmation-conditions', 'trace-conditions', 'no-confirmation',
                       'other-arrangement', 'argument', 'accuracy', 'accuracy-worse', 'token', 'exact-token',
                       'no-diagnostics', 'stop', 'hf', 'set-aside-binary', 'failed-receipt'):
            record = copy.deepcopy(original)
            failed = next(r for r in record['rejected_captures'] if r['status'] == 'failed during capture')
            if damage == 'sample': record['timing']['samples'].pop()
            elif damage == 'confirmation-sample': record['confirmation']['samples'].pop()
            elif damage == 'block': record['timing']['blocks'].pop()
            elif damage == 'capture': record['captures'].pop()
            elif damage == 'dispatch': record['captures'][2]['samples'].pop()
            elif damage == 'provenance': record['captures'][2]['provenance']['binary']['sha256'] = '0'*64
            elif damage == 'conditions': record['timing']['blocks'][3]['after']['power_mode_raw'] = '1'
            elif damage == 'confirmation-conditions': record['confirmation']['blocks'][0]['before']['power_mode_raw'] = '1'
            elif damage == 'trace-conditions': record['captures'][0]['conditions']['before']['battery']['power_source'] = 'Battery Power'
            elif damage == 'no-confirmation': record['confirmation'] = None
            elif damage == 'other-arrangement': record['confirmation']['argument'] = 'reordered-confirm:7'
            elif damage == 'argument': record['timing']['argument'] = 'projections'
            elif damage == 'accuracy': record['timing']['accuracy'].pop()
            elif damage == 'accuracy-worse': next(r for r in record['timing']['accuracy'] if r['arrangement'] == 8)['worst_ulps'] += 1
            elif damage == 'token': record['timing']['token_differences'].pop()
            elif damage == 'exact-token': next(r for r in record['timing']['token_differences'] if r['arrangement'] == 7)['differing'] = 1
            elif damage == 'no-diagnostics': record['diagnostics'] = None
            elif damage == 'stop': record['diagnostics']['stop'] = True
            elif damage == 'hf': record['diagnostics']['hf']['8']['agree'] -= 1
            elif damage == 'set-aside-binary': record['rejected_captures'][0]['receipt']['profile']['binary']['sha256'] = '0'*64
            elif damage == 'failed-receipt': failed['receipt']['capture']['status'] = 'complete'
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                raw = json.dumps(record).encode()
                packed = gzip.compress(raw, mtime=0)
                (directory/'batch-reordered.json.gz').write_bytes(packed)
                (directory/'batch-reordered.json').write_text(json.dumps(dict(kind=record['kind'],
                    sha256=hashlib.sha256(packed).hexdigest(), uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with self.subTest(damage=damage), redirect_stdout(StringIO()):
                    if damage is None:
                        reordered_replay(directory)
                        self.assertEqual(json.loads((directory/'batch-reordered-summary.json').read_text()), retained)
                    else:
                        with self.assertRaises((ValueError, RuntimeError)): reordered_replay(directory)

    def test_retained_single_sequence_check(self):
        """1e's adoption gate: Fast decode of one sequence in arrangement 8 against 5, from the raw steps."""
        from llm_mojo._repository import repository_root
        record = json.loads((repository_root()/'studies/model_generation/batch-reordered-single-sequence.json').read_text())
        def summary(record):
            runs = record['runs']
            if ([r['run'] for r in runs] != list(range(1, 17))
                    or [[r['arrangement'] for r in runs if r['block'] == b] for b in range(1, 5)] != [[5, 8, 8, 5], [8, 5, 5, 8]]*2):
                raise ValueError('the runs are not four alternating blocks')
            if any(len(r['decode_step_ns']) != record['prompt']['max_new_tokens']-1
                   or r['median_ms'] != statistics.median(r['decode_step_ns'])/1e6 for r in runs):
                raise ValueError('a run lost a decode step or misstates its median')
            ratios = [statistics.mean(r['median_ms'] for r in runs if r['block'] == b and r['arrangement'] == 8)
                      / statistics.mean(r['median_ms'] for r in runs if r['block'] == b and r['arrangement'] == 5)
                      for b in range(1, 5)]
            median = statistics.median(ratios)
            verdict = ('regression' if all(r > 1 for r in ratios) and median > 1.05 else
                       'consistent slowdown below the floor' if all(r > 1 for r in ratios) else 'no regression')
            return ratios, median, verdict
        self.assertEqual(summary(record), (record['block_ratios'], record['median_block_ratio'], record['verdict']))
        self.assertEqual(record['verdict'], 'no regression')
        self.assertTrue(all(r < 1 for r in record['block_ratios']) and record['texts_identical'])
        self.assertEqual({a: (b['decode_projection'], b['dirty']) for a, b in record['binaries'].items()},
                         {'5': (5, False), '8': (8, False)})
        self.assertNotEqual(record['binaries']['5']['sha256'], record['binaries']['8']['sha256'])
        for damage in ('step', 'order'):
            damaged = copy.deepcopy(record)
            if damage == 'step': damaged['runs'][1]['decode_step_ns'].pop()
            else: damaged['runs'][0]['arrangement'] = 8
            with self.subTest(damage=damage), self.assertRaises(ValueError): summary(damaged)

    def test_batch_support_distinguishes_backend_failure_from_bad_results(self):
        from llm_mojo.benchmarks.model_profile import batch_support_parse
        base='device: Apple M4 Pro\napi: metal\nBATCH_EAGER_PASS 15\n'
        unsupported=base+'BATCH_GRAPH_ERROR createGraphBuilder() not supported on this device context\nBATCH_SUPPORT_COMPLETE\n'
        self.assertEqual(batch_support_parse(unsupported)['status'],'graph-unsupported')
        supported=base+'BATCH_BUILDER_ENTERED\nBATCH_GRAPH_PASS 15 63\nBATCH_SUPPORT_COMPLETE\n'
        self.assertEqual(batch_support_parse(supported)['status'],'graph-replay-supported')
        for invalid in (unsupported.replace('api: metal','api: cpu'), unsupported.replace('BATCH_EAGER_PASS 15\n',''), unsupported.replace('createGraphBuilder() not supported on this device context','graph first replay failed'), unsupported+'BATCH_BUILDER_ENTERED\n', supported.replace('15 63','15 15')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): batch_support_parse(invalid)


    def test_retained_enqueue_evidence(self):
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import enqueue_summary
        original=json.loads(gzip.decompress((repository_root()/'studies/model_generation/runtime-enqueue.json.gz').read_bytes()))
        self.assertEqual(enqueue_summary(original),original['summary'])
        for damage in ('run','call','boundary','numerical','state'):
            record=copy.deepcopy(original)
            probe=next(r['probe'] for r in record['runs'] if r['probe'])
            if damage=='run': record['runs'].pop()
            elif damage=='call': probe['groups'][0].pop()
            elif damage=='boundary': probe['groups'][0][0][0]=1
            elif damage=='numerical': record['numerical'][0]['observations'][0]['inactive_exact']=False
            elif damage=='state': next(r for r in record['runs'] if r['kind']=='micro')['state']='disabled'
            with self.subTest(damage=damage), self.assertRaises(ValueError): enqueue_summary(record)


    def test_enqueue_partition_rejects_boundary_and_count_errors(self):
        from llm_mojo.benchmarks.model_profile import enqueue_partition, enqueue_windows
        rows=[dict(start_ns=100,end_ns=200),dict(start_ns=300,end_ns=400)]
        call=lambda a,b:[a,b,7,1,1,1,128,1,1,7,0]
        calls=[call(110,130),call(140,160),call(310,330),call(340,360)]
        self.assertEqual(enqueue_partition(rows,calls,2),[calls[:2],calls[2:]])
        for broken in (calls[:-1],[call(90,110)]+calls,calls[:1]+[call(120,145)]+calls[2:],
                       [calls[0][:-1]+[1]]+calls[1:]):
            with self.assertRaises(ValueError): enqueue_partition(rows,broken,2)
        text='device: Apple M4 Pro\napi: metal\n'
        text+=''.join(f'LAUNCH_SAMPLE {a} {i} {1000*(a*10+i)+1} {1000*(a*10+i)+100} {1000*(a*10+i)+200}\n' for a in (0,1) for i in range(10))
        text+='LAUNCH_MICRO_COMPLETE\n'
        self.assertEqual(len(enqueue_windows(text,'micro')),20)
        for invalid in (text.replace('api: metal','api: cpu'),text.replace('LAUNCH_SAMPLE 0 0 1 100 200\n',''),text.replace('0 0 1 100 200','0 0 1 200 100')):
            with self.assertRaises(ValueError): enqueue_windows(invalid,'micro')


    def test_retained_scheduling_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import scheduling_replay
        original=json.loads(gzip.decompress((repository_root()/'studies/model_generation/projection-scheduling.json.gz').read_bytes()))
        for damage in (None,'sample','token','marks','numerical','cache','capture','fragment','provenance','host','target','receipt','conditions'):
            record=copy.deepcopy(original)
            if damage=='sample': record['timing']['samples'].pop()
            elif damage=='token': record['timing']['samples'][0]['token']+=1
            elif damage=='marks': next(x for x in record['timing']['samples'] if x['marks'])['marks'][0]=-1
            elif damage=='numerical': record['timing']['numerical'].pop()
            elif damage=='cache': record['timing']['numerical'][0]['observations'][0]['prefix_exact']=False
            elif damage=='capture': record['captures'].pop()
            elif damage=='fragment': record['captures'][0]['samples'][0]['segments']+=1
            elif damage=='provenance': record['captures'][0]['provenance_text']+=' '
            elif damage=='host': record['captures'][0]['host'][0]['elapsed_ns']+=1
            elif damage=='target': record['captures'][0]['target_text']+=' '
            elif damage=='receipt': record['captures'][0]['capture_receipt_text']+=' '
            elif damage=='conditions': record['captures'][0]['conditions']['after']['power_mode_raw']='1'
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp);raw=json.dumps(record).encode();packed=gzip.compress(raw,mtime=0)
                (directory/'projection-scheduling.json.gz').write_bytes(packed)
                (directory/'projection-scheduling.json').write_text(json.dumps(dict(sha256=hashlib.sha256(packed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with redirect_stdout(StringIO()):
                    if damage is None: self.assertFalse(scheduling_replay(directory)['promote'])
                    else:
                        with self.assertRaises(ValueError): scheduling_replay(directory)

    def test_scheduling_parser_and_frozen_census(self):
        from llm_mojo.benchmarks.model_profile import scheduling_parse, scheduling_summary, SCHEDULING_MODES, scheduling_host
        stdout='device: Apple M4 Pro\napi: metal\n'
        stdout+=''.join(f'SCHED_SAMPLE {a} {i} 42 100\n' for a in (0,1) for i in range(64))
        stdout+='SCHEDULING_COMPLETE\n'
        self.assertEqual(len(scheduling_parse(stdout,'fixed',64,0,1)),128)
        for invalid in (stdout.replace('SCHED_SAMPLE 1 0 42','SCHED_SAMPLE 1 0 43'),
                        stdout.replace('SCHED_SAMPLE 1 0 42 100\n',''),stdout.replace('api: metal','api: cpu')):
            with self.assertRaises(ValueError): scheduling_parse(invalid,'fixed',64,0,1)
        rows=[dict(mode=m,prefix=p,block=b,comparison=c,arm=a,sample=i,token=42,
                   elapsed_ns=80 if c and a else 100,marks=list(range(10)) if m.startswith('observed-') else [])
              for m in SCHEDULING_MODES for p in contract.PREFIXES for b in range(4)
              for c in (0,1) for a in (0,1) for i in range(64)]
        self.assertEqual(len(rows),12288)
        self.assertTrue(all(x['qualifies'] for x in scheduling_summary(rows)))
        with self.assertRaises(ValueError): scheduling_summary(rows[:-1])
        broken=copy.deepcopy(rows);broken[0]['token']=43
        with self.assertRaises(ValueError): scheduling_summary(broken)
        text=''.join('SCHED_HOST '+str(i)+' 100 '+' '.join(map(str,range(10)))+'\n' for i in range(8))
        self.assertEqual(len(scheduling_host(text)),8)
        with self.assertRaises(ValueError): scheduling_host(text+text)

    def test_scheduling_timeline_unions_overlapping_fragments(self):
        from llm_mojo.benchmarks.model_profile import scheduling_timeline
        rows=[]
        for i in range(8):
            for stage,parts in [('gate projection',[[0,5],[7,2]]),('FP32 GQA',[[4,4]])]:
                rows.append(dict(iteration=i,kind='compute',stage=stage,active_intervals=parts,
                    duration_ns=sum(d for _,d in parts),segments=len(parts),submission_start_ns=0,submission_duration_ns=2))
        row=scheduling_timeline(dict(samples=rows))[0]
        self.assertEqual((row['active_ns'],row['span_ns'],row['uncovered_ns']),(9,9,0))
        self.assertEqual((row['projections_ns'],row['attention_ns'],row['fragmented_commands']),(7,4,1))

    def test_projection_screen_requires_all_contexts_and_confirmation(self):
        from llm_mojo.benchmarks.model_profile import projection_summary
        implementation='qwen_model_all_three'
        provenance=dict(implementation=implementation,entrypoint=contract.ENTRYPOINTS[implementation],
            **contract.specification(1024,*contract.options(implementation)),profile_iterations=8,
            profile_warmup_iterations=10,projection_variant=5)
        self.assertEqual(contract.configuration(provenance)['projection_variant'],5)
        for invalid in (-1,6,True):
            with self.assertRaises(ValueError): contract.configuration(dict(provenance,projection_variant=invalid))
        output='''device: Apple M4 Pro
api: metal
correctness: passed
profile implementation: QwenModel.forward+greedy-all-three
rows: 1
hidden: 896
key value rows: 1025
profile workload: model-p1024-all-three
profile dispatches per iteration: 245
warmup iterations: 10
profile iterations: 8
post-profile idle milliseconds: 250
projection arrangement: 5
'''
        self.assertEqual(parse_target_identity(output)['projection_variant'],5)
        with self.assertRaises(ValueError): parse_target_identity(output+'projection arrangement: 5\n')
        samples=[dict(prefix=p,block=b,comparison=c,arm=a,sample=i,marks=[],
                      elapsed_ns=8000 if a and c==1 else 9000 if a and c else 10000)
                 for p in contract.PREFIXES for b in range(4) for c in range(6)
                 for a in range(2) for i in range(10)]
        summary=projection_summary(samples)
        self.assertEqual(summary['screen_selected'],1)
        self.assertFalse(summary['promote'])
        self.assertTrue(summary['confirmation_required'])
        with self.assertRaises(ValueError): projection_summary(samples[:-1])
        for r in samples:
            if r['prefix']==3968 and r['arm']==1: r['elapsed_ns']=10000
        self.assertEqual(projection_summary(samples)['screen_selected'],0)

    def test_retained_projection_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import selection_replay
        source=repository_root()/'studies/model_generation/projection-arrangements.json.gz'
        original=json.loads(gzip.decompress(source.read_bytes()))
        damages=[None,'sample','numerical','cache','layer','dispatch','provenance',
                 'provenance-bytes','variant','target-variant','fragment','terminal','conditions','confirmation']
        for damage in damages:
            record=copy.deepcopy(original)
            if damage=='sample': record['timing']['samples'].pop()
            elif damage=='numerical': record['timing']['numerical'].pop()
            elif damage=='cache': record['timing']['numerical'][0]['observations'][0]['exact']=False
            elif damage=='layer': next(n for n in record['timing']['numerical'] if n['prefix']==64)['layers'].pop()
            elif damage=='dispatch': record['captures'][5]['samples'].pop()
            elif damage=='provenance': record['captures'][5]['provenance']['binary']['sha256']='0'*64
            elif damage=='provenance-bytes': record['captures'][5]['provenance_text']+=' '
            elif damage=='variant': record['captures'][5]['provenance']['projection_variant']=4
            elif damage=='target-variant': record['captures'][5]['analysis']['capture_identity']['workload']['projection_variant']=4
            elif damage=='fragment': record['captures'][5]['samples'][0]['segments']+=1
            elif damage=='terminal': record['terminal']['blocks'][0]['arms'][5]['turns'][0]['generated'][0]=0
            elif damage=='conditions': record['captures'][0]['conditions']['after']['power_mode_raw']='1'
            elif damage=='confirmation':
                record['confirmation']=None if record['confirmation'] else {'selected':1}
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                raw=json.dumps(record).encode(); packed=gzip.compress(raw,mtime=0)
                (directory/'projection-arrangements.json.gz').write_bytes(packed)
                (directory/'projection-arrangements.json').write_text(json.dumps(dict(sha256=hashlib.sha256(packed).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with redirect_stdout(StringIO()):
                    if damage is None:
                        self.assertFalse(selection_replay(directory,projection=True)['confirmation_required'])
                    else:
                        with self.assertRaises(ValueError): selection_replay(directory,projection=True)

    def test_retained_residual_norm_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import selection_replay
        source=repository_root()/'studies/model_generation/residual-norm.json.gz'
        original=json.loads(gzip.decompress(source.read_bytes()))
        for damage in (None,'sample','numerical','cache','layer','lifecycle','owners',
                       'dispatch','provenance','provenance-bytes','fragment','terminal','conditions'):
            record=copy.deepcopy(original)
            if damage=='sample': record['timing']['samples'].pop()
            elif damage=='numerical': record['timing']['numerical'].pop()
            elif damage=='cache': record['timing']['numerical'][0]['observations'][0]['exact']=False
            elif damage=='layer': record['timing']['numerical'][0]['swap_checks']['layers'].pop()
            elif damage=='lifecycle': record['timing']['numerical'][0]['swap_checks']['states'][1][2]=3
            elif damage=='owners': record['timing']['numerical'][0]['swap_checks']['owners_checked']=False
            elif damage=='dispatch': record['captures'][3]['samples'].pop()
            elif damage=='provenance': record['captures'][3]['provenance']['binary']['sha256']='0'*64
            elif damage=='provenance-bytes': record['captures'][3]['provenance_text']+=' '
            elif damage=='fragment': record['captures'][3]['samples'][0]['segments']+=1
            elif damage=='terminal': record['terminal']['blocks'][0]['arms'][3]['turns'][0]['generated'][0]=0
            elif damage=='conditions': record['captures'][0]['conditions']['after']['power_mode_raw']='1'
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                raw=json.dumps(record).encode(); packed=gzip.compress(raw,mtime=0)
                (directory/'residual-norm.json.gz').write_bytes(packed)
                (directory/'residual-norm.json').write_text(json.dumps(dict(sha256=hashlib.sha256(packed).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with redirect_stdout(StringIO()):
                    if damage is None:
                        self.assertEqual(selection_replay(directory,composition=True)['selected'],3)
                    else:
                        with self.assertRaises(ValueError): selection_replay(directory,composition=True)

    def test_residual_norm_composition_geometry_and_choice(self):
        from llm_mojo.benchmarks.model_profile import composition_summary, swap_capture_names
        self.assertEqual(len(swap_capture_names(extra_norm=True)),195)
        self.assertEqual([len(contract.stages(*contract.options(i))) for i in contract.COMPOSITION_IMPLEMENTATIONS],[314,266,293,245])
        for implementation in contract.COMPOSITION_IMPLEMENTATIONS:
            fields=contract.specification(1024,*contract.options(implementation))
            contract.configuration(dict(implementation=implementation,entrypoint=contract.ENTRYPOINTS[implementation],
                                        **fields,profile_iterations=8,profile_warmup_iterations=10))
        latencies=[10000,9000,8500,7500]
        samples=[dict(prefix=p,block=b,comparison=c,arm=a,sample=s,marks=[],
                      elapsed_ns=latencies[contract.COMPOSITION_PAIRS[c][a]])
                 for p in contract.PREFIXES for b in range(4) for c in range(6) for a in range(2) for s in range(10)]
        self.assertEqual(composition_summary(samples)['selected'],3)
        with self.assertRaises(ValueError): composition_summary(samples[:-1])
        # A qualifying combined route must also beat other qualifiers directly.
        damaged=copy.deepcopy(samples)
        for row in damaged:
            if row['comparison']==4 and row['arm']==1: row['elapsed_ns']=10000
        self.assertEqual(composition_summary(damaged)['selected'],0)
        for row in samples:
            variant=contract.COMPOSITION_PAIRS[row['comparison']][row['arm']]
            if variant in (1,3): row['elapsed_ns']=11000
        self.assertEqual(composition_summary(samples)['selected'],2)
        for row in samples:
            if row['comparison']==0 and row['arm']==1: row['elapsed_ns']=14000
        self.assertEqual(composition_summary(samples)['selected'],0)

    def test_retained_buffer_swap_evidence_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import fusion_replay
        source=repository_root()/'studies/model_generation'
        original=json.loads(gzip.decompress((source/'buffer-swap.json.gz').read_bytes()))
        for damage in (None,'sample','cache','layer','lifecycle','owners','dispatch','provenance','terminal','conditions','trace-conditions','block'):
            record=copy.deepcopy(original)
            if damage=='sample': record['timing']['samples'].pop()
            elif damage=='cache': record['timing']['numerical'][0]['observations'][0]['exact']=False
            elif damage=='layer': record['timing']['numerical'][0]['swap_checks']['layers'].pop()
            elif damage=='lifecycle': record['timing']['numerical'][0]['swap_checks']['states'][1][2]=3
            elif damage=='owners': record['timing']['numerical'][0]['swap_checks']['owners_checked']=False
            elif damage=='dispatch': record['captures'][1]['samples'].pop()
            elif damage=='provenance': record['captures'][1]['provenance']['binary']['sha256']='0'*64
            elif damage=='terminal': record['terminal']['blocks'][0]['arms'][0]['turns'][0]['generated'][0]=0
            elif damage=='conditions': record['timing']['blocks'][0]['before']['power_mode_raw']='1'
            elif damage=='trace-conditions': record['captures'][0]['conditions']['after']['power_mode_raw']='1'
            elif damage=='block': record['timing']['blocks'].pop()
            with tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                raw=json.dumps(record).encode()
                packed=gzip.compress(raw,mtime=0)
                (directory/'buffer-swap.json.gz').write_bytes(packed)
                (directory/'buffer-swap.json').write_text(json.dumps(dict(sha256=hashlib.sha256(packed).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with redirect_stdout(StringIO()):
                    if damage is None:
                        fusion_replay(directory,copy_free=True)
                        self.assertFalse(json.loads((directory/'buffer-swap-summary.json').read_text())['promote'])
                    else:
                        with self.assertRaises(ValueError): fusion_replay(directory,copy_free=True)

    def test_buffer_swap_trace_geometry_and_lifecycle_census(self):
        from llm_mojo.benchmarks.model_profile import validate_swap_checks, swap_capture_names, swap_lifecycle_names
        stages=contract.stages(copy_free=True)
        self.assertEqual(len(stages),291)
        self.assertNotIn('inter-layer copy',[stage for _,stage in stages])
        implementation='qwen_model_buffer_swap'
        fields=contract.specification(1024,copy_free=True)
        contract.configuration(dict(implementation=implementation,entrypoint=contract.ENTRYPOINTS[implementation],
            **fields,profile_iterations=8,profile_warmup_iterations=10))
        check=dict(owners_checked=True,rejection_checked=True,
            layers=[dict(name=n,exact=True,bytes=2,sha256='0'*64) for n in swap_capture_names()],
            lifecycle=[dict(name=n,exact=True,bytes=2,sha256='0'*64) for n in swap_lifecycle_names()],
            states=[[i,r,t,0] for i,r,t in [(0,3,3),(1,1,4),(2,1,5),(3,2,7),(4,1,8),(6,1,1),(7,2,3),(8,1,4)]])
        validate_swap_checks(check)
        for field in ('layers','lifecycle'):
            damaged=copy.deepcopy(check);damaged[field].pop()
            with self.assertRaises(ValueError): validate_swap_checks(damaged)
        check['owners_checked']=False
        with self.assertRaises(ValueError): validate_swap_checks(check)

    def test_retained_selection_evidence_rejects_missing_or_changed_records(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import selection_replay
        source=repository_root()/'studies/model_generation'
        original=json.loads(gzip.decompress((source/'token-selection.json.gz').read_bytes()))
        for damage in (None,'sample','cache','actual','nonfinite','dispatch','provenance','terminal','conditions'):
            record=copy.deepcopy(original)
            if damage=='sample': record['timing']['samples'].pop()
            elif damage=='cache': record['timing']['numerical'][0]['observations'][0]['exact']=False
            elif damage=='actual': record['timing']['numerical'][1]['actual'][0]['exact']=False
            elif damage=='nonfinite': record['timing']['numerical'][0]['nonfinite_invalidates']=False
            elif damage=='dispatch': record['captures'][1]['samples'].pop()
            elif damage=='provenance': record['captures'][2]['provenance']['binary']['sha256']='0'*64
            elif damage=='terminal': record['terminal']['blocks'][0]['arms'][0]['turns'][0]['generated'][0]=0
            elif damage=='conditions': record['timing']['blocks'][0]['before']['power_mode_raw']='1'
            with tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                raw=json.dumps(record).encode()
                packed=gzip.compress(raw,mtime=0)
                (directory/'token-selection.json.gz').write_bytes(packed)
                (directory/'token-selection.json').write_text(json.dumps(dict(sha256=hashlib.sha256(packed).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with redirect_stdout(StringIO()):
                    if damage is None:
                        self.assertEqual(selection_replay(directory)['selected'],0)
                    else:
                        with self.assertRaises(ValueError): selection_replay(directory)

    def test_selection_geometry_and_conservative_choice(self):
        from llm_mojo.benchmarks.model_profile import selection_summary
        self.assertEqual([len(contract.stages(True,True,s)) for s in range(3)],[314,316,315])
        for selection,implementation in enumerate(['qwen_model_combined','qwen_model_gpu_argmax','qwen_model_fused_head']):
            fields=contract.specification(1024,True,True,selection)
            contract.configuration(dict(implementation=implementation,entrypoint=contract.ENTRYPOINTS[implementation],
                **fields,profile_iterations=8,profile_warmup_iterations=10))
        samples=[dict(prefix=p,block=b,comparison=c,arm=a,sample=s,marks=[],elapsed_ns=10000000)
                 for p in contract.PREFIXES for b in range(4) for c in range(4) for a in range(2) for s in range(10)]
        self.assertEqual(selection_summary(samples)['selected'],0)
        for row in samples:
            if row['comparison'] in (1,2) and row['arm']==1:
                row['elapsed_ns']=9000000
        self.assertEqual(selection_summary(samples)['selected'],1)
        for row in samples:
            if row['comparison']==3 and row['arm']==1:
                row['elapsed_ns']=9000000
        self.assertEqual(selection_summary(samples)['selected'],2)
        # A single losing context prevents a promotion; noise is not ignored.
        for row in samples:
            if row['prefix']==3968 and row['comparison']==0 and row['arm']==1:
                row['elapsed_ns']=12000000
        self.assertEqual(selection_summary(samples)['selected'],0)
        with self.assertRaises(ValueError): selection_summary(samples[:-1])
        samples[0]['marks']=[1]
        with self.assertRaises(ValueError): selection_summary(samples)

    def test_retained_combined_fusion_integrity(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import fusion_replay
        source=repository_root()/'studies/model_generation'
        original=json.loads(gzip.decompress((source/'combined-fusion.json.gz').read_bytes()))
        for damage in (None,'ablation','sample','cache','dispatch','terminal','provenance'):
            record=copy.deepcopy(original)
            if damage=='ablation':
                record['timing']['samples']=[r for r in record['timing']['samples'] if r['comparison']!=2]
            elif damage=='sample': record['timing']['samples'].pop()
            elif damage=='cache': record['timing']['numerical'][0]['observations'][0]['exact']=False
            elif damage=='dispatch': record['captures'][1]['samples'].pop()
            elif damage=='terminal': record['terminal']['blocks'][0]['arms'][0]['turns'][0]['generated'][0]=0
            elif damage=='provenance': record['captures'][1]['provenance']['binary']['sha256']='0'*64
            with tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)
                raw=json.dumps(record).encode()
                packed=gzip.compress(raw,mtime=0)
                (directory/'combined-fusion.json.gz').write_bytes(packed)
                (directory/'combined-fusion.json').write_text(json.dumps(dict(sha256=hashlib.sha256(packed).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with redirect_stdout(StringIO()):
                    if damage is None:
                        fusion_replay(directory,True)
                        self.assertTrue(json.loads((directory/'combined-fusion-summary.json').read_text())['promote'])
                    else:
                        with self.assertRaises(ValueError): fusion_replay(directory,True)

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

    def test_retained_fusion_rejects_rehashed_missing_or_changed_evidence(self):
        from llm_mojo._repository import repository_root
        from llm_mojo.benchmarks.model_profile import fusion_replay
        path = repository_root()/'studies/model_generation/qkv-fusion.json.gz'
        if not path.exists():
            self.skipTest('fusion evidence has not yet been collected')
        original = json.loads(gzip.decompress(path.read_bytes()))
        for damage in ('sample','cache','dispatch','terminal'):
            record = copy.deepcopy(original)
            if damage=='sample': record['timing']['samples'].pop()
            elif damage=='cache': record['timing']['numerical'][0]['observations'][0]['prefix_exact']=False
            elif damage=='dispatch': record['captures'][1]['samples'].pop()
            else: record['terminal']['blocks'][0]['arms'][0]['turns'][0]['generated'][0]=0
            with tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp)
                raw=json.dumps(record).encode()
                packed=gzip.compress(raw,mtime=0)
                (d/'qkv-fusion.json.gz').write_bytes(packed)
                (d/'qkv-fusion.json').write_text(json.dumps(dict(sha256=hashlib.sha256(packed).hexdigest(),
                                                              uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
                with self.assertRaises(ValueError): fusion_replay(d)

    def test_fusion_contract_and_promotion_require_all_contexts_and_calibration(self):
        from llm_mojo.benchmarks.model_profile import fusion_summary
        self.assertEqual(len(contract.stages(True)),338)
        self.assertEqual(len(contract.command_stages(True)),342)
        samples = [dict(prefix=p,block=b,comparison=c,arm=a,sample=s,elapsed_ns=(80 if c==1 and a==1 else 100),marks=[])
                   for p in contract.PREFIXES for b in range(4) for c in range(2) for a in range(2) for s in range(10)]
        self.assertTrue(all(r['promote'] for r in fusion_summary(samples)))
        for row in samples:
            if row['comparison']==0 and row['arm']==1: row['elapsed_ns']=130
        self.assertFalse(any(r['promote'] for r in fusion_summary(samples)))
        with self.assertRaises(ValueError): fusion_summary(samples[:-1])

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

    def test_combined_contract_and_ablation_census(self):
        from llm_mojo.benchmarks.model_profile import combined_ablation
        self.assertEqual(len(contract.stages(True,True)),314)
        self.assertEqual(len(contract.command_stages(True,True)),318)
        stages=contract.stages(True,True)
        self.assertEqual(sum(name=='fused SiLU/multiply' for _,name in stages),24)
        self.assertFalse(any(name in ('SiLU','multiply') for _,name in stages))
        data=dict(implementation='qwen_model_combined',entrypoint='QwenModel.forward+greedy-combined',
                  **contract.specification(1024,True,True),profile_iterations=8,profile_warmup_iterations=10)
        contract.configuration(data)
        with self.assertRaises(ValueError):
            contract.configuration({**data,'dispatches_per_iteration':338})
        samples=[dict(prefix=p,block=b,comparison=c,arm=a,sample=s,
                      elapsed_ns=90 if c==2 and a==1 else 100,marks=[])
                 for p in contract.PREFIXES for b in range(4) for c in range(3)
                 for a in range(2) for s in range(10)]
        self.assertTrue(all(r['all_faster'] for r in combined_ablation(samples)))
        with self.assertRaises(ValueError): combined_ablation(samples[:-1])
        for row in samples:
            if row['prefix']==64 and row['block']==0 and row['comparison']==2 and row['arm']==1:
                row['elapsed_ns']=110
        self.assertFalse(combined_ablation(samples)[0]['all_faster'])

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
