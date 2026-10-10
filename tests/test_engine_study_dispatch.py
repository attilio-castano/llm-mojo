"""Study options must select their actual collector before any work starts."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from llm_mojo.benchmarks import engine_async, engine_budget, model_profile as profile


class EngineStudyDispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.output = self.root/'output-must-remain-absent'
        self.build = self.root/'missing-build'
        self.trace = self.root/'missing-trace.json'
        self.prepared = self.root/'missing-prepared'
        self.qualification = self.root/'missing-qualification.json'
        self.policy = self.root/'missing-policy.json'

    def invoke(self, command, options):
        """Mock dispatch and all work boundaries; leave every input absent."""
        argv = ['model_profile', command, '--output', str(self.output),
                '--build', str(self.build), '--trace', str(self.trace),
                '--prepared', str(self.prepared), *options]
        calls = {}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, 'argv', argv))
            stack.enter_context(mock.patch.dict(os.environ))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for name in ('engine_build', 'engine_collect', 'engine_replay',
                         'engine_specification', 'assets', 'verify_build', 'execute',
                         'ensure_record_location', 'write'):
                calls[name] = stack.enter_context(mock.patch.object(profile, name))
            calls['engine_replay'].return_value = {}
            for name in ('fast_collect', 'scheduling_collect'):
                calls[name] = stack.enter_context(mock.patch.object(engine_budget, name))
            calls['async_collect'] = stack.enter_context(mock.patch.object(engine_async, 'collect'))
            calls['async_qualify'] = stack.enter_context(mock.patch.object(engine_async, 'qualify'))
            try:
                profile.main()
            except SystemExit as error:
                code = error.code
            else:
                code = 0
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        return code, calls

    def assert_rejected(self, command, options):
        code, calls = self.invoke(command, options)
        self.assertEqual(code, 2)
        for name, call in calls.items():
            with self.subTest(boundary=name):
                call.assert_not_called()

    def test_collection_options_reject_build_replay_and_specification_before_work(self):
        options = [('--token-budget', '32'),
                   ('--qualification', str(self.qualification)),
                   ('--budget-stage', 'calibration'),
                   ('--policy', str(self.policy)),
                   ('--admission-policy', 'reserved')]
        for command in ('engine-build', 'engine-qualify', 'engine-replay', 'engine-specification'):
            for option in options:
                with self.subTest(command=command, option=option[0]):
                    self.assert_rejected(command, list(option))

    def test_fast_only_options_reject_legacy_and_budget_collection(self):
        for prefix in ([], ['--admission-pair'], ['--admission-range'],
                       ['--budget-study', '--budget-stage', 'calibration',
                        '--admission-policy', 'reserved']):
            for option in (['--token-budget', '32'],
                           ['--qualification', str(self.qualification)]):
                with self.subTest(prefix=prefix, option=option[0]):
                    self.assert_rejected('engine-collect', prefix+option)

    def test_budget_only_options_reject_legacy_and_fast_collection(self):
        for prefix in ([], ['--admission-pair'], ['--admission-range'],
                       ['--fast-study', '--token-budget', '64',
                        '--qualification', str(self.qualification),
                        '--admission-policy', 'reserved']):
            for option in (['--budget-stage', 'evaluation'],
                           ['--policy', str(self.policy)]):
                with self.subTest(prefix=prefix, option=option[0]):
                    self.assert_rejected('engine-collect', prefix+option)

    def test_admission_policy_rejects_collectors_that_do_not_consume_it(self):
        for prefix in ([], ['--admission-pair'], ['--admission-range']):
            with self.subTest(prefix=prefix):
                self.assert_rejected('engine-collect', prefix+['--admission-policy', 'reserved'])

    def test_new_study_selectors_reject_non_build_collection_commands(self):
        for command in ('engine-qualify', 'engine-replay', 'engine-specification'):
            for selector in ('--budget-study', '--fast-study', '--async-study'):
                with self.subTest(command=command, selector=selector):
                    self.assert_rejected(command, [selector])

    def test_async_checkpoint_qualification_dispatches_before_collection(self):
        code, calls = self.invoke('engine-qualify', [])
        self.assertEqual(code, 0)
        calls['async_qualify'].assert_called_once_with(self.build, self.output)
        for name, call in calls.items():
            if name != 'async_qualify':
                call.assert_not_called()

    def test_explicit_fast_and_budget_collection_dispatch_preserves_parameters(self):
        code, calls = self.invoke('engine-collect',
            ['--fast-study', '--token-budget', '64', '--admission-policy', 'reserved',
             '--qualification', str(self.qualification), '--blocks', '40'])
        self.assertEqual(code, 0)
        calls['fast_collect'].assert_called_once_with(self.build, self.trace, self.output,
            self.qualification, 64, 'reserved', 40, 8, 10)
        calls['scheduling_collect'].assert_not_called()
        calls['engine_collect'].assert_not_called()
        for stage, policy_options, expected_policy in (
                ('calibration', [], None),
                ('evaluation', ['--policy', str(self.policy)], self.policy)):
            with self.subTest(stage=stage):
                code, calls = self.invoke('engine-collect',
                    ['--budget-study', '--budget-stage', stage,
                     '--admission-policy', 'reserved', '--blocks', '40', *policy_options])
                self.assertEqual(code, 0)
                calls['scheduling_collect'].assert_called_once_with(self.build, self.trace,
                    self.output, stage, 'reserved', expected_policy, 40, 8, 'greedy', 10)
                calls['fast_collect'].assert_not_called()
                calls['engine_collect'].assert_not_called()

    def test_legacy_commands_and_new_build_selectors_still_dispatch(self):
        for options, pair, operating_range in (([], False, False),
                (['--admission-pair'], True, False), (['--admission-range'], False, True)):
            with self.subTest(options=options):
                code, calls = self.invoke('engine-collect', options)
                self.assertEqual(code, 0)
                calls['engine_collect'].assert_called_once_with(self.build, self.trace,
                    self.output, 128, 8, 'greedy', 10, pair, operating_range)
        for options, budget, fast in (([], False, False),
                (['--budget-study'], True, False), (['--fast-study'], False, True)):
            with self.subTest(options=options):
                code, calls = self.invoke('engine-build', options)
                self.assertEqual(code, 0)
                calls['engine_build'].assert_called_once_with(self.output, self.prepared,
                    False, False, budget, fast)
        code, calls = self.invoke('engine-replay', [])
        self.assertEqual(code, 0)
        calls['engine_replay'].assert_called_once_with(self.output)
        code, calls = self.invoke('engine-specification', [])
        self.assertEqual(code, 0)
        calls['engine_specification'].assert_called_once_with(self.output, 7, None)

    def test_async_build_and_collection_dispatch_consumes_explicit_gate(self):
        code, calls = self.invoke('engine-build', ['--async-study'])
        self.assertEqual(code, 0)
        calls['engine_build'].assert_called_once_with(self.output, self.prepared, async_study=True)
        code, calls = self.invoke('engine-collect', ['--async-study', '--qualification', str(self.qualification),
                                                    '--admission-policy', 'reserved', '--blocks', '40'])
        self.assertEqual(code, 0)
        calls['async_collect'].assert_called_once_with(self.build, self.trace, self.output,
                                                       self.qualification, 40, 8, 10)
        calls['fast_collect'].assert_not_called()
        calls['scheduling_collect'].assert_not_called()
        calls['engine_collect'].assert_not_called()

    def test_async_options_reject_unsupported_configuration_before_work(self):
        valid = ['--async-study', '--qualification', str(self.qualification), '--admission-policy', 'reserved']
        for options in (['--async-study'], ['--async-study', '--qualification', str(self.qualification)],
                ['--async-study', '--qualification', str(self.qualification), '--admission-policy', 'incremental'],
                valid+['--token-budget', '32'], valid+['--policy', str(self.policy)],
                valid+['--budget-stage', 'calibration'], valid+['--engine-mode', 'scripted']):
            with self.subTest(options=options):
                self.assert_rejected('engine-collect', options)


if __name__ == '__main__':
    unittest.main()
