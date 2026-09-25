"""Public configuration, input boundaries and native launch handoff."""
import json
from contextlib import chdir
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from typer.testing import CliRunner
from llm_mojo.cli.app import app
from llm_mojo.configuration import resolve_run, resolve_bench
from llm_mojo.runtime import launch


class ConfigurationTests(unittest.TestCase):
    def test_preset_then_explicit_options_without_leaking_between_calls(self):
        a = resolve_run('chat', preset='short', max_new_tokens=17)
        self.assertEqual((a.workload.max_new_tokens, a.workload.chunk_rows), (17, 64))
        self.assertEqual(resolve_run('chat').workload.max_new_tokens, 256)
        self.assertEqual(resolve_run('generate', preset='whole-prompt').workload.chunk_rows, 0)
        with self.assertRaises(ValueError): resolve_run('chat', preset='whole-prompt')

    def test_unsupported_selections_and_conflicting_inputs(self):
        for options in ({'model': 'llama'}, {'mode': 'auto'}, {'preset': 'unknown'},
                        {'prompt': 'hello', 'prompt_file': 'file'}, {'system_file': 'file'},
                        {'chunk_rows': 4097}, {'max_new_tokens': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_run('generate', **options)

    def test_generate_accepts_reference_modes_and_chat_stays_fast(self):
        self.assertEqual(resolve_run('chat').mode.name, 'fast')
        self.assertEqual(resolve_run('generate').mode.name, 'fast')
        for mode in ('fast', 'baseline', 'consistent'):
            self.assertEqual(resolve_run('generate', mode=mode).mode.name, mode)
        for mode in ('baseline', 'consistent', 'auto'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'supported: fast$'):
                resolve_run('chat', mode=mode)
        with self.assertRaisesRegex(ValueError, 'supported: fast, baseline, consistent$'):
            resolve_run('generate', mode='projection-0')

    def test_inspection_is_read_only_and_keeps_literal_prompt(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as directory, chdir(directory), \
             patch.object(launch, 'verify_prepared') as verify, \
             patch.object(launch, 'ensure_binary') as build:
            result = runner.invoke(app, ['generate', '--preset', 'short', '--prompt',
                                        'Hello, [world] = ${not_a_resolver}', '--show-config'])
            self.assertEqual(result.exit_code, 0, result.output)
            cfg = json.loads(result.output)
            self.assertEqual(cfg['workload']['prompt'], 'Hello, [world] = ${not_a_resolver}')
            self.assertEqual(list(Path(directory).iterdir()), [])
            verify.assert_not_called(); build.assert_not_called()

    def test_cli_rejects_inputs_before_launch(self):
        runner = CliRunner()
        with patch.object(launch, 'launch_generate') as generate, patch.object(launch, 'launch_chat') as chat:
            for arguments in (['generate'], ['generate', '--prompt', 'x', '--mode', 'auto'],
                              ['generate', '--prompt', 'x', '--prompt-file', 'x'],
                              ['chat', '--mode', 'consistent']):
                result = runner.invoke(app, arguments)
                self.assertNotEqual(result.exit_code, 0)
            generate.assert_not_called(); chat.assert_not_called()

    def test_benchmark_presets_preserve_existing_grids(self):
        from llm_mojo.benchmarks.study import STUDIES
        original = json.dumps(STUDIES, sort_keys=True)
        cfg = resolve_bench('attention', studies=['rms_norm'])
        self.assertEqual(cfg.studies, ['rms_norm'])
        self.assertEqual(resolve_bench('attention').studies, ['gqa_decode', 'gqa_prefill'])
        self.assertEqual(json.dumps(STUDIES, sort_keys=True), original)
        for studies in (['bogus'], ['rms_norm', 'rms_norm'], []):
            with self.assertRaises(ValueError): resolve_bench(studies=studies)

    def test_benchmark_passes_resolved_selection_to_existing_runner(self):
        with patch('llm_mojo.benchmarks.run.run') as run:
            result = CliRunner().invoke(app, ['bench', 'run', '--study', 'rms_norm',
                                             '--build-dir', '/build', '--output', '/output'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.args, (Path('/build'), Path('/output'), ['rms_norm']))
        self.assertEqual(run.call_args.kwargs['resolved_configuration']['studies'], ['rms_norm'])

    def test_replay_only_studies_are_listed_but_never_run(self):
        from llm_mojo.benchmarks.study import REPLAY_ONLY
        from llm_mojo.benchmarks.run import run
        self.assertTrue(REPLAY_ONLY)
        self.assertTrue(all(name.startswith(('decoder_selection_', 'decoder_policies_round2_'))
                            for name in REPLAY_ONLY))
        name = sorted(REPLAY_ONLY)[0]
        with self.assertRaises(ValueError): resolve_bench(studies=[name])
        with self.assertRaisesRegex(ValueError, 'edb610a'): run(Path('/build'), Path('/output'), [name])
        listed = json.loads(CliRunner().invoke(app, ['bench', 'list']).output)
        self.assertEqual(set(listed['replay_only']), REPLAY_ONLY)
        self.assertFalse(set(listed['studies']) & REPLAY_ONLY)
        self.assertIn('decoder_selection_calibration', listed['studies'])

    def test_setup_passes_options_and_returns_readiness(self):
        with patch('llm_mojo.models.qwen2.assets.setup', return_value=1) as setup:
            result = CliRunner().invoke(app, ['setup', '--check', '--offline', '--no-build', '--store', '/s',
                                             '--import-from', '/a', '--import-from', '/b'])
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertEqual(setup.call_args.args, (Path('/s'),))
        options = setup.call_args.kwargs
        self.assertEqual((options['offline'], options['check'], options['build']), (True, True, False))
        self.assertEqual(list(options['import_from']), [Path('/a'), Path('/b')])
        with patch('llm_mojo.models.qwen2.assets.setup', return_value=0) as setup:
            self.assertEqual(CliRunner().invoke(app, ['setup']).exit_code, 0)
        self.assertEqual(setup.call_args.kwargs['import_from'], ())
        self.assertFalse(setup.call_args.kwargs['offline'])

    def test_prepare_is_explicit_and_local_by_default(self):
        with patch('llm_mojo.models.qwen2.assets.prepare') as prepare:
            result = CliRunner().invoke(app, ['models', 'prepare', 'qwen2.5-0.5b-instruct'])
        self.assertEqual(result.exit_code, 0, result.output)
        prepare.assert_called_once_with(None, download=False)


    def test_validate_passes_fixture_options_and_rejects_conflicts(self):
        with patch('llm_mojo.validation.suite.main') as main:
            for arguments, expected in ((['validate'], []),
                                        (['validate', '--prepare-only', '--regenerate-fixtures'],
                                         ['--prepare-only', '--regenerate-fixtures']),
                                        (['validate', '--no-fixture-cache'], ['--no-fixture-cache'])):
                result = CliRunner().invoke(app, arguments)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(main.call_args.args, (expected,))
            result = CliRunner().invoke(app, ['validate', '--regenerate-fixtures', '--no-fixture-cache'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(main.call_count, 3)

    def test_fixture_detach_reports_its_result(self):
        with patch('llm_mojo.validation.fixtures.detach', return_value='mlp: now a writable copy') as detach:
            result = CliRunner().invoke(app, ['fixtures', 'detach', 'mlp'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('mlp: now a writable copy', result.output)
        detach.assert_called_once_with('mlp')
        self.assertIsInstance(CliRunner().invoke(app, ['fixtures', 'detach', 'decoder']).exception, ValueError)

    def test_fixture_list_and_prune_report_and_pass_the_confirmation(self):
        with patch('llm_mojo.validation.fixtures.report', return_value='Shared oracle fixtures in /s'):
            result = CliRunner().invoke(app, ['fixtures', 'list'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Shared oracle fixtures in /s', result.output)
        with patch('llm_mojo.validation.fixtures.prune', return_value='Nothing to prune.') as prune:
            for arguments in (['fixtures', 'prune'], ['fixtures', 'prune', '--yes']):
                self.assertEqual(CliRunner().invoke(app, arguments).exit_code, 0)
        self.assertEqual([call.kwargs for call in prune.call_args_list], [{'yes': False}, {'yes': True}])


class GenerationTests(unittest.TestCase):
    def test_prompt_file_snapshot_and_report_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); prompt = root/'prompt.txt'; prompt.write_text('Hello é')
            report = root/'events.tsv'
            cfg = resolve_run('generate', prompt_file=prompt, report=report, max_new_tokens=12)
            def execute(command, **kwargs):
                self.assertEqual(Path(command[3]).read_text(), 'Hello é')
                self.assertEqual(command[4:7], ['12', '256', 'fast'])
                self.assertEqual(command[-1], str(report))
            with patch.object(launch, 'verify_prepared', return_value=(Path('/weights'), {})), \
                 patch.object(launch, 'ensure_prepared', return_value=Path('/tables')), \
                 patch.object(launch, 'ensure_binary', return_value=Path('/native')), \
                 patch.object(launch.subprocess, 'run', side_effect=execute):
                launch.launch_generate(cfg)
            saved = json.loads((root/'events.tsv.config.json').read_text())
            self.assertEqual(saved['configuration']['workload']['max_new_tokens'], 12)
            with self.assertRaises(ValueError): launch.report_paths(str(report))

    def test_empty_prompt_rejected_without_assets(self):
        with patch.object(launch, 'verify_prepared') as verify:
            with self.assertRaises(ValueError): launch.launch_generate(resolve_run('generate', prompt=''))
            verify.assert_not_called()


if __name__ == '__main__': unittest.main()
