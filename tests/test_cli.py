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
        for options in ({'model': 'llama'}, {'mode': 'consistent'}, {'preset': 'unknown'},
                        {'prompt': 'hello', 'prompt_file': 'file'}, {'system_file': 'file'},
                        {'chunk_rows': 4097}, {'max_new_tokens': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_run('generate', **options)

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
        with patch.object(launch, 'launch_generate') as generate:
            for arguments in (['generate'], ['generate', '--mode', 'consistent'],
                              ['generate', '--prompt', 'x', '--prompt-file', 'x']):
                result = runner.invoke(app, arguments)
                self.assertNotEqual(result.exit_code, 0)
            generate.assert_not_called()

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

    def test_prepare_is_explicit_and_local_by_default(self):
        with patch('llm_mojo.models.qwen2.assets.prepare') as prepare:
            result = CliRunner().invoke(app, ['models', 'prepare', 'qwen2.5-0.5b-instruct'])
        self.assertEqual(result.exit_code, 0, result.output)
        prepare.assert_called_once_with(None, download=False)


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
