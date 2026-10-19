"""Running plot without arguments renders exactly the paired-benchmark studies."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo._repository import repository_root
from llm_mojo.benchmarks.study import run_directories


class PlotSelectionTests(unittest.TestCase):
    def test_only_folders_with_a_run_record_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('flat', 'nested/data', 'archive'):
                (root / name).mkdir(parents=True)
            (root / 'flat/run.json').write_text('{}')
            (root / 'nested/data/run.json').write_text('{}')
            (root / 'archive/study.json.gz').write_bytes(b'')
            (root / 'notes.md').write_text('')
            self.assertEqual([p.name for p in run_directories(root)], ['flat', 'nested'])

    def test_repository_default_skips_archive_studies(self):
        names = [p.name for p in run_directories(repository_root() / 'studies')]
        self.assertEqual(names, ['attention_sublayer', 'decoder_layer', 'gqa_decode', 'gqa_prefill',
                                 'linear_decode', 'linear_prefill', 'mlp_sublayer', 'rms_norm', 'rope'])

    def test_resident_token_selection_calls_are_not_labelled_as_distinct_buffers(self):
        # Render only simulated samples in a temporary folder. This checks plot
        # semantics, not measured GPU performance or the native kernel.
        from llm_mojo.benchmarks.study import STUDIES, summarize, BLOCKS, REPETITIONS
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {'MPLCONFIGDIR': temporary}):
            try:
                from llm_mojo.benchmarks import plot
            except ModuleNotFoundError as error:
                if error.name == 'matplotlib':
                    self.skipTest('optional matplotlib is not installed')
                raise
            titles = {
                'token_selection': '24 calls · same resident buffers · one synchronization',
                'rms_norm': '24 buffers · one synchronization per sweep',
            }
            for operation, expected in titles.items():
                with self.subTest(operation=operation):
                    directory = Path(temporary) / operation
                    directory.mkdir()
                    spec = STUDIES[operation]
                    samples = [dict(block=block, rows=rows, layers=layers, candidate=candidate,
                                    arm=arm, variant=spec['control'] if arm == 'control' else candidate,
                                    repetition=repetition, us=100.0)
                               for block in range(1, BLOCKS + 1) for rows in spec['rows']
                               for layers in spec.get('layers', (1, 24)) for candidate in spec['candidates']
                               for arm in ('control', 'candidate') for repetition in range(REPETITIONS)]
                    record = dict(study=operation, specification=spec,
                                  runtime={'device': 'Apple Test GPU'}, repository={'commit': 'a' * 40})
                    closed = []
                    real_close = plot.plt.close
                    def capture_and_close(figure=None):
                        if hasattr(figure, 'axes'):
                            closed.append(figure)
                        real_close(figure)
                    with patch.object(plot, 'load_run', return_value=(record, samples, summarize(samples, spec))), \
                            patch.object(plot.plt, 'close', side_effect=capture_and_close):
                        plot.render(directory)
                    self.assertEqual(len(closed), 1)
                    self.assertEqual(closed[0].axes[0].get_title(), 'Hot call · one synchronization')
                    self.assertEqual(closed[0].axes[1].get_title(), expected)
                    self.assertTrue((directory / 'latency.png').is_file())


if __name__ == '__main__':
    unittest.main()
