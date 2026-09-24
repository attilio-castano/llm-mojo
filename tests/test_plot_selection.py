"""Running plot without arguments renders exactly the paired-benchmark studies."""
from pathlib import Path
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
