import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo.benchmarks import run as runner
from llm_mojo.benchmarks.study import load_run, load_profile, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_retained_studies_are_complete(self):
        count = 0
        for directory in (ROOT / 'studies').glob('*/'):
            _, samples, _ = load_run(directory)
            count += len(samples)
        self.assertEqual(count, 10560)
        profile = load_profile(ROOT / 'studies/gqa_decode')
        self.assertEqual(sum(row['count'] for row in profile), 3000)

    def test_profile_corruption_and_duplicate_dispatch_rejected(self):
        source = ROOT / 'studies/gqa_decode'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = json.loads((source / 'profiles.json').read_text())
            path = directory / 'profile_samples.csv.gz'
            path.write_bytes((source / path.name).read_bytes())
            write_json(directory / 'profiles.json', record)
            self.assertEqual(len(load_profile(directory)), 6)
            raw = gzip.decompress(path.read_bytes()).decode()
            path.write_bytes(gzip.compress((raw + raw.splitlines()[1] + '\n').encode()))
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_profile(directory)
            record['samples_sha256'] = sha(path)
            write_json(directory / 'profiles.json', record)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_profile(directory)

    def test_runner_rejects_foreign_source_environment_and_binary_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            binary = directory / 'operations'
            binary.write_bytes(b'fixture binary')
            repo = dict(commit='a' * 40, branch='codex/test', dirty=False)
            sources, environment = {'source': 'hash'}, {'hardware': 'test'}
            base = dict(repository=repo, sources=sources, environment=environment,
                        binaries={'operations': sha(binary)})
            bad = []
            for key, value in [('repository', {**repo, 'commit': 'b' * 40}),
                               ('repository', {**repo, 'dirty': True}),
                               ('sources', {}), ('environment', {}),
                               ('binaries', {'operations': 'wrong'})]:
                item = copy.deepcopy(base); item[key] = value; bad.append(item)
            for record in bad:
                write_json(directory / 'build.json', record)
                with patch.object(runner, 'repository_state', return_value=repo), \
                     patch.object(runner, 'source_hashes', return_value=sources), \
                     patch.object(runner, 'stable_environment', return_value=environment), \
                     patch.object(runner, 'checked_conditions', side_effect=AssertionError('reached GPU phase')), \
                     self.assertRaises(RuntimeError):
                    runner.run(directory, directory / 'output', ['gqa_decode'])
