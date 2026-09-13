"""Native import closure invalidation and cached binary integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_mojo.runtime import build


class NativeBuildTests(unittest.TestCase):
    def test_tracks_transitive_imports_and_package_initializers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); src = root/'src/llm_mojo'; src.mkdir(parents=True)
            (root/'uv.lock').write_text('locked')
            (src/'__init__.mojo').write_text('')
            (src/'app.mojo').write_text('from llm_mojo.layer import compute\n')
            (src/'layer.mojo').write_text('from llm_mojo.kernel import compute\n')
            kernel = src/'kernel.mojo'; kernel.write_text('version 1')
            benchmark = src/'bench.mojo'; benchmark.write_text('benchmark 1')
            entry = 'src/llm_mojo/app.mojo'
            first = build.build_sources(entry, root)
            benchmark.write_text('benchmark 2')
            self.assertEqual(build.build_sources(entry, root), first)
            kernel.write_text('version 2')
            self.assertNotEqual(build.build_sources(entry, root), first)
            kernel.unlink()
            with self.assertRaisesRegex(ValueError, 'missing native'): build.build_sources(entry, root)

    def test_package_member_and_multiple_imports_are_tracked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); src = root/'src/llm_mojo'; src.mkdir(parents=True)
            (root/'uv.lock').write_text('locked')
            (src/'__init__.mojo').write_text('')
            (src/'app.mojo').write_text('from llm_mojo import (first, second)\nimport llm_mojo.third, llm_mojo.fourth as fourth\n')
            for name in ('first', 'second', 'third', 'fourth'):
                (src/(name+'.mojo')).write_text('')
            sources = build.build_sources('src/llm_mojo/app.mojo', root)
            for name in ('first', 'second', 'third', 'fourth'):
                self.assertIn('src/llm_mojo/'+name+'.mojo', sources)

    def test_chat_closure_excludes_benchmarks_and_generation_entrypoint(self):
        sources = build.build_sources('src/llm_mojo/cli/chat_cli.mojo')
        self.assertIn('src/llm_mojo/models/qwen2/model.mojo', sources)
        self.assertIn('src/llm_mojo/kernels/linear.mojo', sources)
        self.assertNotIn('src/llm_mojo/cli/generate_cli.mojo', sources)
        self.assertFalse(any('/benchmarks/' in p for p in sources))

    def test_tokenizer_benchmark_tracks_relocated_helpers_and_native_packages(self):
        from llm_mojo.benchmarks.tokenizer_contract import sources
        identity = sources()
        for name in ('runtime/artifacts.py', 'runtime/build.py', 'models/qwen2/tokenizer.mojo',
                     'models/qwen2/__init__.mojo', 'benchmarks/__init__.mojo'):
            self.assertIn('src/llm_mojo/' + name, identity)

    def test_changed_binary_is_rebuilt_even_with_matching_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); directory=root/'build/chat'; directory.mkdir(parents=True)
            binary=directory/'chat'; binary.write_bytes(b'changed')
            (directory/'binary.json').write_text(json.dumps(dict(sources={'a':'b'},binary_sha256='stale')))
            def compile(command, **kwargs): Path(command[-1]).write_bytes(b'fresh')
            with patch.object(build, 'repository_root', return_value=root), \
                 patch.object(build, 'build_sources', return_value={'a':'b'}), \
                 patch.object(build, 'environment_tool', return_value='mojo'), \
                 patch.object(build.subprocess, 'run', side_effect=compile) as run:
                self.assertEqual(build.ensure_binary('chat', 'entry.mojo'), binary)
                self.assertEqual(build.ensure_binary('chat', 'entry.mojo'), binary)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(binary.read_bytes(), b'fresh')
