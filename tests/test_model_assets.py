"""Malformed model metadata must fail before any native model is launched."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_mojo import model_assets as assets
from llm_mojo.model_assets import CHECKPOINT_SHA, REVISION, tensor_shapes, validate_manifest


class ModelAssetTests(unittest.TestCase):
    def manifest(self):
        return dict(format='qwen-model-prepared-v1', model_revision=REVISION,
            checkpoint_sha256=CHECKPOINT_SHA, tensors={name: dict(shape=list(shape),
                dtype='BF16', bytes=2*math.prod(shape), sha256='a'*64)
                for name,shape in tensor_shapes().items()})

    def test_complete_geometry(self):
        result = validate_manifest(self.manifest())
        self.assertEqual(len(result), 196)
        self.assertEqual(result['layer_23_down'], (896,4864))
        # The head shares the embedding allocation, not a duplicated tensor.
        self.assertNotIn('lm_head',result)

    def test_wrong_checkpoint_missing_layer_and_transposed_weight(self):
        original=self.manifest()
        cases=[]
        bad=copy.deepcopy(original); bad['checkpoint_sha256']='0'*64; cases.append(bad)
        bad=copy.deepcopy(original); del bad['tensors']['layer_23_qkv']; cases.append(bad)
        bad=copy.deepcopy(original); bad['tensors']['layer_0_down']['shape']=[4864,896]; cases.append(bad)
        bad=copy.deepcopy(original); bad['tensors']['embedding']['dtype']='F16'; cases.append(bad)
        bad=copy.deepcopy(original); bad['tensors']['cosine']['bytes']-=2; cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad.keys()), self.assertRaises(ValueError):
                validate_manifest(bad)


class GenerationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / 'prepared model'
        self.prepared.mkdir()
        self.prompt = self.root / 'prompt text.txt'
        self.prompt.write_text('Hello')
        self.payloads = {'embedding': b'\x80\x3f\x00\x40', 'final_norm': b'\x00\x3f\x80\xbf'}
        self.manifest = dict(format='qwen-model-prepared-v1', model_revision=REVISION,
            checkpoint_sha256=CHECKPOINT_SHA, tensors={name: dict(shape=[2], dtype='BF16',
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                for name, data in self.payloads.items()})
        # Exercise real manifest and byte verification with tiny tensors;
        # ModelAssetTests separately checks the complete 196-tensor geometry.
        shapes = patch.object(assets, 'tensor_shapes', return_value={name: (2,) for name in self.payloads})
        shapes.start()
        self.addCleanup(shapes.stop)
        self.restore()

    def restore(self):
        for name, data in self.payloads.items():
            (self.prepared / (name + '.bin')).write_bytes(data)
        (self.prepared / 'manifest.json').write_text(json.dumps(self.manifest))

    def launch(self):
        assets.main(['--prepared', str(self.prepared), '--prompt', str(self.prompt),
                     '--max-new-tokens', '8', '--chunk-rows', '16', '--policy', 'consistent'])

    def test_default_directory_uses_documented_output_and_verifies_bytes(self):
        expected = self.root / 'build/model-prepared-v1'
        expected.parent.mkdir()
        self.prepared.rename(expected)
        with patch.object(assets, 'repository_root', return_value=self.root):
            directory, manifest = assets.verify_prepared()
            self.assertEqual(directory, expected)
            self.assertEqual(manifest, self.manifest)
            (expected / 'embedding.bin').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum/extent'):
                assets.verify_prepared()

    def test_verified_inputs_reach_native_driver_as_separate_arguments(self):
        tables = self.root / 'tokenizer tables.bin'
        with patch.object(assets, 'ensure_prepared', return_value=tables) as tokenizer, \
                patch.object(assets, 'environment_tool', return_value='/locked/bin/mojo'), \
                patch.object(assets.subprocess, 'run') as run:
            self.launch()
        tokenizer.assert_called_once_with(download=False)
        root = assets.repository_root()
        run.assert_called_once_with([
            '/locked/bin/mojo', 'run', '-I', 'src', str(root / 'src/llm_mojo/generate_cli.mojo'),
            str(self.prepared.resolve()), str(tables), str(self.prompt.resolve()), '8', '16', 'consistent',
        ], cwd=root, check=True)

    def test_corrupt_swapped_missing_and_wrong_checkpoint_fail_before_launch(self):
        for damage in ('corrupt', 'swap', 'missing', 'identity', 'manifest'):
            with self.subTest(damage=damage):
                self.restore()
                weight = self.prepared / 'embedding.bin'
                if damage == 'corrupt':
                    weight.write_bytes(b'\x81' + self.payloads['embedding'][1:])
                elif damage == 'swap':
                    weight.write_bytes(self.payloads['final_norm'])
                elif damage == 'missing':
                    weight.unlink()
                elif damage == 'identity':
                    bad = dict(self.manifest, checkpoint_sha256='0' * 64)
                    (self.prepared / 'manifest.json').write_text(json.dumps(bad))
                else:
                    (self.prepared / 'manifest.json').unlink()
                with patch.object(assets, 'ensure_prepared') as tokenizer, \
                        patch.object(assets.subprocess, 'run') as run:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.launch()
                    tokenizer.assert_not_called()
                    run.assert_not_called()

    def test_tokenizer_verification_failure_prevents_native_launch(self):
        with patch.object(assets, 'ensure_prepared', side_effect=ValueError('damaged tokenizer')), \
                patch.object(assets.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'damaged tokenizer'):
                self.launch()
            run.assert_not_called()


if __name__=='__main__': unittest.main()
