"""Malformed model metadata must fail before any native model is launched."""
import copy
import math
import unittest
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


if __name__=='__main__': unittest.main()
