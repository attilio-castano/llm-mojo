"""Verify the fixed Qwen prepared-weight boundary before native execution.

Preparation and verification are development/initialization work. Native model
execution receives only checked binary tensors and runs without Python interop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tokenizer_assets import REVISION, asset_directory

CHECKPOINT_SHA = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'


def tensor_shapes():
    shapes = {'embedding': (151936, 896), 'final_norm': (896,),
              'cosine': (4096, 64), 'sine': (4096, 64)}
    for layer in range(24):
        for name, shape in {
            'attention_norm': (896,), 'mlp_norm': (896,), 'qkv': (1152,896),
            'bias': (1152,), 'wo': (896,896), 'gate': (4864,896),
            'up': (4864,896), 'down': (896,4864),
        }.items():
            shapes[f'layer_{layer}_{name}'] = shape
    return shapes


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_manifest(manifest):
    if (manifest.get('format') != 'qwen-model-prepared-v1'
            or manifest.get('model_revision') != REVISION
            or manifest.get('checkpoint_sha256') != CHECKPOINT_SHA):
        raise ValueError('prepared model identity mismatch')
    expected = tensor_shapes()
    actual = manifest.get('tensors', {})
    if set(actual) != set(expected):
        raise ValueError('missing or extra prepared model tensors')
    for name, shape in expected.items():
        record = actual[name]
        size = 2
        for dim in shape:
            size *= dim
        digest = record.get('sha256')
        if (record.get('shape') != list(shape) or record.get('dtype') != 'BF16'
                or record.get('bytes') != size
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid prepared tensor geometry/identity: '+name)
    return expected


def verify_prepared(directory=None):
    directory = Path(directory) if directory else asset_directory()/'model-prepared-v1'
    manifest = json.loads((directory/'manifest.json').read_text())
    expected = validate_manifest(manifest)
    for name in expected:
        path = directory/(name+'.bin')
        record = manifest['tensors'][name]
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('prepared tensor checksum/extent mismatch: '+name)
    return directory, manifest
