"""Build/check MLP development references; holdout outputs are never executed here.

Uses local assets only. --characterize writes ignored evidence; --freeze creates
new anchors exclusively after budgets are selected. Default verifies anchors.
"""
import os
# Set before NumPy/BLAS import. Torch is separately constrained below.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from contract import DEVELOPMENT, MUTATIONS, STAGES, WEIGHTS, BUDGETS, specification
from numerics import differences, fp64_stages, bf16_bits, round_bf16
from reference import UpstreamMLP, inputs, mutate, same_bits, provenance, activation_probe, projection_tail_probe, array

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO / 'build/oracle_data/mlp'
FROZEN = HERE / 'checksums.json'
EVIDENCE = HERE / 'development.json.gz'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_identity():
    files = list(HERE.glob('*.py')) + [HERE.parent/'generate.py',
                                     HERE.parent/'attention_sublayer/upstream.py',
                                     HERE.parent/'attention_sublayer/checkpoint_checksums.json']
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(files)}


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write_json(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save_arrays(directory, data):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, value in data.items():
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite fixture: {name}')
        bf16_bits(value)
        path = directory / (name+'.npy')
        np.save(path, value, allow_pickle=False)
        result[str(path.relative_to(ROOT))] = dict(shape=list(value.shape), storage_dtype='float32',
                                                 logical_dtype='bfloat16', sha256=sha(path))
    return result


def validate_anchors(record, expected, root):
    if record != expected:
        raise RuntimeError('MLP reference differs from frozen contract or evidence; anchors are not updated')
    for name, spec in expected['arrays'].items():
        path = root / name
        if sha(path) != spec['sha256']:
            raise RuntimeError(f'MLP fixture hash mismatch: {name}')


def frozen_record(include_checkpoint):
    anchor = json.loads(FROZEN.read_text())
    payload = EVIDENCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != anchor['evidence_sha256']:
        raise RuntimeError('compressed MLP evidence hash mismatch')
    raw = gzip.decompress(payload)
    if hashlib.sha256(raw).hexdigest() != anchor['uncompressed_sha256']:
        raise RuntimeError('uncompressed MLP evidence hash mismatch')
    expected = json.loads(raw)
    for name in ('specification', 'upstream', 'source_sha256'):
        if expected[name] != anchor[name]:
            raise RuntimeError(f'MLP summary/evidence mismatch: {name}')
    if not include_checkpoint:
        expected['cases'] = {k: v for k, v in expected['cases'].items() if not k.startswith('checkpoint_')}
        expected['arrays'] = {k: v for k, v in expected['arrays'].items() if not k.startswith('checkpoint_')}
        expected['checkpoint'] = dict(status='pending', reason='no local checkpoint directory supplied')
    return expected


def check_case(case_id, data, arrays, records):
    rows, h = data['X'].shape
    runner = UpstreamMLP(data)
    full = runner.run(data['X'])
    plain = runner.run(data['X'], observe=False)
    if not same_bits(full['Y'], plain['Y']):
        raise RuntimeError('instrumentation changed upstream result')
    local = fp64_stages(data, full)
    composed = fp64_stages(data)
    record = dict(rows=rows, hidden=h, intermediate=data['gate'].shape[0],
                  observation_exact=True,
                  local_fp64={k: differences(full[k], local[k]) for k in STAGES},
                  composed_fp64={k: differences(full[k], composed[k]) for k in STAGES})
    chunk_sizes = [1]*rows if rows <= 17 else [rows-18, 17, 1]
    chunked = {k: [] for k in STAGES}
    offset = 0
    for size in chunk_sizes:
        part = runner.run(data['X'][offset:offset+size])
        for k in STAGES:
            chunked[k].append(part[k])
        offset += size
    record['full_chunked'] = {k: differences(np.concatenate(chunked[k]), full[k]) for k in STAGES}
    record['chunk_sizes'] = chunk_sizes
    if BUDGETS is not None:
        checks = dict(
            local={k: differences(full[k], local[k], BUDGETS['operation'][k]) for k in STAGES},
            composition={k: differences(full[k], composed[k], BUDGETS['composition'][k]) for k in ('D', 'Y')},
            chunks={k: differences(np.concatenate(chunked[k]), full[k], BUDGETS['composition'][k]) for k in ('D', 'Y')})
        record['reference_budget_checks'] = checks
        if any(m['failed'] for group in checks.values() for m in group.values()):
            raise RuntimeError(f'declared reference budget failed: {case_id}')
    arrays.update(save_arrays(ROOT / case_id, {**data, **full}))
    records[case_id] = record
    print('MLP reference', case_id, 'R/H/I', rows, h, data['gate'].shape[0],
          'D/Y max scaled vs FP64', record['composed_fp64']['D']['max_scaled'],
          record['composed_fp64']['Y']['max_scaled'], flush=True)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_cases(directory):
    """Verify the historical prefix and all companion assets without mutating them."""
    authority = json.loads((HERE.parent / 'attention_sublayer/checkpoint_checksums.json').read_text())
    if directory is None:
        return [], dict(status='pending', reason='no local checkpoint directory supplied')
    directory = directory.resolve()
    for name, digest in authority['asset_sha256'].items():
        if name != 'model.safetensors' and sha(directory/name) != digest:
            raise RuntimeError(f'checkpoint asset mismatch: {name}')
    source = dict(mode='attention-prefix', full_file_sha256_verified=False,
                  downloaded_prefix_bytes=302126368,
                  downloaded_prefix_sha256='0d3c86fcaa9573dbac31055974018e4d1a94a07124e5d124747feed78b51f6fa')
    path = directory / 'model.attention-prefix.bin'
    if path.stat().st_size != source['downloaded_prefix_bytes'] or sha(path) != source['downloaded_prefix_sha256']:
        raise RuntimeError('checkpoint prefix identity mismatch')
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        start = n+8
        def load(name):
            spec = header[name]
            a, b = spec['data_offsets']
            if spec['dtype'] != 'BF16' or not 0 <= a <= b <= path.stat().st_size-start:
                raise RuntimeError(f'incomplete BF16 checkpoint tensor: {name}')
            f.seek(start+a)
            payload = bytearray(f.read(b-a))
            if len(payload) != b-a:
                raise RuntimeError('truncated checkpoint tensor')
            return torch.frombuffer(payload, dtype=torch.bfloat16).reshape(spec['shape']).clone()
        p = 'model.layers.0.'
        names = [p+'post_attention_layernorm.weight'] + [p+'mlp.'+k+'_proj.weight' for k in ('gate', 'up', 'down')]
        mlp = {k: array(load(name)) for k, name in zip(WEIGHTS, names)}
        emb = load('model.embed_tokens.weight')
        attention_inputs = dict(norm_weight=array(load(p+'input_layernorm.weight')),
                                output_weight=array(load(p+'self_attn.o_proj.weight')))
        attention_inputs['weight'] = array(torch.cat([load(p+f'self_attn.{k}_proj.weight') for k in ('q', 'k', 'v')]))
        attention_inputs['bias'] = array(torch.cat([load(p+f'self_attn.{k}_proj.bias') for k in ('q', 'k', 'v')]))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    attention = module_at('mlp_attention_reference', HERE.parent/'attention_sublayer/upstream.py')
    def tokens(prompt):
        return tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'},
                                              {'role': 'user', 'content': prompt}],
                                             tokenize=True, add_generation_prompt=True)[:4096]
    cases, prompts = [], []
    for i, old in enumerate(authority['prompts']):
        ids = tokens(old['prompt'])
        digest = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
        if digest != old['token_ids_sha256']:
            raise RuntimeError('checkpoint development token IDs changed')
        data = dict(attention_inputs, input=array(torch.nn.functional.embedding(torch.tensor(ids), emb)))
        x = attention.UpstreamAttention(data, 14, 2, 64, fp32_attention=True).run(data['input'])['output']
        cases.append((f'checkpoint_{i}', dict(mlp, X=x)))
        prompts.append(dict(token_ids=ids, token_ids_sha256=digest, prompt=old['prompt']))
    from contract import HOLDOUT_PROMPT
    held_ids = tokens(HOLDOUT_PROMPT)  # Tokenization only; no held-out model execution.
    info = dict(status='complete', source=source, model=authority['model'], revision=authority['revision'],
                mlp_source_tensors=names, asset_sha256=authority['asset_sha256'],
                mlp_tensor_bits_sha256={k: hashlib.sha256(bf16_bits(v).tobytes()).hexdigest() for k, v in mlp.items()},
                attention_policy='explicit FP32 SDPA; BF16 attention output before Wo',
                attention_reference_sha256=sha(HERE.parent/'attention_sublayer/upstream.py'),
                prompts=prompts, holdout_token_ids=held_ids,
                holdout_token_ids_sha256=hashlib.sha256(np.asarray(held_ids, dtype=np.int64).tobytes()).hexdigest(),
                holdout_outputs_observed=False)
    return cases, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--characterize', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--tiny-only', action='store_true', help='diagnostic only; cannot freeze or verify')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py'))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    if args.tiny_only and not args.characterize:
        parser.error('--tiny-only requires --characterize')
    if args.freeze and (FROZEN.exists() or EVIDENCE.exists() or BUDGETS is None):
        parser.error('--freeze requires selected budgets and absent anchors; never overwrites')
    if not args.characterize and not args.freeze and not FROZEN.exists():
        parser.error('missing frozen anchors; characterize and select budgets first')
    torch.set_num_threads(1)
    commit_before = git_commit()
    before = source_identity()
    origin = provenance()
    expected = None
    if not args.characterize and not args.freeze:
        expected = frozen_record(args.checkpoint_dir is not None)
        for key, value in (('source_sha256', before), ('upstream', origin),
                           ('specification', jsonable(specification()))):
            if expected[key] != value:
                raise RuntimeError(f'frozen MLP {key} changed; refusing generation')
    ROOT.mkdir(parents=True, exist_ok=True)
    arrays, records = {}, {}
    for h, i, r, seed in DEVELOPMENT:
        if args.tiny_only and h == 896:
            continue
        check_case(f'h{h}_i{i}_r{r}_s{seed}', inputs(h, i, r, seed), arrays, records)
    for h, i in ((8, 12), (7, 11), (896, 4864)):
        if args.tiny_only and h == 896:
            continue
        base = inputs(h, i, 17, 1601)
        for kind in MUTATIONS:
            check_case(f'h{h}_i{i}_{kind}', mutate(base, kind), arrays, records)
    sweep, probes = activation_probe()
    probes['projection_tail'] = projection_tail_probe()
    arrays.update(save_arrays(ROOT/'activation', sweep))
    # Exact cancellation at the real down-projection reduction length.
    s = torch.ones((17, 4864), dtype=torch.bfloat16)
    w = torch.tensor([1/32, -1/32], dtype=torch.bfloat16).repeat(896, 2432)
    down = array(torch.nn.functional.linear(s, w))
    if np.any(down != 0):
        raise RuntimeError('paired down-projection cancellation is not zero')
    arrays.update(save_arrays(ROOT/'down_cancellation', dict(S=array(s), down=array(w), D=down)))
    checkpoint, assets = checkpoint_cases(args.checkpoint_dir)
    for case_id, data in checkpoint:
        check_case(case_id, data, arrays, records)
    record = jsonable(dict(specification=specification(), upstream=origin, arrays=arrays,
                          cases=records, activation=probes, checkpoint=assets,
                          source_sha256=before, scope='development references only; no Mojo MLP results'))
    if before != source_identity() or commit_before != git_commit():
        raise RuntimeError('MLP fixture sources changed during generation')
    if args.freeze:
        raw = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        payload = gzip.compress(raw, mtime=0)
        with EVIDENCE.open('xb') as f:
            f.write(payload)
        summary = {key: record[key] for key in ('specification', 'upstream', 'source_sha256')}
        summary.update(evidence_file=EVIDENCE.name, evidence_sha256=hashlib.sha256(payload).hexdigest(),
                       uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                       development_case_count=len(records), array_count=len(arrays),
                       holdout_outputs_observed=False)
        write_json(FROZEN, summary, exclusive=True)
    elif not args.characterize:
        validate_anchors(record, expected, ROOT)
    write_json(ROOT/('tiny_manifest.json' if args.tiny_only else 'manifest.json'), record)
    run = dict(commit_before=commit_before, commit_after=git_commit(),
               status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).splitlines(),
               source_before=before, source_after=source_identity(), checkpoint_dir=str(args.checkpoint_dir),
               command=sys.argv, anchors_verified=not args.characterize and not args.freeze,
               anchors_created=args.freeze)
    write_json(ROOT/'last_run.json', run)
    print('MLP development references complete; holdout outputs remain unopened.', flush=True)


if __name__ == '__main__':
    main()
