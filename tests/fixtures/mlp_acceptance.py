# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Explicit one-time holdout capture against a frozen, clean Mojo candidate.

The adjacent lock is a symlink to the existing reference script lock. This
entrypoint extends evaluation without modifying any frozen reference source.
Completion means fixtures were captured; candidate execution is recorded
separately by llm_mojo.mlp_validation in the locked project environment.
"""
import os
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / 'mlp'))
import numpy as np
import torch
from contract import HOLDOUT, STAGES, specification
from reference import UpstreamMLP, inputs, array, provenance, same_bits
from numerics import bf16_bits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def frozen():
    anchor = json.loads((HERE / 'mlp/checksums.json').read_text())
    raw = (HERE / 'mlp/development.json.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == anchor['evidence_sha256']
    payload = gzip.decompress(raw)
    assert hashlib.sha256(payload).hexdigest() == anchor['uncompressed_sha256']
    data = json.loads(payload)
    for name, digest in data['source_sha256'].items():
        assert sha(REPO / name) == digest, name
    return data


def checkpoint_input(directory, reference, token_ids=None):
    identity = reference['checkpoint']
    for name, digest in identity['asset_sha256'].items():
        if name != 'model.safetensors':
            assert sha(directory / name) == digest, name
    path = directory / 'model.attention-prefix.bin'
    assert path.stat().st_size == identity['source']['downloaded_prefix_bytes']
    assert sha(path) == identity['source']['downloaded_prefix_sha256']
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
        start = length + 8
        def load(name):
            spec = header[name]
            a, b = spec['data_offsets']
            assert spec['dtype'] == 'BF16' and 0 <= a <= b <= path.stat().st_size-start
            f.seek(start+a)
            raw = bytearray(f.read(b-a))
            assert len(raw) == b-a
            return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(spec['shape']).clone()
        p = 'model.layers.0.'
        values = dict(norm=array(load(p+'post_attention_layernorm.weight')))
        values.update({k: array(load(p+'mlp.'+k+'_proj.weight')) for k in ('gate','up','down')})
        for k, value in values.items():
            assert hashlib.sha256(bf16_bits(value).tobytes()).hexdigest() == identity['mlp_tensor_bits_sha256'][k]
        ids = identity['holdout_token_ids'] if token_ids is None else token_ids
        if token_ids is None:
            assert hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest() == identity['holdout_token_ids_sha256']
        data = dict(input=array(torch.nn.functional.embedding(torch.tensor(ids), load('model.embed_tokens.weight'))),
                    norm_weight=array(load(p+'input_layernorm.weight')),
                    output_weight=array(load(p+'self_attn.o_proj.weight')),
                    weight=array(torch.cat([load(p+f'self_attn.{k}_proj.weight') for k in ('q','k','v')])),
                    bias=array(torch.cat([load(p+f'self_attn.{k}_proj.bias') for k in ('q','k','v')])))
    spec = importlib.util.spec_from_file_location('holdout_attention', HERE/'attention_sublayer/upstream.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values['X'] = module.UpstreamAttention(data, 14, 2, 64, fp32_attention=True).run(data['input'])['output']
    return values


def optimization_spec(directory, reference, *, freeze=False):
    """Tokenization only; this must precede candidate or held-out model output."""
    from transformers import AutoTokenizer
    path = HERE / 'mlp_optimization_holdout.json'
    prompt = 'Describe how a lever can lift a heavy object, using a simple numerical example.'
    for name, digest in reference['checkpoint']['asset_sha256'].items():
        if name != 'model.safetensors':
            assert sha(directory / name) == digest, name
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    ids = tokenizer.apply_chat_template([
        dict(role='system', content='You are a helpful assistant.'),
        dict(role='user', content=prompt)], tokenize=True, add_generation_prompt=True)
    value = dict(schema=1, baseline_commit='ee3b99f', reference_sha256=sha(HERE/'mlp/checksums.json'),
                 cases=[[896,4864,r,seed] for seed in (3037,3041) for r in (1,17,4096)],
                 checkpoint_prompt=prompt, checkpoint_token_ids=ids,
                 checkpoint_token_ids_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
                 model_outputs_observed_at_declaration=False)
    if freeze:
        with path.open('x') as stream:
            stream.write(json.dumps(value,indent=2)+'\n')
    if json.loads(path.read_text()) != value:
        raise ValueError('optimization holdout declaration changed')
    return value



def decode_spec(directory, reference, *, freeze=False):
    """Declare recipes and tokenize only; do not compute model outputs."""
    from transformers import AutoTokenizer
    path = HERE / 'mlp_decode_holdout.json'
    prompt = 'Explain why a bicycle stays easier to balance while moving.'
    for name, digest in reference['checkpoint']['asset_sha256'].items():
        if name != 'model.safetensors':
            assert sha(directory / name) == digest, name
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    ids = tokenizer.apply_chat_template([
        dict(role='system', content='You are a helpful assistant.'),
        dict(role='user', content=prompt)], tokenize=True, add_generation_prompt=True)
    value = dict(schema=1, baseline_commit='5d6a3ee', reference_sha256=sha(HERE/'mlp/checksums.json'),
                 cases=[[896,4864,1,seed] for seed in (4051,4057,4073)],
                 checkpoint_prompt=prompt, checkpoint_token_ids=ids, checkpoint_row='last',
                 checkpoint_token_ids_sha256=hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
                 model_outputs_observed_at_declaration=False)
    if freeze:
        with path.open('x') as stream:
            stream.write(json.dumps(value,indent=2)+'\n')
    if json.loads(path.read_text()) != value:
        raise ValueError('decode holdout declaration changed')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-binary', type=Path)
    p.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--check-env', action='store_true')
    p.add_argument('--optimization', action='store_true')
    p.add_argument('--decode', action='store_true')
    p.add_argument('--freeze-decode-spec', action='store_true')
    p.add_argument('--freeze-optimization-spec', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    reference = frozen()
    origin = provenance()
    assert origin == reference['upstream']
    extra = None
    if (args.decode or args.freeze_decode_spec) and (args.optimization or args.freeze_optimization_spec):
        p.error('select one holdout campaign')
    if args.decode or args.freeze_decode_spec:
        if args.checkpoint_dir is None:
            p.error('local checkpoint directory required')
        extra = decode_spec(args.checkpoint_dir, reference, freeze=args.freeze_decode_spec)
    if args.optimization or args.freeze_optimization_spec:
        if args.checkpoint_dir is None:
            p.error('local checkpoint directory required for optimization holdouts')
        extra = optimization_spec(args.checkpoint_dir, reference, freeze=args.freeze_optimization_spec)
    if args.freeze_optimization_spec or args.freeze_decode_spec:
        print('Optimization recipes and checkpoint tokens frozen; no model outputs generated.')
        return
    if args.check_env:
        print('Pinned acceptance environment and frozen reference verified; no holdouts executed.')
        return
    if args.candidate_binary is None or args.checkpoint_dir is None:
        p.error('explicit candidate binary and local checkpoint directory are required')
    if git('status', '--porcelain'):
        raise RuntimeError('holdouts require a clean frozen candidate')
    # Read the numerical build receipt before any holdout exposure. The
    # verifier uses only stdlib dependencies in this isolated reference env.
    sys.path.insert(0, str(REPO / 'src'))
    from llm_mojo.mlp_validation import verify_build
    build_record = verify_build(args.candidate_binary)
    candidate = dict(commit=git('rev-parse','HEAD'), binary_sha256=sha(args.candidate_binary),
                     generator_sha256=sha(__file__), reference_sha256=sha(HERE/'mlp/checksums.json'))
    declaration = HERE / ('mlp_decode_holdout.json' if args.decode else 'mlp_optimization_holdout.json')
    if extra is not None:
        candidate['holdout_spec_sha256'] = sha(declaration)
    output = REPO / 'build/oracle_data' / ('mlp_decode_holdout' if args.decode else 'mlp_optimization_holdout' if extra else 'mlp_holdout')
    output.mkdir(exist_ok=False)
    # Write the exposure event before any holdout model computation.
    record = dict(candidate=candidate, upstream=origin, specification=specification(),
                  status='started', holdout_outputs_observed=True, arrays={}, cases={})
    record['candidate_build'] = build_record
    if extra is not None:
        record['additional_holdout_specification'] = extra
    def save_record():
        (output/'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    save_record()
    prefix = 'decode_holdout' if args.decode else 'optimization_holdout' if extra else 'holdout'
    recipes = [(f'{prefix}_h{h}_i{i}_r{r}_s{seed}', lambda h=h,i=i,r=r,seed=seed: inputs(h,i,r,seed))
               for h,i,r,seed in (extra['cases'] if extra else HOLDOUT)]
    recipes.append((prefix+'_checkpoint', lambda: checkpoint_input(
        args.checkpoint_dir, reference, extra['checkpoint_token_ids'] if extra else None)))
    for name, recipe in recipes:
        values = recipe()
        if args.decode and name.endswith('_checkpoint'):
            values['X'] = values['X'][-1:].copy()
        module = UpstreamMLP(values)
        full = module.run(values['X'])
        r,h = values['X'].shape
        i = values['gate'].shape[0]
        chunks = [1]*r if r <= 17 else [r-18,17,1]
        pieces = []
        start = 0
        for length in chunks:
            pieces.append(module.run(values['X'][start:start+length]))
            start += length
        exact = {k:same_bits(full[k],np.concatenate([piece[k] for piece in pieces])) for k in STAGES}
        directory = output / name
        directory.mkdir()
        for key, value in {**values, **full}.items():
            path = directory / (key+'.npy')
            np.save(path, value, allow_pickle=False)
            record['arrays'][str(path.relative_to(output))] = dict(sha256=sha(path), shape=list(value.shape))
        record['cases'][name] = dict(rows=r, hidden=h, intermediate=i, upstream_chunk_exact=exact)
        save_record()
        print('captured',name,flush=True)
    assert not git('status','--porcelain') and git('rev-parse','HEAD') == candidate['commit']
    assert sha(args.candidate_binary) == candidate['binary_sha256']
    assert verify_build(args.candidate_binary) == build_record
    assert sha(__file__) == candidate['generator_sha256']
    if extra is not None:
        assert sha(declaration) == candidate['holdout_spec_sha256']
    frozen()
    record['status'] = 'complete'
    record['manifest_payload_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    save_record()


if __name__ == '__main__':
    main()
