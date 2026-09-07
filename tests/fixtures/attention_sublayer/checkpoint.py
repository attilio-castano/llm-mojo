"""Generate first-layer Qwen fixtures from verified checkpoint assets.

Assets and arrays stay in ignored build/. Only explicitly requested --download
fetches assets; ordinary regeneration verifies local files without networking.
No model text generation or full-model quality claim is made.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import struct
import urllib.request

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from generate import ROOT
from precision import PRECISION_CONTRACT, precision_case, write_arrays, upstream_contract, verify_frozen
from upstream import array, provenance

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
ASSETS = ROOT.parents[1] / 'checkpoints' / 'qwen2.5-0.5b-instruct' / REVISION
# Header, embeddings and all first-layer attention tensors fit in this prefix.
# This is a separately checksummed byte range, not a verified full checkpoint.
PREFIX_BYTES = 302126368
PREFIX_SHA256 = '0d3c86fcaa9573dbac31055974018e4d1a94a07124e5d124747feed78b51f6fa'
ASSET_HASHES = {
    'model.safetensors': 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
    'config.json': '18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
    'generation_config.json': 'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}
PROMPTS = [
    'Explain why the sky appears blue in two sentences.',
    'Write a Python function that computes the mean of a list of numbers. Explain how it handles an empty list.',
    'Summarize these laboratory notes and identify what should be measured next.\n' +
    ('We compare attention computations using the same model weights and token inputs. '
     'Record the tensor dimensions, rounding points, cache positions and numerical errors. '
     'Repeat the measurements across short and long contexts before drawing a conclusion.\n') * 128,
]


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def assets(download=False, attention_prefix=False):
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSET_HASHES.items():
        if attention_prefix and name == 'model.safetensors':
            continue
        path = ASSETS / name
        if not path.exists():
            if not download:
                raise RuntimeError(f'missing verified checkpoint asset: {path}; use --download explicitly')
            print('download', name, 'at pinned revision', flush=True)
            hf_hub_download(MODEL, name, revision=REVISION, local_dir=ASSETS, token=False)
        if sha256(path) != expected:
            raise RuntimeError(f'checkpoint hash mismatch: {name}')
    if attention_prefix:
        path = ASSETS / 'model.attention-prefix.bin'
        if not path.exists() and download:
            request = urllib.request.Request(
                f'https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors?download=true',
                headers={'Range': f'bytes=0-{PREFIX_BYTES-1}'},
            )
            temporary = path.with_suffix('.part')
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes 0-{PREFIX_BYTES-1}/988097824':
                    raise RuntimeError('server did not return the declared checkpoint prefix')
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size != PREFIX_BYTES or sha256(temporary) != PREFIX_SHA256:
                raise RuntimeError('checkpoint prefix hash/size mismatch')
            temporary.replace(path)
        if not path.exists() or path.stat().st_size != PREFIX_BYTES or sha256(path) != PREFIX_SHA256:
            raise RuntimeError('missing or changed checkpoint prefix; use --download --attention-prefix explicitly')
    elif (ASSETS / 'model.safetensors').stat().st_size != 988097824:
        raise RuntimeError('checkpoint size differs from pinned model contract')
    return ASSETS


@contextmanager
def open_checkpoint(directory, attention_prefix):
    if not attention_prefix:
        with safe_open(directory / 'model.safetensors', framework='pt', device='cpu') as checkpoint:
            yield checkpoint
        return
    # Read only complete BF16 tensors described by the original safetensors
    # header. A truncated file must never be presented as a complete checkpoint.
    with (directory / 'model.attention-prefix.bin').open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size

        class PrefixTensors:
            def get_tensor(self, name):
                spec = header[name]
                start, end = spec['data_offsets']
                if spec['dtype'] != 'BF16' or not 0 <= start <= end <= PREFIX_BYTES-data_start:
                    raise RuntimeError(f'tensor not fully present in checkpoint prefix: {name}')
                stream.seek(data_start + start)
                payload = bytearray(stream.read(end-start))
                if len(payload) != end-start:
                    raise RuntimeError(f'incomplete checkpoint tensor: {name}')
                return torch.frombuffer(payload, dtype=torch.bfloat16).reshape(spec['shape'])

        yield PrefixTensors()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--attention-prefix', action='store_true',
                        help='use the separately checksummed prefix containing embeddings and layer-0 attention')
    args = parser.parse_args()
    directory = assets(args.download, args.attention_prefix)
    if args.download_only:
        return
    torch.set_num_threads(1)
    config = json.loads((directory / 'config.json').read_text())
    for key, expected in dict(hidden_size=896, num_attention_heads=14, num_key_value_heads=2,
                              rope_theta=1000000., rms_norm_eps=1e-6, use_sliding_window=False).items():
        if config[key] != expected:
            raise RuntimeError(f'checkpoint architecture mismatch: {key}')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    p = 'model.layers.0.'
    tensor_names = ['model.embed_tokens.weight', p+'input_layernorm.weight', p+'self_attn.o_proj.weight']
    tensor_names += [p+f'self_attn.{name}_proj.{kind}' for kind in ('weight','bias') for name in ('q','k','v')]
    hashes, checks, cases, prompts = {}, {}, [], []
    ROOT.mkdir(parents=True, exist_ok=True)
    with open_checkpoint(directory, args.attention_prefix) as checkpoint:
        weights = {name: checkpoint.get_tensor(name) for name in tensor_names}
        if any(value.dtype != torch.bfloat16 for value in weights.values()):
            raise RuntimeError('checkpoint tensor dtype changed')
        for i, prompt in enumerate(PROMPTS):
            ids = tokenizer.apply_chat_template(
                [{'role':'system','content':'You are a helpful assistant.'},
                 {'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True)
            original_length = len(ids)
            ids = ids[:4096]
            token_ids = torch.tensor(ids, dtype=torch.int64)
            inputs = dict(
                input=array(torch.nn.functional.embedding(token_ids, weights['model.embed_tokens.weight'])),
                weight=array(torch.cat([weights[p+f'self_attn.{name}_proj.weight'] for name in ('q','k','v')])),
                bias=array(torch.cat([weights[p+f'self_attn.{name}_proj.bias'] for name in ('q','k','v')])),
                output_weight=array(weights[p+'self_attn.o_proj.weight']),
                norm_weight=array(weights[p+'input_layernorm.weight']),
            )
            case_id, spec = 17+i, (14,2,64,len(ids),None)
            # This is the input of the first attention block, so no later
            # decoder layers need to run to obtain checkpoint activations.
            eager, _, chunk_checks = crosscheck_checkpoint(inputs, len(ids))
            hashes.update(write_arrays(case_id, inputs))
            hashes.update(write_arrays(case_id, eager, 'upstream_'))
            added, checks[case_id] = precision_case(case_id, spec, inputs, eager)
            checks[case_id]['eager_full_vs_chunked'] = chunk_checks
            hashes.update(added)
            cases.append(spec)
            prompts.append(dict(case=case_id, prompt=prompt, original_token_count=original_length,
                                token_count=len(ids), truncation='take first 4096 chat-template tokens',
                                token_ids_sha256=hashlib.sha256(token_ids.numpy().tobytes()).hexdigest()))
    record = dict(
        cases=cases, case_offset=17, contract=PRECISION_CONTRACT, upstream=provenance(),
        fp32_upstream_contract=upstream_contract(),
        model=MODEL, revision=REVISION, layer=0, asset_sha256=ASSET_HASHES,
        checkpoint_source=dict(
            mode='attention-prefix' if args.attention_prefix else 'full-file',
            full_file_sha256_verified=not args.attention_prefix,
            downloaded_prefix_bytes=PREFIX_BYTES if args.attention_prefix else None,
            downloaded_prefix_sha256=PREFIX_SHA256 if args.attention_prefix else None,
        ),
        source_tensors=tensor_names, prompts=prompts, array_sha256=hashes, diagnostics=checks,
        packages={name: importlib.metadata.version(name) for name in ('safetensors','tokenizers','huggingface-hub')},
        source_sha256={name:sha256(Path(__file__).with_name(name))
                       for name in ('generate.py','upstream.py','precision.py','checkpoint.py')},
    )
    (ROOT / 'checkpoint_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    verify_frozen(record, 'checkpoint_checksums.json')


def crosscheck_checkpoint(inputs, t):
    # Unlike synthetic fixtures, checkpoint inputs have no handwritten NumPy
    # composition. Execute the official modules and check persistent-cache use.
    from upstream import UpstreamAttention, errors
    eager = UpstreamAttention(inputs,14,2,64).run(inputs['input'])
    runner = UpstreamAttention(inputs,14,2,64)
    chunks = [t-18,17,1] if t>65 else [1]*t
    records, start = [], 0
    for rows in chunks:
        result = runner.run(inputs['input'][start:start+rows])
        stages = {name:errors(result[name],eager[name][start:start+rows],.03125)
                  for name in ('attention','projected','output')}
        if any(stages[name]['failed'] for name in ('projected','output')):
            raise RuntimeError('official eager checkpoint full/chunked mismatch')
        records.append(dict(start=start,rows=rows,stages=stages))
        start += rows
    return eager, None, records


if __name__ == '__main__':
    main()
