"""Compare a declared FP32 attention boundary with actual pinned upstream.

The official Qwen modules own normalization, projections, RoPE and cache flow.
Only SDPA's inputs are promoted to FP32 and its result cast back to BF16.
This is an explicit experimental inference policy, not the default BF16 eager
path or a claim about Qwen's training kernels. No model downloads here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from generate import ROOT, CASES, TOLERANCES, bf16, fixtures, crosscheck
from upstream import UpstreamAttention, errors, provenance

# Declared before observing these GPU results.
PRECISION_CASES = CASES + [(14, 2, 64, 4096, seed) for seed in (1009, 1237)]
PRECISION_CONTRACT = dict(
    authority='actual pinned Qwen2SdpaAttention with explicit FP32 SDPA inputs and BF16 output',
    backend='CPU SDPBackend.MATH; Torch 2.4.0; Transformers 4.43.1',
    storage='BF16 X, weights, Q/K/V, cache, GQA output, Wo output, residual output',
    attention_intermediates='FP32 scaled scores, softmax, and probability-times-V accumulation',
    operation_atol=.0078125, operation_rtol=.0078125,
    composition_atol=.03125, composition_rtol=.03125,
    cache='exact copies, prefix preservation and unused capacity poison',
    eager='retained as a separate compatibility comparison; existing gates unchanged',
    numpy_fp64='independent diagnostic on identical captured Q/K/V',
    promotion='experimental route; no default arithmetic change or performance claim',
)


def write_arrays(case_id, arrays, prefix=''):
    hashes = {}
    for name, a in arrays.items():
        path = ROOT / f'{prefix}{case_id}_{name}.npy'
        if not np.isfinite(a).all():
            raise ValueError(f'nonfinite array: {path.name}')
        np.save(path, a.astype(np.float32), allow_pickle=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def verify_frozen(record, filename):
    """Check the declared contract and every array against reviewed anchors."""
    frozen = json.loads(Path(__file__).with_name(filename).read_text())
    # Compare JSON values: in-memory case tuples serialize as JSON arrays.
    record = json.loads(json.dumps(record))
    for key, expected in frozen.items():
        if record.get(key) != expected:
            raise RuntimeError(f'precision fixture contract changed: {filename}: {key}')
    for name, expected in frozen['array_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'precision fixture array changed: {name}')


def upstream_contract():
    return {key: value for key, value in provenance(True).items()
            if key not in ('platform', 'machine')}


def fp64_attention(captured):
    q, k = captured['query'], captured['rotated_key']
    t, nq, d = q.shape
    nk = k.shape[1]
    v = captured['raw_value'].reshape(t, nk, d)
    result = np.empty(q.shape, dtype=np.float32)
    for head in range(nq):
        kh = head // (nq // nk)
        for start in range(0, t, 32):
            stop = min(t, start+32)
            scores = q[start:stop, head].astype(np.float64) @ k[:, kh].astype(np.float64).T / np.sqrt(d)
            scores[np.arange(t)[None, :] > np.arange(start, stop)[:, None]] = -np.inf
            p = np.exp(scores - scores.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            result[start:stop, head] = bf16(p @ v[:, kh].astype(np.float64))
    return result


def precision_case(case_id, spec, inputs, eager):
    nq, nk, d, t, _ = spec
    full = UpstreamAttention(inputs, nq, nk, d, fp32_attention=True).run(inputs['input'])
    shared = ['normalized', 'raw_query', 'raw_key', 'raw_value', 'cosine', 'sine',
              'query', 'rotated_key', 'cache_key', 'cache_value']
    for name in shared:
        if not np.array_equal(full[name], eager[name]):
            raise RuntimeError(f'precision experiment changed its input boundary: {case_id} {name}')
    chunks = [t-18, 17, 1] if t > 65 else [1] * t
    runner = UpstreamAttention(inputs, nq, nk, d, fp32_attention=True)
    start = 0
    chunk_checks = []
    for rows in chunks:
        result = runner.run(inputs['input'][start:start+rows])
        checks = {name: errors(result[name], full[name][start:start+rows], TOLERANCES[name])
                  for name in ('attention', 'projected', 'output')}
        if any(checks[name]['failed'] for name in ('projected', 'output')):
            raise RuntimeError(f'upstream FP32 full/chunked composition mismatch: {case_id} {start}')
        chunk_checks.append(dict(start=start, rows=rows, stages=checks))
        start += rows
    independent = fp64_attention(eager)
    record = dict(
        shared_upstream_inputs_exact=True,
        fp32_vs_eager={name: errors(full[name], eager[name], .03125)
                       for name in ('attention', 'projected', 'output')},
        fp32_vs_fp64=errors(full['attention'], independent, .0078125),
        eager_vs_fp64=errors(eager['attention'], independent, .03125),
        upstream_full_vs_chunked=chunk_checks,
    )
    hashes = write_arrays(case_id, {name: full[name] for name in ('attention', 'projected', 'output')}, 'fp32_')
    print('precision fixture', case_id, 'shape', nq, nk, d, t, 'generated', flush=True)
    return hashes, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, help='one diagnostic case; does not replace the primary manifest')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    frozen = json.loads(Path(__file__).with_name('checksums.json').read_text())
    hashes, checks = {}, {}
    for case_id, spec in enumerate(PRECISION_CASES):
        if args.case is not None and args.case != case_id:
            continue
        nq, nk, d, t, seed = spec
        if case_id < len(CASES):
            loaded = {}
            for name, expected_hash in frozen['array_sha256'].items():
                if name.startswith(f'{case_id}_') or name.startswith(f'upstream_{case_id}_'):
                    path = ROOT / name
                    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                        raise RuntimeError(f'changed source fixture: {name}')
                    hashes[name] = expected_hash
                    loaded[name[:-4]] = np.load(path, allow_pickle=False)
            inputs = {name[len(str(case_id))+1:]: a for name, a in loaded.items() if name.startswith(f'{case_id}_')}
            eager = {name[len(f'upstream_{case_id}_'):]: a for name, a in loaded.items() if name.startswith(f'upstream_{case_id}_')}
        else:
            inputs = fixtures(*spec)
            eager, _, _ = crosscheck(inputs, nq, nk, d, t)
            hashes.update(write_arrays(case_id, inputs))
            hashes.update(write_arrays(case_id, eager, 'upstream_'))
        added, checks[case_id] = precision_case(case_id, spec, inputs, eager)
        hashes.update(added)
    record = dict(
        cases=PRECISION_CASES, contract=PRECISION_CONTRACT, upstream=provenance(),
        fp32_upstream_contract=upstream_contract(),
        array_sha256=hashes, diagnostics=checks,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('generate.py', 'upstream.py', 'precision.py')},
    )
    name = 'precision_manifest.json' if args.case is None else f'precision_case_{args.case}.json'
    if args.case is None:
        verify_frozen(record, 'precision_checksums.json')
    (ROOT / name).write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
