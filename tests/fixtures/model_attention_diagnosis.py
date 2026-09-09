"""ATen-stage helpers for model_reference_diagnosis.py --backend.

Uses actual PyTorch operations under pass-through observation. No replacement
attention implementation supplies model outputs.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import model_reference as reference

CONTRACT = Path(__file__).with_name('model_backend_contract.json')
OPS = ['aten.mul.Scalar', 'aten.mul.Scalar', 'aten.bmm.default',
       'aten._softmax.default', 'aten.bmm.default']


def verify_sources(declaration, download=False):
    if torch.version.git_version != declaration['torch_git_version']:
        raise ValueError('PyTorch binary/source revision mismatch')
    root = reference.ROOT / 'build/upstream_sources'
    root.mkdir(parents=True, exist_ok=True)
    for name, record in declaration['source_files'].items():
        path = root / name
        if not path.exists() and download:
            data = urlopen(record['url'], timeout=45).read()
            if len(data) != record['bytes'] or reference.hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('downloaded source identity mismatch: ' + name)
            path.write_bytes(data)
        if not path.exists():
            raise ValueError('missing pinned source; rerun with --download-sources: ' + name)
        if path.stat().st_size != record['bytes'] or reference.sha(path) != record['sha256']:
            raise ValueError('pinned source identity mismatch: ' + name)
    if reference.sha(inspect.getfile(reference.qwen)) != declaration['source_files']['modeling_qwen2.py']['sha256']:
        raise ValueError('installed HF implementation differs from pinned upstream source')


class AttentionOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.values = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if str(func) in OPS:
            self.values.append(dict(op=str(func), output=output.detach().clone(),
                tensors=[value.detach().clone() for value in args if isinstance(value, torch.Tensor)],
                geometry=[dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
                          for value in args if isinstance(value, torch.Tensor)]))
        return output


def trace(q, k, v, mask):
    observer = AttentionOps()
    with reference.sdpa_kernel(backends=[reference.SDPBackend.MATH]):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        with observer:
            actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        repeated = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if not torch.equal(expected, actual) or not torch.equal(expected, repeated):
        raise ValueError('ATen observation or repeat changed SDPA output')
    if [v['op'] for v in observer.values] != OPS:
        raise ValueError('unexpected math-SDPA operation census')
    return expected, observer.values


@contextmanager
def deterministic():
    before = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn_only)


@contextmanager
def canonical_queries():
    original = torch.nn.functional.scaled_dot_product_attention

    def canonical(q, k, v, **kwargs):
        if kwargs.get('dropout_p', 0.) or kwargs.get('is_causal', False):
            raise ValueError('canonical diagnostic requires explicit mask and no dropout')
        rows, total = q.shape[-2], k.shape[-2]
        result = []
        for row in range(rows):
            end = total - rows + row + 1
            options = dict(kwargs)
            options['attn_mask'] = options['attn_mask'][..., row:row + 1, :end].contiguous()
            result.append(original(q[..., row:row + 1, :].contiguous(),
                k[..., :end, :].contiguous(), v[..., :end, :].contiguous(), **options))
        return torch.cat(result, dim=-2)

    with patch.object(torch.nn.functional, 'scaled_dot_product_attention', canonical):
        yield


def thresholds(ops, declaration, position, difference):
    a, b = ops[2]['tensors']
    records = []
    first = None
    for rows in declaration['duplicate_query_rows']:
        duplicated = a.repeat(1, rows, 1)
        actual = torch.bmm(duplicated, b)
        forced_mm = torch.stack([torch.mm(duplicated[h], b[h]) for h in range(len(a))])
        current = actual[:, :1]
        if first is None:
            first = current.clone()
        work = duplicated.shape[1] * duplicated.shape[2] * b.shape[2]
        records.append(dict(position=position, query_rows=rows, contraction=a.shape[2], keys=b.shape[2],
            work=work, source_predicted_path='small_bmm' if work < 400 else 'per_batch_mm',
            forced_per_head_mm_equal=torch.equal(actual, forced_mm),
            versus_one_row=difference(first.numpy(), current.numpy())))
    return records


def localize(model, spec, declaration, traced_forward, difference):
    length = spec['length']
    ids = np.random.default_rng(spec['seed']).integers(0, 151643, size=length).tolist()
    _, captures = traced_forward(model, ids, [length])
    _, cached_captures = traced_forward(model, ids, [1] * length)
    f = captures[0]
    full_output, full_ops = trace(f['query'], f['key'], f['value'], f['mask'])
    if not torch.equal(full_output, f['output']):
        raise ValueError('replaying captured operands changed upstream attention')
    with deterministic():
        deterministic_output, deterministic_ops = trace(f['query'], f['key'], f['value'], f['mask'])
    full_deterministic_equal = all(torch.equal(a['output'], b['output'])
                                  for a, b in zip(full_ops, deterministic_ops, strict=True))
    records, probes = [], []
    for position in range(length):
        end = position + 1
        c = cached_captures[position * 24]
        output, ops = trace(c['query'], c['key'], c['value'], c['mask'])
        if not torch.equal(output, c['output']):
            raise ValueError('cached captured replay changed upstream attention')
        stages = []
        for i, name in enumerate(declaration['operations']):
            a, b = full_ops[i]['output'], ops[i]['output']
            if i == 0:
                a = a[..., position:position + 1, :]
            elif i == 1:
                a = a[..., :end]
            elif i == 2:
                a = a[:, position:position + 1, :end]
            elif i == 3:
                a = a[..., position:position + 1, :end]
            else:
                a = a[:, position:position + 1, :]
            stages.append(dict(stage=name, **difference(a.numpy(), b.numpy())))
        with deterministic():
            _, dops = trace(c['query'], c['key'], c['value'], c['mask'])
        # Isolate downstream shape effects with identical contributing values.
        same_scores = full_ops[3]['tensors'][0][..., position:position + 1, :]
        p_full = torch.softmax(same_scores, dim=-1)
        p_prefix = torch.softmax(same_scores[..., :end].contiguous(), dim=-1)
        fixed_probabilities = full_ops[3]['output'][..., position:position + 1, :end]
        only_pv = torch.matmul(fixed_probabilities, f['value'][..., :end, :])
        original = full_output[..., position:position + 1, :]
        records.append(dict(position=position, stages=stages,
            full_geometry=[v['geometry'] for v in full_ops], cached_geometry=[v['geometry'] for v in ops],
            deterministic_equal=all(torch.equal(a['output'], b['output']) for a, b in zip(ops, dops, strict=True)),
            fixed_scores_softmax=difference(p_full[..., :end].numpy(), p_prefix.numpy()),
            fixed_probabilities_pv=difference(original.numpy(), only_pv.numpy())))
        if position in declaration['threshold_positions']:
            probes.extend(thresholds(ops, declaration, position, difference))
    return dict(**spec, ids=ids, observer_and_repeat_bitwise_equal=True,
                full_deterministic_equal=full_deterministic_equal,
                records=records, threshold_probes=probes)


@torch.no_grad()
def canonical_case(model, spec, difference):
    length = spec['length']
    ids = np.random.default_rng(spec['seed']).integers(0, 151643, size=length).tolist()
    with canonical_queries():
        full = reference.forward(model, ids, [length], all_logits=False)[0][2]
        repeat = reference.forward(model, ids, [length], all_logits=False)[0][2]
        repeat_equal = all(np.array_equal(full[name], repeat[name]) for name in full)
        cached = reference.forward(model, ids, [1] * length, all_logits=False)
    records = []
    for position, rows, values in cached:
        if rows != 1 or len(values) != 75 or set(values) != set(full):
            raise ValueError('canonical comparison boundary census changed')
        for name, actual in values.items():
            if name.startswith('cache_'):
                expected = full[name][:position + 1]
            elif name == 'logits':
                normalized = torch.from_numpy(full['final_norm'][position:position + 1]).bfloat16()
                expected = model.lm_head(normalized).float().numpy()
            else:
                expected = full[name][position:position + 1]
            records.append(dict(position=position, stage=name,
                                exact=bool(np.array_equal(expected, actual)),
                                **difference(expected, actual)))
    return dict(**spec, ids=ids, repeat_bitwise_equal=repeat_equal, checks=records,
                schedule_bitwise_equal=all(r['exact'] for r in records))


@torch.no_grad()
def run(model, traced_forward, difference, download=False):
    declaration = json.loads(CONTRACT.read_text())
    verify_sources(declaration, download)
    source_hash, contract_hash = reference.sha(__file__), reference.sha(CONTRACT)
    library = Path(torch.__file__).parent / 'lib/libtorch_cpu.dylib'
    metadata = dict(torch_git_version=torch.version.git_version,
        torch_build_configuration=torch.__config__.show(),
        cpu_library_sha256=reference.sha(library), intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic_initially_enabled=torch.are_deterministic_algorithms_enabled())
    localized, canonical = [], []
    for spec in declaration['trace_cases']:
        localized.append(localize(model, spec, declaration, traced_forward, difference))
        print('ATen localization complete:', spec, flush=True)
    for spec in declaration['canonical_cases']:
        canonical.append(canonical_case(model, spec, difference))
        print('Canonical query comparison complete:', spec,
              'bitwise equal:', canonical[-1]['schedule_bitwise_equal'], flush=True)
    if source_hash != reference.sha(__file__) or contract_hash != reference.sha(CONTRACT):
        raise ValueError('backend diagnostic source changed during execution')
    return dict(source_sha256=source_hash, declaration_sha256=contract_hash,
                declaration=declaration, environment=metadata, localized=localized, canonical=canonical)


def self_test():
    import unittest

    class AttentionDiagnosticTests(unittest.TestCase):
        def test_actual_operations_and_determinism_restore(self):
            generator = torch.Generator().manual_seed(812)
            values = [torch.randn(1, 2, 7, 64, generator=generator) for _ in range(3)]
            mask = torch.zeros(1, 1, 7, 7).masked_fill(torch.ones(7, 7).triu(1).bool(), float('-inf'))
            _, ops = trace(*values, mask)
            self.assertEqual([r['op'] for r in ops], OPS)
            original = torch.are_deterministic_algorithms_enabled()
            with deterministic():
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), original)

        def test_canonical_execution_preserves_causal_prefix(self):
            generator = torch.Generator().manual_seed(813)
            q, k, v = [torch.randn(1, 2, 7, 64, generator=generator) for _ in range(3)]
            mask = torch.zeros(1, 1, 7, 7).masked_fill(torch.ones(7, 7).triu(1).bool(), float('-inf'))
            with reference.sdpa_kernel(backends=[reference.SDPBackend.MATH]), canonical_queries():
                full = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                for row in range(7):
                    cached = torch.nn.functional.scaled_dot_product_attention(q[..., row:row + 1, :],
                        k[..., :row + 1, :], v[..., :row + 1, :], attn_mask=mask[..., row:row + 1, :row + 1])
                    self.assertTrue(torch.equal(full[..., row:row + 1, :], cached))
                changed = v.clone()
                changed[..., 3:, :] += 100
                other = torch.nn.functional.scaled_dot_product_attention(q, k, changed, attn_mask=mask)
                self.assertTrue(torch.equal(full[..., :3, :], other[..., :3, :]))

    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(AttentionDiagnosticTests))
    if not result.wasSuccessful():
        raise ValueError('ATen diagnostic self-test failed')
