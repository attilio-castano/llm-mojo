"""Actual pinned upstream capture, deterministic inputs, and boundary probes."""

from contextlib import ExitStack
import hashlib
import inspect
import platform

import numpy as np
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as qwen
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from numerics import bf16_bits, from_bits, round_bf16, silu64, differences
from contract import STAGES, UP_VALUES


def tensor(a):
    return torch.from_numpy(np.array(a, copy=True)).to(torch.bfloat16)


def array(t):
    if t.dtype != torch.bfloat16 or t.device.type != 'cpu':
        raise ValueError('upstream boundary must be CPU BF16')
    return t.detach().float().numpy().copy()


def stored(a):
    return array(tensor(np.asarray(a, dtype=np.float32)))


def inputs(h, i, rows, seed):
    result = {}
    for tag, (name, shape, scale) in enumerate((
        ('X', (rows, h), 1), ('norm', (h,), 1/64),
        ('gate', (i, h), 1/np.sqrt(h)), ('up', (i, h), 1/np.sqrt(h)),
        ('down', (h, i), 1/np.sqrt(i)),
    )):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, h, i, tag])))
        value = rng.standard_normal(shape) * scale
        if name == 'norm':
            value += 1
        result[name] = stored(value)
    return result


def mutate(source, kind):
    data = {k: v.copy() for k, v in source.items()}
    if kind.startswith('zero_'):
        data[{'input': 'X'}.get(kind[5:], kind[5:])].fill(0)
    elif kind == 'swap_gate_up':
        data['gate'], data['up'] = data['up'], data['gate']
    elif kind in ('small_input', 'large_input'):
        data['X'] = stored(data['X'] * (1/64 if kind == 'small_input' else 16))
    elif kind == 'cancellation':
        data['norm'].fill(1)
        data['X'][:, ::2], data['X'][:, 1::2] = 1, -1
        h = data['X'].shape[1]
        if h % 2:
            data['X'][:, -1] = 0
        for name in ('gate', 'up'):
            data[name][:, 1::2] = data[name][:, :h-h%2:2]
    else:
        raise ValueError(f'unknown mutation: {kind}')
    return data


def same_bits(a, b):
    return a.shape == b.shape and np.array_equal(bf16_bits(a), bf16_bits(b))


class UpstreamMLP:
    def __init__(self, data):
        h, i = data['X'].shape[1], data['gate'].shape[0]
        self.h, self.i = h, i
        self.module = qwen.Qwen2MLP(Qwen2Config(hidden_size=h, intermediate_size=i, hidden_act='silu')).to(torch.bfloat16).eval()
        self.norm = qwen.Qwen2RMSNorm(h, eps=1e-6).to(torch.bfloat16).eval()
        with torch.no_grad():
            self.norm.weight.copy_(tensor(data['norm']))
            for name in ('gate', 'up', 'down'):
                getattr(self.module, name+'_proj').weight.copy_(tensor(data[name]))

    @torch.no_grad()
    def run(self, values, observe=True):
        x = tensor(values)
        captured = dict(X=array(x))
        with ExitStack() as stack:
            def hook(name):
                def save(module, args, result):
                    captured[name] = array(result)
                return save
            if observe:
                for name, module in [('N', self.norm), ('G', self.module.gate_proj),
                                     ('U', self.module.up_proj), ('D', self.module.down_proj)]:
                    stack.callback(module.register_forward_hook(hook(name)).remove)
                def pre_down(module, args):
                    captured['S'] = array(args[0])
                stack.callback(self.module.down_proj.register_forward_pre_hook(pre_down).remove)
                stack.callback(self.module.act_fn.register_forward_hook(hook('A')).remove)
            d = self.module(self.norm(x))
            captured['Y'] = array(x+d)
        if observe:
            rows = values.shape[0]
            for name in STAGES:
                shape = (rows, self.h if name in ('N', 'D', 'Y') else self.i)
                if captured[name].shape != shape or not np.isfinite(captured[name]).all():
                    raise ValueError(f'invalid captured boundary {name}')
            if not same_bits(captured['S'], array(tensor(captured['A'])*tensor(captured['U']))):
                raise ValueError('captured gating operands do not reconstruct down input')
            if not same_bits(captured['Y'], array(x+tensor(captured['D']))):
                raise ValueError('captured residual operands do not reconstruct output')
        return captured


def provenance():
    versions = (torch.__version__, transformers.__version__, np.__version__)
    if versions != ('2.4.0', '4.43.1', '1.26.4'):
        raise RuntimeError(f'wrong reference environment: {versions}')
    return dict(torch=versions[0], transformers=versions[1], numpy=versions[2],
                python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                device='cpu', backend='eager', threads=torch.get_num_threads(),
                module_sha256=hashlib.sha256(inspect.getsource(qwen).encode()).hexdigest(),
                torch_build=torch.__config__.show(), training_arithmetic_reproduced=False)


def activation_probe():
    bits = np.arange(65536, dtype=np.uint16)
    g = from_bits(bits[(bits & 0x7f80) != 0x7f80])
    a = array(torch.nn.functional.silu(tensor(g)))
    diagnostic = round_bf16(silu64(g))
    record = dict(silu_vs_fp64=differences(a, diagnostic), gating={})
    changed = bf16_bits(a) != bf16_bits(diagnostic)
    record['fp64_discrepancy_inputs'] = g[changed].tolist()
    if np.any(a[g <= -89] != 0) or not np.signbit(a[g <= -89]).all():
        raise RuntimeError('pinned SiLU negative-tail policy changed')
    # FP32 exp overflows for sufficiently negative G; this is a named backend
    # policy discrepancy, distinct from BF16 subnormal flushing.
    for name, mask in [('central', (g >= -16) & (g <= 16)),
                       ('negative_tail', g < -16), ('positive_tail', g > 16),
                       ('input_subnormal', (np.abs(g) < 2.0**-126) & (g != 0)),
                       ('zero', g == 0)]:
        record[name] = differences(a[mask], diagnostic[mask])
    for value in UP_VALUES:
        u = np.full_like(g, value)
        with np.errstate(over='ignore'):
            product = array(tensor(a)*tensor(u))
        mask = np.isfinite(product)
        expected = round_bf16(a[mask].astype(np.float64)*value)
        local = differences(product[mask], expected)
        if local['bit_differences']:
            raise RuntimeError('BF16 multiply differs from direct rounded exact product')
        unrounded = round_bf16(silu64(g[mask])*value)
        record['gating'][str(value)] = dict(finite=int(mask.sum()), overflow=int((~mask).sum()),
                                          exact_multiply=local,
                                          omitted_silu_boundary=differences(unrounded, product[mask]))
    # Use a non-power-of-two multiplier: scaling by powers of two usually
    # commutes with rounding and would be a weak omitted-boundary control.
    central = (np.abs(g) >= 1/64) & (np.abs(g) < 16)
    gc, ac = g[central], a[central]
    raw = torch.nn.functional.silu(tensor(gc).float())
    correct = array(tensor(ac)*tensor(np.full_like(ac, 1.5)))
    wrong = array((raw*1.5).to(torch.bfloat16))
    mismatch = (bf16_bits(correct) != bf16_bits(wrong)) & (bf16_bits(array(raw.to(torch.bfloat16))) == bf16_bits(ac))
    skipped = None
    if mismatch.any():
        j = int(np.flatnonzero(mismatch)[0])
        skipped = dict(g=float(gc[j]), up=1.5, rounded_silu=float(ac[j]),
                       correct=float(correct[j]), skipped=float(wrong[j]))
    if skipped is None:
        raise RuntimeError('activation boundary negative control did not discriminate')
    # An exact FP32 accumulator available from BF16 products: 1 + 2^-8.
    # D rounds to 1 before adding X=-1; skipping D rounding gives 2^-8.
    d_acc, x = np.float64(1+2**-8), np.float64(-1)
    correct = round_bf16(round_bf16(d_acc).astype(np.float64)+x)
    wrong = round_bf16(d_acc+x)
    assert correct != wrong
    record['rounding_regressions'] = dict(silu=skipped, down_residual=dict(
        accumulator=float(d_acc), residual=float(x), correct=float(correct), skipped=float(wrong)))
    return dict(sweep_G=g, sweep_A=a), record


def projection_tail_probe():
    """Reproduce the pinned ARM BF16 scalar-tail product rounding on seven terms."""
    data = inputs(7, 11, 17, 1601)
    captured = UpstreamMLP(data).run(data['X'])
    x, w = captured['N'][12], data['gate'][2]
    products = x.astype(np.float64)*w
    # v2.4.0 BlasKernel.cpp vectorizes groups of four, then evaluates the
    # remaining products as BFloat16 before adding to its float accumulator.
    tail_sum = np.sum(products[:4]) + np.sum(round_bf16(products[4:]).astype(np.float64))
    predicted = float(round_bf16(tail_sum))
    exact = float(round_bf16(np.sum(products)))
    actual = float(captured['G'][12, 2])
    fp32 = float(array(torch.nn.functional.linear(tensor(x).float(), tensor(w).float()).bfloat16()))
    if predicted != actual or exact != fp32 or predicted == exact:
        raise RuntimeError('pinned scalar-tail diagnostic no longer explains the discrepancy')
    return dict(input=x.tolist(), weight=w.tolist(), exact_dot=float(np.sum(products)),
                bf16_tail_dot=float(tail_sum), upstream=actual,
                direct_fp32=fp32, rounded_fp64=exact,
                source='https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/BlasKernel.cpp#L502-L507',
                interpretation='CPU scalar tail multiplies in BF16; keep all-FP32 products in the intended Mojo contract')
