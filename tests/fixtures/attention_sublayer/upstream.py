"""Capture the pinned Qwen eager implementation without rewriting its math.

CPU BF16 compatibility reference, not a reproduction of Qwen training kernels.
Hooks observe module boundaries; the rotary wrapper calls the original function.
Only fixture generation and numerical diagnostics import this module.
"""

import hashlib
import inspect
import platform
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2 import modeling_qwen2 as qwen
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


def tensor(array):
    return torch.from_numpy(np.array(array, copy=True)).to(torch.bfloat16)


def array(value):
    return value.detach().float().cpu().numpy().copy()


def provenance(fp32_attention=False):
    source = Path(inspect.getfile(qwen))
    record = dict(
        implementation="transformers.models.qwen2.modeling_qwen2.Qwen2Attention",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch=torch.__version__, transformers=transformers.__version__,
        numpy=np.__version__, device="cpu", backend="eager", dtype="bfloat16",
        torch_threads=torch.get_num_threads(), platform=platform.platform(),
        machine=platform.machine(), training_arithmetic_reproduced=False,
    )
    if fp32_attention:
        record.update(
            implementation="transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention",
            backend="SDPBackend.MATH",
            attention_boundary="explicit FP32 Q/K/V and mask; result rounded to BF16 before Wo",
        )
    return record


class UpstreamAttention:
    def __init__(self, inputs, nq, nk, d, fp32_attention=False):
        self.nq, self.nk, self.d = nq, nk, d
        self.fp32_attention = fp32_attention
        h, k = nq * d, nk * d
        config = Qwen2Config(
            hidden_size=h, num_attention_heads=nq, num_key_value_heads=nk,
            max_position_embeddings=4096, rope_theta=1000000.,
            attention_dropout=0., use_sliding_window=False,
        )
        cls = qwen.Qwen2SdpaAttention if fp32_attention else qwen.Qwen2Attention
        self.module = cls(config, layer_idx=0).to(torch.bfloat16).eval()
        self.norm = qwen.Qwen2RMSNorm(h, eps=1e-6).to(torch.bfloat16).eval()
        with torch.no_grad():
            for name, start, stop in [("q_proj", 0, h), ("k_proj", h, h+k),
                                      ("v_proj", h+k, h+2*k)]:
                getattr(self.module, name).weight.copy_(tensor(inputs["weight"][start:stop]))
                getattr(self.module, name).bias.copy_(tensor(inputs["bias"][start:stop]))
            self.module.o_proj.weight.copy_(tensor(inputs["output_weight"]))
            self.norm.weight.copy_(tensor(inputs["norm_weight"]))
        self.cache = DynamicCache()

    @torch.no_grad()
    def run(self, inputs):
        x = tensor(inputs)[None]
        r, h = inputs.shape
        p = self.cache.get_seq_length()
        positions = torch.arange(p, p+r)[None]
        mask = torch.zeros((1, 1, r, p+r), dtype=torch.bfloat16)
        mask.masked_fill_(torch.arange(p+r)[None] > positions[0, :, None], float("-inf"))
        captured = {}
        handles = []

        def save_projection(name):
            def capture(module, args, result):
                captured[name] = array(result[0])
            return capture

        for name, attr in [("raw_query", "q_proj"), ("raw_key", "k_proj"),
                           ("raw_value", "v_proj")]:
            handles.append(getattr(self.module, attr).register_forward_hook(save_projection(name)))

        def save_attention(module, args):
            captured["attention"] = array(args[0][0]).reshape(r, self.nq, self.d)

        handles.append(self.module.o_proj.register_forward_pre_hook(save_attention))
        original = qwen.apply_rotary_pos_emb
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        def fp32_sdpa(q, k, v, **kwargs):
            # The boundary casts are the experiment. The actual attention math
            # remains in the pinned upstream implementation, forced to math.
            mask = kwargs.get('attn_mask')
            if mask is not None and mask.is_floating_point():
                kwargs['attn_mask'] = mask.float()
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return original_sdpa(q.float(), k.float(), v.float(), **kwargs).to(q.dtype)

        def save_rotation(q, k, cos, sin, position_ids, unsqueeze_dim=1):
            rotated_q, rotated_k = original(q, k, cos, sin, position_ids, unsqueeze_dim)
            captured["cosine"] = array(cos[p:p+r])
            captured["sine"] = array(sin[p:p+r])
            captured["query"] = array(rotated_q[0].transpose(0, 1))
            captured["rotated_key"] = array(rotated_k[0].transpose(0, 1))
            return rotated_q, rotated_k

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(qwen, "apply_rotary_pos_emb", save_rotation))
                if self.fp32_attention:
                    stack.enter_context(patch.object(torch.nn.functional, 'scaled_dot_product_attention', fp32_sdpa))
                normalized = self.norm(x)
                branch = self.module(
                    normalized, attention_mask=mask, position_ids=positions,
                    past_key_value=self.cache, use_cache=True,
                    cache_position=positions[0],
                )[0]
            captured["normalized"] = array(normalized[0])
            captured["projected"] = array(branch[0])
            captured["output"] = array((x + branch)[0])
            captured["cache_key"] = array(self.cache.key_cache[0][0].transpose(0, 1))
            captured["cache_value"] = array(self.cache.value_cache[0][0].transpose(0, 1))
        finally:
            for handle in handles:
                handle.remove()
        return captured


def errors(actual, expected, tolerance):
    if actual.shape != expected.shape:
        raise ValueError(f"comparison shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite reference or implementation result")
    delta = np.abs(actual - expected)
    scaled = delta / (1 + np.abs(expected))
    worst = np.unravel_index(np.argmax(scaled), scaled.shape)
    return dict(
        elements=int(actual.size), failed=int(np.count_nonzero(scaled > tolerance)),
        max_abs=float(delta.max()), max_scaled=float(scaled.max()),
        rms_abs=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
        index=[int(i) for i in worst], actual=float(actual[worst]), expected=float(expected[worst]),
    )


def reference_regressions():
    """Execute upstream for both RoPE examples; no handwritten rotation oracle."""
    x = tensor([[[[4.59375, -0.03466796875]]]])
    c = tensor([[0.796875, 0.796875]])
    s = tensor([[0.60546875, 0.60546875]])
    rotated, _ = qwen.apply_rotary_pos_emb(x, x, c, s, torch.tensor([[0]]))
    if rotated[0,0,0,0].item() != 3.671875:
        raise RuntimeError("pinned eager RoPE rounding changed")
    rotary = qwen.Qwen2RotaryEmbedding(64, max_position_embeddings=4096, base=1000000)
    x = torch.zeros((1,1,1,64), dtype=torch.bfloat16)
    x[0,0,0,7], x[0,0,0,39] = 3.46875, 3.65625
    c, s = rotary(x, seq_len=4096)
    rotated, _ = qwen.apply_rotary_pos_emb(x, x, c, s, torch.tensor([[2370]]))
    if rotated[0,0,0,39].item() != 0.09375:
        raise RuntimeError("pinned rotary-table regression changed")
    return dict(rounding_output=3.671875, table_output=0.09375,
                sine_at_2370_7=float(s[2370,7]))


def compare_attention_backends(inputs, captured, nq, nk, d, tolerances):
    """Run actual pinned official math SDPA; diagnostic only.

    Torch 2.4.0 scales BF16 Q and K before matmul in this backend. Do not infer
    its precision from newer Torch versions that upcast these intermediates.
    """
    runner = UpstreamAttention(inputs, nq, nk, d)
    module = qwen.Qwen2SdpaAttention(runner.module.config, layer_idx=0).to(torch.bfloat16).eval()
    module.load_state_dict(runner.module.state_dict())
    runner.module = module
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        result = runner.run(inputs['input'])
    return dict(
        implementation='transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention',
        backend='SDPBackend.MATH', device='cpu', diagnostic_only=True,
        atol=tolerances, rtol=tolerances,
        sdpa_vs_eager={name: errors(result[name], captured[name], tol)
                       for name, tol in tolerances.items()},
    )


def score_boundary_diagnostic(captured, position, head, key_position, tolerance):
    """Explain one score change and propagate it through actual Torch operations.

    The FP64 calculation diagnoses proximity to a BF16 midpoint. It does not
    replace the upstream reference or prescribe GPU accumulation order.
    """
    q = captured['query'][position]
    k = captured['rotated_key'][:position+1]
    nq, d = q.shape
    nk = k.shape[1]
    v = captured['raw_value'][:position+1].reshape(position+1,nk,d)
    products = q[head].astype(np.float64) * k[key_position,head//(nq//nk)].astype(np.float64)
    serial = np.float32(0)
    for product in products:
        serial = np.float32(serial + np.float32(product))
    tq = tensor(q)
    tk = tensor(k).repeat_interleave(nq//nk,dim=1)
    tv = tensor(v).repeat_interleave(nq//nk,dim=1).permute(1,0,2)
    scores = torch.bmm(tq[:,None,:],tk.permute(1,2,0))[:,0,:] / np.sqrt(d)
    official_score = float(scores[head,key_position])
    official_probability = torch.softmax(scores.float(),dim=-1).bfloat16()
    official_output = torch.bmm(official_probability[:,None,:],tv)[:,0,:]
    if not np.array_equal(array(official_output), captured['attention'][position]):
        raise RuntimeError('single-row diagnostic does not reproduce the captured eager row')
    changed = scores.clone()
    changed[head,key_position] = float(serial / np.sqrt(d))
    probability = torch.softmax(changed.float(),dim=-1).bfloat16()
    output = torch.bmm(probability[:,None,:],tv)[:,0,:]
    comparison = errors(array(output),array(official_output),tolerance)
    return dict(
        position=position, head=head, key_position=key_position,
        fp64_scaled_dot=float(products.sum()/np.sqrt(d)),
        serial_fp32_scaled_dot=float(serial/np.sqrt(d)),
        upstream_bf16_score=official_score, serial_bf16_score=float(changed[head,key_position]),
        upstream_probability=float(official_probability[head,key_position]),
        changed_probability=float(probability[head,key_position]),
        atol=tolerance, rtol=tolerance,
        changed_output_vs_upstream=comparison,
        changed_dimensions=np.argwhere(np.abs(array(output)-array(official_output)) >
                                       tolerance*(1+np.abs(array(official_output)))).tolist(),
        changed_score_output=array(output).tolist(),
    )
