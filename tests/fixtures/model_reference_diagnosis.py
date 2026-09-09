# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==1.26.4", "torch==2.4.0", "transformers==4.43.1"]
# ///
"""Reproduce the 17-token upstream scheduling ablation; no candidate outputs.

These alternate execution shapes diagnose reference variation. They are not
replacement oracles and do not change the accepted component arithmetic.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import model_reference as reference


def run(model, mode, ids):
    linear = torch.nn.functional.linear
    norm = reference.qwen.Qwen2RMSNorm.forward
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def rowwise_linear(value, weight, bias=None):
        flat = value.reshape(-1, value.shape[-1])
        result = torch.cat([linear(row[None], weight, bias) for row in flat])
        return result.reshape(*value.shape[:-1], weight.shape[0])

    def rowwise_norm(self, value):
        flat = value.reshape(-1, value.shape[-1])
        return torch.cat([norm(self, row[None]) for row in flat]).reshape(value.shape)

    def rowwise_sdpa(query, key, value, **kwargs):
        rows = query.shape[-2]
        past = key.shape[-2] - rows
        result = []
        for index in range(rows):
            extent = past + index + 1
            options = dict(kwargs)
            if options.get('attn_mask') is not None:
                options['attn_mask'] = options['attn_mask'][..., index:index + 1, :extent]
            options['is_causal'] = False
            result.append(sdpa(query[..., index:index + 1, :], key[..., :extent, :],
                               value[..., :extent, :], **options))
        return torch.cat(result, dim=-2)

    contexts = {
        'original': nullcontext(),
        'rowwise_linear': patch.object(torch.nn.functional, 'linear', rowwise_linear),
        'rowwise_norm': patch.object(reference.qwen.Qwen2RMSNorm, 'forward', rowwise_norm),
        'rowwise_sdpa': patch.object(torch.nn.functional, 'scaled_dot_product_attention', rowwise_sdpa),
    }
    with contexts[mode]:
        full = reference.forward(model, ids, [len(ids)])[0][2]
        repeated = reference.forward(model, ids, [len(ids)])[0][2]
        cached = reference.forward(model, ids, [1] * len(ids))[-1][2]
    stages = []
    for name in sorted(full):
        expected = full[name] if name.startswith('cache_') else full[name][-1:]
        actual = cached[name]
        error = expected.astype(np.float64) - actual.astype(np.float64)
        signal = float(np.linalg.norm(expected.astype(np.float64)))
        stages.append(dict(stage=name, different=int(np.count_nonzero(expected != actual)),
                           max_abs=float(np.abs(error).max()),
                           relative_rms=float(np.linalg.norm(error) / signal) if signal else 0.,
                           repeat_bitwise_equal=bool(np.array_equal(full[name], repeated[name]))))
    if len(stages) != 75:
        raise ValueError('incomplete diagnostic boundary census')
    return dict(stages=stages,
                full_top1=int(np.argmax(full['logits'][-1])),
                cached_top1=int(np.argmax(cached['logits'][-1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to replace diagnosis evidence')
    reference.verify_assets()
    source = reference.sha(__file__)
    provenance = reference.provenance()
    ids = np.random.default_rng(9120).integers(0, 151643, size=17).tolist()
    model = reference.load_model()
    records = {}
    for mode in ('original', 'rowwise_linear', 'rowwise_norm', 'rowwise_sdpa'):
        records[mode] = run(model, mode, ids)
        print('reference ablation complete:', mode, flush=True)
    if source != reference.sha(__file__) or provenance != reference.provenance():
        raise ValueError('diagnosis source changed during execution')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(source_sha256=source, reference=provenance,
        ids=ids, candidate_outputs_observed=False, reserved_outputs_observed=False,
        modes=records), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
