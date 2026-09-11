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
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import model_reference as reference

DETAIL_CONTRACT = Path(__file__).with_name('model_diagnosis_contract.json')


def difference(expected, actual):
    a, b = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or mismatched diagnostic values')
    error = np.abs(a - b)
    signal = float(np.linalg.norm(a))
    if signal == 0 and np.any(error):
        raise ValueError('nonzero diagnostic error on zero signal')
    return dict(different=int(np.count_nonzero(a != b)), max_abs=float(error.max()),
                p99_abs=float(np.quantile(error, .99)),
                relative_rms=float(np.linalg.norm(error) / signal) if signal else 0.)


def prediction(expected, actual):
    a, b = np.asarray(expected).reshape(-1), np.asarray(actual).reshape(-1)
    stats = difference(a, b)
    winner = int(np.argmax(a))
    runner_up = float(np.partition(a, -2)[-2])
    margin = float(a[winner]) - runner_up
    other = int(np.argmax(b))
    return dict(**stats, full_token=winner, cached_token=other, same_token=winner == other,
                full_margin=margin, margin_certified=margin > 2 * stats['max_abs'])


def traced_forward(model, ids, schedule):
    traces = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def capture(query, key, value, **options):
        output = sdpa(query, key, value, **options)
        if query.dtype != torch.float32 or output.dtype != torch.float32:
            raise ValueError('trace must observe the declared FP32 SDPA boundary')
        traces.append(dict(query=query.detach().clone(), key=key.detach().clone(),
                           value=value.detach().clone(), mask=options['attn_mask'].detach().clone(),
                           output=output.detach().clone(),
                           geometry={name: dict(shape=list(tensor.shape), stride=list(tensor.stride()))
                                     for name, tensor in [('query', query), ('key', key), ('value', value),
                                                          ('mask', options['attn_mask']), ('output', output)]}))
        return output

    with patch.object(torch.nn.functional, 'scaled_dot_product_attention', capture):
        calls = reference.forward(model, ids, schedule)
    if len(traces) != 24 * len(schedule):
        raise ValueError('incomplete attention trace')
    # Instrumentation must not change any observed computation.
    control = reference.forward(model, ids, schedule)
    for observed, unobserved in zip(calls, control, strict=True):
        if observed[:2] != unobserved[:2] or set(observed[2]) != set(unobserved[2]):
            raise ValueError('observation changed call geometry')
        for name in observed[2]:
            if not np.array_equal(observed[2][name], unobserved[2][name]):
                raise ValueError('observation changed boundary: ' + name)
    return calls, traces


@torch.no_grad()
def detailed_case(model, spec):
    length = spec['length']
    ids = np.random.default_rng(spec['seed']).integers(0, 151643, size=length).tolist()
    full_calls, full_trace = traced_forward(model, ids, [length])
    cached_calls, cached_trace = traced_forward(model, ids, [1] * length)
    full = full_calls[0][2]
    attention, hidden, predictions, normalization, shapes = [], [], [], [], []
    for position in range(length):
        cached = cached_calls[position][2]
        for layer in range(24):
            f, c = full_trace[layer], cached_trace[position * 24 + layer]
            a, b = f['output'][..., position:position + 1, :], c['output']
            before = difference(a.numpy(), b.numpy())
            rounded_a, rounded_b = a.bfloat16().float(), b.bfloat16().float()
            after = difference(rounded_a.numpy(), rounded_b.numpy())
            flips = []
            indices = torch.nonzero(rounded_a != rounded_b)
            for idx in indices[:3]:
                index = tuple(int(i) for i in idx)
                lower, upper = sorted([float(rounded_a[index]), float(rounded_b[index])])
                flips.append(dict(index=list(index), full_fp32=float(a[index]), cached_fp32=float(b[index]),
                                  full_bf16=float(rounded_a[index]), cached_bf16=float(rounded_b[index]),
                                  midpoint=(lower + upper) / 2))
            attention.append(dict(position=position, layer=layer, before_bf16=before, after_bf16=after,
                query_equal=torch.equal(f['query'][..., position:position + 1, :], c['query']),
                key_equal=torch.equal(f['key'][..., :position + 1, :], c['key']),
                value_equal=torch.equal(f['value'][..., :position + 1, :], c['value']),
                flip_examples=flips, full_geometry=f['geometry'], cached_geometry=c['geometry']))
        for layer in range(25):
            name = f'hidden_{layer}'
            hidden.append(dict(position=position, layer=layer,
                               **difference(full[name][position:position + 1], cached[name])))
        predictions.append(dict(position=position,
                                **prediction(full['logits'][position], cached['logits'][0])))
        # Reproduce the upstream norm expression before explaining its scaling.
        a = torch.from_numpy(full['hidden_24'][position:position + 1]).bfloat16()
        b = torch.from_numpy(cached['hidden_24']).bfloat16()
        epsilon = model.model.norm.variance_epsilon
        inv_a = torch.rsqrt(a.float().square().mean(-1, keepdim=True) + epsilon)
        inv_b = torch.rsqrt(b.float().square().mean(-1, keepdim=True) + epsilon)
        norm_a = model.model.norm.weight * (a.float() * inv_a).bfloat16()
        norm_b = model.model.norm.weight * (b.float() * inv_b).bfloat16()
        if (not np.array_equal(norm_a.float().numpy(), full['final_norm'][position:position + 1])
                or not np.array_equal(norm_b.float().numpy(), cached['final_norm'])):
            raise ValueError('norm diagnostic does not reproduce upstream')
        index = int((norm_a.float() - norm_b.float()).abs().argmax())
        # A large absolute error at a large reference value can pass rtol.
        # Also identify the actual worst pointwise-budget coordinate.
        error = (norm_a.float() - norm_b.float()).abs()
        required = error - reference.GATES['hidden']['rtol'] * norm_a.float().abs()
        gate_index = int(required.argmax())
        normalization.append(dict(position=position, index=index, gamma=float(model.model.norm.weight[index]),
            full_input=float(a[0, index]), cached_input=float(b[0, index]),
            full_inverse_rms=float(inv_a[0, 0]), cached_inverse_rms=float(inv_b[0, 0]),
            full_output=float(norm_a[0, index]), cached_output=float(norm_b[0, index]),
            pointwise_coordinate=dict(index=gate_index, required_atol=max(0., float(required[0, gate_index])),
                gamma=float(model.model.norm.weight[gate_index]),
                full_input=float(a[0, gate_index]), cached_input=float(b[0, gate_index]),
                full_output=float(norm_a[0, gate_index]), cached_output=float(norm_b[0, gate_index])),
            **difference(norm_a.float().numpy(), norm_b.float().numpy())))
        # Isolate shape from operand changes using ONLY the first layer's full inputs.
        f = full_trace[0]
        q = f['query'][..., position:position + 1, :]
        k, v = f['key'], f['value']
        mask = f['mask'][..., position:position + 1, :]
        with reference.sdpa_kernel(backends=[reference.SDPBackend.MATH]):
            one_query = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            prefix = torch.nn.functional.scaled_dot_product_attention(q, k[..., :position + 1, :],
                        v[..., :position + 1, :], attn_mask=mask[..., :position + 1])
        high = torch.softmax(q.double() @ k.double().transpose(-2, -1) / math.sqrt(q.shape[-1])
                             + mask.double(), dim=-1) @ v.double()
        original = f['output'][..., position:position + 1, :]
        shapes.append(dict(position=position,
            one_query_full_keys=difference(original.numpy(), one_query.numpy()),
            one_query_prefix_keys=difference(original.numpy(), prefix.numpy()),
            query_only_bf16_flips=int((original.bfloat16() != one_query.bfloat16()).sum()),
            prefix_bf16_flips=int((original.bfloat16() != prefix.bfloat16()).sum()),
            prefix_matches_cached=torch.equal(prefix, cached_trace[position * 24]['output']),
            full_vs_fp64=difference(high.numpy(), original.numpy()),
            query_vs_fp64=difference(high.numpy(), one_query.numpy()),
            prefix_vs_fp64=difference(high.numpy(), prefix.numpy())))
    return dict(**spec, ids=ids, observer_bitwise_equal=True, attention=attention,
                hidden=hidden, predictions=predictions, normalization=normalization, shapes=shapes)


class CachedReference:
    def __init__(self, model):
        self.model, self.cache, self.length = model, None, 0

    @torch.no_grad()
    def consume(self, ids):
        rows = len(ids)
        positions = torch.arange(self.length, self.length + rows)
        mask = torch.zeros((1, 1, rows, self.length + rows), dtype=torch.bfloat16)
        mask.masked_fill_(torch.arange(self.length + rows)[None] > positions[:, None], float('-inf'))
        with reference.precision_policy() as count:
            result = self.model.model(input_ids=torch.tensor([ids]), attention_mask=mask,
                position_ids=positions[None], past_key_values=self.cache, use_cache=True, return_dict=True)
        if count[0] != 24:
            raise ValueError('incomplete SDPA coverage in generation diagnosis')
        self.cache = result.past_key_values
        self.length += rows
        logits = self.model.lm_head(result.last_hidden_state[:, -1:])[0, 0].float().numpy().copy()
        if not np.isfinite(logits).all():
            raise ValueError('nonfinite generation logits')
        return logits


def initialized(model, ids, tokenwise):
    state = CachedReference(model)
    for part in ([ [i] for i in ids ] if tokenwise else [ids]):
        logits = state.consume(part)
    return state, logits


def text_case(model, tokenizer, text, declaration):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    history = ids.copy()
    baseline, cached = initialized(model, ids, False)
    tokenwise, incremental = initialized(model, ids, True)
    # Qualify the new persistent-cache harness against the existing capture path.
    for schedule, logits in [([len(ids)], cached), ([1] * len(ids), incremental)]:
        expected = reference.forward(model, ids, schedule, all_logits=False)[-1][2]['logits'][0]
        if not np.array_equal(expected, logits):
            raise ValueError('generation harness differs from established upstream capture')
    comparisons, full_tokens = [], []
    for step in range(declaration['greedy_steps']):
        _, full = initialized(model, history, False)
        comparisons.append(dict(step=step, history=history.copy(),
            cached_full_prompt=prediction(full, cached), cached_tokenwise_prompt=prediction(full, incremental)))
        token = int(np.argmax(full))
        full_tokens.append(token)
        if token in declaration['stop_ids'] or step + 1 == declaration['greedy_steps']:
            break
        history.append(token)
        cached, incremental = baseline.consume([token]), tokenwise.consume([token])
    trajectories = {}
    for name, mode in [('cached_full_prompt', False), ('cached_tokenwise_prompt', True)]:
        state, logits = initialized(model, ids, mode)
        tokens = []
        for step in range(declaration['greedy_steps']):
            token = int(np.argmax(logits))
            tokens.append(token)
            if token in declaration['stop_ids'] or step + 1 == declaration['greedy_steps']:
                break
            logits = state.consume([token])
        trajectories[name] = dict(tokens=tokens, text=tokenizer.decode(tokens, skip_special_tokens=True),
                                  matches_full=tokens == full_tokens)
    return dict(text=text, ids=ids, comparisons=comparisons, trajectories=trajectories,
                full_tokens=full_tokens, full_text=tokenizer.decode(full_tokens, skip_special_tokens=True))


def detail(model):
    from tokenizers import Tokenizer
    declaration = json.loads(DETAIL_CONTRACT.read_text())
    tokenizer_path = reference.ASSETS / 'tokenizer.json'
    # The pinned tokenizer is an input decoder, not an alternative model oracle.
    tokenizer_hash = reference.sha(tokenizer_path)
    if tokenizer_hash != 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539':
        raise ValueError('pinned tokenizer identity mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cases, texts = [], []
    for spec in declaration['random_cases']:
        cases.append(detailed_case(model, spec))
        print('Detailed reference trace complete:', spec, flush=True)
    for index, text in enumerate(declaration['text_cases']):
        texts.append(text_case(model, tokenizer, text, declaration))
        print('Prediction diagnosis complete: text', index, flush=True)
    return dict(declaration=declaration, declaration_sha256=reference.sha(DETAIL_CONTRACT),
                tokenizer_sha256=tokenizer_hash, cases=cases, texts=texts)


def self_test():
    import unittest

    class DiagnosticTests(unittest.TestCase):
        def test_bf16_midpoint_crossing(self):
            midpoint = 1.00390625
            a = torch.tensor([midpoint - 2**-23], dtype=torch.float32)
            b = torch.tensor([midpoint + 2**-23], dtype=torch.float32)
            self.assertLess(difference(a.numpy(), b.numpy())['max_abs'], 1e-6)
            self.assertEqual(difference(a.bfloat16().float().numpy(), b.bfloat16().float().numpy())['max_abs'], 1/128)

        def test_margin_distinguishes_stability_from_small_error(self):
            stable = prediction(np.array([2., 0.]), np.array([1.9, 0.1]))
            self.assertTrue(stable['same_token'] and stable['margin_certified'])
            close = prediction(np.array([1., 0.999]), np.array([0.999, 1.]))
            self.assertFalse(close['same_token'] or close['margin_certified'])
            tie = prediction(np.array([1., 1.]), np.array([1., 1.]))
            self.assertEqual(tie['full_token'], 0)
            self.assertFalse(tie['margin_certified'])

        def test_nonfinite_or_wrong_shape_fails(self):
            with self.assertRaises(ValueError):
                difference(np.ones(2), np.ones(3))
            with self.assertRaises(ValueError):
                difference(np.ones(2), np.array([np.nan, 1.]))

    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests))
    if not result.wasSuccessful():
        raise ValueError('diagnostic self-test failed')
    import model_attention_diagnosis as backend
    backend.self_test()


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


@torch.no_grad()
def native_operations(model, output):
    """Observe actual HF operations on the already exposed one-token input."""
    import subprocess
    from model_attention_diagnosis import canonical_queries, exact_array
    contract = Path(__file__).with_name('model_consistency.json')
    source, contract_hash, before = reference.sha(__file__), reference.sha(contract), reference.provenance()
    canonical_source = Path(__file__).with_name('model_attention_diagnosis.py')
    canonical_hash = reference.sha(canonical_source)
    spec = json.loads(contract.read_text())['development_cases'][0]
    if spec['length'] != 1:
        raise ValueError('operation diagnosis is scoped to the exposed one-token case')
    ids = np.random.default_rng(spec['seed']).integers(0, 151643, size=1).tolist()
    captured, handles = {}, []
    def retain(name, value):
        if name in captured or value.dtype != torch.bfloat16:
            raise ValueError('duplicate or non-BF16 operation boundary')
        captured[name] = value.detach().float().numpy().reshape(1, -1).copy()
    def hook(module, name, before=False, tuple_output=False):
        if before:
            handles.append(module.register_forward_pre_hook(lambda module, args: retain(name, args[0])))
        else:
            handles.append(module.register_forward_hook(
                lambda module, args, result: retain(name, result[0] if tuple_output else result)))
    with canonical_queries():
        unobserved = reference.forward(model, ids, [1], all_logits=False)[0][2]
        for i, layer in enumerate(model.model.layers):
            prefix = f'layer_{i}_'
            hook(layer.input_layernorm, prefix+'X', before=True)
            hook(layer.input_layernorm, prefix+'N_att')
            for short, proj in [('Q_raw','q'),('K_raw','k'),('V_raw','v')]:
                hook(getattr(layer.self_attn, proj+'_proj'), prefix+short)
            hook(layer.self_attn.o_proj, prefix+'O', before=True)
            hook(layer.self_attn.o_proj, prefix+'B_att')
            hook(layer.post_attention_layernorm, prefix+'Z', before=True)
            hook(layer.post_attention_layernorm, prefix+'N_mlp')
            hook(layer.mlp.gate_proj, prefix+'G')
            hook(layer.mlp.up_proj, prefix+'U')
            hook(layer.mlp.act_fn, prefix+'A')
            hook(layer.mlp.down_proj, prefix+'S', before=True)
            hook(layer.mlp.down_proj, prefix+'B_mlp')
            hook(layer, prefix+'Y', tuple_output=True)
        try:
            observed = reference.forward(model, ids, [1], all_logits=False)[0][2]
        finally:
            for handle in handles:
                handle.remove()
    if len(captured) != 24*15 or not all(exact_array(observed[n], unobserved[n]) for n in observed):
        raise ValueError('operation observation changed model outputs or is incomplete')
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for name, value in captured.items():
        if not np.isfinite(value).all():
            raise ValueError('nonfinite operation fixture')
        path = output/(name+'.bin')
        ((value.view(np.uint32) >> 16).astype('<u2')).tofile(path)
        arrays[name] = dict(shape=list(value.shape), sha256=reference.sha(path))
    if (source != reference.sha(__file__) or contract_hash != reference.sha(contract)
            or canonical_hash != reference.sha(canonical_source) or before != reference.provenance()):
        raise ValueError('operation reference source changed during execution')
    report = dict(kind='model_identical_operand_reference', source_sha256=source,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference.ROOT,text=True).strip(),
        reference=before, contract_sha256=contract_hash, canonical_source_sha256=canonical_hash, case=spec, ids=ids,
        observer_bitwise_equal=True, arrays=arrays, reserved_outputs_observed=False,
        scope='Actual upstream intermediate operands for the already exposed native one-token failure; no native outputs read by this generator.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Observed actual HF operations:',len(arrays),'boundaries; model observation unchanged',flush=True)


@torch.no_grad()
def fast_affine_diagnosis(model, qualification):
    """Reproduce the exposed calibration maximum; inspect identical operands.

    This is diagnosis on already observed reference inputs, never a new
    qualification or authorization to change any numerical threshold.
    """
    from fractions import Fraction
    import model_calibration as calibration
    from model_consistency import calls
    from model_attention_diagnosis import exact_array
    report = json.loads(qualification.read_text())
    if report['passed'] or report['source'] != calibration.fast_identity():
        raise ValueError('require the original failed Fast qualification')
    raw_path = qualification.parent/'observations.jsonl'
    if reference.sha(raw_path) != report['observations_sha256']:
        raise ValueError('calibration observation hash mismatch')
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    worst = max(rows, key=lambda r: r['relative_rms'])
    if worst['arm'] != 'fp32_split2' or worst['schedule'] != [worst['length']]:
        raise ValueError('this bounded diagnosis requires an exposed split-affine full call')
    ids = dict(report['cases']['calibration'])[worst['case']]
    captured, handles = [], []
    def hook(name):
        def observe(module, args, output):
            captured.append((name, module, args[0].detach().clone(), output.detach().clone()))
        return observe
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        observed = next(calls(model, ids, [len(ids)], canonical=False))[2]
    finally:
        for handle in handles:
            handle.remove()
    baseline = next(calls(model, ids, [len(ids)], canonical=False))[2]
    if len(captured) != 169 or any(not exact_array(baseline[n], observed[n]) for n in baseline):
        raise ValueError('observation altered the reference or missed an affine operation')
    with calibration.split_linear():
        changed = next(calls(model, ids, [len(ids)], canonical=False))[2]
    reproduced = []
    for name in sorted(baseline):
        value = calibration.metrics(changed[name], baseline[name], report['declaration']['rtol'])
        old = next(r for r in rows if r['case'] == worst['case'] and r['arm'] == worst['arm']
                   and r['schedule'] == worst['schedule'] and r['stage'] == name)
        if any(value[k] != old[k] for k in ('max_abs', 'required_atol', 'relative_rms')):
            raise ValueError('exposed calibration result did not reproduce')
        reproduced.append(dict(stage=name, **value))
    operations, witnesses = [], []
    for name, module, x, expected in captured:
        with calibration.split_linear() as count:
            actual = module(x)
        if count[0] != 1 or actual.shape != expected.shape or actual.dtype != torch.bfloat16:
            raise ValueError('invalid local affine replacement')
        a, b = actual.float().numpy().reshape(-1, actual.shape[-1]), expected.float().numpy().reshape(-1, expected.shape[-1])
        record = dict(module=name, shape=list(actual.shape), different=int(np.count_nonzero(a != b)),
                      **calibration.metrics(a, b, report['declaration']['rtol']))
        operations.append(record)
        if record['different'] and len(witnesses) < 16:
            # Freeze at most the first 16 differing affine operations in
            # execution order, and the first differing element in each.
            row, column = map(int, np.argwhere(a != b)[0])
            operands = x.reshape(-1, x.shape[-1])[row].float().tolist()
            weights = module.weight[column].float().tolist()
            exact = sum((Fraction(v)*Fraction(w) for v, w in zip(operands, weights, strict=True)), Fraction())
            if module.bias is not None:
                exact += Fraction(float(module.bias[column]))
            hf, split = Fraction(float(b[row,column])), Fraction(float(a[row,column]))
            dh, ds = abs(hf-exact), abs(split-exact)
            witnesses.append(dict(module=name, row=row, column=column,
                exact_numerator=str(exact.numerator), exact_denominator=str(exact.denominator),
                hf=float(hf), split=float(split), hf_absolute_error=float(dh), split_absolute_error=float(ds),
                nearer='hf' if dh < ds else 'split' if ds < dh else 'tie'))
    return dict(qualification_sha256=reference.sha(qualification), worst_record=worst,
        ids=ids, observation_exact=True, reproduced_boundaries=reproduced,
        operations=operations, exact_dot_witnesses=witnesses,
        calibration_source=calibration.fast_identity(), confirmation_executed=False,
        scope='169 identical-operand affine checks and at most 16 exact dot witnesses on one exposed reference case; no native results, no changed gates')


def runtime_propagation(model, native_result, reference_directory):
    """Explain the exposed early-layer amplification using HF on native operands."""
    from model_consistency import calls
    report=json.loads(native_result.read_text())
    manifest_path=reference_directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if report['reference_sha256']!=reference.sha(manifest_path):
        raise ValueError('native/reference binding mismatch')
    worst=max((r for r in report['diagnostics'] if r['comparison']=='hf_same_history'
        and r['mode']=='full' and r['configuration']==0 and r['stage'].startswith('hidden_')),key=lambda r:r['max_abs'])
    case=next(c for c in manifest['cases'] if c['name']==worst['case'])
    ids=case['ids'];length=len(ids)
    baseline=next(calls(model,ids,[length],canonical=False))[2]
    for name,record in case['modes']['full'][0]['arrays'].items():
        path=reference_directory/record['path']
        if reference.sha(path)!=record['sha256'] or baseline[name].tobytes()!=np.load(path,allow_pickle=False).tobytes():
            raise ValueError('unmodified HF capture did not reproduce')
    native={};hashes={}
    for i in range(5):
        path=native_result.parent/case['name']/'full-0/call_0'/f'hidden_{i}.bin'
        data=np.fromfile(path,dtype='<u2')
        if data.size!=length*896: raise ValueError('incomplete native propagation input')
        native[i]=(data.astype(np.uint32)<<16).view(np.float32).reshape(length,896)
        hashes[f'hidden_{i}']=reference.sha(path)
    def metrics(a,b):
        a,b=a.astype(np.float64),b.astype(np.float64)
        norm=np.linalg.norm(b,axis=1);error=np.linalg.norm(a-b,axis=1)
        relative=np.divide(error,norm,out=np.zeros_like(error),where=norm!=0)
        return dict(max_abs=float(np.abs(a-b).max()),max_row_relative_l2=float(relative.max()),
            row_relative_l2=relative.tolist(),zero_reference_nonzero_rows=int(np.sum((norm==0)&(error!=0))))
    rows=[]
    for i in range(4):
        seen=[0]
        def replace(module,args):
            seen[0]+=1
            return (torch.from_numpy(native[i]).bfloat16().unsqueeze(0),*args[1:])
        hook=model.model.layers[i].register_forward_pre_hook(replace)
        try:
            injected=next(calls(model,ids,[length],canonical=False))[2]
        finally:
            hook.remove()
        if seen[0]!=1: raise ValueError('incomplete decoder input replacement')
        expected=baseline[f'hidden_{i+1}'];propagated=injected[f'hidden_{i+1}']
        rows.append(dict(layer=i,input_difference=metrics(native[i],baseline[f'hidden_{i}']),
            observed_output_difference=metrics(native[i+1],expected),
            hf_propagated_difference=metrics(propagated,expected),
            identical_input_residual=metrics(native[i+1],propagated)))
    return dict(native_result_sha256=reference.sha(native_result),reference_manifest_sha256=reference.sha(manifest_path),
        selected_record=worst,ids=ids,native_input_sha256=hashes,baseline_reproduced_boundaries=75,layers=rows,
        scope='First four layers of the exposed full-call maximum; replace each HF layer input with the recorded native input. No arithmetic or threshold changes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--detail', action='store_true', help='trace rounding, propagation and prediction impact')
    modes.add_argument('--backend', action='store_true', help='localize ATen operations and test fixed query execution')
    modes.add_argument('--native-operations', action='store_true', help='export identical upstream operands for the exposed one-token native case')
    modes.add_argument('--fast-affine', type=Path, help='diagnose the already exposed failed Fast qualification result.json')
    modes.add_argument('--runtime-propagation',type=Path,help='diagnose the exposed native early-layer amplification')
    parser.add_argument('--runtime-reference',type=Path)
    parser.add_argument('--download-sources', action='store_true', help='download hash-pinned upstream sources for --backend')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error('--output is required')
    if args.download_sources and not args.backend:
        parser.error('--download-sources requires --backend')
    if args.output.exists():
        raise ValueError('refusing to replace diagnosis evidence')
    reference.verify_assets()
    source = reference.sha(__file__)
    provenance = reference.provenance()
    ids = np.random.default_rng(9120).integers(0, 151643, size=17).tolist()
    model = reference.load_model()
    if args.runtime_propagation:
        if not args.runtime_reference: parser.error('--runtime-propagation requires --runtime-reference')
        result=runtime_propagation(model,args.runtime_propagation,args.runtime_reference)
        if source!=reference.sha(__file__) or provenance!=reference.provenance():
            raise ValueError('source changed during propagation diagnosis')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(kind='model-runtime-propagation-v1',source_sha256=source,
            reference=provenance,reserved_outputs_observed=False,native_outputs_observed=True,result=result),indent=2,allow_nan=False)+'\n')
        return
    if args.fast_affine:
        result = fast_affine_diagnosis(model, args.fast_affine)
        if source != reference.sha(__file__) or provenance != reference.provenance():
            raise ValueError('reference source changed during Fast diagnosis')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(source_sha256=source, reference=provenance,
            new_candidate_outputs_observed=False, reserved_outputs_observed=False,
            fast_affine=result), indent=2, allow_nan=False)+'\n')
        return
    if args.native_operations:
        native_operations(model,args.output)
        return
    if args.backend:
        import model_attention_diagnosis as backend
        result = backend.run(model, traced_forward, difference, args.download_sources)
        if source != reference.sha(__file__) or provenance != reference.provenance():
            raise ValueError('reference entrypoint changed during backend diagnosis')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(source_sha256=source, reference=provenance,
            candidate_outputs_observed=False, reserved_outputs_observed=False,
            backend=result), indent=2, allow_nan=False) + '\n')
        return
    if args.detail:
        declaration_hash = reference.sha(DETAIL_CONTRACT)
        result = detail(model)
        if (source != reference.sha(__file__) or provenance != reference.provenance()
                or declaration_hash != reference.sha(DETAIL_CONTRACT)):
            raise ValueError('diagnosis source or declaration changed during execution')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(source_sha256=source, reference=provenance,
            candidate_outputs_observed=False, reserved_outputs_observed=False,
            detail=result), indent=2, allow_nan=False) + '\n')
        return
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
