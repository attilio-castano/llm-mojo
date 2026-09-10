# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==1.26.4", "torch==2.4.0", "transformers==4.43.1"]
# ///
"""Pinned full Qwen reference, preparation and reference-only qualification.

Run with the shared script lock. No Mojo outputs are read here.
"""
from contextlib import contextmanager
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import transformers
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2 as qwen

ROOT = Path(__file__).resolve().parents[2]
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
ASSETS = ROOT/'build/checkpoints/qwen2.5-0.5b-instruct'/REVISION
WEIGHT_SHA = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
CONFIG_SHA = '18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45'
CONTRACT_PATH = Path(__file__).with_name('model_contract.json')
CONTRACT = json.loads(CONTRACT_PATH.read_text())
GATES = CONTRACT['initial_qualification_gates']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_assets():
    for name, expected in [('model.safetensors', WEIGHT_SHA), ('config.json', CONFIG_SHA),
            ('generation_config.json','e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6')]:
        if sha(ASSETS/name) != expected:
            raise ValueError('pinned artifact mismatch: '+name)
    if (torch.__version__, transformers.__version__, np.__version__) != ('2.4.0','4.43.1','1.26.4'):
        raise ValueError('wrong oracle environment')
    torch.set_num_threads(1)


def provenance():
    return dict(model_revision=REVISION, checkpoint_sha256=WEIGHT_SHA,
                source_sha256=sha(__file__), contract_sha256=sha(CONTRACT_PATH), upstream_sha256=sha(inspect.getfile(qwen)),
                torch=torch.__version__, transformers=transformers.__version__,
                numpy=np.__version__, device='cpu', threads=1,
                policy='BF16 stored boundaries; math SDPA FP32 Q/K/V/mask, output BF16',
                gates=GATES)


@contextmanager
def precision_policy():
    original = torch.nn.functional.scaled_dot_product_attention
    counts = [0]
    def fp32(query, key, value, **kwargs):
        counts[0] += 1
        if kwargs.get('attn_mask') is not None:
            kwargs['attn_mask'] = kwargs['attn_mask'].float()
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return original(query.float(), key.float(), value.float(), **kwargs).to(query.dtype)
    with patch.object(torch.nn.functional, 'scaled_dot_product_attention', fp32):
        yield counts


def load_model():
    model, info = Qwen2ForCausalLM.from_pretrained(ASSETS, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
    if any(info[key] for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('incomplete checkpoint loading: '+str(info))
    model.eval()
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError('checkpoint embeddings/LM head must be tied')
    return model


def bytes_bf16(value):
    return value.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()


def prepare(model, output):
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    def save(name, value, sources):
        if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError('invalid prepared tensor '+name)
        path = output/(name+'.bin')
        path.write_bytes(bytes_bf16(value))
        records[name] = dict(shape=list(value.shape), dtype='BF16', bytes=path.stat().st_size,
                             sha256=sha(path), source_tensors=sources)
    save('embedding', model.model.embed_tokens.weight, ['model.embed_tokens.weight'])
    save('final_norm', model.model.norm.weight, ['model.norm.weight'])
    for i, layer in enumerate(model.model.layers):
        prefix = f'model.layers.{i}.'
        for name, attr in [('attention_norm','input_layernorm.weight'),
                           ('mlp_norm','post_attention_layernorm.weight'),
                           ('wo','self_attn.o_proj.weight'),('gate','mlp.gate_proj.weight'),
                           ('up','mlp.up_proj.weight'),('down','mlp.down_proj.weight')]:
            value = layer
            for part in attr.split('.'):
                value = getattr(value, part)
            save(f'layer_{i}_{name}', value, [prefix+attr])
        for name, attr in [('qkv','weight'),('bias','bias')]:
            save(f'layer_{i}_{name}', torch.cat([getattr(getattr(layer.self_attn,p+'_proj'),attr)
                                               for p in ('q','k','v')]),
                 [prefix+'self_attn.'+p+'_proj.'+attr for p in ('q','k','v')])
    cos, sin = model.model.layers[0].self_attn.rotary_emb(
        torch.zeros((1,2,1,64),dtype=torch.bfloat16), seq_len=4096)
    save('cosine', cos, ['Qwen2RotaryEmbedding(seq_len=4096)'])
    save('sine', sin, ['Qwen2RotaryEmbedding(seq_len=4096)'])
    manifest = dict(**provenance(), format='qwen-model-prepared-v1', tensors=records)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('prepared', len(records), 'BF16 tensors', flush=True)


@torch.no_grad()
def forward(model, ids, schedule, all_logits=True):
    if sum(schedule) != len(ids) or min(schedule) < 1 or len(ids) > 4096:
        raise ValueError('invalid schedule')
    offset, cache, calls = 0, None, []
    for rows in schedule:
        block = torch.tensor([ids[offset:offset+rows]], dtype=torch.long)
        # Explicit 4D mask avoids upstream implicit non-square causal choices.
        positions = torch.arange(offset,offset+rows)
        mask = torch.zeros((1,1,rows,offset+rows),dtype=torch.bfloat16)
        mask.masked_fill_(torch.arange(offset+rows)[None] > positions[:,None],float('-inf'))
        last = []
        handle = model.model.layers[23].register_forward_hook(
            lambda module,args,result: last.append(result[0][0].float().numpy().copy()))
        try:
            with precision_policy() as count:
                result = model.model(input_ids=block, attention_mask=mask,
                    position_ids=positions[None], past_key_values=cache, use_cache=True,
                    output_hidden_states=True, return_dict=True)
        finally:
            handle.remove()
        if count[0] != 24:
            raise ValueError('missing full-model SDPA coverage')
        cache = result.past_key_values
        values = {f'hidden_{i}': h[0].float().numpy().copy()
                  for i,h in enumerate(result.hidden_states)}
        if len(last) != 1:
            raise ValueError('missing final decoder boundary')
        values['final_norm'] = values.pop('hidden_24')
        values['hidden_24'] = last[0]
        head_input = result.last_hidden_state if all_logits else result.last_hidden_state[:,-1:]
        values['logits'] = model.lm_head(head_input)[0].float().numpy().copy()
        for layer,(key,value) in enumerate(cache):
            values[f'cache_key_{layer}'] = key[0].transpose(0,1).float().numpy().copy()
            values[f'cache_value_{layer}'] = value[0].transpose(0,1).float().numpy().copy()
        calls.append((offset,rows,values))
        offset += rows
    return calls


def qualify(model, output):
    """Compare full and split upstream schedules before observing Mojo."""
    output.mkdir(parents=True, exist_ok=False)
    records=[]
    # Fixed IDs independent of tokenizer and model outputs. This is development
    # qualification, never the final reserved acceptance set.
    for length in CONTRACT['qualification_lengths']:
        rng=np.random.default_rng(9103+length)
        ids=rng.integers(0,151643,size=length).tolist()
        full=forward(model,ids,[length])[0][2]
        schedules=([1]*length,) if length<=17 else ([length-17,16,1],)
        for schedule in schedules:
            for start,rows,values in forward(model,ids,schedule):
                for name,actual in values.items():
                    expected = full[name][:start+rows] if name.startswith('cache_') else full[name][start:start+rows]
                    gate=GATES['logits' if name=='logits' else 'hidden']
                    scaled=np.abs(actual-expected)/(gate['atol']+gate['rtol']*np.abs(expected))
                    maximum=float(scaled.max())
                    records.append(dict(length=length,start=start,rows=rows,stage=name,
                                        scaled_error=maximum,passed=maximum<=1))
        print('qualified reference schedules',length,flush=True)
    report=dict(**provenance(), candidate_outputs_observed=False, checks=records,
                passed=all(r['passed'] for r in records))
    (output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise ValueError('reference-only schedule qualification failed; do not compare Mojo')


def capture(model, output, ids, schedule, qualification):
    output.mkdir(parents=True, exist_ok=False)
    qualified = json.loads(qualification.read_text())
    if (not qualified.get('passed') or qualified.get('gates') != GATES
            or qualified.get('source_sha256') != sha(__file__)
            or qualified.get('contract_sha256') != sha(CONTRACT_PATH)
            or qualified.get('candidate_outputs_observed') is not False):
        raise ValueError('missing compatible reference-only qualification')
    cases = {}
    for mode, chunks in [('full',[len(ids)]),('scheduled',schedule)]:
        calls=[]
        for index,(start,rows,values) in enumerate(forward(model,ids,chunks,all_logits=False)):
            arrays={}
            directory=output/mode/f'call_{index}'
            directory.mkdir(parents=True)
            for name,value in values.items():
                path=directory/(name+'.npy')
                np.save(path,value,allow_pickle=False)
                arrays[name]=dict(path=str(path.relative_to(output)),shape=list(value.shape),sha256=sha(path))
            calls.append(dict(start=start,rows=rows,arrays=arrays))
        cases[mode]=calls
    (output/'manifest.json').write_text(json.dumps(dict(**provenance(),ids=ids,
        schedule=schedule,kind='model_development_reference',cases=cases,
        qualification_sha256=sha(qualification),qualification=qualified),indent=2)+'\n')


def self_test():
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    torch.set_num_threads(1)
    torch.manual_seed(18803)
    config=Qwen2Config(hidden_size=16,intermediate_size=37,num_hidden_layers=24,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,
        max_position_embeddings=64,rope_theta=1000000.,rms_norm_eps=1e-6,
        attention_dropout=0.,use_sliding_window=False)
    config._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
    ids=[1,2,3,2,1,4,5]
    full=forward(model,ids,[7])[0][2]
    checked=0
    for start,rows,values in forward(model,ids,[3,3,1]):
        assert len(values)==75
        for name,actual in values.items():
            expected=full[name][:start+rows] if name.startswith('cache_') else full[name][start:start+rows]
            assert actual.shape==expected.shape and np.isfinite(actual).all()
            gate=GATES['logits' if name=='logits' else 'hidden']
            assert np.all(np.abs(actual-expected)<=gate['atol']+gate['rtol']*np.abs(expected)), name
            checked+=1
    print('synthetic upstream 24-layer capture self-test:',checked,'boundaries; not checkpoint qualification',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','qualify','capture','self-test'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--length',type=int,default=17)
    parser.add_argument('--seed',type=int,default=9103)
    parser.add_argument('--schedule',default='')
    parser.add_argument('--qualification',type=Path)
    args=parser.parse_args()
    if args.command=='self-test':
        self_test()
        return
    if not args.output:
        parser.error('--output is required')
    verify_assets()
    model=load_model()
    if args.command=='prepare': prepare(model,args.output)
    elif args.command=='qualify': qualify(model,args.output)
    else:
        if not 1<=args.length<=4096:
            parser.error('length must be 1..4096')
        ids=np.random.default_rng(args.seed+args.length).integers(0,151643,size=args.length).tolist()
        schedule=[int(x) for x in args.schedule.split(',')] if args.schedule else (
            [1]*args.length if args.length<=17 else [args.length-17,16,1])
        if not args.qualification:
            parser.error('capture requires --qualification from this source')
        capture(model,args.output,ids,schedule,args.qualification)


if __name__=='__main__': main()
