"""Observe the actual pinned decoder; no Mojo or held-out evaluation here."""
from contextlib import ExitStack
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import struct
from unittest.mock import patch

import numpy as np
import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2 import modeling_qwen2 as qwen
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from contract import STAGES, QWEN, SYSTEM, HOLDOUT_PROMPT, case_spec

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('decoder_bf16_numerics', HERE.parent/'mlp/numerics.py')
_numerics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_numerics)
bf16_bits, differences = _numerics.bf16_bits, _numerics.differences


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor(a):
    return torch.from_numpy(np.array(a, copy=True)).to(torch.bfloat16)


def array(t):
    if t.dtype != torch.bfloat16 or t.device.type != 'cpu':
        raise ValueError('capture requires CPU BF16 tensors')
    return t.detach().float().numpy().copy()


def same_bits(a, b):
    return a.shape == b.shape and np.array_equal(bf16_bits(a), bf16_bits(b))


def inputs(spec):
    h,nq,nk,d,i,t,seed = (spec[k] for k in ('h','nq','nk','d','i','rows','seed'))
    values = {}
    for tag,(name,shape,scale,offset) in enumerate((
        ('X',(t,h),1,0), ('input_norm',(h,),1/64,1),
        ('qkv',(h+2*nk*d,h),1/np.sqrt(h),0), ('bias',(h+2*nk*d,),1/64,0),
        ('wo',(h,h),1/np.sqrt(h),0), ('post_norm',(h,),1/64,1),
        ('gate',(i,h),1/np.sqrt(h),0), ('up',(i,h),1/np.sqrt(h),0),
        ('down',(h,i),1/np.sqrt(i),0),
    )):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,h,i,nq,nk,d,tag])))
        values[name] = array(tensor((rng.standard_normal(shape)*scale+offset).astype(np.float32)))
    mutation = spec['mutation']
    if mutation.startswith('zero_'):
        for name in {'zero_x':['X'], 'zero_wo':['wo'], 'zero_down':['down'], 'zero_branches':['wo','down']}[mutation]:
            values[name].fill(0)
    elif mutation in ('swap_gate_up', 'swap_norms'):
        a,b = ('gate','up') if mutation == 'swap_gate_up' else ('input_norm','post_norm')
        values[a],values[b] = values[b],values[a]
    elif mutation in ('small_x','large_x'):
        values['X'] = array(tensor(values['X']*(1/64 if mutation == 'small_x' else 16)))
    elif mutation != 'base':
        raise ValueError('unknown fixture mutation')
    return values


def check_capture(c, spec, p, r):
    if set(c) != set(STAGES):
        raise ValueError('missing or extra decoder boundary')
    h,nq,nk,d,i = (spec[k] for k in ('h','nq','nk','d','i'))
    shapes = {name:(r,h) for name in STAGES}
    shapes.update({name:(r,i) for name in ('G','U','A','S')})
    shapes.update({name:(r,nk*d) for name in ('K_raw','V_raw')})
    shapes.update(Q=(r,nq,d), K_rot=(r,nk,d), O=(r,nq,d),
                  cosine=(p+r,d), sine=(p+r,d),
                  cache_key=(p+r,nk,d), cache_value=(p+r,nk,d))
    for name, a in c.items():
        if a.shape != shapes[name] or not np.isfinite(a).all():
            raise ValueError('invalid captured boundary '+name)
        bf16_bits(a)
    for name,expected in (
        ('Z', array(tensor(c['X'])+tensor(c['B_att']))),
        ('S', array(tensor(c['A'])*tensor(c['U']))),
        ('Y', array(tensor(c['Z'])+tensor(c['B_mlp']))),
    ):
        if not same_bits(c[name],expected):
            raise ValueError('captured residual/gating operands do not reconstruct '+name)
    if not same_bits(c['cache_key'][p:],c['K_rot']) or not same_bits(c['cache_value'][p:],c['V_raw'].reshape(r,nk,d)):
        raise ValueError('upstream cache append does not match produced K/V')


class UpstreamDecoder:
    def __init__(self, spec, data):
        self.spec = spec
        h,nq,nk,d,i = (spec[k] for k in ('h','nq','nk','d','i'))
        config = Qwen2Config(hidden_size=h, intermediate_size=i,
            num_attention_heads=nq, num_key_value_heads=nk, hidden_act='silu',
            rms_norm_eps=1e-6, max_position_embeddings=4096, rope_theta=1000000.,
            attention_dropout=0., use_sliding_window=False, sliding_window=None)
        config._attn_implementation = 'sdpa'
        self.module = qwen.Qwen2DecoderLayer(config, layer_idx=0).to(torch.bfloat16).eval()
        if type(self.module.self_attn) is not qwen.Qwen2SdpaAttention:
            raise ValueError('decoder did not select the SDPA implementation')
        with torch.no_grad():
            self.module.input_layernorm.weight.copy_(tensor(data['input_norm']))
            self.module.post_attention_layernorm.weight.copy_(tensor(data['post_norm']))
            for name,a,b in [('q',0,h),('k',h,h+nk*d),('v',h+nk*d,h+2*nk*d)]:
                projection = getattr(self.module.self_attn,name+'_proj')
                projection.weight.copy_(tensor(data['qkv'][a:b]))
                projection.bias.copy_(tensor(data['bias'][a:b]))
            self.module.self_attn.o_proj.weight.copy_(tensor(data['wo']))
            for name in ('gate','up','down'):
                getattr(self.module.mlp,name+'_proj').weight.copy_(tensor(data[name]))
        self.cache = DynamicCache()
        self.sdpa_calls = 0

    @torch.no_grad()
    def run(self, values, observe=True):
        r,h = values.shape
        p = self.cache.get_seq_length()
        if h != self.spec['h'] or r < 1 or p+r > 4096:
            raise ValueError('invalid decoder input/cache extent')
        x = tensor(values)[None]
        positions = torch.arange(p,p+r)[None]
        mask = torch.zeros((1,1,r,p+r),dtype=torch.bfloat16)
        mask.masked_fill_(torch.arange(p+r)[None] > positions[0,:,None], float('-inf'))
        c = dict(X=array(x[0])) if observe else {}
        counts = {}
        original_rope = qwen.apply_rotary_pos_emb
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        def fp32_sdpa(q,k,v,**kwargs):
            self.sdpa_calls += 1
            if kwargs.get('attn_mask') is not None:
                kwargs['attn_mask'] = kwargs['attn_mask'].float()
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return original_sdpa(q.float(),k.float(),v.float(),**kwargs).to(q.dtype)
        def rotation(q,k,cos,sin,position_ids,unsqueeze_dim=1):
            qr,kr = original_rope(q,k,cos,sin,position_ids,unsqueeze_dim)
            if observe:
                c.update(Q=array(qr[0].transpose(0,1)), K_rot=array(kr[0].transpose(0,1)),
                         cosine=array(cos), sine=array(sin))
                counts['rope'] = counts.get('rope',0)+1
            return qr,kr
        def hook(name, before=False, attention=False):
            def capture(module,args,result=None):
                value = args[0] if before else result
                value = array(value[0])
                c[name] = value.reshape(r,self.spec['nq'],self.spec['d']) if attention else value
                counts[name] = counts.get(name,0)+1
            return capture
        before_calls = self.sdpa_calls
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.nn.functional,'scaled_dot_product_attention',fp32_sdpa))
            stack.enter_context(patch.object(qwen,'apply_rotary_pos_emb',rotation))
            if observe:
                modules = [('N_att',self.module.input_layernorm),('N_mlp',self.module.post_attention_layernorm),
                           ('B_att',self.module.self_attn.o_proj)]
                modules += [(name,getattr(self.module.self_attn,attr+'_proj')) for name,attr in [('Q_raw','q'),('K_raw','k'),('V_raw','v')]]
                modules += [(name,getattr(self.module.mlp,attr+'_proj')) for name,attr in [('G','gate'),('U','up'),('B_mlp','down')]]
                modules += [('A',self.module.mlp.act_fn)]
                for name,module in modules:
                    stack.callback(module.register_forward_hook(hook(name)).remove)
                for name,module in [('Z',self.module.post_attention_layernorm),('S',self.module.mlp.down_proj),('O',self.module.self_attn.o_proj)]:
                    stack.callback(module.register_forward_pre_hook(hook(name,before=True,attention=name=='O')).remove)
            result = self.module(x, attention_mask=mask, position_ids=positions,
                                 past_key_value=self.cache, use_cache=True, cache_position=positions[0])[0]
        if self.sdpa_calls != before_calls+1 or (observe and (len(counts)!=14 or any(n!=1 for n in counts.values()))):
            raise ValueError('missing, duplicate or bypassed decoder observation')
        c.update(Y=array(result[0]), cache_key=array(self.cache.key_cache[0][0].transpose(0,1)),
                 cache_value=array(self.cache.value_cache[0][0].transpose(0,1)))
        if observe:
            check_capture(c,self.spec,p,r)
        return c


def provenance():
    versions = (torch.__version__,transformers.__version__,np.__version__)
    if versions != ('2.4.0','4.43.1','1.26.4') or torch.get_num_threads()!=1:
        raise ValueError('wrong pinned decoder reference environment')
    return dict(torch=versions[0],transformers=versions[1],numpy=versions[2],
                python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
                implementation='Qwen2DecoderLayer', source_sha256=sha(inspect.getfile(qwen)),
                device='cpu', backend='SDPBackend.MATH', threads=1, storage='bfloat16',
                policy='FP32 Q/K/V/mask at actual SDPA; BF16 output before Wo',
                training_arithmetic_reproduced=False, torch_build=torch.__config__.show())


def checkpoint(directory):
    """Read the existing independently hashed prefix; never download or mutate."""
    directory = Path(directory)
    authority = json.loads((HERE.parent/'attention_sublayer/checkpoint_checksums.json').read_text())
    assets={k:v for k,v in authority['asset_sha256'].items() if k!='model.safetensors'}
    for name,digest in assets.items():
        if sha(directory/name)!=digest:
            raise ValueError('checkpoint asset changed: '+name)
    path = directory/'model.attention-prefix.bin'
    if path.stat().st_size != 302126368 or sha(path) != '0d3c86fcaa9573dbac31055974018e4d1a94a07124e5d124747feed78b51f6fa':
        raise ValueError('checkpoint prefix identity mismatch')
    identities = {}
    with path.open('rb') as f:
        n = struct.unpack('<Q',f.read(8))[0]
        header = json.loads(f.read(n))
        def load(name, shape):
            spec = header[name]; a,b = spec['data_offsets']
            if spec['dtype']!='BF16' or spec['shape']!=list(shape) or not 0<=a<=b<=path.stat().st_size-n-8 or b-a!=2*int(np.prod(shape)):
                raise ValueError('invalid/incomplete checkpoint tensor: '+name)
            f.seek(n+8+a); raw = bytearray(f.read(b-a))
            if len(raw)!=b-a:
                raise ValueError('truncated checkpoint tensor')
            identities[name] = dict(shape=list(shape),dtype='BF16',sha256=hashlib.sha256(raw).hexdigest(),byte_range=[n+8+a,n+8+b])
            return torch.frombuffer(raw,dtype=torch.bfloat16).reshape(shape).clone()
        prefix = 'model.layers.0.'
        data = {k:array(load(prefix+suffix,shape)) for k,suffix,shape in (
            ('input_norm','input_layernorm.weight',(896,)), ('post_norm','post_attention_layernorm.weight',(896,)),
            ('wo','self_attn.o_proj.weight',(896,896)), ('gate','mlp.gate_proj.weight',(4864,896)),
            ('up','mlp.up_proj.weight',(4864,896)), ('down','mlp.down_proj.weight',(896,4864)))}
        data['qkv'] = array(torch.cat([load(prefix+f'self_attn.{k}_proj.weight',(out,896)) for k,out in [('q',896),('k',128),('v',128)]]))
        data['bias'] = array(torch.cat([load(prefix+f'self_attn.{k}_proj.bias',(out,)) for k,out in [('q',896),('k',128),('v',128)]]))
        embedding = load('model.embed_tokens.weight',(151936,896))
    tokenizer = transformers.AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    def tokens(prompt):
        return tokenizer.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
    cases = []
    for idx,old in enumerate(authority['prompts']):
        ids = tokens(old['prompt'])[:4096]
        if hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()!=old['token_ids_sha256']:
            raise ValueError('checkpoint development token IDs changed')
        item = dict(data,X=array(torch.nn.functional.embedding(torch.tensor(ids),embedding)))
        cases.append((f'checkpoint_{idx}',case_spec(QWEN,len(ids),0),item,ids))
    held_ids = tokens(HOLDOUT_PROMPT)  # Declaration only: no held-out decoder execution.
    return cases, dict(model=authority['model'], revision=authority['revision'],
        source=dict(prefix_sha256=sha(path),prefix_bytes=path.stat().st_size,full_file_sha256_verified=False),
        asset_sha256=assets, tensors=identities,
        holdout_token_ids=held_ids,holdout_token_ids_sha256=hashlib.sha256(np.asarray(held_ids,dtype='<i8').tobytes()).hexdigest())
