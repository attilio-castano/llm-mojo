"""Fixed decoder baseline geometry, workload identities and fixture transport."""
import gzip
import hashlib
import json
from pathlib import Path

from .._repository import repository_root

OPERATION = 'decoder_layer'
VARIANTS = {0}  # One caller policy: attention (0,0), MLP 7 for R>1, otherwise 0.
ENTRYPOINTS = {'decoder_layer_0': 'enqueue_decoder_layer'}
WORKLOADS = [(256,256),(4096,4096),(16,256),(64,4096),(1,256),(1,4096)]
PROFILES = [(256,256,25),(64,4096,25),(1,4096,100)]
STAGES = ['attention RMSNorm','packed QKV projection','QKV unpack','Q RoPE',
          'K RoPE','KV append','FP32 GQA','output projection','attention residual',
          'MLP RMSNorm','gate projection','up projection','SiLU','multiply',
          'down projection','MLP residual']
TARGET_FIELDS = ('profile_workload','dispatches_per_iteration','key_value_rows',
                 'query_heads','key_value_heads','intermediate_size','mlp_mapping')
CASE = 'h896_i4864_nq14_nk2_d64_t4096_s4001_base'
ARITHMETIC = 'BF16 stored boundaries/cache; FP32 reductions and SDPA; materialized BF16 SiLU, product and residual operands.'
INPUTS = 'Frozen seed 4001, original X prefix/suffix; GPU-produced cache prefix. Hot one allocation set; ring24 distinct input/weight/cache sets with identical contents and shared scratch. Untimed adversarial smoke uses 24 distinct hidden-coordinate sign patterns.'
TIMING = 'Host enqueue through completion; hot one call, ring24 one synchronization per sweep divided by 24. Prefix preparation, allocation, verification and printing excluded. Each sample overwrites the same suffix at P=T-R after the previous sample completes; not growing-context generation.'
LABELS = ['input_'+x for x in ('X','input_norm','qkv','bias','wo','post_norm','gate','up','down')]
LABELS += ['full_'+x for x in ('cosine','sine','B_att','Z','B_mlp','Y')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_identity():
    root=repository_root()
    anchor=root/'tests/fixtures/decoder_layer/checksums.json'
    identity=json.loads(anchor.read_text())
    evidence=root/'tests/fixtures/decoder_layer/development.json.gz'
    raw=evidence.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=identity['evidence_sha256']:
        raise ValueError('decoder benchmark anchor changed')
    raw=gzip.decompress(raw)
    if hashlib.sha256(raw).hexdigest()!=identity['uncompressed_sha256']:
        raise ValueError('decoder evidence changed')
    frozen=json.loads(raw)
    case=frozen['cases'][CASE]
    arrays={label:case['arrays'][label] for label in LABELS}
    for label,spec in arrays.items():
        if sha(root/'build/oracle_data/decoder_layer'/CASE/(label+'.npy'))!=spec['sha256']:
            raise ValueError('decoder benchmark array changed: '+label)
    return dict(case=CASE,reference_sha256=sha(anchor),arrays=arrays)


def specification(variant,r,t):
    if type(variant)is not int or variant!=0 or type(r)is not int or type(t)is not int or not 1<=r<=t<=4096:
        raise ValueError('invalid decoder baseline route or shape')
    return dict(profile_rows=r,hidden_size=896,key_value_rows=t,query_heads=14,key_value_heads=2,
                intermediate_size=4864,mlp_mapping=7 if r>1 else 0,
                profile_workload=f'decoder-r{r}-t{t}-v0',dispatches_per_iteration=16)


def configuration(data):
    if data.get('implementation')!='decoder_layer_0' or data.get('entrypoint')!='enqueue_decoder_layer':
        raise ValueError('decoder implementation identity changed')
    spec=specification(0,data.get('profile_rows'),data.get('key_value_rows'))
    if any(data.get(k)!=v or type(data.get(k))is not type(v) for k,v in spec.items()):
        raise ValueError('decoder profile configuration mismatch')
    n,w=data.get('profile_iterations'),data.get('profile_warmup_iterations')
    if type(n)is not int or not 1<=n*16<=5000 or type(w)is not int or not 0<=w<=100:
        raise ValueError('decoder profile dispatch/warmup budget exceeded')
    return spec


def _array(label):
    import numpy as np
    return np.load(repository_root()/'build/oracle_data/decoder_layer'/CASE/(label+'.npy'),allow_pickle=False)


def signs(layer):
    import numpy as np
    return np.where(((np.arange(896)*(layer+1)+layer)%31)<15,-1,1).astype(np.float32)


def load(address,label,count,adversarial,layer):
    import ctypes
    import numpy as np
    data=_array(label)
    if adversarial:
        sign=signs(layer)
        if label in ('input_X','input_qkv','input_gate','input_up'):
            data=data*sign
        elif label in ('input_wo','input_down'):
            data=data*sign[:,None]
    bits=(data.reshape(-1)[:count].view(np.uint32)>>16).astype(np.uint16)
    if bits.size!=count:
        raise ValueError('decoder benchmark load extent mismatch')
    dst=np.ctypeslib.as_array((ctypes.c_uint16*count).from_address(address))
    dst[:]=bits


def check(address,stage,start,rows,adversarial,layer):
    import ctypes
    import numpy as np
    bits=np.ctypeslib.as_array((ctypes.c_uint16*(rows*896)).from_address(address))
    actual=(bits.astype(np.uint32)<<16).view(np.float32).reshape(rows,896)
    expected=_array('full_'+stage).reshape(4096,896)[start:start+rows]
    if adversarial:
        expected=expected*signs(layer)
    error=np.abs(actual.astype(np.float64)-expected)/(1+np.abs(expected))
    if not np.isfinite(error).all() or np.any(error>2**-5):
        raise ValueError('decoder benchmark numerical gate failed: '+stage)
