"""Fixed decoder baseline geometry, workload identities and fixture transport."""
import gzip
import hashlib
import json
from pathlib import Path

from .._repository import repository_root

OPERATION = 'decoder_layer'
VARIANTS = {0,1,2,3,4,8,12,14}
# VARIANTS is the frozen historical selection registry.
MEASUREMENT_VARIANTS = VARIANTS | {20}
ENTRYPOINTS = {f'decoder_layer_{v}':'enqueue_decoder_layer' for v in MEASUREMENT_VARIANTS}
NAMES = {0:'integrated control',1:'both 16x16 attention projections',
         2:'split8 attention',3:'split8 + both 16x16 projections',4:'rowwise MLP',
         8:'combined gate/up',12:'cooperative down G4',14:'combined gate/up + down G4',
         20:'consistent G32 attention and rowwise projections'}
POLICY_PATH = 'tests/fixtures/decoder_policies.json'


def policy_declaration():
    return json.loads((repository_root()/POLICY_PATH).read_text())


def policy_schedules(rows):
    """Finite declared schedules; each call sees its own absolute causal prefix."""
    result={'policy_repeat':[(0,rows)], 'policy_tokenwise':[(p,1) for p in range(rows)]}
    calls=[];p=0
    for size in policy_declaration()['schedules']['irregular_chunks']:
        if p==rows:break
        size=min(size,rows-p);calls.append((p,size));p+=size
    if p<rows:calls.append((p,rows-p))
    result['policy_irregular']=calls
    return result
SELECTION_PATH = 'studies/decoder_layer/selection-declaration.json'
SELECTION_PROMPT = ('A train travels 60 kilometers in 45 minutes. Explain its average '
                    'speed in kilometers per hour and why the units matter.')
SELECTION_SEEDS = (6011,6029)
SELECTION_ROWS = (1,17,257,4096)
SCREEN_GRIDS = {
    'full': ([(r,r) for r in (16,64,256,1024,4096)],[0,1]),
    'short': ([(r,r) for r in (7,15,16,17)],[0,4]),
    'cached': ([(16,256)]+[(r,t) for r in (16,64,256) for t in (1024,4096)],[0,1,2,3]),
    'decode': ([(1,t) for t in (64,256,1024,4096)],[0,8,12,14]),
}
NEIGHBORS = {'full':[(257,257),(1023,1023)],'short':[],
             'cached':[(15,256),(17,256),(65,4096),(255,4096)],'decode':[(1,257),(1,4095)]}
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
    if type(variant)is not int or variant not in MEASUREMENT_VARIANTS or type(r)is not int or type(t)is not int or not 1<=r<=t<=4096:
        raise ValueError('invalid decoder baseline route or shape')
    return dict(profile_rows=r,hidden_size=896,key_value_rows=t,query_heads=14,key_value_heads=2,
                intermediate_size=4864,mlp_mapping=mappings(variant,r)[2],
                profile_workload=f'decoder-r{r}-t{t}-v{variant}',dispatches_per_iteration=len(stages(variant,r)))


def configuration(data):
    if data.get('implementation') not in ENTRYPOINTS or data.get('entrypoint')!='enqueue_decoder_layer':
        raise ValueError('decoder implementation identity changed')
    variant=int(data['implementation'].rsplit('_',1)[1])
    spec=specification(variant,data.get('profile_rows'),data.get('key_value_rows'))
    if any(data.get(k)!=v or type(data.get(k))is not type(v) for k,v in spec.items()):
        raise ValueError('decoder profile configuration mismatch')
    n,w=data.get('profile_iterations'),data.get('profile_warmup_iterations')
    if type(n)is not int or not 1<=n*spec['dispatches_per_iteration']<=5000 or type(w)is not int or not 0<=w<=100:
        raise ValueError('decoder profile dispatch/warmup budget exceeded')
    return spec


def mappings(variant,rows):
    if type(variant)is not int or variant not in MEASUREMENT_VARIANTS or type(rows)is not int or rows<1:
        raise ValueError('invalid decoder configuration')
    if variant==20:return (5,0,0)
    return (4 if variant in (2,3) else 0,5 if variant in (1,3) else 0,
            variant if rows==1 and variant in (8,12,14) else 0 if rows==1 or variant==4 else 7)


def stages(variant,rows):
    result=list(STAGES)
    if variant==20:
        result[1:3]=['Q projection','K projection','V projection']
    if variant in (2,3) and rows>1:
        result[6:7]=['FP32 GQA split','FP32 GQA merge']
    if variant in (8,14) and rows==1:
        index=result.index('gate projection')
        result[index:index+2]=['gate/up projections']
    return result


def selection_declaration():
    record=json.loads((repository_root()/SELECTION_PATH).read_text())
    expected=dict(variants=sorted(VARIANTS),seeds=list(SELECTION_SEEDS),rows=list(SELECTION_ROWS),
                  prompt=SELECTION_PROMPT,screen_grids=json.loads(json.dumps(SCREEN_GRIDS)),
                  neighbors=json.loads(json.dumps(NEIGHBORS)),
                  reference_sha256=sha(repository_root()/'tests/fixtures/decoder_layer/checksums.json'))
    # This is the frozen declaration-time assertion, not current observation state.
    if record.get('reserved_outputs_observed') is not False or any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('decoder selection declaration changed')
    import struct
    ids=record['checkpoint_token_ids']
    if not ids or hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()!=record['checkpoint_token_ids_sha256']:
        raise ValueError('decoder selection token identity changed')
    return record


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


def screen_decision(directory,build):
    """Deterministic selection; independent modes and no promotion from calibration."""
    from .study import STUDIES,load_run
    records=[];proposals=[]
    for family in SCREEN_GRIDS:
        name='decoder_selection_'+family+'_screen'
        path,prefix=selection_run_location(directory,name)
        run,_,summary=load_run(path,prefix)
        if run['study']!=name or run['build']!=build or run['specification']!=json.loads(json.dumps(STUDIES[name])):
            raise ValueError('decoder screen identity changed')
        records.append(dict(study=name,run_sha256=sha(path/(prefix+'run.json')),samples_sha256=run['samples_sha256']))
        for r,t in SCREEN_GRIDS[family][0]:
            for layers in (1,24):
                rows=[x for x in summary if (x['query_rows'],x['rows'],x['layers'])==(r,t,layers) and x['decision']=='faster']
                winner=min(rows,key=lambda x:(x['ratio'],x['candidate']))['candidate'] if rows else 0
                proposals.append(dict(family=family,query_rows=r,rows=t,layers=layers,candidate=winner,neighbor=False))
        for r,t in NEIGHBORS[family]:
            available=SCREEN_GRIDS[family][0]
            if family=='cached':available=[w for w in available if w[1]==t]
            rr,tt=min(available,key=lambda w:(abs((w[1] if family=='decode' else w[0])-(t if family=='decode' else r)),w))
            for layers in (1,24):
                candidate=next(p['candidate'] for p in proposals if p['family']==family and (p['query_rows'],p['rows'],p['layers'])==(rr,tt,layers))
                proposals.append(dict(family=family,query_rows=r,rows=t,layers=layers,candidate=candidate,neighbor=True,from_shape=[rr,tt]))
    # Offline reconstruction binds the declaration to the measured build,
    # without requiring a source checkout or consulting its current files.
    return dict(schema=1,kind='decoder_selection',build_sha256=hashlib.sha256(json.dumps(build,sort_keys=True).encode()).hexdigest(),declaration_sha256=build['sources'][SELECTION_PATH],
        screens=records,proposals=proposals,rule='Per exact shape and mode: all four ratios below one and median reduction exceeds calibrated max(5%, self-pair deviation); lowest qualifying median ratio then ID. Nearest declared neighbor, baseline fallback.')


def confirmation_spec(family,decision):
    from .study import STUDIES
    proposals=[p for p in decision['proposals'] if p['family']==family]
    pairs=list(dict.fromkeys((p['query_rows'],p['rows']) for p in proposals))
    comparisons=[dict(query_rows=p['query_rows'],rows=p['rows'],layers=p['layers'],candidate=v)
                 for p in proposals for v in sorted({0,p['candidate']})]
    return {**STUDIES['decoder_selection_'+family+'_confirmation'],
        'workloads':[dict(query_rows=r,rows=t) for r,t in pairs],
        'candidates':sorted({c['candidate'] for c in comparisons}),'comparisons':comparisons}


def confirmed_selection(decision,directory):
    from .study import load_run
    cells=[];receipts=[];build=None
    for family in SCREEN_GRIDS:
        name='decoder_selection_'+family+'_confirmation'
        path,prefix=selection_run_location(directory,name)
        run,_,summary=load_run(path,prefix)
        if (run['study']!=name or run.get('selection')!=decision
            or run['specification']!=json.loads(json.dumps(confirmation_spec(family,decision)))):
            raise ValueError('decoder confirmation differs from frozen proposal')
        if (hashlib.sha256(json.dumps(run['build'],sort_keys=True).encode()).hexdigest()!=decision['build_sha256']
            or (build is not None and run['build']!=build)):raise ValueError('confirmation build differs')
        build=run['build']
        receipts.append(dict(study=name,run_sha256=sha(path/(prefix+'run.json')),samples_sha256=run['samples_sha256']))
        for proposal in (p for p in decision['proposals'] if p['family']==family):
            row=next(x for x in summary if all(x[k]==proposal[k] for k in ('query_rows','rows','layers','candidate')))
            accepted=proposal['candidate'] if row['decision']=='faster' else 0
            cells.append({**proposal,'accepted':accepted,'confirmation':row})
    # Short-row mechanism takes precedence only where independently confirmed.
    lookup={}
    for cell in cells:
        key=(cell['query_rows'],cell['rows'],cell['layers'])
        if key not in lookup or (cell['family']=='short' and cell['accepted']!=0):lookup[key]=cell['accepted']
    return dict(schema=1,kind='decoder_confirmed_selection',selection=decision,confirmations=receipts,
        cells=cells,lookup=[dict(query_rows=r,rows=t,layers=l,variant=v) for (r,t,l),v in sorted(lookup.items())],
        shared_lookup=[dict(query_rows=r,rows=t,variant=lookup[r,t,1] if lookup[r,t,1]==lookup[r,t,24] else 0)
                       for r,t in sorted({(r,t) for r,t,l in lookup})],
        fallback=0,scope='Exact measured cells only; hot layers=1 and ring24 layers=24 are independent boundaries. No continuous shape ranges or model throughput claim.')


def profile_selection(record):
    if record.get('kind')!='decoder_confirmed_selection' or record.get('fallback')!=0:
        raise ValueError('invalid confirmed decoder selection')
    grid=[]
    for r,t,_ in PROFILES:
        cells=[x for x in record['lookup'] if (x['query_rows'],x['rows'])==(r,t)]
        if sorted(x['layers'] for x in cells)!=[1,24]:raise ValueError('missing representative selection cells')
        variants=sorted({0,*(x['variant'] for x in cells)})
        if not set(variants)<=VARIANTS:raise ValueError('unknown accepted decoder configuration')
        grid.extend((r,t,v) for v in variants)
    return grid


def selection_run_location(directory,name):
    root=Path(directory)
    return (root/name,'') if (root/name).is_dir() else (root,name+'_')
