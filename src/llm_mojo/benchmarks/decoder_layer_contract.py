"""Fixed decoder baseline geometry, workload identities and fixture transport."""
import gzip
import hashlib
import json
from pathlib import Path

from .._repository import repository_root

OPERATION = 'decoder_layer'
VARIANTS = {0,1,2,3,4,8,12,14}
# VARIANTS is the frozen historical selection registry.
MEASUREMENT_VARIANTS = VARIANTS | {20,21,22,23,24}
# Numerical harness IDs for the public lookup: Fast/Deterministic, hot/ring24.
POLICY_EXECUTIONS = {100:(False,1),101:(True,1),102:(False,24),103:(True,24)}
POLICY_TEST_VARIANTS = MEASUREMENT_VARIANTS | POLICY_EXECUTIONS.keys()
ENTRYPOINTS = {f'decoder_layer_{v}':'enqueue_decoder_layer' for v in MEASUREMENT_VARIANTS}
NAMES = {0:'integrated control',1:'both 16x16 attention projections',
         2:'split8 attention',3:'split8 + both 16x16 projections',4:'rowwise MLP',
         8:'combined gate/up',12:'cooperative down G4',14:'combined gate/up + down G4',
         20:'consistent G32 attention and rowwise projections',
         21:'consistent G32 and fixed MMA projections',
         22:'consistent G32 and four-row weight reuse',
         23:'consistent G32 and eight-row weight reuse',
         24:'consistent G32 and sixteen-row weight reuse'}
POLICY_PATH = 'tests/fixtures/decoder_policies.json'


def policy_declaration():
    return json.loads((repository_root()/POLICY_PATH).read_text())


def policy_profile_grid(spec):
    declared=spec['policies']
    if sha_json(declared)!=spec.get('policies_sha256'):
        raise ValueError('decoder policy profile declaration changed')
    grid=[(r,t,v) for r,t,v,n in declared['profiles']]
    if len(set(grid))!=len(grid) or sorted(map(list,grid))!=sorted(map(list,spec.get('captures',[]))):
        raise ValueError('decoder policy profile census changed')
    return grid


def sha_json(value):
    return hashlib.sha256((json.dumps(value,indent=2)+'\n').encode()).hexdigest()


def policy_schedules(rows,declaration=None):
    """Finite declared schedules; each call sees its own absolute causal prefix."""
    declaration=policy_declaration() if declaration is None else declaration
    declared=declaration['schedules']
    result={'policy_repeat':[(0,rows)], 'policy_tokenwise':[(p,1) for p in range(rows)]}
    calls=[];p=0
    for size in declared['irregular_chunks']:
        if p==rows:break
        size=min(size,rows-p);calls.append((p,size));p+=size
    if p<rows:calls.append((p,rows-p))
    result['policy_irregular']=calls
    if 'lookup_chunks' in declared:
        calls=[];p=0
        for size in declared['lookup_chunks']:
            if p==rows:break
            size=min(size,rows-p);calls.append((p,size));p+=size
        if p<rows:calls.append((p,rows-p))
        result['policy_lookup']=calls
    if 'prefill_prefix_rows' in declared:
        prefix=min(rows,declared['prefill_prefix_rows'])
        result['policy_prefix']=[(0,prefix)]+([(prefix,rows-prefix)] if prefix<rows else [])
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
    if variant==21:return (5,6,7)
    if variant>=22:return (5,variant-15,variant-3)
    return (4 if variant in (2,3) else 0,5 if variant in (1,3) else 0,
            variant if rows==1 and variant in (8,12,14) else 0 if rows==1 or variant==4 else 7)



# Loaded once; archived numerical replay can provide its own frozen table.
_POLICY_LOOKUP=policy_declaration()['accepted_lookup']


def policy_configuration(deterministic,rows,total_rows,layers=1,lookup=None):
    """Mirror the native lookup so recorded routes expose selector drift."""
    if type(deterministic)is not bool or type(rows)is not int or type(total_rows)is not int or not 1<=rows<=total_rows<=4096 or type(layers)is not int or layers not in (1,24):
        raise ValueError('invalid decoder policy workload')
    lookup=_POLICY_LOOKUP if lookup is None else lookup
    policy='deterministic' if deterministic else 'fast'
    for cell in lookup['cells']:
        if (cell['query_rows'],cell['rows'],cell['layers'])==(rows,total_rows,layers):
            return cell[policy]
    return lookup['fallbacks'][policy]


def execution_mappings(variant,rows,total_rows,lookup=None):
    if variant in POLICY_EXECUTIONS:
        deterministic,layers=POLICY_EXECUTIONS[variant]
        variant=policy_configuration(deterministic,rows,total_rows,layers,lookup)
    return mappings(variant,rows)


def select_policy_round1(summaries,invariant_variants,compatible_family,declaration=None):
    """Apply the frozen first-round rule to complete, reconstructed summaries."""
    declaration=policy_declaration() if declaration is None else declaration
    declared=declaration['rounds'][0]
    if declared['round']!=1 or declared['new_candidates']!=[21,22]:
        raise ValueError('unsupported decoder policy round')
    screens={screen['name']:screen for screen in declared['screens']}
    if set(summaries)!=set(screens):raise ValueError('missing policy screen')
    tables={}
    for name,screen in screens.items():
        rows=summaries[name]
        table={(x['query_rows'],x['rows'],x['layers'],x['candidate']):x for x in rows}
        expected={(r,t,l,v) for r,t in screen['workloads']
                  for l in declaration['modes'] for v in screen['candidates']}
        if len(table)!=len(rows) or set(table)!=expected:
            raise ValueError('incomplete policy screen comparison census')
        tables[name]=table
    det=tables['decoder_policies_round1_det']
    cells=[(r,t,l) for r,t in screens['decoder_policies_round1_det']['workloads']
           for l in declaration['modes']]
    compatible=compatible_family is True and {20,22}<=set(invariant_variants)
    use22={cell:compatible and det[*cell,22]['decision']=='faster' for cell in cells}
    use21=21 in invariant_variants and all(
        det[*cell,21]['decision']=='faster' and (
            not use22[cell] or det[*cell,21]['ratio_max'] <
            (1-det[*cell,21]['noise_floor'])*det[*cell,22]['ratio_min']) for cell in cells)
    proposals=[]
    for r,t,l in cells:
        control=next(v for rr,tt,v in declaration['workloads'] if (r,t)==(rr,tt))
        fast=tables[f'decoder_policies_round1_fast_{control}']
        eligible=[fast[r,t,l,v] for v in (21,22) if fast[r,t,l,v]['decision']=='faster']
        winner=min(eligible,key=lambda x:(x['ratio'],x['candidate']))['candidate'] if eligible else control
        proposals.append(dict(query_rows=r,rows=t,layers=l,fast=winner,
            deterministic=21 if use21 else 22 if use22[r,t,l] else 20))
    return dict(round=1,proposals=proposals,deterministic_family=[21] if use21 else [20,22] if compatible else [20],
        deterministic_fallback=21 if use21 else 20,
        fixed_mma_global_qualified=use21,compatible_row_reuse=compatible)


def select_policy_round2(summaries,invariant_variants,compatible_family,declaration=None):
    """Compare each new tile directly with its frozen incumbent, per mode."""
    import copy
    declaration=policy_declaration() if declaration is None else declaration
    declared=declaration['rounds'][1]
    incumbent=declared['incumbent_decision']
    if (declared['round']!=2 or declared['new_candidates']!=[23,24]
        or sha_json(incumbent)!=declared['incumbent_decision_sha256']):
        raise ValueError('round-two incumbent or candidate declaration changed')
    screens={screen['name']:screen for screen in declared['screens']}
    if set(summaries)!=set(screens):raise ValueError('missing policy screen')
    proposals=copy.deepcopy(incumbent['proposals'])
    cells={(p['query_rows'],p['rows'],p['layers']):p for p in proposals}
    visited=set()
    compatible=compatible_family is True and {20,22,23,24}<=set(invariant_variants)
    for name,screen in screens.items():
        table={(x['query_rows'],x['rows'],x['layers'],x['candidate']):x for x in summaries[name]}
        expected={(r,t,l,v) for r,t in screen['workloads']
                  for l in screen['layers'] for v in screen['candidates']}
        if len(table)!=len(summaries[name]) or set(table)!=expected:
            raise ValueError('incomplete policy screen comparison census')
        policy='deterministic' if name.endswith('_det') else 'fast'
        for r,t in screen['workloads']:
            for l in screen['layers']:
                key=(policy,r,t,l)
                cell=cells[r,t,l]
                if key in visited or cell[policy]!=screen['control']:
                    raise ValueError('policy screen does not compare the frozen incumbent')
                visited.add(key)
                eligible=[table[r,t,l,v] for v in (23,24)
                          if table[r,t,l,v]['decision']=='faster'
                          and (policy=='fast' or compatible)]
                if eligible:
                    cell[policy]=min(eligible,key=lambda x:(x['ratio'],x['candidate']))['candidate']
    if visited!={(policy,r,t,l) for r,t,l in cells if r>1 for policy in ('fast','deterministic')}:
        raise ValueError('missing multirow policy comparison')
    return dict(round=2,proposals=proposals,deterministic_fallback=20,
        deterministic_family=[20,22,23,24] if compatible else incumbent['deterministic_family'],
        compatible_row_reuse=compatible)


def policy_round1_decision(directory,numerical_directories,build):
    return policy_round_decision(directory,numerical_directories,build,1)


def policy_round_decision(directory,numerical_directories,build,round_number):
    """Bind selection to raw timing, full numerical census and one clean build."""
    import copy
    from .. import decoder_validation as validation
    from .study import STUDIES,load_run
    numerical_directories=list(map(Path,numerical_directories))
    if not numerical_directories:raise ValueError('missing policy numerical split')
    evidence=[validation.read_policy_evidence(p) for p in numerical_directories]
    declared=evidence[0][0]['declaration']
    if round_number not in (1,2):raise ValueError('unknown policy round')
    round_spec=declared['rounds'][round_number-1]
    if build['repository']['dirty'] or build['sources'].get(POLICY_PATH)!=sha_json(declared):
        raise ValueError('policy declaration differs from measurement build')
    anchor_path=repository_root()/'tests/fixtures/decoder_layer/checksums.json'
    evidence_path=anchor_path.with_name('development.json.gz')
    anchor=json.loads(anchor_path.read_text())
    if (sha(anchor_path)!=build['sources'].get(str(anchor_path.relative_to(repository_root())))
        or sha(evidence_path)!=anchor['evidence_sha256']):
        raise ValueError('policy numerical reference changed')
    raw=gzip.decompress(evidence_path.read_bytes())
    if hashlib.sha256(raw).hexdigest()!=anchor['uncompressed_sha256']:
        raise ValueError('policy numerical evidence changed')
    frozen=json.loads(raw);numerical=[];splits=set()
    invariant=set(round_spec['numerical']['variants']);compatible=True
    for record,checks_path,original_receipt_sha in evidence:
        source=record['build']['source'];split=record['split']
        if (record.get('kind')!='decoder_policy_evaluation' or record.get('status')!='passed'
            or record.get('exit_code')!=0 or record.get('declaration')!=declared
            or source['repository']!=build['repository']
            or any(source['sources'].get(k)!=v for k,v in build['sources'].items())
            or record.get('variants')!=round_spec['numerical']['variants']
            or record.get('invariant_variants')!=round_spec['numerical']['required_invariants']
            or record.get('comparison_family')!=round_spec['numerical']['comparison_family']
            or split in splits or split not in round_spec['numerical']['splits']):
            raise ValueError('policy numerical evaluation identity changed')
        splits.add(split)
        cases={name:copy.deepcopy(case) for name,case in frozen['cases'].items()
               if case['spec']['nq']==14 and name.startswith('checkpoint_')==(split=='checkpoint')}
        for name,case in cases.items():
            if any(record['fixtures'].get(name+'/'+label+'.npy')!=spec['sha256'] for label,spec in case['arrays'].items()):
                raise ValueError('policy evaluation used different numerical inputs')
            case['schedules'].update({key:[dict(start=p,rows=r) for p,r in calls]
                for key,calls in policy_schedules(case['spec']['rows'],declared).items()})
        replay=validation.validate_results(checks_path,cases,True,
            record['variants'],record['invariant_variants'],record['comparison_family'])
        if any(record.get(k)!=v for k,v in replay.items()):
            raise ValueError('policy numerical summary differs from complete records')
        invariant &= {int(v) for v,result in replay['schedule_invariance'].items() if result['mismatches']==0}
        compatible &= replay['family_compatible']
        numerical.append(dict(split=split,evaluation_sha256=original_receipt_sha,checks_sha256=record['checks_sha256'],
            binary_sha256=record['build']['binary_sha256']))
    if splits!=set(round_spec['numerical']['splits']):raise ValueError('missing policy numerical split')
    summaries={};runs=[]
    for screen in round_spec['screens']:
        name=screen['name'];path,prefix=selection_run_location(directory,name)
        run,_,summary=load_run(path,prefix)
        if (run['study']!=name or run['build']!=build
            or run['specification']!=json.loads(json.dumps(STUDIES[name]))):
            raise ValueError('policy timing screen identity changed')
        summaries[name]=summary
        runs.append(dict(study=name,run_sha256=sha(path/(prefix+'run.json')),samples_sha256=run['samples_sha256']))
    return dict(schema=1,kind='decoder_policy_round_decision',declaration_sha256=sha_json(declared),
        build_sha256=hashlib.sha256(json.dumps(build,sort_keys=True).encode()).hexdigest(),
        numerical=numerical,screens=runs,invariant_variants=sorted(invariant),
        **(select_policy_round1 if round_number==1 else select_policy_round2)(
            summaries,invariant,compatible,declared))


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


def _policy_matrix(name,control,cells,field):
    """One control and reuse mode, with complete per-cell self calibration."""
    layers=sorted({p['layers'] for p in cells})
    if len(layers)!=1:raise ValueError('policy matrix must separate reuse modes')
    comparisons=[dict(query_rows=p['query_rows'],rows=p['rows'],layers=p['layers'],candidate=v)
        for p in cells for v in sorted({control,p[field]})]
    variants=sorted({p['candidate'] for p in comparisons})
    return dict(name=name,control=control,candidates=variants,layers=layers,
        names={v:NAMES[v] for v in variants},
        workloads=[dict(query_rows=p['query_rows'],rows=p['rows']) for p in cells],
        comparisons=comparisons)


def propose_policy_confirmation(decision,declaration=None):
    """Freeze the final table and the previously unmeasured long extension."""
    declaration=policy_declaration() if declaration is None else declaration
    if decision.get('round')!=2 or not decision.get('compatible_row_reuse'):
        raise ValueError('final proposal requires the complete compatible second round')
    cells=[]
    source={(p['query_rows'],p['rows'],p['layers']):p for p in decision['proposals']}
    expected={(r,t,l) for r,t,_ in declaration['workloads'] if r!=4096 for l in declaration['modes']}
    if len(source)!=len(decision['proposals']) or set(source)!=expected:
        raise ValueError('incomplete final proposal basis')
    for r,t,control in declaration['workloads']:
        for l in declaration['modes']:
            p=source[256,256,l] if r==4096 else source[r,t,l]
            cells.append(dict(query_rows=r,rows=t,layers=l,fast_control=control,
                fast=p['fast'],deterministic=p['deterministic'],
                unmeasured_extension=r==4096))
    screens=[]
    for l in declaration['modes']:
        mode='hot' if l==1 else 'ring'
        det=[p for p in cells if p['layers']==l]
        screens.append(dict(**_policy_matrix('decoder_policies_confirmation_det_'+mode,20,det,'deterministic'),policy='deterministic'))
        for control in sorted({p['fast_control'] for p in cells}):
            fast=[p for p in det if p['fast_control']==control and p['fast']!=control]
            if fast:screens.append(dict(**_policy_matrix(f'decoder_policies_confirmation_fast_{control}_{mode}',control,fast,'fast'),policy='fast'))
    return dict(schema=1,kind='decoder_policy_confirmation_proposal',basis=decision,
        basis_sha256=sha_json(decision),cells=cells,screens=screens,
        deterministic_fallback=20,numerical=dict(variants=[100,101,102,103],
            required_invariants=[101,103],comparison_family=[101,103],split='holdout'),
        rule='One independent serialized timing session. Qualify each proposed deterministic configuration directly against20 and every changed Fast configuration against its original Fast incumbent, with self calibration. Apply the existing faster rule; otherwise retain the fallback. Then measure the accepted deterministic configuration directly against accepted Fast at all seven cells in both modes, self calibrated. This cost phase cannot change selection. No new candidates or repeated confirmation.',
        numerical_order='Encode exactly the mechanically accepted lookup, run full regression and commit. Capture the already reserved fresh inputs once for that final numerical executable and execute the actual lookups across every declared schedule. Kernel timing and final lookup build identities are recorded separately.')


def policy_confirmation_specs(declaration=None):
    from .study import STUDIES
    declaration=policy_declaration() if declaration is None else declaration
    final=declaration['final_confirmation']
    if final!=json.loads(json.dumps(propose_policy_confirmation(final['basis'],declaration))):
        raise ValueError('policy confirmation proposal changed')
    return {s['name']:{**STUDIES['decoder_layer'],**{k:v for k,v in s.items() if k not in ('name','policy')}}
            for s in final['screens']}


def policy_confirmed_selection(directory,build,declaration=None):
    """Derive acceptance only from all frozen, completed qualification runs."""
    import copy
    from .study import load_run
    declaration=policy_declaration() if declaration is None else declaration
    final=declaration['final_confirmation']
    if build['repository']['dirty'] or build['sources'].get(POLICY_PATH)!=sha_json(declaration):
        raise ValueError('confirmation build does not bind this proposal')
    specs=policy_confirmation_specs(declaration);tables={};receipts=[]
    for name,spec in specs.items():
        path,prefix=selection_run_location(directory,name);run,_,summary=load_run(path,prefix)
        if (run['study']!=name or run['build']!=build or run['specification']!=json.loads(json.dumps(spec))):
            raise ValueError('policy qualification identity changed')
        tables[name]={(p['query_rows'],p['rows'],p['layers'],p['candidate']):p for p in summary}
        receipts.append(dict(study=name,run_sha256=sha(path/(prefix+'run.json')),samples_sha256=run['samples_sha256']))
    cells=copy.deepcopy(final['cells'])
    for screen in final['screens']:
        policy=screen['policy'];table=tables[screen['name']]
        for w in screen['workloads']:
            for l in screen['layers']:
                cell=next(p for p in cells if (p['query_rows'],p['rows'],p['layers'])==(w['query_rows'],w['rows'],l))
                proposed=cell[policy];row=table[w['query_rows'],w['rows'],l,proposed]
                cell[policy]=proposed if proposed==screen['control'] or row['decision']=='faster' else screen['control']
    return dict(schema=1,kind='decoder_policy_confirmed_selection',proposal_sha256=sha_json(final),
        build_sha256=hashlib.sha256(json.dumps(build,sort_keys=True).encode()).hexdigest(),
        qualification=receipts,cells=cells,fallbacks=dict(fast=0,deterministic=20))


def policy_cost_specs(accepted):
    from .study import STUDIES
    if accepted.get('kind')!='decoder_policy_confirmed_selection':
        raise ValueError('policy cost needs a confirmed lookup')
    result={}
    for l in (1,24):
        for control in sorted({p['fast'] for p in accepted['cells'] if p['layers']==l}):
            cells=[p for p in accepted['cells'] if p['layers']==l and p['fast']==control]
            name=f'decoder_policies_cost_{control}_'+('hot' if l==1 else 'ring')
            spec=_policy_matrix(name,control,cells,'deterministic')
            result[name]={**STUDIES['decoder_layer'],**{k:v for k,v in spec.items() if k!='name'}}
    return result


def policy_cost_report(directory,accepted,build):
    from .study import load_run
    rows=[];receipts=[]
    for name,spec in policy_cost_specs(accepted).items():
        path,prefix=selection_run_location(directory,name);run,_,summary=load_run(path,prefix)
        if (run['study']!=name or run['build']!=build or run.get('selection')!=accepted
            or run['specification']!=json.loads(json.dumps(spec))):
            raise ValueError('policy cost does not execute the accepted configurations')
        selected={(p['query_rows'],p['rows'],p['layers']):p['deterministic'] for p in accepted['cells']}
        rows.extend(dict(**p,fast=spec['control'],deterministic=p['candidate']) for p in summary
            if p['candidate']==selected[p['query_rows'],p['rows'],p['layers']])
        receipts.append(dict(study=name,run_sha256=sha(path/(prefix+'run.json')),samples_sha256=run['samples_sha256']))
    expected={(p['query_rows'],p['rows'],p['layers'],p['fast'],p['deterministic']) for p in accepted['cells']}
    actual={(p['query_rows'],p['rows'],p['layers'],p['fast'],p['deterministic']) for p in rows}
    if len(rows)!=len(expected) or actual!=expected:raise ValueError('incomplete final policy cost table')
    return dict(kind='decoder_policy_cost',accepted=accepted,runs=receipts,rows=rows)


def replay_policy_campaign(directory):
    """Reconstruct the entire bounded campaign from adjacent retained evidence."""
    import copy
    from .. import decoder_validation as validation
    from .study import load_run,load_decoder_profile,load_decoder_windows
    directory=Path(directory)
    index=json.loads((directory/'policies-evidence.json').read_text())
    if index.get('kind')!='decoder_policy_campaign' or index.get('schema')!=1:
        raise ValueError('unsupported decoder policy campaign')
    for name,digest in index['files'].items():
        if name!=Path(name).name or sha(directory/name)!=digest:
            raise ValueError('decoder policy evidence index changed: '+name)
    runs={}
    for name in index['timing_studies']:
        run,samples,summary=load_run(directory,name+'_')
        if run['study']!=name:raise ValueError('decoder campaign timing identity changed')
        runs[name]=(run,samples,summary)
    numerical=[]
    anchor_path=repository_root()/'tests/fixtures/decoder_layer/checksums.json'
    evidence_path=anchor_path.with_name('development.json.gz')
    anchor=json.loads(anchor_path.read_text())
    if sha(evidence_path)!=anchor['evidence_sha256']:
        raise ValueError('decoder policy reference evidence changed')
    raw=gzip.decompress(evidence_path.read_bytes())
    if hashlib.sha256(raw).hexdigest()!=anchor['uncompressed_sha256']:
        raise ValueError('decoder policy reference payload changed')
    frozen=json.loads(raw)
    for name in index['incomplete_numerical']:
        record,_,_=validation.read_policy_evidence(directory/name)
        if record.get('status')!='failed' or record.get('exit_code')!=-15:
            raise ValueError('interrupted baseline was reclassified as accepted evidence')

    def numeric_row(phase,record,replay):
        return dict(phase=phase,split=record['split'],checks=replay['checks'],
            variants=','.join(map(str,record['variants'])),
            schedule_comparisons=sum(v['comparisons'] for v in replay['schedule_invariance'].values()),
            schedule_mismatches=sum(v['mismatches'] for v in replay['schedule_invariance'].values()),
            required_invariants=','.join(map(str,record['invariant_variants'])),
            required_invariant_mismatches=sum(replay['schedule_invariance'][str(v)]['mismatches'] for v in record['invariant_variants']),
            family_comparisons=replay.get('family_comparisons',0),family_compatible=replay.get('family_compatible',''),
            commit=record['build']['source']['repository']['commit'],binary_sha256=record['build']['binary_sha256'])

    def replay_numerical(name,cases,phase):
        record,checks,_=validation.read_policy_evidence(directory/name)
        if record.get('status')!='passed' or record.get('exit_code')!=0:
            raise ValueError('decoder policy numerical run did not pass')
        declared=record['declaration']
        source=record['build']['source']
        if (source['repository']['dirty'] or source['sources'].get(POLICY_PATH)!=sha_json(declared)
            or source['sources'].get('tests/fixtures/decoder_layer/checksums.json')!=sha(anchor_path)):
            raise ValueError('decoder policy numerical source is not bound')
        cases=copy.deepcopy(cases)
        for case in cases.values():
            case['schedules'].update({key:[dict(start=p,rows=r) for p,r in calls]
                for key,calls in policy_schedules(case['spec']['rows'],declared).items()})
        result=validation.validate_results(checks,cases,True,record['variants'],
            record['invariant_variants'],record.get('comparison_family',()),
            lookup=declared.get('accepted_lookup'))
        if any(k in record and record[k]!=v for k,v in result.items()):
            raise ValueError('decoder policy numerical summary changed')
        numerical.append(numeric_row(phase,record,result))
        return record,result

    baseline_build=runs['decoder_policies_baseline_0'][0]['build']
    for name in index['baseline_numerical']:
        # The earliest receipts predate individual-array hashes and summary
        # fields. Their complete raw checks are replayed with that declaration.
        record=json.loads((directory/name).read_text())['evaluation']
        split=record['split']
        cases={n:c for n,c in frozen['cases'].items()
            if c['spec']['nq']==14 and n.startswith('checkpoint_')==(split=='checkpoint')}
        record,_=replay_numerical(name,cases,'baseline')
        if (record['build']['source']['repository']!=baseline_build['repository']
            or any(record['build']['source']['sources'].get(k)!=v for k,v in baseline_build['sources'].items())):
            raise ValueError('baseline numerical and timing source differ')
    for phase in index['rounds']:
        number=phase['round'];name=f'decoder_policies_round{number}_det'
        build=runs[name][0]['build']
        expected=json.loads((directory/phase['decision']).read_text())
        decision=policy_round_decision(directory,[directory/n for n in phase['numerical']],build,number)
        if decision!=expected:raise ValueError('decoder optimization decision changed')
        for name in phase['numerical']:
            record=json.loads((directory/name).read_text())['evaluation']
            numerical.append(numeric_row('round'+str(number),record,record))
        adversarial=json.loads((directory/phase['adversarial']).read_text())
        if (adversarial.get('status')!='passed'
            or adversarial['binary_sha256']!=build['binaries']['decoder_layer']
            or adversarial['build_sha256']!=hashlib.sha256(json.dumps(build,sort_keys=True).encode()).hexdigest()):
            raise ValueError('adversarial policy benchmark identity changed')
        expected_cases={(r,t,24,v) for r,t in phase['adversarial_shapes']
            for v in record['declaration']['rounds'][number-1]['numerical']['variants']}
        actual_cases={(p['query_rows'],p['rows'],p['layers'],p['variant']) for p in adversarial['cases']}
        if actual_cases!=expected_cases or len(actual_cases)!=len(adversarial['cases']):
            raise ValueError('adversarial policy case census changed')
        for case in adversarial['cases']:
            text=adversarial['retained_outputs'][case['output']]
            if (hashlib.sha256(text.encode()).hexdigest()!=case['output_sha256']
                or 'correctness: passed' not in text or 'BENCHMARK_COMPLETE' not in text
                or case['backend']!='metal' or not case['device'].startswith('Apple ')):
                raise ValueError('adversarial benchmark output changed')
    declared=json.loads((directory/index['confirmation_declaration']).read_text())
    if declared['final_confirmation']['basis']!=decision:
        raise ValueError('final proposal is not the reconstructed last-round decision')
    build=runs['decoder_policies_confirmation_det_hot'][0]['build']
    accepted=policy_confirmed_selection(directory,build,declared)
    if accepted!=json.loads((directory/index['accepted']).read_text()):
        raise ValueError('confirmed decoder policy lookup changed')
    cost=policy_cost_report(directory,accepted,build)
    if cost!=json.loads((directory/index['cost']).read_text()):
        raise ValueError('final decoder policy cost changed')
    final=json.loads((directory/index['holdout_numerical']).read_text())['evaluation']
    if final['declaration'].get('accepted_lookup')!=accepted:
        raise ValueError('final numerical execution used a different lookup')
    regression=json.loads((directory/index['repository_validation']).read_text())
    if (regression.get('status')!='passed' or regression.get('exit_code')!=0
        or regression['source']['sources']!=final['build']['source']['sources']):
        raise ValueError('final numerical source did not pass the repository regression')
    bridge=json.loads((directory/index['build_bridge']).read_text())
    native='src/llm_mojo/decoder_layer.mojo'
    if (bridge.get('kind')!='decoder_policy_build_bridge'
        or bridge['timing_commit']!=build['repository']['commit']
        or bridge['timing_binary_sha256']!=build['binaries']['decoder_layer']
        or bridge['numerical_commit']!=final['build']['source']['repository']['commit']
        or bridge['numerical_binary_sha256']!=final['build']['binary_sha256']
        or bridge['environment']!=build['environment']
        or bridge['changed_native_sources']!=([native] if build['sources'][native]!=final['build']['source']['sources'][native] else [])
        or bridge['timing_source_sha256']!=build['sources'][native]
        or bridge['numerical_source_sha256']!=final['build']['source']['sources'][native]):
        raise ValueError('timing and numerical build identities are not bridged')
    if any(final['build']['source']['sources'].get(k)!=v for k,v in build['sources'].items()
        if k.startswith('src/') and k.endswith('.mojo') and k!=native):
        raise ValueError('native arithmetic changed between timing and final confirmation')
    manifest,hashes=validation.holdout_manifest(directory/index['holdout_manifest'],
        verify_arrays=False,policy_declaration=final['declaration'])
    if (manifest['candidate']['binary_sha256']!=final['build']['binary_sha256']
        or manifest['candidate']['commit']!=final['build']['source']['repository']['commit']
        or hashes!=final['fixtures']):
        raise ValueError('fresh decoder inputs are not bound to the final executable')
    expected=declared['final_confirmation']['numerical']
    if (final['variants']!=expected['variants'] or final['invariant_variants']!=expected['required_invariants']
        or final.get('comparison_family')!=expected['comparison_family'] or final['split']!=expected['split']):
        raise ValueError('final policy numerical contract changed')
    _,result=replay_numerical(index['holdout_numerical'],manifest['cases'],'confirmation')
    if result.get('family_compatible')is not True:
        raise ValueError('final deterministic reuse modes are not compatible')
    profile_identity=declared['rounds'][0]['baseline_profile_sha256']
    for name,digest in profile_identity.items():
        if sha(directory/('policies_'+name))!=digest:
            raise ValueError('optimization rationale names different baseline profiles')
    profile=load_decoder_profile(directory,'policies_')
    windows=load_decoder_windows(directory,'policies_')
    return dict(accepted=accepted,cost=cost,runs=runs,numerical=numerical,profile=profile,windows=windows)
