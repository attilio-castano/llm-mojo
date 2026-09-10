"""Freeze and evaluate an exact decoder binary with complete numerical coverage."""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import lzma
import os
from pathlib import Path
import subprocess

from ._repository import environment_tool, repository_root
from .mlp_validation import sha, write, source_identity as base_source_identity
from .benchmarks.environment import ensure_record_location, utc_now
from .benchmarks import decoder_layer_contract as selection_contract
from .benchmarks.study import load_numerical_record

BOUNDARIES=('B_att','Z','B_mlp','Y')
STAGES=('N_att','Q_raw','K_raw','V_raw','Q','K_rot','O','B_att','Z','N_mlp','G','U','A','S','B_mlp','Y')


def environment():
    return {k:v for k,v in os.environ.items() if not k.startswith('DECODER_') and k!='MODULAR_DEBUG'}


def source_identity():
    record=base_source_identity()
    path=repository_root()/selection_contract.SELECTION_PATH
    if path.exists():record['sources'][selection_contract.SELECTION_PATH]=sha(path)
    return record


def build(binary,selection=False):
    binary=Path(binary).resolve();ensure_record_location(binary)
    receipt=Path(str(binary)+'.provenance.json')
    if binary.exists() or receipt.exists():
        raise ValueError('refusing to overwrite decoder build')
    source=source_identity()
    if source['repository']['dirty']:
        raise ValueError('decoder build requires clean source')
    binary.parent.mkdir(parents=True,exist_ok=True)
    test='tests/test_decoder_selection.mojo' if selection else 'tests/test_decoder_layer.mojo'
    command=[environment_tool('mojo'),'build','-I','src','-I','tests',test,'-o',str(binary)]
    subprocess.run(command,cwd=repository_root(),env=environment(),check=True)
    if source_identity()!=source:
        raise ValueError('decoder source changed during build')
    write(receipt,dict(schema=1,kind='decoder_numerical_build',source=source,selection=selection,
          binary_sha256=sha(binary),command=command,created_utc=utc_now()))


def verify_build(binary):
    binary=Path(binary).resolve()
    record=json.loads(Path(str(binary)+'.provenance.json').read_text())
    if (record.get('schema')!=1 or record.get('kind')!='decoder_numerical_build'
        or record.get('binary_sha256')!=sha(binary) or record['source']['repository']['dirty']
        or record['source']!=source_identity()):
        raise ValueError('decoder binary or build source differs from receipt')
    return record


def holdout_manifest(root,verify_arrays=True,policy_declaration=None):
    root=Path(root)
    manifest=root if root.is_file() else root/'manifest.json'
    root=manifest.parent;record=load_numerical_record(manifest)
    payload={k:v for k,v in record.items() if k!='payload_sha256'}
    import hashlib
    selection=record.get('kind')=='decoder_selection_holdout'
    policy=record.get('kind')=='decoder_policy_holdout'
    if (record.get('status')!='complete' or record.get('kind') not in ('decoder_holdout','decoder_selection_holdout','decoder_policy_holdout')
        or hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=record.get('payload_sha256')):
        raise ValueError('decoder holdout is incomplete or changed')
    repo=repository_root();anchor=json.loads((repo/'tests/fixtures/decoder_layer/checksums.json').read_text())
    if record['candidate']['reference_sha256']!=sha(repo/'tests/fixtures/decoder_layer/checksums.json'):
        raise ValueError('decoder reference identity changed')
    expected={'h{h}_i{i}_nq{nq}_nk{nk}_d{d}_t{rows}_s{seed}_{mutation}'.format(**s):s
              for s in anchor['specification']['holdout']}
    ids=anchor['checkpoint']['holdout_token_ids'];checkpoint_name='checkpoint_holdout'
    if selection or policy:
        if policy:
            declared_policy=selection_contract.policy_declaration() if policy_declaration is None else policy_declaration
            if record.get('policies')!=declared_policy:raise ValueError('policy holdout declaration changed')
            declared=declared_policy['confirmation']
        else:
            declared=selection_contract.selection_declaration()
            if record.get('selection')!=declared:raise ValueError('selection holdout declaration changed')
        ids=declared['checkpoint_token_ids']
        import struct
        if not ids or hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()!=declared['checkpoint_token_ids_sha256']:
            raise ValueError('decoder reserved token identity changed')
        checkpoint_name='checkpoint_policy_holdout' if policy else 'checkpoint_selection_holdout'
        expected={f'h896_i4864_nq14_nk2_d64_t{t}_s{seed}_base':dict(h=896,nq=14,nk=2,d=64,i=4864,rows=t,seed=seed,mutation='base')
                  for seed in declared['seeds'] for t in declared['rows']}
    expected[checkpoint_name]=dict(h=896,nq=14,nk=2,d=64,i=4864,
        rows=len(ids),seed=0,mutation='base')
    if {name:case['spec'] for name,case in record['cases'].items()}!=expected:
        raise ValueError('decoder reserved case census changed')
    if record['checkpoint']!=anchor['checkpoint'] or record['cases'][checkpoint_name]['token_ids']!=ids:
        raise ValueError('decoder reserved checkpoint identity changed')
    if record['specification']!=anchor['specification'] or record['sources']!=anchor['sources'] or record['upstream']!=anchor['upstream']:
        raise ValueError('decoder reserved reference policy changed')
    for case in record['cases'].values():
        t=case['spec']['rows']
        chunks=[1]*t if t<=17 else [t-17,16,1]
        required={'full':[(0,t)],'chunk':[]}
        position=0
        for r in chunks:
            required['chunk'].append((position,r));position+=r
        if policy and t==33:required['threshold']=[(0,16),(16,1),(17,15),(32,1)]
        if policy and t==65:required['reuse']=[(0,53)]+[(p,1) for p in range(53,65)]
        actual={key:[(c['start'],c['rows']) for c in calls] for key,calls in case['schedules'].items()}
        if actual!=required:
            raise ValueError('decoder reserved schedule changed')
    for case in record['cases'].values():
        required={'input_X',*('input_'+k for k in anchor['specification']['weights'])}
        required.update('full_'+k for k in anchor['specification']['stages'])
        for schedule,calls in case['schedules'].items():
            if schedule!='full':
                required.update(f"{schedule}_{call['start']}_{stage}" for call in calls for stage in anchor['specification']['stages'])
        if set(case['arrays'])!=required:
            raise ValueError('decoder reserved array census changed')
        if any(a.get('dtype')!='float32' or a.get('logical_dtype')!='bfloat16' for a in case['arrays'].values()):
            raise ValueError('decoder reserved storage dtype changed')
    # The loader above verified both compressed and original payload hashes.
    # Fixture identity names the experiment's manifest, not its storage wrapper.
    stored=json.loads(manifest.read_text())
    manifest_sha=stored['uncompressed_sha256'] if stored.get('format')=='lossless-json-gzip-v1' else sha(manifest)
    hashes={'manifest.json':manifest_sha}

    for name,case in record['cases'].items():
        for label,spec in case['arrays'].items():
            path=root/name/(label+'.npy')
            if verify_arrays and sha(path)!=spec['sha256']:
                raise ValueError('decoder reserved array changed')
            hashes[name+'/'+label+'.npy']=spec['sha256']
    return record,hashes


def policies(spec,selection,variants=None):
    if variants is not None:return variants
    return sorted(selection_contract.VARIANTS) if selection else [0,7] if spec['nq']==14 and spec['rows']>1 else [0]


def expected_checks(cases,selection=False,variants=None,comparison_family=()):
    expected=Counter()
    for name,case in cases.items():
        spec=case['spec'];t=spec['rows']
        for policy in policies(spec,selection,variants):
            for schedule,calls in case['schedules'].items():
                for call in calls:
                    p,r=call['start'],call['rows']
                    expected[name,policy,schedule,'layer','route','',p,r]+=1
                    for stage in STAGES:
                        expected[name,policy,schedule,'layer','boundary',stage,p,r]+=1
                    if schedule=='full' and policy in comparison_family[1:]:
                        for stage in (*STAGES,'consistent_cache_key','consistent_cache_value'):
                            expected[name,policy,schedule,'layer','family_exact',stage,p,r]+=1
                    if schedule!='full':
                        for stage in STAGES if variants is not None else BOUNDARIES:
                            expected[name,policy,schedule,'layer','full_vs_chunk',stage,p,r]+=1
                        if variants is not None:
                            for stage in (*STAGES,'consistent_cache_key','consistent_cache_value'):
                                expected[name,policy,schedule,'layer','schedule_exact',stage,p,r]+=1
                    for stage in ('cache_key','cache_value'):
                        label=('full_' if schedule=='full' else f'{schedule}_{p}_')+stage
                        expected[name,policy,schedule,'layer','cache',label,p,r]+=1
            for stage in STAGES:
                expected[name,policy,'full','operation','boundary',stage,0,t]+=1
            for stage in ('B_mlp','Y'):
                expected[name,policy,'full','mlp','boundary',stage,0,t]+=1
    return expected


def protected_extents(spec):
    """Full allocations snapshotted by the decoder case, including rotary guards."""
    t, h, i = spec['rows'], spec['h'], spec['i']
    k, d = spec['nk'] * spec['d'], spec['d']
    capacity = min(t + 1, 4096)
    return dict(xb=t*h, aw_norm=h, aw_qkv=(h+2*k)*h, aw_bias=h+2*k,
                aw_output=h*h, mw_norm=h, mw_gate=i*h, mw_up=i*h,
                mw_down=h*i, a_cosine=capacity*d, a_sine=capacity*d)


_COLUMN_HEADER=b'{"format":"decoder-checks-columns-v1"}\n'
_COLUMN_SUFFIX='.columns.jsonl.xz'


def column_records(path):
    """Transpose bounded column blocks back into ordered check dictionaries."""
    with lzma.open(path,'rb') as stream:
        if stream.readline()!=_COLUMN_HEADER:raise ValueError('unsupported column archive')
        for line in stream:
            order,tables=json.loads(line)
            if not isinstance(order,list) or not order or len(order)>8192 or not tables:
                raise ValueError('invalid column block')
            positions=[0]*len(tables)
            for keys,columns in tables:
                if (not keys or not all(isinstance(k,str) for k in keys)
                    or len(set(keys))!=len(keys) or len(keys)!=len(columns)
                    or not all(isinstance(c,list) for c in columns)
                    or not columns[0] or any(len(c)!=len(columns[0]) for c in columns)):
                    raise ValueError('invalid column table')
            for index in order:
                if type(index) is not int or not 0<=index<len(tables):
                    raise ValueError('invalid column order')
                keys,columns=tables[index];position=positions[index]
                if position>=len(columns[0]):raise ValueError('column order exceeds table')
                yield dict(zip(keys,(column[position] for column in columns)))
                positions[index]+=1
            if any(position!=len(table[1][0]) for position,table in zip(positions,tables)):
                raise ValueError('column order omits records')


def compact_checks(source,destination):
    """Losslessly transpose canonical JSONL; return original and encoded hashes.

    Each block stores key layouts once, values by column, and the original row
    order. No check, field, floating-point value or repeated row is discarded.
    Reject noncanonical input instead of silently changing its original bytes.
    """
    source,destination=Path(source),Path(destination)
    if not destination.name.endswith(_COLUMN_SUFFIX):raise ValueError('wrong column archive suffix')
    original=hashlib.sha256();count=0;size=0
    # Exclusive creation keeps an existing evidence artifact untouched.
    with destination.open('xb') as target:
        try:
            with (gzip.open(source,'rb') if source.suffix=='.gz' else source.open('rb')) as stream, \
                lzma.open(target,'wb',preset=6) as encoded:
                encoded.write(_COLUMN_HEADER)
                while block:=list(itertools.islice(stream,8192)):
                    layouts={};tables=[];order=[]
                    for line in block:
                        row=json.loads(line)
                        if not isinstance(row,dict) or not row or (json.dumps(row)+'\n').encode()!=line:
                            raise ValueError('checks are not canonical JSONL')
                        original.update(line);count+=1;size+=len(line)
                        keys=tuple(row)
                        if keys not in layouts:
                            layouts[keys]=len(tables);tables.append((keys,[[] for _ in keys]))
                        index=layouts[keys];order.append(index)
                        for column,value in zip(tables[index][1],row.values()):column.append(value)
                    encoded.write((json.dumps([order,tables],separators=(',',':'))+'\n').encode())
            target.flush()
            restored=hashlib.sha256();restored_count=0
            for row in column_records(destination):
                restored.update((json.dumps(row)+'\n').encode());restored_count+=1
            if restored.digest()!=original.digest() or restored_count!=count:
                raise ValueError('column archive did not reproduce original checks')
        except Exception:
            destination.unlink()
            raise
    return dict(file=destination.name,sha256=sha(destination),
        uncompressed_sha256=original.hexdigest(),records=count,uncompressed_bytes=size)


def check_records(path):
    """Stream complete raw, gzip or lossless columnar numerical records."""
    path=Path(path)
    if path.name.endswith(_COLUMN_SUFFIX):
        yield from column_records(path)
        return
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        for line in stream:yield json.loads(line)


def validate_results(path,cases,selection=False,variants=None,invariant_variants=(),comparison_family=(),lookup=None):
    observed=Counter();runtimes=set();aux=Counter();protected=Counter()
    async_rows=[];family=[];schedule_counts=Counter();schedule_mismatches=Counter()
    for row in check_records(path):
        if selection and row.get('mode')=='async':async_rows.append(row)
        if comparison_family and row.get('kind')=='family_exact':family.append(row)
        if (variants is not None and row.get('kind')=='schedule_exact'
            and row.get('case') in cases and row.get('mode')=='layer'):
            schedule_counts[row['policy']]+=1
            schedule_mismatches[row['policy']]+=row.get('exact')is not True
        if row.get('expected_failure'):
            if row.get('mode')!='negative' or row.get('failed',0)<=0:
                raise ValueError('invalid expected negative control')
            continue
        if row.get('failed',0):
            raise ValueError('decoder numerical failure')
        if row.get('kind')=='negative_control':
            if row.get('rejected')is not True:
                raise ValueError('negative control was not rejected')
            aux[row['label']]+=1
        if row.get('case') not in cases or row.get('mode') not in ('layer','operation','mlp'):
            continue
        kind=row['kind']
        if kind=='exact':
            if row.get('failed')!=0 or type(row.get('elements')) is not int or row['elements']<=0:
                raise ValueError('missing exact preservation check')
            protected[row['case'],row['policy'],row['label'],row['elements']]+=1
            continue
        key=tuple(row.get(k,'') for k in ('case','policy','schedule','mode','kind','stage','start','rows'))
        observed[key]+=1
        spec=cases[row['case']]['spec'];r=row['rows'];h=spec['h'];stage=row.get('stage','')
        if kind=='route':
            gqa,_,mlp=selection_contract.execution_mappings(row['policy'],r,row['start']+r,lookup) if selection else (0,0,row['policy'] if r>1 else 0)
            if row['attention']!=(11 if gqa==5 else 4 if r==1 else 6+gqa) or row['mlp']!=mlp:
                raise ValueError('decoder route changed')
            if row['backend']!='metal' or not row['device'].startswith('Apple '):
                raise ValueError('decoder execution did not prove Metal')
            runtimes.add((row['device'],row['backend']))
        elif kind=='cache':
            if row['elements']!=(row['start']+r)*spec['nk']*spec['d'] or any(row.get(k)is not True for k in ('prefix_exact','append_exact','inactive_exact')):
                raise ValueError('incomplete cache preservation check')
        elif kind in ('schedule_exact','family_exact'):
            width=spec['i'] if stage in ('G','U','A','S') else spec['nk']*spec['d'] if stage in ('K_raw','V_raw','K_rot','consistent_cache_key','consistent_cache_value') else h
            count=(row['start']+r)*width if stage.startswith('consistent_cache_') else r*width
            if type(row.get('exact'))is not bool or type(row.get('elements'))is not int or row.get('elements')!=count:
                raise ValueError('incomplete decoder schedule comparison')
            if kind=='family_exact' and (not comparison_family or type(row.get('reference_variant'))is not int or row.get('reference_variant')!=comparison_family[0]):
                raise ValueError('decoder family comparison identity changed')
            if kind=='schedule_exact' and row['policy'] in invariant_variants and not row['exact']:
                raise ValueError('deterministic decoder schedule mismatch')
        elif kind in ('boundary','full_vs_chunk'):
            width=spec['i'] if stage in ('G','U','A','S') else spec['nk']*spec['d'] if stage in ('K_raw','V_raw','K_rot') else h
            if kind=='boundary' and row.get('inactive_exact')is not True:
                raise ValueError('missing workspace guard check')
            if row['elements']!=r*width:
                raise ValueError('decoder incomplete element coverage')
            if (row['mode']=='operation' or stage in BOUNDARIES) and row.get('failed')!=0:
                raise ValueError('missing decoder gate')
    expected_protected=Counter((name,policy,label,elements) for name,case in cases.items()
        for policy in policies(case['spec'],selection,variants)
        for label,elements in protected_extents(case['spec']).items())
    if variants is not None:
        expected_protected.update((name,policy,label,protected_extents(case['spec'])['a_cosine'])
            for name,case in cases.items() if 'policy_tokenwise' in case['schedules']
            for policy in variants for label in ('tokenwise_cosine','tokenwise_sine'))
    if protected!=expected_protected:
        raise ValueError('incomplete protected decoder storage coverage')
    if observed!=expected_checks(cases,selection,variants,comparison_family) or len(runtimes)!=1:
        raise ValueError('decoder missing, duplicate or unexpected numerical coverage')
    negatives={'second residual uses X','second residual omitted','first residual omitted',
               'second norm uses X','wrong norm weights','wrong absolute RoPE position',
               'mask exposes future rows','cache prefix changed'}
    if aux!=Counter({n:1 for n in negatives}):
        raise ValueError('incomplete decoder negative controls')
    if selection:
        for row in async_rows:
            if row.get('failed')!=0 or type(row.get('elements')) is not int or row['elements']<=0:
                raise ValueError('missing selection asynchronous gate')
            if row['kind']=='boundary' and row['elements']!=row['rows']*896:
                raise ValueError('incomplete asynchronous elements')
        expected=Counter()
        for v in variants if variants is not None else selection_contract.VARIANTS:
            for j in range(13):
                start,rows=(0,53) if j==0 else (52+j,1)
                for stage in BOUNDARIES:
                    expected[v,'boundary',stage,start,rows,rows*896]+=1
                    expected[v,'exact','async '+stage+' vs separate workspace','','',rows*896]+=1
            expected[v,'boundary','Y',0,1,896]+=1
            for label in ('async cache key','async cache value'):
                expected[v,'exact',label,'','',66*128]+=1
        actual=Counter((r['policy'],r['kind'],r.get('stage',r.get('label')),r.get('start',''),r.get('rows',''),r['elements']) for r in async_rows)
        if actual!=expected:raise ValueError('incomplete selection asynchronous coverage')
    result=dict(checks=sum(observed.values()),runtime=dict(zip(('device','backend'),next(iter(runtimes)))))
    if variants is not None:
        result['schedule_invariance']={str(v):dict(
            comparisons=schedule_counts[v],mismatches=schedule_mismatches[v]) for v in variants}
    if comparison_family:
        result.update(comparison_family=list(comparison_family),family_comparisons=len(family),
                      family_compatible=bool(family) and all(r['exact'] for r in family))
    return result


def read_policy_evidence(location):
    """Return a receipt, complete checks path and original receipt identity."""
    import hashlib
    location=Path(location)
    if location.is_dir():
        receipt=location/'evaluation.json';record=json.loads(receipt.read_text())
        checks=location/'checks.jsonl';log=location/'output.log'
        if sha(checks)!=record['checks_sha256'] or sha(log)!=record['output_sha256']:
            raise ValueError('policy numerical raw records changed')
        return record,checks,sha(receipt)
    wrapper=load_numerical_record(location)
    if wrapper.get('format') not in ('decoder-policy-numerics-gzip-v1','decoder-policy-numerics-columns-v1'):
        raise ValueError('unsupported policy numerical archive')
    record=wrapper['evaluation']
    paths={}
    for field in ('checks','output'):
        spec=wrapper[field];name=spec['file']
        if name!=Path(name).name:raise ValueError('policy archive payload must be adjacent')
        path=location.parent/name
        if sha(path)!=spec['sha256']:raise ValueError('compressed policy evidence changed')
        h=hashlib.sha256()
        try:
            if field=='checks' and wrapper['format']=='decoder-policy-numerics-columns-v1':
                if not path.name.endswith(_COLUMN_SUFFIX):raise ValueError('wrong column archive suffix')
                count=0;size=0
                for row in column_records(path):
                    data=(json.dumps(row)+'\n').encode();h.update(data);count+=1;size+=len(data)
                if count!=spec['records'] or size!=spec['uncompressed_bytes']:
                    raise ValueError('column archive census changed')
            else:
                with gzip.open(path,'rb') as stream:
                    for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
        except (OSError,EOFError,lzma.LZMAError) as error:
            raise ValueError('invalid compressed policy evidence') from error
        if h.hexdigest()!=spec['uncompressed_sha256']:
            raise ValueError('uncompressed policy evidence changed')
        paths[field]=path
    if (wrapper['checks']['uncompressed_sha256']!=record['checks_sha256']
        or wrapper['output']['original_sha256']!=record['output_sha256']):
        raise ValueError('policy archive changed the original check/output identity')
    with gzip.open(paths['output'],'rt') as stream:log=stream.read()
    if record.get('status')=='passed' and '0 failed , 0 skipped' not in log:
        raise ValueError('policy archive lacks complete native suite output')
    return record,paths['checks'],wrapper['original_evaluation_sha256']


def evaluate_policies(binary,fixtures,output,variants,invariant_variants,split='development',comparison_family=()):
    """Reuse the layer suite with an explicit complete policy/schedule census."""
    import copy
    binary,fixtures,output=map(lambda p:Path(p).resolve(),(binary,fixtures,output))
    ensure_record_location(output);candidate=verify_build(binary)
    if not candidate.get('selection'):
        raise ValueError('policy evaluation requires the configuration suite')
    if (not variants or len(set(variants))!=len(variants)
        or not set(variants)<=selection_contract.POLICY_TEST_VARIANTS
        or not set(invariant_variants)<=set(variants)):
        raise ValueError('invalid policy evaluation family')
    if comparison_family and (len(comparison_family)<2 or len(set(comparison_family))!=len(comparison_family)
        or not set(comparison_family)<=set(variants)
        or [v for v in variants if v in comparison_family]!=list(comparison_family)):
        raise ValueError('invalid ordered decoder comparison family')
    raw=json.loads((fixtures/'manifest.json').read_text())
    if raw.get('status')!='complete':raise ValueError('incomplete policy inputs')
    if split=='holdout':
        verified,_=holdout_manifest(fixtures)
        if (verified!=raw or raw.get('kind')!='decoder_policy_holdout'
            or raw['candidate']['binary_sha256']!=candidate['binary_sha256']
            or raw['candidate']['commit']!=candidate['source']['repository']['commit']):
            raise ValueError('policy confirmation names a different frozen candidate')
    cases={name:copy.deepcopy(case) for name,case in raw['cases'].items()
           if case['spec']['nq']==14 and (split=='holdout' or name.startswith('checkpoint_')==(split=='checkpoint'))}
    if not cases:raise ValueError('empty policy split')
    hashes={'manifest.json':sha(fixtures/'manifest.json')}
    for name,case in cases.items():
        for label,spec in case['arrays'].items():
            if sha(fixtures/name/(label+'.npy'))!=spec['sha256']:
                raise ValueError('policy input changed')
            hashes[name+'/'+label+'.npy']=spec['sha256']
        case['schedules'].update({key:[dict(start=p,rows=r) for p,r in calls]
            for key,calls in selection_contract.policy_schedules(case['spec']['rows']).items()})
    output.mkdir(parents=True,exist_ok=False)
    log=output/'output.log';results=output/'checks.jsonl'
    env=environment();env.update(DECODER_SPLIT=split,DECODER_FIXTURES=str(fixtures),
        DECODER_RECORDS=str(results),DECODER_POLICY_STUDY='1',
        DECODER_VARIANTS=','.join(map(str,variants)),
        DECODER_INVARIANT_VARIANTS=','.join(map(str,invariant_variants)),
        DECODER_COMPARISON_FAMILY=','.join(map(str,comparison_family)))
    record=dict(kind='decoder_policy_evaluation',status='started',build=candidate,
        declaration=selection_contract.policy_declaration(),fixtures=hashes,split=split,
        variants=variants,invariant_variants=invariant_variants,comparison_family=list(comparison_family),started_utc=utc_now())
    try:
        with log.open('w') as stream:
            process=subprocess.run([str(binary)],cwd=repository_root(),env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=14400)
        record['exit_code']=process.returncode
        if verify_build(binary)!=candidate or sha(fixtures/'manifest.json')!=hashes['manifest.json']:
            raise ValueError('policy candidate or fixtures changed during evaluation')
        if process.returncode or '0 failed , 0 skipped' not in log.read_text():
            raise ValueError('decoder policy suite failed or truncated')
        for name,case in cases.items():
            for label,spec in case['arrays'].items():
                if sha(fixtures/name/(label+'.npy'))!=spec['sha256']:
                    raise ValueError('policy input changed during evaluation')
        record.update(validate_results(results,cases,True,variants,invariant_variants,comparison_family),status='passed')
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:
        record.update(finished_utc=utc_now(),output_sha256=sha(log) if log.exists() else None,
                      checks_sha256=sha(results) if results.exists() else None)
        write(output/'evaluation.json',record)
    return record


def evaluate(binary,fixtures,output):
    binary,fixtures,output=map(lambda p:Path(p).resolve(),(binary,fixtures,output))
    ensure_record_location(output);candidate=verify_build(binary)
    manifest,hashes=holdout_manifest(fixtures)
    selection=bool(candidate.get('selection'))
    if selection!=(manifest['kind']=='decoder_selection_holdout'):raise ValueError('wrong reserved candidate kind')
    if manifest['candidate']['binary_sha256']!=candidate['binary_sha256'] or manifest['candidate']['commit']!=candidate['source']['repository']['commit']:
        raise ValueError('decoder holdout names a different candidate')
    output.mkdir(parents=True,exist_ok=False)
    log=output/'output.log';results=output/'checks.jsonl'
    env=environment();env.update(DECODER_SPLIT='holdout',DECODER_FIXTURES=str(fixtures),DECODER_RECORDS=str(results))
    record=dict(kind='decoder_evaluation',status='started',build=candidate,fixtures=hashes,started_utc=utc_now())
    try:
        with log.open('w') as stream:
            process=subprocess.run([str(binary)],cwd=repository_root(),env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        record['exit_code']=process.returncode
        if verify_build(binary)!=candidate or holdout_manifest(fixtures)!=(manifest,hashes):
            raise ValueError('decoder candidate or inputs changed during evaluation')
        if process.returncode or '0 failed , 0 skipped' not in log.read_text():
            raise ValueError('decoder numerical suite failed or truncated')
        record.update(validate_results(results,manifest['cases'],selection),status='passed')
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:
        record.update(finished_utc=utc_now(),output_sha256=sha(log) if log.exists() else None,
                      checks_sha256=sha(results) if results.exists() else None)
        write(output/'evaluation.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('compact-checks','expand-checks'):
        c=sub.add_parser(command);c.add_argument('--input',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('build');b.add_argument('--binary',type=Path,required=True);b.add_argument('--selection',action='store_true')
    e=sub.add_parser('evaluate')
    for flag in ('binary','fixtures','output'):
        e.add_argument('--'+flag,type=Path,required=True)
    e=sub.add_parser('evaluate-policies')
    for flag in ('binary','fixtures','output'):
        e.add_argument('--'+flag,type=Path,required=True)
    e.add_argument('--variants',type=int,nargs='+',required=True)
    e.add_argument('--invariant-variants',type=int,nargs='*',default=[])
    e.add_argument('--comparison-family',type=int,nargs='*',default=[])
    e.add_argument('--split',choices=['development','checkpoint','holdout'],default='development')
    args=p.parse_args()
    if args.command=='compact-checks':print(json.dumps(compact_checks(args.input,args.output),indent=2))
    elif args.command=='expand-checks':
        with args.output.open('xb') as stream:
            for row in check_records(args.input):stream.write((json.dumps(row)+'\n').encode())
    elif args.command=='build':build(args.binary,args.selection)
    elif args.command=='evaluate-policies':
        evaluate_policies(args.binary,args.fixtures,args.output,args.variants,args.invariant_variants,args.split,args.comparison_family)
    else:evaluate(args.binary,args.fixtures,args.output)


if __name__=='__main__':main()
