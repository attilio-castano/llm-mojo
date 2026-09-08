"""Freeze and evaluate an exact decoder binary with complete numerical coverage."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess

from ._repository import environment_tool, repository_root
from .mlp_validation import sha, write, source_identity
from .benchmarks.environment import ensure_record_location, utc_now

BOUNDARIES=('B_att','Z','B_mlp','Y')
STAGES=('N_att','Q_raw','K_raw','V_raw','Q','K_rot','O','B_att','Z','N_mlp','G','U','A','S','B_mlp','Y')


def environment():
    return {k:v for k,v in os.environ.items() if not k.startswith('DECODER_') and k!='MODULAR_DEBUG'}


def build(binary):
    binary=Path(binary).resolve();ensure_record_location(binary)
    receipt=Path(str(binary)+'.provenance.json')
    if binary.exists() or receipt.exists():
        raise ValueError('refusing to overwrite decoder build')
    source=source_identity()
    if source['repository']['dirty']:
        raise ValueError('decoder build requires clean source')
    binary.parent.mkdir(parents=True,exist_ok=True)
    command=[environment_tool('mojo'),'build','-I','src','-I','tests','tests/test_decoder_layer.mojo','-o',str(binary)]
    subprocess.run(command,cwd=repository_root(),env=environment(),check=True)
    if source_identity()!=source:
        raise ValueError('decoder source changed during build')
    write(receipt,dict(schema=1,kind='decoder_numerical_build',source=source,
          binary_sha256=sha(binary),command=command,created_utc=utc_now()))


def verify_build(binary):
    binary=Path(binary).resolve()
    record=json.loads(Path(str(binary)+'.provenance.json').read_text())
    if (record.get('schema')!=1 or record.get('kind')!='decoder_numerical_build'
        or record.get('binary_sha256')!=sha(binary) or record['source']['repository']['dirty']
        or record['source']!=source_identity()):
        raise ValueError('decoder binary or build source differs from receipt')
    return record


def holdout_manifest(root,verify_arrays=True):
    root=Path(root);record=json.loads((root/'manifest.json').read_text())
    payload={k:v for k,v in record.items() if k!='payload_sha256'}
    import hashlib
    if (record.get('status')!='complete' or record.get('kind')!='decoder_holdout'
        or hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=record.get('payload_sha256')):
        raise ValueError('decoder holdout is incomplete or changed')
    repo=repository_root();anchor=json.loads((repo/'tests/fixtures/decoder_layer/checksums.json').read_text())
    if record['candidate']['reference_sha256']!=sha(repo/'tests/fixtures/decoder_layer/checksums.json'):
        raise ValueError('decoder reference identity changed')
    expected={'h{h}_i{i}_nq{nq}_nk{nk}_d{d}_t{rows}_s{seed}_{mutation}'.format(**s):s
              for s in anchor['specification']['holdout']}
    expected['checkpoint_holdout']=dict(h=896,nq=14,nk=2,d=64,i=4864,
        rows=len(anchor['checkpoint']['holdout_token_ids']),seed=0,mutation='base')
    if {name:case['spec'] for name,case in record['cases'].items()}!=expected:
        raise ValueError('decoder reserved case census changed')
    if record['checkpoint']!=anchor['checkpoint'] or record['cases']['checkpoint_holdout']['token_ids']!=anchor['checkpoint']['holdout_token_ids']:
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
    hashes={'manifest.json':sha(root/'manifest.json')}

    for name,case in record['cases'].items():
        for label,spec in case['arrays'].items():
            path=root/name/(label+'.npy')
            if verify_arrays and sha(path)!=spec['sha256']:
                raise ValueError('decoder reserved array changed')
            hashes[name+'/'+label+'.npy']=spec['sha256']
    return record,hashes


def expected_checks(cases):
    expected=Counter()
    for name,case in cases.items():
        spec=case['spec'];t=spec['rows']
        policies=[0,7] if spec['nq']==14 and t>1 else [0]
        for policy in policies:
            for schedule,calls in case['schedules'].items():
                for call in calls:
                    p,r=call['start'],call['rows']
                    expected[name,policy,schedule,'layer','route','',p,r]+=1
                    for stage in STAGES:
                        expected[name,policy,schedule,'layer','boundary',stage,p,r]+=1
                    if schedule!='full':
                        for stage in BOUNDARIES:
                            expected[name,policy,schedule,'layer','full_vs_chunk',stage,p,r]+=1
                    for stage in ('cache_key','cache_value'):
                        label=('full_' if schedule=='full' else f'{schedule}_{p}_')+stage
                        expected[name,policy,schedule,'layer','cache',label,p,r]+=1
            for stage in STAGES:
                expected[name,policy,'full','operation','boundary',stage,0,t]+=1
            for stage in ('B_mlp','Y'):
                expected[name,policy,'full','mlp','boundary',stage,0,t]+=1
    return expected


def validate_results(path,cases):
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    observed=Counter();runtimes=set();aux=Counter();protected=Counter()
    for row in records:
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
            if row.get('failed')!=0 or row.get('elements',0)<=0:
                raise ValueError('missing exact preservation check')
            protected[row['case'],row['policy'],row['label']]+=1
            continue
        key=tuple(row.get(k,'') for k in ('case','policy','schedule','mode','kind','stage','start','rows'))
        observed[key]+=1
        spec=cases[row['case']]['spec'];r=row['rows'];h=spec['h'];stage=row.get('stage','')
        if kind=='route':
            if row['attention']!=(4 if r==1 else 6) or row['mlp']!=(row['policy'] if r>1 else 0):
                raise ValueError('decoder route changed')
            if row['backend']!='metal' or not row['device'].startswith('Apple '):
                raise ValueError('decoder execution did not prove Metal')
            runtimes.add((row['device'],row['backend']))
        elif kind=='cache':
            if row['elements']!=(row['start']+r)*spec['nk']*spec['d'] or any(row.get(k)is not True for k in ('prefix_exact','append_exact','inactive_exact')):
                raise ValueError('incomplete cache preservation check')
        elif kind in ('boundary','full_vs_chunk'):
            width=spec['i'] if stage in ('G','U','A','S') else spec['nk']*spec['d'] if stage in ('K_raw','V_raw','K_rot') else h
            if kind=='boundary' and row.get('inactive_exact')is not True:
                raise ValueError('missing workspace guard check')
            if row['elements']!=r*width:
                raise ValueError('decoder incomplete element coverage')
            if (row['mode']=='operation' or stage in BOUNDARIES) and row.get('failed')!=0:
                raise ValueError('missing decoder gate')
    labels=('xb','aw_norm','aw_qkv','aw_bias','aw_output','mw_norm','mw_gate','mw_up','mw_down','a_cosine','a_sine')
    expected_protected=Counter((name,policy,label) for name,case in cases.items()
        for policy in ([0,7] if case['spec']['nq']==14 and case['spec']['rows']>1 else [0]) for label in labels)
    if protected!=expected_protected:
        raise ValueError('incomplete protected decoder storage coverage')
    if observed!=expected_checks(cases) or len(runtimes)!=1:
        raise ValueError('decoder missing, duplicate or unexpected numerical coverage')
    negatives={'second residual uses X','second residual omitted','first residual omitted',
               'second norm uses X','wrong norm weights','wrong absolute RoPE position',
               'mask exposes future rows','cache prefix changed'}
    if aux!=Counter({n:1 for n in negatives}):
        raise ValueError('incomplete decoder negative controls')
    return dict(checks=sum(observed.values()),runtime=dict(zip(('device','backend'),next(iter(runtimes)))))


def evaluate(binary,fixtures,output):
    binary,fixtures,output=map(lambda p:Path(p).resolve(),(binary,fixtures,output))
    ensure_record_location(output);candidate=verify_build(binary)
    manifest,hashes=holdout_manifest(fixtures)
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
        record.update(validate_results(results,manifest['cases']),status='passed')
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:
        record.update(finished_utc=utc_now(),output_sha256=sha(log) if log.exists() else None,
                      checks_sha256=sha(results) if results.exists() else None)
        write(output/'evaluation.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--binary',type=Path,required=True)
    e=sub.add_parser('evaluate')
    for flag in ('binary','fixtures','output'):
        e.add_argument('--'+flag,type=Path,required=True)
    args=p.parse_args()
    if args.command=='build':build(args.binary)
    else:evaluate(args.binary,args.fixtures,args.output)


if __name__=='__main__':main()
