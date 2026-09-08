"""Generate development decoder references; reserved inputs are never evaluated."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

from contract import DEVELOPMENT, GATES, STAGES, case_id, schedules, specification
from reference import UpstreamDecoder, inputs, same_bits, differences, provenance, checkpoint, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]/'build/oracle_data/decoder_layer'
ANCHOR = HERE/'checksums.json'
EVIDENCE = HERE/'development.json.gz'


def write_json(path, record):
    Path(path).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def freeze(record):
    if ANCHOR.exists() or EVIDENCE.exists():
        raise ValueError('refusing to overwrite frozen decoder evidence')
    raw=(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    EVIDENCE.write_bytes(gzip.compress(raw,mtime=0))
    write_json(ANCHOR,dict(schema=1,status=record['status'],cases=len(record['cases']),
        specification=record['specification'],upstream=record['upstream'],sources=record['sources'],
        checkpoint=record.get('checkpoint'),evidence_sha256=sha(EVIDENCE),
        uncompressed_sha256=hashlib.sha256(raw).hexdigest()))


def load_frozen():
    summary=json.loads(ANCHOR.read_text())
    payload=EVIDENCE.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=summary['evidence_sha256']:
        raise ValueError('compressed decoder evidence changed')
    raw=gzip.decompress(payload)
    if hashlib.sha256(raw).hexdigest()!=summary['uncompressed_sha256']:
        raise ValueError('uncompressed decoder evidence changed')
    record=json.loads(raw)
    if summary['schema']!=1 or record['status']!='complete' or len(record['cases'])!=summary['cases']:
        raise ValueError('incomplete decoder reference evidence')
    if any(record.get(k)!=summary[k] for k in ('specification','upstream','sources','checkpoint','status')):
        raise ValueError('decoder summary disagrees with evidence')
    return record


def sources():
    paths = [HERE/name for name in ('contract.py','reference.py','generate.py')]
    paths += [HERE.parent/'generate.py.lock',HERE.parent/'mlp/numerics.py',HERE.parent/'attention_sublayer/checkpoint_checksums.json']
    return {str(p.relative_to(HERE.parents[2])):sha(p) for p in paths}


def verify_arrays(root, record):
    for name,case in record['cases'].items():
        for key,description in case['arrays'].items():
            path=root/name/(key+'.npy')
            if sha(path)!=description['sha256']:
                raise ValueError('decoder array changed: '+str(path))


def execute_case(directory, name, spec, data, token_ids=None):
    directory.mkdir(parents=True,exist_ok=False)
    record = dict(spec=spec,arrays={},schedules={},token_ids=token_ids)
    def save(key, value):
        path = directory/(key+'.npy')
        np.save(path,value.astype(np.float32),allow_pickle=False)
        record['arrays'][key] = dict(shape=list(value.shape),dtype='float32',logical_dtype='bfloat16',sha256=sha(path))
    for key,value in data.items():
        save('input_'+key,value)
    full = UpstreamDecoder(spec,data).run(data['X'])
    plain = UpstreamDecoder(spec,data).run(data['X'],observe=False)
    for key in ('Y','cache_key','cache_value'):
        if not same_bits(full[key],plain[key]):
            raise ValueError('observation changed upstream '+key)
    del plain
    for key,value in full.items():
        save('full_'+key,value)
    for schedule,chunks in schedules(spec['rows']).items():
        if schedule=='full':
            record['schedules'][schedule] = [dict(start=0,rows=spec['rows'])]
            continue
        runner = UpstreamDecoder(spec,data)
        p=0; old_k=old_v=None; calls=[]
        for r in chunks:
            c=runner.run(data['X'][p:p+r])
            if old_k is not None and (not same_bits(old_k,c['cache_key'][:p]) or not same_bits(old_v,c['cache_value'][:p])):
                raise ValueError('upstream mutated cache prefix')
            old_k,old_v=c['cache_key'].copy(),c['cache_value'].copy()
            checks={}
            for key in STAGES:
                expected=full[key][:p+r] if key in ('cosine','sine','cache_key','cache_value') else full[key][p:p+r]
                checks[key]=differences(c[key],expected,GATES.get(key))
                save(f'{schedule}_{p}_{key}',c[key])
            call=dict(start=p,rows=r,checks=checks)
            calls.append(call)
            # Save evidence before enforcing the qualification gate.
            record['schedules'][schedule]=calls
            write_json(directory/'record.json',record)
            failed=[key for key in GATES if checks[key]['failed']]
            if failed:
                raise ValueError(f'upstream full/chunk qualification failed: {name} {schedule} P={p} R={r}: {failed}')
            p+=r
    write_json(directory/'record.json',record)
    print('decoder reference',name,'qualified',flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--freeze-development',action='store_true')
    parser.add_argument('--checkpoint-assets',type=Path)
    parser.add_argument('--inspect-assets',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.self_test:
        subprocess.run([sys.executable,str(HERE/'test_reference.py')],check=True)
        return
    torch.set_num_threads(1)
    if args.inspect_assets:
        if args.checkpoint_assets is None:
            parser.error('--inspect-assets requires --checkpoint-assets')
        cases,origin=checkpoint(args.checkpoint_assets)
        print(json.dumps(dict(cases=[(name,spec['rows']) for name,spec,_,_ in cases],
            source=origin['source'],tensors=len(origin['tensors']),
            holdout_token_count=len(origin['holdout_token_ids']),
            holdout_token_ids_sha256=origin['holdout_token_ids_sha256']),indent=2))
        return
    initial=dict(specification=specification(),upstream=provenance(),sources=sources())
    initial=json.loads(json.dumps(initial))
    if args.freeze_development:
        if ANCHOR.exists() or EVIDENCE.exists():
            raise ValueError('refusing to overwrite decoder anchors')
        if args.checkpoint_assets is None:
            raise ValueError('initial freeze requires verified local checkpoint assets')
    else:
        frozen=load_frozen()
        if any(frozen[k]!=initial[k] for k in initial):
            raise ValueError('decoder reference source/contract changed')
    root=args.output
    if root.exists():
        if args.freeze_development:
            raise ValueError('refusing to overwrite decoder generation; choose a fresh --output directory')
        manifest=json.loads((root/'manifest.json').read_text())
        expected={k:v for k,v in frozen['cases'].items() if args.checkpoint_assets or not k.startswith('checkpoint_')}
        present={k:v for k,v in manifest['cases'].items() if args.checkpoint_assets or not k.startswith('checkpoint_')}
        if manifest['status']!='complete' or any(manifest[k]!=initial[k] for k in initial) or present!=expected:
            raise ValueError('existing decoder capture is incomplete or changed; use a fresh --output')
        if args.checkpoint_assets:
            _,origin=checkpoint(args.checkpoint_assets)
            if origin!=frozen['checkpoint']:
                raise ValueError('checkpoint input identity changed')
        verify_arrays(root,dict(cases=present))
        print('Decoder reference arrays match frozen anchors.',flush=True)
        return
    root.mkdir(parents=True)
    record=dict(initial,status='running',cases={})
    write_json(root/'manifest.json',record)
    try:
        for spec in DEVELOPMENT:
            name=case_id(spec)
            record['cases'][name]=execute_case(root/name,name,spec,inputs(spec))
        if args.checkpoint_assets:
            cases,origin=checkpoint(args.checkpoint_assets)
            record['checkpoint']=origin
            for name,spec,data,ids in cases:
                record['cases'][name]=execute_case(root/name,name,spec,data,ids)
        if sources()!=initial['sources']:
            raise ValueError('decoder source changed during reference generation')
        record['status']='complete'
        # Compare canonical JSON types, including checkpoint tensor shapes.
        record=json.loads(json.dumps(record))
        if args.freeze_development:
            freeze(record)
        else:
            expected={k:v for k,v in frozen['cases'].items() if args.checkpoint_assets or not k.startswith('checkpoint_')}
            if record['cases']!=expected or (args.checkpoint_assets and record['checkpoint']!=frozen['checkpoint']):
                raise ValueError('decoder fixture differs from frozen reference')
        write_json(root/'manifest.json',record)
    except Exception as exc:
        record.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        write_json(root/'manifest.json',record)
        raise


if __name__=='__main__':
    main()
