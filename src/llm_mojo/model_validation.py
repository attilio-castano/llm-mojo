"""Execute a receipted model driver against complete development references."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ._repository import environment_tool, repository_root
from .mlp_validation import source_identity, sha, write
from .model_assets import verify_prepared


def environment():
    return {k:v for k,v in os.environ.items() if k!='MODULAR_DEBUG'}


def build(binary):
    binary=Path(binary).resolve()
    receipt=Path(str(binary)+'.provenance.json')
    if binary.exists() or receipt.exists():
        raise ValueError('refusing to overwrite a model build')
    source=source_identity()
    if source['repository']['dirty']:
        raise ValueError('model numerical build requires clean source')
    binary.parent.mkdir(parents=True,exist_ok=True)
    command=[environment_tool('mojo'),'build','-I','src','tests/model_driver.mojo','-o',str(binary)]
    subprocess.run(command,cwd=repository_root(),env=environment(),check=True)
    if source_identity()!=source:
        raise ValueError('source changed during model compilation')
    write(receipt,dict(kind='model-development-build',source=source,command=command,binary_sha256=sha(binary)))


def verify_build(binary):
    receipt=json.loads(Path(str(binary)+'.provenance.json').read_text())
    if (receipt.get('kind')!='model-development-build' or receipt['binary_sha256']!=sha(binary)
            or receipt['source']!=source_identity() or receipt['source']['repository']['dirty']):
        raise ValueError('model source or executable differs from build receipt')
    return receipt


def bf16(path, shape):
    data=np.fromfile(path,dtype='<u2')
    if data.size!=int(np.prod(shape)):
        raise ValueError('incomplete native capture: '+str(path))
    return (data.astype(np.uint32)<<16).view(np.float32).reshape(shape)


def compare(actual,expected,gate):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite or mismatched numerical boundary')
    error=np.abs(actual-expected)
    scaled=error/(gate['atol']+gate['rtol']*np.abs(expected))
    return dict(max_abs=float(error.max()),max_scaled=float(scaled.max()),passed=bool(np.all(scaled<=1)))


def evaluate(binary,reference,output,configurations,prepared=None):
    binary=Path(binary).resolve();reference=Path(reference).resolve();output=Path(output).resolve()
    receipt=verify_build(binary)
    prepared,model=verify_prepared(prepared)
    manifest_path=reference/'manifest.json'
    manifest_hash=sha(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    if (manifest['kind']!='model_development_reference'
            or manifest['source_sha256']!=sha(repository_root()/'tests/fixtures/model_reference.py')):
        raise ValueError('reference identity changed')
    qualification=manifest.get('qualification',{})
    if (not qualification.get('passed') or qualification.get('gates')!=manifest['gates']
            or qualification.get('source_sha256')!=manifest['source_sha256']
            or qualification.get('candidate_outputs_observed') is not False):
        raise ValueError('reference has not passed independent qualification')
    ids=manifest['ids']
    if len(configurations)!=len(manifest['schedule']):
        raise ValueError('supply exactly one configuration per scheduled call')
    # Complete array verification precedes execution.
    for calls in manifest['cases'].values():
        for call in calls:
            for record in call['arrays'].values():
                path=reference/record['path']
                if sha(path)!=record['sha256']:
                    raise ValueError('reference array changed')
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for mode in ('full','scheduled'):
        schedule=[len(ids)] if mode=='full' else manifest['schedule']
        variants=[0] if mode=='full' else configurations
        calls=manifest['cases'][mode]
        if [c['rows'] for c in calls]!=schedule or sum(schedule)!=len(ids):
            raise ValueError('reference schedule census mismatch')
        directory=output/mode
        for i in range(len(calls)):
            (directory/f'call_{i}').mkdir(parents=True)
        command=[str(binary),str(prepared),','.join(map(str,ids)),','.join(map(str,schedule)),
                 ','.join(map(str,variants)),str(directory)]
        with (directory/'execution.log').open('w') as log:
            subprocess.run(command,cwd=repository_root(),env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
        log=(directory/'execution.log').read_text()
        if 'model device Apple M4 Pro backend metal' not in log:
            raise ValueError('missing reference-device Metal execution identity')
        previous={}
        for i,call in enumerate(calls):
            start,rows=call['start'],call['rows']
            if f'cache_length {start+rows} submitted_layer_rows {(start+rows)*24}' not in log:
                raise ValueError('missing execution/cache accounting')
            native=directory/f'call_{i}'
            required={f'hidden_{layer}' for layer in range(25)}|{'final_norm','logits'}
            required|={f'cache_{kind}_{layer}' for kind in ('key','value') for layer in range(24)}
            if set(call['arrays'])!=required:
                raise ValueError('incomplete reference boundary census')
            for name in sorted(required):
                record=call['arrays'][name]
                expected=np.load(reference/record['path'],allow_pickle=False)
                if name.startswith('cache_'):
                    actual=bf16(native/(name+'.bin'),(min(4096,len(ids)+3),2,64))
                    kind,layer=name.split('_')[1:]
                    appended=bf16(native/(f'append_{kind}_{layer}.bin'),(rows,2,64))
                    preserved = start==0 or np.array_equal(actual[:start],previous[name][:start])
                    exact=preserved and np.array_equal(actual[start:start+rows],appended) and np.all(actual[start+rows:]==123)
                    records.append(dict(mode=mode,call=i,stage=name+'_storage',passed=bool(exact)))
                    previous[name]=actual
                    actual=actual[:start+rows]
                else:
                    if name=='final_norm': expected=expected[-1:]
                    actual=bf16(native/(name+'.bin'),expected.shape)
                result=compare(actual,expected,manifest['gates']['logits' if name=='logits' else 'hidden'])
                records.append(dict(mode=mode,call=i,stage=name,**result))
        # No numerical candidate is adjusted or promoted by this development tool.
    verify_build(binary)
    if sha(manifest_path)!=manifest_hash:
        raise ValueError('reference changed during execution')
    report=dict(kind='model_development_evaluation',build=receipt,reference_sha256=manifest_hash,
                prepared_manifest_sha256=sha(prepared/'manifest.json'),configurations=configurations,
                checks=records,passed=all(r['passed'] for r in records))
    write(output/'evaluation.json',report)
    if not report['passed']:
        raise ValueError('full-model development numerical checks failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--binary',required=True,type=Path)
    e=sub.add_parser('evaluate');e.add_argument('--binary',required=True,type=Path)
    e.add_argument('--reference',required=True,type=Path);e.add_argument('--output',required=True,type=Path)
    e.add_argument('--configurations',required=True,nargs='+',type=int)
    e.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    if args.command=='build':build(args.binary)
    else:evaluate(args.binary,args.reference,args.output,args.configurations,args.prepared)


if __name__=='__main__':main()
