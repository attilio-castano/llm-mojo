"""Execute a receipted model driver against complete development references."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ._repository import environment_tool, repository_root
from .mlp_validation import source_identity, sha, write
from .model_assets import verify_prepared

CONSISTENCY_BOUNDARIES = ({f'hidden_{i}' for i in range(25)} | {'final_norm', 'logits'} |
                          {f'cache_{kind}_{i}' for kind in ('key', 'value') for i in range(24)})


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
    command=[environment_tool('mojo'),'build','-I','src','-I','tests','tests/model_driver.mojo','-o',str(binary)]
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


def consistency_accuracy(actual, expected, gate):
    """Pointwise and per-row error; accuracy is separate from byte equality."""
    result = compare(actual, expected, gate)
    error = (actual.astype(np.float64)-expected.astype(np.float64)).reshape(len(actual), -1)
    signal = np.linalg.norm(expected.astype(np.float64).reshape(len(expected), -1), axis=1)
    delta = np.linalg.norm(error, axis=1)
    relative = np.divide(delta, signal, out=np.zeros_like(delta), where=signal != 0)
    zero_ok = not np.any((signal == 0) & (delta != 0))
    exact = actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    result.update(relative_rms=float(relative.max()), exact=exact)
    result['passed'] = (result['passed'] and zero_ok and result['relative_rms'] <= gate['relative_rms']
                        and (not gate['exact'] or exact))
    return result


def required_consistency_schedules(length, declaration):
    """Independently reconstruct required schedules, not just reported ones."""
    settings = declaration['schedules']
    required = {(length,)}
    if length <= settings['exhaustive_through']:
        from itertools import combinations
        for count in range(length):
            for interior in combinations(range(1,length),count):
                cuts = (0,*interior,length)
                required.add(tuple(b-a for a,b in zip(cuts,cuts[1:])))
    elif length <= settings['tokenwise_through']:
        required.add((1,)*length)
    if length > 17:
        required.add((length-sum(settings['long_suffix']),*settings['long_suffix']))
    if length > 2:
        required.add((1,length-2,1))
    remaining, chunks = length, []
    while remaining:
        rows = min(remaining,settings['ragged_cycle'][len(chunks)%len(settings['ragged_cycle'])])
        chunks.append(rows)
        remaining -= rows
    required.add(tuple(chunks))
    return required


def verify_consistency_observations(report, path, declaration=None):
    if declaration is None:
        declaration = json.loads((repository_root()/'tests/fixtures/model_consistency.json').read_text())
    required = Counter()
    for case in report['cases']:
        if set(case['arrays']) != CONSISTENCY_BOUNDARIES:
            raise ValueError('incomplete canonical reference boundary census')
        seen = set()
        for schedule in case['schedules']:
            chunks = tuple(schedule['rows'])
            if (not chunks or min(chunks) < 1 or sum(chunks) != case['length'] or chunks in seen
                    or schedule['checks'] != len(chunks)*75 or schedule['failures']):
                raise ValueError('invalid or failed reference schedule')
            seen.add(chunks)
            start = 0
            for rows in chunks:
                for stage in CONSISTENCY_BOUNDARIES:
                    required[(case['length'],case['seed'],chunks,start,rows,stage)] += 1
                start += rows
        if seen != required_consistency_schedules(case['length'],declaration):
            raise ValueError('incomplete declared reference schedule census')
    observed = Counter()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['exact'] is not True or row['max_abs'] != 0:
                raise ValueError('failed exact reference observation')
            observed[(row['length'],row['seed'],tuple(row['schedule']),row['start'],row['rows'],row['stage'])] += 1
    if observed != required:
        raise ValueError('incomplete or duplicated reference observations')


def consistency_reference(directory):
    """Verify the complete declared qualification before candidate exposure."""
    root = repository_root()
    contract = root/'tests/fixtures/model_consistency.json'
    declaration = json.loads(contract.read_text())
    report = json.loads((directory/'qualification.json').read_text())
    if (report.get('kind') != 'model_consistency_reference' or report.get('passed') is not True
            or report.get('candidate_outputs_observed') is not False
            or report.get('reserved_outputs_observed') is not False
            or report.get('version') != declaration['version']):
        raise ValueError('canonical reference has not qualified')
    for name, digest in report['source'].items():
        if sha(root/name) != digest:
            raise ValueError('canonical reference source changed: '+name)
    if sha(directory/'observations.jsonl') != report['observations_sha256']:
        raise ValueError('reference qualification observations changed')
    if [{k: c[k] for k in ('length', 'seed')} for c in report['cases']] != declaration['development_cases']:
        raise ValueError('incomplete reference qualification case census')
    verify_consistency_observations(report, directory/'observations.jsonl', declaration)
    for case in report['cases']:
        for record in case['arrays'].values():
            if sha(directory/record['path']) != record['sha256']:
                raise ValueError('canonical reference array changed')
    return declaration, report


def evaluate_consistency(binary, reference, output, length, prepared=None):
    """Initial full-forward accuracy gate for the consistent native candidate."""
    binary, reference, output = (Path(p).resolve() for p in (binary, reference, output))
    receipt = verify_build(binary)
    declaration, qualified = consistency_reference(reference)
    prepared, _ = verify_prepared(prepared)
    case = next(c for c in qualified['cases'] if c['length'] == length)
    output.mkdir(parents=True, exist_ok=False)
    native = output/'call_0'
    native.mkdir()
    command = [str(binary), str(prepared), ','.join(map(str, case['ids'])), str(length), '20', str(output)]
    with (output/'execution.log').open('w') as log:
        subprocess.run(command, cwd=repository_root(), env=environment(), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    log = (output/'execution.log').read_text()
    if ('model device Apple M4 Pro backend metal' not in log
            or f'cache_length {length} submitted_layer_rows {length*24}' not in log):
        raise ValueError('missing native Metal identity or cache accounting')
    records = []
    required = CONSISTENCY_BOUNDARIES
    if set(case['arrays']) != required:
        raise ValueError('incomplete canonical reference boundary census')
    for name in sorted(required):
        expected = np.load(reference/case['arrays'][name]['path'], allow_pickle=False)
        if name.startswith('cache_'):
            actual = bf16(native/(name+'.bin'), (min(4096, length+3), 2, 64))
            kind, layer = name.split('_')[1:]
            appended = bf16(native/(f'append_{kind}_{layer}.bin'), (length, 2, 64))
            storage = actual[:length].tobytes() == appended.tobytes() and np.all(actual[length:] == 123)
            records.append(dict(stage=name+'_storage', passed=bool(storage)))
            actual = actual[:length]
        else:
            if name == 'final_norm':
                expected = expected[-1:]
            actual = bf16(native/(name+'.bin'), expected.shape)
        records.append(dict(stage=name, **consistency_accuracy(actual, expected, declaration['accuracy']['gates'][name])))
    verify_build(binary)
    # Recheck input identities after execution as well as before it.
    consistency_reference(reference)
    report = dict(kind='model_consistency_accuracy', build=receipt, command=command,
        qualification_sha256=sha(reference/'qualification.json'),
        prepared_manifest_sha256=sha(prepared/'manifest.json'), length=length,
        configuration=20, reserved_outputs_observed=False, checks=records,
        passed=all(r['passed'] for r in records))
    write(output/'evaluation.json', report)
    print('consistent native accuracy', length, 'checks', len(records),
          'failures', sum(not r['passed'] for r in records), flush=True)
    if not report['passed']:
        raise ValueError('consistent full-model accuracy gates failed')


def evaluate_operations(binary, reference, output, prepared=None):
    """Isolate all layer operations on identical, actually observed HF inputs."""
    import importlib.util
    binary, reference, output = (Path(p).resolve() for p in (binary, reference, output))
    receipt = verify_build(binary)
    prepared, prepared_manifest = verify_prepared(prepared)
    root = repository_root()
    manifest = json.loads((reference/'manifest.json').read_text())
    declaration = json.loads((root/'tests/fixtures/model_consistency.json').read_text())
    if (manifest.get('kind') != 'model_identical_operand_reference'
            or manifest['case'] != declaration['development_cases'][0]
            or manifest['observer_bitwise_equal'] is not True
            or manifest['reserved_outputs_observed'] is not False
            or manifest['source_sha256'] != sha(root/'tests/fixtures/model_reference_diagnosis.py')
            or manifest['canonical_source_sha256'] != sha(root/'tests/fixtures/model_attention_diagnosis.py')
            or manifest['reference']['source_sha256'] != sha(root/'tests/fixtures/model_reference.py')
            or manifest['contract_sha256'] != sha(root/'tests/fixtures/model_consistency.json')):
        raise ValueError('operation reference identity or scope mismatch')
    stages = ('N_att','Q_raw','K_raw','V_raw','O','B_att','Z','N_mlp','G','U','A','S','B_mlp','Y')
    required = {f'layer_{i}_{s}' for i in range(24) for s in ('X', *stages)}
    if set(manifest['arrays']) != required:
        raise ValueError('incomplete operation reference census')
    for name, record in manifest['arrays'].items():
        if sha(reference/(name+'.bin')) != record['sha256']:
            raise ValueError('operation input bytes changed')
    output.mkdir(parents=True, exist_ok=False)
    command = [str(binary), '--operations', str(prepared), str(reference), str(output)]
    with (output/'execution.log').open('w') as log:
        subprocess.run(command, cwd=root, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (output/'execution.log').read_text()
    if ('operation device Apple M4 Pro backend metal' not in log
            or any(f'completed identical-operand layer {i}\n' not in log for i in range(24))):
        raise ValueError('incomplete native operation execution')
    spec = importlib.util.spec_from_file_location('model_operation_numerics',root/'tests/fixtures/mlp/numerics.py')
    numerics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numerics)
    records, rounding = [], []
    for i in range(24):
        for stage in stages:
            name = f'layer_{i}_{stage}'
            shape = manifest['arrays'][name]['shape']
            actual, expected = bf16(output/(name+'.bin'),shape), bf16(reference/(name+'.bin'),shape)
            # These are the existing decoder identical-operand operation gates.
            gate = dict(atol=2**-7,rtol=2**-7)
            if stage == 'B_att':
                gate = dict(atol=2**-5,rtol=2**-5)
            elif stage in ('Z','S','Y'):
                gate = dict(atol=0.,rtol=0.,exact_bits=True)
            elif stage == 'A':
                gate = dict(atol=2**-133,rtol=2**-7,max_bf16_steps=1,exact_reference_zero=True)
            records.append(dict(layer=i,stage=stage,gate=gate,
                output_sha256=sha(output/(name+'.bin')),
                **numerics.differences(actual,expected,gate)))
            # Offline exact rational sums diagnose rounding; they are never
            # inference operands or an alternative acceptance oracle.
            if stage in ('Q_raw','K_raw','V_raw','G','U','B_mlp'):
                from fractions import Fraction
                source_stage = 'N_att' if stage.endswith('_raw') else 'S' if stage == 'B_mlp' else 'N_mlp'
                tensor = 'qkv' if stage.endswith('_raw') else {'G':'gate','U':'up','B_mlp':'down'}[stage]
                tensor_name = f'layer_{i}_{tensor}'
                changed = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
                if len(changed):
                    weights = bf16(prepared/(tensor_name+'.bin'),prepared_manifest['tensors'][tensor_name]['shape'])
                    inputs = bf16(reference/f'layer_{i}_{source_stage}.bin',(weights.shape[1],))
                    offset = {'Q_raw':0,'K_raw':896,'V_raw':1024}.get(stage,0)
                    for coordinate in changed:
                        total = sum((Fraction(float(x))*Fraction(float(w))
                                     for x,w in zip(inputs,weights[offset+coordinate])),Fraction())
                        if tensor == 'qkv':
                            bias_name = f'layer_{i}_bias'
                            bias = bf16(prepared/(bias_name+'.bin'),prepared_manifest['tensors'][bias_name]['shape'])
                            total += Fraction(float(bias[offset+coordinate]))
                        av, rv = Fraction(float(actual.flat[coordinate])), Fraction(float(expected.flat[coordinate]))
                        rounding.append(dict(layer=i,stage=stage,coordinate=int(coordinate),
                            exact_numerator=total.numerator,exact_denominator=total.denominator,
                            exact_sum=float(total),native=float(av),upstream=float(rv),
                            midpoint_distance=float(total-(av+rv)/2),
                            closer_to_exact='native' if abs(total-av)<abs(total-rv) else
                                'upstream' if abs(total-rv)<abs(total-av) else 'tie'))
    verify_build(binary)
    for name, record in manifest['arrays'].items():
        if sha(reference/(name+'.bin')) != record['sha256']:
            raise ValueError('operation input changed during execution')
    result = dict(kind='model_identical_operand_evaluation',build=receipt,command=command,
        reference_sha256=sha(reference/'manifest.json'),checks=records,rounding=rounding,
        prepared_manifest_sha256=sha(prepared/'manifest.json'),
        reserved_outputs_observed=False,passed=all(r['failed']==0 for r in records))
    write(output/'evaluation.json',result)
    print('Identical-operand checks',len(records),'failed',sum(r['failed'] for r in records),flush=True)
    if not result['passed']:
        raise ValueError('identical-operand accuracy failure')


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
    c=sub.add_parser('consistency');c.add_argument('--binary',required=True,type=Path)
    c.add_argument('--reference',required=True,type=Path);c.add_argument('--output',required=True,type=Path)
    c.add_argument('--length',required=True,type=int);c.add_argument('--prepared',type=Path)
    o=sub.add_parser('operations');o.add_argument('--binary',required=True,type=Path)
    o.add_argument('--reference',required=True,type=Path);o.add_argument('--output',required=True,type=Path)
    o.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    if args.command=='build':build(args.binary)
    elif args.command=='consistency':evaluate_consistency(args.binary,args.reference,args.output,args.length,args.prepared)
    elif args.command=='operations':evaluate_operations(args.binary,args.reference,args.output,args.prepared)
    else:evaluate(args.binary,args.reference,args.output,args.configurations,args.prepared)


if __name__=='__main__':main()
