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


def build(binary, generation=False):
    binary=Path(binary).resolve()
    receipt=Path(str(binary)+'.provenance.json')
    if binary.exists() or receipt.exists():
        raise ValueError('refusing to overwrite a model build')
    source=source_identity()
    if source['repository']['dirty']:
        raise ValueError('model numerical build requires clean source')
    binary.parent.mkdir(parents=True,exist_ok=True)
    entry='src/llm_mojo/generate_cli.mojo' if generation else 'tests/model_driver.mojo'
    command=[environment_tool('mojo'),'build','-I','src','-I','tests',entry,'-o',str(binary)]
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


def numerical_diagnostic(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite or mismatched diagnostic boundary')
    a,b = actual.astype(np.float64),expected.astype(np.float64)
    error = np.linalg.norm((a-b).reshape(len(a),-1),axis=1)
    norm = np.linalg.norm(b.reshape(len(b),-1),axis=1)
    relative = np.divide(error,norm,out=np.zeros_like(error),where=norm!=0)
    return dict(max_abs=float(np.max(np.abs(a-b))),max_row_relative_l2=float(relative.max()),
        zero_reference_nonzero_rows=int(np.sum((norm==0)&(error!=0))),exact=actual.tobytes()==expected.tobytes())


def runtime_specification(output, generations=None):
    declaration=json.loads((repository_root()/'tests/fixtures/model_runtime.json').read_text())
    cases=[]
    if generations is not None:
        report=json.loads(Path(generations).read_text())
        for i,r in enumerate(report['records']):
            consumed=r['prompt_ids']+r['tokens'][:-1]
            remaining=len(r['prompt_ids']);schedule=[]
            while remaining:
                rows=min(remaining,r['chunk_rows'] or remaining)
                schedule.append(rows);remaining-=rows
            schedule += [1]*(len(r['tokens'])-1)
            cases.append(dict(name=f'history-{i}',ids=consumed,schedule=schedule,
                configurations=[0]*len(schedule),full_configurations=[0]))
        declaration['prompts']=[]
        write(output,dict(declaration=declaration,cases=cases,measurements=[],history_source_sha256=sha(generations)))
        return
    def ids(length):
        return np.random.default_rng(declaration['seed']+length).integers(0,151643,size=length).tolist()
    for length in declaration['lengths']:
        schedule=[1]*length if length<=17 else [length-17,16,1]
        cases.append(dict(name=f'length-{length}',ids=ids(length),schedule=schedule,
            configurations=[0]*len(schedule),full_configurations=[0]))
    for w in declaration['measurements']:
        prefix=w['total']-w['rows']
        for candidate in w['candidates']:
            cases.append(dict(name=f"cell-{w['rows']}-{w['total']}-{candidate}",ids=ids(w['total']),
                schedule=[prefix,w['rows']],configurations=[0,candidate],full_configurations=[0]))
    write(output,dict(declaration=declaration,cases=cases,measurements=declaration['measurements']))


def prediction_diagnostic(actual, expected):
    a,b=actual.astype(np.float64).ravel(),expected.astype(np.float64).ravel()
    def log_softmax(x):
        x=x-x.max()
        return x-np.log(np.exp(x).sum())
    la,lb=log_softmax(a),log_softmax(b)
    p,q=np.exp(la),np.exp(lb)
    ordered=np.sort(b)
    return dict(kl_nats=float(np.sum(q*(lb-la))),total_variation=float(np.abs(p-q).sum()/2),
        token=int(a.argmax()),reference_token=int(b.argmax()),reference_margin=float(ordered[-1]-ordered[-2]))


def storage_diagnostic(actual, appended, previous, start, rows):
    # Byte equality preserves signed zero, unlike numeric array_equal.
    return dict(prefix=start==0 or actual[:start].tobytes()==previous[:start].tobytes(),
        append=actual[start:start+rows].tobytes()==appended.tobytes(),
        inactive=bool(np.all(actual[start+rows:]==123)))


def diagnose(binary, reference, output, prepared=None):
    """Execute every declared schedule; numerical differences remain measurements."""
    binary,reference,output=map(lambda p:Path(p).resolve(),(binary,reference,output))
    receipt=verify_build(binary)
    prepared,_=verify_prepared(prepared)
    manifest=json.loads((reference/'manifest.json').read_text())
    from .tokenizer_assets import SOURCE_SHA
    if manifest.get('tokenizer_sha256')!=SOURCE_SHA: raise ValueError('wrong diagnostic tokenizer')
    if manifest['kind']!='model-diagnostic-reference-v1':
        raise ValueError('wrong diagnostic reference kind')
    for name,digest in manifest['sources'].items():
        if sha(repository_root()/name)!=digest: raise ValueError('diagnostic source changed')
    for case in manifest['cases']:
        for calls in case['modes'].values():
            for call in calls:
                if set(call['arrays'])!=CONSISTENCY_BOUNDARIES: raise ValueError('incomplete boundary census')
                for record in call['arrays'].values():
                    if sha(reference/record['path'])!=record['sha256']: raise ValueError('diagnostic array changed')
    output.mkdir(parents=True,exist_ok=False)
    diagnostics,storage=[],[]
    for case,spec in zip(manifest['cases'],manifest['specification']['cases'],strict=True):
        if case['name']!=spec['name'] or case['ids']!=spec['ids']: raise ValueError('case binding mismatch')
        full_native={}
        for mode in ('full','scheduled'):
            calls=case['modes'][mode]
            schedule=[len(case['ids'])] if mode=='full' else spec['schedule']
            variants=spec['full_configurations'] if mode=='full' else [None]
            if [c['rows'] for c in calls]!=schedule or sum(schedule)!=len(case['ids']):
                raise ValueError('diagnostic schedule mismatch')
            for variant in variants:
                configs=[variant] if mode=='full' else spec['configurations']
                if len(configs)!=len(schedule): raise ValueError('missing call configuration')
                directory=output/case['name']/(mode+('-'+str(variant) if variant is not None else ''))
                for index in range(len(calls)): (directory/f'call_{index}').mkdir(parents=True)
                command=[str(binary),str(prepared),','.join(map(str,case['ids'])),','.join(map(str,schedule)),
                    ','.join(map(str,configs)),str(directory)]
                with (directory/'execution.log').open('w') as log:
                    subprocess.run(command,cwd=repository_root(),env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
                log=(directory/'execution.log').read_text()
                if 'model device Apple M4 Pro backend metal' not in log: raise ValueError('missing Metal identity')
                previous={}
                for index,call in enumerate(calls):
                    start,rows=call['start'],call['rows']
                    if start!=sum(schedule[:index]): raise ValueError('diagnostic position mismatch')
                    if f'call {index} token ' not in log or f'cache_length {start+rows} submitted_layer_rows {(start+rows)*24}' not in log:
                        raise ValueError('missing model accounting')
                    native=directory/f'call_{index}'
                    for name,record in sorted(call['arrays'].items()):
                        expected=np.load(reference/record['path'],allow_pickle=False)
                        tag=dict(case=case['name'],mode=mode,configuration=configs[index],call=index,start=start,rows=rows,stage=name)
                        if name.startswith('cache_'):
                            actual=bf16(native/(name+'.bin'),(min(4096,len(case['ids'])+3),2,64))
                            kind,layer=name.split('_')[1:]
                            appended=bf16(native/(f'append_{kind}_{layer}.bin'),(rows,2,64))
                            check=storage_diagnostic(actual,appended,previous.get(name),start,rows)
                            storage.append(dict(**tag,**check))
                            previous[name]=actual
                            actual=actual[:start+rows]
                        else:
                            if name=='final_norm': expected=expected[-1:]
                            actual=bf16(native/(name+'.bin'),expected.shape)
                        metrics=numerical_diagnostic(actual,expected)
                        if name=='hidden_0' and not metrics['exact']: raise ValueError('embedding lookup mismatch')
                        if name=='logits': metrics.update(prediction_diagnostic(actual,expected))
                        diagnostics.append(dict(**tag,comparison='hf_same_history',**metrics))
                        if mode=='full' and variant==0: full_native[name]=actual.copy()
                        if mode=='scheduled' and name in full_native:
                            full=full_native[name]
                            if name.startswith('cache_'): full=full[:start+rows]
                            elif name in ('logits','final_norm'):
                                if start+rows!=len(case['ids']): continue
                            else: full=full[start:start+rows]
                            metrics=numerical_diagnostic(actual,full)
                            if name=='logits': metrics.update(prediction_diagnostic(actual,full))
                            diagnostics.append(dict(**tag,comparison='native_full',**metrics))
                print('diagnosed',case['name'],mode,configs,flush=True)
    verify_build(binary)
    report=dict(kind='model-runtime-diagnostics-v1',build=receipt,reference_sha256=sha(reference/'manifest.json'),
        specification=manifest['specification'],prepared_manifest_sha256=sha(prepared/'manifest.json'),
        diagnostics=diagnostics,storage=storage,invariants_passed=all(r['prefix'] and r['append'] and r['inactive'] for r in storage),
        numerical_policy='diagnostic only; no full-model error gate')
    write(output/'result.json',report)
    if not report['invariants_passed']: raise ValueError('cache storage invariant failed')


def benchmark(binary, specification, output, prepared=None):
    from .benchmarks.environment import stable_environment, conditions_snapshot, require_ac
    binary,output=Path(binary).resolve(),Path(output).resolve()
    receipt=verify_build(binary)
    prepared,_=verify_prepared(prepared)
    spec=json.loads(Path(specification).read_text())
    output.mkdir(parents=True,exist_ok=False)
    samples,conditions=[],[]
    hardware=stable_environment()
    for block in range(4):
        before=conditions_snapshot();require_ac(before)
        plan=[]
        workloads=spec['measurements'][::-1] if block in (1,2) else spec['measurements']
        for workload in workloads:
            arms=[0,0,*workload['candidates']]
            order=list(enumerate(arms))
            if block in (1,2): order.reverse()
            for arm,config in order:
                plan.append([workload['total']-workload['rows'],workload['rows'],config,block,arm])
        path=output/f'block_{block}.txt'
        path.write_text(''.join(','.join(map(str,row))+'\n' for row in plan))
        command=[str(binary),'--bench',str(prepared),str(path),'10','10']
        with (output/f'block_{block}.log').open('w') as log:
            subprocess.run(command,cwd=repository_root(),env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
        lines=(output/f'block_{block}.log').read_text().splitlines()
        if 'model device Apple M4 Pro backend metal' not in lines: raise ValueError('missing Metal identity')
        block_rows=[]
        for line in lines:
            if line.startswith('sample '):
                values=list(map(int,line.split()[1:]))
                block_rows.append(dict(zip(('block','arm','prefix','rows','configuration','sample','nanoseconds'),values,strict=True)))
        expected=Counter(tuple(row+[sample]) for row in plan for sample in range(10))
        actual=Counter((r['prefix'],r['rows'],r['configuration'],r['block'],r['arm'],r['sample']) for r in block_rows)
        if expected!=actual or any(r['nanoseconds']<=0 for r in block_rows): raise ValueError('incomplete benchmark census')
        samples.extend(block_rows)
        conditions.append(dict(before=before,after=conditions_snapshot()))
        print('measured block',block,len(block_rows),'samples',flush=True)
    verify_build(binary)
    write(output/'result.json',dict(kind='model-runtime-measurements-v1',build=receipt,
        prepared_manifest_sha256=sha(prepared/'manifest.json'),specification=spec,environment=hardware,conditions=conditions,
        model_geometry=dict(layers=24,hidden=896,intermediate=4864,query_heads=14,kv_heads=2,head_width=64,vocabulary=151936),
        dtype='BF16 stored tensors; FP32 reductions',layout='row-major activations and output-row-major affine weights',
        warmups=10,samples_per_arm=10,samples=samples,
        boundary='resident 24-layer forward including token upload and tied head, ending at device synchronization; prefix setup and greedy readback excluded'))


def generation_events(path, maximum):
    import csv
    events=list(csv.DictReader(Path(path).read_text().splitlines(),delimiter='\t'))
    def group(name): return [r for r in events if r['event']==name]
    prompt,tokens=group('prompt'),group('token')
    for values in (prompt,tokens):
        if [int(r['index']) for r in values]!=list(range(len(values))): raise ValueError('invalid token event order')
        if any(not 0<=int(r['value'])<151936 for r in values): raise ValueError('invalid reported token')
    budget=min(maximum,4096-len(prompt))
    if not 1<=len(prompt)<=4096 or not 0<=len(tokens)<=budget: raise ValueError('generation budget mismatch')
    finish=group('finish')
    if len(finish)!=1 or int(finish[0]['index'])!=len(tokens): raise ValueError('missing generation completion')
    ids=[int(r['value']) for r in tokens]
    if any(t in (151643,151645) for t in ids[:-1]): raise ValueError('generation continued after stop')
    if finish[0]['value']=='stop':
        if not ids or ids[-1] not in (151643,151645): raise ValueError('false stop')
    elif finish[0]['value']!='limit' or len(tokens)!=budget: raise ValueError('early generation termination')
    if budget:
        if len(group('device'))!=1 or group('device')[0]['value']!='Apple M4 Pro/metal': raise ValueError('missing generation Metal identity')
        expected=len(prompt)+len(tokens)-1
        if len(group('cache'))!=1 or int(group('cache')[0]['value'])!=expected: raise ValueError('generation cache mismatch')
        if len(group('submitted'))!=1 or int(group('submitted')[0]['value'])!=24*expected: raise ValueError('generation submission mismatch')
        if len(group('decode'))!=len(tokens)-1: raise ValueError('generation decode count mismatch')
    return dict(prompt_ids=[int(r['value']) for r in prompt],tokens=ids,events=events)


def require_empty_prompt_rejection(result):
    # The native Mojo runtime reports uncaught exceptions on stdout on this release.
    if result.returncode==0 or b'prompt must encode to 1..4096 tokens' not in result.stdout+result.stderr:
        raise ValueError('missing empty prompt rejection')


def generation_study(binary, output, prepared=None, policy='fast'):
    from .tokenizer_assets import ensure_prepared
    from .benchmarks.environment import stable_environment, conditions_snapshot
    binary,output=Path(binary).resolve(),Path(output).resolve()
    receipt=verify_build(binary)
    prepared,_=verify_prepared(prepared)
    tables=ensure_prepared(download=False)
    declaration=json.loads((repository_root()/'tests/fixtures/model_runtime.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    conditions_before=conditions_snapshot()
    records=[]
    for i,text in enumerate(declaration['prompts']):
        prompt=output/f'prompt_{i}.txt';prompt.write_text(text)
        for chunk in (0,4):
            report=output/f'events_{i}_{chunk}.tsv'
            command=[str(binary),str(prepared),str(tables),str(prompt),str(declaration['max_new_tokens']),str(chunk),policy,str(report)]
            result=subprocess.run(command,cwd=repository_root(),env=environment(),capture_output=True,check=True)
            generated=result.stdout.decode('utf-8',errors='strict')
            unobserved=subprocess.run(command[:-1],cwd=repository_root(),env=environment(),capture_output=True,check=True)
            if unobserved.stdout!=result.stdout: raise ValueError('reporting changed generated output')
            record=generation_events(report,declaration['max_new_tokens'])
            records.append(dict(prompt=text,chunk_rows=chunk,text=generated,unobserved_output_exact=True,**record))
            print('generated',i,'chunk',chunk,len(record['tokens']),'tokens',flush=True)
    prompt=output/'zero.txt';prompt.write_text('Hello')
    report=output/'zero.tsv'
    result=subprocess.run([str(binary),str(prepared),str(tables),str(prompt),'0','0',policy,str(report)],
        cwd=repository_root(),env=environment(),capture_output=True,check=True)
    if result.stdout: raise ValueError('zero budget emitted text')
    zero=generation_events(report,0)
    prompt.write_text('')
    invalid=subprocess.run([str(binary),str(prepared),str(tables),str(prompt),'1','0',policy],
        cwd=repository_root(),env=environment(),capture_output=True)
    require_empty_prompt_rejection(invalid)
    verify_build(binary)
    write(output/'result.json',dict(kind='model-runtime-generations-v1',build=receipt,policy=policy,
        environment=stable_environment(),conditions_before=conditions_before,conditions_after=conditions_snapshot(),
        dtype='BF16 stored tensors; FP32 reductions',layout='row-major',
        prepared_manifest_sha256=sha(prepared/'manifest.json'),tokenizer_sha256=sha(tables),
        records=records,zero_budget=zero,empty_prompt_rejected=True))


def lifecycle_study(binary, output, prepared=None):
    binary=Path(binary).resolve()
    receipt=verify_build(binary)
    prepared,_=verify_prepared(prepared)
    result=subprocess.run([str(binary),'--lifecycle',str(prepared)],cwd=repository_root(),
        env=environment(),capture_output=True,text=True,check=True)
    if ('model device Apple M4 Pro backend metal' not in result.stdout or
            'lifecycle passed:' not in result.stdout): raise ValueError('missing lifecycle completion')
    verify_build(binary)
    write(output,dict(kind='model-runtime-lifecycle-v1',build=receipt,
        prepared_manifest_sha256=sha(prepared/'manifest.json'),stdout=result.stdout))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('specification');s.add_argument('--output',required=True,type=Path)
    s.add_argument('--generations',type=Path)
    b=sub.add_parser('build');b.add_argument('--binary',required=True,type=Path)
    b.add_argument('--generation',action='store_true')
    g=sub.add_parser('generate');g.add_argument('--binary',required=True,type=Path)
    g.add_argument('--output',required=True,type=Path);g.add_argument('--prepared',type=Path)
    g.add_argument('--policy',default='fast')
    l=sub.add_parser('lifecycle');l.add_argument('--binary',required=True,type=Path)
    l.add_argument('--output',required=True,type=Path);l.add_argument('--prepared',type=Path)
    d=sub.add_parser('diagnose');d.add_argument('--binary',required=True,type=Path)
    d.add_argument('--reference',required=True,type=Path);d.add_argument('--output',required=True,type=Path)
    d.add_argument('--prepared',type=Path)
    m=sub.add_parser('benchmark');m.add_argument('--binary',required=True,type=Path)
    m.add_argument('--specification',required=True,type=Path);m.add_argument('--output',required=True,type=Path)
    m.add_argument('--prepared',type=Path)
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
    if args.command=='specification':runtime_specification(args.output,args.generations)
    elif args.command=='build':build(args.binary,args.generation)
    elif args.command=='generate':generation_study(args.binary,args.output,args.prepared,args.policy)
    elif args.command=='lifecycle':lifecycle_study(args.binary,args.output,args.prepared)
    elif args.command=='diagnose':diagnose(args.binary,args.reference,args.output,args.prepared)
    elif args.command=='benchmark':benchmark(args.binary,args.specification,args.output,args.prepared)
    elif args.command=='consistency':evaluate_consistency(args.binary,args.reference,args.output,args.length,args.prepared)
    elif args.command=='operations':evaluate_operations(args.binary,args.reference,args.output,args.prepared)
    else:evaluate(args.binary,args.reference,args.output,args.configurations,args.prepared)


if __name__=='__main__':main()
