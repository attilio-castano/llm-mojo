"""Host-only BF16 transport and recorded decoder assertions; no inference."""
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('DECODER_FIXTURES',REPO/'build/oracle_data/decoder_layer'))
_spec=importlib.util.spec_from_file_location('decoder_transport_numerics',REPO/'tests/fixtures/mlp/numerics.py')
_num=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_num)
bf16_bits,from_bits,differences=_num.bf16_bits,_num.from_bits,_num.differences
RECORDS=[]
VERIFIED=set()
FULL={}
IDENTITY={}
MANIFEST=None
EXPECTED_CASES=set()
FROZEN=None
DEVELOPMENT_ROOT=REPO/"build/oracle_data/decoder_layer"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    global MANIFEST, EXPECTED_CASES, FROZEN
    if MANIFEST is None:
        record=json.loads((ROOT/'manifest.json').read_text())
        if record['status']!='complete':
            raise ValueError('decoder fixtures are incomplete')
        anchor=json.loads((REPO/'tests/fixtures/decoder_layer/checksums.json').read_text())
        raw=(REPO/'tests/fixtures/decoder_layer/development.json.gz').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=anchor['evidence_sha256']:
            raise ValueError('decoder anchor hash mismatch')
        raw=gzip.decompress(raw)
        if hashlib.sha256(raw).hexdigest()!=anchor['uncompressed_sha256']:
            raise ValueError('decoder evidence hash mismatch')
        frozen=json.loads(raw)
        FROZEN=frozen
        EXPECTED_CASES=set(frozen['cases'])
        for k in ('specification','upstream','sources'):
            if record[k]!=frozen[k]:
                raise ValueError('decoder fixture contract mismatch: '+k)
        for path,digest in frozen['sources'].items():
            if sha(REPO/path)!=digest:
                raise ValueError('decoder reference source changed: '+path)
        if record.get('kind') in ('decoder_holdout','decoder_selection_holdout'):
            from llm_mojo.decoder_validation import holdout_manifest
            holdout_manifest(ROOT)
            EXPECTED_CASES=set(record['cases'])
        else:
            for name,case in record['cases'].items():
                if name not in frozen['cases'] or case!=frozen['cases'][name]:
                    raise ValueError('unknown or changed decoder case: '+name)
        MANIFEST=record
    return MANIFEST


def cases():
    split=os.environ.get('DECODER_SPLIT','development')
    if split not in ('development','checkpoint','holdout'):
        raise ValueError('unsupported decoder split; reserved evaluation is not enabled')
    selected=[name for name in manifest()['cases'] if split=='holdout' or name.startswith('checkpoint_')==(split=='checkpoint')]
    expected={name for name in EXPECTED_CASES if split=='holdout' or name.startswith('checkpoint_')==(split=='checkpoint')}
    if set(selected)!=expected:
        raise ValueError('decoder split has incomplete fixture coverage')
    if os.environ.get('DECODER_CASE'):
        name=os.environ['DECODER_CASE']
        if name not in selected:
            raise ValueError('case is not in selected split')
        selected=[name]
    return [(name,*(manifest()['cases'][name]['spec'][k] for k in ('rows','h','nq','nk','d','i'))) for name in selected]


def selection_variants():
    from llm_mojo.benchmarks.decoder_layer_contract import VARIANTS
    selected=sorted(VARIANTS)
    if os.environ.get('DECODER_VARIANTS'):
        selected=[int(v) for v in os.environ['DECODER_VARIANTS'].split(',')]
        if len(set(selected))!=len(selected) or not set(selected)<=VARIANTS:
            raise ValueError('invalid decoder selection variants')
    return selected


def verify_case(name):
    if name not in VERIFIED:
        for label,spec in case_record(name)['arrays'].items():
            if sha(case_directory(name)/(label+'.npy'))!=spec['sha256']:
                raise ValueError('decoder fixture array changed: '+label)
        VERIFIED.add(name)


def schedules(name):
    declared=case_record(name)['schedules']
    return [(key,[(call['start'],call['rows']) for call in declared[key]])
            for key in ('full','chunk','threshold','reuse') if key in declared]


def configure(name, policy, schedule, mode):
    global IDENTITY
    IDENTITY=dict(case=name,policy=policy,schedule=schedule,mode=mode)
    if schedule=='full' and mode=='layer':
        FULL.clear()


def read(name,label):
    a=np.load(case_directory(name)/(label+'.npy'),allow_pickle=False)
    spec=case_record(name)['arrays'][label]
    if a.shape!=tuple(spec['shape']) or a.dtype!=np.float32:
        raise ValueError('fixture shape/storage dtype changed')
    return a


def at(address,count):
    return np.ctypeslib.as_array((ctypes.c_uint16*count).from_address(address))


def load(address,name,label,start,count):
    src=bf16_bits(read(name,label)).reshape(-1)[start:start+count]
    if src.size!=count:
        raise ValueError('decoder load extent mismatch')
    at(address,count)[:]=src


def poison(address,active,capacity):
    if not 0<=active<=capacity:
        raise ValueError('invalid poison extent')
    at(address,capacity)[:]=0x42f6
    at(address,active)[:]=0x7fc0


def snapshot(address,count):
    return at(address,count).copy()


def exact(address,expected,label):
    actual=at(address,len(expected))
    failed=int(np.count_nonzero(actual!=expected))
    emit(dict(kind='exact',label=label,elements=len(expected),failed=failed))
    if failed:
        raise AssertionError('decoder changed protected bytes: '+label)


def emit(record):
    value={**IDENTITY,**record}
    if IDENTITY.get("mode")=="negative" and "failed" in value:
        value["expected_failure"]=True
    RECORDS.append(value)
    path=os.environ.get('DECODER_RECORDS')
    if path:
        with Path(path).open('a') as f:
            f.write(json.dumps(value,allow_nan=False)+'\n')


def route(actual, mapping, start, rows, device, backend):
    emit(dict(kind='route', attention=actual, mlp=mapping, start=start, rows=rows,
              device=device, backend=backend))


def check(address,name,stage,schedule,start,rows,width,capacity,mode='layer'):
    count=rows*width
    bits=at(address,capacity).copy()
    if np.any(bits[count:]!=0x42f6):
        raise AssertionError('decoder wrote inactive workspace '+stage)
    actual=from_bits(bits[:count]).reshape(rows,width)
    label=f'full_{stage}' if schedule in ('full','full_slice') else f'{schedule}_{start}_{stage}'
    expected=read(name,label)
    if schedule=='full_slice':
        expected=expected.reshape(-1,width)[start:start+rows]
    expected=expected.reshape(rows,width)
    gate=manifest()['specification']['whole_layer_gates'].get(stage)
    if mode=='mlp' and stage=='B_mlp':
        gate={'atol':2**-6,'rtol':2**-6}
    if mode=='operation':
        gate={'atol':2**-7,'rtol':2**-7}
        if stage=='B_att':
            # Preserve the existing isolated Wo contract; other projections
            # retain their 2^-7 gate.
            gate={'atol':2**-5,'rtol':2**-5}
        elif stage in ('Q','K_rot','S','Y','Z'):
            gate={'atol':0,'rtol':0,'exact_bits':True}
        elif stage=='A':
            gate={'atol':2**-133,'rtol':2**-7,'max_bf16_steps':1,'exact_reference_zero':True}
    result=differences(actual,expected,gate)
    emit(dict(kind='boundary',stage=stage,start=start,rows=rows,inactive_exact=True,**result))
    if result.get('failed',0):
        print('DECODER FAILURE',json.dumps(RECORDS[-1]),flush=True)
        raise AssertionError('decoder numerical gate: '+stage)
    if mode=='layer' and stage in ('B_att','Z','B_mlp','Y'):
        if schedule=='full':
            FULL[stage]=actual.copy()
        elif stage in FULL:
            comparison=differences(actual,FULL[stage][start:start+rows],gate)
            emit(dict(kind='full_vs_chunk',stage=stage,start=start,rows=rows,**comparison))
            if comparison['failed']:
                raise AssertionError('decoder full/chunk mismatch: '+stage)


def check_cache(address,produced,previous,name,label,start,rows,width,capacity):
    bits=at(address,capacity*width).copy()
    if not np.array_equal(bits[:start*width],previous):
        raise AssertionError('decoder cache prefix changed')
    if not np.array_equal(bits[start*width:(start+rows)*width],produced):
        raise AssertionError('decoder cache append changed produced bits')
    if np.any(bits[(start+rows)*width:]!=0x42f6):
        raise AssertionError('decoder wrote inactive cache capacity')
    actual=from_bits(bits[:(start+rows)*width]).reshape(start+rows,width)
    expected=read(name,label).reshape(start+rows,width)
    emit(dict(kind='cache',stage=label,start=start,rows=rows,prefix_exact=True,append_exact=True,inactive_exact=True,
              **differences(actual,expected)))


def result_summary():
    print('Decoder checks:',len(RECORDS),'records,',sum(r.get('failed',0) for r in RECORDS if not r.get('expected_failure')),'failed elements',flush=True)


def negative_passed(label):
    emit(dict(kind="negative_control",label=label,rejected=True))


def case_record(name):
    active=manifest()['cases']
    return active[name] if name in active else FROZEN['cases'][name]


def case_directory(name):
    return (ROOT if name in manifest()['cases'] else DEVELOPMENT_ROOT)/name
