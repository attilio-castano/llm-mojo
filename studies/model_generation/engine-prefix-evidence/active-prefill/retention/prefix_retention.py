#!/usr/bin/env python3
"""Lossless local active-prefill evidence packaging and guarded source/CPU restoration.

Pack only after the declared 80-run campaign has completed. The measured Git
commit supplies source bytes; current working files are not silently substituted.
Measurements remain a separate original gzip. Text receipts and explicitly
selected compact JSON/gzip acceptance evidence are retained without binaries,
weights or generated oracle arrays. Restore requires a fresh /private/tmp path
and an independently supplied archive SHA-256, and runs no native model work.
"""
import argparse
import base64
import builtins
import datetime
import gzip
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
KIND='engine-prefix-active-validation-v1'
ARCHIVE='engine-prefix-active-validation.json.gz'
MANIFEST='engine-prefix-active-validation.json'
TEXT_SUFFIXES={'.py','.mojo','.json','.log','.txt','.tsv','.md','.toml','.lock','.csv'}
POLICY='tests/fixtures/decoder_policies.json'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def unique(items):
    result={}
    for key,value in items:
        require(key not in result,'duplicate JSON key: '+key)
        result[key]=value
    return result


def decode(raw):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)


def write(path,value):
    Path(path).write_bytes(encode(value))


def sha(path):
    return digest(Path(path).read_bytes())


def safe(name):
    require(type(name) is str and name and '\\' not in name and '\x00' not in name,'invalid retained path')
    value=PurePosixPath(name)
    require(not value.is_absolute() and '..' not in value.parts and str(value)==name and name!='.','unsafe retained path: '+name)
    return Path(name)


def fresh(path):
    path=Path(path).absolute()
    require(path.resolve().is_relative_to(Path('/private/tmp').resolve()),'output must remain under /private/tmp')
    require(not path.exists(),'refusing to replace '+str(path))
    path.mkdir(parents=True)
    return path


def binding(path):
    path=Path(path)
    return dict(bytes=path.stat().st_size,sha256=sha(path))


def checked(raw,expected,label):
    require(type(expected.get('bytes')) is int and len(raw)==expected['bytes'] and digest(raw)==expected.get('sha256'),
            'retained identity differs: '+label)
    return raw


def measurement(path):
    path=Path(path)
    raw=path.read_bytes(); manifest_raw=path.with_suffix('').read_bytes()
    manifest=decode(manifest_raw); unpacked=gzip.decompress(raw)
    checked(raw,manifest,str(path))
    require(digest(unpacked)==manifest['uncompressed_sha256'],'measurement uncompressed identity differs')
    record=decode(unpacked)
    require(record.get('kind')==manifest.get('kind')=='qwen-engine-prefix-active-v1','wrong measurement kind')
    require(len(record.get('cells',[]))==5 and all(len(c.get('runs',[]))==16 for c in record['cells'])
            and record.get('summary',{}).get('measured_runs')==80,'measurement is not the complete 80-run campaign')
    return record,raw,manifest_raw


class Inventory:
    def __init__(self):
        self.files={}

    def add(self,name,raw,origin='',allow_gzip=False):
        safe(name)
        entry=dict(name=name,bytes=len(raw),sha256=digest(raw),original_path=origin)
        try:
            text=raw.decode('utf-8')
            require(text.encode()==raw,'non-lossless UTF-8')
            entry['text']=text
        except UnicodeDecodeError:
            require(allow_gzip and name.endswith('.json.gz'),'binary/array/weight bytes are excluded: '+name)
            require(isinstance(decode(gzip.decompress(raw)),dict),'compact gzip evidence must contain a JSON object')
            entry['encoding']='base64'
            entry['data']=base64.b64encode(raw).decode('ascii')
        require(name not in self.files or self.files[name]==entry,'conflicting duplicate retained name')
        self.files[name]=entry

    def file(self,path,name):
        path=Path(path)
        require(path.is_file() and not path.is_symlink(),'explicit evidence must be a regular file: '+str(path))
        require(path.suffix in TEXT_SUFFIXES or path.name.endswith('.json.gz'),'excluded evidence type: '+str(path))
        self.add(name,path.read_bytes(),str(path.absolute()),allow_gzip=True)


def git(root,*arguments):
    return subprocess.run(['git',*arguments],cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout


def committed_sources(root,commit,expected,inventory):
    rows=git(root,'ls-tree','-r','-z',commit).split(b'\0')
    modes={}
    for row in rows:
        if row:
            metadata,name=row.split(b'\t',1)
            modes[name.decode()]=metadata.split()[0].decode()
    def read(name,depth=0):
        require(depth<12 and name in modes,'missing/cyclic source identity in measured commit: '+name)
        raw=git(root,'show',commit+':'+name)
        if modes[name]=='120000':
            target=raw.decode()
            require(not PurePosixPath(target).is_absolute(),'source symlink escapes measured tree')
            resolved=os.path.normpath(str(PurePosixPath(name).parent/target))
            safe(resolved)
            return read(resolved,depth+1)
        require(modes[name].startswith('100'),'source is not a file: '+name)
        return raw
    omitted=[]
    for name,expected_sha in sorted(expected.items()):
        safe(name)
        raw=read(name)
        require(digest(raw)==expected_sha,'measurement source differs from measured Git blob: '+name)
        retain=Path(name).suffix not in ('.json','.gz') or name==POLICY
        if retain:
            inventory.add('checkout/'+name,raw,commit+':'+name)
        else:
            omitted.append(dict(name=name,bytes=len(raw),sha256=digest(raw),note='generated source fixture: hash-only; unused by CPU replay'))
    for name in sorted(modes):
        if (name in ('README.md','AGENTS.md','studies/model_generation/engine-core.md',
                     'src/llm_mojo/benchmarks/README.md') or name.startswith('docs/')) and Path(name).suffix=='.md':
            inventory.add('checkout/'+name,read(name),commit+':'+name)
    return omitted


def pack(args):
    root=Path(args.root).resolve()
    commit=git(root,'rev-parse',args.commit+'^{commit}').decode().strip()
    prefix=Path(args.campaign_prefix).absolute()
    measured=Path(str(prefix)+'-measurements')/'engine-prefix-active.json.gz'
    record,raw,manifest_raw=measurement(measured)
    require(record['build']['source']['repository']['commit']==commit
            and record['build']['source']['repository']['dirty'] is False,'measurement commit/clean-source identity differs')
    require(record['trace_sha256']==sha(Path(str(prefix)+'-campaign.json')),'frozen campaign source document differs')
    qualification_path=Path(str(prefix)+'-qualification')/'qualification.json'
    require(sha(qualification_path)==record['qualification_sha256'],'qualification document differs')
    inventory=Inventory()
    omitted=committed_sources(root,commit,record['build']['source']['sources'],inventory)
    helper=Path(__file__).resolve()
    helper_name=helper.relative_to(root).as_posix()
    require(digest(git(root,'show',commit+':'+helper_name))==sha(helper),
            'retention helper differs from the measured commit')
    inventory.file(Path(__file__),'retention/prefix_retention.py')
    build_dir=Path(str(prefix)+'-build')
    require(decode((build_dir/'build.json').read_bytes())==record['build'],'original build receipt differs')
    for name in ('build.json','engine-build.log'):
        inventory.file(build_dir/name,'build/'+name)
    validation=Path(args.validation_receipt).resolve()
    validated=decode(validation.read_bytes())
    require(validated.get('kind')=='active-prefill-full-validation-v1'
            and validated.get('status')=='passed'
            and validated.get('source_unchanged') is True
            and validated.get('execution',{}).get('exit_code')==0
            and validated.get('source',{}).get('sources')==record['build']['source']['sources'],
            'full validation must pass for the exact measured source bytes')
    inventory.file(validation,'acceptance/full-validation/'+validation.name)
    evidence=[]
    for index,item in enumerate(args.evidence):
        path=Path(item).resolve(); label=f'acceptance/evidence-{index}'
        if path.is_dir():
            for entry in sorted(path.rglob('*')):
                if entry.is_file() and (entry.suffix in TEXT_SUFFIXES or entry.name.endswith('.json.gz')):
                    inventory.file(entry,label+'/'+entry.relative_to(path).as_posix())
        else:
            inventory.file(path,label+'/'+path.name)
        evidence.append(dict(original_path=str(path),retained_prefix=label))
    stage=fresh(args.output)
    target=stage/'measurements'; target.mkdir()
    (target/'engine-prefix-active.json.gz').write_bytes(raw)
    (target/'engine-prefix-active.json').write_bytes(manifest_raw)
    bundle=dict(kind=KIND,schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=commit,source_map=record['build']['source']['sources'],
        files=list(inventory.files.values()),hash_only_sources=omitted,
        measurement=dict(path='measurements/engine-prefix-active.json.gz',**binding(measured)),
        measurement_manifest=dict(path='measurements/engine-prefix-active.json',**binding(measured.with_suffix(''))),
        embedded_documents=[dict(name='records/campaign.json',member='trace_document',sha256=record['trace_sha256']),
                            dict(name='records/qualification.json',member='qualification_document',sha256=record['qualification_sha256'])],
        validation_receipt='acceptance/full-validation/'+validation.name,evidence_inputs=evidence,
        custody='local-only; no remote backup or fresh GPU rerun')
    unpacked=encode(bundle); compressed=gzip.compress(unpacked,mtime=0)
    (stage/ARCHIVE).write_bytes(compressed)
    write(stage/MANIFEST,dict(kind=KIND,schema_version=1,source_commit=commit,bytes=len(compressed),
        sha256=digest(compressed),uncompressed_bytes=len(unpacked),uncompressed_sha256=digest(unpacked),
        retained_files=len(inventory.files),hash_only_source_files=len(omitted)))
    write(stage/'retention-inventory.json',dict(kind='engine-prefix-active-retention-inventory-v1',source_commit=commit,
        files={name:binding(stage/name) for name in (ARCHIVE,MANIFEST,'measurements/engine-prefix-active.json.gz','measurements/engine-prefix-active.json')}))
    print(json.dumps(dict(directory=str(stage),archive_sha256=digest(compressed),source_commit=commit,
        restore_command=[str(Path(sys.executable).absolute()),'-B',str(Path(__file__).resolve()),'restore',
                         '--archive',str(stage/ARCHIVE),'--manifest',str(stage/MANIFEST),
                         '--expected-sha256',digest(compressed),'--directory',str(stage)+'-restored',
                         '--python',str(Path(sys.executable).absolute())]),indent=2))


def restore(args):
    archive=Path(args.archive).resolve(); manifest=decode(Path(args.manifest).read_bytes())
    compressed=archive.read_bytes()
    require(digest(compressed)==args.expected_sha256,'independent archive SHA differs')
    checked(compressed,manifest,'validation archive')
    raw=gzip.decompress(compressed)
    require(digest(raw)==manifest['uncompressed_sha256'] and len(raw)==manifest['uncompressed_bytes'],'validation payload differs')
    bundle=decode(raw)
    require(bundle.get('kind')==KIND and bundle['source_commit']==manifest['source_commit'],'validation identity differs')
    files={}
    for entry in bundle['files']:
        name=entry['name']; safe(name)
        require(name not in files,'duplicate retained path')
        body=entry['text'].encode() if 'text' in entry else base64.b64decode(entry['data'],validate=True)
        files[name]=checked(body,entry,name)
    originals={}
    for key in ('measurement','measurement_manifest'):
        entry=bundle[key]; name=entry['path']; safe(name)
        originals[name]=checked((archive.parent/safe(name)).read_bytes(),entry,name)
    require(set(files).isdisjoint(originals),'measurement aliases conflict with retained files')
    stage=fresh(args.directory)
    for name,body in {**files,**originals}.items():
        path=stage/safe(name); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(body)
    record,_,_=measurement(stage/bundle['measurement']['path'])
    require(record['build']['source']['repository']['commit']==bundle['source_commit'],'restored measurement source differs')
    require(record['build']['source']['sources']==bundle['source_map'],'restored measurement source map differs')
    for entry in bundle['embedded_documents']:
        body=record[entry['member']].encode()
        require(digest(body)==entry['sha256'],'embedded original document differs')
        path=stage/safe(entry['name']); require(not path.exists(),'embedded document overwrites a retained file')
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(body)
    for name,value in bundle['source_map'].items():
        path=stage/'checkout'/safe(name)
        if path.exists():
            require(sha(path)==value,'restored source differs: '+name)
    helper=stage/'retention/prefix_retention.py'
    require(Path(args.python).is_absolute(),'--python must name an absolute locked-environment executable')
    command=[str(Path(args.python).absolute()),'-B',str(helper),'cpu-check','--directory',str(stage)]
    started=time.monotonic_ns()
    result=subprocess.run(command,cwd=stage,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
    elapsed=time.monotonic_ns()-started
    (stage/'cpu-replay.log').write_text(result.stdout)
    write(stage/'cpu-replay-execution.json',dict(command=command,exit_code=result.returncode,timeout_seconds=600,
        wall_elapsed_ns=elapsed,log_sha256=sha(stage/'cpu-replay.log'),archive_sha256=digest(compressed),
        source_commit=bundle['source_commit'],native_execution=False,asset_access=False))
    require(result.returncode==0,'guarded CPU replay failed; see '+str(stage/'cpu-replay.log'))
    print(json.dumps(dict(directory=str(stage),source_commit=bundle['source_commit'],actual_exit_code=result.returncode,
                         native_execution=False,asset_access=False),indent=2))


def cpu_check(args):
    stage=Path(args.directory).resolve(); checkout=stage/'checkout'
    record,_,_=measurement(stage/'measurements/engine-prefix-active.json.gz')
    blocked=[Path(record['build']['assets'][key]).resolve() for key in ('prepared','tables') if key in record['build']['assets']]
    attempts=[]
    def deny(name):
        def rejected(*arguments,**keywords):
            attempts.append(name)
            raise RuntimeError('CPU replay prohibits '+name)
        return rejected
    def asset(path):
        if isinstance(path,(str,bytes,os.PathLike)):
            value=Path(os.fsdecode(path)).resolve()
            if value.suffix in ('.bin','.npy','.npz','.safetensors','.pt','.pth') or any(value==p or value.is_relative_to(p) for p in blocked):
                deny('asset read '+str(value))()
    real_open,real_io_open,real_os_open=builtins.open,io.open,os.open
    def guarded_open(path,*arguments,**keywords):
        asset(path); return real_open(path,*arguments,**keywords)
    def guarded_io_open(path,*arguments,**keywords):
        asset(path); return real_io_open(path,*arguments,**keywords)
    def guarded_os_open(path,*arguments,**keywords):
        asset(path); return real_os_open(path,*arguments,**keywords)
    builtins.open,io.open,os.open=guarded_open,guarded_io_open,guarded_os_open
    for name in ('Popen','run','call','check_call','check_output'):
        setattr(subprocess,name,deny('subprocess.'+name))
    for name in ('system','popen','execv','execve','execl','execlp','execvp','execvpe','posix_spawn','posix_spawnp'):
        if hasattr(os,name): setattr(os,name,deny('os.'+name))
    import socket
    class NoNetworkSocket(socket.socket):
        def __init__(self,*arguments,**keywords):
            deny('network socket')()
    socket.socket=NoNetworkSocket
    socket.create_connection=deny('network connection')
    class NoGPU(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('max','mojo','torch','tensorflow','cupy'):
                deny('GPU/native framework import '+fullname)()
    sys.meta_path.insert(0,NoGPU())
    sys.path.insert(0,str(checkout/'src'))
    from llm_mojo.benchmarks import model_profile,engine_prefix,model_contract
    for module in (model_profile,engine_prefix,model_contract):
        require(Path(module.__file__).resolve().is_relative_to(checkout),'checker imported outside restored source')
    for name in ('execute','checked_execution','assets','verify_prepared','ensure_prepared','verify_build','verify_engine_build','engine_build','environment_tool','source_identity'):
        if hasattr(model_profile,name): setattr(model_profile,name,deny('model_profile.'+name))
    actual=model_profile.engine_replay(stage/'measurements/engine-prefix-active.json.gz')
    require(actual==record['summary'],'restored CPU summary differs')
    preliminary=None
    helpers=list((stage/'acceptance').rglob('engine-prefix-active-preliminary-replay.py'))
    if helpers:
        require(len(helpers)==1,'ambiguous preliminary replay helper')
        script=helpers[0]
        spec=importlib.util.spec_from_file_location('restored_prefix_preliminary_replay',script)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preliminary=module.verify(script.with_name('engine-prefix-active-preliminary-acceptance.json'))
        require(preliminary.get('status')=='passed-cpu-replay','preliminary acceptance CPU replay failed')
    require(not attempts,'prohibited action was attempted during CPU replay')
    result=dict(kind='engine-prefix-active-guarded-cpu-replay-v1',source_commit=record['build']['source']['repository']['commit'],
        measurement_sha256=sha(stage/'measurements/engine-prefix-active.json.gz'),measured_runs=actual['measured_runs'],
        untimed_preflights=actual['untimed_preflights'],same_delivered_histories=actual['same_delivered_histories'],
        same_reuse_work=actual['same_reuse_work'],usefulness=actual['usefulness'],
        native_execution=False,asset_access=False,network_access=False,prohibited_attempts=attempts,
        preliminary_acceptance=preliminary)
    write(stage/'cpu-replay-result.json',result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    create=commands.add_parser('pack')
    create.add_argument('--root',required=True)
    create.add_argument('--commit',required=True)
    create.add_argument('--campaign-prefix',required=True)
    create.add_argument('--validation-receipt',required=True)
    create.add_argument('--evidence',action='append',default=[],help='Explicit compact acceptance bundle/receipt file or directory; repeatable')
    create.add_argument('--output',required=True)
    recover=commands.add_parser('restore')
    recover.add_argument('--archive',required=True)
    recover.add_argument('--manifest',required=True)
    recover.add_argument('--expected-sha256',required=True)
    recover.add_argument('--directory',required=True)
    recover.add_argument('--python',default=sys.executable)
    check=commands.add_parser('cpu-check')
    check.add_argument('--directory',required=True)
    args=parser.parse_args()
    {'pack':pack,'restore':restore,'cpu-check':cpu_check}[args.command](args)


if __name__=='__main__':
    main()
