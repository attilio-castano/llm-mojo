# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Capture declared reserved decoder outputs only for a verified clean candidate."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/fixtures/decoder_layer'))
import numpy as np
import torch
from contract import HOLDOUT, QWEN, case_id, case_spec
from reference import inputs, checkpoint, provenance
from generate import execute_case, load_frozen, sources, write_json
from llm_mojo.decoder_validation import verify_build, sha
from llm_mojo.benchmarks.environment import ensure_record_location, utc_now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-binary',type=Path,required=True)
    p.add_argument('--checkpoint-assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();ensure_record_location(out)
    if out.exists():raise ValueError('refusing to overwrite reserved decoder outputs')
    candidate=verify_build(args.candidate_binary)
    torch.set_num_threads(1)
    frozen=load_frozen()
    if sources()!=frozen['sources'] or provenance()!=frozen['upstream']:
        raise ValueError('decoder reference changed before reserved capture')
    cases,origin=checkpoint(args.checkpoint_assets)
    if origin!=frozen['checkpoint']:
        raise ValueError('decoder checkpoint assets or declared tokens changed')
    # Read only the already-verified embedding rows for the reserved prompt.
    ids=origin['holdout_token_ids']
    with (args.checkpoint_assets/'model.attention-prefix.bin').open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
        offset=8+size+header['model.embed_tokens.weight']['data_offsets'][0]
        x=[]
        for token in ids:
            f.seek(offset+token*896*2)
            bits=np.frombuffer(f.read(896*2),dtype='<u2').astype(np.uint32)<<16
            x.append(bits.view(np.float32))
    data={k:v for k,v in cases[0][2].items() if k!='X'};data['X']=np.stack(x)
    out.mkdir(parents=True)
    record=dict(kind='decoder_holdout',status='started',cases={},checkpoint=origin,
        specification=frozen['specification'],sources=frozen['sources'],upstream=frozen['upstream'],
        candidate=dict(binary_sha256=candidate['binary_sha256'],commit=candidate['source']['repository']['commit'],
            reference_sha256=sha(ROOT/'tests/fixtures/decoder_layer/checksums.json')),
        started_utc=utc_now())
    torch.set_num_threads(1)
    try:
        for spec in HOLDOUT:
            name=case_id(spec)
            record['cases'][name]=execute_case(out/name,name,spec,inputs(spec))
            write_json(out/'manifest.json',record)
        spec=case_spec(QWEN,len(ids),0)
        record['cases']['checkpoint_holdout']=execute_case(out/'checkpoint_holdout','checkpoint_holdout',spec,data,ids)
        if verify_build(args.candidate_binary)!=candidate or sources()!=frozen['sources']:
            raise ValueError('candidate/reference changed during reserved execution')
        record.update(status='complete',finished_utc=utc_now())
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:
        record['payload_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        write_json(out/'manifest.json',record)


if __name__=='__main__':main()
