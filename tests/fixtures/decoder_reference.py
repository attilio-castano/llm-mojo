# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Decoder references through the shared lock without editing frozen sources."""
from pathlib import Path
import runpy
import sys


def checkpoint_only():
    """Restore the frozen checkpoint split without regenerating synthetic data."""
    import argparse
    import json
    import torch
    from generate import load_frozen, sources, execute_case, write_json, verify_arrays
    from reference import checkpoint, provenance
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-only',action='store_true')
    parser.add_argument('--checkpoint-assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    frozen=load_frozen()
    if sources()!=frozen['sources'] or provenance()!=frozen['upstream']:
        raise ValueError('pinned decoder oracle identity changed')
    cases,origin=checkpoint(args.checkpoint_assets)
    if json.loads(json.dumps(origin))!=frozen['checkpoint']:
        raise ValueError('checkpoint input identity changed')
    expected={name:case for name,case in frozen['cases'].items() if name.startswith('checkpoint_')}
    record={**frozen,'cases':expected}
    out=args.output
    if out.exists():
        if json.loads((out/'manifest.json').read_text())!=record:
            raise ValueError('checkpoint capture is incomplete or changed; use a fresh output')
        verify_arrays(out,record)
        print('Checkpoint reference arrays match frozen anchors.',flush=True)
        return
    out.mkdir(parents=True)
    record.update(status='running',cases={})
    write_json(out/'manifest.json',record)
    try:
        for name,spec,data,ids in cases:
            case=execute_case(out/name,name,spec,data,ids)
            if json.loads(json.dumps(case))!=expected[name]:
                raise ValueError('checkpoint reference differs from frozen evidence: '+name)
            record['cases'][name]=case
            write_json(out/'manifest.json',record)
        if sources()!=frozen['sources'] or provenance()!=frozen['upstream']:
            raise ValueError('decoder reference changed during checkpoint capture')
        record['status']='complete'
    except Exception as error:
        record.update(status='failed',error=str(error))
        raise
    finally:
        write_json(out/'manifest.json',record)

if __name__ == '__main__':
    directory = Path(__file__).resolve().parent/'decoder_layer'
    sys.path.insert(0, str(directory))
    if '--checkpoint-only' in sys.argv:
        checkpoint_only()
    else:
        runpy.run_path(str(directory/'generate.py'), run_name='__main__')
