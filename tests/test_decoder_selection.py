"""Selection requires complete calibrated evidence and independent confirmation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_mojo.benchmarks import decoder_layer_contract as contract
from llm_mojo.benchmarks import study
from llm_mojo import decoder_validation as validation
import test_decoder_validation as baseline_tests


def observations(spec):
    return [dict(**w,layers=l,candidate=c,block=b,arm=a,variant=0 if a=='control' else c,repetition=n,
                 us=100. if a=='control' or c==0 else 90.)
        for w,l,c in study.comparisons(spec) for b in range(1,5)
        for a in ('control','candidate') for n in range(10)]


class SelectionTests(unittest.TestCase):
    def test_declared_budget_and_route_dispatches(self):
        contract.selection_declaration()
        self.assertEqual(sum(len(observations(study.STUDIES['decoder_selection_'+f+'_screen'])) for f in contract.SCREEN_GRIDS),9920)
        self.assertEqual(sum(len(observations(study.STUDIES['decoder_selection_'+f])) for f in ('calibration','buffered')),480)
        expected={0:(0,0,0,16),1:(0,5,0,16),2:(4,0,0,17),3:(4,5,0,17),4:(0,0,0,16),8:(0,0,8,16),12:(0,0,12,16),14:(0,0,14,16)}
        for v,(g,p,m,n) in expected.items():
            self.assertEqual(contract.mappings(v,1),(g,p,m))
            self.assertEqual(len(contract.stages(v,17)),n)
            self.assertEqual(len(contract.stages(v,1)),15 if v in (8,14) else 16)
        for v in (-1,7,9,True):
            with self.assertRaises(ValueError):contract.mappings(v,1)

    def test_modes_select_independently_and_neighbors_are_frozen(self):
        build={'frozen':True}
        def load(path,prefix=''):
            name=prefix.removesuffix('_') if prefix else Path(path).name;spec=study.STUDIES[name]
            data=observations(spec)
            # Hot has no effect; ring24 has a reproducible ten percent effect.
            for row in data:
                if row['layers']==1:row['us']=100.
            return dict(study=name,build=build,specification=json.loads(json.dumps(spec)),samples_sha256='samples'),data,study.summarize(data,spec)
        with patch.object(study,'load_run',side_effect=load),patch.object(contract,'sha',return_value='sha'):
            decision=contract.screen_decision(Path('/unused'),build)
        self.assertEqual(len(decision['proposals']),56)
        self.assertTrue(all(p['candidate']==0 for p in decision['proposals'] if p['layers']==1))
        for family in contract.SCREEN_GRIDS:
            spec=contract.confirmation_spec(family,decision)
            data=observations(spec);study.summarize(data,spec)
            for bad in (data[:-1],data+[data[0]]):
                with self.assertRaises(ValueError):study.summarize(bad,spec)
            bad=copy.deepcopy(spec);bad['comparisons']=bad['comparisons'][1:]
            with self.assertRaises(ValueError):study.comparisons(bad)
        neighbor=next(p for p in decision['proposals'] if p['family']=='cached' and p['query_rows']==15 and p['layers']==24)
        self.assertEqual(neighbor['from_shape'],[16,256])
        def confirm(path,prefix=''):
            name=prefix.removesuffix('_') if prefix else Path(path).name;family=name.split('_')[2];spec=contract.confirmation_spec(family,decision)
            data=observations(spec)
            # A failing neighboring shape must fall back, without changing its proposal.
            for row in data:
                if row['query_rows']==15 and row['rows']==256:row['us']=100.
            return dict(study=name,build=build,selection=decision,specification=json.loads(json.dumps(spec)),samples_sha256='samples'),data,study.summarize(data,spec)
        with patch.object(study,'load_run',side_effect=confirm),patch.object(contract,'sha',return_value='sha'):
            selected=contract.confirmed_selection(decision,Path('/unused'))
        cell=next(c for c in selected['cells'] if c['family']=='cached' and c['query_rows']==15 and c['layers']==24)
        self.assertNotEqual(cell['candidate'],0);self.assertEqual(cell['accepted'],0)
        self.assertEqual(selected['fallback'],0)
        self.assertLessEqual(len(contract.profile_selection(selected)),9)

    def test_noise_and_one_bad_block_prevent_promotion(self):
        spec=study.STUDIES['decoder_selection_full_screen']
        for kind in ('noise','regression'):
            data=observations(spec)
            for row in data:
                if row['arm']=='candidate' and row['block']==1 and row['candidate']==(0 if kind=='noise' else 1):row['us']=120.
            results=study.summarize(data,spec)
            self.assertFalse(any(r['decision']=='faster' for r in results))

    def test_profile_census_includes_split_and_combined_launches(self):
        import csv,gzip,hashlib,io
        selected=dict(kind='decoder_confirmed_selection',fallback=0,lookup=[
            dict(query_rows=r,rows=t,layers=l,variant=(3 if r==64 else 14 if r==1 else 0) if l==24 else 0)
            for r,t,_ in contract.PROFILES for l in (1,24)])
        grid=contract.profile_selection(selected);captures=[];rows=[]
        source=dict(commit='frozen',dirty=False)
        for r,t,v in grid:
            n=next(n for rr,tt,n in contract.PROFILES if (r,t)==(rr,tt))
            captures.append(dict(query_rows=r,rows=t,variant=v,capture=dict(repository=source,
                operation='decoder_layer',implementation=f'decoder_layer_{v}',entrypoint='enqueue_decoder_layer',
                runtime=dict(backend='metal',device='Apple Test GPU'),workload=dict(**contract.specification(v,r,t),
                    rows=r,warmup_iterations=10,profile_iterations=n))))
            for j in range(n):
                for k,stage in enumerate(contract.stages(v,r)):
                    rows.append(dict(query_rows=r,rows=t,variant=v,iteration=j,stage=stage,
                        duration_ns=1,start_ns=j*40+k*2,end_ns=j*40+k*2+1))
        self.assertEqual(len(rows),4325)
        record=dict(schema=6,common=dict(repository=source),captures=captures,
            specification=dict(selection=selected,captures=grid,selection_sha256=hashlib.sha256((json.dumps(selected,indent=2)+'\n').encode()).hexdigest()))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def check(data):
                out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(data)
                raw=gzip.compress(out.getvalue().encode());(root/'profile_samples.csv.gz').write_bytes(raw)
                record['samples_sha256']=hashlib.sha256(raw).hexdigest();(root/'profiles.json').write_text(json.dumps(record))
                return study.load_decoder_profile(root)
            self.assertEqual(sum(x['count'] for x in check(rows)),4325)
            windows=study.load_decoder_windows(root)
            self.assertEqual(sum(w['dispatches'] for w in windows),4325)
            self.assertEqual({(w['query_rows'],w['rows'],w['variant']) for w in windows},set(grid))
            self.assertTrue(all(w['active_us']==w['dispatches']/1000 for w in windows))
            for bad in (rows[:-1],rows+[rows[-1]],[r for r in rows if r['stage']!='FP32 GQA merge']):
                with self.assertRaises(ValueError):check(bad)

    def test_all_ids_and_async_required_for_numerical_acceptance(self):
        base=baseline_tests.NumericalReceiptTests();base.setUp()
        records=[r for r in base.records if r['mode']=='negative']
        for v in sorted(contract.VARIANTS):
            for source in base.records:
                if source['mode']=='negative':continue
                row={**source,'policy':v}
                if row['kind']=='route':row['mlp']=contract.mappings(v,1)[2]
                records.append(row)
            for j in range(13):
                p,r=(0,53) if j==0 else (52+j,1)
                for stage in validation.BOUNDARIES:
                    records.append(dict(mode='async',policy=v,kind='boundary',stage=stage,start=p,rows=r,elements=r*896,failed=0))
                    records.append(dict(mode='async',policy=v,kind='exact',label='async '+stage+' vs separate workspace',elements=r*896,failed=0))
            records.append(dict(mode='async',policy=v,kind='boundary',stage='Y',start=0,rows=1,elements=896,failed=0))
            for label in ('async cache key','async cache value'):
                records.append(dict(mode='async',policy=v,kind='exact',label=label,elements=66*128,failed=0))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checks.jsonl'
            def check(rows):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                return validation.validate_results(path,base.cases,True)
            self.assertEqual(check(records)['checks'],480)
            for bad in (records[:-1],records+[records[-1]],[r for r in records if r.get('policy')!=14]):
                with self.assertRaises(ValueError):check(bad)
            bad=copy.deepcopy(records);del bad[-1]['failed']
            with self.assertRaises(ValueError):check(bad)
