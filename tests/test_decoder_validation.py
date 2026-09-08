"""Acceptance must reject incomplete numerical evidence even after exit code zero."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from llm_mojo.decoder_validation import validate_results


class NumericalReceiptTests(unittest.TestCase):
    def setUp(self):
        self.cases={'qwen':dict(spec=dict(rows=1,h=896,nq=14,nk=2,d=64,i=4864),
            schedules={'full':[dict(start=0,rows=1)],'chunk':[dict(start=0,rows=1)]})}
        self.records=[]
        stages=['N_att','Q_raw','K_raw','V_raw','Q','K_rot','O','B_att','Z','N_mlp','G','U','A','S','B_mlp','Y']
        boundaries=['B_att','Z','B_mlp','Y']
        def append(schedule,mode,kind,stage='',**kwargs):
            self.records.append(dict(case='qwen',policy=0,schedule=schedule,mode=mode,kind=kind,stage=stage,start=0,rows=1,**kwargs))
        def check(schedule,mode,kind,stage):
            width=4864 if stage in ('G','U','A','S') else 128 if stage in ('K_raw','V_raw','K_rot') else 896
            append(schedule,mode,kind,stage,elements=width,failed=0,inactive_exact=True)
        for schedule in ('full','chunk'):
            append(schedule,'layer','route',attention=4,mlp=0,device='Apple M4 Pro',backend='metal')
            for stage in stages:check(schedule,'layer','boundary',stage)
            if schedule=='chunk':
                for stage in boundaries:check(schedule,'layer','full_vs_chunk',stage)
            for field in ('cache_key','cache_value'):
                append(schedule,'layer','cache',('full_' if schedule=='full' else 'chunk_0_')+field,
                       elements=128,prefix_exact=True,append_exact=True,inactive_exact=True)
        for stage in stages:check('full','operation','boundary',stage)
        for stage in ('B_mlp','Y'):check('full','mlp','boundary',stage)
        sizes = dict(xb=896, aw_norm=896, aw_qkv=1152*896, aw_bias=1152,
                     aw_output=896*896, mw_norm=896, mw_gate=4864*896,
                     mw_up=4864*896, mw_down=896*4864, a_cosine=2*64, a_sine=2*64)
        for label, elements in sizes.items():
            append('chunk','layer','exact',label=label,elements=elements,failed=0)
        for label in ('second residual uses X','second residual omitted','first residual omitted',
                      'second norm uses X','wrong norm weights','wrong absolute RoPE position',
                      'mask exposes future rows','cache prefix changed'):
            append('full','negative','boundary',failed=1,expected_failure=True)
            append('full','negative','negative_control',label=label,rejected=True)

    def validate(self,records):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checks.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            return validate_results(path,self.cases)

    def test_complete_receipt(self):
        self.assertEqual(self.validate(self.records)['checks'],60)

    def test_protected_extents_must_match_complete_allocations(self):
        for index, record in enumerate(self.records):
            if record['kind'] != 'exact':
                continue
            for elements in (1, record['elements'] - 1, record['elements'] + 1,
                             True, float(record['elements']), None):
                with self.subTest(label=record['label'], elements=elements):
                    bad = copy.deepcopy(self.records)
                    bad[index]['elements'] = elements
                    with self.assertRaises(ValueError):
                        self.validate(bad)
        # Checking only initialized rotary rows omits the allocated guard row.
        bad = copy.deepcopy(self.records)
        next(r for r in bad if r.get('label') == 'a_cosine')['elements'] = 64
        with self.assertRaises(ValueError):
            self.validate(bad)

    def test_missing_duplicate_wrong_route_and_backend(self):
        bads=[self.records[1:],self.records+[self.records[0]],self.records[:-1]]
        for field,value in [('attention',6),('mlp',7),('backend','cpu')]:
            bad=copy.deepcopy(self.records);bad[0][field]=value;bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):self.validate(bad)

    def test_missing_guards_gates_and_elements(self):
        for field,value in [('elements',895),('failed',1),('inactive_exact',False)]:
            bad=copy.deepcopy(self.records);bad[1][field]=value
            with self.assertRaises(ValueError):self.validate(bad)
        bad=copy.deepcopy(self.records)
        del next(r for r in bad if r['kind']=='boundary' and r['stage']=='Y')['failed']
        with self.assertRaises(ValueError):self.validate(bad)
        bad=[r for r in self.records if r['kind']!='exact']
        with self.assertRaises(ValueError):self.validate(bad)
        bad=copy.deepcopy(self.records)
        next(r for r in bad if r['kind']=='cache')['prefix_exact']=False
        with self.assertRaises(ValueError):self.validate(bad)


if __name__=='__main__':unittest.main()
