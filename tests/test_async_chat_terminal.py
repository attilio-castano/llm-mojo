"""Terminal receipts must charge discarded async work and preserve exact history."""
import copy
import unittest

from chat_terminal import validate_engine
from test_engine_chat_launcher import engine_rows


def async_rows(reason='stop', token=151643, discarded=1, zero_work=False):
    rows=engine_rows(reason,token)
    next(r for r in rows if r['event']=='engine_mode')['value']='reference-27/async-two-context/recompute-history'
    for turn in ('1','2'):
        group=[r for r in rows if r['turn']==turn]
        generated=[r for r in group if r['event']=='token']
        prompt=[r for r in group if r['event']=='prompt']
        necessary=len(prompt)+len(generated)-1 if generated else 0
        total=0 if zero_work else necessary+discarded if generated else 2
        discard_rows=total-necessary
        discarded_tokens=discarded if generated and total else int(total==len(prompt)) if total else 0
        heads=len(generated)+discarded_tokens
        submissions=2 if total and (discarded_tokens or not generated) else 1 if total else 0
        counters=dict(submissions=submissions,completions=submissions,peak_pending=2 if submissions==2 else submissions,
            pending=0,submitted_rows=total,selected_heads=heads,delivered_tokens=len(generated),
            chained_rows=max(0,heads-1),discarded_tokens=discarded_tokens,discarded_rows=discard_rows)
        next(r for r in group if r['event']=='engine_rows')['value']=str(total)
        next(r for r in group if r['event']=='submitted')['value']=str(24*total)
        next(r for r in group if r['event']=='engine_steps')['value']=str(max(1,submissions))
        for name,value in counters.items():
            rows.append(dict(event='async_'+name,turn=turn,index='0',value=str(value),nanoseconds='0'))
    return rows


class AsyncChatTerminalTests(unittest.TestCase):
    def test_stop_keeps_delivered_history_and_charges_discarded_decode(self):
        turns=validate_engine(async_rows(),3,async_steps=True)
        self.assertEqual(turns[0]['generated'],[151643])
        self.assertEqual(turns[0]['executed_rows'],len(turns[0]['prompt'])+1)
        self.assertEqual(turns[0]['async']['discarded_tokens'],1)
        self.assertEqual(turns[1]['prompt'][:len(turns[0]['history'])],turns[0]['history'])

    def test_limit_suppresses_known_extra_step(self):
        turns=validate_engine(async_rows('limit',42,discarded=0),1,async_steps=True)
        self.assertEqual(turns[0]['async']['discarded_tokens'],0)
        self.assertEqual(turns[0]['executed_rows'],len(turns[0]['prompt']))
        with self.assertRaises(ValueError):
            validate_engine(async_rows('limit',42),1,async_steps=True)

    def test_partial_prefill_abort_counts_all_cancelled_rows(self):
        turns=validate_engine(async_rows('interrupted',None),3,async_steps=True)
        self.assertEqual(turns[0]['generated'],[])
        self.assertEqual(turns[0]['async']['discarded_rows'],2)
        self.assertEqual(turns[0]['async']['selected_heads'],1)
        self.assertEqual(turns[0]['async']['discarded_tokens'],1)
        self.assertEqual(turns[1]['async']['discarded_tokens'],0)

    def test_abort_before_any_gpu_work_has_zero_census(self):
        turns=validate_engine(async_rows('interrupted',None,zero_work=True),3,async_steps=True)
        self.assertEqual(turns[0]['executed_rows'],0)
        self.assertFalse(any(turns[0]['async'].values()))

    def test_async_needs_explicit_mode_and_preserves_sync_contract(self):
        with self.assertRaises(ValueError): validate_engine(async_rows(),3)
        with self.assertRaises(ValueError): validate_engine(engine_rows(),1,async_steps=True)
        rows=engine_rows()
        rows.append(dict(event='async_pending',turn='1',index='0',value='0',nanoseconds='0'))
        with self.assertRaises(ValueError): validate_engine(rows,1)

    def test_counterfeit_async_census_and_ownership_fail(self):
        for name,value in (('async_submissions',3),('async_completions',1),('async_peak_pending',3),
                ('async_pending',1),('async_submitted_rows',0),('async_selected_heads',1),
                ('async_delivered_tokens',2),('async_chained_rows',2),('async_discarded_tokens',0),
                ('async_discarded_rows',0),('kv_written',1),('kv_owned',1),('engine_live',1)):
            rows=copy.deepcopy(async_rows())
            next(r for r in rows if r['event']==name)['value']=str(value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_engine(rows,3,async_steps=True)

    def test_missing_duplicate_and_reordered_histories_fail(self):
        rows=async_rows()
        for corrupted in ([r for r in rows if r['event']!='async_completions'],
                rows+[copy.deepcopy(next(r for r in rows if r['event']=='async_pending'))]):
            with self.assertRaises(ValueError): validate_engine(corrupted,3,async_steps=True)
        rows=async_rows()
        next(r for r in rows if r['event']=='history')['index']='2'
        with self.assertRaises(ValueError): validate_engine(rows,3,async_steps=True)


if __name__=='__main__': unittest.main()
