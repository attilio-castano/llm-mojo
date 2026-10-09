"""Engine chat dispatch and the separate recomputing terminal report contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from llm_mojo.configuration import resolve_run, resolved_dict
from llm_mojo.cli import launch
from chat_terminal import validate_engine


def engine_rows(reason='limit', token=42):
    rows=[]
    def event(name,turn,index,value,ns=0):
        rows.append(dict(event=name,turn=str(turn),index=str(index),value=str(value),nanoseconds=str(ns)))
    event('load',0,0,'Apple M4 Pro/metal')
    event('engine_mode',0,0,'reference-27/recompute-history')
    previous=[]
    for turn in (1,2):
        prompt=previous+[5,6]
        generated=[] if reason=='interrupted' and token is None else [token]
        history=prompt+generated+([] if generated and generated[-1]==151645 else [151645])+[198]
        executed=1 if not generated else len(prompt)
        event('begin',turn,0,len(prompt))
        for i,t in enumerate(prompt): event('prompt',turn,i,t)
        for i,t in enumerate(generated): event('token',turn,i,t,100)
        event('finish',turn,0,reason,200)
        for name,value in [('submitted',24*executed),('engine_rows',executed),('engine_steps',1),
                           ('kv_free',128),('kv_total',128),('kv_owned',0),('kv_written',0),('engine_live',0)]:
            event(name,turn,0,value)
        for i,t in enumerate(history): event('history',turn,i,t)
        previous=history
    return rows


class EngineChatConfigurationTests(unittest.TestCase):
    def test_optional_engine_is_explicit_reference_and_direct_default_is_fast(self):
        direct=resolve_run('chat'); engine=resolve_run('chat',engine=True)
        self.assertEqual(direct.mode.name,'fast'); self.assertFalse(direct.workload.engine)
        self.assertEqual(engine.mode.name,'reference'); self.assertTrue(engine.workload.engine)
        self.assertTrue(resolved_dict(engine)['workload']['engine'])
        for options in ({'engine':1},{'engine':True,'mode':'fast'}):
            with self.assertRaises(ValueError): resolve_run('chat',**options)
        with self.assertRaises(ValueError): resolve_run('generate',engine=True)

    def test_launcher_keeps_public_chat_sidecar_and_selects_separate_native_binary(self):
        with tempfile.TemporaryDirectory() as d:
            report=Path(d)/'reply.tsv'
            cfg=resolve_run('chat',engine=True,report=report)
            with mock.patch.object(launch,'verify_prepared',return_value=(Path('/prepared'),{})), \
                 mock.patch.object(launch,'ensure_prepared',return_value=Path('/tables')), \
                 mock.patch.object(launch,'ensure_binary',return_value=Path('/chat-engine')) as build, \
                 mock.patch.object(launch.os,'execv') as execute:
                launch.launch_chat(cfg)
            build.assert_called_once_with('chat_engine','src/llm_mojo/cli/chat_engine_cli.mojo')
            self.assertEqual(execute.call_args.args[1][0],'/chat-engine')
            sidecar=json.loads(report.with_name(report.name+'.config.json').read_text())
            self.assertEqual(sidecar['command'],'chat')
            self.assertEqual(sidecar['configuration']['mode']['name'],'reference')
            self.assertTrue(sidecar['configuration']['workload']['engine'])

    def test_public_show_config_does_not_verify_or_build(self):
        from typer.testing import CliRunner
        from llm_mojo.cli.app import app
        with mock.patch.object(launch,'launch_chat') as native:
            result=CliRunner().invoke(app,['chat','--engine','--show-config'])
        self.assertEqual(result.exit_code,0,result.output)
        self.assertEqual(json.loads(result.output)['mode']['name'],'reference')
        native.assert_not_called()


class EngineChatReportTests(unittest.TestCase):
    def test_fresh_prefix_each_turn_and_exact_history(self):
        turns=validate_engine(engine_rows(),1)
        self.assertEqual([t['cached'] for t in turns],[0,0])
        self.assertEqual(turns[1]['prompt'][:len(turns[0]['history'])],turns[0]['history'])
        self.assertEqual(turns[1]['executed_rows'],len(turns[1]['prompt']))

    def test_eos_and_prefill_abort(self):
        self.assertEqual(validate_engine(engine_rows('stop',151643),1)[0]['reason'],'stop')
        self.assertEqual(validate_engine(engine_rows('interrupted',None),1)[0]['generated'],[])

    def test_counterfeit_lifecycle_and_rows_are_rejected(self):
        for name,value in [('kv_owned',1),('kv_written',1),('engine_live',1),('kv_free',127),
                           ('submitted',24),('engine_rows',3)]:
            with self.subTest(name=name):
                rows=copy.deepcopy(engine_rows())
                next(r for r in rows if r['event']==name)['value']=str(value)
                with self.assertRaises(ValueError): validate_engine(rows,1)
        rows=engine_rows(); next(r for r in rows if r['event']=='begin')['index']='1'
        with self.assertRaises(ValueError): validate_engine(rows,1)
        rows=[r for r in engine_rows() if r['event']!='engine_mode']
        with self.assertRaises(ValueError): validate_engine(rows,1)


if __name__=='__main__': unittest.main()
