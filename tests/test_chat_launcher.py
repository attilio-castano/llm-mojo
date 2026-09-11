"""Launcher boundaries: local verification precedes native execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from llm_mojo import chat


class ChatLauncherTests(unittest.TestCase):
    def test_invalid_limits_do_not_load_or_compile(self):
        for option in ('--max-new-tokens','--chunk-rows'):
            with self.subTest(option=option), patch.object(chat,'verify_prepared') as verify:
                with self.assertRaises(SystemExit): chat.main([option,'0'])
                verify.assert_not_called()

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'report.tsv';path.write_text('original')
            with patch.object(chat,'verify_prepared') as verify:
                with self.assertRaises(SystemExit): chat.main(['--report',str(path)])
                verify.assert_not_called()
            self.assertEqual(path.read_text(),'original')

    def test_native_exec_receives_verified_paths(self):
        with patch.object(chat,'verify_prepared',return_value=(Path('/prepared'),{})) as verify, \
             patch.object(chat,'ensure_prepared',return_value=Path('/tables')) as tables, \
             patch.object(chat,'ensure_binary',return_value=Path('/native')), \
             patch.object(chat.os,'execv') as execute:
            chat.main(['--prepared','/prepared','--max-new-tokens','12','--chunk-rows','16'])
        verify.assert_called_once_with(Path('/prepared'))
        tables.assert_called_once_with(download=False)
        execute.assert_called_once_with(Path('/native'),['/native','/prepared','/tables','12','16','',''])

    def test_changed_binary_is_rebuilt_even_with_matching_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/'build/chat';directory.mkdir(parents=True)
            binary=directory/'chat';binary.write_bytes(b'changed')
            (directory/'binary.json').write_text(json.dumps(dict(sources={'a':'b'},binary_sha256='stale')))
            def compile(command,**kwargs): Path(command[-1]).write_bytes(b'fresh')
            with patch.object(chat,'repository_root',return_value=root), \
                 patch.object(chat,'build_sources',return_value={'a':'b'}), \
                 patch.object(chat,'environment_tool',return_value='mojo'), \
                 patch.object(chat.subprocess,'run',side_effect=compile) as run:
                self.assertEqual(chat.ensure_binary(),binary)
                self.assertEqual(chat.ensure_binary(),binary)
            self.assertEqual(run.call_count,1)
            self.assertEqual(binary.read_bytes(),b'fresh')

class ChatEventTests(unittest.TestCase):
    def test_interrupt_after_last_token_was_consumed_is_valid(self):
        from chat_terminal import validate
        rows=[]
        def event(name,index,value,ns=0):
            rows.append(dict(event=name,turn='1',index=str(index),value=str(value),nanoseconds=str(ns)))
        event('load',0,'Apple M4 Pro/metal')
        event('begin',0,2)
        event('prompt',0,5);event('prompt',1,6)
        event('token',0,42,100)
        event('finish',3,'interrupted',200)
        event('submitted',0,72)
        for i,token in enumerate([5,6,42,151645,198]): event('history',i,token)
        self.assertEqual(validate(rows,1)[0]['cached'],3)
        # A normal budget stop leaves its last token pending instead.
        next(r for r in rows if r['event']=='finish')['value']='limit'
        with self.assertRaisesRegex(ValueError,'accounting'): validate(rows,1)

if __name__=='__main__': unittest.main()
