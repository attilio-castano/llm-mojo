"""The fitted cost policy must retain a numerical and evaluation boundary."""
import copy
import unittest

import numpy as np

from llm_mojo.benchmarks.engine_budget import (
    COEFFICIENTS, features, nonnegative_fit, summary, validate_policy, workload_identity,
)


class EngineBudgetTests(unittest.TestCase):
    def test_recovers_independent_nonnegative_cost_model(self):
        rng = np.random.default_rng(19)
        x = rng.integers(1,100,size=(100,5))
        x[:,0] = 1
        expected = [7000,110,3,900,270]
        actual=nonnegative_fit(x,x@expected)
        self.assertTrue(all(0<=a-b<=1 for a,b in zip(actual,expected)))
        self.assertTrue(np.all(x@actual>=x@expected))

    def test_nonnegative_fit_accepts_rank_deficiency_without_negative_cost(self):
        x = np.tile([1,1,1,1,1],(10,1))
        actual = nonnegative_fit(x,np.full(10,10))
        self.assertTrue(all(v>=0 for v in actual))
        self.assertEqual(sum(actual),10)
        with self.assertRaises(ValueError):
            nonnegative_fit(x,np.zeros(10))

    def test_features_follow_physical_attention_partitions_and_step_ownership(self):
        step = dict(step_id=1,decode_seqs=2,prefill_tokens=1,total_tokens=3,attended_positions=42)
        events = [dict(kind='token',step_id=0),dict(kind='token',step_id=1),dict(kind='finish',step_id=1)]
        self.assertEqual(features(step,events),[1,3,42,1,1])
        step['prefill_tokens']=128
        step['total_tokens']=130
        self.assertEqual(features(step,events),[1,130,42,2,1])

    def test_policy_rejects_missing_negative_and_unbounded_coefficients(self):
        policy = dict(kind='engine-step-cost-v1',schema_version=1,
                      cost=dict(zip(COEFFICIENTS,[1,2,3,4,5]),target_ns=25_000_000),
                      calibration_sha256='a'*64,calibration_trace_sha256='b'*64,
                      calibration_workload_sha256='c'*64)
        self.assertEqual(validate_policy(policy),policy['cost'])
        for value in [-1,1_000_000_000_001,True]:
            broken=copy.deepcopy(policy)
            broken['cost']['per_row_ns']=value
            with self.assertRaises(ValueError): validate_policy(broken)
        with self.assertRaises(ValueError):
            summary(dict(policy=policy,trace={},trace_sha256='b'*64))

    def test_workload_identity_ignores_json_metadata_and_formatting(self):
        from tests.test_engine_trace import trace_fixture
        first=trace_fixture()
        second=copy.deepcopy(first)
        second['seed']=123
        second['scripted_tokens']=[999]
        self.assertEqual(workload_identity(first),workload_identity(second))
        second['requests'][0]['prompt_ids']=[99]
        self.assertNotEqual(workload_identity(first),workload_identity(second))


def scheduling_native_fixture(budget=256, admission='reserved', mode='greedy', cost=None):
    from tests.test_engine_trace import native_fixture
    stdout = native_fixture('chunked', mode).replace('config chunked 4 256 2',
                f'config {"adaptive" if cost is not None else "chunked"} 4 {budget} 8')
    lines = stdout.splitlines()
    if cost is not None:
        from llm_mojo.benchmarks import model_contract as contract
        for i,line in enumerate(lines):
            if line.startswith('step '):
                fields=line.split()
                step=dict(zip(contract.ENGINE_STEP_FIELDS,map(int,fields[1:])))
                prediction=cost['fixed_ns']+cost['per_row_ns']+cost['per_position_ns']+cost['per_partition_ns']+cost['per_logit_ns']
                fields[contract.ENGINE_STEP_FIELDS.index('predicted_ns')+1]=str(prediction)
                lines[i]=' '.join(fields)
        lines.append('policy '+' '.join(str(cost[k]) for k in contract.ENGINE_POLICY_FIELDS))
    lines.extend(['admission '+admission, 'study engine-budget-v1', 'work_capacity 256 8'])
    return '\n'.join(lines)


def scheduling_fixture(stage='calibration', policy=None, calibration_bytes=None):
    import hashlib,json,base64
    from tests.test_engine_trace import study_fixture
    from llm_mojo.benchmarks import model_contract as contract, model_profile as profile
    from llm_mojo.benchmarks.engine_budget import scheduling_cells, scheduling_summary, _budget_command
    record=study_fixture()
    record.update(kind=contract.ENGINE_BUDGET_DECLARATION['kind'],declaration=contract.ENGINE_BUDGET_DECLARATION,
                  stage=stage,admission='reserved',max_sequences=8,native_trace_path='/collection/trace.tsv',runs=[])
    record['build']['declaration']=contract.ENGINE_BUDGET_DECLARATION
    record['build']['assets']['prepared']='/prepared'
    record['build']['command']=['mojo','build','-I','src','src/llm_mojo/benchmarks/engine_trace.mojo','-o','/build/engine']
    if stage=='evaluation':
        record['build']=copy.deepcopy(policy['build'])
        record['trace']['requests'][0]['prompt_ids']=[99]
        document=json.dumps(policy)
        record.update(policy=policy,policy_document=document,policy_sha256=hashlib.sha256(document.encode()).hexdigest(),
                      calibration_archive_base64=base64.b64encode(calibration_bytes).decode(),native_policy_path='/collection/policy.tsv',
                      native_policy_sha256=hashlib.sha256(('cost '+' '.join(str(policy['cost'][k]) for k in (*COEFFICIENTS,'target_ns'))+'\n').encode()).hexdigest())
    record['trace_document']=json.dumps(record['trace'])
    record['trace_sha256']=hashlib.sha256(record['trace_document'].encode()).hexdigest()
    record['native_trace_sha256']=hashlib.sha256(profile.engine_trace_tsv(record['trace']).encode()).hexdigest()
    conditions=dict(battery={'power_source':'AC Power'},power_mode_raw='0',thermal=[
                    'No thermal warning level has been recorded','No performance warning level has been recorded'])
    for block in range(4):
        for name,arm,budget,calibration in scheduling_cells(stage,block):
            cost=policy['cost'] if arm=='adaptive' else None
            stdout=scheduling_native_fixture(budget,cost=cost)
            run=dict(block=block,budget_arm=name,arm=arm,token_budget=budget,calibration=calibration,stdout=stdout,
                     parsed=profile.parse_engine_run(stdout,record['trace'],arm,4,8,'greedy',policy=cost,
                        admission='reserved',study=contract.ENGINE_BUDGET_STUDY,token_budget=budget),
                     conditions_before=conditions,conditions_after=conditions)
            run['execution']=dict(command=_budget_command(record,run),timeout_seconds=180,wall_elapsed_ns=100,exit_code=0)
            record['runs'].append(run)
    record['summary']=scheduling_summary(record)
    return record


def write_scheduling_archive(root,record):
    import hashlib,json,gzip
    raw=json.dumps(record).encode()
    compressed=gzip.compress(raw,mtime=0)
    (root/'engine-budget.json.gz').write_bytes(compressed)
    (root/'engine-budget.json').write_text(json.dumps(dict(kind=record['kind'],bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
    return compressed


class SchedulingStudyTests(unittest.TestCase):
    def fitted_fixture(self,root):
        from llm_mojo.benchmarks.engine_budget import scheduling_fit
        calibration=scheduling_fixture()
        compressed=write_scheduling_archive(root,calibration)
        policy=scheduling_fit(root,root/'policy.json')
        return scheduling_fixture('evaluation',policy,compressed),policy

    def test_fixed_workspace_budget_parse_is_explicit_and_legacy_unchanged(self):
        from llm_mojo.benchmarks import model_profile as profile
        from tests.test_engine_trace import trace_fixture,native_fixture
        for budget in (32,64,128,256):
            stdout=scheduling_native_fixture(budget)
            parsed=profile.parse_engine_run(stdout,trace_fixture(),'chunked',4,8,'greedy',admission='reserved',
                                            study='engine-budget-v1',token_budget=budget)
            self.assertEqual(parsed['token_budget'],budget)
            self.assertEqual(parsed['work_capacity'],dict(token_rows=256,max_sequences=8))
            with self.assertRaises(ValueError):
                profile.parse_engine_run(stdout,trace_fixture(),'chunked',4,8,'greedy',admission='reserved')
            with self.assertRaises(ValueError):
                profile.parse_engine_run(stdout.replace('work_capacity 256 8','work_capacity 32 8'),trace_fixture(),
                    'chunked',4,8,'greedy',admission='reserved',study='engine-budget-v1',token_budget=budget)
        self.assertNotIn('study',profile.parse_engine_run(native_fixture('chunked'),trace_fixture(),'chunked',4,2,'greedy'))

    def test_calibration_twenty_runs_and_embedded_fitted_evaluation_twenty_four_replay(self):
        import tempfile
        from pathlib import Path
        from llm_mojo.benchmarks import model_profile as profile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            evaluation,policy=self.fitted_fixture(root)
            self.assertEqual(policy['kind'],'engine-step-cost-v2')
            self.assertEqual(len(policy['calibration_samples']),40)
            self.assertEqual(len(evaluation['runs']),24)
            write_scheduling_archive(root,evaluation)
            (root/'policy.json').unlink()
            self.assertEqual(profile.engine_replay(root),evaluation['summary'])
            self.assertEqual(evaluation['summary']['target_ns'],25_000_000)
            self.assertTrue(all(c['same_generated_histories'] for c in evaluation['summary']['comparisons']))

    def test_corruption_rejected_with_fresh_envelope_hashes(self):
        import tempfile
        from pathlib import Path
        from llm_mojo.benchmarks import model_profile as profile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            evaluation,policy=self.fitted_fixture(root)
            for kind in ('budget','order','type','workspace','admission','history','policy','calibration','wall','summary'):
                broken=copy.deepcopy(evaluation)
                if kind=='budget': broken['runs'][2]['token_budget']=256
                if kind=='order': broken['runs'][0],broken['runs'][1]=broken['runs'][1],broken['runs'][0]
                if kind=='type': broken['runs'][1]['calibration']=1
                if kind=='workspace': broken['runs'][0]['stdout']=broken['runs'][0]['stdout'].replace('work_capacity 256 8','work_capacity 32 8')
                if kind=='admission': broken['admission']='incremental'
                if kind=='history': broken['runs'][2]['stdout']=broken['runs'][2]['stdout'].replace('token 0 11','token 0 12')
                if kind=='policy': broken['policy']['cost']['per_row_ns']+=1
                if kind=='calibration': broken['calibration_archive_base64']='AAAA'
                if kind=='wall': broken['runs'][0]['execution']['wall_elapsed_ns']=1
                if kind=='summary': broken['summary']['runs'][0]['duration_ns']=1
                write_scheduling_archive(root,broken)
                with self.subTest(kind=kind),self.assertRaises((ValueError,EOFError,OSError)):
                    profile.engine_replay(root)

    def test_collector_rejects_mutation_of_generated_native_input(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks.engine_budget import scheduling_collect
        record=scheduling_fixture()
        receipt=copy.deepcopy(record['build'])
        def mutate(command,log,timeout):
            Path(command[2]).write_text('script 999\nrequest 0 0 1 - 99 -\n')
            return scheduling_native_fixture(int(command[5]))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            receipt['command'][-1]=str(root/'build'/'engine')
            trace=root/'trace.json';trace.write_text(record['trace_document'])
            with mock.patch('llm_mojo.benchmarks.model_profile.environment_tool',return_value='mojo'), \
                 mock.patch('llm_mojo.benchmarks.model_profile.verify_build',return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions',return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'generated input changed during'):
                    scheduling_collect(root/'build',trace,root/'mutated','calibration','reserved',blocks=4)
            self.assertFalse((root/'mutated'/'engine-budget.json.gz').exists())

    def test_new_policy_cannot_enter_legacy_adaptive_evaluation(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            evaluation,policy=self.fitted_fixture(Path(d))
            with self.assertRaisesRegex(ValueError,'v1 policy'):
                summary(dict(kind='engine-adaptive-evaluation-v1',policy=policy))

    def test_failed_cell_retains_partial_log_receipt_and_no_complete_archive(self):
        import tempfile,json
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks.engine_budget import scheduling_collect
        from llm_mojo.benchmarks.model_profile import NativeCommandError
        record=scheduling_fixture()
        receipt=copy.deepcopy(record['build'])
        def failed(command,log,timeout):
            log.write_text('partial native output\n')
            raise NativeCommandError(7,log)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            receipt['command'][-1]=str(root/'build'/'engine')
            trace=root/'trace.json';trace.write_text(record['trace_document'])
            with mock.patch('llm_mojo.benchmarks.model_profile.environment_tool',return_value='mojo'), \
                 mock.patch('llm_mojo.benchmarks.model_profile.verify_build',return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions',return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute',side_effect=failed):
                with self.assertRaisesRegex(RuntimeError,'command failed'):
                    scheduling_collect(root/'build',trace,root/'failed','calibration','reserved',blocks=4)
            self.assertEqual((root/'failed'/'block-0-fixed-256.log').read_text(),'partial native output\n')
            execution=json.loads((root/'failed'/'block-0-fixed-256.execution.json').read_text())
            self.assertEqual(execution['exit_code'],7)
            self.assertEqual(execution['timeout_seconds'],180)
            self.assertFalse((root/'failed'/'engine-budget.json.gz').exists())

    def test_updated_policy_document_still_has_to_reproduce_the_calibration_fit(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from llm_mojo.benchmarks.engine_budget import scheduling_summary
        with tempfile.TemporaryDirectory() as d:
            evaluation,policy=self.fitted_fixture(Path(d))
            evaluation['policy']['cost']['per_row_ns']+=1
            document=json.dumps(evaluation['policy'])
            evaluation.update(policy_document=document,policy_sha256=hashlib.sha256(document.encode()).hexdigest(),
                native_policy_sha256=hashlib.sha256(('cost '+' '.join(str(evaluation['policy']['cost'][k]) for k in (*COEFFICIENTS,'target_ns'))+'\n').encode()).hexdigest())
            with self.assertRaisesRegex(ValueError,'does not reproduce'):
                scheduling_summary(evaluation)

    def test_calibration_cannot_fit_virtual_time_or_change_target(self):
        import tempfile
        from pathlib import Path
        from llm_mojo.benchmarks.engine_budget import scheduling_fit,budget_calibration_samples
        record=scheduling_fixture()
        record['mode']='scripted'
        with self.assertRaises(ValueError): budget_calibration_samples(record)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): scheduling_fit(Path(d),Path(d)/'p.json',target_ns=10_000_000)

    def test_evaluation_requires_different_canonical_native_workload(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from llm_mojo.benchmarks.engine_budget import scheduling_summary
        with tempfile.TemporaryDirectory() as d:
            evaluation,policy=self.fitted_fixture(Path(d))
            evaluation['trace']['requests'][0]['prompt_ids']=[42]
            evaluation['trace']['seed']=999
            from llm_mojo.benchmarks import model_profile as profile
            evaluation['native_trace_sha256']=hashlib.sha256(profile.engine_trace_tsv(evaluation['trace']).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError,'separate native workload'):
                scheduling_summary(evaluation)

    def test_evaluation_collector_runs_all_twenty_four_cells_with_frozen_policy(self):
        import tempfile,json,base64,gzip
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks.engine_budget import scheduling_fit,scheduling_collect
        record=scheduling_fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            record['build']['command'][-1]=str(root/'build'/'engine')
            for run in record['runs']:
                run['execution']['command'][0]=record['build']['command'][-1]
            compressed=write_scheduling_archive(root,record)
            policy=scheduling_fit(root,root/'policy.json')
            trace=copy.deepcopy(record['trace']);trace['requests'][0]['prompt_ids']=[99]
            trace_path=root/'heldout.json';trace_path.write_text(json.dumps(trace))
            observed=[]
            def execute(command,log,timeout):
                observed.append(command)
                adaptive=command[3]=='adaptive'
                self.assertEqual(command[-2:],['reserved','engine-budget-v1'])
                if adaptive:
                    self.assertEqual(Path(command[9]).name,'policy.tsv')
                    self.assertTrue(Path(command[9]).read_text().startswith('cost '))
                return scheduling_native_fixture(int(command[5]),cost=policy['cost'] if adaptive else None)
            with mock.patch('llm_mojo.benchmarks.model_profile.environment_tool',return_value='mojo'), \
                 mock.patch('llm_mojo.benchmarks.model_profile.verify_build',return_value=record['build']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions',return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute',side_effect=execute):
                result=scheduling_collect(root/'build',trace_path,root/'evaluation','evaluation','reserved',
                                          root/'policy.json',blocks=4)
            self.assertEqual(len(observed),24)
            self.assertEqual(sum(c[3]=='adaptive' for c in observed),4)
            self.assertEqual(len(result['comparisons']),4)
            collected=json.loads(gzip.decompress((root/'evaluation'/'engine-budget.json.gz').read_bytes()))
            self.assertEqual(base64.b64decode(collected['calibration_archive_base64']),compressed)

    def test_collector_uses_fixed_build_workspace_balanced_grid_and_actual_receipts(self):
        import tempfile,json
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks.engine_budget import scheduling_collect
        record=scheduling_fixture()
        receipt=copy.deepcopy(record['build'])
        observed=[]
        def execute(command,log,timeout):
            self.assertEqual(command[6:9],['8','10','greedy'])
            self.assertEqual(command[-2:],['reserved','engine-budget-v1'])
            self.assertEqual(timeout,180)
            observed.append(int(command[5]))
            return scheduling_native_fixture(int(command[5]))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            receipt['command'][-1]=str(root/'build'/'engine')
            trace=root/'trace.json';trace.write_text(record['trace_document'])
            with mock.patch('llm_mojo.benchmarks.model_profile.environment_tool',return_value='mojo'), \
                 mock.patch('llm_mojo.benchmarks.model_profile.verify_build',return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions',return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute',side_effect=execute):
                result=scheduling_collect(root/'build',trace,root/'out','calibration','reserved',blocks=4)
            self.assertEqual(len(observed),20)
            self.assertEqual(observed[:5],[256,256,32,64,128])
            self.assertEqual(result,record['summary'])
            self.assertEqual(len(list((root/'out').glob('*.execution.json'))),20)
            with self.assertRaises(ValueError): scheduling_collect(root/'build',trace,root/'bad','calibration',None)


def fast_native_fixture(runner='reference', budget=64, token=11, partial=False):
    from llm_mojo.benchmarks import model_contract as contract
    if not partial:
        stdout=scheduling_native_fixture(budget).replace('study engine-budget-v1','study engine-fast-v1')
        return stdout+'\nrunner '+runner+'\n'+'\n'.join(f'route {i} {26 if runner=="fast-decode" else 27} 1 1 1' for i in range(2)) if token==11 else fast_native_fixture(runner,budget).replace('token 0 11','token 0 '+str(token)).replace('token 1 11','token 1 '+str(token))
    lines=['device Apple M4 Pro/metal','mode greedy',f'config chunked 4 {budget} 8',
           'admission reserved','study engine-fast-v1','work_capacity 256 8','runner '+runner,
           'arrival 0 0 0','arrival 1 0 0','arrival 2 0 0']
    for i in range(4):
        active=i<3
        fields=dict(step_id=i,decode_seqs=0,prefill_seqs=int(active),prefill_tokens=int(active),
                    total_tokens=int(active),attended_positions=2 if i==1 else int(active),admitted=int(i in (0,2)),
                    preempted=0,finished=int(i>0),aborted=0,waiting=max(2-i,0),blocks_free=3 if i==0 else 4,
                    begin_ns=i*10,schedule_ns=1,build_ns=int(active),execute_ns=6 if active else 0,
                    postprocess_ns=1,end_ns=(i+1)*10,predicted_ns=0,budget_limited=0)
        lines.append('step '+' '.join(str(fields[k]) for k in contract.ENGINE_STEP_FIELDS))
        if active:
            selected=int(i>0);configuration=26 if runner=='fast-decode' and selected else 27
            lines.append(f'route {i} {configuration} 1 1 {selected}')
        if i in (1,2):
            identifier=i-1;prompt=2 if identifier==0 else 1
            lines += [f'token {identifier} 11 {prompt} 1 0 {(i+1)*10}',f'finish {identifier} length {prompt} 1 0 {(i+1)*10}']
        if i==3: lines.append('finish 2 length 1 0 0 40')
    lines.append('drained 3 4 4 40')
    return '\n'.join(lines)


def fast_fixture(divergent=False):
    import hashlib,json
    from llm_mojo.benchmarks import model_contract as contract,model_profile as profile
    from llm_mojo.benchmarks.engine_budget import fast_cells,fast_summary,_fast_command,workload_identity
    record=scheduling_fixture()
    record.pop('stage')
    record.update(kind=contract.ENGINE_FAST_DECLARATION['kind'],declaration=contract.ENGINE_FAST_DECLARATION,
                  token_budget=64,runs=[])
    record['build']['declaration']=contract.ENGINE_FAST_DECLARATION
    record['build']['source']['sources']['tests/engine_metal_driver.mojo']='9'*64
    conditions=dict(battery={'power_source':'AC Power'},power_mode_raw='0',thermal=[
                    'No thermal warning level has been recorded','No performance warning level has been recorded'])
    routes=[]
    for runner in contract.ENGINE_FAST_ROUTES:
        token=12 if divergent and runner=='fast-decode' else 11
        log='device Apple M4 Pro/metal\nqualification engine-fast-checkpoint-v1\nrunner '+runner+'\n'
        log+=''.join('check '+name+' checked 100 mismatches 0 unexpected-nonfinite 0\n' for name in contract.ENGINE_FAST_CHECKPOINT_CHECKS)
        driver_build=dict(source=copy.deepcopy(record['build']['source']),assets=copy.deepcopy(record['build']['assets']),
                          environment=copy.deepcopy(record['build']['environment']),
                          command=['mojo','build','-I','src','-I','tests','tests/engine_metal_driver.mojo','-o','/qualification/checkpoint-driver'],
                          binaries={'checkpoint-driver':dict(sha256='f'*64,bytes=1)})
        natural_path='/qualification/'+runner+'/trace.tsv'
        native=fast_native_fixture(runner,token=token)
        parsed=profile.parse_engine_run(native,record['trace'],'chunked',4,8,'greedy',admission='reserved',
                            study=contract.ENGINE_FAST_STUDY,token_budget=64,runner=runner)
        routes.append(dict(runner=runner,
                           natural_run=dict(native_trace_path=natural_path,native_trace_sha256=record['native_trace_sha256'],
                               stdout=native,parsed=parsed,execution=dict(command=_fast_command(dict(record,native_trace_path=natural_path),dict(runner=runner)),
                                         timeout_seconds=180,wall_elapsed_ns=100,exit_code=0)),
                           checkpoint_receipt=dict(build=driver_build,stdout=log,stdout_sha256=hashlib.sha256(log.encode()).hexdigest(),
                               execution=dict(command=['/qualification/checkpoint-driver','/prepared','fast-qualification',runner],
                                              exit_code=0,timeout_seconds=180,wall_elapsed_ns=100))))
    qualification=dict(kind='engine-fast-qualification-v1',schema_version=1,build=copy.deepcopy(record['build']),
                       workload_sha256=workload_identity(record['trace']),native_trace_sha256=record['native_trace_sha256'],
                       token_budget=64,blocks=4,admission='reserved',work_capacity=dict(token_rows=256,max_sequences=8),routes=routes)
    document=json.dumps(qualification)
    record.update(qualification=qualification,qualification_document=document,qualification_sha256=hashlib.sha256(document.encode()).hexdigest())
    for block in range(4):
        for runner,calibration in fast_cells(block):
            token=12 if divergent and runner=='fast-decode' else 11
            stdout=fast_native_fixture(runner,token=token)
            run=dict(block=block,runner=runner,calibration=calibration,stdout=stdout,
                     parsed=profile.parse_engine_run(stdout,record['trace'],'chunked',4,8,'greedy',admission='reserved',
                            study=contract.ENGINE_FAST_STUDY,token_budget=64,runner=runner),
                     conditions_before=conditions,conditions_after=conditions)
            run['execution']=dict(command=_fast_command(record,run),timeout_seconds=180,wall_elapsed_ns=100,exit_code=0)
            record['runs'].append(run)
    record['summary']=fast_summary(record)
    return record


def write_fast_archive(root,record):
    import hashlib,json,gzip
    raw=json.dumps(record).encode();compressed=gzip.compress(raw,mtime=0)
    (root/'engine-fast.json.gz').write_bytes(compressed)
    (root/'engine-fast.json').write_text(json.dumps(dict(kind=record['kind'],bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest())))


class FastStudyTests(unittest.TestCase):
    def test_partial_singleton_zero_heads_and_empty_steps_follow_actual_routes(self):
        from tests.test_engine_trace import trace_fixture
        from llm_mojo.benchmarks import model_profile as profile
        trace=trace_fixture();trace['requests'][0]['prompt_ids']=[42,43]
        request=copy.deepcopy(trace['requests'][1]);request.update(request_id=2,max_new_tokens=0);trace['requests'].append(request)
        for runner in ('reference','fast-decode'):
            parsed=profile.parse_engine_run(fast_native_fixture(runner,partial=True),trace,'chunked',4,8,'greedy',
                       admission='reserved',study='engine-fast-v1',token_budget=64,runner=runner)
            self.assertEqual([r['selected_logits'] for r in parsed['routes']],[0,1,1])
            self.assertEqual([r['configuration'] for r in parsed['routes']],[27,26,26] if runner=='fast-decode' else [27]*3)
            self.assertEqual(len(parsed['steps']),4)
        broken=fast_native_fixture('fast-decode',partial=True).replace('route 0 27 1 1 0','route 0 27 1 0 0')
        lines=broken.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('step 0 '):
                fields=line.split();fields[profile.contract.ENGINE_STEP_FIELDS.index('prefill_seqs')+1]='0'
                lines[i]=' '.join(fields)
        with self.assertRaisesRegex(ValueError,'completed route'):
            profile.parse_engine_run('\n'.join(lines),trace,'chunked',4,8,'greedy',admission='reserved',
                       study='engine-fast-v1',token_budget=64,runner='fast-decode')

    def test_routes_reject_missing_duplicate_false_geometry_and_counterfeit_simulation(self):
        from tests.test_engine_trace import trace_fixture
        from llm_mojo.benchmarks import model_profile as profile
        stdout=fast_native_fixture('fast-decode')
        for broken in [stdout.replace('route 0 26 1 1 1',''),stdout+'\nroute 0 26 1 1 1',
                       stdout.replace('route 0 26 1 1 1','route 0 27 1 1 1'),
                       stdout.replace('route 0 26 1 1 1','route 0 26 2 1 1'),
                       stdout.replace('route 0 26 1 1 1','route 0 26 1 2 1'),
                       stdout.replace('route 0 26 1 1 1','route 0 26 1 1 0'),
                       stdout.replace('runner fast-decode','runner reference')]:
            with self.subTest(broken=broken[-50:]),self.assertRaises(ValueError):
                profile.parse_engine_run(broken,trace_fixture(),'chunked',4,8,'greedy',admission='reserved',
                        study='engine-fast-v1',token_budget=64,runner='fast-decode')
        with self.assertRaises(ValueError):
            profile.parse_engine_run(stdout.replace('device Apple M4 Pro/metal','device simulated/virtual').replace('mode greedy','mode scripted'),
                       trace_fixture(),'chunked',4,8,'scripted',admission='reserved',study='engine-fast-v1',token_budget=64,runner='fast-decode')
        with self.assertRaises(ValueError):
            profile.parse_engine_run(stdout.replace('study engine-fast-v1','study engine-budget-v1'),trace_fixture(),'chunked',4,8,'greedy',
                        admission='reserved',study='engine-budget-v1',token_budget=64)

    def test_matched_twelve_cell_archive_replays_routes_qualification_and_work(self):
        import tempfile
        from pathlib import Path
        from llm_mojo.benchmarks import model_profile as profile
        record=fast_fixture()
        self.assertEqual(len(record['runs']),12)
        self.assertTrue(record['summary']['performance_eligible'])
        self.assertFalse(record['summary']['promotion'])
        self.assertTrue(all(record['summary']['comparison']['matched_work']))
        self.assertEqual(record['summary']['runs'][2]['route_steps'],{'26':2,'27':0})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fast_archive(root,record)
            self.assertEqual(profile.engine_replay(root),record['summary'])

    def test_cross_route_arithmetic_difference_is_archived_without_speed_verdict(self):
        import tempfile
        from pathlib import Path
        from llm_mojo.benchmarks import model_profile as profile
        record=fast_fixture(divergent=True)
        self.assertFalse(record['summary']['performance_eligible'])
        self.assertIsNone(record['summary']['comparison']['speed_verdict'])
        self.assertTrue(all(r['matches_qualified_history'] for r in record['summary']['runs']))
        self.assertEqual(sum(d['kind']=='cross-route-history-difference' for d in record['summary']['diagnostics']),4)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fast_archive(root,record)
            self.assertEqual(profile.engine_replay(root),record['summary'])

    def test_own_route_repeat_difference_retains_expected_and_actual_histories(self):
        from llm_mojo.benchmarks import model_profile as profile
        from llm_mojo.benchmarks.engine_budget import fast_summary
        record=fast_fixture();run=record['runs'][2]
        run['stdout']=run['stdout'].replace('token 0 11','token 0 12')
        run['parsed']=profile.parse_engine_run(run['stdout'],record['trace'],'chunked',4,8,'greedy',admission='reserved',
                         study='engine-fast-v1',token_budget=64,runner='fast-decode')
        result=fast_summary(record)
        self.assertIsNone(result['comparison']['speed_verdict'])
        self.assertIn('own-route-repeat-history-difference',[d['kind'] for d in result['diagnostics']])
        self.assertIn('own-route-qualified-history-difference',[d['kind'] for d in result['diagnostics']])

    def test_fresh_envelope_does_not_hide_qualification_receipt_route_or_grid_corruption(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from llm_mojo.benchmarks import model_profile as profile
        for kind in ('order','type','budget','receipt','route','qualification','log','stop'):
            record=fast_fixture()
            if kind=='order':record['runs'][0],record['runs'][1]=record['runs'][1],record['runs'][0]
            if kind=='type':record['runs'][0]['calibration']=0
            if kind=='budget':record['token_budget']=32
            if kind=='receipt':record['runs'][0]['execution']['wall_elapsed_ns']=1
            if kind=='route':record['runs'][0]['stdout']=record['runs'][0]['stdout'].replace('route 0 27 1 1 1','route 0 26 1 1 1')
            if kind in ('qualification','log','stop'):
                if kind=='qualification':record['qualification']['schema_version']=1.0
                if kind=='log':record['qualification']['routes'][0]['checkpoint_receipt']['stdout']='changed'
                if kind=='stop':record['qualification']['routes'][0]['natural_run']['stdout']=record['qualification']['routes'][0]['natural_run']['stdout'].replace('finish 0 length','finish 0 stop')
                document=json.dumps(record['qualification']);record.update(qualification_document=document,qualification_sha256=hashlib.sha256(document.encode()).hexdigest())
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);write_fast_archive(root,record)
                with self.subTest(kind=kind),self.assertRaises(ValueError):profile.engine_replay(root)

    def test_collector_uses_explicit_shape_runner_and_retains_complete_divergence_archive(self):
        import tempfile,json,gzip
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks.engine_budget import fast_collect
        record=fast_fixture(divergent=True)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);receipt=record['build'];receipt['command'][-1]=str(root/'build'/'engine')
            record['qualification']['build']=copy.deepcopy(receipt)
            for route in record['qualification']['routes']:
                route['natural_run']['execution']['command'][0]=receipt['command'][-1]
            trace=root/'trace.json';trace.write_text(record['trace_document'])
            qualification=root/'qualification.json';qualification.write_text(json.dumps(record['qualification']))
            import hashlib
            binary=root/'checkpoint-driver';binary.write_bytes(b'fixture driver')
            for route in record['qualification']['routes']:
                checkpoint=route['checkpoint_receipt'];checkpoint['build']['command'][-1]=str(binary)
                checkpoint['build']['binaries']['checkpoint-driver']=dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),bytes=binary.stat().st_size)
                checkpoint['execution']['command'][0]=str(binary)
            qualification.write_text(json.dumps(record['qualification']))
            observed=[]
            def execute(command,log,timeout):
                observed.append(command);self.assertEqual(command[-3:-1],['reserved','engine-fast-v1'])
                return fast_native_fixture(command[-1],int(command[5]),12 if command[-1]=='fast-decode' else 11)
            with mock.patch('llm_mojo.benchmarks.model_profile.environment_tool',return_value='mojo'), \
                 mock.patch('llm_mojo.benchmarks.model_profile.verify_build',return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions',return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute',side_effect=execute):
                result=fast_collect(root/'build',trace,root/'collected',qualification,64,'reserved',blocks=4)
            self.assertEqual(len(observed),12);self.assertIsNone(result['comparison']['speed_verdict'])
            retained=json.loads(gzip.decompress((root/'collected'/'engine-fast.json.gz').read_bytes()))
            self.assertEqual(retained['qualification'],record['qualification'])

    def test_failed_numerical_qualification_stops_before_native_and_retains_diagnostics(self):
        import tempfile,json
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks.engine_budget import fast_collect
        import hashlib
        record=fast_fixture();receipt=record['qualification']['routes'][1]['checkpoint_receipt']
        receipt['stdout']=receipt['stdout'].replace('check mixed-fallback checked 100 mismatches 0 unexpected-nonfinite 0','check mixed-fallback checked 100 mismatches 1 unexpected-nonfinite 0')
        receipt['stdout_sha256']=hashlib.sha256(receipt['stdout'].encode()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);receipt=record['build'];receipt['command'][-1]=str(root/'build'/'engine')
            record['qualification']['build']=copy.deepcopy(receipt)
            for route in record['qualification']['routes']:
                route['natural_run']['execution']['command'][0]=receipt['command'][-1]
            trace=root/'trace.json';trace.write_text(record['trace_document'])
            qualification=root/'qualification.json';qualification.write_text(json.dumps(record['qualification']))
            with mock.patch('llm_mojo.benchmarks.model_profile.environment_tool',return_value='mojo'), \
                 mock.patch('llm_mojo.benchmarks.model_profile.verify_build',return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute') as execute:
                with self.assertRaisesRegex(ValueError,'qualification failed'):
                    fast_collect(root/'build',trace,root/'failed',qualification,64,'reserved',blocks=4)
                execute.assert_not_called()
            self.assertTrue((root/'failed'/'qualification.json').exists())
            self.assertTrue((root/'failed'/'engine-fast-failure.json').exists())


    def test_qualification_binds_checked_native_outputs_driver_build_and_actual_history_command(self):
        import json,hashlib
        from llm_mojo.benchmarks.engine_budget import fast_summary
        for kind in ('arbitrary-command','other-source','other-assets','other-entrypoint','missing-check',
                     'unordered-check','negative-check','wrong-natural-command','wrong-natural-runner'):
            record=fast_fixture();route=record['qualification']['routes'][0];receipt=route['checkpoint_receipt']
            if kind=='arbitrary-command':receipt['execution']['command']=['true']
            if kind=='other-source':receipt['build']['source']['sources']['tests/engine_metal_driver.mojo']='8'*64
            if kind=='other-assets':receipt['build']['assets']['prepared_sha256']='8'*64
            if kind=='other-entrypoint':receipt['build']['command'][-3]='tests/unrelated.mojo'
            if kind=='missing-check':receipt['stdout']='\n'.join(receipt['stdout'].splitlines()[:-1])
            if kind=='unordered-check':
                lines=receipt['stdout'].splitlines();lines[3],lines[4]=lines[4],lines[3];receipt['stdout']='\n'.join(lines)
            if kind=='negative-check':receipt['stdout']=receipt['stdout'].replace('checked 100','checked -1',1)
            if kind=='wrong-natural-command':route['natural_run']['execution']['command'][0]='/unrelated/engine'
            if kind=='wrong-natural-runner':route['natural_run']['stdout']=route['natural_run']['stdout'].replace('runner reference','runner fast-decode')
            receipt['stdout_sha256']=hashlib.sha256(receipt['stdout'].encode()).hexdigest()
            document=json.dumps(record['qualification']);record.update(qualification_document=document,qualification_sha256=hashlib.sha256(document.encode()).hexdigest())
            with self.subTest(kind=kind),self.assertRaises(ValueError):fast_summary(record)

    def test_changed_self_control_work_disables_speed_even_with_same_histories(self):
        from llm_mojo.benchmarks import model_profile as profile
        from llm_mojo.benchmarks.engine_budget import fast_summary
        record=fast_fixture();run=record['runs'][1]
        lines=run['stdout'].splitlines()
        for i,line in enumerate(lines):
            if line.startswith('step 0 '):
                fields=line.split();index=profile.contract.ENGINE_STEP_FIELDS.index('attended_positions')+1
                fields[index]=str(int(fields[index])+1);lines[i]=' '.join(fields)
        run['stdout']='\n'.join(lines)
        run['parsed']=profile.parse_engine_run(run['stdout'],record['trace'],'chunked',4,8,'greedy',admission='reserved',
                         study='engine-fast-v1',token_budget=64,runner='reference')
        summary=fast_summary(record)
        self.assertIsNone(summary['comparison']['speed_verdict'])
        self.assertIn('self-reference-work-difference',[d['kind'] for d in summary['diagnostics']])

    def test_equal_aggregate_work_cannot_hide_different_per_step_attention(self):
        import tempfile
        from pathlib import Path
        from llm_mojo.benchmarks import model_profile as profile
        from llm_mojo.benchmarks.engine_budget import fast_summary
        for cell,runner,diagnostic in ((1,'reference','self-reference-step-work-difference'),
                                       (2,'fast-decode','cross-route-step-work-difference')):
            record=fast_fixture();run=record['runs'][cell]
            before=profile.engine_run_summary(run['parsed'],record['trace'])
            lines=run['stdout'].splitlines()
            for i,line in enumerate(lines):
                if line.startswith('step '):
                    fields=line.split();step_id=int(fields[1])
                    # Redistributing attention work preserves every aggregate,
                    # token history, timing and route/head count in the archive.
                    fields[profile.contract.ENGINE_STEP_FIELDS.index('attended_positions')+1]=str(2 if step_id==0 else 0)
                    lines[i]=' '.join(fields)
            run['stdout']='\n'.join(lines)
            run['parsed']=profile.parse_engine_run(run['stdout'],record['trace'],'chunked',4,8,'greedy',
                         admission='reserved',study='engine-fast-v1',token_budget=64,runner=runner)
            after=profile.engine_run_summary(run['parsed'],record['trace'])
            self.assertEqual([before[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions')],
                             [after[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions')])
            self.assertEqual(before['requests'],after['requests'])
            record['summary']=fast_summary(record)
            self.assertFalse(record['summary']['performance_eligible'])
            self.assertIsNone(record['summary']['comparison']['speed_verdict'])
            self.assertIn(diagnostic,[d['kind'] for d in record['summary']['diagnostics']])
            if cell==2:
                self.assertFalse(record['summary']['comparison']['matched_step_work'][0])
                self.assertFalse(record['summary']['comparison']['matched_work'][0])
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);write_fast_archive(root,record)
                self.assertEqual(profile.engine_replay(root),record['summary'])

    def test_failed_fast_cell_retains_partial_log_and_actual_failed_execution_receipt(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks.engine_budget import fast_collect
        from llm_mojo.benchmarks.model_profile import NativeCommandError
        record=fast_fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);receipt=record['build'];receipt['command'][-1]=str(root/'build'/'engine')
            record['qualification']['build']=copy.deepcopy(receipt)
            binary=root/'checkpoint-driver';binary.write_bytes(b'fixture driver')
            for route in record['qualification']['routes']:
                route['natural_run']['execution']['command'][0]=receipt['command'][-1]
                checkpoint=route['checkpoint_receipt'];checkpoint['build']['command'][-1]=str(binary)
                checkpoint['build']['binaries']['checkpoint-driver']=dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),bytes=binary.stat().st_size)
                checkpoint['execution']['command'][0]=str(binary)
            trace=root/'trace.json';trace.write_text(record['trace_document'])
            qualification=root/'qualification.json';qualification.write_text(json.dumps(record['qualification']))
            def failed(command,log,timeout):
                log.write_text('partial Fast native output\n')
                raise NativeCommandError(9,log)
            with mock.patch('llm_mojo.benchmarks.model_profile.environment_tool',return_value='mojo'), \
                 mock.patch('llm_mojo.benchmarks.model_profile.verify_build',return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions',return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute',side_effect=failed):
                with self.assertRaisesRegex(RuntimeError,'command failed'):
                    fast_collect(root/'build',trace,root/'failed',qualification,64,'reserved',blocks=4)
            self.assertEqual((root/'failed'/'block-0-reference.log').read_text(),'partial Fast native output\n')
            execution=json.loads((root/'failed'/'block-0-reference.execution.json').read_text())
            self.assertEqual(execution['exit_code'],9);self.assertEqual(execution['command'][-1],'reference')
            self.assertFalse((root/'failed'/'engine-fast.json.gz').exists())


class EngineCollectorBuildTests(unittest.TestCase):
    def test_budget_and_fast_preflight_accept_zero_output_beyond_pool_capacity(self):
        import json, tempfile
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks import engine_budget as budget
        from llm_mojo.benchmarks import model_profile as profile
        from tests.test_engine_trace import trace_fixture

        for operation in ('budget', 'fast'):
            for admission in profile.contract.ENGINE_ADMISSION_POLICIES:
                for maximum in (0, 1):
                    with self.subTest(operation=operation, admission=admission, maximum=maximum), \
                            tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        trace = trace_fixture()
                        trace['requests'] = trace['requests'][:1]
                        trace['requests'][0].update(prompt_ids=[42]*33, max_new_tokens=maximum)
                        trace_path = root/'trace.json'
                        trace_path.write_text(json.dumps(trace))
                        qualification = root/'qualification.json'
                        qualification.write_text('{}')
                        # Stop at the first boundary after preflight, without
                        # compiling, qualifying or executing a model.
                        target, name = (profile, 'conditions') if operation == 'budget' else (budget, '_fast_qualification')
                        with mock.patch.object(profile, 'verify_engine_build', return_value=scheduling_fixture()['build']), \
                                mock.patch.object(target, name, side_effect=RuntimeError('after preflight')) as collect, \
                                mock.patch.object(profile, 'execute') as execute:
                            expected = RuntimeError if maximum == 0 else ValueError
                            message = 'after preflight' if maximum == 0 else 'cannot fit alone'
                            with self.assertRaisesRegex(expected, message):
                                if operation == 'budget':
                                    budget.scheduling_collect(root/'build', trace_path, root/'output',
                                                              'calibration', admission, blocks=1)
                                else:
                                    budget.fast_collect(root/'build', trace_path, root/'output',
                                                        qualification, 64, admission, blocks=1)
                            self.assertEqual(collect.call_count, int(maximum == 0))
                            execute.assert_not_called()

    def test_copied_build_cannot_execute_its_changed_origin_binary(self):
        import hashlib, json, tempfile
        from pathlib import Path
        from unittest import mock
        from llm_mojo.benchmarks import model_profile as profile
        from llm_mojo.benchmarks.engine_budget import scheduling_collect, fast_collect

        for operation, declaration in (('budget', profile.contract.ENGINE_BUDGET_DECLARATION),
                                       ('fast', profile.contract.ENGINE_FAST_DECLARATION)):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                origin, copied = root/'original-build', root/'copied-build'
                origin.mkdir()
                copied.mkdir()
                body = b'verified engine fixture'
                (origin/'engine').write_bytes(body)
                (copied/'engine').write_bytes(body)
                receipt = copy.deepcopy(scheduling_fixture()['build'])
                receipt['declaration'] = declaration
                receipt['command'] = ['mojo', 'build', '-I', 'src',
                    'src/llm_mojo/benchmarks/engine_trace.mojo', '-o', str(origin/'engine')]
                receipt['binaries'] = {'engine': dict(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))}
                (copied/'build.json').write_text(json.dumps(receipt))
                # The copy still matches the receipt; its origin no longer does.
                (origin/'engine').write_bytes(b'changed unverified origin engine')
                output = root/'output'
                with mock.patch.object(profile, 'source_identity', return_value=receipt['source']), \
                        mock.patch.object(profile, 'assets', return_value=receipt['assets']), \
                        mock.patch.object(profile, 'stable_environment', return_value=receipt['environment']), \
                        mock.patch.object(profile, 'environment_tool', return_value='mojo'), \
                        mock.patch.object(profile, 'execute') as execute:
                    self.assertEqual(profile.verify_build(copied), receipt)
                    with self.assertRaisesRegex(ValueError, 'verified live build binary'):
                        if operation == 'budget':
                            scheduling_collect(copied, root/'missing-trace', output,
                                               'calibration', 'reserved', blocks=4)
                        else:
                            fast_collect(copied, root/'missing-trace', output,
                                         root/'missing-qualification', 64, 'reserved', blocks=4)
                    execute.assert_not_called()
                    self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
