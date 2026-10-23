"""Synthetic CPU tests: no subprocess training or GPU access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('hc05_run',ROOT/'tools/unique_study_run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FakeProcess:
    pid=123456
    def __init__(self, action, timeout=False):self.action=action;self.timeout=timeout
    def wait(self,timeout=None):
        if self.timeout and timeout is not None:raise subprocess.TimeoutExpired('synthetic',timeout)
        if self.action:self.action();self.action=None
        return 0

class RunGuardTests(unittest.TestCase):
    def fixture(self,root):
        run=root/'protocol';run.mkdir()
        for name in ['runner.py','review.json','parity.json']:(run/name).write_text('{}')
        (run/'runner_commands.json').write_text(json.dumps({'42':['never-executed']}))
        p={'status':'READY_REVIEWED','review_required':False,'runs':{'42':{'tag':'synthetic','expected_attempts':3448,'expected_successful_updates':3448}}}
        for name,key in [('runner.py','runner'),('runner_commands.json','commands')]:p[key+'_sha256']=m.sha(run/name)
        p['outer_wrapper_sha256']=m.sha(m.__file__)
        for name in ['review','parity']:p[name+'_path']='protocol/'+name+'.json';p[name+'_sha256']=m.sha(run/(name+'.json'))
        path=run/'manifest.json';path.write_text(json.dumps(p));return path
    def outputs(self,root,skips=0,attempts=3448,omit=None):
        out=root/'artifacts/experiments/synthetic';out.mkdir(parents=True)
        (out/'summary.json').write_text(json.dumps({'optimizer_updates':attempts,'skipped_updates':skips}))
        (out/'runner-complete.json').write_text(json.dumps({'successful_updates':attempts-skips,'expected_successful_updates':3448}))
        for name in ['provenance.json','history.json','model_best.pt','oof_best.csv','oof_k36.csv']:(out/name).write_text('synthetic')
        names=['summary.json','provenance.json','history.json','model_best.pt','oof_best.csv','oof_k36.csv']
        (out/'output-inventory.json').write_text(json.dumps({n:{'sha256':m.sha(out/n),'bytes':(out/n).stat().st_size} for n in names if n!=omit}))
        return out
    def invoke(self,root,path,process):
        with patch.object(m,'ROOT',root),patch.object(sys,'argv',['test','--protocol',str(path),'--seed','42']),patch.object(m.subprocess,'Popen',return_value=process):
            return m.main()
    def test_validated_success_requires_output_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=self.fixture(root)
            self.assertEqual(self.invoke(root,path,FakeProcess(lambda:self.outputs(root))),0)
            self.assertTrue((root/'artifacts/experiments/synthetic/VALIDATED_SUCCESS.json').exists())
            self.assertEqual(json.loads(path.with_name('execution_budget.json').read_text())['completed_seeds'],[42])
    def test_success_exit_with_wrong_update_count_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=self.fixture(root)
            self.assertEqual(self.invoke(root,path,FakeProcess(lambda:self.outputs(root,1))),1)
            self.assertFalse((root/'artifacts/experiments/synthetic/VALIDATED_SUCCESS.json').exists())
            self.assertTrue(json.loads(path.with_name('execution_budget.json').read_text())['failed'])
    def test_extra_attempt_with_equal_success_count_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=self.fixture(root)
            self.assertEqual(self.invoke(root,path,FakeProcess(lambda:self.outputs(root,skips=1,attempts=3449))),1)
    def test_incomplete_inventory_is_invalid_despite_zero_exit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=self.fixture(root)
            self.assertEqual(self.invoke(root,path,FakeProcess(lambda:self.outputs(root,omit='model_best.pt'))),1)
    def test_timeout_kills_only_own_process_tree_and_records_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=self.fixture(root)
            with patch.object(m.subprocess,'run') as stop:
                self.assertEqual(self.invoke(root,path,FakeProcess(None,True)),1)
                stop.assert_called_once()
                self.assertEqual(stop.call_args[0][0],['taskkill.exe','/PID','123456','/T','/F'])
            record=json.loads((root/'artifacts/experiments/synthetic/INVALID.json').read_text())
            self.assertEqual(record['failure'],'aggregate hard timeout')
    def test_changed_code_and_unreviewed_protocol_never_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=self.fixture(root);p=json.loads(path.read_text());p['status']='DRAFT';path.write_text(json.dumps(p))
            with patch.object(m.subprocess,'Popen') as launch:
                with self.assertRaises(SystemExit):self.invoke(root,path,None)
                launch.assert_not_called()
            p['status']='READY_REVIEWED';p['runner_sha256']='bad';path.write_text(json.dumps(p))
            with self.assertRaises(SystemExit):self.invoke(root,path,None)

if __name__=='__main__':unittest.main()
