import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tools import h216_oai_pair_runtime_v101 as m
from tools import h216_oai_pair_runtime_v1 as old
from tools import job_runtime as jobs

torch.set_num_threads(1)


class RestartContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.p=patch.object(m,'HOME_OUT',self.base);self.p.start();self.addCleanup(self.p.stop)
        m.configure('a002')
        self.addCleanup(lambda:m.configure('a001'))

    def fixture(self,arm='rsna_control'):
        folder=self.base/'a001';ap=folder/arm;ap.mkdir(parents=True)
        reg=dict(version=m.VERSION,attempt='a001',resume=None,input_contract='a'*64)
        reg['contract_id']=m.digest(reg);m.write(folder/'REGISTRATION.json',reg)
        state=dict(arm=arm,phase='adapt',step=7,identity='b'*64,input_contract='a'*64,model={'synthetic':torch.tensor([1.,2.])})
        torch.save(state,ap/'resume.pt')
        side=dict(phase=state['phase'],step=7,identity=state['identity'],contract_id=reg['contract_id'],
                  sha256=m.sha(ap/'resume.pt'),worker_identity=dict(pid=33,create_time=3.))
        m.write(ap/'resume.json',side)
        m.write(ap/'PAUSED.json',dict(status='PAUSED_AFTER_COMMIT',contract_id=reg['contract_id'],checkpoint=side))
        m.write(folder/'PAUSED.json',dict(status='PAUSED_AFTER_COMMIT',contract_id=reg['contract_id'],arm=arm))
        if arm=='oai_native':
            cp=folder/'rsna_control';cp.mkdir()
            (cp/'model.pt').write_bytes(b'synthetic completed model');(cp/'holdout.csv').write_bytes(b'synthetic predictions')
            m.write(cp/'RESULT.json',dict(status='COMPLETE',arm='rsna_control',contract_id=reg['contract_id'],
                    holdout_rows=977,finite=True,model_sha256=m.sha(cp/'model.pt'),holdout_sha256=m.sha(cp/'holdout.csv')))
        terminal=dict(exit_code=3,wrapper_identity=dict(pid=11,create_time=1.),child_identity=dict(pid=22,create_time=2.))
        logs=self.base/'logs';(logs/'H216_OAI_PAIR_V101_A001').mkdir(parents=True)
        m.write(logs/'H216_OAI_PAIR_V101_A001/receipt.json',terminal)
        m.write(logs/'H216_OAI_PAIR_V101_A001/job.json',{})
        return folder,reg,state,logs,terminal

    def guards(self,logs,state='FAILED',live='dead'):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(jobs,'LOGS',logs))
        stack.enter_context(patch.object(jobs,'job_status',return_value=dict(state=state,receipt_kind='bound')))
        stack.enter_context(patch.object(jobs,'liveness',return_value=live))
        return stack

    def test_active_unknown_or_success_job_rejected_before_output_read(self):
        for state in ('running','unknown','stale','SUCCESS'):
            with self.guards(self.base,state),patch.object(m,'read',side_effect=AssertionError('must not inspect live outputs')):
                with self.assertRaisesRegex(ValueError,'TERMINAL_PAUSE'):m.resume_info('a001')

    def test_failure_code_or_live_owner_cannot_be_resumed(self):
        _,_,_,logs,t=self.fixture()
        with self.guards(logs,live='alive'),self.assertRaisesRegex(ValueError,'OWNER_NOT_DEAD'):m.resume_info('a001')
        t['exit_code']=1;m.write(logs/'H216_OAI_PAIR_V101_A001/receipt.json',t)
        with self.guards(logs),self.assertRaisesRegex(ValueError,'OWNER_NOT_DEAD'):m.resume_info('a001')

    def test_bound_checkpoint_roundtrip_and_wrong_arm_no_load(self):
        _,reg,state,logs,_=self.fixture()
        with self.guards(logs):info=m.resume_info('a001')
        bound=dict(input_contract=reg['input_contract'],resume=info)
        loaded=m.load_resume(bound,'rsna_control')
        self.assertTrue(torch.equal(loaded['model']['synthetic'],state['model']['synthetic']))
        with patch.object(torch,'load',side_effect=AssertionError('unrelated arm must start from base')):
            self.assertIsNone(m.load_resume(bound,'oai_native'))
            self.assertIsNone(m.load_resume(dict(resume=None),'rsna_control'))
        Path(info['checkpoint']).write_bytes(b'changed')
        with patch.object(torch,'load',side_effect=AssertionError('hash before deserialize')):
            with self.assertRaisesRegex(ValueError,'HASH_AT_LOAD'):m.load_resume(bound,'rsna_control')

    def test_resume_identity_and_sidecar_tampering_rejected(self):
        folder,reg,_,logs,_=self.fixture()
        with self.guards(logs):info=m.resume_info('a001')
        bad=copy.deepcopy(info);bad['step']=8
        with self.assertRaisesRegex(ValueError,'LOADED_CHECKPOINT_IDENTITY'):m.load_resume(dict(resume=bad,input_contract='a'*64),'rsna_control')
        side=m.read(folder/'rsna_control/resume.json');side['identity']='c'*64
        m.write(folder/'rsna_control/resume.json',side)
        with self.guards(logs),self.assertRaisesRegex(ValueError,'SIDECAR_BINDING'):m.resume_info('a001')

    def test_completed_control_is_referenced_and_change_is_rejected(self):
        folder,_,_,logs,_=self.fixture('oai_native')
        with self.guards(logs):info=m.resume_info('a001')
        self.assertEqual(list(info['completed']),['rsna_control'])
        self.assertEqual(Path(info['completed']['rsna_control']['path']),folder/'rsna_control')
        (folder/'rsna_control/holdout.csv').write_bytes(b'changed')
        with self.guards(logs),self.assertRaisesRegex(ValueError,'COMPLETED_ARM_DRIFT'):m.resume_info('a001')

    def test_attempt_path_and_owner_command_are_bound(self):
        for token in ('../a001','a000','a002/extra','A001'):
            with self.assertRaisesRegex(ValueError,'ATTEMPT_NAME'):m.configure(token)
        m.configure('a002')
        with self.assertRaisesRegex(ValueError,'SEQUENCE'):m.resume_info('a000')
        current=dict(pid=20,create_time=2.);owner=dict(pid=10,create_time=1.)
        argv=['python','gpu_job.py','--_run-job',m.TAG,'run1']
        j=dict(schema='gpu-job/v2',tag=m.TAG,phase='running',child_identity=current,**owner,
               command=m.command(),command_sha256=jobs.command_digest(m.command(),m.ROOT),cwd=str(m.ROOT),run_id='run1')
        m.verify_job(j,current,owner,argv)
        j['command']=m.command('a001')
        with self.assertRaisesRegex(ValueError,'COMMAND'):m.verify_job(j,current,owner,argv)

    def test_unchanged_cache_inference_and_predictions_functions(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
        a=functions(Path(old.__file__));b=functions(Path(m.__file__))
        for name in ('inputs','cache_receipt_check','qualified_cache','infer_values','save_predictions'):
            self.assertEqual(a[name],b[name],name)

    def test_outer_restart_skips_completed_control_and_uses_new_admission(self):
        from types import SimpleNamespace
        from tools import l1_rco_runtime_v120 as runtime
        folder,_,_,logs,_=self.fixture('oai_native')
        with self.guards(logs):info=m.resume_info('a001')
        original=(folder/'rsna_control/model.pt').read_bytes()
        reg=dict(contract_id='c'*64,resume=info);m.OUT.mkdir()
        calls=[]
        def spawn(command,**kwargs):
            calls.append(command)
            self.assertIn('oai_native',command);self.assertEqual(command[-2:],['--attempt','a002'])
            ap=m.OUT/'oai_native';ap.mkdir()
            (ap/'model.pt').write_bytes(b'synthetic new arm');(ap/'holdout.csv').write_bytes(b'synthetic new predictions')
            m.write(ap/'RESULT.json',dict(status='COMPLETE',arm='oai_native',contract_id='c'*64,
                holdout_rows=977,finite=True,model_sha256=m.sha(ap/'model.pt'),holdout_sha256=m.sha(ap/'holdout.csv')))
            return SimpleNamespace(poll=lambda:0,returncode=0)
        with patch.object(m,'require_job',return_value=123),patch.object(m,'admit_cpu',return_value=(reg,{})),\
             patch.object(runtime,'active_ml_processes',return_value=[]),\
             patch.object(runtime,'gpu_snapshot',return_value=dict(desktop_bytes=0,total_bytes=16*2**30)),\
             patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=200*2**30)),\
             patch.object(m.subprocess,'Popen',side_effect=spawn):
            m.run()
            self.assertEqual(len(calls),1)
            self.assertEqual(m.read(m.OUT/'RESULT.json')['arms']['rsna_control'],info['completed']['rsna_control'])
            self.assertEqual((folder/'rsna_control/model.pt').read_bytes(),original)
            with self.assertRaisesRegex(ValueError,'ATTEMPT_ALREADY_ADMITTED'):m.run()
            self.assertEqual(len(calls),1)

    def test_missing_cache_cannot_register_or_read_weights(self):
        m.configure('a001')
        with patch.object(m,'inputs',return_value={'contract_id':'a'*64}),patch.object(m,'qualified_cache',side_effect=FileNotFoundError),\
             patch.object(m,'source_paths',return_value=[]),patch.object(torch,'load',side_effect=AssertionError('weight load')):
            self.assertEqual(m.prepare()['status'],'BLOCKED')
            with self.assertRaises(FileNotFoundError):m.register()
            self.assertFalse((m.OUT/'REGISTRATION.json').exists());self.assertFalse((m.OUT/'ADMISSION.json').exists())

    def test_disk_serialized_commit_reaches_numerical_core_without_replay(self):
        # New disk/outer-load integration, reusing the existing miniature test model.
        spec=importlib.util.spec_from_file_location('h216_tiny_resume',m.ROOT/'tests/test_h216_oai_pair_core_v101.py')
        tiny=importlib.util.module_from_spec(spec);spec.loader.exec_module(tiny)
        from tools import h216_oai_pair_core_v101 as core
        torch.manual_seed(52);initial=copy.deepcopy(tiny.Base().state_dict())
        targets=torch.linspace(.05,.95,108).reshape(9,12);saved={};last={}
        def fit(resume=None,pause=False):
            base=tiny.Base();base.load_state_dict(initial);ds=tiny.Dataset()
            with patch.object(core.v1,'crop',tiny.small_crop):
                result=core.pair_arm(base,ds,None,targets,None,tiny.CT,arm='rsna_control',device='cpu',
                    input_contract='a'*64,resume=resume,progress=lambda s:last.update(s),
                    checkpoint=lambda s:saved.update(copy.deepcopy(s)),pause_requested=lambda:pause and last.get('steps')==7)
            return copy.deepcopy(base.state_dict()),result,ds.seen
        reference=fit()
        with self.assertRaises(core.PausedAtCommit):fit(pause=True)
        p=self.base/'synthetic-resume.pt';torch.save(saved,p)
        r=dict(checkpoint=str(p),checkpoint_sha256=m.sha(p),arm='rsna_control',phase='adapt',step=7,identity=saved['identity'])
        resumed=fit(m.load_resume(dict(resume=r,input_contract='a'*64),'rsna_control'))
        self.assertTrue(tiny.equal(reference[:2],resumed[:2]))
        self.assertEqual(len(resumed[2]),2*(400-7+800))

    def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
