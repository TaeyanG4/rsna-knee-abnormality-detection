import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tools import h216_oai_pair_runtime_v1 as m
from tools.job_runtime import command_digest
torch.set_num_threads(1)


class DS:
    def __len__(self):return 3
    def __getitem__(self,i):
        x=torch.arange(384,dtype=torch.int64).remainder(256).byte()[None,None,None,:].expand(1,3,384,384).clone()+i
        return x,torch.zeros(1,6),i


class CT:
    @staticmethod
    def collate(rows):
        return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows]),torch.ones(len(rows),1,dtype=torch.bool),torch.tensor([r[2] for r in rows])


class Model(torch.nn.Module):
    def forward(self,img,meta,mask,chunk):
        assert img.shape==(1,1,3,352,352) and chunk==32
        return img.float().mean((1,2,3,4))[:,None].expand(-1,12)/255


class OuterContract(unittest.TestCase):
    def cache_fixture(self):
        ids=[f'synthetic-{i}' for i in range(800)]
        reg=dict(contract_id='a'*64,schedules=dict(oai_required_ids=ids))
        index=dict(order=ids,blocks={s:[dict(views=[[0,i]],n_slices=1)] for i,s in enumerate(ids)},
                   done={s:[i,1,1] for i,s in enumerate(ids)},errors={},cursor=800)
        files={n:{} for n in ('px.u8.npy','meta.f32.npy','index.json')}
        r=dict(status='QUALIFIED_ACTUAL_V4_CACHE',input_contract='a'*64,ids=ids,actual_anchor=True,
               loader_parity='PASS',views='v4',size=384,geometry_errors=0,model_errors=0,out_of_field_count=0,
               anchor_model_studies=790,explicit_slab_prior_studies=10,hashes={n:'a'*64 for n in files},files=files)
        return r,index,reg

    def test_cache_incomplete_fallback_and_missing_study_rejected(self):
        r,idx,b=self.cache_fixture();m.cache_receipt_check(r,idx,b)
        for key,value in [('actual_anchor',False),('out_of_field_count',1),('model_errors',1),
                          ('status','V4_PILOT_PREPARED'),('loader_parity','HOLD'),('explicit_slab_prior_studies',9)]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):m.cache_receipt_check(bad,idx,b)
        bad=copy.deepcopy(idx);del bad['blocks'][b['schedules']['oai_required_ids'][0]]
        with self.assertRaisesRegex(ValueError,'INDEX'):m.cache_receipt_check(r,bad,b)

    def test_cache_overlapping_bounds_rejected(self):
        r,idx,b=self.cache_fixture();idx['done'][b['schedules']['oai_required_ids'][1]][0]=0
        with self.assertRaisesRegex(ValueError,'PARTITION'):m.cache_receipt_check(r,idx,b)

    def test_process_pid_reuse_command_or_owner_mismatch_rejected(self):
        current=dict(pid=20,create_time=2.);owner=dict(pid=10,create_time=1.)
        argv=['python','gpu_job.py','--_run-job',m.TAG,'run1']
        job=dict(schema='gpu-job/v2',tag=m.TAG,phase='running',child_identity=current,**owner,
                 command=m.command(),command_sha256=command_digest(m.command(),m.ROOT),cwd=str(m.ROOT),run_id='run1')
        m.verify_job(job,current,owner,argv)
        for key,value in [('create_time',5.),('command',['wrong']),('phase','success')]:
            bad=dict(job);bad[key]=value
            with self.assertRaises(ValueError):m.verify_job(bad,current,owner,argv)

    def test_exact_center_crop_chunk_and_ordered_predictions(self):
        events=[];ds=DS();v=m.infer_values(Model(),ds,CT,'cpu',events.append)
        expected=np.array([float(torch.sigmoid(ds[i][0][...,16:368,16:368].float().mean()/255)) for i in range(3)])
        self.assertTrue(np.array_equal(v,np.repeat(expected[:,None].astype(np.float32),12,axis=1)))
        self.assertEqual([e['eval_batches'] for e in events],[1,2,3])

    def test_nonfinite_and_reordered_inference_fail(self):
        class Bad(Model):
            def forward(self,*a,**kw):return torch.full((1,12),float('nan'))
        with self.assertRaisesRegex(ValueError,'LOGITS'):m.infer_values(Bad(),DS(),CT,'cpu',lambda s:None)
        class Wrong(DS):
            def __getitem__(self,i):x,meta,_=super().__getitem__(i);return x,meta,i+1
        with self.assertRaisesRegex(ValueError,'ORDER'):m.infer_values(Model(),Wrong(),CT,'cpu',lambda s:None)

    def test_missing_cache_preparation_does_not_admit_or_register(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            with patch.object(m,'OUT',Path(tmp)),patch.object(m,'inputs',return_value={'contract_id':'a'*64}),\
                 patch.object(m,'qualified_cache',side_effect=FileNotFoundError),patch.object(m,'source_paths',return_value=[]):
                r=m.prepare();self.assertEqual(r['status'],'BLOCKED');self.assertFalse(r['training_ready'])
                self.assertFalse((Path(tmp)/'ADMISSION.json').exists())
                with self.assertRaises(FileNotFoundError):m.register()
                self.assertFalse((Path(tmp)/'REGISTRATION.json').exists())

    def test_csv_roundtrip_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'.local/scratch/slot-b') as tmp:
            path=Path(tmp)/'pred.csv';values=np.linspace(.1,.9,36,dtype=np.float32).reshape(3,12)
            m.save_predictions(path,['synthetic-a','synthetic-b','synthetic-c'],values)
            old=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'EXIST'):m.save_predictions(path,['synthetic-a','synthetic-b','synthetic-c'],values)
            self.assertEqual(path.read_bytes(),old)
            with path.open(newline='',encoding='utf-8') as f:rows=list(__import__('csv').DictReader(f))
            self.assertTrue(np.array_equal(np.array([[float(r[c]) for c in m.LABELS] for r in rows],np.float32),values))

    def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
