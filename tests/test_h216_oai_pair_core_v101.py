import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import io
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from tools import h216_oai_pair_core_v1 as old
from tools import h216_oai_pair_core_v101 as m
torch.set_num_threads(1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.num_features=4
        self.stem=nn.Conv2d(3,4,1); self.stages=nn.ModuleList([nn.Conv2d(4,4,1) for _ in range(4)])
    def forward(self,x):
        x=self.stem(x)
        for s in self.stages:x=torch.tanh(s(x))
        return x


class Base(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=Encoder(); self.sp_head_norm=nn.LayerNorm(4)
        self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(4,12))
        self.register_buffer('mean',torch.zeros(1,3,1,1)); self.register_buffer('std',torch.ones(1,3,1,1))
    def forward(self,img,meta,mask):
        f=self.enc(img[:,0].float()/255).mean((-1,-2))
        return self.head(self.sp_head_norm(f))


class Dataset:
    mode='train';n_win=64;train_sampling='cover';cover_stride=3;slice_jitter=.25;phys_mm=3.5;slices=3
    seed=0;epoch=0
    def __init__(self):self.seen=[]
    def __len__(self):return 9
    def __getitem__(self,i):
        self.seen.append((self.epoch,i))
        rng=np.random.default_rng(self.seed+1000003*self.epoch+i)
        x=torch.from_numpy(rng.integers(0,255,(1,3,12,12),dtype=np.uint8))
        return x,torch.zeros(1,6),i


class CT:
    @staticmethod
    def collate(rows):
        return torch.stack([r[0] for r in rows]),torch.stack([r[1] for r in rows]),torch.ones(len(rows),1,dtype=torch.bool),torch.tensor([r[2] for r in rows])


def small_crop(img,gen):
    y=int(torch.randint(0,5,(1,),generator=gen));x=int(torch.randint(0,5,(1,),generator=gen))
    return img[...,y:y+8,x:x+8]


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class ResumeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(52); cls.initial=copy.deepcopy(Base().state_dict())
        cls.rt=torch.linspace(.05,.95,108).reshape(9,12)
        cls.ot=torch.arange(270).reshape(9,30)%4;cls.ot[:,4]=-1
        cls.reference={};cls.snapshots={}
        for arm in ('rsna_control','oai_native'):
            out,snaps,_,_=cls.execute(arm)
            cls.reference[arm]=out;cls.snapshots[arm]=snaps

    @classmethod
    def execute(cls,arm,resume=None,stop=None,legacy=False):
        base=Base();base.load_state_dict(cls.initial);rs=Dataset();oa=Dataset();snaps={};last={}
        def progress(s):last.update(s)
        def save(s):snaps[(s['phase'],s['step'])]=copy.deepcopy(s)
        kw=dict(arm=arm,device='cpu',progress=progress,checkpoint=save)
        if not legacy:kw.update(input_contract='a'*64,resume=resume,pause_requested=lambda:(last.get('phase'),last.get('steps'))==stop)
        with patch.object(old,'crop',small_crop):
            result=(old if legacy else m).pair_arm(base,rs,oa,cls.rt,cls.ot,CT,**kw)
        return (copy.deepcopy(base.state_dict()),result),snaps,rs.seen,oa.seen

    def test_recipe_matches_original_both_arms(self):
        for arm in self.reference:
            result,snaps,_,_=self.execute(arm,legacy=True)
            self.assertTrue(equal(result[0],self.reference[arm][0]))
            for k,s in snaps.items():
                new=self.snapshots[arm][k]
                for field in ('model','optimizer','scheduler','ema','losses','cpu_rng','crop_rng'):
                    self.assertTrue(equal(s[field],new[field]),(arm,k,field))

    def test_resume_after_update_and_at_phase_boundary_both_arms(self):
        for arm in self.reference:
            for boundary in (('adapt',100),('adapt',400),('recovery',100),('recovery',800)):
                # Serialize only synthetic miniature state in memory, never a live or disease checkpoint.
                stream=io.BytesIO();torch.save(self.snapshots[arm][boundary],stream);stream.seek(0)
                resume=torch.load(stream,weights_only=False,map_location='cpu')
                random.seed(18);np.random.seed(19);torch.manual_seed(20)
                out,snaps,rs,oa=self.execute(arm,resume)
                self.assertTrue(equal(out,self.reference[arm]),(arm,boundary))
                if boundary[0]=='recovery':self.assertEqual(len(rs),2*(800-boundary[1]));self.assertEqual(len(oa),0)
                for k,s in snaps.items():
                    self.assertTrue(equal(s,self.snapshots[arm][k]),(arm,boundary,k))

    def test_explicit_pause_persists_committed_nonperiodic_step(self):
        saved={};last={};base=Base();base.load_state_dict(self.initial)
        def save(s):saved.update(copy.deepcopy(s))
        with patch.object(old,'crop',small_crop),self.assertRaises(m.PausedAtCommit):
            m.pair_arm(base,Dataset(),Dataset(),self.rt,self.ot,CT,arm='rsna_control',device='cpu',
                input_contract='a'*64,progress=lambda s:last.update(s),checkpoint=save,
                pause_requested=lambda:last['steps']==7)
        self.assertEqual(saved['step'],7);self.assertEqual(len(saved['losses']),7)
        out,_,_,_=self.execute('rsna_control',saved)
        self.assertTrue(equal(out,self.reference['rsna_control']))

    def test_wrong_identity_schedule_and_commit_rejected(self):
        for field,value,code in [('input_contract','b'*64,'IDENTITY'),('step',99,'COMMIT'),
                                 ('loader_seed',123,'SCHEDULE'),('arm','oai_native','ARM_PHASE')]:
            ck=copy.deepcopy(self.snapshots['rsna_control'][('adapt',100)]);ck[field]=value
            with self.assertRaisesRegex(ValueError,code):self.execute('rsna_control',ck)

    def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
