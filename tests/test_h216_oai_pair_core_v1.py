import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import unittest
import torch
from torch import nn
from tools import h216_oai_pair_core_v1 as m
torch.set_num_threads(1)


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.num_features=8
        self.stem=nn.Conv2d(3,8,1)
        self.stages=nn.ModuleList([nn.Conv2d(8,8,1) for _ in range(4)])
    def forward(self,x):
        x=self.stem(x)
        for stage in self.stages: x=torch.tanh(stage(x))
        return x


class SmallBase(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=SmallEncoder(); self.sp_head_norm=nn.LayerNorm(8)
        self.head=nn.Linear(8,12)
        self.register_buffer('mean',torch.zeros(1,3,1,1)); self.register_buffer('std',torch.ones(1,3,1,1))


class PairContract(unittest.TestCase):
    def test_native_missing_has_zero_gradient_and_no_negative_fill(self):
        logits=torch.zeros(2,120,requires_grad=True); grades=torch.full((2,30),-1,dtype=torch.long)
        grades[0,0]=3; grades[1,1]=0
        loss=m.native_loss(logits,grades); loss.backward()
        self.assertAlmostEqual(float(loss.detach()),__import__('math').log(4),places=6)
        grad=logits.grad.reshape(2,30,4)
        self.assertEqual(float(grad[1,0].abs().sum()),0)
        self.assertEqual(float(grad[0,1].abs().sum()),0)
        self.assertGreater(float(grad[0,0].abs().sum()),0)

    def test_native_each_task_has_equal_weight_despite_coverage(self):
        z=torch.randn(3,120); g=torch.full((3,30),-1,dtype=torch.long); g[:,0]=1; g[0,1]=2
        expected=(torch.nn.functional.cross_entropy(z.reshape(3,30,4)[:,0],g[:,0])+
                  torch.nn.functional.cross_entropy(z.reshape(3,30,4)[:1,1],g[:1,1]))/2
        self.assertTrue(torch.allclose(m.native_loss(z,g),expected))

    def test_invalid_or_empty_native_targets_fail(self):
        z=torch.zeros(2,120); g=torch.full((2,30),-1,dtype=torch.long)
        with self.assertRaisesRegex(ValueError,'NO_OBSERVED'): m.native_loss(z,g)
        g[0,0]=4
        with self.assertRaisesRegex(ValueError,'NATIVE_GRADES'): m.native_loss(z,g)

    def test_rsna_matches_parent_weighted_bce(self):
        z=torch.randn(2,12); t=torch.rand(2,12)
        reference=(torch.nn.functional.binary_cross_entropy_with_logits(z,t,reduction='none').sum(0)/2).mean()
        self.assertTrue(torch.equal(m.rsna_loss(z,t),reference))
        t[0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'RSNA_TARGET'): m.rsna_loss(z,t)

    def test_head_restored_and_adapted_encoder_retained(self):
        b=SmallBase(); head=m.original_head(b)
        with torch.no_grad():
            b.head.bias.add_(7); b.enc.stages[3].weight.add_(1)
        adapted=b.enc.stages[3].weight.detach().clone(); m.restore_head(b,head)
        self.assertTrue(torch.equal(b.head.bias,head['head.bias']))
        self.assertTrue(torch.equal(b.enc.stages[3].weight,adapted))

    def test_temporary_head_backprop_only_updates_last_stage_and_own_head(self):
        b=SmallBase(); saved={n:p.clone() for n,p in b.named_parameters()}
        a=m.TemporaryAdapter(b,120,hidden=16); m.configure_trainable(a,True); m.train_mode(a)
        opt,_,manifest=m.optimizer(a,400)
        img=torch.randint(0,255,(2,3,3,16,16),dtype=torch.uint8)
        meta=torch.zeros(2,3,6); mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
        z=a(img,meta,mask); g=torch.zeros(2,30,dtype=torch.long)
        m.native_loss(z,g).backward(); opt.step()
        self.assertTrue(torch.equal(b.enc.stem.weight,saved['enc.stem.weight']))
        self.assertFalse(torch.equal(b.enc.stages[3].weight,saved['enc.stages.3.weight']))
        self.assertTrue(torch.equal(b.head.weight,saved['head.weight']))
        self.assertFalse(b.enc.stages[0].training); self.assertTrue(b.enc.stages[3].training)
        names=[x['name'] for x in manifest['parameters']]
        self.assertFalse(any(n.startswith('enc.stages.0.') for n in names))
        self.assertTrue(any(n.startswith('cls.') for n in names))

    def test_padding_cannot_change_temporary_predictions(self):
        a=m.TemporaryAdapter(SmallBase(),12,hidden=16).eval()
        x=torch.randint(0,255,(1,2,3,8,8),dtype=torch.uint8); meta=torch.zeros(1,2,6); mask=torch.tensor([[1,0]],dtype=torch.bool)
        z=a(x,meta,mask); x[:,1]=255; meta[:,1,2:]=99
        self.assertTrue(torch.equal(z,a(x,meta,mask)))
        with self.assertRaisesRegex(ValueError,'EMPTY_STUDY'): a(x,meta,torch.zeros_like(mask))

    def test_recovery_rng_and_order_do_not_depend_on_adaptation(self):
        m.reset_rng(21601); torch.rand(37); a=m.batch_schedule(19,800,21602)
        ga=m.reset_rng(21602); randa=torch.rand(6); cropa=torch.rand(3,generator=ga)
        m.reset_rng(21601); torch.rand(911); b=m.batch_schedule(19,800,21602)
        gb=m.reset_rng(21602); randb=torch.rand(6); cropb=torch.rand(3,generator=gb)
        self.assertEqual(a,b); self.assertTrue(torch.equal(randa,randb)); self.assertTrue(torch.equal(cropa,cropb))
        self.assertEqual(len(a),800); self.assertTrue(all(i!=j for _,(i,j) in a))

    def test_recovery_reenables_original_head_and_preserves_frozen_encoder(self):
        b=SmallBase(); m.configure_trainable(b,False)
        self.assertTrue(b.head.weight.requires_grad); self.assertTrue(b.sp_head_norm.weight.requires_grad)
        self.assertFalse(b.enc.stem.weight.requires_grad); self.assertTrue(b.enc.stages[3].weight.requires_grad)

    def test_ema_only_applies_selected_parameters(self):
        b=SmallBase(); m.configure_trainable(b,False); ema=m.ema_start(b)
        stem=b.enc.stem.weight.detach().clone(); old=b.head.bias.detach().clone()
        with torch.no_grad(): b.head.bias.add_(1)
        m.ema_update(b,ema,1); m.ema_apply(b,ema)
        self.assertTrue(torch.allclose(b.head.bias,old+9/11))
        self.assertTrue(torch.equal(b.enc.stem.weight,stem))

    def test_no_cuda(self): self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__': unittest.main()
