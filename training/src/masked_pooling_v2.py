"""Mask-safe pooling used by new experiments; historical models remain immutable."""
import torch

def masked_attention_max(features: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor):
    if features.shape[:-1] != scores.shape or scores.shape != mask.shape:
        raise ValueError('Expected [...,K,D] features and [...,K] scores/mask')
    valid=mask.bool(); any_valid=valid.any(dim=-1,keepdim=True)
    clean=features.masked_fill(~valid[...,None],0)
    safe_scores=scores.float().masked_fill(~valid,torch.finfo(torch.float32).min)
    a=torch.softmax(safe_scores,dim=-1)*valid
    a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
    mean=(clean*a.to(features.dtype)[...,None]).sum(-2)
    maximum=features.masked_fill(~valid[...,None],torch.finfo(features.dtype).min).amax(-2)
    maximum=torch.where(any_valid,maximum,torch.zeros_like(maximum))
    return mean,maximum

if __name__=='__main__':
    torch.manual_seed(37)
    for dtype in (torch.float32,torch.float16):
        x=torch.randn(2,6,9,7,dtype=dtype,requires_grad=True)
        s=torch.randn(2,6,9,dtype=dtype,requires_grad=True)
        m=torch.ones(2,6,9,dtype=torch.bool); m[:,5]=False; m[0,2,:3]=False
        a,b=masked_attention_max(x,s,m)
        assert torch.equal(a[:,5],torch.zeros_like(a[:,5]))
        assert torch.equal(b[:,5],torch.zeros_like(b[:,5]))
        (a.float().sum()+b.float().sum()).backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(s.grad).all()
        assert torch.count_nonzero(x.grad[:,5])==0 and torch.count_nonzero(s.grad[:,5])==0
        xx=x.detach().clone(); xx[~m]=9999
        aa,bb=masked_attention_max(xx,s.detach(),m)
        assert torch.equal(aa,a.detach()) and torch.equal(bb,b.detach())
        z=torch.zeros_like(m); aa,bb=masked_attention_max(xx,s.detach(),z)
        assert torch.count_nonzero(aa)==0 and torch.count_nonzero(bb)==0
        print('PASS',dtype,'empty slot, empty study, invalid-value invariance, finite backward')
