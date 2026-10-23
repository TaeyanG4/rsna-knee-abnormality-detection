"""CPU tests for native input, paired recipe and effective-batch semantics."""
import copy
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="2"
import numpy as np
import pytest
import torch
torch.set_num_threads(2)
from training.arms import native_resolution_v1 as n
from training.arms import mil_screen_hooks_v2 as runner
from tools.hb147_go_v1 import decide
from tools.hb147_native_audit_v1 import geometry, required_cache_side


def test_native_crop_is_direct_and_repeated_center():
    # Spatial signal distinguishes direct native detail from 224-equivalent upsampling.
    plane = np.random.default_rng(0).integers(0,256,(448,448),dtype=np.uint8)
    volume = np.broadcast_to(plane,(64,448,448))
    centers = [10]*12
    x = n.scale_windows(volume,centers,320)
    np.testing.assert_array_equal(x[1,0],plane[64:384,64:384].astype(np.float32)/255.)
    np.testing.assert_array_equal(x[:,0],x[:,1])
    np.testing.assert_array_equal(x[:,1],x[:,2])
    assert x.shape == (12,3,320,320)
    y = n.scale_windows(volume,centers,224)
    up = torch.nn.functional.interpolate(torch.from_numpy(y[1:2]),size=(320,320),mode="bilinear",align_corners=False)
    assert float(np.abs(x[1]-up[0].numpy()).mean()) > .05


@pytest.mark.parametrize("res",[224,320])
def test_padding_and_old_cache_rejected(res):
    with pytest.raises(n.h.life.ContractError):
        n.scale_windows(np.zeros((64,336,336),np.uint8),[1]*12,res)
    x=n.scale_windows(np.zeros((64,448,448),np.uint8),[-1]*12,res)
    assert not x.any()
    assert required_cache_side(res)==448


def test_native_geometry_is_not_blanket_320():
    assert geometry((512,512),.25)["inner_320"]
    assert not geometry((320,320),.5)["inner_320"]
    assert not geometry((128,128),.5)["full_140mm"]


def test_mask_difference_requires_exact_source_replay(monkeypatch):
    from tools import hb147_native_cache_v1 as cache
    from training.src import raptor_common
    frozen = np.array([0,1],np.uint8)
    raster = np.array([1,1],np.uint8)
    pixels = np.arange(8,dtype=np.uint8)
    monkeypatch.setattr(cache.raw, '_WORKER', {'spec':{'img':448}})
    def build(task):
        return 'study','train',pixels.copy(),(raster if cache.raw._WORKER['spec']['img']==448 else frozen),[]
    monkeypatch.setattr(cache.raw, 'build_study', build)
    class Corpus:
        def __init__(self,*args): pass
        def volume(self,*args): return pixels
    monkeypatch.setattr(raptor_common,'Corpus',Corpus)
    result=cache.checked_study((('train','study',[]),frozen))
    assert result[-1]==[0] and cache.raw._WORKER['spec']['img']==448
    monkeypatch.setattr(Corpus,'volume',lambda *args: pixels+1)
    with pytest.raises(AssertionError,match='pixels differ'):
        cache.checked_study((('train','study',[]),frozen))
    assert cache.raw._WORKER['spec']['img']==448


def test_accumulated_gradient_equals_effective_batch_without_dropout():
    torch.manual_seed(42)
    model=torch.nn.Linear(5,12)
    ref=copy.deepcopy(model)
    x,y=torch.randn(8,5),torch.rand(8,12)
    weights=torch.linspace(.5,3,12)
    class Arm:
        microbatch=2
        def forward(self,model,x,extra): return model(x)
    class Sched:
        n=0
        def step(self): self.n+=1
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    ropt=torch.optim.SGD(ref.parameters(),lr=.01)
    loss=torch.nn.BCEWithLogitsLoss(pos_weight=weights)(ref(x),y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ref.parameters(),3.)
    ropt.step()
    sched=Sched(); counts=dict(updates=0,skips=0)
    result=runner.train_epoch_accum(Arm(),model,[(x,y,list(range(8)))],opt,sched,
        torch.amp.GradScaler("cuda",enabled=False),weights,"cpu","none",0,counts)
    for actual,expected in zip(model.parameters(),ref.parameters()):
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    assert counts==dict(updates=1,skips=0) and sched.n==1 and result["batches"]==1


def test_native_dataset_real_pixels_and_rng():
    corpus=n.h.ROOT / n.WORK / "cache-smoke"
    if not (corpus / "manifest.json").exists(): pytest.skip("real cache smoke not built")
    ids=np.load(corpus / "ids.npy").tolist()
    zeros={u:np.zeros(12,np.float32) for u in ids}
    for res in (224,320):
        inner=n.h.MILWindows(corpus,ids,zeros,12,True,42)
        ds=n.NativeWindows(inner,res)
        x=ds[0][0]
        assert x.shape==(12,3,res,res) and torch.isfinite(x).all()
        torch.testing.assert_close(x,ds[0][0],rtol=0,atol=0)
        ds.set_epoch(1)
        assert not torch.equal(x,ds[0][0])
        centers=n.h.sampled_centers(ds,0)
        other=n.NativeWindows(n.h.MILWindows(corpus,ids,zeros,12,True,42),544-res)
        other.set_epoch(1)
        np.testing.assert_array_equal(centers,n.h.sampled_centers(other,0))
    for k in (24,36):
        ds=n.NativeWindows(n.h.MILWindows(corpus,ids,zeros,k,False,42),320)
        assert ds[0][0].shape==(k,3,320,320)


def test_gate_requires_same_three_rulers_and_image_reads():
    d=np.full((2,4,12),.003)
    args=dict(all_lenses_complete=True,original_harm_guard_pass=True,image_reads_eligible=True)
    assert decide(d,[0,0],**args)["status"]=="GO_CONFIRM_OTHER_FOLD"
    assert decide(d,[0,0])["status"]=="HOLD"
    assert decide(d,[-.0001,0],**args)["status"]=="NO_GO"
    d[:,0,0]=-.02
    assert decide(d,[0,0],**args)["status"]=="NO_GO"
    d[:]=.003; d[:,0,:]=0; d[:,1,:]=0
    assert decide(d,[0,0],**args)["same_rulers_passing"]==2
    d[0,0,0]=np.nan
    assert decide(d,[0,0],**args)["status"]=="HOLD"


def test_runtime_gate_counts_all_four_fits_and_rejects_missing_memory():
    from tools.hb147_build_v1 import project_runtime
    row=dict(engineering_targets=True,results=dict(optimizer_steps=2),step_seconds=[.1,.2],val=16,
             peak_reserved_bytes=8*(1<<30),validation_seconds={'k24':.1,'k36':.1,'tta':.1},elapsed_seconds=1.)
    result=project_runtime([row]*4)
    assert result['status']=='RUNTIME_SCENARIO_PASS'
    assert result['train_hours']==pytest.approx(4*8620*.2/3600)
    slow={**row,'step_seconds':[2,2]}
    assert project_runtime([slow]*4)['status']=='NO_GO'
    with pytest.raises(n.h.life.ContractError):
        project_runtime([{**row,'peak_reserved_bytes':12*(1<<30)}]*4)


def test_final_spec_pairs_and_reject_changed_epochs():
    from tools.hb147_build_v1 import spec_for
    if not (n.h.ROOT/n.WORK/"configs/native320-s42.json").exists():
        pytest.skip("full native cache/config registration not complete")
    for seed in (42,43):
        a,b=spec_for(224,seed),spec_for(320,seed)
        n.ARM.validate_recipe(a);n.ARM.validate_recipe(b)
        ca,cb=copy.deepcopy(a.document["config"]),copy.deepcopy(b.document["config"])
        ca["windows"]["res"]=cb["windows"]["res"]
        assert ca==cb and a.document["inputs"]["data"]==b.document["inputs"]["data"]
        bad=copy.deepcopy(b.document);bad["config"]["updates"]["epochs"]=21
        with pytest.raises(n.h.life.ContractError):
            n.ARM.validate_recipe(n.h.life.ArmSpec.from_document(bad))
