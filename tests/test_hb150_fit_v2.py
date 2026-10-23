"""CPU tests for H-B-150 safe progress, fixed recipe and full-state resume."""
import json
from pathlib import Path

import pytest

from tools import hb150_fit_v2 as f


@pytest.mark.parametrize('passed,code', [(True,0),(False,1)])
def test_cli_export_failure_stops_pair(monkeypatch,passed,code):
    value=dict(passed=passed,arm='global',seed=42,device='cpu',engineering_only=True,steps=1,seconds=.1)
    monkeypatch.setattr(f,'fit',lambda *args,**kwargs:value)
    assert f.main(['smoke','--config','synthetic.json','--out','synthetic'])==code


def test_progress_phase_switch_no_fake_commit(tmp_path):
    p=f.Progress(tmp_path/"heartbeat.json","test")
    p.completed("training_step",{"update":1})
    first=json.loads(p.path.read_text())
    p.phase_start("validation")
    entered=json.loads(p.path.read_text())
    assert entered["sequence"] == first["sequence"]
    assert entered["completed_at"] == first["completed_at"]
    assert entered["detail"]["cumulative_units"] == 0
    p.completed("validation",{"batch":1})
    p.completed("validation",{"batch":2})
    p.completed("training_step",{"update":2})
    assert json.loads(p.path.read_text())["detail"]["cumulative_units"] == 2


def test_heartbeat_failure_is_nonfatal(tmp_path,monkeypatch):
    seen=[]
    monkeypatch.setattr(f,"atomic_json",lambda *a,**kw: seen.append(kw))
    f.Progress(tmp_path/"beat","token").completed("validation")
    assert seen == [{"critical":False}]


def test_full_state_checkpoint_restores_optimizer_scheduler_rng_and_counters(tmp_path):
    h=f.h
    h.seed_all(42)
    model=f.TinyModel()
    opt,sched,scaler=h.optimizer_stack(model,4,"cpu","none")
    gen=h.torch.Generator().manual_seed(42)
    identity={"config":"test","device":"cpu","engineering_only":True}
    x=h.torch.ones(2,12,3,8,8)
    y=h.torch.full((2,12),.5)
    loader=[(x,y,["one","two"])]
    old=h.PROGRESS
    h.PROGRESS=f.Progress()
    counters={"updates":0,"skips":0}
    try:
        result=h.train_epoch_accum(f.native.NativeArm(),model,loader,opt,sched,scaler,
            h.torch.ones(12),"cpu","none",0,counters)
        path=tmp_path/"epoch.pt"
        f.save_epoch(path,epoch=0,model=model,opt=opt,sched=sched,scaler=scaler,generator=gen,
            history=[result],counters=counters,identity=identity,device="cpu",progress=h.PROGRESS)
        expected={key:value.clone() for key,value in model.state_dict().items()}
        target=f.TinyModel()
        opt2,sched2,scaler2=h.optimizer_stack(target,4,"cpu","none")
        epoch,history,count=h.restore_epoch(path,model=target,opt=opt2,sched=sched2,scaler=scaler2,
            generator=gen,identity=identity,device="cpu")
        assert epoch == 1 and count == counters and history == [result]
        assert sched2.state_dict() == sched.state_dict()
        assert len(opt2.state) == len(opt.state)
        for key,value in target.state_dict().items():
            h.torch.testing.assert_close(value,expected[key])
        with pytest.raises(Exception,match="resume receipt differs"):
            h.restore_epoch(path,model=target,opt=opt2,sched=sched2,scaler=scaler2,
                generator=gen,identity={**identity,"config":"other"},device="cpu")
    finally:
        h.PROGRESS=old


def test_prediction_export_failure_is_nonfatal(tmp_path,monkeypatch):
    monkeypatch.setattr(f,"safe_replace",lambda *a,**kw:False)
    assert not f.export_prediction(tmp_path/"val.csv",["test"],f.h.np.full((1,12),.5),
        tmp_path/"absent.pt",{},k=24,tta=False,engineering=True)


def test_scientific_targets_read_only_sealed_table(tmp_path,monkeypatch):
    from tools import hb146_pl_labels_v111 as pl
    monkeypatch.setattr(f,"ROOT",tmp_path)
    monkeypatch.setattr(pl,"registration",lambda *a:pytest.fail("source traversal forbidden"))
    path=tmp_path/"synthetic.csv"
    path.write_bytes(f.h.life.prediction_csv_bytes(["one"],f.h.np.full((1,12),.5),f.h.TARGETS,id_column=f.h.UID))
    c={"label_path":"synthetic.csv","train":["one"]}
    assert len(f.scientific_targets(c)) == 1
    with pytest.raises(Exception):
        f.scientific_targets({**c,"train":["missing"]})


def test_no_scientific_cpu_or_gpu_admission_bypass(tmp_path,monkeypatch):
    monkeypatch.setattr(f,"read_config",lambda path:{})
    with pytest.raises(Exception,match="CUDA"):
        f.fit(tmp_path/"c.json",tmp_path/"out",device="cpu",smoke=False)
    from tools import hb150_build_v2 as build
    def blocked(*args):
        raise ValueError("independent review artifact required")
    monkeypatch.setattr(build,"require_fit",blocked)
    with pytest.raises(ValueError,match="independent review"):
        f.fit(tmp_path/"c.json",tmp_path/"out",device="cuda",smoke=False)
    assert not (tmp_path/"out").exists()


def test_real_native_pixel_cpu_path(tmp_path,monkeypatch):
    """Real native448 volume, train augmentation, K24/K36/flip; no label rows."""
    h=f.h
    h.torch.set_num_threads(2)
    corpus=h.common.Corpus(f.ROOT/f.native.CORPUS)
    ids=corpus.ids.tolist()[:2]
    labels={uid:h.np.full(12,.5,h.np.float32) for uid in ids}
    arm=f.native.NativeArm()
    model=f.TinyModel()
    ds=f.native.NativeWindows(h.MILWindows(f.ROOT/f.native.CORPUS,ids,labels,12,True,42),320)
    loader=h.DataLoader(ds,batch_size=2,collate_fn=arm.collate_fn)
    opt,sched,scaler=h.optimizer_stack(model,1,"cpu","none")
    monkeypatch.setattr(h,"PROGRESS",f.Progress(tmp_path/"hb.json","test"))
    counters={"updates":0,"skips":0}
    result=h.train_epoch_accum(arm,model,loader,opt,sched,scaler,h.torch.ones(12),"cpu","none",0,counters)
    assert result["batches"] == counters["updates"] == 1
    for k,tta in ((24,False),(36,False),(36,True)):
        ds=f.native.NativeWindows(h.MILWindows(f.ROOT/f.native.CORPUS,ids[:1],labels,k,False,42),320)
        loader=h.DataLoader(ds,batch_size=1,collate_fn=arm.collate_fn)
        values,ordered,batches=h.evaluate_with(arm,model,loader,"cpu","none",corpus,tta=tta)
        assert ordered == ids[:1] and values.shape == (1,12) and h.np.isfinite(values).all() and batches == 1
    assert h.torch.cuda.is_initialized() is False
