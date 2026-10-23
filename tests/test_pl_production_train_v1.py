"""Production wrapper: CPU resume journal and exact existing hook binding."""
import json

import pytest

from tools import pl_production_train_v1 as train


def test_counter_progress_is_completed_updates_minus_skips(monkeypatch):
    seen=[]
    monkeypatch.setattr(train.watchdog.PROGRESS,"completed",lambda *a:seen.append(a))
    counters=train.Counters(updates=0,skips=0)
    counters["skips"]+=1
    assert not seen
    counters["updates"]+=1
    assert seen==[("training_step",dict(update=1,skips=1))]
    resumed=train.Counters(updates=100,skips=2)
    resumed["updates"]+=1
    assert seen[-1]==("training_step",dict(update=1,skips=0))


def test_resume_identity_refuses_cross_spec_fold_and_go():
    spec=dict(spec_sha256="a"*64,member=dict(id="mil-s42-f0",fold=0,seed=42))
    go={"go":{"sha256":"b"*64}}
    identity=train.identity(spec,go)
    record=dict(identity=identity,next_epoch=2,counters=dict(updates=8,skips=0))
    assert train.validate_resume(record,identity,4,20)==2
    for changed in (dict(identity,spec_sha256="c"*64),dict(identity,go_sha256="d"*64),
                    dict(identity,member=dict(id="mil-s42-f1",fold=1,seed=42))):
        with pytest.raises(ValueError,match="identity"):
            train.validate_resume(record,changed,4,20)
    with pytest.raises(ValueError,match="counters"):
        train.validate_resume(record,identity,5,20)


def test_cpu_resume_journal_roundtrip_preserves_previous_on_replace_failure(tmp_path,monkeypatch):
    import torch
    path=tmp_path/"resume.pt"
    first=dict(next_epoch=1,tensor=torch.tensor([1.,2.]),counters=dict(updates=5,skips=0))
    train.save_resume(path,first)
    loaded=torch.load(path,map_location="cpu",weights_only=False)
    assert loaded["next_epoch"]==1
    torch.testing.assert_close(loaded["tensor"],first["tensor"])
    def fail(*args,**kwargs):
        raise OSError("sharing failure")
    monkeypatch.setattr(train,"safe_replace",fail)
    with pytest.raises(OSError):
        train.save_resume(path,dict(first,next_epoch=2))
    assert torch.load(path,map_location="cpu",weights_only=False)["next_epoch"]==1
    assert path.with_name("resume.pt.tmp").exists()


def test_fit_requires_go_before_training(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(train,"adapters",nullcontext)
    spec=dict(member=dict(kind="MIL"))
    monkeypatch.setattr(train.base,"validate",lambda _: (spec,{},{}))
    monkeypatch.setattr(train.base,"admit",lambda *a:dict(status="HOLD"))
    monkeypatch.setattr(train,"mil_fit",lambda *a,**kw:pytest.fail("training reached without GO"))
    with pytest.raises(ValueError,match="not admitted"):
        train.fit("spec","out",device="cuda",smoke=False)


def test_export_serialization_instrumentation_keeps_payload_and_restores_save(tmp_path):
    import torch
    original=torch.save
    path=tmp_path/"checkpoint.pt"
    with train.export_progress(torch):
        with path.open("xb") as stream:
            torch.save(dict(model=torch.tensor([1.,2.])),stream)
    assert torch.save is original
    torch.testing.assert_close(torch.load(path,weights_only=True)["model"],torch.tensor([1.,2.]))


def test_wrapper_reuses_frozen_scientific_hooks_and_pins_sources():
    import inspect
    source=inspect.getsource(train.mil_fit)
    assert "h.train_epoch_with(" in source and "base.export(" in source
    assert "optimizer_stack(model,steps,device,amp)" in source
    assert not any(name in source for name in ("backward(","BCEWithLogitsLoss","sched.step("))
    assert {p.name for p in train.EXTRA_SOURCES} >= {
        "pl_production_train_v1.py","pl_production_targets_v1.py","pl_production_watchdog_v1.py"}
