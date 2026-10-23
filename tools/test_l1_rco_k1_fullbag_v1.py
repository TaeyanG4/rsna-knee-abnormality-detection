"""D-A-052 CPU regression tests; synthetic receipts never qualify production."""
import ast
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tools import l1_rco_k1_fullbag_v1 as f
from tools import l1_rco_k1_fullbag_replay_v1 as r
from tools.l1_rco_k1_fullbag_check_v1 import TinyNet, compare_model


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("lengths", [(1, 3), (4, 2)])
def test_stochastic_replay_all_gradients_update_and_rng(amp, lengths):
    torch.set_num_threads(2)
    torch.manual_seed(52)
    mask = torch.arange(max(lengths))[None,:] < torch.tensor(lengths)[:,None]
    raw = torch.randint(0,256,(int(mask.sum()),3,16,16),dtype=torch.uint8)
    def chunks():
        for part in raw.split(2):
            # Include zero-padding like the real model.
            n = len(part)
            yield torch.cat([part,part.new_zeros((2-n,3,16,16))]), n
    result = compare_model(TinyNet().train(),chunks,torch.zeros(2,max(lengths),6),mask,amp=amp)
    assert result["status"] == "PASS"


@pytest.mark.parametrize("aug,res", [("crop",0),("v2",0),("v2",1)])
def test_streamed_augmentation_exact_frozen_pixels_and_generator(aug,res):
    source = (f.k.parent.OUT.parent/"compact_train_l1.py").read_text()
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=="augment")
    a = SimpleNamespace(crop=16,aug=aug,aug_res=res)
    namespace = dict(torch=torch,F=F,math=math,a=a)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"frozen-augment","exec"),namespace)
    raw = torch.randint(0,256,(2,4,3,20,20),dtype=torch.uint8)
    g1,g2 = [torch.Generator().manual_seed(51) for _ in range(2)]
    expected = namespace["augment"](raw,g1)
    plan = r.augmentation_plan(raw,g2,a)
    mask = torch.tensor([[True,False,True,False],[True,True,True,True]])
    locations = mask.nonzero()
    actual = []
    for loc in locations.split(2):
        actual.append(r.augment_chunk(raw[loc[:,0],loc[:,1]],loc[:,0],plan,a,torch.device("cpu")))
    assert torch.equal(expected[mask],torch.cat(actual))
    assert torch.equal(g1.get_state(),g2.get_state())


def test_reject_batchnorm_and_stateful_encoder():
    model = TinyNet()
    model.enc.add_module("bn",torch.nn.BatchNorm1d(8))
    with pytest.raises(ValueError,match="BatchNorm"):
        r.backward(model,lambda:[],torch.zeros(1,1,6),torch.ones(1,1,dtype=torch.bool),lambda z:z.sum())


def test_bce_retains_frozen_autocast_scope():
    model = TinyNet().train()
    raw = torch.zeros(2,3,16,16,dtype=torch.uint8)
    def loss(logits):
        assert torch.is_autocast_enabled("cpu")
        result = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        assert result.dtype == torch.float32
        return result
    r.backward(model, lambda: iter([(raw,2)]), torch.zeros(1,2,6),
               torch.ones(1,2,dtype=torch.bool), loss, amp=True)


def test_trainer_delta_and_recipe():
    source = (f.ROOT/f.k.s.SOURCE/"compact_train.py").read_text()
    before = f.k.parent.derive_trainer(source)
    after = f.derive_trainer(source)
    def classes(text):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
                if isinstance(n,ast.ClassDef) and n.name in ("Net","OFLEncoder")}
    assert classes(before) == classes(after)
    assert 'rng.choice(len(cand), self.n_win' not in after
    assert "fullbag candidate cap exceeded" in after
    assert after.count("loss = training_backward(") == 1
    tail = "                    if a.forward_only:"
    assert after[after.index(tail,after.index("loss = training_backward(")):] == before[
        before.index(tail,before.index("for img, meta, mask, idx in l1_state.batches():")):]
    reg = f.expected_registration()
    old = f.k.read(f.k.PARENT)
    assert reg["train"] == old["train"] and reg["holdout"] == old["holdout"]
    for key in old["recipe"]:
        if key not in ("windows","unchanged"):
            assert reg["recipe"][key] == old["recipe"][key],key
    assert (len(reg["train"]),len(reg["holdout"])) == (3190,977)
    argv = f.trainer_argv(f.k.CACHE,reg)
    assert argv[argv.index("--train-windows")+1] == "640"
    assert argv[argv.index("--eval-stride")+1] == "3"


def test_cap_exceeded_fails_before_image_allocation(monkeypatch):
    ct = f.k.import_trainer()
    obj = object.__new__(ct.Studies)
    obj.uids=["synthetic"]
    obj.px=[]
    obj.seed=1
    obj.epoch=0
    obj.blocks={"synthetic":[]}
    obj.meta=[]
    obj.mode="train"
    obj.slices=3
    obj.eval_stride=obj.cover_stride=3
    obj.phys_mm=3.5
    obj.slice_jitter=.25
    obj.train_sampling="cover"
    obj.n_win=640
    monkeypatch.setattr(ct,"window_rows",lambda *args:[(0,0)]*641)
    with pytest.raises(ValueError,match="cap exceeded"):
        obj[0]


def test_fold0_subprocess_import_preserves_environment_and_live_modules():
    code = ("import os; from tools import l1_rco_k1_fullbag_v1 as f; "
            "from tools import l1_rco_k1_folds_v1 as k; "
            "assert f.k is not k and f.k.FOLD==0 and k.FOLD==1; "
            "assert os.environ['K1_FOLD']=='0'; print('PASS')")
    result = subprocess.run([sys.executable,"-B","-c",code],cwd=f.ROOT,
                            env={**os.environ,"K1_FOLD":"0","CUDA_VISIBLE_DEVICES":""},
                            capture_output=True,text=True,check=True)
    assert result.stdout.strip()=="PASS"


@pytest.mark.parametrize("wins,status",[(2,"HOLD"),(3,"SCREEN_PASS_PENDING_REVIEW"),(4,"SCREEN_PASS_PENDING_REVIEW")])
def test_registered_three_of_four_rule(wins,status):
    names=["jev_hybrid","aquila_v1","dread","consensus"]
    assert f.decide(dict.fromkeys(names,.8),{n:.8 if i<wins else .79 for i,n in enumerate(names)})["status"]==status


def test_invalid_macro_cannot_pass():
    names=["jev_hybrid","aquila_v1","dread","consensus"]
    with pytest.raises(ValueError):
        f.decide(dict.fromkeys(names,.8),dict.fromkeys(names,float("nan")))


def test_meta_missing_blocks_before_gpu(monkeypatch,tmp_path):
    def fail(cache):
        raise ValueError("missing META_EPS")
    monkeypatch.setattr(f,"meta_binding",fail)
    monkeypatch.setattr(f,"original_admit",lambda *args:pytest.fail("must not enter GPU admission"))
    assert f.admit({},tmp_path,tmp_path/"review")["status"]=="BLOCKED"
    with pytest.raises(ValueError,match="META_EPS"):
        f.admit({},tmp_path,tmp_path/"review",False)


def test_sequence_own_consumed_admission(monkeypatch,tmp_path):
    from tools import l1_rco_k1_foldseq_v1 as seq
    monkeypatch.setattr(f.k,"OUT",tmp_path)
    (tmp_path/"admission.json").write_text("{}")
    monkeypatch.setattr(f,"predecessor_success",lambda spec:{"bound":True})
    monkeypatch.setattr(seq,"check_fresh",lambda spec:None)
    with pytest.raises(ValueError,match="consumed"):
        f.preflight({})


def test_sequence_failure_is_durable_no_retry(monkeypatch,tmp_path):
    from tools import l1_rco_k1_foldseq_v1 as seq
    spec=dict(owner=str(tmp_path/"owner"),ledger=str(tmp_path/"once.json"),
              predecessor={"run_id":"f3"},jobs=[dict(tag=f.k.TAG,argv=["never"],fold="fullbag")])
    def preflight(spec):
        if Path(spec["ledger"]).exists():
            raise ValueError("no retry")
        return {"run_id":"f3"}
    monkeypatch.setattr(f,"preflight",preflight)
    def dispatch(*args,**kwargs):
        assert seq.runtime.read_json(Path(spec["ledger"]))["state"]=="STARTED"
        raise RuntimeError("synthetic dispatch failure")
    monkeypatch.setattr(f.subprocess,"run",dispatch)
    with pytest.raises(RuntimeError):
        f.run_sequence(spec)
    assert seq.runtime.read_json(Path(spec["ledger"]))["state"]=="STOPPED"
    with pytest.raises(ValueError,match="no retry"):
        f.run_sequence(spec)


def test_direct_run_requires_bound_canonical_sequence(monkeypatch,tmp_path):
    monkeypatch.setattr(f,"WORK",tmp_path)
    f.k.write(tmp_path/"after-f3-once.json",dict(state="STARTED",spec=dict(predecessor=dict(run_id="fake"))))
    with pytest.raises(ValueError,match="canonical"):
        f.require_sequence()


@pytest.fixture
def sequence_rig(monkeypatch, tmp_path):
    from tools.test_l1_rco_k1_device_plan_v1 import terminal
    monkeypatch.setattr(f, "WORK", tmp_path/"fullbag")
    monkeypatch.setattr(f, "CANONICAL_OWNER", tmp_path/"owner")
    monkeypatch.setattr(f.k, "WORK", f.WORK)
    monkeypatch.setattr(f.k, "OUT", f.WORK/"fit")
    spec = f.plan("bound-f3")
    pred = dict(tag=spec["predecessor"]["tag"],
                argv=["gpu_job.py", "--"]+spec["predecessor"]["command"])
    directory, job = terminal(spec, pred, "bound-f3")
    f.k.write(f.WORK/"registration.json", {})
    monkeypatch.setattr(f.k, "validate", lambda reg: reg)
    monkeypatch.setattr(f, "admit", lambda *a: dict(status="CPU_ADMISSIBLE"))
    calls = []
    def dispatch(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["KNEE_WORKSPACE_ROOT"] == spec["owner"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        f.require_sequence()
        terminal(spec, spec["jobs"][0], "bound-fullbag")
    monkeypatch.setattr(f.subprocess, "run", dispatch)
    return spec, directory, job, calls


def test_fullbag_bound_success_and_no_retry(sequence_rig):
    spec, _, _, calls = sequence_rig
    result = f.run_sequence(spec)
    assert result["state"] == "SUCCESS" and len(calls) == 1
    assert result["jobs"][0]["run_id"] == "bound-fullbag"
    with pytest.raises(ValueError, match="no retry"):
        f.preflight(spec)


@pytest.mark.parametrize("field", ["run_id", "command", "cwd", "command_sha256"])
def test_fullbag_rejects_wrong_f3_binding(sequence_rig, field):
    spec, directory, job, calls = sequence_rig
    job[field] = ["wrong"] if field == "command" else "wrong"
    f.k.write(directory/"job.json", job)
    with pytest.raises(ValueError, match="binding mismatch"):
        f.preflight(spec)
    assert not calls


@pytest.mark.parametrize("failure", [None, "memory", "crop", "code", "review", "run", "predecessor"])
def test_fullshape_evidence_binding(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(f, "WORK", tmp_path)
    monkeypatch.setattr(f, "code_pins", lambda: {"code": "fresh"})
    pred = {"run_id": "f3"}
    calls = []
    def predecessor(spec):
        calls.append(spec)
        return pred
    monkeypatch.setattr(f, "predecessor_success", predecessor)
    f.k.write(tmp_path/"REVIEW-QUALIFY.json", {"verdict": "PASS"})
    memory = dict(contract_id="contract", gpu="4090", status="PASS",
                  code_pins=f.code_pins(), batch=2, windows_per_study=640,
                  crop=352, amp=True, peak_reserved=8*2**30, baseline_used=0,
                  total_bytes=24*2**30, run_id="qualify", f3_run_id="f3",
                  qualification_review_sha256=f.k.sha(tmp_path/"REVIEW-QUALIFY.json"))
    once = dict(run_id="qualify", predecessor=pred)
    if failure == "memory":
        memory["peak_reserved"] = 13*2**30
    elif failure == "crop":
        memory["crop"] = 64
    elif failure == "code":
        memory["code_pins"] = {}
    elif failure == "review":
        memory["qualification_review_sha256"] = "stale"
    elif failure == "run":
        once["run_id"] = "different"
    elif failure == "predecessor":
        once["predecessor"] = {}
    f.k.write(tmp_path/"fullshape-cuda.json", memory)
    f.k.write(tmp_path/"fullshape-once.json", once)
    review = dict(fullbag_memory_sha256=f.k.sha(tmp_path/"fullshape-cuda.json"))
    if failure:
        with pytest.raises(ValueError):
            f.fullshape_evidence({"contract_id": "contract"}, review)
    else:
        assert f.fullshape_evidence({"contract_id": "contract"}, review) == memory
        assert calls[-1]["predecessor"]["run_id"] == "qualify"
        assert calls[-1]["predecessor"]["command"] == f.fullshape_command("f3")


def test_fullbag_export_packaging(tmp_path):
    from tools import k1_fold_dataset_v1 as a
    import importlib.util
    spec = importlib.util.spec_from_file_location("fullbag_asset_fixture", f.ROOT/"tests/test_k1_fold_assets_v1.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    synthetic_export,seal,write = fixture.synthetic_export,fixture.seal,fixture.write
    export = synthetic_export(tmp_path/"export",1)
    root=export.parent
    info=a.read(export)
    old="fit/models/E_prior_s0_f1.pt"
    new="fit/models/E_prior_s0_f0.pt"
    (root/new).write_bytes((root/old).read_bytes())
    info["sha256"][new]=info["sha256"].pop(old)
    reg=a.read(root/"registration.json")
    reg.pop("contract_id")
    reg.update(model_fold=0,variant="fullbag",train=["t"+str(i) for i in range(3190)],
               holdout=["h"+str(i) for i in range(977)])
    reg["recipe"].update(model_fold=0,windows=640,encoder_microbatch=32)
    reg["contract_id"]=a.digest(reg)
    write(root/"registration.json",reg)
    info.update(fold=0,contract_id=reg["contract_id"])
    for name in ("fit/result.json","fit/fit-complete.json","prepare.json"):
        data=a.read(root/name)
        data.update(contract_id=reg["contract_id"])
        if name=="prepare.json":
            import hashlib
            data["trainer_sha256"]=hashlib.sha256(a.derive_training_source(a.SERVING.read_bytes(),0)).hexdigest()
        else:
            data.update(model_fold=0,holdout_rows=977,train_studies=3190)
        write(root/name,data)
    seal(root,info)
    result=a.package(export,tmp_path/"assets","taeyangg4/knee-k1-v3gfb-v1-assets")
    assert result["id"]=="k1_v3g_fullbag"
    assert a.verify_dataset(tmp_path/"assets")==result
    reg["variant"]="seed0"
    reg.pop("contract_id")
    reg["contract_id"]=a.digest(reg)
    write(root/"registration.json",reg)
    info["contract_id"]=reg["contract_id"]
    seal(root,info)
    with pytest.raises(ValueError,match="explicit fullbag"):
        a.verify_export(export)
