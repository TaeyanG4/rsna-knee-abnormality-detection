"""v1.0.0: local Python3.12 failure-weight export packaging and fixed p25 fill.

Changelog: additive adapter; no upload, checkpoint deserialization or tuning.
"""
import argparse
import ast
import hashlib
import shutil
import sys
from pathlib import Path

from tools import k1_fold_dataset_v1 as assets
from tools import k1n_config_v1 as schema
from tools import l1_rco_k1_failure_gate_v1 as gate
from tools import l1_rco_k1_failure_v1 as producer

VERSION = "1.0.0"
MODEL = "k1_v3g_failure"
SLUG = "taeyangg4/knee-k1-v3gfailure-v1-assets"
TEMPLATE = assets.ROOT/"configs/k1n/p25-failure-dropo-v10.template.json"
need, read, sha = assets.need, assets.read, assets.sha


def verify_export(export):
    export = Path(export)
    need(export.name == "EXPORT.json", "EXPORT.json required")
    root = export.parent
    info = read(export)
    expected = {"fit/models/E_prior_s0_f0.pt", "fit/holdout.csv", "fit/result.json",
                "fit/fit-complete.json", "fit/runtime.json", "registration.json", "prepare.json"}
    need(info.get("version") == "1.0.0" and info.get("fold") == 0
         and set(info["sha256"]) == expected, "failure export inventory")
    sums = "".join(h+"  "+n+"\n" for n,h in sorted(info["sha256"].items()))
    need(assets.member(root,"SHA256SUMS").read_text() == sums, "SHA256SUMS mismatch")
    for name,pin in info["sha256"].items():
        need(schema.valid_hash(pin) and sha(assets.member(root,name)) == pin, "export hash mismatch: "+name)
    reg, result, complete, prep = (read(root/n) for n in
        ("registration.json","fit/result.json","fit/fit-complete.json","prepare.json"))
    need(reg.get("variant") == "failure_weights" and not reg["test_only"], "production failure variant required")
    need(all(r["contract_id"] == info["contract_id"] for r in (reg,result,complete,prep)), "receipt contract mismatch")
    for r in (result,complete):
        need(r["status"] == "COMPLETE" and r["model_fold"] == 0
             and r["model_sha256"] == info["sha256"]["fit/models/E_prior_s0_f0.pt"], "complete model required")
    need(complete["epochs"] == 10 and complete["optimizer_steps"] == 15950
         and complete["train_studies"] == 3190, "exact completed seed0 recipe required")
    need(result["cpu_test"] is False and result["clean_oof"] is False
         and result["exposure"] == "EXPOSED_EXPLORATORY", "production exposure metadata required")
    screen = gate.evaluate(root)  # authenticates host-bound registration, targets, argv and 977 rows
    need(screen["status"] == "SCREEN_PASS_PENDING_REVIEW", "preregistered replacement screen HOLD")
    serving = assets.SERVING.read_bytes()
    need(sha(assets.SERVING) == schema.TRAINER_SHA, "frozen serving source changed")
    training = producer.derive_trainer(serving.decode()).encode()
    need(hashlib.sha256(training).hexdigest() == prep["trainer_sha256"], "training source changed")
    for name in ("Net","OFLEncoder"):
        def definition(source):
            nodes = [n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name == name]
            need(len(nodes) == 1, "architecture definition missing")
            return ast.dump(nodes[0],include_attributes=False)
        need(definition(serving) == definition(training), "serving architecture drift")
    return info, reg, screen


def package(export, output):
    info, reg, screen = verify_export(export)
    root, output = Path(export).parent, Path(output)
    assets.member(output,"dataset-metadata.json")
    need(not output.exists(), "existing output; preserve it and inspect")
    files = {"weights/E_prior_s0_f0.pt":info["sha256"]["fit/models/E_prior_s0_f0.pt"],
             "source/compact_train.py":schema.TRAINER_SHA}
    manifest = dict(schema="k1-v3g-failure-assets/v1",version=VERSION,files=files,
        model_fold=0,model_seed=0,variant="failure_weights",clean_oof=False,exposure="EXPOSED_EXPLORATORY",
        training_contract=reg["contract_id"],loss_weights=reg["loss_weights"],
        export_sha256=sha(export),export_files=info["sha256"],screen=screen,
        independent_result_review="REQUIRED BEFORE RELEASE",independent_efficacy_evidence=False)
    output.mkdir(parents=True)
    for name,src in (("weights/E_prior_s0_f0.pt",root/"fit/models/E_prior_s0_f0.pt"),
                     ("source/compact_train.py",assets.SERVING)):
        dst=assets.member(output,name);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst);need(sha(dst)==files[name],"copy hash mismatch")
    (output/"asset-manifest.json").write_bytes(assets.encoded(manifest))
    (output/"dataset-metadata.json").write_bytes(assets.encoded(dict(id=SLUG,title=SLUG.split('/')[1],
        isPrivate=True,licenses=[dict(name="other")],
        description="Private failure-weight fold0 checkpoint/source. EXPOSED_EXPLORATORY; no independent gain claim.")))
    return verify_dataset(output)


def verify_dataset(directory):
    model = assets.verify_dataset(directory,MODEL)
    m=read(Path(directory)/"asset-manifest.json")
    need(model["dataset_slug"]==SLUG and m["variant"]=="failure_weights"
         and m["model_fold"]==m["model_seed"]==0
         and m["screen"]["status"]=="SCREEN_PASS_PENDING_REVIEW", "failure package identity")
    return model


def fill(directory, output):
    model=verify_dataset(directory)
    config=read(TEMPLATE)
    config["models"][1]=model
    config=schema.validate(config)
    output=Path(output);need(not output.exists(),"config destination exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(assets.encoded(config))
    return dict(status="FILLED_LOCAL_ONLY_PENDING_INDEPENDENT_REVIEW",path=str(output),sha256=sha(output))


def main():
    need(sys.version_info[:2]==(3,12),"local Python3.12 required")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["verify-export","package","verify-dataset","fill"])
    p.add_argument("--export",type=Path);p.add_argument("--dataset",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.action=="verify-export":
        verify_export(a.export);result=dict(status="PASS_EXPORT")
    elif a.action=="package":result=package(a.export,a.output)
    elif a.action=="verify-dataset":result=verify_dataset(a.dataset)
    else:result=fill(a.dataset,a.output)
    import json
    print(json.dumps(result))


if __name__=="__main__":main()
