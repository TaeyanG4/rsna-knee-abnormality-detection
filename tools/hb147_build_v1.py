"""H-B-147 paired native-resolution lifecycle v1.0.0 (2026-10-13).

Changelog: four bound members, register/validate/prepare and engineering CPU
cards; no GPU dispatch. Full CUDA commands remain B's decision.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
from training.arms import native_resolution_v1 as native
from tools.hb122_build_v1 import same_or_new

h, life = native.h, native.h.life
ROOT, WORK = h.ROOT, h.ROOT / native.WORK
CORPUS = ROOT / native.CORPUS
VERSION = "1.0.0"


def member(resolution, seed):
    h.need(resolution in (224,320) and seed in (42,43), "unregistered member")
    return f"native{resolution}-s{seed}"


def spec_for(resolution, seed):
    return life.ArmSpec((WORK / "configs" / (member(resolution,seed) + ".json")).read_bytes())


def configure():
    manifest = h.read(CORPUS / "manifest.json")
    h.need(manifest["finished"] and manifest["studies"] == 4410, "complete native cache required")
    result = []
    for resolution in (224,320):
        for seed in (42,43):
            control = h.control_spec(seed)
            doc = control.document
            c = doc["config"]
            oldmanifest = c["roots"]["manifest"]
            c["roots"].update(corpus=native.CORPUS, manifest=native.CORPUS + "/manifest.json")
            c["windows"].update(res=resolution, native_source=native.CONTRACT)
            c["sampler"]["microbatch"] = 2
            c["updates"]["accum"] = 4
            c["model"]["memory_format"] = "channels_last"
            del doc["inputs"]["data"][oldmanifest]
            doc["inputs"]["data"][c["roots"]["manifest"]] = h.sha(CORPUS / "manifest.json")
            evidence = doc["inputs"]["evidence"]
            targethash = doc["target_environment"]["evidence"]["declaration.json"]
            target = next(p for p,d in evidence.items() if d == targethash)
            recipepath = next(p for p in evidence if p != target)
            recipe = life.ArmSpec((ROOT / recipepath).read_bytes()).document
            recipe["config"] = copy.deepcopy(c)
            recipe["inputs"]["data"] = copy.deepcopy(doc["inputs"]["data"])
            recipe["arm"] = f"HB147_NATIVE{resolution}_RECIPE_S{seed}_F0"
            recipefile = WORK / "configs" / (member(resolution,seed) + "-recipe.json")
            same_or_new(recipefile, life.ArmSpec.from_document(recipe).document)
            doc["arm"] = f"HB147_NATIVE{resolution}_S{seed}_F0"
            # control_spec returns a fresh object: configure must not contaminate its reference.
            sources = h.expected_sources(native.ARM, h.control_spec(seed))
            doc["inputs"]["sources"] = {p:h.sha(ROOT / p) for p in sorted(sources)}
            doc["inputs"]["evidence"] = {recipefile.relative_to(ROOT).as_posix():h.sha(recipefile), target:targethash}
            spec = life.ArmSpec.from_document(doc)
            native.validate_data(spec, ROOT, CORPUS, engineering=True)
            path = WORK / "configs" / (member(resolution,seed) + ".json")
            same_or_new(path, spec.document)
            result.append(dict(member=member(resolution,seed), spec_sha256=spec.sha256))
    return dict(status="FOUR_PAIRED_SPECS_REGISTERED", members=result)


def prepare(resolution, seed):
    spec = spec_for(resolution,seed)
    run = WORK / member(resolution,seed)
    life.validate_inputs(spec,ROOT)
    same_or_new(run / "registered.json",spec.document)
    if not (run / "stage").exists():
        life.prepare(spec,ROOT,run / "stage")
    life.validate_stage(spec,run / "stage")
    _,train,val,_,_ = native.validate_data(spec,run / "stage",CORPUS,engineering=True)
    return dict(status="REGISTER_VALIDATE_PREPARE_PASS", member=member(resolution,seed),train=len(train),val=len(val))


def admit(resolution,seed,evidence="cpu"):
    run=WORK / member(resolution,seed)
    name="verification-pass.json" if evidence == "pass" else "verification.json"
    return life.admission_dry_run(spec_for(resolution,seed),run / "stage",h.read(run / evidence / name),run / evidence)


def cpu_card(resolution, seed):
    prepare(resolution,seed)
    run=WORK / member(resolution,seed)
    if not (run / "cpu").exists():
        command=[sys.executable,"-B","-m","training.runners.mil_screen_hooks_v2","smoke",
            "--arm","training.arms.native_resolution_v1:ARM","--spec",
            str(WORK / "configs" / (member(resolution,seed)+".json")),"--stage",str(run / "stage"),
            "--corpus",str(CORPUS),"--device","cpu","--workers","2","--out",str(run / "cpu")]
        subprocess.run(command,cwd=ROOT,check=True)
    admission=admit(resolution,seed)
    h.need(admission["status"] == "HOLD","CPU cannot grant target admission")
    summary=h.read(run / "cpu/outputs/summary.json")
    h.need(summary["cpu_cuda_initialized"] is False and summary["results"]["optimizer_steps"] == 1,
           "CPU smoke incomplete")
    record=dict(status="CPU_CARD_PASS",member=member(resolution,seed),admission=admission,
                summary=summary,labels="surrogate targets only; metadata and label-file hashes validated")
    same_or_new(run / "cpu-card.json",record)
    return record


def project_runtime(summaries):
    """Conservative two-step scenario; not a statistical runtime guarantee."""
    h.need(len(summaries) == 4, "four member target smokes required")
    training = validation = overhead = 0.
    for summary in summaries:
        h.need(summary['engineering_targets'] and summary['results']['optimizer_steps'] == 2
               and len(summary['step_seconds']) == 2 and summary['val'] == 16,
               "two-step, sixteen-validation-study target smoke required")
        h.need(0 < summary['peak_reserved_bytes'] <= 11 * (1 << 30), "11 GiB fit not proven")
        training += 8620 * max(summary['step_seconds'])
        # Deliberately includes repeated worker startup at every projected block.
        # This is pessimistic, not a fitted scaling law.
        validation += (888/16) * sum(summary['validation_seconds'].values())
        overhead += 20 * max(0., summary['elapsed_seconds'] - sum(summary['step_seconds'])
                            - sum(summary['validation_seconds'].values()))
    panel_reserve = 1800.
    total = training + validation + overhead + panel_reserve
    return dict(status="RUNTIME_SCENARIO_PASS" if total <= 12*3600 else "NO_GO",
                train_hours=training/3600, validation_hours=validation/3600,
                checkpoint_loading_hours=overhead/3600, panel_reserve_hours=.5,
                projected_pilot_hours=total/3600, limit_hours=12,
                note="Four fits; max of two steps, pessimistic 16-study readout scaling; panel reserve unmeasured. No CI.")


def measure():
    paths = [WORK/member(r,s)/'target/outputs/summary.json' for r in (224,320) for s in (42,43)]
    if not all(p.exists() for p in paths):
        return dict(status="HOLD",reason="four successful target smoke summaries required")
    result=project_runtime([h.read(p) for p in paths])
    same_or_new(WORK/'runtime-gate.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("configure","prepare","cpu-card","admit","measure"))
    ap.add_argument("--resolution",type=int,choices=(224,320),default=320)
    ap.add_argument("--seed",type=int,choices=(42,43),default=42)
    ap.add_argument("--evidence",choices=("cpu","pass"),default="cpu")
    a=ap.parse_args()
    if a.command == "configure":
        result=configure()
    elif a.command == "measure":
        result=measure()
    elif a.command == "admit":
        result=admit(a.resolution,a.seed,a.evidence)
    else:
        result=(prepare if a.command == "prepare" else cpu_card)(a.resolution,a.seed)
    print(life.canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
