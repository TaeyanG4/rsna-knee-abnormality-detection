"""Create and verify the EXP182 seed-43 fixed-SWA/BN release checkpoint.

The trainer already ran update_bn and saved its result as S43/model_best.pt.
This CPU-only wrapper follows EXP152's S42v2 averaging arithmetic, then copies
the authenticated trainer BN buffers and proves exact final-state equality.
It never performs a second BN pass or initializes CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from tools import exp152_own0949_release as s42
except ImportError:
    import exp152_own0949_release as s42

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/experiments/EXP182_OWN0949_S43/protocol-v1/manifest.json"
RUN_REL = "artifacts/experiments/EXP182_OWN0949_S43/S43"
RUN = ROOT / RUN_REL
OUT = ROOT / "artifacts/experiments/EXP182_OWN0949_S43/average-v1"
NAME = "exp182-s43-e14-e11-e12-bn.pt"
EPOCHS = [14, 11, 12]
VERSION = "1.0.0"
need, sha, read_json, write_json = s42.need, s42.sha, s42.read_json, s42.write_json


def registered_inputs():
    p = read_json(PROTOCOL)
    need(p.get("schema") == "exp182-own0949-s43-v1" and
         p.get("status") == "REGISTERED_NO_TRAINING" and p.get("seed") == 43 and
         p.get("run_dirs", {}).get("local") == RUN_REL and
         p.get("recipe", {}).get("swa_1based") == EPOCHS and
         p["recipe"].get("bn_batch") == 2 and
         p["recipe"].get("seed_local") == 43 and
         len(p.get("train_ids", [])) == 4347 and
         len(set(p["train_ids"])) == 4347,
         "EXP182 S43 registration differs")
    protocol_sha = sha(PROTOCOL)
    summary = read_json(RUN / "summary.json")
    provenance = read_json(RUN / "provenance.json")
    need(summary.get("status") == "COMPLETE_TRAINING_NO_EVALUATION" and
         summary.get("epochs") == 16 and summary.get("updates") == 4352 and
         summary.get("skips") == 0 and
         isinstance(summary.get("identity"), dict) and
         all(provenance.get(k) == v for k, v in summary["identity"].items()) and
         provenance.get("protocol_sha256") == protocol_sha and
         provenance.get("seed") == 43 and
         provenance.get("run_dir") == RUN_REL and
         provenance.get("swa_epochs_1based") == EPOCHS and
         provenance.get("checkpoint_selection") == "fixed-SWA" and
         provenance.get("validation") == "none" and
         provenance.get("gold_access") == "blocked" and
         provenance.get("oai") is False,
         "S43 training identity differs")
    n = len(p["train_ids"])
    bn = summary.get("bn", {})
    batches = math.ceil(n / p["recipe"]["bn_batch"])
    expected_ids = hashlib.sha256("\n".join(sorted(p["train_ids"])).encode()).hexdigest()
    need(bn.get("function") == "torch.optim.swa_utils.update_bn" and
         bn.get("studies") == bn.get("unique_studies") == n and
         bn.get("batches") == batches and
         bn.get("num_batches_tracked") == [batches] and
         bn.get("studies_sha256") == expected_ids and
         bn.get("mode") == "train" and bn.get("gradients") == "none" and
         bn.get("bn_modules") == 25,
         "S43 trainer BN replay incomplete or cohort differs")
    final = RUN / "model_best.pt"
    need(summary.get("model_sha256") == sha(final), "S43 final checkpoint SHA differs")
    return p, summary, final, protocol_sha, batches


def checkpoint_meta(ck, protocol_sha):
    expected = dict(arch=s42.ARCH, res=384, inference_res=320,
                    lab=s42.TARGETS, contract=s42.CONTRACT, k_eval=94,
                    norm="imagenet", window_policy="legacy", oai=False,
                    checkpoint_kind="fixed-swa-bn", swa_epochs_1based=EPOCHS,
                    seed=43)
    need(all(ck.get(k) == v for k, v in expected.items()) and
         ck.get("provenance", {}).get("protocol_sha256") == protocol_sha and
         ck["provenance"].get("run_dir") == RUN_REL and
         isinstance(ck.get("model"), dict) and ck["model"],
         "S43 checkpoint metadata differs")


def bn_buffer_keys(state, expected_modules):
    train = s42.load_module("exp182_training_base", ROOT / "training/src/exp143_own0949.py")
    model = train.build_model(320)  # CPU construction only.
    model.load_state_dict(state, strict=True)
    from torch.nn.modules.batchnorm import _BatchNorm
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, _BatchNorm)]
    keys = {name + "." + key for name, m in modules for key in
            ("running_mean", "running_var", "num_batches_tracked")
            if getattr(m, key) is not None}
    need(len(modules) == expected_modules, "S43 BN module count differs")
    del model
    return keys


def create():
    import torch
    need(not OUT.exists(), "S43 average-v1 exists; create-only output")
    p, summary, final, protocol_sha, batches = registered_inputs()
    ck = s42.safe_load(final)
    checkpoint_meta(ck, protocol_sha)
    need(summary["identity"] == ck["provenance"], "S43 final provenance differs")
    states, members = [], {}
    for epoch in EPOCHS:
        path = RUN / f"epoch_{epoch:02d}.pt"
        members[str(epoch)] = sha(path)
        need(summary.get("swa_member_sha256", {}).get(str(epoch)) == members[str(epoch)],
             "S43 epoch SHA differs: " + str(epoch))
        item = s42.safe_load(path)
        need(set(item) == {"model", "epoch"} and item["epoch"] == epoch,
             "S43 epoch payload differs: " + str(epoch))
        states.append(item["model"])
    avg = s42.average_states(states)
    del states
    keys = bn_buffer_keys(ck["model"], summary["bn"]["bn_modules"])
    reconstructed = s42.merge_replayed_bn(avg, ck["model"], keys, batches)
    payload = dict(ck, model=reconstructed)
    OUT.mkdir(parents=True)
    with (OUT / NAME).open("xb") as f:
        torch.save(payload, f)
    receipt = dict(schema="exp152-average-v1", version=VERSION,
                   status="VERIFIED_FIXED_SWA_BN", checkpoint=NAME,
                   checkpoint_sha256=sha(OUT / NAME), source_final_sha256=sha(final),
                   summary_sha256=sha(RUN / "summary.json"),
                   protocol_sha256=protocol_sha, run_dir=RUN_REL,
                   epochs_1based=EPOCHS, epoch_sha256=members,
                   state_schema=s42.state_schema(reconstructed),
                   bn_buffer_keys=sorted(keys), bn=summary["bn"],
                   bn_handling="reconstructed float64 SWA + authenticated trainer replay buffers; exact final-state equality",
                   source_bindings=p["source_bindings"], oai=False)
    write_json(OUT / "average-receipt.json", receipt)
    return dict(status=receipt["status"], checkpoint=str(OUT / NAME),
                checkpoint_sha256=receipt["checkpoint_sha256"])


def verify():
    import torch
    p, summary, final, protocol_sha, batches = registered_inputs()
    r = read_json(OUT / "average-receipt.json")
    need(r.get("schema") == "exp152-average-v1" and
         r.get("status") == "VERIFIED_FIXED_SWA_BN" and
         r.get("checkpoint") == NAME and r.get("run_dir") == RUN_REL and
         r.get("epochs_1based") == EPOCHS and r.get("protocol_sha256") == protocol_sha and
         r.get("summary_sha256") == sha(RUN / "summary.json") and
         r.get("source_final_sha256") == sha(final) and
         r.get("checkpoint_sha256") == sha(OUT / NAME) and
         r.get("source_bindings") == p["source_bindings"] and r.get("bn") == summary["bn"] and
         r.get("oai") is False,
         "S43 average receipt differs")
    for epoch in EPOCHS:
        path = RUN / f"epoch_{epoch:02d}.pt"
        need(r["epoch_sha256"].get(str(epoch)) == sha(path) ==
             summary["swa_member_sha256"][str(epoch)], "S43 average member differs")
    ck = s42.safe_load(OUT / NAME)
    source = s42.safe_load(final)
    checkpoint_meta(ck, protocol_sha)
    need(s42.state_schema(ck["model"]) == r.get("state_schema") and
         set(r.get("bn_buffer_keys", [])) == bn_buffer_keys(source["model"], summary["bn"]["bn_modules"]) and
         all(torch.equal(ck["model"][k], source["model"][k]) for k in source["model"]) and
         all(not k.endswith("num_batches_tracked") or
             int(ck["model"][k].item()) == batches for k in r["bn_buffer_keys"]),
         "S43 average state differs from authenticated trainer final")
    try:
        from tools import exp169_release_v4 as v4
    except ImportError:
        import exp169_release_v4 as v4
    need(v4.s43_ready(ROOT) is not None, "EXP169 v4 S43 gate failed")
    return dict(status="PASS_S43_AVERAGE_V4_GATE", checkpoint_sha256=r["checkpoint_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    args = parser.parse_args(argv)
    result = create() if args.command == "create" else verify()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
