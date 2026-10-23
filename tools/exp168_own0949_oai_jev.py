"""EXP168: EXP155 v1.1 with Dread75/JEV25 RSNA labels. CPU-passive CLI."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "training/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tools import exp143_v2 as base
from tools import exp155_own0949_oai_v11 as prior
from tools import exp168_labels as label_builder

SCHEMA, VERSION = "exp168-own0949-oai-jev-v1", "1.0.0"
BASE = ROOT / "artifacts/experiments/EXP168_OWN0949_OAI_JEV"
PROTOCOL = BASE / "protocol-v1/manifest.json"
RUN_DIRS = {"local": "artifacts/experiments/EXP168_OWN0949_OAI_JEV/S42"}
LABELS = label_builder.OUT
OAI_LABELS = prior.OAI_LABELS
CORPUS, OAI_CORPUS, INIT, BASE_PROTOCOL = prior.CORPUS, prior.OAI_CORPUS, prior.INIT, prior.BASE_PROTOCOL
RECIPE = dict(prior.RECIPE, labels="l2-dread75-jev25-plus-oai-three-target-b1")
CODE = list(dict.fromkeys([*prior.CODE, "tools/exp168_labels.py",
    "tools/exp168_own0949_oai_jev.py", "training/src/exp168_own0949_oai_jev.py",
    "tools/exp168_own0949_oai_jev_release.py",
    "artifacts/experiments/EXP168_OWN0949_OAI_JEV/PROTOCOL.md"]))
sha256, json_hash, read_json = base.sha256, base.json_hash, base.read_json
save_torch_new, save_torch_atomic = base.save_torch_new, base.save_torch_atomic
projection, TARGETS = prior.projection, prior.TARGETS


def scoped_path(path, root=None):
    path, root = Path(path).resolve(), Path(root or BASE).resolve()
    if path == root or root not in path.parents:
        raise ValueError("output must be a child of EXP168")
    return path


def write_new_json(path, value):
    path = Path(path).resolve()
    if not any(r.resolve() in path.parents for r in (BASE, ROOT / ".local/exp168-probes")):
        raise ValueError("EXP168 JSON output outside experiment/probe subtree")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def registered_run(p, lane="local"):
    if lane != "local" or p["run_dirs"] != RUN_DIRS:
        raise ValueError("EXP168 requires the registered local seed42 run")
    return scoped_path(ROOT / RUN_DIRS[lane])


def execution_settings(p, world, precision, seed, checkpointing=None):
    if world != 1 or precision != "bf16" or seed != 42 or checkpointing not in (None, False):
        raise ValueError("EXP168: one local GPU, bf16, seed42, checkpointing off")
    return prior.execution_settings(p, world, precision, seed, False)


def source_bindings():
    return {name: sha256(ROOT / name) for name in CODE}


def label_provenance():
    record = read_json(label_builder.PROVENANCE)
    baseline = base.load_registration(BASE_PROTOCOL)
    if (record.get("name") != "l2_dread75_jev25_v1" or record.get("rows") != 4347
            or record.get("gold58_included") is not False or record.get("oai") is not False
            or record.get("csv_sha256") != sha256(LABELS)
            or record.get("baseline_protocol_sha256") != sha256(BASE_PROTOCOL)
            or record.get("source_sha256") != {
                str(label_builder.DREAD.relative_to(ROOT)): sha256(label_builder.DREAD),
                str(label_builder.JEV.relative_to(ROOT)): sha256(label_builder.JEV)}
            or record.get("baseline_train_ids_sha256") != __import__("hashlib").sha256(
                "\n".join(baseline["train_ids"]).encode()).hexdigest()):
        raise ValueError("EXP168 label provenance drift")
    return record


def label_table(ids):
    label_provenance()
    actual = base.read_label_table(LABELS)
    if set(actual) != set(ids):
        raise ValueError("EXP168 label roster differs from EXP143 v2 non-Gold cohort")
    return {uid: actual[uid] for uid in ids}


def prepare(args):
    from tools.exp155_oai_data import check_cache
    protocol = scoped_path(args.protocol)
    if protocol != PROTOCOL.resolve() or protocol.exists() or registered_run({"run_dirs": RUN_DIRS}).exists():
        raise FileExistsError("EXP168 registered protocol/run already exists or path changed")
    if (Path(args.labels).resolve() != LABELS.resolve() or
            Path(args.oai_labels).resolve() != OAI_LABELS.resolve()):
        raise ValueError("EXP168 requires exact RSNA and OAI label files")
    if args.max_hours != 7.0:
        raise ValueError("EXP168 registers exactly a 7 h wall cap")
    baseline = base.load_registration(args.base_protocol, args.corpus, args.init_weights)
    if Path(args.base_protocol).resolve() != BASE_PROTOCOL.resolve():
        raise ValueError("baseline protocol changed")
    if baseline["recipe"] != dict(base.RECIPE, labels="registered-replacement-soft"):
        raise ValueError("EXP143 v2 A2.1 baseline changed")
    rsna = label_table(baseline["train_ids"])
    base.check_corpus(args.corpus, full_hash=True)
    cache = check_cache(args.oai_corpus, full_hash=True)
    oai_labels, oai_masks, oai_prov, oai_qa, lp, qp = prior.verify_v2_labels(
        args.oai_labels, len(cache["ids"]), sha256(Path(args.oai_corpus) / "manifest.json"))
    ids, labels, masks, oai_ids = prior.combine_cohorts(
        baseline["train_ids"], rsna, baseline["excluded_gold_ids"], cache["ids"], oai_labels, oai_masks)
    if len(oai_ids) != oai_qa["admitted_oai_n"] or set(oai_ids) != set(oai_labels):
        raise ValueError("OAI v2 QA cohort changed")
    p = dict(schema=SCHEMA, version=VERSION, status="REGISTERED_NO_TRAINING",
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), recipe=RECIPE,
        budgets=dict(local=args.max_hours, reserve_seconds=600), run_dirs=RUN_DIRS,
        train_ids=ids, rsna_train_ids=baseline["train_ids"], oai_train_ids=oai_ids,
        excluded_gold_ids=baseline["excluded_gold_ids"], train_labels=labels, train_masks=masks,
        pos_weight=base.pos_weights(rsna, baseline["train_ids"]),
        label_sha256=sha256(LABELS), label_provenance_sha256=sha256(label_builder.PROVENANCE),
        oai_label_sha256=sha256(OAI_LABELS), oai_label_provenance=oai_prov,
        oai_label_provenance_sha256=sha256(lp), oai_label_qa=oai_qa,
        oai_label_qa_sha256=sha256(qp), init_sha256=baseline["init_sha256"],
        init_provenance=baseline["init_provenance"],
        corpus_manifest_sha256=sha256(Path(args.corpus) / "manifest.json"),
        corpus_build_sha256=sha256(Path(args.corpus) / "build-registration.json"),
        oai_corpus_manifest_sha256=sha256(Path(args.oai_corpus) / "manifest.json"),
        oai_corpus_build_sha256=sha256(Path(args.oai_corpus) / "build-registration.json"),
        base_protocol=str(BASE_PROTOCOL.relative_to(ROOT)), base_protocol_sha256=sha256(BASE_PROTOCOL),
        public_gold_sha256=baseline["public_gold_sha256"], source_bindings=source_bindings(),
        oai_membership=dict(cache_n=len(cache["ids"]), labels_n=len(oai_labels), admitted_n=len(oai_ids),
            cache_without_observed_label=len(cache["ids"]) - len(oai_ids),
            labelled_without_cache=sum(any(oai_masks[u]) and u not in set(cache["ids"]) for u in oai_masks)),
        evaluation=dict(gold_n=58, target_n=12, macro="all12", exposure="report-only, exposed",
            comparison="EXP155 v1.1 S42v1_1, label substitution only", selection="none"))
    p["projection_4070"] = prior.projection(execution_settings(p, 1, "bf16", 42), 3.15)
    p["content_sha256"] = json_hash(p)
    write_new_json(protocol, p)
    print(json.dumps(dict(status=p["status"], rsna=len(p["rsna_train_ids"]), oai=len(oai_ids),
                          protocol_sha256=sha256(protocol), projection=p["projection_4070"])))


def load_registration(protocol, corpus=None, init_weights=None, oai_corpus=None):
    if Path(protocol).resolve() != PROTOCOL.resolve():
        raise ValueError("EXP168 protocol path changed")
    p = read_json(protocol)
    digest = p.pop("content_sha256")
    if (json_hash(p) != digest or p.get("schema") != SCHEMA or p.get("version") != VERSION
            or p.get("recipe") != RECIPE or p.get("run_dirs") != RUN_DIRS):
        raise ValueError("EXP168 registration changed")
    p["content_sha256"] = digest
    if p.get("source_bindings") != source_bindings():
        raise ValueError("EXP168 registered source changed")
    if (p["label_sha256"] != sha256(LABELS) or
            p["label_provenance_sha256"] != sha256(label_builder.PROVENANCE)):
        raise ValueError("EXP168 label binding changed")
    actual = label_table(p["rsna_train_ids"])
    if any(p["train_labels"][uid] != actual[uid] for uid in p["rsna_train_ids"]):
        raise ValueError("EXP168 RSNA labels differ from derived CSV")
    actual_oai, actual_masks, provenance, qa, lp, qp = prior.verify_v2_labels(
        OAI_LABELS, p["oai_membership"]["cache_n"], p["oai_corpus_manifest_sha256"])
    if (p["oai_label_sha256"] != sha256(OAI_LABELS) or p["oai_label_provenance_sha256"] != sha256(lp)
            or p["oai_label_qa_sha256"] != sha256(qp) or p["oai_label_provenance"] != provenance
            or p["oai_label_qa"] != qa or set(actual_oai) != set(p["oai_train_ids"])
            or any(p["train_labels"][uid] != actual_oai[uid] or p["train_masks"][uid] != actual_masks[uid]
                   for uid in p["oai_train_ids"])):
        raise ValueError("EXP155 v2 OAI binding changed")
    baseline = base.load_registration(BASE_PROTOCOL)
    if (p["base_protocol"] != str(BASE_PROTOCOL.relative_to(ROOT))
            or p["base_protocol_sha256"] != sha256(BASE_PROTOCOL)
            or p["rsna_train_ids"] != baseline["train_ids"]
            or p["excluded_gold_ids"] != baseline["excluded_gold_ids"]
            or p["init_sha256"] != baseline["init_sha256"]
            or p["init_provenance"] != baseline["init_provenance"]
            or p["public_gold_sha256"] != baseline["public_gold_sha256"]
            or p["corpus_manifest_sha256"] != baseline["corpus_manifest_sha256"]
            or p["corpus_build_sha256"] != baseline["corpus_build_sha256"]
            or p["budgets"]["reserve_seconds"] != 600
            or p["budgets"]["local"] != 7.0):
        raise ValueError("baseline/Gold/init/budget binding changed")
    registered_run(p)
    prior.validate_membership(p)
    for path, prefix in ((corpus, ""), (oai_corpus, "oai_")):
        if path is not None:
            for filename, key in (("manifest.json", "manifest"), ("build-registration.json", "build")):
                if sha256(Path(path) / filename) != p[prefix + "corpus_" + key + "_sha256"]:
                    raise ValueError("corpus binding changed")
    if init_weights is not None and sha256(init_weights) != p["init_sha256"]:
        raise ValueError("initialization changed")
    return p


def validation_path(protocol):
    return Path(protocol).with_name("validation.json")


def validate(args):
    from exp155_own0949_oai_v11 import tiny_validation
    result = tiny_validation()
    if args.fakes_only:
        result["status"] = "FAKES_VALIDATED_NO_TRAINING"
    else:
        from exp143_v2 import build_model, load_imagenet
        from tools.exp155_oai_data import check_cache
        load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
        model = build_model()
        load_imagenet(model, args.init_weights)
        del model
        base.check_corpus(args.corpus, full_hash=True)
        check_cache(args.oai_corpus, full_hash=True)
        result.update(status="VALIDATED_NO_TRAINING", protocol_sha256=sha256(args.protocol),
            init_sha256=sha256(args.init_weights), full_corpus_hashes=True,
            full_oai_corpus_hashes=True, real_backbone_keys_loaded_cpu=True)
        write_new_json(validation_path(args.protocol), result)
    print(json.dumps(result, indent=2))


def require_validation(protocol):
    v = read_json(validation_path(protocol))
    if (v.get("status") != "VALIDATED_NO_TRAINING" or v.get("protocol_sha256") != sha256(protocol)
            or not all(v.get(k) for k in ("full_corpus_hashes", "full_oai_corpus_hashes",
                "real_backbone_keys_loaded_cpu", "masked_loss_verified"))):
        raise ValueError("full EXP168 CPU validation receipt missing/stale")


def _queue_command(args, action):
    return [args.python, "-m", "tools.exp168_own0949_oai_jev", action,
        "--protocol", str(Path(args.protocol).resolve()), "--corpus", str(Path(args.corpus).resolve()),
        "--oai-corpus", str(Path(args.oai_corpus).resolve()),
        "--init-weights", str(Path(args.init_weights).resolve())]


def probe_queue_json(args):
    p = load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    require_validation(args.protocol)
    execution_settings(p, 1, "bf16", 42, False)
    if not 1 <= args.steps <= 64 or not 0 <= args.workers <= 12:
        raise ValueError("invalid probe steps/workers")
    scratch = ROOT / ".local/exp168-probes/probe-s42"
    if scratch.exists():
        raise FileExistsError("EXP168 scratch probe exists")
    command = _queue_command(args, "probe") + ["--precision", "bf16", "--workers", str(args.workers),
        "--steps", str(args.steps), "--out", str(scratch), "--no-grad-checkpointing"]
    write_new_json(scoped_path(args.out), [dict(tag="EXP168_OAI_JEV_PROBE_S42", after=args.after,
        continue_on_failure=False, command=command)])
    print(json.dumps(dict(status="PROBE_QUEUE_JSON_ONLY_NOT_LAUNCHED", out=str(args.out),
        probe_report=str(scratch / "report.json"))))


def queue_json(args):
    p = load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    require_validation(args.protocol)
    settings = execution_settings(p, 1, "bf16", 42, False)
    probe = read_json(args.probe_report)
    result = probe.get("results", [])
    if (probe.get("protocol_sha256") != sha256(args.protocol) or len(result) != 1
            or result[0].get("mode") != "off" or result[0].get("admissible") is not True
            or result[0].get("status") != "MEASURED_SCRATCH_ONLY"
            or result[0].get("precision") != "bf16" or result[0].get("skipped_updates") != 0
            or result[0].get("execution") != settings):
        raise ValueError("EXP168 fit requires admissible same-protocol scratch probe")
    if not 0 <= args.workers <= 12 or registered_run(p).exists():
        raise ValueError("workers invalid or fresh run already exists")
    command = _queue_command(args, "train") + ["--device", "cuda", "--precision", "bf16",
        "--seed", "42", "--workers", str(args.workers), "--no-grad-checkpointing"]
    write_new_json(scoped_path(args.out), [dict(tag="EXP168_OWN0949_OAI_JEV_S42",
        after=args.after, continue_on_failure=False, command=command)])
    print(json.dumps(dict(status="QUEUE_JSON_ONLY_NOT_LAUNCHED", out=str(args.out))))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("labels")
    def common(p):
        p.add_argument("--protocol", default=str(PROTOCOL))
        p.add_argument("--corpus", default=str(CORPUS))
        p.add_argument("--oai-corpus", default=str(OAI_CORPUS))
        p.add_argument("--init-weights", default=str(INIT))
    p = sub.add_parser("prepare"); common(p)
    p.add_argument("--base-protocol", default=str(BASE_PROTOCOL))
    p.add_argument("--labels", default=str(LABELS))
    p.add_argument("--oai-labels", default=str(OAI_LABELS))
    p.add_argument("--max-hours", type=float, default=7.0)
    p = sub.add_parser("validate"); common(p)
    p.add_argument("--fakes-only", action="store_true")
    for name in ("probe-queue-json", "queue-json"):
        p = sub.add_parser(name); common(p)
        p.add_argument("--after", required=True)
        p.add_argument("--out", default=str(BASE / "protocol-v1" / ("probe-queue-s42.json" if name.startswith("probe") else "queue-s42.json")))
        p.add_argument("--python", default=base.DEFAULT_PYTHON)
        p.add_argument("--workers", type=int, default=12)
        if name.startswith("probe"):
            p.add_argument("--steps", type=int, default=20)
        else:
            p.add_argument("--probe-report", default=str(ROOT / ".local/exp168-probes/probe-s42/report.json"))
    for name in ("train", "probe", "_probe-mode"):
        p = sub.add_parser(name); common(p)
        p.add_argument("--precision", choices=["bf16"], default="bf16")
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=False)
        if name == "train":
            p.add_argument("--device", choices=["cuda"], default="cuda")
            p.add_argument("--seed", type=int, choices=[42], default=42)
            p.add_argument("--resume", action="store_true")
        else:
            p.add_argument("--steps", type=int, default=20)
            p.add_argument("--out")
            p.add_argument("--mode", choices=["off"], default="off")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "labels":
        print(json.dumps(label_builder.build(), indent=2)); return
    if args.command in ("train", "probe", "_probe-mode"):
        if args.grad_checkpointing is not False:
            raise ValueError("EXP168 forbids gradient checkpointing")
        import exp168_own0949_oai_jev as fit
        return getattr(fit, args.command.replace("_probe-mode", "probe_mode"))(args)
    return {"prepare": prepare, "validate": validate, "probe-queue-json": probe_queue_json,
            "queue-json": queue_json}[args.command](args)


if __name__ == "__main__":
    main()
