"""EXP163: EXP143 v2 plus eight-target OAI supervision.

CPU-passive CLI. Only explicit probe/train/predict-gold58 may initialize CUDA.
No networking, uploads, credential lookup or automatic submission.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "training/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from tools import exp143_v2 as base

VERSION = "1.1.0"
SCHEMA = "exp163-own0949-oai-ext-v1.1"
UID, TARGETS = base.UID, base.TARGETS
OAI_TARGETS = ("ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
               "PF OA", "Effusion", "Synovitis")
BASE = ROOT / "artifacts/experiments/EXP163_OWN0949_OAI_EXT"
BASE_PROTOCOL = ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json"
PROTOCOL_DOC = "artifacts/experiments/EXP163_OWN0949_OAI_EXT/PROTOCOL.md"
PROTOCOL = BASE / "protocol-v1/manifest.json"
CORPUS = base.CORPUS
OAI_CORPUS = ROOT / "storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02"
LABELS = ROOT / "storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv"
LABEL_SHA = "79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9"
OAI_LABELS = ROOT / "storage/labels/derived_constellation/exp163_oai_masked_v1_1.csv"
INIT = ROOT / "storage/weights/exp143-imagenet-init/coatnet384-imagenet.pt"
RUN_DIRS = {"local": "artifacts/experiments/EXP163_OWN0949_OAI_EXT/S42"}
RECIPE = dict(base.RECIPE, labels="l2-dread75-aquila25-plus-oai-eight-target-ext-v1.1",
              oai=True, grad_checkpointing_default=False, grad_checkpointing_allowed=[False],
              oai_targets=list(OAI_TARGETS), oai_mix="one-copy-per-study-per-epoch",
              loss="fixed-12-denominator-masked-bce-then-study-mean",
              class_weights="RSNA-training-only", bn_cohort="all-mixed-training-studies")
CODE = list(dict.fromkeys([*base.CODE, base.PROTOCOL,
    "tools/extcache/common.py", "tools/extcache/sources.py", "tools/extlabels.py",
    "tools/exp155_oai_data.py", "tools/exp155_oai_labels_v2.py", "tools/exp163_oai_labels.py", "tools/exp163_own0949_oai_ext.py",
    "training/src/exp163_own0949_oai_ext.py", "tools/exp163_own0949_oai_ext_release.py",
    "tools/exp152_own0949_release.py", PROTOCOL_DOC]))

sha256, json_hash, read_json = base.sha256, base.json_hash, base.read_json
series_records = base.series_records  # shared EXP152 Public3 guard interface
save_torch_new, save_torch_atomic = base.save_torch_new, base.save_torch_atomic
atomic_json = base.atomic_json


def scoped_path(path, root=None):
    path, root = Path(path).resolve(), Path(root or BASE).resolve()
    if path == root or root not in path.parents:
        raise ValueError("output must be a new child of " + str(root))
    return path


def write_new_json(path, value):
    # Training/protocol/evaluation outputs never write a bound EXP143 path.
    path = Path(path).resolve()
    if not any(r.resolve() in path.parents for r in (BASE, ROOT / ".local/exp163-probes")):
        raise ValueError("EXP163 output outside its experiment/probe subtree")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def registered_run(p, lane="local"):
    if lane != "local" or p["run_dirs"] != RUN_DIRS:
        raise ValueError("EXP163 is registered only on the local 4070, seed42")
    return scoped_path(ROOT / p["run_dirs"][lane])


def execution_settings(p, world, precision, seed, checkpointing=None):
    if world != 1 or precision != "bf16" or seed != 42 or checkpointing not in (None, False):
        raise ValueError("EXP163: one local GPU, bf16, seed42, no gradient checkpointing")
    return base.execution_settings(p, world, precision, seed, False)


projection = base.projection


def read_oai_labels(path):
    """Unknown labels stay mask=0; placeholder zeros never become supervision."""
    labels, masks = {}, {}
    expected = {"study_key", *TARGETS, *("mask_" + t for t in TARGETS)}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != 25 or set(reader.fieldnames) != expected:
            raise ValueError("OAI table must have study_key, twelve targets and twelve masks")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("invalid OAI row width")
            uid = row["study_key"]
            import re
            if not re.fullmatch(r"OAI_[0-9]+_[RL]_V00", uid) or uid in labels:
                raise ValueError("duplicate/invalid/non-baseline OAI study key")
            ys, ms = [], []
            for target in TARGETS:
                m = float(row["mask_" + target])
                if m not in (0., 1.) or (target not in OAI_TARGETS and m != 0):
                    raise ValueError("OAI supervision outside the eight registered targets")
                raw = row[target].strip()
                if m:
                    y = float(raw)
                    if not math.isfinite(y) or y not in (0., 1.):
                        raise ValueError("observed OAI labels must be binary")
                else:
                    # Blanks/NaN are unknown; a finite placeholder must be zero.
                    if raw and raw.lower() not in ("nan", "na") and float(raw) != 0:
                        raise ValueError("masked OAI cell must be blank/NaN or canonical zero")
                    y = 0.
                ys.append(y)
                ms.append(int(m))
            labels[uid], masks[uid] = ys, ms
    if not labels:
        raise ValueError("empty OAI table")
    return labels, masks


def combine_cohorts(rsna_ids, rsna_labels, gold_ids, cache_ids, oai_labels, oai_masks):
    for ids in (rsna_ids, gold_ids, cache_ids):
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("duplicate/empty cohort")
    if set(rsna_ids) & set(gold_ids) or set(cache_ids) & (set(rsna_ids) | set(gold_ids)):
        raise ValueError("cohort overlap/Gold leakage")
    if set(rsna_labels) != set(rsna_ids) or set(oai_labels) != set(oai_masks):
        raise ValueError("label membership differs")
    # Cache order fixes mixed membership. Unlabelled/all-unknown images are excluded,
    # never interpreted as negative examples. Every eligible image appears once.
    oai_ids = [u for u in cache_ids if u in oai_masks and any(oai_masks[u])]
    if not oai_ids:
        raise ValueError("no observed OAI labels matched a QA-passed cache study")
    ids = list(rsna_ids) + oai_ids
    labels = {u: list(rsna_labels[u]) for u in rsna_ids}
    masks = {u: [1] * 12 for u in rsna_ids}
    labels.update({u: oai_labels[u] for u in oai_ids})
    masks.update({u: oai_masks[u] for u in oai_ids})
    return ids, labels, masks, oai_ids


def source_bindings():
    return {s: sha256(ROOT / s) for s in CODE}


def verify_v2_labels(path, cache_n, manifest_sha):
    from tools.exp163_oai_labels import (SCHEMA as LABEL_SCHEMA, VERSION as LABEL_VERSION,
                                         aggregate_qa, clinical_inputs, DEFAULT_RECEIPT)
    if Path(path).resolve() != OAI_LABELS.resolve():
        raise ValueError("EXP163 requires the fixed extended label path")
    lp, qp = Path(path).with_suffix(".provenance.json"), Path(path).with_suffix(".qa.json")
    provenance, qa = read_json(lp), read_json(qp)
    if (provenance.get("schema") != LABEL_SCHEMA or provenance.get("version") != LABEL_VERSION
            or provenance.get("targets") != list(OAI_TARGETS)
            or provenance.get("other_targets_masked") is not True
            or provenance.get("csv_sha256") != sha256(path)
            or provenance.get("qa_sha256") != sha256(qp)
            or provenance.get("cache_manifest_sha256") != manifest_sha
            or provenance.get("exp136_receipt_sha256") != sha256(DEFAULT_RECEIPT)):
        raise ValueError("EXP163 label provenance/hash mismatch")
    clinical = clinical_inputs(DEFAULT_RECEIPT)
    if provenance.get("clinical_inputs_sha256") != {name: digest for name, (_, digest) in clinical.items()}:
        raise ValueError("EXP163 clinical input hash drift")
    labels, masks = read_oai_labels(path)
    mapped = {uid: {target: int(labels[uid][TARGETS.index(target)])
                    if masks[uid][TARGETS.index(target)] else None for target in OAI_TARGETS}
              for uid in labels}
    expected = aggregate_qa(mapped, cache_n)
    if any(qa.get(key) != value for key, value in expected.items()):
        raise ValueError("EXP163 aggregate label QA does not match CSV/cache")
    return labels, masks, provenance, qa, lp, qp


def prepare(args):
    from tools.exp155_oai_data import check_cache
    protocol = scoped_path(args.protocol)
    if protocol.exists() or (ROOT / RUN_DIRS["local"]).exists():
        raise FileExistsError("registered protocol/run already exists")
    if not math.isfinite(args.max_hours) or args.max_hours <= 600 / 3600:
        raise ValueError("finite wall budget above reserve required")
    baseline = base.load_registration(args.base_protocol, args.corpus, args.init_weights)
    if baseline["label_sha256"] != LABEL_SHA or sha256(args.labels) != LABEL_SHA:
        raise ValueError("EXP163 requires exact A2.1 L2 labels, not a new label selection")
    if baseline["recipe"] != dict(base.RECIPE, labels="registered-replacement-soft"):
        raise ValueError("base recipe is not EXP143 v2 Amendment A2.1")
    if baseline["run_dirs"]["local"] != "artifacts/experiments/EXP143_OWN0949/S42v2":
        raise ValueError("wrong baseline run")
    base.check_corpus(args.corpus, full_hash=True)
    cache = check_cache(args.oai_corpus, full_hash=True)
    oai_labels, oai_masks, label_provenance, label_qa, lp, qp = verify_v2_labels(
        args.oai_labels, len(cache["ids"]), sha256(Path(args.oai_corpus) / "manifest.json"))
    ids, labels, masks, oai_ids = combine_cohorts(baseline["train_ids"], baseline["train_labels"],
        baseline["excluded_gold_ids"], cache["ids"], oai_labels, oai_masks)
    if len(oai_ids) != label_qa["admitted_oai_n"] or set(oai_ids) != set(oai_labels):
        raise ValueError("EXP163 QA/admitted OAI cohort mismatch")
    p = dict(schema=SCHEMA, version=VERSION, status="REGISTERED_NO_TRAINING",
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), recipe=RECIPE,
        budgets=dict(local=args.max_hours, reserve_seconds=600), run_dirs=RUN_DIRS,
        train_ids=ids, rsna_train_ids=baseline["train_ids"], oai_train_ids=oai_ids,
        excluded_gold_ids=baseline["excluded_gold_ids"], train_labels=labels, train_masks=masks,
        pos_weight=base.pos_weights(baseline["train_labels"], baseline["train_ids"]),
        label_sha256=LABEL_SHA, oai_label_sha256=sha256(args.oai_labels),
        oai_label_provenance=label_provenance, oai_label_provenance_sha256=sha256(lp),
        oai_label_qa=label_qa, oai_label_qa_sha256=sha256(qp),
        init_sha256=baseline["init_sha256"], init_provenance=baseline["init_provenance"],
        corpus_manifest_sha256=sha256(Path(args.corpus) / "manifest.json"),
        corpus_build_sha256=sha256(Path(args.corpus) / "build-registration.json"),
        oai_corpus_manifest_sha256=sha256(Path(args.oai_corpus) / "manifest.json"),
        oai_corpus_build_sha256=sha256(Path(args.oai_corpus) / "build-registration.json"),
        base_protocol=str(Path(args.base_protocol).resolve().relative_to(ROOT)),
        base_protocol_sha256=sha256(args.base_protocol), public_gold_sha256=baseline["public_gold_sha256"],
        source_bindings=source_bindings(),
        oai_membership=dict(cache_n=len(cache["ids"]), labels_n=len(oai_labels), admitted_n=len(oai_ids),
            cache_without_observed_label=len(cache["ids"]) - len(oai_ids),
            labelled_without_cache=sum(any(oai_masks[u]) and u not in set(cache["ids"]) for u in oai_masks)),
        evaluation=dict(gold_n=58, target_n=12, macro="all12", exposure="report-only, exposed",
            comparison="EXP143 v2 and EXP155 v1.1, fixed epochs and registered blends",
            expected_oai_delta=None, causal_claim="extended target hypothesis; no author efficacy claim"))
    p["projection_4070"] = projection(execution_settings(p, 1, "bf16", 42), 3.15)
    p["content_sha256"] = json_hash(p)
    write_new_json(protocol, p)
    print(json.dumps(dict(status=p["status"], rsna=len(p["rsna_train_ids"]), oai=len(oai_ids),
                          label_qa=label_qa, projection=p["projection_4070"],
                          protocol_sha256=sha256(protocol))))


def validate_membership(p):
    rsna, oai, gold = p["rsna_train_ids"], p["oai_train_ids"], p["excluded_gold_ids"]
    ids, labels, masks = p["train_ids"], p["train_labels"], p["train_masks"]
    if (len(rsna) != 4347 or len(gold) != 58 or not oai or ids != rsna + oai
            or any(len(x) != len(set(x)) for x in (ids, gold))
            or set(ids) & set(gold) or set(labels) != set(ids) or set(masks) != set(ids)):
        raise ValueError("mixed training membership/Gold exclusion changed")
    allowed = {TARGETS.index(t) for t in OAI_TARGETS}
    rsna_set = set(rsna)
    for uid in ids:
        ys, ms = labels[uid], masks[uid]
        if (len(ys) != 12 or len(ms) != 12 or not any(ms)
                or any(m not in (0, 1) for m in ms)
                or any(not math.isfinite(y) or not 0 <= y <= 1 for y in ys)):
            raise ValueError("invalid registered labels/masks")
        if uid in rsna_set:
            if ms != [1] * 12:
                raise ValueError("RSNA cells must remain fully supervised")
        elif any((ms[i] and (i not in allowed or ys[i] not in (0, 1)))
                 or (not ms[i] and ys[i] != 0) for i in range(12)):
            raise ValueError("OAI target mask/unknown values changed")
    if p["pos_weight"] != base.pos_weights(labels, rsna):
        raise ValueError("class weights must be derived only from RSNA training studies")


def load_registration(protocol, corpus=None, init_weights=None, oai_corpus=None):
    scoped_path(protocol)
    p = read_json(protocol)
    digest = p.pop("content_sha256")
    if json_hash(p) != digest or p.get("schema") != SCHEMA or p.get("version") != VERSION or p.get("recipe") != RECIPE:
        raise ValueError("registration content/recipe changed")
    p["content_sha256"] = digest
    if p.get("source_bindings") != source_bindings():
        raise ValueError("registered source changed")
    actual_labels, actual_masks, provenance, qa, lp, qp = verify_v2_labels(
        OAI_LABELS, p["oai_membership"]["cache_n"], p["oai_corpus_manifest_sha256"])
    if (set(actual_labels) != set(p["oai_train_ids"])
            or qa["admitted_oai_n"] != p["oai_membership"]["admitted_n"]
            or sha256(OAI_LABELS) != p["oai_label_sha256"] or sha256(lp) != p["oai_label_provenance_sha256"]
            or sha256(qp) != p["oai_label_qa_sha256"] or provenance != p["oai_label_provenance"]
            or qa != p["oai_label_qa"] or any(p["train_labels"][u] != actual_labels[u]
            or p["train_masks"][u] != actual_masks[u] for u in p["oai_train_ids"])):
        raise ValueError("EXP163 label/QA registration binding changed")
    if (sha256(ROOT / p["base_protocol"]) != p["base_protocol_sha256"] or p["label_sha256"] != LABEL_SHA
            or p["budgets"]["reserve_seconds"] != 600 or not math.isfinite(p["budgets"]["local"])
            or p["budgets"]["local"] <= 600 / 3600):
        raise ValueError("baseline binding/labels/budget changed")
    baseline = base.load_registration(ROOT / p["base_protocol"])
    if (p["rsna_train_ids"] != baseline["train_ids"] or p["excluded_gold_ids"] != baseline["excluded_gold_ids"]
            or p["init_sha256"] != baseline["init_sha256"]
            or any(p["train_labels"][u] != baseline["train_labels"][u] for u in p["rsna_train_ids"])):
        raise ValueError("RSNA baseline cohort/labels/initialization changed")
    registered_run(p)
    validate_membership(p)
    for path, prefix in ((corpus, ""), (oai_corpus, "oai_")):
        if path is not None:
            for filename, key in (("manifest.json", "manifest"), ("build-registration.json", "build")):
                if sha256(Path(path) / filename) != p[prefix + "corpus_" + key + "_sha256"]:
                    raise ValueError("corpus binding changed")
            if prefix:
                from tools.exp155_oai_data import check_cache
                cp = check_cache(path)
                if not set(p["oai_train_ids"]) <= set(cp["ids"]):
                    raise ValueError("OAI cache cohort changed")
            else:
                cp = base.check_corpus(path)
                if set(cp["ids"]) != set(p["rsna_train_ids"]) | set(p["excluded_gold_ids"]):
                    raise ValueError("RSNA corpus cohort changed")
    if init_weights is not None and sha256(init_weights) != p["init_sha256"]:
        raise ValueError("initialization changed")
    return p


def validation_path(protocol):
    return Path(protocol).with_name("validation.json")


def validate(args):
    from exp163_own0949_oai_ext import tiny_validation
    result = tiny_validation()
    if not args.fakes_only:
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
    else:
        result["status"] = "FAKES_VALIDATED_NO_TRAINING"
    print(json.dumps(result, indent=2))


def require_validation(protocol):
    v = read_json(validation_path(protocol))
    if (v.get("status") != "VALIDATED_NO_TRAINING" or v.get("protocol_sha256") != sha256(protocol)
            or not all(v.get(k) for k in ("full_corpus_hashes", "full_oai_corpus_hashes",
                       "real_backbone_keys_loaded_cpu", "masked_loss_verified"))):
        raise ValueError("full EXP163 CPU validation receipt missing/stale")


def queue_json(args):
    p = load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    require_validation(args.protocol)
    execution_settings(p, 1, "bf16", 42, False)
    probe = read_json(args.probe_report)
    result = probe.get("results", [])
    if (probe.get("protocol_sha256") != sha256(args.protocol) or len(result) != 1
            or result[0].get("mode") != "off" or result[0].get("admissible") is not True
            or result[0].get("status") != "MEASURED_SCRATCH_ONLY"
            or result[0].get("precision") != "bf16" or result[0].get("skipped_updates") != 0
            or result[0].get("execution") != execution_settings(p, 1, "bf16", 42, False)):
        raise ValueError("queue requires an admissible mixed EXP163 probe report")
    if not 0 <= args.workers <= 12 or registered_run(p).exists():
        raise ValueError("invalid workers or existing run; fresh queue only")
    command = [args.python, "-m", "tools.exp163_own0949_oai_ext", "train",
        "--protocol", str(Path(args.protocol).resolve()), "--corpus", str(Path(args.corpus).resolve()),
        "--oai-corpus", str(Path(args.oai_corpus).resolve()), "--init-weights", str(Path(args.init_weights).resolve()),
        "--device", "cuda", "--precision", "bf16", "--seed", "42", "--workers", str(args.workers),
        "--no-grad-checkpointing"]
    write_new_json(scoped_path(args.out), [dict(tag="EXP163_OWN0949_OAI_EXT_S42", after=args.after,
                                              continue_on_failure=False, command=command)])
    print(json.dumps(dict(status="QUEUE_JSON_ONLY_NOT_LAUNCHED", out=str(args.out))))


def probe_queue_json(args):
    p = load_registration(args.protocol, args.corpus, args.init_weights, args.oai_corpus)
    require_validation(args.protocol)
    execution_settings(p, 1, "bf16", 42, False)
    if not 1 <= args.steps <= 64 or not 0 <= args.workers <= 12:
        raise ValueError("invalid probe steps/workers")
    scratch = (ROOT / ".local/exp163-probes/probe-s42").resolve()
    if scratch.exists():
        raise FileExistsError("EXP163 scratch probe already exists")
    command = [args.python, "-m", "tools.exp163_own0949_oai_ext", "probe",
        "--protocol", str(Path(args.protocol).resolve()), "--corpus", str(Path(args.corpus).resolve()),
        "--oai-corpus", str(Path(args.oai_corpus).resolve()), "--init-weights", str(Path(args.init_weights).resolve()),
        "--precision", "bf16", "--workers", str(args.workers), "--steps", str(args.steps),
        "--out", str(scratch), "--no-grad-checkpointing"]
    write_new_json(scoped_path(args.out), [dict(tag="EXP163_OAI_EXT_PROBE_S42", after=args.after,
                                              continue_on_failure=False, command=command)])
    print(json.dumps(dict(status="PROBE_QUEUE_JSON_ONLY_NOT_LAUNCHED", out=str(args.out),
                          probe_report=str(scratch / "report.json"))))


def check_checkpoint(ck, p, protocol, lane):
    from tools.exp163_own0949_oai_ext_release import own_meta
    own_meta(ck, sha256(protocol), p["run_dirs"][lane])


def predict_gold58(args):
    # Same v2 decode/model/inference and checks, with isolated function globals.
    # Never monkey-patch the imported baseline module or edit its bound files.
    environment = dict(base.predict.__globals__, load_registration=load_registration,
        registered_run=registered_run, check_checkpoint=check_checkpoint,
        guard_output=scoped_path, write_new_json=write_new_json, CODE=CODE)
    return types.FunctionType(base.predict.__code__, environment)(args, True)


def report(args):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    p = load_registration(args.protocol)
    baseline = base.load_registration(ROOT / p["base_protocol"])
    _, ids, y = base.official_population(args.competition)
    if set(ids) != set(p["excluded_gold_ids"]):
        raise ValueError("Gold membership changed")
    own = registered_run(p) / "gold58.csv"
    control = base.registered_run(baseline) / "gold58.csv"
    for path, reg, protocol in ((own, p, args.protocol), (control, baseline, ROOT / p["base_protocol"])):
        r = read_json(str(path) + ".receipt.json")
        if (r.get("protocol_sha256") != sha256(protocol) or r.get("output_sha256") != sha256(path)
                or r.get("ranks") or r.get("swa_epochs_1based") != [14, 11, 12]
                or r.get("run_dir") != reg["run_dirs"]["local"]):
            raise ValueError("report requires authenticated fixed-SWA raw predictions")
    if sha256(args.public) != p["public_gold_sha256"]:
        raise ValueError("public comparator changed")
    public = base.read_predictions(args.public, ids)
    values = {"exp143_v2": base.read_predictions(control, ids)}
    values.update(base.fixed_blends(public, base.read_predictions(own, ids)))
    values["exp163"] = values.pop("own143")
    values["exp143_public50_own50"] = base.fixed_blends(public, values["exp143_v2"])["primary_public50_own50"]
    values["public_exp143_exp163_equal_rank"] = (
        base.rank_matrix(public) + base.rank_matrix(values["exp143_v2"])
        + base.rank_matrix(values["exp163"])) / 3
    results = {}
    for name, val in values.items():
        per = [float(roc_auc_score(y[:, i], np.asarray(val)[:, i])) for i in range(12)]
        results[name] = dict(macro12=float(np.mean(per)), per_target=dict(zip(TARGETS, per)))
    output = dict(status="REPORT_ONLY_NO_SELECTION", n=58, targets=TARGETS, results=results,
        delta_own_macro12=results["exp163"]["macro12"] - results["exp143_v2"]["macro12"],
        delta_fixed50_macro12=results["primary_public50_own50"]["macro12"] - results["exp143_public50_own50"]["macro12"],
        protocol_sha256=sha256(args.protocol), control_sha256=sha256(control), own_sha256=sha256(own),
        public_sha256=sha256(args.public), official_labels_sha256=json_hash(y.astype(int).tolist()),
        caveat="Exposed Gold58; not independent validation. OAI arm has more optimizer steps. No tuning or private-LB claim.")
    write_new_json(args.out or registered_run(p) / "gold58-report.json", output)
    print(json.dumps(output, indent=2))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="command", required=True)
    from tools.exp155_oai_data import DEFAULT_CACHE
    cache = sub.add_parser("build-oai-cache", help="resume the unchanged v1 cache builder")
    cache.add_argument("--root", type=Path, default=ROOT)
    cache.add_argument("--oai-cache", type=Path, default=DEFAULT_CACHE)
    cache.add_argument("--min-free-gib", type=float, default=20.0)
    cache.add_argument("--resume", action="store_true")
    def common(p):
        p.add_argument("--protocol", default=str(PROTOCOL))
        p.add_argument("--corpus", default=str(CORPUS))
        p.add_argument("--oai-corpus", default=str(OAI_CORPUS))
        p.add_argument("--init-weights", default=str(INIT))
    p = sub.add_parser("prepare")
    common(p)
    p.add_argument("--base-protocol", default=str(BASE_PROTOCOL))
    p.add_argument("--labels", default=str(LABELS))
    p.add_argument("--oai-labels", default=str(OAI_LABELS))
    p.add_argument("--max-hours", type=float, default=14.)
    p = sub.add_parser("validate")
    common(p)
    p.add_argument("--fakes-only", action="store_true")
    p = sub.add_parser("queue-json")
    common(p)
    p.add_argument("--after", required=True)
    p.add_argument("--probe-report", required=True)
    p.add_argument("--out", default=str(BASE / "protocol-v1/queue-s42.json"))
    p.add_argument("--python", default=base.DEFAULT_PYTHON)
    p.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("probe-queue-json")
    common(p)
    p.add_argument("--after", required=True)
    p.add_argument("--out", default=str(BASE / "protocol-v1/probe-queue-s42.json"))
    p.add_argument("--python", default=base.DEFAULT_PYTHON)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--steps", type=int, default=20)
    for name in ("train", "probe", "_probe-mode"):
        p = sub.add_parser(name)
        common(p)
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
    p = sub.add_parser("predict-gold58")
    common(p)
    p.add_argument("--competition", default=str(ROOT / "storage/competition"))
    p.add_argument("--lane", choices=["local"], default="local")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["fp16"], default="fp16")
    p.add_argument("--checkpoint")
    p.add_argument("--out")
    p = sub.add_parser("report")
    common(p)
    p.add_argument("--competition", default=str(ROOT / "storage/competition"))
    p.add_argument("--public", default=str(base.PUBLIC_GOLD))
    p.add_argument("--out")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "build-oai-cache":
        from tools import exp155_oai_data as data
        result = data.build_cache(args)
        print(json.dumps(result, indent=2))
        if result.get("status") != "COMPLETE":
            raise SystemExit(1)
        return result
    if args.command in ("train", "probe", "_probe-mode"):
        if args.grad_checkpointing is not False:
            raise ValueError("EXP163 registers gradient checkpointing off only")
        import exp163_own0949_oai_ext as fit
        return getattr(fit, args.command.replace("_probe-mode", "probe_mode"))(args)
    return {"prepare": prepare, "validate": validate, "queue-json": queue_json,
            "probe-queue-json": probe_queue_json,
            "predict-gold58": predict_gold58, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
