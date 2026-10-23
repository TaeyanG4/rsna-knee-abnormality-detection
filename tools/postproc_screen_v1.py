"""CPU-only fixed post-processing screen, v1.0.0.

Changelog 1.0.0: initial registered sex/pair/aggregation screen. Aggregates only.
Pre-result correction: compare reconstruction after original CSV/float32 roundtrip.
Header correction: local official ZIP uses test_series, also accept test_images.
No training, network, GPU, release mutation or submission entry points.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
import zipfile

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata
from tools import h216_pair_eval_v1 as context_helper
from tools import l1_rco_k1_eval_v1 as ev
from tools import hb128_gold58_pbranch_v1 as gold
from tools import l1_rco_stage0_v1 as stage0

VERSION = "1.0.0"
WORK = ROOT / ".local/scratch/astra-b5/postproc-screen"
REPORT = ROOT / ".local/scratch/slot-b/past-rsna-1021/POSTPROC-SCREEN.md"
REPLAY = ROOT / ".local/scratch/slot-b/blend-corr-1019/blend_corr_v1.py"
META = ROOT / "storage/labels/derived_meta/study_meta.csv"
TRAIN_REG = ROOT / stage0.REG
STRIDE = ROOT / ".local/scratch/slot-b/k1-stride-screen/preds"
TEST_SERIES = ROOT / "storage/competition/test_series.csv"
RAW_ZIP = ROOT / stage0.RAW
UID = "StudyInstanceUID"
TARGETS = list(ev.TARGETS)
RULERS = ("jev_hybrid", "aquila_v1", "dread", "consensus")
BETAS = (.03, .05, .10)
LAMBDAS = (.1, .2)
PAIRS = ((8, 7), (9, 7), (10, 11), (2, 4))
SPEC = dict(
    version=VERSION,
    base976="r(.8*r(R7-f0)+.2*r(mean(r(v2_s3),r(s0_s3),r(s1_s3))))",
    base58="exact existing blend_corr_v1.py INF3 replay, CSV roundtrip retained",
    rank="average ties / n independently per target on full fixed panel",
    sex=dict(beta=list(BETAS), source=ev.rulers.JEV,
             training="sealed K1 3190 training roster; no fold0, no Gold58",
             prior="binary JEV prevalence by M/F; O/unknown/missing -> overall TRAIN prevalence",
             formula="r((1-beta)*r(base)+beta*r(prior))"),
    pairs=dict(pairs=[[TARGETS[a], TARGETS[b]] for a, b in PAIRS],
               lambdas=list(LAMBDAS), formula="only a changes: r((1-lambda)*r(base_a)+lambda*r(base_b))"),
    aggregation=dict(
        gauss="outer two branches only; r(weighted mean(ndtri(clip(pct_rank,.5/n,1-.5/n)))); original weights",
        trimmed="outer eligible member mean: K1 three on976, O three on58; remove one min and one max per study/target; renormalize surviving original weights; other hierarchy unchanged",
        trim_two_members="not eligible; unchanged, no separate two-member variant",
        gold_weights="P .85/O .15; P unchanged; O weights 17/17/12",
        q_inf3="unchanged for both aggregation variants"),
    pass_rule="all FOUR ruler macro12 deltas > 0 AND Gold58 paired bootstrap strict P(delta>0) >= .8",
    uncertainty=dict(panel976="paired exact delete-one-study jackknife, fixed ranks/masks; 95% normal CI",
                     gold58="existing helper: 300 paired study bootstrap draws, seed20261003; fixed ranks; percentile95 CI; strict >0; invalid macro draws excluded and counted",
                     target_set="all12 fixed; never nanmean or target omission"),
    decision_status="needs verifier; provisional screen only, not a trusted result or release qualification",
    exposure="EXPOSED_EXPLORATORY; R7-f0 clean_oof=false; Gold58 exposed; correlated rulers; no multiple-testing correction",
)


def need(ok, text):
    if not ok:
        raise ValueError(text)


def rank(x):
    x = np.asarray(x, float)
    need(x.ndim in (1, 2) and len(x) > 0 and np.isfinite(x).all(), "Invalid rank input")
    return rankdata(x, method="average", axis=0) / len(x)


def mix(a, b, weight):
    return rank((1 - weight) * rank(a) + weight * rank(b))


def aggregate(members, weights, mode="mean"):
    """Member axis first; trimming never splits tied members by arbitrary identity."""
    x = np.asarray([rank(m) for m in members], float)
    w = np.asarray(weights, float)
    need(x.ndim == 3 and w.shape == (len(x),) and np.isfinite(w).all()
         and (w > 0).all(), "Invalid aggregation inputs")
    if mode == "gauss":
        n = x.shape[1]
        x = ndtri(np.clip(x, .5 / n, 1 - .5 / n))
    elif mode == "trimmed":
        need(len(x) >= 3, "Trim requires at least three members")
        # Exactly one minimum/maximum removed, sharing removal mass across ties.
        lo, hi = x == x.min(0), x == x.max(0)
        keep = 1 - lo / lo.sum(0) - hi / hi.sum(0)
        ww = w[:, None, None] * keep
        return (x * ww).sum(0) / ww.sum(0)
    else:
        need(mode == "mean", "Unknown aggregation mode")
    return np.average(x, axis=0, weights=w)


def fit_prior(train_ids, labels, metadata, excluded):
    """Select training IDs before parsing labels; never use evaluation outcomes."""
    need(len(train_ids) == len(set(train_ids)) and train_ids, "Invalid train roster")
    need(not set(train_ids) & set(excluded), "Train/evaluation overlap")
    need(all(stage0.fold(u) != 0 for u in train_ids), "Fold0 prior leakage")
    need(labels.index.is_unique and metadata.index.is_unique, "Duplicate input IDs")
    y = labels.loc[train_ids, TARGETS].to_numpy(float)
    need(np.isfinite(y).all() and np.isin(y, [0, 1]).all(), "Binary complete training labels required")
    sex = metadata.reindex(train_ids).fillna("").str.strip().str.upper().to_numpy()
    overall = y.mean(0)
    priors = {"overall": overall}
    counts = {"train_n": len(y)}
    for key in ("M", "F"):
        mask = sex == key
        need(mask.any(), "Training sex stratum empty")
        priors[key] = y[mask].mean(0)
        counts["train_" + key] = int(mask.sum())
    counts["train_other_unknown"] = int((~np.isin(sex, ["M", "F"])).sum())
    return priors, counts


def predict_prior(ids, metadata, priors):
    sex = metadata.reindex(ids).fillna("").str.strip().str.upper().to_numpy()
    out = np.stack([priors.get(s, priors["overall"]) if s in ("M", "F")
                    else priors["overall"] for s in sex])
    return out, dict(n=len(ids), known_mf=int(np.isin(sex, ["M", "F"]).sum()),
                     other_unknown=int((~np.isin(sex, ["M", "F"])).sum()))


def postprocess(base, prior):
    preds = {f"sex_b{b:.2f}": mix(base, prior, b) for b in BETAS}
    for a, b in PAIRS:
        for lam in LAMBDAS:
            p = base.copy()
            p[:, a] = mix(base[:, a], base[:, b], lam)
            preds[f"pair_{a}from{b}_l{lam:.1f}"] = p
    return preds


def passes(deltas, probability):
    return bool(set(deltas) == set(RULERS) and
                all(np.isfinite(x) and x > 0 for x in deltas.values()) and
                probability is not None and np.isfinite(probability) and probability >= .8)


def paired_panel(pred, base, refs):
    result = {}
    for ruler in RULERS:
        y, masks, _ = refs[ruler]
        rows, leaves = [], []
        for j, target in enumerate(TARGETS):
            m = masks[:, j]
            yy, pp, bb = y[m, j], pred[m, j], base[m, j]
            a, b = float(ev.auc(yy, pp)), float(ev.auc(yy, bb))
            d = a - b
            deleted = np.full(len(base), d)
            deleted[m] = ev.auc_delete_one(yy, pp) - ev.auc_delete_one(yy, bb)
            leaves.append(deleted)
            rows.append(dict(target=target, n=int(m.sum()), base=b, auc=a, delta=d,
                             ci95=ev.normal_summary(d, deleted)["ci95"]))
        delta = float(np.mean([v["delta"] for v in rows]))
        result[ruler] = dict(base=float(np.mean([v["base"] for v in rows])),
                             auc=float(np.mean([v["auc"] for v in rows])), delta=delta,
                             ci95=ev.normal_summary(delta, np.mean(leaves, axis=0))["ci95"],
                             targets=rows)
    return result


def replay_gold():
    """Execute the existing read-only replay prefix, stopping BEFORE any scoring."""
    text = REPLAY.read_text(encoding="utf-8")
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "members" for t in node.targets
        ):
            break
        nodes.append(node)
    else:
        raise ValueError("Gold replay prefix boundary missing")
    scope = {"__file__": str(REPLAY), "__name__": "_postproc_replay"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPLAY), "exec"), scope)
    ids, arrays, inf3 = scope["ids"], scope["a"], scope["INF3"]
    need(len(ids) == len(set(ids)) == 58 and inf3.shape == (58, 12), "Gold prediction roster")
    g_ids, yy = gold.gold()
    need(set(g_ids) == set(ids), "Gold roster mismatch")
    y = yy[[g_ids.index(u) for u in ids]]
    wt = np.asarray((.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5))
    public = wt * (.65 * rank(arrays["R"]) + .35 * rank(arrays["R224"])) + (1 - wt) * rank(arrays["S_R7"])
    own_members = [arrays[n] for n in ("EXP143v2", "EXP155", "Q_INF3")]
    own = aggregate(own_members, (17, 17, 12))
    reconstructed = .85 * rank(public) + .15 * rank(own)
    # The original helper deliberately reloads CSV as float32. Comparing the
    # pre-CSV float64 ranks creates spurious differences at quantized ties.
    replayed = scope["table"](scope["ns"]["p"].csv_bytes(ids, reconstructed))
    max_error = float(np.abs(replayed - inf3).max())
    need(max_error == 0 and np.array_equal(rank(replayed), rank(inf3)),
         "INF3 reconstructed base rank parity failed")
    pins = {str(scope["h208"].INVENTORY.relative_to(ROOT)): ev.sha(scope["h208"].INVENTORY),
            str((scope["h208"].INF3 / "kernel/release.ipynb").relative_to(ROOT)): scope["h208"].INF3_SHA,
            str((scope["h208"].K3 / "kernel/release.ipynb").relative_to(ROOT)): scope["h208"].K3_SHA}
    for rel in (
        ".local/scratch/hb167-common/hb167/gpu/predictions.npz",
        ".local/scratch/hb167-common/hb167/gpu/q-leaves.npz",
        ".local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output/predictions.csv",
        ".local/scratch/slot-b/research-resume-1018/run1/predictions.csv",
    ):
        pins[rel] = scope["inv"]["files"][rel]["sha256"]
    return ids, y, inf3, public, own, own_members, max_error, pins


def test_header_counts():
    """Only read PatientSex from local test-series DICOM headers, never pixels/text."""
    import pydicom
    series = pd.read_csv(TEST_SERIES, usecols=[UID, "SeriesInstanceUID"], dtype=str)
    expected = set(zip(series[UID], series.SeriesInstanceUID))
    studies = set(series[UID])
    read_studies, present, nonempty, found_series = set(), set(), set(), set()
    scanned = failures = 0
    # The local test_series extraction is empty; the locally stored official ZIP
    # is used without extracting or downloading anything.
    with zipfile.ZipFile(RAW_ZIP) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) < 4 or parts[-4] not in ("test_series", "test_images") or not parts[-1].lower().endswith(".dcm"):
                continue
            uid, sid = parts[-3:-1]
            if (uid, sid) not in expected:
                continue
            try:
                with archive.open(info) as stream:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ds = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=["PatientSex"])
                scanned += 1
                read_studies.add(uid)
                found_series.add((uid, sid))
                if "PatientSex" in ds:
                    present.add(uid)
                    if str(ds.PatientSex).strip():
                        nonempty.add(uid)
            except Exception:
                failures += 1
    return dict(test_studies=len(studies), test_series=len(expected), readable_studies=len(read_studies),
                readable_series=len(found_series), header_files_read=scanned, header_read_failures=failures,
                studies_with_PatientSex_tag=len(present), studies_with_nonempty_PatientSex=len(nonempty),
                studies_without_readable_header=len(studies - read_studies),
                scope="local visible test only, not actual hidden availability",
                source="local official ZIP test_series/test_images paths matched to local test_series.csv; PatientSex tag only")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gold.clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def register():
    path = WORK / "REGISTRATION.json"
    if path.exists():
        need(json.loads(path.read_text(encoding="utf-8")) == SPEC, "Registration drift")
    else:
        write_json(path, SPEC)
    return dict(status="REGISTERED", spec_sha256=ev.sha(path), version=VERSION)


def summarize_gold(draws):
    return gold.block(draws)


def run():
    need(json.loads((WORK / "REGISTRATION.json").read_text(encoding="utf-8")) == SPEC,
         "Run register first; immutable spec must match")
    ids977, ids, refs, r7, take = context_helper.context()
    need(ev.ids_digest(ids) == "fb04d36607a6fb9130357449dadb4f7c573183fdc033a1549975bd328a21942a", "976 roster drift")
    paths = [STRIDE / f"{name}_s3.csv" for name in ("v2", "s0", "s1")]
    members = [context_helper.load(p, ids977)[take] for p in paths]
    k1 = rank(aggregate(members, (1, 1, 1)))
    base = mix(r7, k1, .2)
    gids, gy, gbase, public, own, own_members, replay_error, gold_pins = replay_gold()
    full = json.loads(TRAIN_REG.read_text(encoding="utf-8"))
    stage0.validate_roster(full)
    need(ev.sha(TRAIN_REG) == stage0.PINS[stage0.REG], "Sealed train roster hash mismatch")
    meta = pd.read_csv(META, usecols=[UID, "sex"], dtype=str).set_index(UID)["sex"]
    labels = pd.read_csv(ROOT / ev.rulers.JEV, usecols=[UID, *TARGETS], dtype={UID: str}).set_index(UID)
    priors, counts = fit_prior(full["train"], labels, meta, ids977 + gids)
    prior, panel_sex = predict_prior(ids, meta, priors)
    gprior, gold_sex = predict_prior(gids, meta, priors)
    preds, gpreds = postprocess(base, prior), postprocess(gbase, gprior)
    preds["aggregation_gauss"] = rank(aggregate([r7, k1], (.8, .2), "gauss"))
    gpreds["aggregation_gauss"] = rank(aggregate([public, own], (.85, .15), "gauss"))
    preds["aggregation_trimmed"] = mix(r7, rank(aggregate(members, (1, 1, 1), "trimmed")), .2)
    gpreds["aggregation_trimmed"] = .85 * rank(public) + .15 * rank(aggregate(own_members, (17, 17, 12), "trimmed"))
    need(set(preds) == set(gpreds) and len(preds) == 13, "Variant grid mismatch")
    need(all(v.shape == (976, 12) and np.isfinite(v).all() for v in preds.values()), "976 outputs invalid")
    need(all(v.shape == (58, 12) and np.isfinite(v).all() for v in gpreds.values()), "Gold outputs invalid")
    paths += [META, TRAIN_REG, REPLAY, TEST_SERIES, ROOT / "storage/competition/train.csv",
              Path(__file__), ROOT / "tests/test_postproc_screen_v1.py"]
    paths += [ROOT / p for p in {ev.rulers.JEV, ev.rulers.AQUILA, *ev.rulers.CONSENSUS}]
    paths += [ev.TEACHER, ev.TEACHER_RECEIPT, ev.Q12B_REG, context_helper.SEED0]
    # Track all imported local Python sources, in addition to data/artifact pins.
    paths += [Path(m.__file__) for m in list(sys.modules.values())
              if getattr(m, "__file__", None) and str(m.__file__).endswith(".py")
              and Path(m.__file__).resolve().is_relative_to(ROOT / "tools")]
    pins = {str(p.resolve().relative_to(ROOT)).replace("\\", "/"): ev.sha(p) for p in paths}
    pins.update(gold_pins)
    write_json(WORK / "INPUT-PINS.json", pins)
    header_counts = test_header_counts()
    weights = gold.bootstrap_weights(gids)
    gscore = gold.auc_table(dict(base=gbase, prior= gprior, **gpreds), gy, weights)
    results = {}
    for name, pred in preds.items():
        panel = paired_panel(pred, base, refs)
        gold_delta = summarize_gold(gscore[name] - gscore["base"])
        prob = gold_delta["macro12"]["share_positive"]
        results[name] = dict(panel976=panel, gold58_delta=gold_delta,
                             gold58_auc=summarize_gold(gscore[name]),
                             provisional_pass=passes({r: panel[r]["delta"] for r in RULERS}, prob),
                             status="needs verifier")
    prior_panel = paired_panel(prior, base, refs)
    base_panel = paired_panel(base, base, refs)
    # Real-data numeric regression: exact existing metrics path and Gold AUC
    # independently checked against sklearn (all targets, all new paths).
    from sklearn.metrics import roc_auc_score
    worst = 0.0
    for pred in [base, prior, *preds.values()]:
        for ruler in RULERS:
            y, mask, _ = refs[ruler]
            for j in range(12):
                m = mask[:, j]
                worst = max(worst, abs(ev.auc(y[m, j], pred[m, j]) - roc_auc_score(y[m, j], pred[m, j])))
    for name, pred in dict(base=gbase, prior=gprior, **gpreds).items():
        for j in range(12):
            worst = max(worst, abs(gscore[name][0, j] - roc_auc_score(gy[:, j], pred[:, j])))
    need(worst < 1e-12, "Real-data AUC reference regression")
    for rel, digest in pins.items():
        need(ev.sha(ROOT / rel) == digest, "Input changed during screen: " + rel)
    result = dict(version=VERSION, status="needs verifier", spec=SPEC,
                  id_hash976=ev.ids_digest(ids), id_hash58=ev.ids_digest(gids), n976=len(ids), n58=len(gids),
                  train_counts=counts, panel_sex=panel_sex, gold_sex=gold_sex,
                  header_counts=header_counts, prior_auc976=prior_panel,
                  prior_auc58=summarize_gold(gscore["prior"]), base976=base_panel,
                  base58=summarize_gold(gscore["base"]), variants=results,
                  smoke=dict(status="PASS", variants=13, real_panel_paths=30,
                             max_auc_error_vs_sklearn=worst, inf3_reconstruction_max_abs=replay_error,
                             inf3_rank_parity=True, input_pins_unchanged=True),
                  input_pins=pins)
    write_json(WORK / "RESULT.json", result)
    report(result)
    return dict(status="needs verifier", variants=13,
                provisional_pass=[k for k, v in results.items() if v["provisional_pass"]],
                report=str(REPORT.relative_to(ROOT)), smoke=result["smoke"])


def fmt(x):
    return "NA" if x is None else f"{x:+.6f}"


def ci(x):
    return "NA" if x is None else "[" + ", ".join(fmt(v) for v in x) + "]"


def report(result):
    lines = ["# 후처리 CPU 스크린 — v" + VERSION, "",
             "**상태: needs verifier. 별도 lane의 코드·결과 검증 전에는 결과를 신뢰하거나 채택하지 않는다.**", "",
             "13개 고정 변형을 모두 실행했다. 아래 PASS/FAIL은 등록된 수치 규칙의 잠정 판정이며 독립 검증 PASS가 아니다.",
             "GPU·Kaggle·네트워크·제출·하위 에이전트를 사용하지 않았다. 기존/동결 파일 수정 없음.", "",
             "## 고정 정의와 한계", "",
             "- 976: 정확한 Q12b 패널. base = rank(.8 rank(R7-f0) + .2 rank(K1x3_s3)); K1x3는 v2/s0/s1의 rank 평균을 다시 rank. 모든 rank는 평균 동률 순위/n.",
             "- Gold58: 지정된 blend_corr_v1.py의 INF3 재현과 CSV roundtrip을 그대로 사용. 계층 재구성 후 rank 완전 일치를 확인했다.",
             "- 성별 prior: 고정 K1 학습 3,190건의 JEV binary 유병률. fold0와 Gold58 전체를 제외. M/F별 학습 평균, O/unknown/누락은 전체 학습 평균. prior 값 자체도 평가 패널 내 rank로 변환한다.",
             "- cross-target: 각 쌍을 개별 실행하며 해당 target만 변경. 쌍들을 합치거나 결과로 가중치를 선택하지 않았다.",
             "- rank-Gauss: 최상위 두 분기의 pct-rank를 [.5/n, 1-.5/n]으로 clip 후 정규 역누적분포 적용, 기존 .8/.2 또는 .85/.15 가중 평균, 최종 rank. 하위 집계는 그대로.",
             "- trimmed: 최상위는 두 분기라 적용 불가. 적용 가능한 976 K1 3멤버 / Gold O 3멤버 평균에서 최소·최대 한 개씩 제거한다. 동률은 제거 질량을 균등 분배하고 잔여 원래 가중치를 재정규화한다. P/O 및 R7/K1 비중은 그대로이고 Q_INF3 내부는 변경하지 않는다.",
             "- 위 집계 적용 위치와 JEV 학습 라벨 선택은 수치 확인 전 등록했다. 두 패널의 멤버 구조와 base가 달라 실제 INF3+K1 후보의 검증으로 볼 수 없다.",
             "- 잠정 PASS: 네 ruler의 macro12 delta가 각각 엄격히 양수 AND Gold58 paired-bootstrap P(delta>0)≥.8.",
             "- 976 CI: 동일 study를 두 모델·모든 target에서 함께 삭제하는 exact jackknife, 95% 정규 근사. rank/mask/prior는 고정한다. 학습 prior 추정 불확실성은 포함하지 않는다.",
             "- Gold58: 기존 helper의 300회 study bootstrap(seed 20261003), 같은 draw로 두 모델 비교, percentile 95% CI. 모든 12 targets를 유지하며 한 target이라도 단일 class인 draw는 macro에서 제외하고 유효 횟수를 표시한다. P>0는 strict >0 빈도이다.",
             "- R7-f0 pooled-prior 노출(clean_oof=false), 작은 exposed Gold58, 상관된 report rulers, 패널 선택·다중비교 영향이 있다. CI는 hidden 성능 예측이나 독립 검증을 뜻하지 않는다.", "",
             "## 요약", "",
             "| 변형 | JEV Δ [95% JK CI] | Aquila Δ [95% JK CI] | Dread Δ [95% JK CI] | consensus Δ [95% JK CI] | Gold Δ [95% bootstrap CI] | Gold P>0 | 유효 draws | 잠정 규칙 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, v in result["variants"].items():
        fields = [name]
        for r in RULERS:
            p = v["panel976"][r]
            fields.append(fmt(p["delta"]) + " " + ci(p["ci95"]))
        g = v["gold58_delta"]["macro12"]
        fields += [fmt(g["observed"]) + " " + ci(g["ci95"]), fmt(g["share_positive"]),
                   f'{g["valid_draws"]}/{g["requested_draws"]}', "PASS (needs verifier)" if v["provisional_pass"] else "FAIL (needs verifier)"]
        lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "쌍 이름: 8←7 Synovitis←Effusion; 9←7 Baker's←Effusion; 10←11 Contusion←Fracture; 2←4 Medial Meniscus←Medial OA.", "",
              "## baseline 및 prior-only AUC", "",
              "| target | JEV base/prior | Aquila base/prior | Dread base/prior | consensus base/prior | Gold base/prior |",
              "|---|---|---|---|---|---|"]
    for j, target in enumerate(TARGETS):
        fields = [target]
        for r in RULERS:
            a = result["prior_auc976"][r]["targets"][j]
            fields.append(f'{a["base"]:.6f}/{a["auc"]:.6f}')
        fields.append(f'{result["base58"]["targets"][target]["observed"]:.6f}/{result["prior_auc58"]["targets"][target]["observed"]:.6f}')
        lines.append("| " + " | ".join(fields) + " |")
    fields = ["macro12"]
    for r in RULERS:
        a = result["prior_auc976"][r]
        fields.append(f'{a["base"]:.6f}/{a["auc"]:.6f}')
    fields.append(f'{result["base58"]["macro12"]["observed"]:.6f}/{result["prior_auc58"]["macro12"]["observed"]:.6f}')
    lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "## 성별 가용성과 학습 범위", "",
              "학습·평가 metadata 집계: `" + json.dumps({k: result[k] for k in ("train_counts", "panel_sex", "gold_sex")}) + "`", "",
              "로컬 test-series 헤더 집계: `" + json.dumps(result["header_counts"]) + "`", "",
              "이는 로컬 visible test 헤더의 관측이다. 실제 hidden-test PatientSex 존재를 보장하지 않는다. 태그 값·환자별 정보는 출력하지 않았다."]
    for name, v in result["variants"].items():
        lines += ["", "## " + name + " — target별 변화", "",
                  "| target | JEV Δ [JK CI] | Aquila Δ [JK CI] | Dread Δ [JK CI] | consensus Δ [JK CI] | Gold Δ [bootstrap CI] | Gold P>0 | 유효 draws |",
                  "|---|---|---|---|---|---|---|---|"]
        for j, target in enumerate(TARGETS):
            fields = [target]
            for r in RULERS:
                a = v["panel976"][r]["targets"][j]
                fields.append(fmt(a["delta"]) + " " + ci(a["ci95"]))
            a = v["gold58_delta"]["targets"][target]
            fields += [fmt(a["observed"]) + " " + ci(a["ci95"]), fmt(a["share_positive"]),
                       f'{a["valid_draws"]}/{a["requested_draws"]}']
            lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "## 구현·재현·검증 증거", "",
              "- Script: `tools/postproc_screen_v1.py`; tests: `tests/test_postproc_screen_v1.py`.",
              "- 등록: `.local/scratch/astra-b5/postproc-screen/REGISTRATION.json`.",
              "- 모든 집계·target 수치 및 입력 SHA256: 같은 폴더 `RESULT.json`, `INPUT-PINS.json`.",
              "- exact976 ID-list SHA256: `" + result["id_hash976"] + "`.",
              "- Gold58 ID-list SHA256: `" + result["id_hash58"] + "`.",
              "- CPU real-data smoke: `" + json.dumps(result["smoke"]) + "`.",
              "- 실측 모든 신규 경로의 AUC를 sklearn으로 대조했다. 단일 bootstrap draw의 class 결손을 처리할 때 macro target 집합을 줄이지 않는다.",
              "- 별도 verifier는 diff(신규 파일 전체), prior train/holdout 분리, 집계 적용 위치, 정확한 INF3 parity, paired CI 및 strict pass 판정, 입력 pins를 확인해야 한다.",
              "- LAUNCH card 명령은 CPU `register`, `run`뿐이다. GPU admission/prepare는 해당 없음.",
              "", "SUB-AGENTS: none.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "run"])
    args = parser.parse_args()
    print(json.dumps(register() if args.action == "register" else run(), ensure_ascii=False))


if __name__ == "__main__":
    main()
