"""CPU-only fixed label-lens screen, v1.0.0.

Changelog 1.0.0: compare three untrusted numeric label lenses with four existing
rulers on two predeclared Q12b976 contrasts; aggregate outputs only.
External dataset code is never imported or executed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr
import sklearn
from sklearn.metrics import roc_auc_score
from tools import h216_pair_eval_v1 as ctx
from tools import l1_rco_k1_eval_v1 as ev

VERSION = "1.0.0"
WORK = ROOT / ".local/scratch/astra-b5/label-lens"
REPORT = ROOT / ".local/scratch/slot-b/past-rsna-1021/LABEL-LENS.md"
DATA = ROOT / "storage/external_data/omerzalman-label-fusion-v2"
STRIDE = ROOT / ".local/scratch/slot-b/k1-stride-screen/preds"
UID = "StudyInstanceUID"
INDICES = [i for i, t in enumerate(ev.TARGETS) if t != "Synovitis"]
TARGETS = [ev.TARGETS[i] for i in INDICES]
RULERS = ["jev_hybrid", "aquila_v1", "dread", "consensus"]
EXTERNAL = dict(L_soft="soft_labels.csv",
                L_strict="strict_mri_oof_soft_labels.csv", L_weak="weak_labels.csv")
SHARE = [.1, .2, .2, .5, .35, .5, .35, .35, .5, .2, .15, .35]
CANDIDATES = ["R7", "v2_s3_flat20", "v2_s3_shareV2", "x3_s3_flat20",
              "x3_s3_shareV2", "x3_s2_flat20", "x3_s2_shareV2"]
PRIMARY = ["R7", "x3_s3_flat20", "x3_s2_shareV2"]
PAIRS = dict(pair1=["x3_s3_flat20", "x3_s2_shareV2"],
             pair2=["x3_s3_flat20", "R7"])
PANEL_SHA = "fb04d36607a6fb9130357449dadb4f7c573183fdc033a1549975bd328a21942a"
SPEC = dict(
    version=VERSION, n=976, panel_sha256=PANEL_SHA, targets=TARGETS,
    external_files=EXTERNAL, threshold=">= 0.5 (diagnostic, not calibrated probability)",
    rank="average ties / n, per target on exact976, before any ruler mask",
    x3="rank(mean(rank(v2), rank(s0), rank(s1))) at the specified stride",
    blend="rank((1-a_t)*rank(R7f0)+a_t*rank(K1)); flat a=.20",
    shareV2=SHARE, pairs=PAIRS, candidates=CANDIDATES,
    lb_facts=dict(pair1="k1x3sr20-v2 0.958 > av2 0.957, D-B-251",
                  pair2="INF3+K1 .20 selected above INF3, D-B-249; magnitude unspecified"),
    track_rule="BOTH macro11 deltas >0 AND paired jackknife normal P(delta>0)>=.8 EACH",
    uncertainty="paired delete-one-study jackknife; same deleted study across11; "
                "fixed full-panel ranks/masks; normal95 CI; no target dropping",
    correlation=dict(primary3=PRIMARY, descriptive7=CANDIDATES,
                     statistic="Pearson and Spearman across candidate macro11 AUC vectors, "
                               "external lens versus each ruler; descriptive only"),
    status="needs verifier",
    exposure="EXPOSED_EXPLORATORY; clean_oof=false R7 proxy; shared source/selection "
             "exposure; no independent clinical or hidden-LB validation",
)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def rank(x):
    x = np.asarray(x, float)
    need(x.ndim in (1, 2) and len(x) > 0 and np.isfinite(x).all(), "Invalid rank input")
    return rankdata(x, axis=0, method="average") / len(x)


def candidates(r7, members):
    need(r7.ndim == 2 and r7.shape[1] == 12, "Expected twelve prediction targets")
    need(set(members) == {f"{m}_{s}" for m in ("v2", "s0", "s1") for s in ("s3", "s2")},
         "Incomplete fixed member set")
    need(all(p.shape == r7.shape for p in members.values()), "Member shape mismatch")
    k = dict(v2_s3=rank(members["v2_s3"]))
    for stride in ("s3", "s2"):
        k["x3_" + stride] = rank(np.mean(
            [rank(members[m + "_" + stride]) for m in ("v2", "s0", "s1")], axis=0))
    rr = rank(r7)
    out = dict(R7=rr)
    for name, p in k.items():
        for suffix, a in (("flat20", .2), ("shareV2", np.array(SHARE))):
            out[name + "_" + suffix] = rank((1 - a) * rr + a * rank(p))
    need(list(out) == CANDIDATES, "Candidate order mismatch")
    return out


def read_lens(path, ids):
    """Parse only the requested numeric CSV columns; sanitize all parser errors."""
    need(Path(path).suffix.lower() == ".csv", "Only CSV input accepted")
    need(len(ids) == len(set(ids)) and len(ids) > 0, "Invalid evaluation roster")
    try:
        df = pd.read_csv(path, usecols=[UID, *TARGETS], dtype={UID: str})
    except Exception:
        raise ValueError("External CSV parsing/schema failure (row details suppressed)") from None
    need(df[UID].notna().all() and not df[UID].duplicated().any(), "Invalid external UID column")
    need(set(ids).issubset(set(df[UID])), "External lens missing panel studies; no panel shrink allowed")
    try:
        values = df.set_index(UID).loc[ids, TARGETS].to_numpy(float)
    except Exception:
        raise ValueError("External numeric conversion failed (row details suppressed)") from None
    need(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
         "External panel targets nonfinite or outside [0,1]")
    y = (values >= .5).astype(int)
    counts = dict(source_rows=len(df), panel_n=len(ids), missing_panel=0,
                  source_rows_outside_panel=len(df) - len(ids),
                  targets=[dict(target=t, n=len(ids), positives=int(y[:, j].sum()),
                                exactly_half=int((values[:, j] == .5).sum()))
                           for j, t in enumerate(TARGETS)])
    return (y, np.ones_like(y, bool)), counts


def paired(a, b, y, masks):
    """All eleven targets remain in the macro; undefined deletions fail closed."""
    a, b, y, masks = map(np.asarray, (a, b, y, masks))
    need(a.ndim == 2 and a.shape == b.shape == y.shape == masks.shape
         and a.shape[1] == 11 and len(a) > 2, "Expected matched n by eleven arrays")
    need(masks.dtype == bool and np.isfinite(a).all() and np.isfinite(b).all(),
         "Invalid mask or prediction")
    need(np.isin(y[masks], [0, 1]).all(), "Nonbinary included labels")
    rows, leaves = [], []
    for j, t in enumerate(TARGETS):
        m = masks[:, j]
        yy, aa, bb = y[m, j], a[m, j], b[m, j]
        pos, neg = int(yy.sum()), int(len(yy) - yy.sum())
        need(pos >= 2 and neg >= 2, "Insufficient target classes for delete-one jackknife")
        va, vb = float(ev.auc(yy, aa)), float(ev.auc(yy, bb))
        d = va - vb
        leave = np.full(len(a), d)
        leave[m] = ev.auc_delete_one(yy, aa) - ev.auc_delete_one(yy, bb)
        need(np.isfinite(leave).all(), "Undefined jackknife; no target dropping allowed")
        leaves.append(leave)
        rows.append(dict(target=t, n=int(m.sum()), positives=pos, negatives=neg,
                         auc_a=va, auc_b=vb, delta=d, **ev.normal_summary(d, leave)))
    delta = float(np.mean([v["delta"] for v in rows]))
    return dict(auc_a=float(np.mean([v["auc_a"] for v in rows])),
                auc_b=float(np.mean([v["auc_b"] for v in rows])), delta=delta,
                **ev.normal_summary(delta, np.mean(leaves, axis=0)), targets=rows)


def tracks(pairs):
    if set(pairs) != set(PAIRS):
        return False
    return all(p["delta"] > 0 and np.isfinite(p["delta"])
               and p["p_delta_positive_normal"] is not None
               and np.isfinite(p["p_delta_positive_normal"])
               and p["p_delta_positive_normal"] >= .8 for p in pairs.values())


def correlations(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    need(a.ndim == 1 and a.shape == b.shape and len(a) >= 3
         and np.isfinite([a, b]).all(), "Invalid correlation vectors")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(n=len(a), pearson=None, spearman=None, status="UNDEFINED_CONSTANT")
    return dict(n=len(a), pearson=float(np.corrcoef(a, b)[0, 1]),
                spearman=float(spearmanr(a, b).statistic), status="DEFINED")


def registration():
    ids = ctx.ids977()
    spec = ev.evaluator_registration(ids)
    need(len(spec["primary"]["ids"]) == 976
         and spec["primary"]["id_list_sha256"] == PANEL_SHA, "Exact Q12b976 mismatch")
    paths = [Path(__file__), ROOT / "tests/test_label_lens_screen_v1.py",
             Path(ctx.__file__), ctx.SEED0,
             ROOT / ".local/scratch/slot-b/codex-1520/DIAG-RESULT.md",
             DATA / "DATASET_CARD.txt", DATA / "THIRD_PARTY_NOTICES.txt",
             *[DATA / f for f in EXTERNAL.values()],
             *[STRIDE / f"{m}_{s}.csv" for m in ("v2", "s0", "s1") for s in ("s3", "s2")]]
    pins = dict(spec["pins"])
    pins.update({str(p): ev.sha(p) for p in paths})
    # Never persist primary IDs, excluded metadata or any label rows.
    return dict(spec=SPEC, pins=pins, runtime=dict(
        python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__))


def register():
    reg = registration()
    path = WORK / "REGISTRATION.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        need(all(old[k] == reg[k] for k in reg), "Existing registration differs; do not overwrite")
        return dict(status="REGISTERED_IDENTICAL", pins=len(reg["pins"]))
    need(not (WORK / "RESULT.json").exists(), "Cannot register after results")
    reg["registered_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, reg)
    return dict(status="REGISTERED", n=976, targets=11, pins=len(reg["pins"]))


def validate_registration():
    old = json.loads((WORK / "REGISTRATION.json").read_text(encoding="utf-8"))
    current = registration()
    need(all(old[k] == current[k] for k in current), "Registered code/data/runtime changed")
    return old


def evaluate():
    reg = validate_registration()
    ids, primary, refs12, r7, take = ctx.context()
    need(len(primary) == 976 and ev.ids_digest(primary) == PANEL_SHA, "Panel identity differs")
    members = {f"{m}_{s}": ctx.load(STRIDE / f"{m}_{s}.csv", ids)[take]
               for m in ("v2", "s0", "s1") for s in ("s3", "s2")}
    full = candidates(r7, members)
    preds = {k: v[:, INDICES] for k, v in full.items()}
    refs = {k: (y[:, INDICES], masks[:, INDICES]) for k, (y, masks, _) in refs12.items()}
    coverage = {}
    for name, filename in EXTERNAL.items():
        refs[name], coverage[name] = read_lens(DATA / filename, primary)
    lenses, max_err, checks = {}, 0., 0
    for name in [*EXTERNAL, *RULERS]:
        y, masks = refs[name]
        aucs, target_aucs = {}, {}
        for candidate, pred in preds.items():
            vals = []
            for j in range(11):
                m = masks[:, j]
                need(np.isin(y[m, j], [0, 1]).all(), "Nonbinary included labels")
                value = float(ev.auc(y[m, j], pred[m, j]))
                trusted = float(roc_auc_score(y[m, j], pred[m, j]))
                need(np.isfinite(value), "Undefined candidate AUC")
                max_err = max(max_err, abs(value - trusted))
                checks += 1
                vals.append(value)
            aucs[candidate] = float(np.mean(vals))
            target_aucs[candidate] = dict(zip(TARGETS, vals))
        pairs = {p: paired(preds[a], preds[b], y, masks) for p, (a, b) in PAIRS.items()}
        lenses[name] = dict(candidate_auc=aucs, candidate_target_auc=target_aucs,
                            pairs=pairs, tracks_lb=tracks(pairs))
    need(max_err < 1e-12, "AUC differs from sklearn")
    corr = {}
    for group, names in (("primary3", PRIMARY), ("descriptive7", CANDIDATES)):
        corr[group] = {lens: {r: correlations(
            [lenses[lens]["candidate_auc"][c] for c in names],
            [lenses[r]["candidate_auc"][c] for c in names]) for r in RULERS} for lens in EXTERNAL}
    # Aggregate parity check with the previously reported full12 DIAG numbers.
    expected = {
        "R7": [.90632, .92548, .89799, .96934],
        "x3_s3_flat20": [.91243, .93088, .90650, .97409],
        "x3_s2_shareV2": [.91568, .93298, .91087, .97638]}
    diag_error = 0.
    for c, values in expected.items():
        for name, wanted in zip(RULERS, values):
            y, masks, _ = refs12[name]
            got = float(np.mean([ev.auc(y[masks[:, j], j], full[c][masks[:, j], j])
                                 for j in range(12)]))
            diag_error = max(diag_error, abs(got - wanted))
    need(diag_error <= .0000051, "Candidate reconstruction differs from rounded DIAG")
    validate_registration()
    result = dict(version=VERSION, status="needs verifier", n=976, targets=TARGETS,
                  clean_oof=False, exposure=SPEC["exposure"],
                  registration_sha256=ev.sha(WORK / "REGISTRATION.json"),
                  evaluated_utc=datetime.now(timezone.utc).isoformat(),
                  coverage=coverage, lenses=lenses, correlations=corr,
                  validation=dict(sklearn_auc_checks=checks, max_abs_auc_error=max_err,
                                  max_abs_diag12_rounded_error=diag_error,
                                  inputs_unchanged=True))
    write_json(WORK / "RESULT.json", result)
    report(result, reg)
    return dict(status="needs verifier", n=976, targets=11,
                tracks={k: v["tracks_lb"] for k, v in lenses.items()},
                validation=result["validation"], report=str(REPORT.relative_to(ROOT)))


def number(x, signed=False):
    return "NA" if x is None else format(x, "+.6f" if signed else ".6f")


def interval(x):
    return "NA" if x is None else "[" + ", ".join(number(v, True) for v in x) + "]"


def report(result, reg):
    lenses = result["lenses"]
    passed = [k for k, v in lenses.items() if v["tracks_lb"]]
    rows = [
        "# Omerzalman label-fusion v2 — 고정 LB 방향 비교",
        "",
        "**상태: needs verifier.** astra-b5 자체 구현·테스트·CPU 실행 결과이며 독립 검증 PASS가 아니다.",
        f"사전 규칙 통과: **{', '.join(passed) if passed else '없음'}** ({len(passed)}/7). "
        "이 판정은 두 알려진 LB 비교의 대리 실험에 한정된다. 새 lens가 hidden LB를 더 잘 예측한다는 일반화 근거로 쓰지 않는다.",
        "",
        "## 고정 설계",
        "",
        "- 정확한 Q12b fold-0 976 studies; 977 예측 파일을 정렬·검증한 뒤 고정 976을 선택했다. "
        "모든 lens에서 Synovitis를 제외한 동일 11개 finding을 단순 평균했다. "
        "consensus는 기존 target별 합의 mask를 그대로 사용한다.",
        f"- Panel ID-list SHA256: `{PANEL_SHA}`. ID 및 label 행은 출력·저장하지 않았다.",
        "- Pair1 = x3_s3_flat20 − x3_s2_shareV2; 알려진 LB 0.958 > 0.957 (D-B-251).",
        "- Pair2 = x3_s3_flat20 − R7; INF3+K1 .20이 INF3보다 위라는 auto-selection 사실 (D-B-249). "
        "R7-f0는 INF3/P의 proxy이므로 실제 INF3 순위를 직접 재현한 실험은 아니다.",
        "- rank = target별 average ties/n, 정확한 976 전체에서 계산. "
        "K1 x3 = rank(mean(rank(v2), rank(s0), rank(s1))). "
        "최종 = rank((1−a)rank(R7)+a rank(K1)); flat a=.20.",
        f"- shareV2(원래 12-target 순서) = `{SHARE}`. Synovitis는 평가에서만 제외. "
        "DIAG-RESULT.md의 중간 재순위 단계를 보존했다.",
        "- 외부 3개 CSV는 모두 >=0.5로 이진화. 카드에 별도 이진 AUC 규칙이 없으며 "
        "soft ranking target은 보정된 임상 확률이 아니다. threshold 탐색 없음. "
        "기존 ruler 변환은 JEV binary / Aquila >=.5 / Dread np.rint / 기존 consensus를 유지했다.",
        "- 통과 = 두 macro11 Δ 모두 >0이고 각각 paired jackknife normal P(Δ>0)>=.8. "
        "P는 delete-one-study SE와 정규 근사 CDF로 계산한 방향 지지도이며, "
        "jackknife 표본 중 양수 비율 또는 Bayesian posterior가 아니다.",
        "- study 하나를 모든 target에서 동시에 제외; 예측/rank/mask 고정. "
        "SE=sqrt((n−1)/n × Σ(leave−mean(leave))²), CI=Δ±1.959964×SE. "
        "mask 밖 study 삭제는 해당 target을 바꾸지 않는다. target 누락·nanmean 없음.",
        f"- 사전 등록 UTC: {reg['registered_utc']}; 실행 UTC: {result['evaluated_utc']}. "
        "등록은 이 작업의 점수 계산 전에 기록했다. 이미 알려진 LB 사실을 이용한 회고적 검증이다.",
        "",
        "## Macro11 결과",
        "",
        "Δ 단위는 AUC이며 .01=1%p. 아래 CI는 95% normal jackknife 구간이다.",
        "",
        "| Lens/ruler | Pair1 Δ | CI | P(>0) | Pair2 Δ | CI | P(>0) | 두 비교 통과 |",
        "|---|---:|---|---:|---:|---|---:|---|",
    ]
    for name, lens in lenses.items():
        a, b = lens["pairs"]["pair1"], lens["pairs"]["pair2"]
        rows.append(f"| {name} | {number(a['delta'], True)} | {interval(a['ci95'])} | "
                    f"{number(a['p_delta_positive_normal'])} | {number(b['delta'], True)} | "
                    f"{interval(b['ci95'])} | {number(b['p_delta_positive_normal'])} | "
                    f"{'YES' if lens['tracks_lb'] else 'NO'} |")
    rows += ["", "## 후보별 macro11 AUC", "",
             "| Candidate | " + " | ".join(lenses) + " |",
             "|---|" + "---:|" * len(lenses)]
    for c in CANDIDATES:
        rows.append("| " + c + " | " + " | ".join(
            number(v["candidate_auc"][c]) for v in lenses.values()) + " |")
    rows += ["", "## Lens–ruler 후보 AUC 상관", "",
             "동일 후보의 macro11 AUC 벡터 사이 Pearson r / Spearman ρ. 핵심 3개는 "
             "`R7, x3_s3_flat20, x3_s2_shareV2`; 기술적 7개는 위 표 전체다. "
             "공통 R7 대비 상승이 상관을 크게 만들 수 있으므로 높은 상관만으로 Pair1 일치를 추론하지 않는다. "
             "n=3/7은 작고 후보들은 강하게 종속되어 있어 일반화 검정으로 해석하지 않는다.", "",
             "| Scope | Lens | Ruler | n | Pearson r | Spearman ρ |",
             "|---|---|---|---:|---:|---:|"]
    for group, entries in result["correlations"].items():
        for lens, rulers in entries.items():
            for r, v in rulers.items():
                rows.append(f"| {group} | {lens} | {r} | {v['n']} | "
                            f"{number(v['pearson'])} | {number(v['spearman'])} |")
    rows += ["", "## Target별 두 비교", "",
             "N과 양성 수는 각 lens의 target별 집계다. A/B/C는 각각 "
             "x3_s3_flat20 / x3_s2_shareV2 / R7의 AUC. 모든 per-target CI/P도 같은 paired jackknife다."]
    for name, lens in lenses.items():
        rows += ["", f"### {name}", "",
                 "| Target | N | 양성 | A AUC | B AUC | C AUC | Pair1 Δ | CI | P(>0) | Pair2 Δ | CI | P(>0) |",
                 "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|"]
        for a, b in zip(lens["pairs"]["pair1"]["targets"], lens["pairs"]["pair2"]["targets"]):
            rows.append(f"| {a['target']} | {a['n']} | {a['positives']} | "
                        f"{number(a['auc_a'])} | {number(a['auc_b'])} | {number(b['auc_b'])} | "
                        f"{number(a['delta'], True)} | {interval(a['ci95'])} | "
                        f"{number(a['p_delta_positive_normal'])} | {number(b['delta'], True)} | "
                        f"{interval(b['ci95'])} | {number(b['p_delta_positive_normal'])} |")
    rows += ["", "## 입력 범위·출처와 한계", "",
             "| Lens | CSV 행 수 | panel 포함 | panel 누락 | panel 밖 | 정확히 .5인 panel cells |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, cov in result["coverage"].items():
        rows.append(f"| {name} | {cov['source_rows']} | {cov['panel_n']} | {cov['missing_panel']} | "
                    f"{cov['source_rows_outside_panel']} | {sum(x['exactly_half'] for x in cov['targets'])} |")
    rows += [
        "",
        "- DATASET_CARD.txt와 THIRD_PARTY_NOTICES.txt를 읽었다. Omerzalman label-fusion v2의 "
        "새 curated prediction은 **CC BY-NC 4.0**, 이번에는 평가 lens로만 사용했다. "
        "원 출처의 licence/attribution도 유지된다. 카드·notices 및 CSV 해시는 등록 파일에 있다.",
        "- 데이터셋 전체는 4,407 studies/11 findings의 pseudo-label이다. soft는 전체, weak는 4,349 weak rows. "
        "58 expert rows의 soft 값도 복사한 정답이 아닌 outer-held prediction이다. "
        "familiar Gold58과 public source/family 선택의 반복 노출은 남아 있다.",
        "- 카드에 따르면 full MRI ensemble의 최대 4/5 reader가 weak study의 이전 report-derived target을 보았다 "
        "(release 집계: 0회 49, 4회 4,296). strict는 weak-training-exposed full ensemble을 제외하지만 "
        "case-UID OOF일 뿐 patient 독립성은 검증되지 않았다. 4 weak studies의 MRI fallback도 명시돼 있다. "
        "이 수치는 카드 전체 집계이며 이 976에 대한 exposure 재감사는 아니다.",
        "- 외부 lens는 JEV/Steven/Pilkwang 등 기존 ruler와 출처를 공유한다. 서로 독립된 정답으로 볼 수 없다. "
        "R7 receipt도 clean_oof=false이고 pooled-prior 노출이 있다. shareV2는 이미 이 panel에서 선택된 벡터다.",
        "- 알려진 LB는 12-target 실제 INF3 계열, 여기서는 11-target R7 proxy다. 따라서 불일치는 label lens, "
        "proxy, target 제외 또는 hidden 분포 차이에서 올 수 있다. 두 사례만으로 원인을 식별하거나 "
        "향후 LB 예측력을 추정할 수 없다. jackknife는 study 표본 변동만 반영하며 "
        "patient clustering·source 선택·훈련 노출·다중 비교를 보정하지 않는다.",
        "- 외부 .py를 읽거나 실행하지 않았다. 외부 읽기는 요청된 두 카드와 지정 CSV만으로 제한했다. "
        "GPU/Kaggle/network/credentials/보고서 원문/금지 디렉터리/서브에이전트 사용 없음.",
        "",
        "## 빌더 실행 증거와 verifier 인계",
        "",
        f"- 실제 7 lens × 7 candidate × 11 target = {result['validation']['sklearn_auc_checks']}개 AUC를 "
        f"sklearn과 대조: 최대 오차 {result['validation']['max_abs_auc_error']:.3g}.",
        f"- 핵심 3개 후보의 full12 기존 DIAG 반올림 수치 대조: 최대 차이 "
        f"{result['validation']['max_abs_diag12_rounded_error']:.3g} (허용 5.1e-6). "
        "이 점검의 full12 수치는 위 macro11 판정에 사용하지 않았다.",
        "- 등록 전후 모든 pin 일치. 예측 6파일 모두 실제 읽기/정렬/976 선택, 3개 외부 CSV 모두 "
        "완전한 panel coverage와 유한 [0,1] 확인, 4 ruler mask 및 모든 비교 경로 실제 CPU 실행.",
        "- 코드: `tools/label_lens_screen_v1.py`; 테스트: `tests/test_label_lens_screen_v1.py`. "
        "기존/frozen/scored 파일 수정 없음. 신규 파일만 추가했으므로 백업 대상 없음.",
        "- 등록·집계 JSON·로그·CPU 명령 카드: `.local/scratch/astra-b5/label-lens/`. "
        "`RESULT.json`은 target별 AUC/CI/P와 집계만 포함한다.",
        "- 독립 verifier 확인 항목: 정확한 panel/hash·11-target 선택, shareV2 및 중간 재순위, "
        ">=.5와 기존 ruler 변환/consensus mask, paired 삭제·macro covariance·방향 P·AND 규칙, "
        "등록 선행·입력 불변·노출 한계. 독립 PASS 전 의사결정에 신뢰된 결과로 사용하지 않는다.",
        "- LAUNCH.md의 pytest/register/evaluate가 이 CPU 작업의 전체 명령이다. "
        "훈련·release stage가 없어 prepare/admit은 해당 없음.",
        "",
        "SUB-AGENTS: none.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "evaluate"])
    action = parser.parse_args().action
    try:
        result = {"register": register, "evaluate": evaluate}[action]()
    except Exception as exc:
        # Third-party/trusted readers can include row values in their exceptions.
        print(json.dumps(dict(status="FAILED", error_type=type(exc).__name__,
                              details="Suppressed to protect row-level data")))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
