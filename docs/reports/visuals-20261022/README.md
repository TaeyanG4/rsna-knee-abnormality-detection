# Research visuals — 2026-10-22

[English report](../project-summary-20261022.md) · [한국어 보고서](../project-summary-20261022.ko.md)

Original diagrams and aggregate-data charts for the **2026-10-22 18:15 KST**
pre-result snapshot. They explain existing records; they do not update scientific
verdicts or imply later outcomes. All illustrations are made from code and simple
shapes. No patient MRI, clinical photographs, labels, report text or model files
were read or embedded. The slice icons are conceptual, not anatomical reference images.

## 1. Study to prediction

![Conceptual study, rendering and 12-target output flow](study-to-prediction.png)

An MRI study contains multiple series/slices. The diagram shows input preparation,
registered model branches, rank fusion and one 12-target output per study. Shared
R6/K1 rendering applies only where registered. It is not an exact trace of every candidate.

한국어: 여러 MRI 시리즈를 처리해 모델별 예측을 만들고 등록된 방식으로 결합하여 검사당
12개 값을 출력하는 흐름입니다. 슬라이스 그림은 실제 영상이 아닌 도형 아이콘입니다.

## 2. Candidate structure

![Candidate family components and registered combinations](candidate-family-map.png)

Percentages summarize nested rank fusion, not direct probability averaging.
The full rank formulas are in report §3. The submitted X uses stride 3, whereas
the `X_u` component of U uses stride 2. K4 adds the full-bag member to K3.

한국어: P·O·K3·K4의 차이와 후보별 고정 조합입니다. U의 내부 `X_u`와 제출된 X를 구분하고
K4는 기존 모델 교체가 아니라 네 번째 모델 추가라는 점을 표시했습니다.

## 3. Recorded public displays

![Recorded public displays, with unrecorded outcomes kept separate](public-score-snapshot.png)

| Candidate | Supplied public display | Source |
|---|---:|---|
| X | 0.958 | D-B-249/260/276 |
| p25 | 0.958 | D-B-260/276 |
| F / P25F | 0.957 | D-B-274 |
| A5 / D | 0.956 | D-B-275/276 |
| DF | 0.955 | D-B-280/281 |

Dot plot with an explicitly restricted 0.950–0.960 axis. No statistical interval
or unrounded score is available. U/P4/U4 have accepted refs but no recorded outcome
at cutoff; F4 has no accepted ref in this snapshot. No leaderboard was queried.

한국어: 반올림된 공개 표시값을 그대로 사용했습니다. 동점이 비공개 성능의 동일함을 뜻하지
않으며 미기록 후보를 0점으로 처리하지 않았습니다.

## 4. Full-bag comparison

![Full-bag replacement and addition have different local results](fullbag-local-evidence.png)

| Registered contrast | Mean change across four rulers | Source |
|---|---:|---|
| Full-bag replacing capped seed0 | approximately -0.0045 | D-B-264, D-A-052 |
| K4 (K3 + full-bag) versus K3 | +0.001449016 | D-B-278 |

The addition comparison used the identical 977-study fold-0 holdout, stride 3,
with JEV +0.000848, Aquila +0.001585, Dread +0.002005 and Consensus +0.001358.
The plot uses the rounded +0.001449. The bars have distinct controls; no confidence
interval is inferred. These are K1-only local effects, not measured final-blend
or private gains. D-B-278's independent result replay does not close its pending
different-host discovery cross-check.

한국어: 교체는 평균 약 -0.0045였지만 네 번째 멤버 추가는 +0.001449였습니다.
두 실험의 대조군이 다르며 반복 사용된 로컬 기준의 결과를 최종 비공개 점수로 환산하지 않습니다.

## Reproduction and asset integrity

Run `python tools/render_research_explainer_20261022.py` from the project root.
Requires Matplotlib, NumPy and Pillow. Windows Malgun Gothic is used when available
for Korean glyphs; another platform must provide a Hangul font for equivalent output.
No network, image-generation API or data/model loading is used. The renderer writes
four PNGs and [asset-manifest.json](asset-manifest.json) with file hashes and provenance.
PNG metadata is stripped by pixel-only re-encoding. Numerical sources remain the
original discoveries and the versioned report.
