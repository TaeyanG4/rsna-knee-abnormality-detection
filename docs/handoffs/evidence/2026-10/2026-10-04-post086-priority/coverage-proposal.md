# 추가 시리즈 CPU 검증 제안 — 고정 K, coronal-fluid 한 슬롯

관측: 2026-10-04 06:46 KST. 작성: dicom_header_io. root의 `root-next-decision.json` 범위에서 기존 메타데이터와 소스만 읽었다. **판정은 GO_CPU_PILOT(후속 별도 CPU 실행 결정용), 학습/HOLD**이다. 이번에는 디코딩·모델·GPU·라벨·예측·새 캐시·등록을 실행하지 않았다. 실험 설계가 확정됐거나 효과가 증명됐다는 뜻이 아니다.

## 1. 확인한 입력 계약과 남은 정보

- `training/src/raptor_corpus.py`의 `pick_series()`는 CSV 순서대로 같은 plane의 첫 fluid 일치 series를 고른다. 없으면 같은 plane 첫 series로 fallback하며 이전 슬롯에서 쓴 UID는 제외한다. **FS는 원 선택 기준에 없다.** 저장 `picks.jsonl`의 실제 선택이 기준이며 v2 inventory의 replay mismatch는 없었다.
- RAPTOR64 슬롯은 Sagittal1/18, Sagittal0/14, Coronal1/12, Coronal0/8, Axial-any/12다. 원 DICOM은 IOP/IPP 투영좌표 순서(실패시 InstanceNumber 등 과거 fallback)로 정렬하고 native 수 N의 `lo=int(.02N)`, `hi=max(lo,int(.98N)-1)`에서 linspace-round한다. 따라서 **cache c−1,c,c+1은 native 연속 3장이라는 보장이 없다.** 적은 native 수에서는 동일 native index를 반복할 수도 있다.
- 선택한 full native planes에 modality LUT/MONOCHROME1을 처리하고, 전체의 2/98 percentile로 정규화·clip → 중심140mm crop → area336 → uint8가 슬롯 단위 계약이다. 서로 다른 series를 합쳐 percentile을 계산하면 안 된다. 원 builder의 결측/실패 동작을 이번 계획에서 조용히 고치지 않는다.
- `raptor_common.select_centers(within-slot-valid)`는 각 슬롯의 mask-positive triplet 중심들을 **하나의 전역 목록**으로 모은다. train은 UID/epoch RNG shuffle, eval은 전역 linspace다. 슬롯별 고정 quota가 아니다. EXP081에서도 한 study의 바뀐 mask가 K24 center 목록을 바꿨다. 슬롯 외 volume bytes 보존만으로 입력 보존을 주장할 수 없다.
- 기존 v2는 included4405/excluded2에서 ambiguity1985, slot2 ambiguity357를 기록한다. 기존 제안의 **1936/4407은 이 조사에서 원 정의/원 집계파일을 확인하지 못했다.** 어느 숫자도 본 제안의 strict same-plane/fluid/FS-unused, 현재 G1 적격/실제 처리비율로 사용할 수 없다. 새 census를 수행하지 않았다.
- EXP081의 옛 F2 868개 중 metadata-comparable63, header129series/3305장, max-span 교체34는 재사용 가능한 좁은 근거다. 현 G1 T/검증 분모와 다르며 T 교집합은 별도 reviewer/후속 metadata gate가 확인한다. EXP081의 고정가중치 최대길이 교체 결과는 본 train-time 양시리즈 질문의 답이 아니다.
- 기존 30-study/187series 물리감사는 header7538장; 선택/대안122쌍 모두 FoR가 다르다. **다른 FoR는 별개 영상의 증명도, 자동 부적격 사유도 아니다.** 두 series를 공통 좌표로 정렬했다고 주장하지 않는다. 길이는 slice-center acquisition span이며 해부구조 포함/화질/임상적 추가정보를 증명하지 않는다.

## 2. 단 하나의 제안: coronal-fluid의 절반 window만 두 번째 series로

범위를 slot2 `[32:44]` 하나로 제한한다. 다른 plane/contrast, native adjacency, K 증가, encoder/학습법 변경을 함께 넣지 않는다.

1. 원 다섯 series 선택은 저장 pick 그대로 고정한다. 원 slot2 metadata가 Coronal/Fluid=1/FS∈{0,1}인 경우에만 후보를 찾는다. 원 슬롯 중 어느 곳에도 사용되지 않았고 **Coronal/Fluid=1/동일 FS**인 series UID를 lexical 정렬하여 첫 하나만 선택한다. 그 하나가 이후 gate를 탈락해도 다음 후보를 탐색하지 않는다. span이나 angle이 유리한 후보를 선택하는 탐색은 없다. CSV중복·상충 metadata는 입력 오류로 실패한다.
2. metadata에 후보가 없거나 original이 fallback/FSunknown이면 원 입력을 유지하고 reason을 저장한다. 정상적으로 읽힌 선택 후보가 사전 geometry/duplicate gate를 탈락하면 역시 명시된 reason으로 원 입력을 유지한다. 파일 누락·DICOM parse/pixel decode 실패·예산초과를 정상 부적격으로 숨기지 않고 **job FAIL**한다. 사후 예측을 보고 후보를 바꾸지 않는다.
3. 내부 geometry는 기존 EXP081의 finite IOP/IPP/positive spacing/matrix, 단일frame·nonMOSAIC, orientation unit/orthogonal tolerance.001, consistent nonempty within-series FoR, dominant coronal normal axis, drift≤1°, ≥3 distinct positions, gaps>.001mm, maxgap/median≤1.5를 재사용한다. cross-series FoR/좌표 일치는 요구하지 않는다. 서로 다른 obliquity는 기록하되 품질점수로 후보를 고르지 않는다.
4. 중복 UID/동일 source member 집합 및 **확인된 SOPInstanceUID 교집합**은 후보를 제외한다. 최종 rendered12 planes가 원 series와 전부 동일하면 sampled duplicate로 제외하며, 12 native pick의 decoded shape/dtype/pixels 동등성도 별도 기록한다. 원시 DICOM 파일hash만 다르다는 이유로 비중복을 선언하지 않는다. 현재 저장 header에는 SOPInstanceUID가 없으므로 이 부분은 후속 bounded header 읽기 필요다. 픽셀 차이는 임상정보 차이의 증명이 아니고 near-duplicate 판정은 여전히 미해결이다.
5. 대안 series도 독립적으로 원 12-plane normalization/crop/render 계약을 적용한다. volume를 6+6장으로 접합하지 않는다. 대안 triplet은 한 series 안에서 mask가 모두 positive이고 선택 native indices가 서로 다른 center만 허용한다. 없다면 `no_valid_alt_triplet`로 원 입력을 유지한다. mask는 영상 비영값이며 병변/획득상태 라벨이 아니다.
6. 먼저 **원 cache/mask와 기존 sampler/RNG로 baseline center 목록 C를 그대로 생성**한다(trainK12, evalK24). C 중33≤c≤42인 window의 출현순번 j=0…m−1만 모아 **j=1,3,5,…**의 floor(m/2)개를 치환한다. m=0/1은 그대로다. 나머지 window·순서·다른 슬롯·padding은 정확히 유지한다. 대안용 RNG를 기존 stream에서 소비하지 않는다.
7. 대안 valid center 목록 A(로컬1…10)에서 원 c의 `q=(c−33)/9`에 가장 가까운 상대 index를 택한다: `A[round_half_to_even(q*(len(A)−1))]`. 같은 상대순서는 좌표등록/같은 해부단면을 뜻하지 않는다. 대안 mask로 baseline C를 다시 계산하지 않는다. intensity/resize224/ImageNet norm은 기존 순서·같은 UID/epoch stream을 적용한다.

이 방식은 전체 K와 원 시리즈의 최소 절반 slot2 windows를 남기면서 추가 입력을 섞는 제한된 질문이다. metadata 적격이어도 eval slot2 window가 0/1개면 처리되지 않는다. 따라서 후속엔 `metadata eligible`, `geometry eligible`, `nonduplicate`, `valid-alt`, `actual replacement windows>0`를 따로 보고해야 한다. 후보가 희소하면 효과검정력이 낮으며 넓은 1936 분모로 정당화할 수 없다.

## 3. 원 EXP084 B8을 쓸 수 있는 조건

원 B8은 동일 cohort/labels/masks/weights-init/seed/순서/학습 budget/epoch 선택 규칙의 기존 control이며, 위 치료군은 입력만 바뀌는 설계다. **그 사실을 lineage와 no-op adapter 시험으로 확인한 경우에만** 기존 B8과 한 후보 fit을 비교한다. 원 checkpoint에서 이어 학습하지 않고 동일 시작 가중치에서 동일 recipe로 시작해야 한다. 과거 완성된 B를 control로 쓰는 것이 새로운 독립 holdout을 만들지는 않는다.

최소 no-op 확인은 adapter disabled 또는 forced-original시 baseline K12(고정UID/여러epoch)/K24 center·window bytes, shuffle/intensity RNG trace가 동일함을 확인하는 것이다. 후보가 없는 study의 enabled path도 같아야 한다. 다만 bitwise input 보존은 수치적으로 동일 training 재현의 보장이 아니다.

공유 preprocessing/슬롯 quota/native adjacency/정규화/샘플러/학습 recipe를 바꿨거나 원B lineage를 결박할 수 없다면 **변경 input contract의 paired control 재학습**이 필요하다. 평가 membership를 새 geometry/개선율 때문에 축소하지 않는다. 현재 G1은 노출된 weak 개발검증이며 independent OOF/Gold/공개.943/Private/D01 개선으로 확대 해석하지 않는다. 최종 B 유효성은 `anchor_inputs` 독립 리뷰와 root가 확정한다.

## 4. 재사용 자산과 CPU pilot 제안(아직 실행 안 함)

- 원 cache/manifest/picks와 archive index를 그대로 참조한다. 메타데이터는 EXP081 `metadata-map.json`, `geometry.json`, `selection-map.json`, `overlay-receipts.json`, `noop-replay.json`을 재사용한다. `tools/exp081_coronal_span.py`는 옛 F2/weight/output에 결박되어 있어 통째로 재실행하거나 import하지 않는다. passive `raptor_dicom.py`의 명시적 adapter/source를 후속 구현에서 검토한다.
- 옛34 overlay 중 **같은 UID/series/native pick/render recipe/source identity**가 일치하는 대안은 향후 출력 재사용 가능하다. 기존 maximum-span 선택이 이번 lexical 선택과 같다고 가정하지 않는다. 이번에는 arrays/volume/weights를 읽지 않았다. baseline replay에는 원 slot2 cache가 참조값이 되지만 새 decoder가 원본을 정확히 재현하는지는 실제 bounded 시험이 필요하다.
- 새 CPU pilot은 **G1 train-only**와 기존 header-audited strict-eligible 쌍의 교집합에서 UID lexical 순 최대8 study를 먼저 freeze한다. 최소4개가 확보되지 않으면 root에 HOLD(더 넓은 metadata 조회 결정), 사례를 labels/prediction으로 보충하지 않는다. 이 선정은 engineering fixture이며 prevalence/효능 표본이 아니다.
- 그 후보를 고정한 뒤 original+alternate 각12장, 최대192 plane source reads를 수행한다. SOP/정렬 확인을 위한 필요 header만 최대2048개/64MiB parser bytes. wall10분, raw returned pixel payload1GiB, peakRSS4GiB, rendered uint8 payload최대21,676,032B(16series×12×336²), 새 artifact directory로 기록한다. ZIP compressed/block traffic은 parser/payload bytes와 분리한다. 예산초과는 FAIL, 임의 fixture 교체/자동 재시도는 없다. CPU worker 상한12 안에서 메모리/실제할당을 먼저 확인한다.
- 기록할 항목: frozen UID/T membership digest, metadata+source/version bindings, all header/SOP geometry reasons, selected native paths/indices, raw and rendered hashes, codec/transfer syntax, pixel/render walltime/RSS/bytes, original replay exactness, full-series duplicate unknown과 sampled duplicate를 구분, valid center 및 실제 교체수, train/eval no-op 비교. GPU/model/label access=0을 receipt에 기록한다.
- prototype adapter/runner와 frozen fixture 아직 없다. 구현·검토 예상30–60분(추정), pilot 실행상한10분. 옛 EXP081은 header4.33초, 전체준비21.68초/1164 pixel reads/891.8MB였으나 이 기계의 새 계약/full-cache 시간 보장은 아니다. **현재 추가디코딩 비용은 미측정**이다.

대안 캐시 한 study는1,354,752B(+mask/receipt)다. 전체는 `strict eligible N × 1,354,752B`; current G1 N이 없으므로 전체 시간/GB를 수치 확정하지 않는다. 기존 B급 full43.5분은 향후 후보1fit 계획근거일 뿐이며 fulldecode·실제 T4 처리량·재현 gate 전에 GPU 진입하지 않는다.

## 5. 테스트 환경과 다음 결정

대회 test에는 Report가 없다. 위 선택은 series CSV + raw DICOM headers/pixels만 사용하며 train supervision과 분리된다. test-time에서 original64와 추가12 planes의 decoder·SOP/geometry gate를 실행할 필요가 있다. 모델 K는 같아도 디코딩은 증가하며 host codec/resize·실제 raw DICOM 예외·9h공동예산은 별도 검증 대상이다. 기존 D01 FAIL/PASS/version과 T4/4070 수치계약을 소급 재사용하지 않는다.

root의 다음 선택은 **bounded CPU pilot 구현/실행 여부**다. GO 근거는 좁은 기존 header 자산·작은 byte상한·baseline 보호 가능한 계약이며, 임상 추가정보/실제 coverage/성능 기대를 이미 확인했다는 뜻이 아니다. pilot 후 (a) 원 replay/no-op 불일치, (b) 대부분 sampled duplicate 또는 처리창 없음, (c) 비용 비현실적이면 HOLD한다. 별개의 수치·lineage 검토와 고정 평가계약을 통과해야 이후 학습 투자 결정을 할 수 있다. M02/M03/M05/M07/M10/Biohub의 정보누락·공정비교·작은이득대비투자·과거실패 범위 제한을 적용했다.

근거 경로/과거 SHA와 잔여 의존성은 `coverage-proposal.json`. 새 full census/ZIP hash sweep 없음. 참고 탐색 중 PowerShell literal wildcard 경로 오류와 과대한 rg 출력은 좁은 명시경로/필드읽기로 교정했으며 분석값으로 사용하지 않았다.
