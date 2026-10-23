# 새 group 대조 실행안 · 2026-10-04

상태: **PROPOSAL_ONLY_NOT_REGISTERED**. CPU 문서·소스·실행 메타데이터 검토다. 새 T/V 배정, label 원자료·Gold·예측값·pixel 열람, 학습, 다운로드, 원격 실행은 하지 않았다. 이 제안은 학습 큐나 실행 허가가 아니다. 기계 판독용 계약과 근거 hash는 [proposal.json](proposal.json)에 있다.

**EXP082의 실제 주 질문을 유지하려면 새 RN clip20, RN ones20, 공통 B8의 3fit을 권고한다.** 같은 새 V에서 `90% B + 10% RN ones`가 `90% B + 10% RN clip`보다 나은지 한 번 비교한다. RN 두 fit만 수행하면 RN 단독 loss 효과라는 다른 질문에 답한다. 2fit 뒤 V 점수를 보고 B를 추가하는 방식은 사용하지 않는다. root가 실행 전에 목적·2/3fit·예산을 선택한다.

## 얻는 정보와 남는 공백

EXP082에서 관측한 blend +0.000534(CI에0 포함), RN 단독 +0.006828은 작은 후보로 보존한다. 새 대조는 알려진 중복 component를 묶고, ImageNet에서 새로 시작한 영상 모델의 직접 학습에서 V를 제외한 상태로 같은 loss 가설을 확인한다. 기존 B가 새 V 일부를 학습했을 가능성이 있으므로 기존 checkpoint를 재사용해서는 이 질문에 답할 수 없다.

이는 untouched holdout이나 환자 독립 검증이 아니다. 과거 연구자의 표본 노출, 보고서 weak 정답, 미확인 환자·근접중복, 한 seed의 변동은 남는다. Dread 외부 report-label 생성기의 prompt/교정/LLM 계보도 V·Gold와 완전히 독립이라고 인증하지 않았다. 새 image teacher·pseudo-labeler·calibrator는 도입하지 않는다. 따라서 **통제된 영상 학습 계보의 재대조**로 보고하며, exact D01(public anchor+옛 R2), 공개0.943의 전이, 영상 재판독/private 성능을 검증했다고 쓰지 않는다.

## 실제 확인한 group·입력 계약

- 기존 eligible 4,343 studies, report hash 4,197개, connected component 4,196개. report hash 동일성과 확인된 duplicate edge 1개를 전이적으로 묶는다. 최대 component는37 studies다. 기존 F0/F1을 가로지르는2-study component가1개 있으며 F2에서는 모두 train이다.
- `VALIDATION_GROUP_CONTRACT_V1`의 groups/components/known-edges/ancestor-exclusions 크기와 SHA256이 저장 계약과 일치했다. group은 존재하지만 새 split은 없다. 기존 Gold·report 중복 quarantine과 손상2건 제외를 유지하며 Gold 파일을 열어 제외목록을 다시 만들지 않는다.
- 분할 제안은 component ID별 `SHA256(UTF8("RN_GROUP_SCREEN_V1:" + lowercase_hex_component_id))`를 unsigned big-endian 정수로 읽어 `%5 == 2`이면 V, 나머지는 T다. **이번에는 실행하지 않았다.** component 약20%라는 설계일 뿐 V study 수나 class 비율을 보장하지 않는다. future registration에서 T/V UID를 각각 문자열 오름차순으로 고정하고 hash를 남긴다. historical fold·residual·label·scanner·점수를 보고 salt를 바꾸지 않는다.
- 고정 Dread soft label을 학습에 쓰고, 평가 reference는 기존 `float32 → numpy.rint` 12-target 이진화다. 새 T/V 크기·P/N·pos_weight는 이번에 계산하지 않았다. 등록 후 single-class target이 있으면 macro12 invalid로 남기며 조용히 salt를 다시 뽑지 않는다.
- corpus는 `RAPTOR64_336_C140_S02`, slots18/14/12/8/12·depth64·336px cache·crop140mm·span .02–.98을 유지한다. `raptor_corpus.py:171–194`의2–98 percentile 정규화는 해당 series의 선택 slice만 사용한다. 점검한 경로에서 전체 study에 fit하는 정규화는 없었다. 이를 영상 모델의 학습 계보와 구별한다.

## 최소 실행 레시피

|조건|RN clip20 / RN ones20|공통 B8|
|---|---|---|
|초기 구조|resnet34.a1_in1k|coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k|
|고정 epoch|20 / 20|8|
|backbone / head LR|5e-4 / 1e-3|3e-5 / 1e-3|
|loss|새 T의 clipped positive BCE / positive weight 모두1|새 T의 clipped positive BCE|
|gradient checkpointing|false|true|
|init full-state SHA prefix|093be508bda4f14b… (두 RN 동일)|e4ea3ae625eacde8…|

공통:224px, train K12/eval K24, within-slot-valid, ImageNet norm, seed42, train batch8/accum1/drop_last=True, eval batch4, AdamW WD.02, OneCycle pct_start.15, gradient clip3, strict FP16 train/eval·scale1024, successful optimizer update에만 scheduler step. intensity jitter .20(배율.9–1.1), UID/epoch/seed 기반 고정 sampling을 유지한다. fixed-final만 저장·평가하고 epoch/SWA/Gold/LB 선택은 하지 않는다. attempted update 수는 `epochs × floor(n_T/8)`이며 옛8,680을 그대로 강제하지 않는다. skipped update·실효 update·학습률 종료점을 보고하며1% 이상 skip은 검토, 연속10 skip·nonfinite·계약 오류는 중단한다.

CPU worker는 사용자 기본 **12**, 실제 allocation·RAM/I/O·동시 작업·내부 thread 효율에 따라10→8 또는 낮춘다. paired RN은 점수 보기 전에 같은 effective worker를 확정한다. historical RN4/B6는 시간 근거이지 새 기본값이 아니다. 평가 worker는 기존 `min(train_workers,2)`다. 로컬 학습은 순차1개다. 00:11경 host snapshot은8 physical/16 logical CPU, RAM 약95GiB,4070TiSUPER16GiB였으며 자원 예약을 의미하지 않는다.

RN ImageNet revision/file SHA와 세 historical run의 초기 full-state hash가 계약과 일치한다는 기록을 확인했다. B는 초기 full-state hash가 일치하지만 읽은 EXP078/080 계약에는 timm 원격 asset의 정확한 revision/file SHA가 없다. 향후 시작 전 실제 cached asset provenance를 묶고, 모델 학습 전 full-state hash를 검증해야 한다. 이미 학습된 B/RN checkpoint로 대체하지 않는다.

## 구현해야 할 최소 부분

`raptor_train.py:103`은 Gold를 읽고 고정 epoch 실행도 마지막에 Gold를 평가한다. 그대로 실행하면 no-Gold가 아니다. `tools/exp082_r34_loss.py`의 source renderer는 Gold 접근을 제거하지만 옛 F2 UID와 ones loss를 고정하므로 새 V/baseline에 그대로 쓸 수 없다.

1. 새 별도 protocol/adapter에서 explicit T/V와 clip/ones를 받게 하고, shared no-Gold 경로·fixed-final·현재 입력/seed/runtime을 유지한다. 원본4 source와 EXP078~083 frozen protocol은 수정하지 않는다. 차이와 source hash를 기록한다.
2. T/V component 불교차·전체 eligible 일치·init state·T-only weight·동일 RN 초기값/sampling·Gold 미접근을 작은 CPU/synthetic 검사로 확인한다. 전체 DICOM 재감사나 새 framework는 만들지 않는다.
3. 모든 선택 arm이 완료되기 전 V 점수를 노출하지 않는다. 최종 고정 예측에 대해 등록된 evaluator를 한 번 실행한다. 옛 F2를 고정한 report/adapter를 우회 재사용하지 않는다.

## 평가와 판정

3fit primary는 `macro12 AUC(.9 rank(B)+.1 rank(RN_ones)) − macro12 AUC(.9 rank(B)+.1 rank(RN_clip))`다. 각 target의 전체 V에서 average tie rank/N을 계산하고 final rerank를 하지 않는다. RN 단독 차이와 각 blend−B는 secondary다. 2fit을 택하면 primary 자체를 RN 단독 차이로 미리 바꾸고 blend 확인이 아니라고 기록한다.

paired connected-component bootstrap500회·seed20261003, 모든 arm/target에 같은 multiplicity를 적용한다. 전체 V의 rank/blend를 먼저 고정하고 subgroup/bootstrap에서 rerank하지 않는다. 한 target이라도 single-class인 draw는 macro12 invalid로 세어 attempted/valid/invalid를 보고한다. 12소견 P/N·AUC·Δ·Δ/12, Fracture/PFOA/MCL과 기존 scanner/mask group의 표본/양음성 수를 유지한다. 작은 group과 탐색적 분해를 새로운 확정 가설로 쓰지 않는다.

작은 gain을 임의+.002선으로 폐기하지 않으며 CI의0 포함은 동등성이 아니다. 양의 CI도 private/임상 개선 증명이 아니다. 반복된 Fracture 손실을 숨기거나 사후 target routing/비율 변경으로 구제하지 않는다. 현재3arm에는 입력 A가 없으므로 B−A 골절 손실을 해결했는지 직접 확인할 수 없다. fixed ratio/epoch를 유지한 채 effect·CI·소견/그룹 위험·실행비용을 root가 검토하고 추가 fold/seed/384 확장은 자동 실행하지 않는다.

## 실측과 계획 비용

독립 메타데이터 감사 관측:2026-10-04 00:10:46 KST. 모두 historical T3475/V868,4070TiSUPER,torch2.5.1+cu121,timm1.0.19다.

|run|summary total|job wall|학습시간 합|
|---|---:|---:|---:|
|EXP080_R34_B_F2|772초|778초|731초|
|EXP082_R34_ONES_F2|691초|698초|662초|
|EXP080_INPUT_B_F2|2,502초|2,507초|2,451초|

RN2의 실제 wall 합24분36초, B 포함3fit은66분23초였다. 두 RN은 각각8,680 optimizer updates/0 skip, B는3,472 updates/0 skip을 기록했다. 계획은 **RN2 26–36분 / 3fit 66–82분**을 유지하되66분은 확정 ETA가 아니다. 새 T/V 크기·worker·cache 상태에 따라 갱신한다. 학습 외 차이는 평가/저장 등 혼합 overhead이며 순수 평가시간이 아니다. ones가69초 빨랐던 것을 loss의 속도 효과로 단정하지 않는다. 별도 adapter·등록·분석20–40분은 미실측 추정이며 B asset 회수나 구현 문제가 생기면 늘 수 있다. free Kaggle/Colab·유료비용은 이 로컬 대조에 포함하지 않는다.

이번 목적에는3fit이 질문 정합성이 높고, RN2는1시간 미만 비용으로 standalone loss 방향만 보려는 경우에만 선택할 가치가 있다. 입력 확대는 EXP081 특정 max-span 미개선으로 전체 기각하지 않고, 표현은 EXP083 특정 head 개입 실패로 전체 기각하지 않는다. Silver 교정은 현재 pilot에서 자동수정 근거가 부족하므로 이번 대조에 섞지 않는다. exact 후보 serving은 별도 작업이다. 새 대조의 장점은 이 넓은 방향들을 대신한다는 데 있지 않고, 작게 관측된 loss 효과를 원래 blend 질문대로 재확인할 수 있다는 데 있다.

## 인계

- 완료: 기존 group/recipe/source/timing 계약을 읽고 실행안 두 파일 작성. split·원라벨·Gold·prediction·pixel·GPU를 열거나 실행하지 않았고 기존 source/protocol을 변경하지 않았다.
- 근거: [proposal.json](proposal.json)의 source SHA·historical receipt와 범위. 00:20:16 KST에 JSON parse와24개 source binding의 존재/SHA, 원본4 source·group output hash 일치를 확인했다.
- 미해결: B 초기 asset 정확 revision/file SHA, 새 V 크기/P/N, Dread label-source 완전 계보, 실제 fresh-run runtime. 모두 완료로 주장하지 않는다.
- root 다음: 목적/2·3fit/비용 선택 → 별도 최소 adapter·실제 split 사전등록과 시작 전 검증 → 허가 범위에서 detached queue·ID/log/ETA·같은 채팅 heartbeat(ETA20분 전)로 실행. 이 준비 작업은 예약을 만들거나 기존 D01 결과를 조기조회하지 않았다.
