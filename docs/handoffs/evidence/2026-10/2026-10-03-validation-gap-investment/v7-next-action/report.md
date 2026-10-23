# 2026-10-03 21:43 KST · 다음 연구 선택

**권고는 B: 고정 EXP080 B F2 encoder에서 slot-aware head와 동일한 재학습 control을 한 번 비교하는 개발 screen이다.** EXP082의 작은 양의 효과를 보존하되 곧바로 새 split에서 같은 loss 가설을 반복하기보다, 현재 코드에서 분리할 수 있는 다른 표현 가설을 확인한다. A는 유망 레시피를 확대하기 전의 후속 grouped 대조로 남긴다. 이는 B가 더 정확하거나 전체 시간이 더 짧다는 예측이 아니다. 본 문서는 설계만이며 코드/학습/데이터/점수를 실행하지 않았다.

## 근거와 정보가치

기존 EXP082 보고서의 주 효과는 +0.000534467, 조건부95%CI[-0.000268009,+0.001293080]이고 RN 단독은 +0.006828056이다. 작은 효과를 무효화하지 않는다. 다만 혼합의 유용성은 아직 불확실하고 B의 A대비 골절 손실을 해결한 근거도 없다. RN 단독 결과가 더 크다는 이유로 주 비교를 바꾸지 않는다.

`raptor_common.py:239–270`의 head는 각 창 특징에 동일 LayerNorm/attention을 적용한 뒤 창 전체 softmax/pooling을 한다. 명시적인 plane/slot ID는 없다. 이미지 특징이 이미 이를 암묵적으로 표현할 수 있으므로 정보가 없다고 단정하지 않는다. `raptor_data.py`는 `within-slot-valid` 창을 만들지만 head에 slot ID를 넘기지 않는다. 따라서 고정 encoder에 작은 slot 조건을 주는 대조는 새 backbone 학습 없이 확인 가능한 가설이다. 잃어버린 영상이나 encoder가 보존하지 않은 골절 정보를 복원하는 실험은 아니다.

|안|추가로 얻는 정보|구현/검토 비용|로컬GPU 비용|중요 한계|
|---|---|---|---|---|
|A: 새 group split RN clip/ones 두fit|다른 고정 개발 분할에서 loss 효과 방향이 반복되는가|20~40분 예상; 기존 no-Gold adapter의 split 계약 분리 필요|26~36분 추정|RN 단독 비교다. EXP082 주 비교인 B+RN 개선을 직접 재검증하지 못함|
|A+새 B8|새 분할에서 같은 B를 공유하는 두90/10 조합 비교|위 준비에 B no-Gold 계보/초기값 검토 추가|총66~82분 추정|더 적절한 확인이지만 한seed·노출weak·미확인 중복/라벨 문제는 그대로|
|B: 고정B encoder+두head|같은 영상표현에서 명시적인 slot 조건이 pooling에 도움이 되는가|60~120분 예상; feature/slot 연결·head 대조 구현 필요|특징추출10~30분+두head5~15분 잠정|구현시간은 A보다 클 수 있음. 기존 F2 screen이며 새 독립검증 아님|

시간은 기존 RN691~772초/B2502초와 v5/v6 예산을 이용한 추정이다. feature 추출/head 학습은 아직 실측하지 않았다. **B는 새 정보의 종류 때문에 우선**하며 GPU시간만으로 싸다고 판단하지 않는다. 구현120분 또는 GPU45분의 제안 예산을 넘길 전망이면 범위를 늘리지 말고 산출물/실측을 보존해 재판단한다.

## B의 구체 최소 계약

1. **사례와 계보:** EXP080 B F2 fixed8 checkpoint·arch·state/파일SHA, train3475/eval868의 원래 ordered UID·입력/라벨/제외집합을 고정한다. `VALIDATION_GROUP_CONTRACT_V1`의 알려진 F0↔F1 component는 양쪽 모두 train에 남는다. 이 계약이 현재 F2 경계를 가로지르지 않는다는 범위만 확인하며 환자독립/untouched 주장은 하지 않는다. F2로 새 split을 골라 기존 B를 재사용하지 않는다.
2. **feature 추출:** backbone은 `eval()`·`requires_grad=False`, BN/stat/dropout을 고정하고 epoch내 갱신하지 않는다. 두head가 **하나의 동일 cache**를 사용한다. 기존 B eval 경로의224px/K24/within-slot-valid/ImageNet norm/strictFP16/배치4를 그대로 써 `encode(x)` 출력과 실제 center/slot ID를 저장한다. train도 동일 고정K24를 사용하며 stochastic K12/jitter를 끈다. 이는 두arm에 공통인 새 head 재학습 조건으로 명시하고 기존 end-to-end B 학습과 같다고 부르지 않는다. 원래 FP16 encode값을 그대로 저장하며 추가 양자화는 하지 않는다. cache 크기는 `4343×24×feature_dim×2bytes`이며 feature_dim은 실제 checkpoint에서 확인한다.
3. **slot 정의:** center가 속한 실제 cache 구간 [0,18),[18,32),[32,44),[44,52),[52,64)에 따라 sagittal-fluid/sagittal-other/coronal-fluid/coronal-other/axial의5 ID를 준다. triplet 전체가 해당slot 안이고 원래 mask-valid인지 확인한다. K24 순번을5등분해서 slot을 추정하면 안 된다. 반복 center도 원래대로 둔다. `center=-1`은 별도 empty ID이며 bias는 고정0; 기존 blank 입력을 새로 제거하거나 softmax-mask하지 않는다. slot별 quota/창선택을 바꾸지 않는다.
4. **단일 개입:** 기존 `att(LayerNorm(features))`의12소견 logits에 `slot_bias[slot,target]`를 더한 뒤 기존 softmax/pooling/classifier를 유지한다. trainable bias는5×12=60개, 모두0 초기화. feature에 embedding을 더하거나 새value projection·position encoding·missingness gating까지 동시에 넣지 않는다. 이는 slot-aware head의 가장 작은 버전이며 slot embedding 전체 가설의 검증은 아니다.
5. **공정 control:** 두head의 LayerNorm/attention/classifier는 **같은 B8 head weights에서 출발해 둘 다 다시 학습**한다. control은 slot bias를0에 고정하고 treatment만 학습한다. 초기 logits 동등성 확인, 동일 minibatch 순서/seed42·dropout RNG·dtype·loss·step budget을 사용한다. 기존 saved head만 control로 삼지 않는다. treatment의 추가60params와 slot정보 효과를 완전히 분해하는 비교는 아니며, capacity만으로도 설명 가능한 부분은 남긴다.
6. **head 최적화 초안:** fixed20epoch, batch64/drop_last=False, AdamW LR.001/WD.02, OneCycle pct_start=.15, gradient clip3, dropout.2, strictFP16/scale1024를 두arm에 동일 적용한다. 새head용 스케줄이며 B8 원학습의 복제라고 부르지 않는다. 원래 trainsoft label 기반 clipped positive BCE weight를 공통으로 사용한다; 아직 미채택인 ones loss를 섞지 않는다. 동결 feature에서 finite/skip/update 수를 기록하고 no-Gold를 유지한다. epoch/학습률을 F2 점수로 고르지 않는다.
7. **평가:** primary는 같은 F2의 `macro12 AUC(slot-bias 재학습 head) − macro12 AUC(control 재학습 head)` 한 개다. 원B8과의 각각 차이는 재학습 공통효과를 보기 위한 secondary이며, B+RN/anchor 혼합비율을 이번에 탐색하지 않는다. 12소견/P/N·macro 기여·골절/MCL/PFOA·기존 mask/scanner군을 함께 보고, component 단위 paired bootstrap500회/seed20261003을 사용한다. 이미 고정된 예측을 resample하며 CI는 weak/노출/한seed 조건부다.
8. **범위 종료:** 두head가 같은 cache와 초기 shared-state를 쓰는지, encoder state 전후불변·slot/center 계약·empty처리·초기logit동등성만 최소 확인한다. 새 full-DICOM 감사는 하지 않는다. 실패한 계약을 먼저 고치고 한 번의 고정 평가 후 종료한다. 작은 양의 효과를 컷으로 폐기하지 않고 CI0포함을 동일성으로 해석하지 않는다. 음의 결과는 이 제한된 head 개입 미채택이며 모든 slot/coverage/표현 전략의 기각이 아니다. alltargets 양수/골절 단독veto·사후routing은 없다.

## A를 실행할 때 필요한 최소 계약

4196개 component는 **그룹계약일 뿐 새 split이 아니다.** 다음 A를 택하면 먼저 UID/component별 split을 실제로 고정하고 기존 folds.csv를 보존한다. 예를 들어 고정 salt `RN_GROUP_SCREEN_V1`와 component ID의 SHA256 정수 mod5==2를 V로 삼는 단일 결정적20% 배정이 가능하다. 이는 제안 규칙이며 이번에 배정하거나 class통계를 계산하지 않았다. 결과/점수를 보고 salt·V를 다시 뽑지 않는다. 단일class 소견이 생기면 macro12는 invalid로 남기고 별도 설계 변경을 기록한다.

두 RN은 같은 ImageNet 초기값에서 T만 사용해 clip/ones로 각각 fixed20 학습한다. 원래Gold/손상제외·reference labels·orderedUID·RNG·optimizer/runtime을 두arm 간 고정하고 class weight는 새T에서만 계산한다. primary를 RN 단독 차이로 등록하면26~36분 대조가 가능하지만 EXP082의 blend primary와는 다르다. **blend primary를 유지하려면 새V를 본 기존 B checkpoint를 재사용하지 말고 T로 B8도 새로 학습해야 한다.** 모든V component는 direct training과 competition-finetuned 초기weights/teacher/pseudo-label ancestor에서 제외돼야 한다. 이 조건도 미확인 환자연결·보고서정답 오류·과거 연구자노출을 없애지는 않는다.

A/B 중 어느 것도 exact D01의 anchor+옛R2 계보나 private/LB 개선을 증명하지 않는다. 다음 단계는 B의 bounded 구현·계약 검토 여부를 root가 정하는 것이며 자동GPU/새split/후속fit 승인으로 읽지 않는다.
