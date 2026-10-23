# PFOA 오류 보완: 추가 학습 전 최소 검증 — 2026-10-03

2026-10-03 15:57 KST 검증 완료. **ResNet에는 R1과 다른 weak-label 순위 오류와 PFOA 이점이 있지만, 전체 모델 교체 근거는 없다.** EXP078의 네 fit과 원래 입력 A/B 판단을 먼저 마치고, 선택한 기준 모델에 저장 ResNet 예측을 한 번만 섞는 CPU 선별을 권한다. 유망할 때 현재 strict 조건의 ResNet 두 fit을 별도로 검토한다. 이번 작업에서는 혼합 점수 계산·GPU 학습·Gold 평가·제출을 하지 않았다.

## 확인한 증거

동일 4,345 study·4,199 report group·12소견, 기존 Dread 라벨의 float32→`np.rint` 규칙과 원래 fold를 사용했다. R1/ResNet 각각 다섯 `oof_best.csv`만 비교했다. ResNet summary의 SWA 선택값과 섞지 않았다. 기존 감사의 원본 22개 SHA와 UID/fold/열 계약을 강제했다. 교집합 축소나 누락 모델 대체는 없다.

| PFOA | R1 AUC | ResNet AUC | 차이 |
|---|---:|---:|---:|
| F0 | 0.850329 | 0.858219 | +0.007890 |
| F1 | 0.814556 | 0.814568 | **+0.000012** |
| F2 | 0.812231 | 0.834746 | +0.022515 |
| F3 | 0.816587 | 0.836995 | +0.020408 |
| F4 | 0.819686 | 0.837605 | +0.017919 |
| pooled | 0.821988 | 0.833992 | +0.012005 |

PFOA 양성1,459·음성2,886의 4,210,674쌍 중 R1이 틀리고 ResNet이 맞힌 쌍은276,060, 반대는225,512, 둘 다 틀린 쌍은473,491이다. R1 오순위 쌍의36.83%를 ResNet이 올바르게 순위화하지만, 새 오류도 상당하다. 전체쌍의11.91%에서 엄격한 순서가 반대이며 Spearman은0.8664이다. **이 수치는 달성 가능한 ensemble 이득이나 oracle 성능이 아니다.** 같은 study를 공유하는 쌍들은 독립 표본이 아니다. F1 이점은 순위쌍 두 개 차이에 불과하므로 ‘다섯 fold에서 확실히 개선’이라고 해석하지 않는다.

반면 pooled 다른11소견 모두 ResNet AUC가 낮다. 전체 macro는0.878263→0.866471(**−0.011792**), 다른11 평균은−0.013955이며, 모든 fold에서 전체 macro/다른11 평균이 낮다. 개별 소견의 모든 fold가 음수인 것은 아니다. 예를 들어 Medial OA는4/5 fold 양수인데 pooled는 음수여서 fold 간 점수 분포 영향도 있다. PFOA만 +0.012를 보존해도 macro 기여는 약+0.001이므로, 모든 새 가설에 +0.002를 요구해 작은 유용한 개선을 버려서는 안 된다. 현재 결과는 PFOA routing의 채택 근거도 아니다.

손상 study 두 개를 제외한4,343건에서도 PFOA 차이는+0.012005로 거의 같다. 이 민감도 확인이 과거 학습의 입력 문제를 고치는 것은 아니다. 기존 conditional report-group bootstrap 구간[+0.00365,+0.02064]은 고정 예측의 보조 근거일 뿐, epoch 선택·다중 탐색·weak 정답·seed/환경 불확실성을 포함하지 않는다. Report group의 환자 독립성도 확인되지 않았다.

## 구조 효과로 단정할 수 없는 이유

| 조건 | 과거 R1 | 과거 ResNet |
|---|---|---|
| backbone | CoAtNet rmlp2, in12k→in1k | ResNet34, a1_in1k |
| backbone LR / epoch | 3e−5 / 8 | 5e−4 / 20 |
| 저장 best epoch(0부터) | 전부7 | 14,8,14,14,14 |
| 장치·AMP | 로컬4070 Ti SUPER·BF16 | Kaggle T4·FP16 |
| torch / timm | 2.5.1+cu121 / 1.0.19 | 2.10+cu128 / 1.0.26 |

224px, k12/eval24, batch8, head LR0.001, wd0.02, seed42, ImageNet norm은 같다. 그러나 사전학습·학습률·학습길이·checkpoint 선택·수치환경이 달라 **encoder와 학습 레시피의 관측 차이**다. 축상면/연골 표현이 원인이라는 주장도 아직 근거가 없다. T4↔4070의 별도 R2 재추론 비교는 이 ResNet 학습 차이를 제거하지 않는다.

## 다음 한 가설과 단계

가설: **현재 입력 기준 모델에 ResNet 레시피의 순위를 소량 고정 혼합하면, 12소견 평균 성능을 해치지 않으면서 PFOA 오류를 일부 보완한다.** 현재까지 비율 성능을 계산하지 않았으며 아래 비율은 계산 전에 고정하는 제안이다.

1. **GPU 0, 저장 예측 한 번 선별.** EXP078 네 fit 완료 후 원래 gate로 입력 arm을 결정한다. 이 결정에 PFOA/ResNet 결과를 사용하지 않는다. 선택 arm F0/F1과 과거 ResNet F0/F1의 정확한869 UID씩을 맞추고, 각 fold·소견 안에서 평균 동순위 percentile rank로 `0.90×baseline + 0.10×ResNet`을 계산한다. 12소견 모두 같은 비율이며 PFOA routing/다른 비율 검색은 하지 않는다. fold별 macro/PFOA/다른11 평균/12개 차이를 한 번 기록한다. 예상 구현·확인10~20분, 계산 수초. 과거 ResNet의 best 선택과 구환경을 재사용한 **개발 선별**이며 통제 효과 추정이 아니다.
2. **조건부 새 ResNet 두 fit.** CPU 결과가 다음 투자 근거를 주면 F0/F1에 현재 선택 입력·strict FP16·full OOF·원래 Dread 평가를 적용한다. ResNet34.a1_in1k, LR5e−4/head0.001/wd0.02, seed42, **고정20epoch**의 한 레시피를 사전 기록한다. best/SWA/Gold 선택은 하지 않는다. 고정20은 과거 best와 다른 평가 조건이므로 결과가 낮아질 수 있다. baseline8epoch와 candidate20epoch 비교는 순수 구조 효과가 아닌 **실용 레시피 보완 효과**다. 위90/10을 유지하며 다시 비율을 맞추지 않는다.
3. **판단·중단.** 두 fold를 모두 본 후 판단한다. 두 fold macro 양수와 PFOA 평균 양수이면 작은 이득도 다음 검토 후보로 남기되, −0.01 이하 소견 하락·다른11의 반복 손실·그룹 편차·조건부 CI를 함께 검토한다. 작은 혼합 부호는 불확실로 보존하고, 두 fold macro 손실 또는 광범위 손실이면 해당 레시피 확대를 보류한다. 이 경우 비율 조정으로 결과를 구제하거나 표현/라벨 전략 전체를 기각하지 않는다. 중간 성능 조기중단은 없으며 NaN/Inf·계약/hash 오류·unsupported AMP는 즉시 중단, optimizer skip>1% 또는10연속은 수치 검토 대상으로 한다. 추가fold/seed/학습·제출은 자동 배정하지 않는다.

CPU 선별에서 기존 ResNet이 실패해도 strict 새 레시피의 실패를 증명하지 않는다. 다만 제한된 GPU 시간에서는 series 선택/감독신호 등 다른 구체적 가설과 투자 가치를 다시 비교할 이유다. F0/F1은 이미 노출된 개발 fold이고 해당 arm도 그 점수로 선택하므로 결과를 독립 검증으로 부르지 않는다.

## EXP078 baseline 재사용 조건과 시간

선택 arm의 F0/F1 **epoch8 최종 OOF**가 모두 있고, label/split·study/12target 순서·mask/corpus·격리2건·창 정책·resize/norm/증강·샘플링 RNG·FP16 train/eval·strict runtime·epoch 선택·모델 코드 의미가 유지될 때만 재사용한다. manifest의 네 source SHA와 data/initial-state hash, 실제 실행 receipt를 함께 확인한다. 미래 공통 코드 수정이 prediction에 영향 없음을 보이지 못하면 baseline이 같다고 가정하지 않는다. 아키텍처가 다른 초기 weights hash는 같을 수 없으며, ResNet 자체의 초기 가중치 출처/버전을 별도 고정한다. 기존 T4 ResNet OOF는 새 strict fit의 대체물이 아니다.

| 범위 | 추가 GPU 예산 | 근거/제한 |
|---|---:|---|
| 저장 OOF90/10 선별 | 0 | CPU만, 아직 미실행 |
| baseline 재사용 + 새 ResNet F0/F1 | 잠정1.5~2시간 | 과거T420epoch F0+F1 실측83.0분, 전체fold39.8~49.9분/fit. 새 로컬strict 미측정, 첫epoch로 ETA갱신 |
| baseline 재사용 불가: 한fold 새 대조 | 2fit | CoAt1+ResNet1, 두fold 확인이 아님 |
| baseline 재사용 불가: 두fold 새 대조 | 4fit, 잠정3~4시간 | CoAt strict8epoch 최초실측45.35분/fit×2 추가, ResNet ETA불확실 |

현재 EXP078 네 fit은 입력 효과를 분리하려는 대조라 타당하다. 같은 조건 baseline을 계속 다시 만드는 것은 비효율적이다. 위 계약을 보존하면 이후 후보마다 baseline 재학습 없이 candidate 두 fit으로 비용을 줄일 수 있다. GPU 작업을 실제 승인·배정하면 백그라운드 job ID/로그/ETA와 같은 채팅 완료 확인 예약을 남기고 반환한다.

근거: [검증본 audit](../handoffs/evidence/2026-10/2026-10-03-pfoa-complementarity/verified/audit.json) · [PFOA 쌍](../handoffs/evidence/2026-10/2026-10-03-pfoa-complementarity/verified/pfoa_pairs.csv) · [전체소견](../handoffs/evidence/2026-10/2026-10-03-pfoa-complementarity/verified/paired_targets.csv) · [macro](../handoffs/evidence/2026-10/2026-10-03-pfoa-complementarity/verified/scope_macros.csv) · [검사결과](../handoffs/evidence/2026-10/2026-10-03-pfoa-complementarity/verified/verification.json) · [도구](../../tools/pfoa_complementarity_audit.py) · [EXP078 manifest](../../artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/manifest.json).

검사: tied self/reversed synthetic pair matrix, sklearn AUC72항목+손상제외PFOA 교차확인 PASS, 원본24개 hash 유지(기존감사22개 일치). 계산0.955초. 첫 실행 산출물은 상위 경로에 보존하고 명시적 prior SHA 강제를 추가한 `verified/` 재실행과 수치가 같음을 확인했다. Windows glob 검색1회 실패는 알려진 정확 경로로 복구했다. 학습·원라벨·기존예측·root 소유 STATE/INDEX/인계는 수정하지 않았다.
