# Baseline 타당성 검증 (2026-10-02)

후속 정정(10-03): [독립 재감사](route-to-first-2026-10-03.md)에서 R2 간접 누출·시리즈 경계/결측 창·full-data Gold 선택 fallback을 확인했다. 아래의 R2 효과 확정 및 라벨 병목 단정은 재검증 전 잠정 관측이다. 외부 LLM API는 주최733965가 조건부 허용했으므로 9절의 ‘주최 확인 없음’은 부정확하다. 과거 수치와 당시 해석은 기록으로 보존한다.

대상: dreaddevelopment Raptor 레시피([학습 노트북](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) version 8, 2026-10-02 수집 sha `6ea209b0…`). 질문 세 가지: 우리 재구현이 원본과 같은가, 재현되는가, 1등까지 갈 상한이 있는가.

## 1. 재구현 충실도

| 항목 | 원본 | 우리 `raptor_train.py` | 판정 |
|---|---|---|---|
| 모델 | timm 백본 + LayerNorm → 소견별 attention(Linear 256, Tanh, Dropout 0.2) → 소견별 선형 | 같은 구조 | 같음 |
| 창 | 3장 연속 슬라이스 RGB, 학습 무작위 k, 평가 균등 k_eval, mask 최소~최대 범위 | 같은 함수 | 같음 |
| 증강 | 밝기 배율 ±10%, flip 없음 | 같음 | 같음 |
| 손실 | BCE, pos_weight = clip((1−p)/p, 1, 10) | 같음 | 같음 |
| 최적화 | AdamW bb 3e-5 / head 1e-3, wd 0.02, OneCycle 15%, clip 3, bf16 | 같음 | 같음 |
| 정규화 | 학습 노트북 기본값은 없음([0,1]) | ImageNet | 공개 추론 노트북(`sankurero/rsna-knee-coatnet-raptor-only`)이 ImageNet을 쓰고 우리 parity가 체크포인트 기록값과 일치하므로 공개 가중치는 ImageNet으로 학습된 것. 같음 |
| 라벨 | 저자 LLM soft label(노트북에는 미포함) | `dreaddevelopment/rsna-knee-labels` CC0, 같은 저자 공개본 | 같은 출처. Gold58 study는 라벨 파일에서 빠져 있음 |
| 학습 데이터 | Gold58 외 전부(4,349) | fold 4/5(약 3,480) | 의도된 차이. 최종 모델은 전체 학습 또는 5 fold |
| epoch | 16(공개 v5는 epoch 12·14·15 평균) | 12 | **차이**. 최종 레시피에서 16 비교 필요 |
| 배치 | bs 8 | bs 4 × accum 2 | 효과 동일(BN 통계 창 48개) |
| 선택 | **Gold58 AUC로 epoch 선택·top-3 평균** | weak OOF로 선택(ADR 0004) | 원본 Gold58 점수는 선택 편향 포함 |
| corpus | 노트북은 44장 corpus, 공개 v5는 64장 maxspan | 64장 maxspan(`raptor64_336_c140_s02`) | v5 계약과 같음 |

## 2. 공개 체크포인트가 말해 주는 것

| 체크포인트 | Gold58(저자 기록) | 약한 소견 |
|---|---|---|
| maxspan-v5 | 0.9214 | Synovitis 0.80, PF OA 0.85, Lateral OA 0.87, Lateral Meniscus 0.88 |
| native384dense-v10 | 0.9174 | Synovitis 0.80, PF OA 0.84, Lateral Meniscus 0.85 |
| native384-v8 | 0.9067 | Synovitis 0.78, Lateral OA 0.79, Lateral Meniscus 0.83 |

- 세 체크포인트 모두 약한 소견이 같다. 보고서가 자주 언급하지 않는 소견(silence)이라 라벨 문제와 일치한다.
- 우리 corpus에서 maxspan-v5 Gold58 fp16 0.9198, study bootstrap 95% 구간 0.895~0.943. **Gold58로는 0.02~0.03 미만 차이를 구분할 수 없다.**
- 공개 LB 단일 Raptor 노트북은 0.928~0.933, Raptor 4개 arm 평균 0.935, Raptor+DINO blend 0.943~0.946. 1위 0.961과의 차이는 약 0.03.
- 라벨 세트 대부분이 Gold58 공식 라벨을 덮어써서 Gold58 채점이 1.0으로 나온다(비교 불가). 덮어쓰지 않은 라벨 중 최고는 Flight hybrid 0.899, stevenleehans v4 0.893. 원저자 라벨은 Gold58을 제외해 직접 채점할 수 없다.

## 3. 재현 (EXP060, fold 0, 384 px, 12 epoch)

EXP060은 우리 실험 번호다. 다른 노트북을 fork한 것이 아니라 위 재구현(`raptor_train.py`)으로 ImageNet 사전학습 CoAtNet에서 처음부터 학습했다. 공개 Raptor 체크포인트는 쓰지 않았다.

| 지표 | 값 | R0 gate |
|---|---|---|
| weak OOF macro AUC(869 study, top-3 평균 선택) | 0.8963 | ≥ 0.88 통과 |
| Gold58(bf16 / fp16 재추론) | 0.9028 / 0.9026 | ≥ 0.90 통과 |
| 공개 v5 Gold58(전체 데이터, 16 epoch, Gold 선택) | 0.9198 | 참고, 95% 구간 안 |

- 약한 소견도 공개 체크포인트와 같다: Gold58 Lateral Meniscus 0.80, PF OA 0.83, Synovitis 0.82. OOF에서는 MCL 0.84, PF OA 0.85, Lateral Meniscus 0.86.
- 학습 시간 3.7시간, 최대 10.9 GiB. epoch 10~11에서 OOF가 평탄해졌다(0.905).

## 4. 공개 노트북·토론이 말해 주는 상한

근거: 노트북 234개 전수(소스·입력), 토론 144개, 과거 RSNA 5개 대회 솔루션. 원문 스크립트 결과는 세션 scratch, 핵심 인용은 아래 topic/notebook id로 다시 확인했다.

- 0.943~0.946 공개 노트북은 모두 추론 전용이며 **다섯 계열**을 섞는다: Raptor CoAtNet 3종, mattiaangeli CoAtNet 판독기 3종(D4·Global96·resgated, 학습 코드 비공개), DINOv2-S 20개(pilkwang), DINOv3 cross-series(학습 코드 비공개), RadImageNet ResNet-50 + 소견별 헤드(sofiaanjenje e11/e13 학습 노트북, antoinegg1·prvsiyan 헤드). `pjmathematician/rsna-knee-d4-blend` 입력 목록으로 확인.
- Raptor 없이 0.93에 닿은 노트북은 없다. Raptor-free 최고 0.922(`mattiaangeli/bend-the-knee-to-dinov3-ensembled`), 이를 Raptor와 순위 평균하면 0.933 → 두 계열은 상보적이다.
- 토론의 0.949 이상 주장은 모두 **라벨 개선**(OOF 예측을 soft label에 섞기, 다중 teacher pseudo-label, 자체 LLM 라벨)과 함께 나왔다: CoAtNet 224 단일 fold 0.950, ResNet 224 5 fold 0.954, CoAtNet 288 0.938→0.949, 소형 ResNet/EffNet 0.949(topic 735304, 740474, 742050). 같은 encoder를 teacher로 쓴 OOF 혼합은 LB가 안 올랐다는 반례 2건(735304, 743148). 전부 자기 보고이며 LB 검증은 없다.
- 서로 다른 계열 앙상블: DINOv2 0.942 단독 + 다른 계열 → 0.949(744511). 큰 encoder는 효과 없음(+0.0011, 잡음 0.002, 735154).
- 저자 자료(737696): 같은 라벨에서 슬라이스 밀도·창 수로 0.917~0.926 → 0.932, SWA 효과 없음. 384 체크포인트를 256 px로 돌리면 −0.029. 우리 R1은 224 전용 체크포인트(`coatnet_rmlp_2_rw_224`)라 이 경우와 다르지만, 384 대비 차이는 R1 gate로 따로 확인해야 한다.
- 우리 제출: 자체 ConvNeXt-S 288 5 fold(이전 캐시) Gold58 평균 0.891 → 공개 LB 0.909. 공개 Raptor v5 단독 Gold58 0.92 → LB 0.928. 두 경우 LB가 Gold58보다 0.01~0.02 높다.

## 5. 판정

- **재구현은 타당하다.** 모델·창·증강·손실·최적화가 원본과 같고, 정규화·라벨 출처도 공개 가중치와 일치한다. 남은 차이는 epoch 수(12 vs 16)뿐이다.
- **학습 baseline으로 Raptor를 유지한다.** 학습 코드가 공개되고 재현된 레시피 중 공개 LB가 가장 높다(단독 0.928). 다른 계열 중 측정된 최고는 0.898~0.922다.
- **그러나 Raptor 단독으로는 1등이 안 된다.** 단독 상한 근거는 0.932~0.941이고 1위는 0.961이다. 1등 경로는 (1) 다른 계열 teacher로 만든 OOF 라벨 혼합, (2) 서로 다른 계열 2~3개 앙상블이다. 아키텍처 자체는 차별점이 아니다.

## 6. 다른 baseline 후보

| 순위 | 후보 | 근거 | 역할 | 비용·위험 |
|---|---|---|---|---|
| 1 | 소형 2.5D CNN 224 px(ResNet34·EffNet-B0·ConvNeXt-T), 같은 corpus·헤드 | 토론 0.936~0.954 주장 다수, 공개 측정은 없음. 우리 ConvNeXt-S는 LB 0.909(이전 캐시) | 빠른 실험, **다른 계열 teacher**, 두 번째 arm | fold당 0.5~1 h 추정. 우리 trainer에 `--arch`만 바꿔 바로 가능 |
| 2 | RadImageNet ResNet-50(고정) + 소견별 attention 헤드 | 0.943 이상 노트북 전부에 포함, 20% 가중 +0.007(prvsiyan) | 앙상블 arm, Synovitis 보강 | 헤드만 학습해 수 분. 일부 공개 헤드는 CC-BY-NC-SA라 직접 학습. encoder 라이선스 확인 필요 |
| 3 | DINOv2-S 다중 slot(pilkwang/kitopl) | 5 fold 0.898, 학습 코드 공개 | 재현 가능한 다양성 arm | 단독은 약함. T4 fold당 약 95분 |
| 4 | DINOv3 cross-series(A5 재구현) | 상위 blend 핵심 단계, 입력 방식이 가장 다름 | 다양성 | 학습 코드 비공개, 가중치 라이선스·오프라인 첨부 문제. 비용 큼 |
| 5 | 내측·외측 공유 헤드(좌우 정규화) | 과거 대회 전이 논리뿐. 외측 소견이 일관되게 약함 | Raptor·CNN 개선안 | 코드 2~3일, 효과 불확실 |

제외: 더 큰 encoder, Pillar-0 3D(Gold58 0.797), 보고서 보조 손실, 라벨 마스킹, VLM 미세조정(정체 시에만).

## 7. 시작 노트북 재확인 (19:30 KST, 수집 갱신 후)

공개 점수 158개 재정렬. 0.943 이상은 모두 같은 계열(입력 데이터셋 14개·커널 2개, probe22 소견별 가중치).

| 노트북 | 공개 LB | 판정 |
|---|---|---|
| `pjmathematician/rsna-knee-d4-blend` | 0.946 | 0.943 스택 + 작성자 비공개 자산(`rsna-knee-ens14-assets`, `rsna-knee-eff6-assets`) 순위 가중 0.45. 자산이 공개 입력에 없어 fork하면 재현 불가. 실제 fork(`yamadan96/rsna-knee-d4-public0946`, `sushanthtiruvaipati/rsna-knee-d4-public0946-no-correction`)는 0.943 |
| `pjmathematician/rsna-knee-d4-lite` | 0.945 | 같은 구조, 비공개 자산 의존 |
| `jiweiliu/rsna-knee-fast-2xt4-inference` | 0.943 | **제출 시작점 권장.** Mattia v34 레시피를 그대로 유지, 2×T4 병렬, 판독기 실패 시 fallback, 자산 manifest 검사. 투표 210 |
| `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` | 0.943 | 원저자판, 같은 점수. 위의 대안 |
| `skarin/rsna-knee-reproducing-the-0-943-public-stack` | 0.943 | 재현 기록용 참고 |

- 우리 기존 제출 anchor는 이전 세대 0.941 스택(maverick probe22)이었다. 시작점을 0.943 스택으로 올리는 것이 첫 단계다.
- 0.945·0.946이 쓴 방식(공개 스택 + 자체 모델 순위 가중)은 ADR 0003 R5와 같다. 우리 모델이 그 자리에 들어간다.
- 주의: probe22 가중치는 공개 LB 탐색으로 맞춘 값이다. 최종 2개 중 하나는 flat 0.60 유지. 스택에 CC-BY-NC-SA 헤드(antoinegg1)가 들어 있어 상금 대상 라이선스 조건은 제출 전에 확인한다.
- 학습 시작점은 그대로: Raptor 학습 노트북(R0 통과). 두 번째 계열은 RadImageNet 학습 노트북(`sofiaanjenje/rsna-knee-e13-train`)을 참고하되, 그 노트북의 Gold58 적합 단계는 빼고 다시 만든다.

## 8. 판독문 라벨은 Gold만큼 정확해질 수 있는가 (10-03 00:20)

Gold58의 58 study는 판독문도 있으므로, 판독문 라벨·이미지 모델·둘의 혼합을 모두 공식 라벨로 채점할 수 있다(이미지 모델은 Gold58을 학습하지 않음, 순위 평균).

| 입력 | Gold58 macro | 약한 소견 |
|---|---|---|
| Flight hybrid 판독문 라벨 | 0.899 | Synovitis 0.79, L.OA 0.83, Contusion 0.86, Effusion 0.88 |
| Raptor384 fold 0 | 0.903 | L.OA 0.79, PF OA 0.83 |
| ResNet34 5 fold 평균 | 0.888 | L.Men 0.77 |
| 판독문 0.5 + 이미지(Raptor+ResNet34) 0.5 | **0.928** | L.OA 0.82 |

- 혼합 − 판독문: +0.029, 95% 구간 [0.013, 0.045]. 혼합 − 이미지: +0.024, 구간 [0.008, 0.040]. 58건에서도 유의하다.
- 보완 관계가 뚜렷하다: 판독문은 ACL·MCL·반월판(0.95~0.99)에 강하고 이미지는 Effusion·Medial OA·Baker's·Fracture(0.93~0.98)에 강하다. 판독문이 약한 소견은 보고서에 잘 안 적히는 것들이다.
- 결론: 4,349 study의 라벨을 Gold 수준(1.0)으로 만들 수는 없지만, 현재 0.90에서 0.93 근처로 올릴 수 있다. 이것이 토론의 0.949~0.954 주장과 같은 기전이며 R2(label_mix blend 0.5)가 정확히 이 작업이다. Lateral OA(0.82)는 어느 방법으로도 약해 별도 lever가 필요하다.
- 주의: 비율 0.5는 Gold58로 고른 것이 아니라 토론 권고값이며, 0.3·0.7도 0.924로 비슷하다. 최종 비율은 OOF 재채점으로 정한다.

## 9. 토론이 가리키는 공략 지점 (10-03)

근거: 저장된 토론 145개 중 공략 지점을 다룬 글. 숫자는 모두 작성자 보고이며 LB로 검증되지 않았다.

- **정답은 판독문이 아니라 영상 재판독이다.** 주최 측 답변(733491): Gold·테스트 라벨은 MSK 영상의학과 의사 2명이 표준 기준으로 영상을 판독하고 3번째가 판정했다. 판독문과 다른 과정이라 불일치가 정상이다. 기준(733343)은 엄격하다: 경계 사례는 음성, ACL·MCL은 고등급 파열만, OA는 연골 50% 초과 소실 1 cm 이상, Effusion·Baker's는 중등도 이상.
- **병목은 라벨이라는 데 대체로 합의**(735826, 733932, 735767 댓글). 큰 encoder는 효과 없음(735154).
- **의 의미가 소견마다 다르다**(733932, 58표). Baker's·Medial OA는 언급이 없으면 거의 음성(Gold 양성 3%, 0%)이므로 그대로 둔다. Synovitis(34%)·PF OA(21%)는 언급 없음이 정보가 없으므로 채워야 한다. 12개 소견 일괄 대치는 오히려 나빴다(0.8873 → 0.8805).
- **Synovitis**: 보고서의 84%가 언급하지 않는다. Effusion 언급으로 미정 셀만 채우면 0.678 → 0.790(733932). RadImageNet 사전학습이 0.62~0.72 → 0.78(737566).
- **외측 소견**: Lateral OA·Lateral Meniscus는 이미지 모델이 판독문 라벨보다 낮다. Lateral OA 양성 11건 중 10건은 판독문에 적혀 있다(740610) → 라벨보다 영상 쪽(외측 구획 관찰) 문제.
- **이미지 대 판독문 라벨, 다른 참가자 측정**(740610): 이미지가 우세한 소견은 MCL·Baker's·Medial OA·Contusion·Effusion·Fracture, 판독문이 우세한 소견은 Lateral Meniscus·Lateral OA. 우리 Gold58 표(8절)와 방향이 같다.
- **상한 전망**: 금 약 0.96, 1위 약 0.975 예상(Chris Deotte, 735767). 막판 상승과 private 변동이 크다는 데 다수 동의.

### 현재 경로에 대한 판정과 수정

방향(라벨 우선, 다른 계열 teacher, 앙상블)은 토론과 일치한다. 그러나 세 군데를 고친다.

1. **소견별 혼합 비율.** 지금 R2는 모든 소견에 0.5를 쓴다. 다음 변형은 외부 근거(733932, 740610)로 미리 정한 비율을 쓴다. 이미지 쪽: Effusion·Synovitis·Contusion·Fracture·Medial OA·PF OA. 판독문 쪽: ACL·Lateral Meniscus·Lateral OA. Gold58로 비율을 고르지 않는다.
2. **Synovitis 전용 처리.** 판독문이 언급하지 않은 셀만 Effusion 정보와 이미지 OOF로 채우고, RadImageNet 계열 모델을 앙상블에 넣는다.
3. **외측 소견은 영상 쪽을 본다.** coronal slot 장수·외측 구획 crop 확인, R3(80~96장)에서 Lateral OA·Lateral Meniscus를 따로 판정한다.
4. 보류: 기준에 맞춘 라벨 재추출(로컬 LLM). 비용이 크고, 상용 LLM에 판독문을 보내는 것은 규칙 위반 소지가 있다는 지적(733652, 주최 확인 없음).

## 10. R2 결과 (10-03 06:00)

라벨 혼합(판독문 0.5 + ResNet34 OOF 0.5)으로 학습한 CoAtNet 224가 기준 라벨 재채점에서 fold 0·1·2 모두 +0.016(95% 구간 최저 +0.010). 소견별 비율은 균일과 차이 없음. 상세는 [handoff](../handoffs/2026-10-03-0600-r2-result.md).
