# 2026-10-03 20:54 KST · RN 양성항 가중치 한 fit 제안

**권고: 기존 EXP080 F2 RN20/B8 산출물을 고정하고, 같은 F2에서 `pos_weight=1` RN20 한 fit을 다음 개발 screen으로 준비한다.** 본 문서는 제안이며 실행 승인·사전등록·구현 완료가 아니다. 이번에는 작은 기록/소스만 읽었으며 학습, GPU, inference, 점수 계산, 원격, 큰 자산 재읽기를 하지 않았다. [검토 근거](evidence.json).

## 새 중복 근거의 적용 범위

[v3](../v3/report.md)는 F0↔F1 두 UID의 과거/현재 전체 cache 행과 mask가 각각 exact임을 확인했다. 현 모델 입력 중복이므로 이 두 fold를 독립 입력으로 해석하면 안 된다. 원본 geometry가 맞는7쌍 pixel 확인은 별도 진행 중이며 본 검토가 회수하지 않았다. 환자 동일성·전체 원본 동일성은 아직 이 문서의 결론이 아니다.

F2 screen에서는 두 사례가 모두 학습 쪽에 있고 기준/대안에 같은 비중으로 남는다. 이를 이유로 F2를 독립 holdout이라고 부르지 않는다. F2는 이미 노출된 weak 개발 fold이며, 다른 반복촬영·라벨·seed·선택 불확실성은 남는다. 최소 screen 전에 전수 중복 감사를 끝내도록 요구하지 않으며, 이번 대조에서 사후 사례 삭제·fold 이동을 하지 않는다. 현재 corpus 연결 오류가 별도 조사에서 드러나면 해당 계약을 먼저 재검토한다.

## 고정 주 비교와 판정

주 비교 하나: **F2 868건에서 `macro12 AUC[.9 rank(B8)+.1 rank(RN_ones20)] − macro12 AUC[.9 rank(B8)+.1 rank(RN_clip20)]`.** 같은 B8을 양쪽에 사용한다. 평균 동순위 rank/n은 전체 fold·target에서 한 번 만들고, subgroup/bootstrap에서 다시 rank하지 않는다. 비율·소견 routing·epoch 선택은 하지 않는다.

RN 단독 차이, 12소견/P/N·macro 기여, PFOA/다른11/골절/MCL, 기존 scanner/mask군은 보조 진단이다. 기존 RN 보완 +.001243은 이 변경의 예상효과가 아니다. paired report-group bootstrap500회/seed20261003은 기존 방식으로 유지하되, 고정 예측에 조건부인 CI이며 환자독립·seed·반복선택·private 불확실성을 포함하지 않는다.

- 고정20epoch 한 fit과 한 번의 등록 평가 뒤 종료. metric 기반 조기중단·추가fit/seed/비율 검색은 없다.
- 작은 양의 주 효과를 .002 미만이라는 이유로 폐기하지 않는다. CI0포함은 불확실성으로 보존한다. 음의 결과는 이번 직접 교체 미채택이며 모든 loss/가중치 전략의 반증이 아니다.
- 모든 소견 양수를 gate로 만들지 않는다. 손실은 P/N·크기·macro 기여·조건부CI로 함께 검토한다. 보조지표만 좋아졌을 때 주 가설 성공으로 바꾸지 않고 별도 후속 근거로 남긴다.
- 실행계약 불일치·비유한값·silent precision fallback은 중단. 기존 skip 기준(1% 초과 검토, 연속10회 중단)을 유지하며, baseline skip0/8680 updates와 대조한다. loss 변경에 따른 skip/유효 update 변화는 감추거나 LR로 보정하지 않는다.

## 재사용할 기준과 유일한 학습 개입

실제 EXP080 RN: ResNet34.a1_in1k/timm ImageNet,224px, B=within-slot-valid, train k12/eval24, train3475/eval868, fixed20, seed42, batch8/accum1, workers4, grad-checkpoint 없음, ImageNet norm, strict FP16 train/eval, loss scale1024. AdamW backbone/head LR=.0005/.001, WD=.02, OneCycleLR pct_start=.15, gradient clip3, drop_last=True를 유지한다. 실제 소요772초, optimizer8680/skip0이다.

`raptor_train.py:127–128`은 **train soft-label 평균**을 [.03,.7]로 자른 뒤 `(1−prev)/prev`를 [1,10]으로 제한한다. `:179,216`의 BCE 양성항에 적용하며 target 전체 가중치나 평가 P/N 가중치가 아니다. 대안은 같은 `BCEWithLogitsLoss(reduction='mean')`에 float32 12개 ones만 준다. loss 재정규화·다른 target mask·LR 조절을 함께 넣지 않는다. 무한 용량의 이상적 per-target 확률변환은 순위를 바꾸지 않을 수 있다. 가설은 유한한 공유 표현/최적화에서의 순위·보완성 변화다.

기준 OOF SHA는 RN `f37a63e49c366dee5c14dd283d7f56cab757819aff49c887bde139366878f876`, B `132a8dd653186bd56931d50388fd77c68bef93c32992533173b653c35ab10291`; 기존 EXP080 report 기록과 직전 검토의 현재 파일 hash가 일치했다. RN final checkpoint의 직전 관측 SHA는 `d198282aa26b2e06a26412942ef61ef318d789c16891801ffe363feb44ca9abe`; 과거 manifest가 final checkpoint를 고정했다는 뜻은 아니다. 초기 full-state SHA `093be508bda4f14b886c7ca28bf504bfdd2bffc95043252a60be10755b927224`와 구분한다. 이번 문서 작업에서는 이 큰 파일들을 다시 읽지 않았다.

## 최소 adapter와 비교를 깨뜨리는 위험

동결4파일은 그대로 둔다. 별도 import-safe entrypoint에 명시적인 loss 정책과 Gold 미접근 경로만 추가하고, 원본 대비 허용 diff·원본4SHA·adapter SHA를 기록한다. acquisition/training script를 import해 준비 작업을 실행하지 않는다. 모델/데이터 로드·CLI 실행은 명시적 main 안으로 제한한다. 공통 모듈 경로와 `__file__` 기반 provenance가 새 위치 때문에 잘못 묶이지 않게 한다.

1. **Gold/제외집합:** train:103→common:319–323은 공식 train.csv의 Gold를 읽으며, train:267–277은 fixed20/oof에도 Gold inference·CSV·로그·checkpoint/summary metadata를 쓴다. 로그만 숨기는 것으로 해결되지 않는다. Gold 읽기/평가를 제거하되 원래 Gold·fold−1·손상2건 제외를 보존한다. frozen corpus ID 순서+fold/exclusion metadata로 집합을 재구성하고 등록 train/eval SHA에 맞지 않으면 중단한다. `gold={}`만 대입하거나 제외집합을 새로 추정하지 않는다. 필요하면 이미 고정된 비라벨 UID allowlist를 사용한다.
2. **순서:** 기존 SHA는 sorted UID 집합이다. 이것만 일치해도 DataLoader shuffle의 행 순서는 달라질 수 있다. 원래 `corpus.ids` 순서의 train/eval 목록을 보존하고 ordered SHA를 추가 기록한다. corpus ids/manifest·fold/exclusion 파일은 기존 등록 SHA와 묶는다.
3. **RNG/초기값:** seed42, 별도 shuffle Generator42, persistent workers, epoch 공유, UID/epoch 기반 window/intensity RNG와 jitter=.20을 유지한다. 모델 head의 dropout=.2와 초기화 순서도 고정한다. 초기 full-state SHA 검사만으로 이후 RNG 상태까지 같아지지는 않는다. fit process 안에서 비교용 추가 model 생성·random smoke·iterator 소비를 하지 않거나 상태를 정확히 복구한다. audit는 별도 CPU process에서 한다.
4. **학습/평가:** offline cached timm revision/weight SHA, strict backend, GPU/torch2.5.1+cu121/timm1.0.19 환경을 보존한다. 첫 optimizer update 전에 초기 state/ID/런타임 계약을 검증한다. OOF는 baseline과 같은 eval batch4, final20, 같은 CSV writer/12target 순서로 기록한다. 평가 batch8로 바꾸거나 기존 OOF를 덮어쓰지 않는다. 기존 Gold 평가가 final OOF 뒤에 실행됐다는 사실을 보존하며 baseline을 새 no-Gold 성능으로 재표기하지 않는다.

구현 전 CPU 검사 범위는 ones BCE와 unweighted BCE 동등성, Gold 접근 금지/UID·순서 보존, 허용 source diff/계약 검증 정도다. 새 label/data/model audit를 무한히 추가하지 않는다. 실제 loss 벡터는 기존 기록에 없어 이번에 복원하지 않았다. 다음 허가된 준비에서 같은 train labels로 baseline 벡터와 ones를 기록하되 학습외 라벨로 가중치를 정하지 않는다.

## 성능 가설3개와 투자 비교

|순위/가설|최소 공정 대조|구현·CPU 비용 추정|GPU 비용 근거/추정|판단|
|---|---|---|---|---|
|1 학습: pos_weight가 RN 보완성을 제한|위 F2 한 fit, EXP080 B/RN 기준 재사용|30~60분; Gold/UID/RNG adapter 검토 포함|F2 실측772초를 근거로13~18분. F0/F1 두fit은 기존25분, 예산25~35분|가장 싼 분리 가능한 개입. F0/F1 확대보다 우선. 비용은 추정이며 실제작업 추가배정 아님|
|2 표현: head가 slot 역할을 이용하지 못함|F2를 보지 않은 B8 backbone 고정, train3475의 같은 K24 특징으로 기존 head와 slot-ID head를 둘 다 새 학습|60~120분|특징10~30분+head2개5~15분의 미실측 예산|기존 saved head만 대조하면 head 재학습 효과가 섞임. frozen backbone eval/BN 고정·train-only head학습·같은 step 예산 필요. pos_weight 결과 뒤 구체화|
|3 감독: MCL 미언급/정의 불확실성이 손실을 왜곡|가림 근거 감사 후 특정 불확실 항목 loss 제외, 다른11소견·class weight 유지|감사60~120분+CPU1~5분|근거 확보 뒤 RN2fit25~35분 또는 CoAt2fit85~100분의 기존 예산|현재 음성55건은 기존 Dread음성이고 unknown을 영상음성으로 바꿀 근거 없음. 바로 fit하지 않음|

추정 시간은 구현/감사와 GPU를 분리했다. 새 RN screen의 긍정 결과도 anchor0.943/D01(anchor+기존R2)의 개선이나 새로운 독립 후보 다양성을 입증하지 않는다. 고정 B+RN 안에서 유용성을 확인하는 단계다.

## 이후 확대

현재 확인된 입력 중복 연결은 향후 group split에 반영하되 무인 fold/seed 확대는 하지 않는다. 유망할 때 확인된 연결+report group을 묶은 사전 고정 split에서 두 레시피를 같은 ImageNet 초기화로 새 학습해야 한다. 과거 checkpoint를 새 split으로 재채점하거나 검증 행만 삭제해서 독립 OOF라고 부를 수 없다. 이 제한된 screen에 전수 환자 연결표를 선행 요구하지 않으며, top20 가림 감사 등 추가 확인은 별도 비용/정보가치로 정한다. 다음 실제 실행은 root의 기존 범위 판단·구체 계약 검토 뒤 GPU job/ETA/같은채팅 heartbeat로 관리한다.
