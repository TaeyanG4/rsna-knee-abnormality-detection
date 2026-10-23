# EXP086 DINOv2-S V2 고정 전이 결과

2026-10-04 06:33 KST · root. 고정 평가 시각 2026-10-03T21:21:29.054969+00:00.

**이번 8epoch 전이 설정은 미채택하며 기존 B를 유지한다.** DINO AUC 0.82882124, B 0.89339653, 차이 −0.06457528. 동일 888 study/873 component의 조건부 95% CI는 [−0.07353142, −0.05398887], bootstrap 500/500 유효다. 12소견 점추정치가 모두 하락했다. 실행 오류였던 V1 pilot과 달리 V2 full은 정상 완료한 성능 비교다.

## 소견별 근거

|소견|P/N|후보 AUC|B AUC|후보−B|macro 기여 Δ/12|명목 95% CI|
|---|---|---:|---:|---:|---:|---|
|ACL|122/766|0.821630|0.917455|-0.095825|-0.007985|[-0.135010, -0.063249]|
|MCL|45/843|0.716041|0.860445|-0.144405|-0.012034|[-0.220570, -0.076131]|
|Medial Meniscus|358/530|0.801939|0.909445|-0.107505|-0.008959|[-0.129555, -0.084211]|
|Lateral Meniscus|118/770|0.741933|0.867433|-0.125501|-0.010458|[-0.162163, -0.086100]|
|Medial OA|195/693|0.860983|0.898257|-0.037274|-0.003106|[-0.054245, -0.021275]|
|Lateral OA|127/761|0.812110|0.850880|-0.038770|-0.003231|[-0.065729, -0.010737]|
|PF OA|298/590|0.799747|0.851618|-0.051871|-0.004323|[-0.074029, -0.030342]|
|Effusion|225/663|0.891245|0.906878|-0.015633|-0.001303|[-0.025820, -0.005259]|
|Synovitis|217/671|0.885730|0.898140|-0.012410|-0.001034|[-0.022290, -0.000705]|
|Baker's|110/778|0.841867|0.947464|-0.105597|-0.008800|[-0.137883, -0.071990]|
|Contusion|136/752|0.874873|0.889696|-0.014823|-0.001235|[-0.034858, +0.006927]|
|Fracture|29/859|0.897756|0.923046|-0.025290|-0.002108|[-0.069498, +0.011658]|

10소견의 명목 구간은 음수이고 Contusion/Fracture는 0을 포함한다. 다중비교 보정·임상 오류율 추정은 하지 않았다. 골절은 P29/N859로 제한되며 input A가 없어 과거 B−A 손실 회복을 판단할 수 없다.

## 그룹·학습 계약

|그룹|N|후보−B macro|
|---|---:|---:|
|all_nonzero|654|-0.06321897|
|some_zero_mask|234|-0.06819041|

12소견 macro를 계산할 수 있는 모든 그룹도 하락했다. single-class가 있는 작은 그룹의 macro=null을 유지하고 그룹 CI를 주장하지 않는다. 개별 작은 그룹 소견의 양수/동률을 전체 개선으로 일반화하지 않는다.

T3455/V888·seed42·원 Dread/캐시/224·trainK12/evalK24·batch8·workers12/internal1·AdamW/OneCycle·8epoch 3448updates를 고정했다. 원 EXP084 B의 완료 예측을 재사용했다. ViT global_pool=token과 384 feature dimension을 포함한 encoder-family transfer package 비교이며 사전학습 효과만의 대조나 동일 FLOPs 비교가 아니다.

V2는 canonical518 trainable 위치표와 strict 초기자산을 보존하고 후보 instance에서 CPU FP32 bicubicAA 미분 보간을 적용했다. 원 generic asset revision4610ca143709d58a633b6397a74412c2c3842454, SHA04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081. CPU/CUDA 보간 bitwise 동등이나 T4 serving parity를 입증한 것은 아니다.

## 실제 비용과 실패 보존

- Full SUCCESS/resource COMPLETE·exit0·cleanup 오류/잔존0, 3448updates/skip0. 감독기 실제635.438초(10.59분), training log629초. 원V1 실패83.235+V2pilot31.641+full=750.314초로 전체5580초 이내.
- 원V1의 strict CUDA bicubicAA backward 미지원 실패/source/protocol을 보존한다. V2 성공은 V1 성공으로 바뀌지 않는다.
- 평가 wrapper 첫 호출은 절대 registration 경로로 entry에서 거부됐다. 상대 경로로 교정한 다음 호출에서만 실제 평가가 실행됐다. CLI 호출2회/실제 점수평가1회이며 재평가하지 않는다.
- RSS 표본의 process-tree 합 최대95,105,126,400B(~88.57GiB)는 공유 mmap 중복 집계 가능성이 있다. unique physical RAM·OOM으로 부르지 않는다. 독립 검토에서 기존 표본의 한계를 확인하고 다음 worker 결정에 반영한다.

## 판단과 재검토 조건

M02/05/07/10과 전략 누락 점검을 적용했다. 이 후보를 B로 교체하거나 추가 fold/seed/384·비율탐색으로 자동 확대하지 않는다. DINO 계열 전체를 기각하지 않으며, 다른 입력 정보·pooling 또는 사전 근거가 있는 최적화/적응은 별도 계약과 비용 비교 뒤 검토한다. 학습 loss1.0311→0.8984 감소는 관측이며 부족한 최적화나 pooling 원인의 증명은 아니다. 기존 B 마지막 loss0.8310과도 일반화 성능의 원인을 분리할 수 없다.

같은 노출 weak 개발V, 한 seed, 미확정 환자/Dread/generic 데이터 계보가 남는다. CI는 고정 라벨·모델·예측에 조건부이며 seed/선택/라벨 오류/private 이동을 포함하지 않는다. V CSV는 읽은 뒤0치환했다. Gold/중간 V 점수로 선택하지 않았다는 사실을 never-read로 확대하지 않는다. 정확 D01·0.943·private·임상 개선 근거가 아니다.

## 원근거·재개

- [원평가 report.json](../handoffs/evidence/2026-10/2026-10-04-exp086-report-tool-v2/evaluation-v1/report.json) · [12소견 CSV](../handoffs/evidence/2026-10/2026-10-04-exp086-report-tool-v2/evaluation-v1/targets.csv) · [SUCCESS](../handoffs/evidence/2026-10/2026-10-04-exp086-report-tool-v2/evaluation-v1/SUCCESS.json)
- [완료/평가 결정과 원실행 기록](../handoffs/evidence/2026-10/2026-10-04-exp086-dinov2s/v2/root-evaluation-decision.json) · [경로 교정](../handoffs/evidence/2026-10/2026-10-04-exp086-dinov2s/v2/root-evaluation-path-correction.json)
- [결과 판정](../handoffs/evidence/2026-10/2026-10-04-exp086-result/root-result-decision.json). [독립 검토](../handoffs/evidence/2026-10/2026-10-04-exp086-result/independent-result-review.md)에서12소견/22그룹/완료계약/저장결과4파일SHA PASS. 원예측 점수·bootstrap 재계산은 하지 않았다.


## 다음 투자 선택

[5안 비교](../handoffs/evidence/2026-10/2026-10-04-post086-priority/brief.md)와 [독립 반론](../handoffs/evidence/2026-10/2026-10-04-post086-priority/independent-critique.md)을 검토했다. root는 **추가 시리즈가 실제로 더 주는 정보와 고정K 입력 계약을 확인하는 CPU20분 한도 단계**를 선택했다. 새 pixel/캐시/GPU/등록은 아직 없다. [결정](../handoffs/evidence/2026-10/2026-10-04-post086-priority/root-next-decision.json).

producer의 patch-token MIL 제안과 독립 검토의 patchmean은 다른 개입이다. 둘을 같은 반박으로 취급하지 않는다. patch-MIL은 보존할 대안이지만 현재 큰 열세만으로 CLS를 원인으로 단정할 수 없고, 저렴한 재학습만 계속하는 것을 피하기 위해 입력 정보의 공백을 먼저 좁힌다.후속원근거검토에서 이전1936/4407의출처/정의를확인하지못했으므로현재가용률근거로사용하지않는다. broad1985/slot2357도다른정의다. [정정/CPU후속](../handoffs/2026-10-04-0656-exp087-cpu.md). EXP081 max-span 교체 결과는 재사용하고 같은 감사는 반복하지 않는다.
