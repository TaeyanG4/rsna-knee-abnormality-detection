# EXP079 고정 ResNet 보완 대조 — 2026-10-03

**저장 예측의 고정90/10 선별이 두 fold에서 양수여서, ResNet 후보 두 fit만 새로 실행했다.** 임시 B 기준 모델을 재사용하며 baseline을 다시 학습하지 않는다. 이는 학습 투자 근거이지 공개/임상/private 개선의 증거가 아니다.

| CPU 선별 | F0 | F1 | 평균 |
|---|---:|---:|---:|
| 12소견 macro 차이 | +0.002189 | +0.000996 | +0.001593 |
| PFOA 차이 | +0.004301 | +0.002943 | +0.003622 |
| 다른11 평균 차이 | +0.001997 | +0.000819 | +0.001408 |

최대 개별 하락은 F1 외측반월판−0.003126이다. 골절은 F0−0.000282/F1+0.002821로 EXP078의 F1 손실을 해소하지 못한다. 90/10과 평균 동순위 rank는 점수 계산 전에 고정했으며 비율 검색·소견 routing·Gold/LB 사용은 없다. 노출된 개발 fold에서 B 선택과 과거 ResNet best epoch를 재사용한 선별임을 유지한다. [CPU 계획](../../artifacts/experiments/EXP079_R34_COMPLEMENT_PROTOCOL/cpu-screen-plan.json) · [결과](../../artifacts/predictions/EXP079_R34_CPU_SCREEN/report.json).

새 실험은 **ResNet34.a1_in1k, 고정20epoch, LR5e−4, seed42, 입력 B, 224px/k12/eval24, strict FP16**이다. 원래 Dread 학습/평가 라벨·보고서 group fold·손상2건 격리·학습4소스는 유지한다. F0/F1만 순차 실행한다. 기존 B는 CoAtNet8epoch이므로 순수 구조 효과가 아니라 실용 레시피의 보완 효과를 비교한다. best/SWA/Gold 선택은 하지 않는다.

기존 timm 가중치 캐시의 revision `0e8757122c4c09b15fb9300c0fd09b3fe89e28bf`와 SHA, 초기 full-model hash를 고정하고 `HF_HUB_OFFLINE=1`로 실행했다. 실제 F0의 코드·초기값·학습/평가 UID 해시가 명세와 일치했다. [프로토콜](../../artifacts/experiments/EXP079_R34_COMPLEMENT_PROTOCOL/manifest.json) · [순차 큐](../../artifacts/experiments/EXP079_R34_COMPLEMENT_PROTOCOL/exp079_queue.json).

16:35 시작, runner29016. 최초 epoch40초·skip0·VRAM1.95GiB. 과거 T4 기반1.5~2시간 예산을 첫 로컬 실측 기준 **전체17:02~17:10 예상**으로 갱신했다. 첫 epoch 외삽이므로 종료 보장이 아니다. ID/ETA 최신값은 [실행 기록](../handoffs/evidence/2026-10/2026-10-03-continuous-research/exp079-status.json), [로그](../../artifacts/logs/queue/exp079_queue.log)에 남긴다.

두 fit 후 새 프로토콜을 검증하는 평가기로 B와 고정90/10을 비교한다. 두 fold macro·PFOA·다른11·12소견/P/N·그룹·조건부CI·optimizer skip을 함께 보고, 골절은 B뿐 아니라 보존 A와의 기술적 차이도 기록한다. 과거 RN 전용 `pfoa_fixed_rank_screen.py`에 새 checkpoint를 끼워 넣지 않는다. 작은 효과를+0.002 미만으로 폐기하지 않으며, 추가fold/seed/해상도 확대·제출은 자동 배정하지 않는다.

다른 가설과 비용: MCL 감독 정의는 현재 Dread 이진오류 수정 근거가 없어2fit 보류; series 선택은 학습 없는20~30건 metadata 감사가 대안; EXP078 추가 창 변형은 이번에는 반복하지 않는다. CPU 선별이 지지한 레시피 한 개에 우선 투자한다. 대회 제출·공개는 매번 별도 승인이다.


## 17:02 결과와 다음 투자

두 fit는17:00:13 종료했다. 실제723+777초=25분, optimizer skip0이며 초기 state·코드·UID·라벨·입력·strict 환경 계약이 일치했다. 전용평가기 합성59+기존16=75검사 후 [고정 평가](../../artifacts/predictions/EXP079_R34_COMPLEMENT_REPORT/report.json)를 한 번 실행했다. bootstrap500회·seed20261003, subgroup/CI에서 rank를 다시 만들지 않았다.

| 고정90/10 혼합의 B 대비 변화 | F0 | F1 | 평균 |
|---|---:|---:|---:|
| 12소견 macro | +0.002097 | +0.002138 | +0.002117 |
| PFOA | +0.003122 | +0.005757 | +0.004440 |
| 다른11 | +0.002003 | +0.001809 | +0.001906 |
| 골절 | −0.003029 | +0.005344 | +0.001157 |

macro 조건부95%CI는[+0.001462,+0.002731]. 12소견 fold평균 모두 양수이며 all-nonzero/일부zero 그룹도 모두 양수다. 하지만 F1 Philips3T(n146) −0.000496이고, 작은/단일class 그룹의12소견macro는 계산하지 않았다. 전체 그룹 무퇴보나 환자독립 검증을 뜻하지 않는다.

원래 A 대비 혼합의 macro는+0.005320이지만 골절은 F0−0.003663(P34/N835),F1−0.022666(P30/N839),평균−0.013165다. 골절 CI[-0.037717,+0.008885]가 넓다는 이유로 손실을 무시하지 않는다. B의 입력 효과와 RN의 보완 효과를 분리한다.

**판정: 작은 일관된 보완 효과를 개발 후보로 보존한다. 전체학습·제출 적용은 아직 하지 않는다.** 노출fold·선택된B·weak 라벨·seed/private 분포 불확실성이 남는다. [동결 결정](../../artifacts/experiments/EXP079_R34_COMPLEMENT_PROTOCOL/decision.json).

후속은 F2 A/B/RN 고정3fit(95~110분)을 별도 사전등록한다. B/RN2fit(55~65분)은 더 저렴하지만 A 대비골절을 분리하지 못한다. 새seed 한fold B/RN(55~65분)은 seed 변동만 줄이며, series 선택20~30건감사(CPU30~60분)는 다음 정보손실 가설로 보존한다. MCL 수정은 새근거부족으로 보류하고 감독 전략전체를 기각하지 않는다. F2도 과거 노출 개발fold이며 공통격리 후 평가수가869와 다르므로 별도UID 계약을 쓴다.
