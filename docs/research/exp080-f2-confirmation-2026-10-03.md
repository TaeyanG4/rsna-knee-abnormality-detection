# EXP080 · F2 입력과 ResNet 보완 확인

EXP079 고정90/10 혼합은 개발 F0/F1의 macro를 각각+0.002097/+0.002138 개선했다. 그러나 원래 입력 A 대비 F1 골절은−0.022666이다. **다음 순서인 F2에서 A·B·RN_B 세 fit를 고정해 두 문제를 분리한다.** 새로운 독립 holdout이나 최종 모델 확대가 아니다.

- A/B: 기존 CoAtNet224, 고정8epoch, 각각 global-valid/within-slot-valid.
- RN_B: ResNet34.a1_in1k224, 입력B, 고정20epoch. 원래 Dread 학습·평가라벨, seed42, strict FP16, 학습4소스와 초기 가중치 계약을 유지한다.
- 기존 명령에서 tag와 fold만 바꿨다. 공통 손상2건 제외 후 train3,475/val868, 평가 report-group840. metadata 검사에서 교차 Series UID·선택 series 집합·report group 중복0, Gold58·보고서중복 격리4건 혼입0. 환자/영상 근접중복은 미검증이며 scanner fingerprint는 환자ID가 아니다.
- 세 fit 모두 끝난 뒤 B−A, 고정90/10(B,RN_B)−B, **같은 혼합−A**를 별개로 기록한다. 12소견/P/N·골절·scanner/자장/제조사/결측 그룹·고정예측 report-group CI500회(seed20261003)를 함께 본다. 다른 비율·A+RN 조합·소견 routing·Gold/중간epoch 선택은 하지 않는다.
- 양의 작은 효과를+0.002 미만으로 폐기하지 않는다. 손실·불확실성을 보존하고 추가학습은 새 판단 후 결정한다. runtime fallback·계약오류·nonfinite는 중단, skip>1%/연속10회는 검토한다.

17:14 시작, A→B→RN 순차 큐. 이번 실측에 근거한 예산95~110분, 전체18:49~19:04 KST 예상이며 초기 throughput으로 갱신한다. queue 성공 표식만으로 연구 성공을 판정하지 않는다. 기존 두fit 비용25분을 재사용한 후속 판단이며, B/RN만 하는55~65분 대조보다 비용이 크지만 A 대비골절 위험을 함께 확인한다. 새seed와 series 감사는 다음 대안으로 보존한다.

[명세](../../artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/manifest.json) · [큐](../../artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/exp080_queue.json) · [실행 결합기록](../../artifacts/experiments/EXP080_F2_CONFIRM_PROTOCOL/launch-receipt.json) · [메타데이터 검사](../handoffs/evidence/2026-10/2026-10-03-exp080-f2/metadata-preflight.json). 제출·공개는 매번 사용자 승인 대상이다.
