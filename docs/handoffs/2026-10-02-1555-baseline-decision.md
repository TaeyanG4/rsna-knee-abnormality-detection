# 2026-10-02 15:55 KST · baseline 최종 선정·1등 경로·차선책

- 상태: 완료(결정 기록). fold 0 학습은 진행 중. 기록자: Claude.
- 요청·결정: baseline 후보를 최종 선정하고 1등을 위한 경로와 막힐 때의 차선책을 정해 기록. 결정 본문은 [ADR 0003](../decisions/0003-baseline-and-route-to-first.md).
- 결정 요약: 학습 baseline = dreaddevelopment Raptor 레시피(단일 모델 공개 0.924, 저자 보고 0.936~0.938). 제출 anchor = 공개 d4-blend 계열(확인 점수 0.943~0.946)의 Raptor arm을 우리 모델로 교체. 경로는 R0 재현 → R1 빠른 루프(224~288) → R2 라벨(OOF 혼합·silence 채움·teacher 교체) → R3 기하(80~96장·384) → R4 5 fold×2 seed → R5 nested-OOF 앙상블 → R6 Kaggle fp16 노트북. 단계마다 gate.
- 차선책 요약: 재현 실패 시 공개 maxspan-v5에서 fine-tune; 메모리·속도 부족 시 224/288·bs 2; 라벨 효과 없으면 다양성(ResNet 2.5D 224, 기존 캐시 모델); 9 h 초과 시 k_eval·arm 축소; 시간 부족 시 fold 0 모델을 anchor의 Raptor arm에 50:50 교체가 최소안.
- 추가 근거(토론·보고서): 0.946 블렌드의 per-label map은 public split 튜닝이라 flat-0.60을 두 번째 최종 선택으로 유지. 보고서 silence(Synovitis 양성 절반, Fracture 7% vs 31%)가 주된 라벨 오류. teacher 단순 평균은 기존 감사에서 실패(0.880 < Flight 0.899), 따라서 라벨 lever는 OOF 혼합 쪽.
- 검증: 이 작업은 문서·결정. pytest 94·Playwright 33은 직전 기록 기준. 학습 EXP060 fold 0: epoch 0 OOF 0.730(n=200)·gold 0.713(warmup 단계), 1.0~1.5 s/step, 15.2 GB, epoch 0 완료 15:32 KST → 12 epoch 완료 예상 19:30 KST 전후.
- 남은 일·다음: (1) EXP060 완료 후 `raptor_infer.py --ckpt artifacts/experiments/EXP060_RAPTOR_F0/model_best.pt --subset gold --amp fp16`과 공개 maxspan-v5 비교, R0 gate 판정. (2) 통과 시 R1 설정(224 px, k 12, bs 8, 8 epoch) fold 0~4 순차 실행. (3) 미달 시 ADR 차선책 1행.
