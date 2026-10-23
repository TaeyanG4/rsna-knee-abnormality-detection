# 2026-10-03 00:40 KST · 토론 기반 공략 지점

- 상태: 완료(조사). 기록자: Claude.
- 요청: 현재 단계가 최적인지, 토론에 어디를 공략할지에 대한 내용이 있는지.
- 결과: [검증 보고서 9절](../research/baseline-validation-2026-10-02.md). 정답은 판독문이 아니라 엄격한 기준의 영상 재판독(주최 답변 733491, 기준 733343). 병목은 라벨, 의 의미가 소견마다 다름(733932), Synovitis·외측 소견이 약점.
- 판정: 방향은 맞음. 수정 3가지: 소견별 혼합 비율(외부 근거로 사전 결정), Synovitis 전용 채움+RadImageNet, 외측 소견은 영상 쪽 점검. 라벨 재추출은 보류.
- 검증: 토론 원문 인용 확인. 새 학습은 하지 않음. 실행 중 작업은 [이전 기록](2026-10-02-2100-anchor-swap-submit.md) 그대로.
- 00:55 계획 변경(사용자 결정): 384 px fold 2~4 취소(러너 pid 37204 종료, fold 1은 완료까지 유지). 옛 라벨 384 5 fold는 teacher로 불필요하고 최종 384 학습은 레시피 확정 후 한 번만(10-15 무렵). 로컬 GPU는 224 px 라벨 실험에 사용.
- `label_mix.py --alpha-map` 추가(소견별 이미지 비중). `derived_mix/mix_r34_pf.csv`: 이미지 0.7 = Effusion·Synovitis·Contusion·Fracture·Medial OA·PF OA, 판독문 0.7 = ACL·Lateral Meniscus·Lateral OA, 나머지 0.5(근거: 토론 733932·740610, Gold58로 고르지 않음).
- 새 로컬 큐 `artifacts/logs/queue-r2-224-20261003.json`: fold 1 뒤 EXP073_R2PF_F0 → F1 → EXP072_R2MIX_R34_F2(균일) → EXP073_R2PF_F2. fold당 41분. Kaggle은 균일 fold 0·1 진행 중.
- 판정: 각 fold를 기준 라벨로 재채점해 R1 같은 fold와 비교(F0 0.8806, F1·F2는 `EXP061_R1_F1/F2` summary). 균일 vs 소견별은 같은 fold 짝 비교.
- 판정 도구 `tools/fold_compare.py`(같은 fold 짝, 기준 라벨 재채점, bootstrap). 검산: EXP060 vs R1 fold 0 +0.0153 [+0.010, +0.020].
- 다음 세션: (1) 제출 점수 1회 확인. (2) `fetch-kernel --tag EXP072_R2MIX_R34_F0`·`_F1`. (3) `python tools/fold_compare.py --baseline EXP061_R1 --candidate EXP073_R2PF EXP072_R2MIX_R34`. 효과 있으면 5 fold·반복 distillation, 없으면 silence 모드·teacher 교체(R1 OOF).
- 01:45 재부팅 후 재개: EXP060_RAPTOR_F1 완료(weak OOF 0.8938 swa, Gold58 0.9023, 3.15 h; fold 0과 거의 같음). `queue_resume.py`로 224 px 큐 재개(러너 pid 14816), EXP073_R2PF_F0 01:43 시작. Kaggle 균일 fold 0·1 계속 실행 중, 제출은 PENDING(01:43 확인).
