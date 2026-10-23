# 2026-10-02 19:00 KST · baseline 타당성 검증·후보

- 상태: 완료(검증). R1은 큐로 진행 중. 기록자: Claude.
- 요청: Colab은 보류. 현재 baseline(Raptor)이 타당한지 검증하고 다른 후보를 제시. 목표 1등.
- 결과: [검증 보고서](../research/baseline-validation-2026-10-02.md).
  - 재구현 충실도: 모델·창·증강·손실·최적화·정규화·라벨 출처가 원본 학습 노트북(v8)과 같다. 차이는 epoch 12 vs 16과 의도된 차이(OOF 선택, fold 학습).
  - R0 통과: EXP060 weak OOF 0.8963, Gold58 fp16 0.9026(bf16 0.9028). EXP060은 우리 실험 번호이며 ImageNet 사전학습에서 직접 학습한 모델이다.
  - Gold58 95% 구간 폭 약 0.05라 0.02~0.03 미만 차이는 판별 불가.
  - 판정: Raptor는 학습 baseline으로 유지. 단독 상한 근거 0.932~0.941이라 1등에는 다른 계열 teacher의 OOF 라벨 혼합과 2~3계열 앙상블이 필요.
  - 후보: ① 소형 2.5D CNN 224(ResNet34 등, 같은 corpus·헤드) ② RadImageNet ResNet-50 고정+헤드 ③ DINOv2-S 다중 slot ④ DINOv3 cross-series ⑤ 내·외측 공유 헤드.
- 검증: 원본 노트북 소스 대조, 공개 체크포인트 메타, 라벨 세트 Gold58 채점, bootstrap, 상위 노트북 입력 목록, 토론 원문 인용 확인(737696·744511). ResNet34 CPU smoke 통과(결과 폴더 삭제). 노트북·토론 전수 조사는 하위 에이전트 2개가 수행.
- Colab: 사용자 결정으로 보류. STATE·작업 목록·runbook에 반영.
- 실행 중: 큐 러너 pid 16136이 EXP061_R1_F0(224 px) 18:59 시작, 이후 F1~F4. `tools/gpu_job.py --list`.
- 다음: (1) 소형 CNN fold 0을 R1 뒤 또는 Kaggle T4에서 실행해 계열 비교. (2) R1 판정 후 다른 계열 teacher OOF로 R2 라벨 혼합. (3) 최종 Raptor는 16 epoch·전체 데이터 비교. ADR 0003 경로 수정은 사용자 확인 후.
- 19:30 추가(시작 노트북 재확인): 0.946·0.945는 비공개 자산에 의존해 fork 시 0.943으로 떨어진다(fork 2건 실측 0.943). 제출 시작점은 `jiweiliu/rsna-knee-fast-2xt4-inference`(0.943), 학습 시작점은 Raptor 유지. 우리 기존 anchor(0.941 세대)를 0.943 스택으로 교체하는 것이 첫 단계. 상세는 보고서 7절.
