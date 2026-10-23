# 2026-10-02 21:00 KST · 제출 anchor 교체·제출, R1 판정, 다음 학습

- 상태: 진행(제출 채점 대기, 학습 진행 중). 기록자: Claude.
- 요청: 0.943 스택으로 교체하고 진행, 제출까지(사용자 명시 승인).
- anchor: `jiweiliu/rsna-knee-fast-2xt4-inference`를 수정 없이 비공개 `taeyangg4/knee-anchor-0943-stack` v1로 복사(`kaggle/packages/KNEE_ANCHOR_0943`). 코드 점검: 외부 통신·비밀 접근 없음. 3 study 실행 4분, 검사 통과, fallback 0.
- 제출: ref 56773226, 2026-10-02 21:00 KST, 오늘 남은 제출 4. 채점 대기 중. 원본 출력과의 비교는 Kaggle 429로 미실행.
- R1 판정: EXP061_R1_F0 weak OOF 0.8806 vs EXP060 0.8963(같은 hash) → 차 0.016 > 0.01, **미채택**. fold당 41분. 5 fold는 teacher·앙상블 재료.
- 실행 중:
  - 로컬 큐 2(pid 37204, `artifacts/logs/queue-r0-384-20261002.json`): R1 F4 뒤 EXP060_RAPTOR_F1~F4(384, EXP060과 같은 명령) → `raptor_report.py --prefix EXP060_RAPTOR`. 약 15시간.
- ResNet34 1차(EXP070_R34_F0, bb lr 1e-4, 10 epoch): weak OOF 0.821, Gold58 0.816, epoch당 107초. 손실 0.898에서 정체해 학습 부족으로 판단, 2차 `taeyangg4/knee-raptor-run-exp071-r34-lr-f0`(bb lr 5e-4, 20 epoch) 실행.
- ResNet34 2차(EXP071_R34_LR_F0): weak OOF 0.8756, Gold58 0.8715, 38분. EXP060과 순위 상관 0.81(CoAt224는 0.91). fold 0 순위 평균 0.7/0.3 → 0.8981(Raptor 단독 0.8963, 같은 fold에서 고른 참고값). 두 번째 계열·teacher로 채택, fold 1·2 Kaggle 실행 중(`knee-raptor-run-exp071-r34-lr-f1`, `-f2`), fold 3·4는 그 뒤.
- 10-03 00:05: ResNet34 5 fold 완료(Kaggle, `EXP071_R34_LR_F0~F4`), 합친 weak OOF 0.8702. R1 5 fold 합친 OOF 0.8783. ResNet34 OOF로 `storage/labels/derived_mix/mix_r34_a50.csv`(blend 0.5) 생성·tools 데이터셋 반영. R2 fold 0 `taeyangg4/knee-raptor-run-exp072-r2mix-r34-f0`(R1 설정+혼합 라벨, T4 fp16) 실행 중. 판정은 원래 라벨로 재채점해 EXP061_R1_F0 0.8806과 비교(정밀도 bf16↔fp16 차이 주의).
- 제출 점수는 기다리지 않는다(사용자 지시, 채점 최대 약 9 h). 다음 작업 시작 때 한 번 확인.
- 다음: (1) 제출 점수 확인·기록(기대 0.943). (2) ResNet34 fold 0을 EXP060·R1과 비교해 두 번째 계열 판정. (3) 384 5 fold 완료 후 다른 계열 OOF로 R2 라벨 혼합, 자체 모델을 anchor에 순위 가중으로 추가.
- 10-03 00:20: Gold58에서 판독문 라벨 0.899 → 판독문+이미지 혼합 0.928(+0.029, 95% [0.013, 0.045]) 확인([보고서 8절](../research/baseline-validation-2026-10-02.md)). 재부팅 대비 `tools/queue_resume.py` 추가. 로컬 EXP060_RAPTOR_F1(pid 30880) epoch 7/12 진행 중, 01:45 완료 예상.
