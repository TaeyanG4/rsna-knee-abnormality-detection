# 2026-10-02 13:10 KST · Raptor baseline 선정·로컬 학습 도구

- 상태: 진행(fold 0 학습 중). 기록자: Claude.
- 요청·결정: baseline 노트북 하나를 골라 1등 목표로 작업. 반복 작업은 도구화, 이미 받은 데이터 사용, 로컬 GPU(4070 Ti SUPER 16 GB) 학습, Kaggle T4(fp16) 결과 차이 유의.
- baseline: dreaddevelopment Raptor 레시피(CoAtNet rmlp_2 384 2.5D 창+소견별 attention pooling, 64×336 corpus, 140 mm, soft LLM 라벨). 근거: 학습 코드가 공개된 유일한 검증 레시피(단일 0.924), 0.943~0.946 공개 앙상블 전부가 이 체크포인트를 arm으로 사용, 토론에서 같은 구조+라벨 개선으로 단일 fold 0.950 보고.
- 도구: `training/src/raptor_{common,corpus,infer,train,report}.py`, [gpu_job.py](../../tools/gpu_job.py). 사용법 [runbook](../runbooks/training-raptor.md). GPU python에 pydicom·pylibjpeg·opencv-headless 설치. cache inspector가 평면 corpus를 인식하도록 수정, 출처 cache_origins 등록.
- corpus: `storage/cache/derived/RAPTOR64_336_C140_S02` 4,410 study(gold58·test3 선두), 12 shard SHA256, 523초. 1 study(…073439)는 전 시리즈 픽셀 해제 실패로 0 volume. 학습 fold −1(Gold58 보고서 중복 4건) 제외.
- parity(공개 maxspan-v5, Gold58, k_eval 62): fp16 0.9198 · bf16 0.9206 · fp32 0.9199, 헤더 0.9214. fp16↔fp32 확률 차 최대 0.0007. 기존 compact cache의 0.864는 corpus 계약 차이였다.
- 검증: 스모크 학습 통과, pytest 94·Playwright 33. 학습 15.0 GB·1.13 s/step → 약 19분/epoch.
- 실행 중: `EXP060_RAPTOR_F0`(fold 0, 384px, k 12/24, bs 4×2, 12 epoch, OOF 선택, bf16) PID 19340, 로그 `artifacts/logs/EXP060_RAPTOR_F0/run.log`, epoch 0 완료 15:32 KST, 12 epoch 완료 예상 19:30 KST 전후. 결과는 `artifacts/experiments/EXP060_RAPTOR_F0/summary.json`; `python tools/gpu_job.py --status EXP060_RAPTOR_F0`. 중단은 해당 PID 종료(체크포인트 epoch마다 저장).
- 추가(14:05): 웹 API(비공식 `/api/i` ListKernels)로 노트북 967개의 공개 점수 확보(465개). 상위 32개 전부 추론 전용 Raptor+DINO blend(0.946×1·0.945×3·0.943×26), 학습 코드가 있는 노트북 최고는 0.917. `source_meta.public_score/medal/runtime_s`, sync 선택 단계 `notebook_scores`, 라이브러리 추가. 원본 `.local/kneeops/web-notebook-scores-20261002.json`.
- 남은 일: fold 0 결과로 재현 여부 판단(기대 OOF≈0.88~0.90 weak, gold≈0.90+). 이후 빠른 루프(224px, bs 8, 8 epoch ≈ 2시간/fold)로 라벨 개선 실험. 다른 두 공개 arm corpus(384/44slot)는 미생성. 로컬 import·Kaggle 제출 미실행.
