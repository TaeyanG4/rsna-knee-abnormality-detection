# EXP086 CPU 준비와 이후 실행

현재 상태: CPU 준비만 완료. GPU/학습/다운로드/원격 실행 0. 프로토콜과 독립 평가기 검토 후 root가 각 phase 결정을 기록한다. 이 문서는 실행 승인 파일이 아니다.

- 프로토콜: `workspace://artifacts/experiments/EXP086_DINOV2S_G1_PROTOCOL/manifest.json`
- SHA256: `0de375945192a050e7ca22944cc1990e3a67d0c125d333e83bf4a796719e5401`
- 정확한 argv/예정 결정 경로: [execution-plan.json](execution-plan.json). argv를 그대로 사용하고 shell 문자열로 재조립하지 않는다.
- AST 인자와 원본/입력 바이트 검증: [static-command-validation.json](static-command-validation.json).

## Root의 다음 단계

1. producer 독립 계약검토와 평가기 registration을 확인한다. 기존 active GPU 작업·출력·resource·로그 경로를 확인한다. pilot/full 결정 파일은 producer가 생성하지 않았다.
2. `root-pilot-decision.json`에 실제 protocol SHA, `allow_phase=pilot`, `retries=0`을 기록한 후 plan의 pilot argv를 `subprocess.run(argv, check=True)`로 실행한다. `gpu_job.py`가 백그라운드를 담당한다.
3. pilot은 전체8epoch와 같은3448step scheduler의 최초32update만 실행한다. warmup8을 제외한24개 구간은 DataLoader 대기를 포함한다. T의 최초4study만 유한 추론 확인하며 V 추론·metric·모델저장·예측저장은 없다.
4. pilot 로그/receipt/provenance와 `_PILOT_RESOURCE/receipt.json`, GPU-job `SUCCESS`/`job.json`을 회수한다. 종료0뿐 아니라 실제 argv, protocol/초기값, 시간, skips0, resource/cleanup을 확인한다. `1.3*(steady*3448)+300<=5400`을 적용한다. 정확한 GPU FP16/시간 gate는 여기서 처음 측정된다.
5. root가 이후 `root-full-decision.json`을 별도로 작성한다. protocol/phase/retries 외 실제 pilot receipt SHA를 묶는다. plan full argv는 고정8epoch/3448update 1fit이며 cap5400초, pilot+full5580초이다. 재시도·precision/batch 전환·추가seed/fold는 없다.
6. 실제 완료시각 추정에 따라 같은 채팅 heartbeat를 예약하고 반환한다. full 첫 확인은 ETA20분 전, 미완료 시 갱신 ETA. 3분 cap pilot은 작업 규모에 맞는 한 번의 완료 확인이다.
7. full `summary/provenance/history/trace/model_best/oof_best/receipt`와 resource/job 근거를 확인하고 사전 등록 평가기를 한 번 실행한다. 후보−원EXP084 B의 raw12macro/각 소견·P/N·Baker/Fracture·group·paired component500 CI만 판정한다. 순위혼합·비율·epoch 탐색은 없다.

## 이미 끝난 확인 — 같은 계약으로 반복하지 않음

- `python -X utf8 -m pytest tests/test_exp086_dinov2s.py -q -p no:cacheprovider --basetemp=.local/pytest-exp086-v3`: 40 passed, 1.79초.
- [CPU probe](cpu-initial-probe.json): 합성1study×2window, 35.235초·12thread. native forward/head exact, FP32 gradients finite/nonzero, CPU FP16 diagnostic PASS, optimizer0·CUDA 미초기화. backward를 위해 모델의 train mode는 사용했다. receipt의 `model_training=false`는 실제 데이터 학습/optimizer 실행이 없다는 뜻으로만 해석한다.
- [1thread 초기값 연결](initial-thread1-binding.json): forward/backward 반복 없이 constructor 0.313초. 실제 runner의1thread와 probe12thread 초기state 동일.
- AST 명령 검증의 최초 harness는 임의로 예상한 argparse 문장40개와 실제38개가 달라 중단했다. 해당 부정확한 count assertion 제거 후 통과했으며 producer/runner/protocol 변경이나 모델 실행은 없었다.

## 범위와 해석

generic DINOv2-S encoder-family transfer package이며 pretraining만 분리한 대조나 FLOPs 일치 대조가 아니다. 기존T3455/V888·원라벨·224/K12/K24·정규화·sampling·loss/optimizer·fixed8을 유지한다. V soft-label CSV 값은 읽고 zero target으로 치환하므로 **V 값 never-read 주장 금지**. V metric/중간 V 추론/Gold 접근이 금지된다.

같은 V는 이미 연구자에게 노출된 weak 개발검증이다. 독립 임상/private/D01/0.943 개선 근거가 아니며, 새 V를 쓰려면 모든 관련 ancestor에서 제외하고 노출 comparator도 재학습해야 한다. 로컬 FP16 pilot은 T4/4070 parity 증명이 아니다.
