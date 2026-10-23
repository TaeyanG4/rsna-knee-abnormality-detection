# EXP086 V2 실행 인계

CPU 준비만 완료했다. GPU를 실행하지 않았으며 root의 새 phase 결정이 필요하다. V1 failed pilot의 승인/retry0을 재사용하지 않는다.

- [execution-plan.json](execution-plan.json)의 `phase_commands.{pilot,full}.argv`가 정확한 외부 명령이다. argv 배열을 `subprocess.run(argv, check=True)`로 전달하면 `gpu_job.py`가 백그라운드 실행한다. shell 문자열로 재조합하지 않는다.
- [static-command-validation.json](static-command-validation.json)은 실제 runner의 argparse AST만 추출해 pilot/full worker 인자를 검증했다. 모델/runner 실행이나 Torch import가 없다.
- 프로토콜: `workspace://artifacts/experiments/EXP086_DINOV2S_G1_V2_PROTOCOL/manifest.json`, SHA `0383b55ba820c19f262fd36e4cc27bf604d2ae109302d7109f47e5423390cbc3`.
- root 결정 예정 경로는 이 디렉터리의 `root-pilot-decision.json`, `root-full-decision.json`이다. producer가 생성하지 않았다. 두 결정은 protocol SHA/allow_phase/retries0을, full은 실제 완료 V2 pilot receipt SHA도 묶는다.

## 실행 순서

1. 독립 producer/평가기 검토·registration 완료, 실제 active GPU jobs와 새 V2 출력/resource/log 경로를 확인한다. CUDA_VISIBLE_DEVICES가 빈값으로 상속되지 않게 한다. 동시학습1개.
2. root pilot 결정 후 plan pilot argv를 실행한다. 한 번의 T32update·full3448 schedule·T4study inference, 최대180초, skip0, strict/deterministic=True, warn_only=False, retry0. V/Gold metrics·checkpoint/예측저장 없음.
3. 실제 worker argv/초기state/position_policy/protocol·resource timeout/소유PID/exit/telemetry/자연종료/cleanup을 확인한다. `SUCCESS`만으로 통과시키지 않는다. V2 producer는 step32구간을 재계산해 projection도 확인하며 비용불합격은 외부job 실패로 남긴다.
4. 실제 steady는 warmup8을 제외한24개 전체update 구간의 평균이다. loader wait·CPU/GPU복사·보간/gradient·GPU동기화 포함. `1.3*(steady*3448)+300<=5300` 확인 후 root가 별도 full 결정한다.
5. full은8epoch/3448update·최대5300초(88분20초). **원V1 실제83.23500000000058초+V2pilot실제+V2full실제<=5580초**. 세 cap 합5563.235초이며 리뷰 대기는 active 실행예산과 구분한다. 자동retry·정밀도/배치변경·추가seed/fold 없음.
6. root가 실제ETA에 맞춰 같은 채팅 heartbeat를 첫ETA20분 전/미완료시갱신ETA로 예약하고 반환한다. 3분cap pilot은한번의완료확인. producer가 별도예약/실행하지 않았다.
7. full 완료후7산출물(`summary/provenance/history/trace/model_best/oof_best/receipt`), job/resource/phase결정/바이트hash를 검사한 뒤 사전등록 evaluator를1회 실행한다. 원 B예측재사용, raw12macro·소견/P/N/Baker/Fracture/group·paired500CI, 비율/순위/epoch탐색 없음.

## 위치 연산 계약

`training/adapters/dinov2_cpu_pos_v2.py`는 후보instance에만 method를 설치한다. canonical518 trainable `[1,1370,384]`와22,163,224개 전체파라미터/초기SHA를유지한다. 같은timm FP32bicubicAA를CPU내부1thread로계산하고 autograd `.to`로원device/원position dtype에복귀한다. detach/NumPy/no_grad/캐시/전역monkeypatch 없음. train/eval은같은adapter를쓴다.

state shape가같아도 이후추론factory가position policy를설치해야한다. 현재shared `raptor_common.load_checkpoint`의GPU보간을그대로쓰는serving은계약상동일하지않다. 새T4serving/parity 준비완료를주장하지않는다.

## 완료 검사와 한계

- 신규15tests PASS2.47초. V1전체40tests/전체CPUprobe는재실행하지않았다.
- changed-op+1study×1window syntheticmodel1회:0.969초, CPU strict/1thread. CPU원보간대비출력/gradient exact, canonical위치gradientfinite/nonzero, initialSHA동일, parameter동일, optimizer0/GPU0.
- CPU검사는실제CUDA→CPU→CUDA copy backward를실행하지않았다. 그경로와CUDA FP16/no-skip·실제시간은V2pilot이처음검증한다.
- 같은노출weakV이며patient/clinical/private/D01/0.943개선이나T4/4070bitwise동등성의증거가아니다.
