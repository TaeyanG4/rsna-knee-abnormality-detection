# 도구 사용법

2026-10-02 기준. 반복 작업과 반복 검증은 아래 도구로 한다. 새 반복 작업이 생기면 먼저 도구로 만들고 이 문서에 한 줄 추가한다. 모든 명령은 워크스페이스 루트에서 실행한다.

2026-10-03 갱신: [지속 연구](continuous-research.md)·장시간 작업 재개·ensemble serving·D01 패키징·저장OOF 오류감사를 반영했다. 아래 과거 실측은 당시 관측이며 현재 할당·성능 보장이 아니다. [변경 전 원문](../handoffs/archive/snapshots/2026-10-03-continuous-research/manifest.json).

## 실행 환경

| 용도 | 인터프리터 |
|---|---|
| 대시보드·수집·Kaggle CLI | `.venv/Scripts/python -X utf8`, `.venv/Scripts/kaggle` |
| 학습·추론·corpus·원격 번들 | GPU Python `python`(system 3.12, torch 2.5.1+cu121, timm 1.0.19). `.venv`에는 torch·pandas가 없다 |

Kaggle 인증은 `~/.kaggle/access_token`(신형 API 토큰). 값을 출력·기록·커밋하지 않는다. Kaggle 작업 전에는 kaggle 스킬을 읽는다. Git Bash에서는 `export PYTHONIOENCODING=utf-8`을 먼저 둔다(Kaggle CLI 출력 인코딩 오류 방지).

## 대시보드 CLI

`.venv/Scripts/python -X utf8 -m monitoring.kneeops <명령>`. 앱 시작·종료는 `KneeOps.cmd`·`KneeOps-Stop.cmd`. [dashboard runbook](dashboard.md).

| 명령 | 용도 |
|---|---|
| `briefing --json` | 현재 순위·남은 제출·자료 수 요약. 작업 시작 시 먼저 본다 |
| `sync --topic-budget N --notebook-budget N --version-budget N` | Kaggle 노트북·토론·LB 수집 |
| `history <competition>` | 과거 대회 솔루션·토론 수집 |
| `sources search "q" --json`, `sources show "topic:ID" --json` | 수집 자료 검색·본문 |
| `experiments list --json`, `experiments compare A B --json` | 실험 목록·계약 비교 |
| `import` | `artifacts/experiments` 결과를 DB로 가져오기 |
| `schedule --json`, `data status --json`, `backup` | 정기 수집 상태, 데이터 상태, DB 백업 |

## 학습·평가 (GPU Python)

| 단계 | 명령 | 주의 |
|---|---|---|
| corpus | `training/src/raptor_corpus.py --contract raptor64_336_c140_s02 --workers 10` | 재실행은 이어서, `--verify` 재해시 |
| parity smoke | `training/src/raptor_infer.py --arm maxspan-v5 --subset gold --amp fp16` | 과거Gold58≈0.92 참고. aggregate AUC 일치만으로 입력/예측 parity를 보장하지 않음 |
| 학습 | `training/src/raptor_train.py --tag T --fold F ...` | `--select oof`만. `--select gold` 금지(ADR 0004). 224 px R1은 `--grad-ckpt` |
| 추론 | `training/src/raptor_infer.py --ckpt <run>/model_best.pt --subset gold\|oof\|test --amp fp16` | GPU가 학습 중이면 `--device cpu` |
| 실제 DICOM 추론 | `training/src/raptor_serve.py --help` | test UID/series와 DICOM 경로·checkpoint; Report 불필요. 단일 모델·strict 오류 중단 |
| DICOM ensemble | `python training/src/raptor_ensemble_serve.py --manifest <members.json> --test-csv <test.csv> --series-csv <test_series.csv> --dicom-root <test_series> --out <new.csv> --amp fp16 --device cuda --runtime-policy strict --batch 1 --residency resident` | study당 decode/입력1회·raw 확률 동일가중평균. 학습 중 로컬GPU에 추가 실행하지 않음 |
| D01 처리량·디코더 | `tools/d01_runtime_benchmark.py build --help` | 고정100건·T4 FP16·cold/VRAM/RSS·구문별 디코더. 라벨/성능평가 없음. full anchor9h 검증은 별도 |
| 입력 A/B 판정 | `tools/input_ab_report.py --manifest <manifest.json> --out <new.json>` | 사전 고정 코드/라벨/분할/초기값·조건 확인 후 report-group bootstrap. 개발fold 진단 |
| fold 합산 | `training/src/raptor_report.py --prefix EXP061_R1 --which best` | weak OOF·target별·스캐너 그룹별 AUC |
| A/B 진단 | `raptor_report.py --prefix A --compare B` | 현재 study bootstrap. CI가0 포함이면 이득 입증 부족, 누출/seed/domain 판단 별도 |
| 재채점 | `raptor_report.py --prefix A --labels <csv>` | 라벨 실험은 기준 라벨로 다시 채점해 비교 |
| fold 짝 비교 | `tools/fold_compare.py --baseline EXP061_R1 --candidate EXP073_R2PF` | 같은 fold·기준라벨 재채점, study bootstrap·Gold58 진단. teacher 계보·독립성은 별도 검사 |
| 고정 RN CPU 선별 | `python tools/pfoa_fixed_rank_screen.py prepare --help`, 이후 `run --plan <json> --plan-sha256 <sha> --out <new-dir>` | EXP078 네fit/선택arm·과거RN best를hash동결,평균동순위90/10 한 번. 새EXP079 fit 평가기로오용금지 |
| 저장OOF 오류감사 | `python tools/oof_error_audit.py --out-dir <new-dir> --bootstrap 500 --seed 20261003` | CPU만 사용. 기존R1/ResNet34/R2 best-only·원라벨 기준12소견/그룹/조건부CI·source hash 기록. 학습/추론/비율튜닝 안 함 |
| 라벨 혼합 | `training/src/label_mix.py --oof artifacts/predictions/<P>_oof_best.csv --alpha 0.5 --mode blend\|silence\|target --out storage/labels/derived_mix/<name>.csv` | UID중복·Gold혼입·95% cover 검사. teacher/ancestor fold 누출 검사는 미구현 |
| 스캐너 메타 | `training/src/study_meta.py` | 결과 `storage/labels/derived_meta/study_meta.csv` |
| hash 재계산 | `tools/rehash_summary.py TAG` | 10-02 이전 summary를 `lf-v2`로 |

ensemble manifest의 `members`에는 각 `name/path/sha256`을 넣고 path는 manifest 기준으로 해석한다. corpus·12target 순서·res·k_eval·norm·window 계약이 맞아야 한다. 원본ZIP은 `--dicom-root` 대신 `--archive <zip> --archive-index <index>`를 함께 쓴다. 출력은 새CSV·동일stem의 `.members.npz`·`.json`이며 ranking·혼합·제출은 하지 않는다. `resident`는 모든weights를GPU에 유지하고 `sequential`은batch마다 모델을 전송한다. 실제5fold 메모리/처리량은 실측하며 acquired DICOM 오류·member 실패·nonfinite는 중단한다.

오류감사는 기존 라벨·fold·metadata·corpus와 저장된15개fold 예측을 읽는다. 출력 경로가 이미 있으면 중단한다. weak rank 오류를 임상FP/FN으로 부르지 않으며 R2의 teacher누출·혼합라벨epoch선택은 재채점으로 제거되지 않는다. Gold58은 별도 노출 진단이다. [첫 감사와 해석](../research/oof-error-contribution-2026-10-03.md).

## D01 패키지와 실제T4 확인

- 로컬stage: `python tools/d01_package.py build --out <new-stage-dir> --wheels <offline-wheel-dir>`. 기존R2 F0~4·공개anchor·로컬F0 FP16 sample3 receipt/CSV가 필요하다. 참조 경로가 다르면 `--expected-receipt <json> --expected-csv <csv>`를 명시한다. timm1.0.19/pydicom3.0.2 wheels와 고정OpenCV dataset을 사용한다. 기존stage를 덮어쓰거나 Kaggle에 자동 업로드하지 않는다.
- 출력 `assets/`·`preflight/`·`candidate/`·`package-review.json`의 hash/계약을 확인한 뒤 원격 절차를 따른다. generated preflight는 실제Kaggle T4에서 `python tools/d01_package.py preflight --assets <assets> --competition <competition-dir> --work <new-work-dir>`를 실행한다. 로컬GPU나CPU 결과로 대체하지 않는다.
- `candidate-r2`는 같은 assets hash의 PASS preflight를 요구하며 `blend`는 완성R2 receipt·anchor를 검사한다. 후보는 `0.9×rank(anchor)+0.1×rank(mean5(raw R2))`, average ties·최종재rank 없음이다. 공개anchor30셀은 그대로 두고 앞/뒤2셀만 붙인다. 대회 제출은 별도 승인이다.
- [14:11 KST 로컬stage 검사](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-package-verification.json): 자산약1.469GB, CPU ensemble16+serving10+package9 통과. [14:15 비공개 업로드 시작](../handoffs/evidence/2026-10/2026-10-03-continuous-research/upload-launch.json) 이후 실제T4 검증은 대기 상태다. sample3 PASS만으로 대표표본/hidden 전체runtime이나 성능 향상을 보장하지 않는다. [원격 절차](remote-gpu.md).

## GPU 작업 관리

- `python tools/gpu_job.py --tag T -- <명령>`: 백그라운드 실행, 로그·환경 기록. `--status T`, `--list`. 결과는 `artifacts/logs/<T>/`의 `run.log`와 `SUCCESS` 또는 `FAILED`.
- `python tools/gpu_queue.py <plan.json>`: 선행 작업 SUCCESS를 기다려 순차 실행. 실패나 실행기 오류면 남은 체인을 중단한다(명시적 continue_on_failure 제외). 로그 `artifacts/logs/queue/<plan>.log`. 계획은 시작할 때 읽으므로 고친 뒤 러너를 재시작한다. 러너를 끄기 전 프로세스 명령줄로 러너인지 확인한다. 학습 프로세스는 따로 돌아 영향받지 않는다.
- 로컬 GPU에는 한 번에 학습 하나만 올린다. 학습 중 확인 작업은 CPU로 한다.
- CPU 확인용으로 `CUDA_VISIBLE_DEVICES="Invalid device id"로 즉시 실패한다(10-03 실제 발생). 러너는 `env -u CUDA_VISIBLE_DEVICES python tools/gpu_queue.py...`로 띄운다.
- 재부팅·강제 종료 뒤 재개는 `python tools/queue_resume.py <plan.json>`: 중단된 작업의 표식을 치우고(부분 결과는 `<TAG>_partial_<시각>`으로 보존) 러너를 다시 띄운다. 끝난 작업은 건너뛰고 중단된 fold는 처음부터 다시 돈다(학습기에 epoch 재개 없음).

## 원격 GPU

`python tools/remote_bundle.py <명령>`. 절차·실측은 [remote-gpu runbook](remote-gpu.md).

| 명령 | 용도 |
|---|---|
| `build --kind tools`, `publish --kind tools` | 코드·라벨이 바뀌면 비공개 tools 데이터셋 갱신. 라벨 혼합 CSV 포함, 라벨 외 열은 거부 |
| `kernel --tag T --train "..." --push` | 비공개 Kaggle T4 노트북으로 학습 하나 실행(fp16 강제) |
| `fetch-kernel --tag T` | 노트북 결과를 `artifacts/experiments/T`로 |
| `fetch-results [--tag T]` | Colab 실행 결과를 가져오기 |

## 장시간 작업·완료 후 재개 (2026-10-03 사용자 지시)

- 학습·대용량 다운로드·추론·제출 채점은 장시간 대기하지 않는다. 백그라운드 작업ID/프로세스, 로그/결과 경로, 예상 종료시각(KST), 허가된 다음 단계, 중단 조건을 같은 handoff에 기록하고 반환한다.
- 실행이 시작되면 Codex `automation_update`의 **현재 채팅 heartbeat**로 **예상 완료 20분 전**에 첫 확인/재개를 예약한다(10-03 사용자 확정). 확인 때 미완료면 실제 진행량/처리속도로 ETA를 갱신해 **갱신된 완료 시각**에 재예약한다. 20분 미만 작업은 실행 규모에 맞는 한 번의 완료 확인으로 처리하고 과거 시각·즉시 반복 예약을 만들지 않는다. 기존 같은 작업 예약을 갱신하고 중복 생성하지 않는다. 새 채팅·OS 예약 작업으로 우회하지 않으며 ID·ETA 없는 일반 방침만으로 예약하지 않는다.
- 예약 prompt에는 상태 확인 방법, 성공/실패 표식, 산출물 검사, 허가된 후속, 완료 시 예약 해제, 판단 필요 시 보고할 내용을 명시한다. 시간 경과만으로 완료 처리하지 않는다. 알림 설정은 prompt와 분리하고 완료·실패·사용자 판단 필요 시 알리며 상태 불변에는 조용히 유지한다.
- 깨어나면 실제 프로세스/원격 상태/로그를 한 번 확인한다. 미완료면 ETA를 갱신해 다음 확인을 예약하고 반환한다. 긴 sleep·촘촘한 polling·채점 대기 반복문을 사용하지 않는다.
- 성공이면 파일/지표/계약 검사 후 이미 허가된 다음 단계를 이어간다. 실행 중인 학습/큐를 먼저 검사해 중복 실행을 피한다. 실패/중단이면 원인을 기록하고 기존 허가 범위 밖의 재시도·유료 사용·새 제출을 시작하지 않는다. 사용자 중단 지시가 우선한다. 후속이 끝났거나 더 진행할 허가가 없으면 예약을 중지/해제한다.
- 로컬 예약은 PC와 앱이 실행 중이어야 한다. 정확한 완료 순간의 이벤트 알림이 아니라 예약 시각에 실제 완료 여부를 확인해 재개한다. 예약 생성에 실패하면 실패를 명시하고 ID/로그를 남긴다.
- [공식 OpenAI Docs](https://learn.chatgpt.com/docs/automations?surface=app): 기존 채팅 장시간 작업 follow-up, 로컬 PC·앱 실행 조건을 확인했다. 실제 예약ID와 갱신 ETA는 해당 작업 handoff에 남긴다.

## Kaggle 쓰기 작업

Notebook 산출물 하나를 회수할 때는 `.venv/Scripts/python.exe -X utf8 tools/kaggle_output_file.py --kernel owner/slug --version N --file relative/result.json --out <new-local-file>`을 쓴다. 정확한 버전/파일만 다운로드하고 SHA·시각을 기록한다. 기본16MiB 상한·기존파일 보존, 필요하면 명시적 `--max-bytes`/`--expected-sha256`를 준다. `kernels output` 전체회수는 설치된 dependency나 proxy 파일까지 열거해429/불필요한 다운로드를 만들 수 있다. 로그가 필요하면 SDK `ListKernelSessionOutput` 1페이지로 log만 회수하고, latest 관측과 버전고정 파일회수를 구분한다.

비공개 학습용 데이터셋·노트북 생성은 원격 학습 범위로 허용됐다(2026-10-02). 대회 제출, 공개 게시, 공개 범위 변경은 매번 사용자 승인을 받는다.

2026-10-03 후속 지시: 서로 다른 사전 고정 가설·검사된 후보가 있으면 하루 최대5회 기회를 활용한다. live 잔여 한도·pending을 확인하고 완성된 후보에 매회 승인을 받는다. 횟수 소진이나 public 결과에 맞춘 비율/소견 반복튜닝은 하지 않는다. [검증 규칙](../decisions/0004-validation-protocol.md).

**제출 후 점수를 기다리지 않는다**(2026-10-03 사용자 지시). 채점은 숨겨진 테스트 재실행으로 최대 약9시간 걸린다. PENDING이면 ref·시각·예상 완료시각을 handoff에 남기고 위 heartbeat 확인을 예약한 뒤 반환한다. 재개 때 `kaggle competitions submissions rsna-knee-abnormality-detection --csv` 한 번으로 실제 상태를 확인한다. 미완료면 다음 확인을 예약하며 대기 반복문은 쓰지 않는다. 점수 확인이 새 제출의 허가는 아니다. 이전 ‘예약 감시 없음’ 지침은 후속 사용자 지시로 대체했다.

## 검사

| 대상 | 명령 |
|---|---|
| 학습 도구 | `python -m pytest tests/test_training_tools.py -q -p no:cacheprovider --basetemp=.local/pytest-tmp` |
| 앱 Python | `.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp=.local/pytest-tmp` |
| 웹 | `monitoring/web`에서 `npm run build`, `npm run test:ui` |

샌드박스에서는 기본 pytest 임시 폴더가 권한 오류를 낸다. `--basetemp`를 쓰고 끝나면 그 폴더를 지운다. 문서만 바꿨으면 링크·분량만 확인한다.
