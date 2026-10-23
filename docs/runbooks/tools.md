# 도구 사용법

반복 작업과 검증은 아래 도구로 한다. 모든 명령은 작업 공간 루트에서 실행한다. 일반 연구 도구는 2026-10-03, 로컬 에이전트 세션은 2026-10-04 갱신했다. 과거 실측은 현재 할당·성능을 보장하지 않는다. [지속 연구](continuous-research.md) · [이전 원문](../handoffs/archive/snapshots/2026-10-03-continuous-research/manifest.json).

## 반복 작업의 도구화

반복 조회·추출·검사·집계는 기존 도구를 검색하고, 반복이 예상되며 구현 비용보다 절약 효과가 크면 작은 CLI로 만들거나 기존 도구를 확장한다. 같은 shell 코드를 매번 새로 작성하거나 대용량 로그를 대화에 반복 출력하지 않는다.

- 입력·출력·적용 범위·실패 시 종료코드·실행 예시를 명시한다. 가능하면 조회는 읽기 전용으로 만들고 변경 동작은 명시적으로 구분한다.
- 한 번 계산한 유효 결과를 재사용한다. 계약/hash가 달라지면 필요한 범위만 다시 검사한다. 새 도구 때문에 동결 실험을 바꾸거나 완료 실험을 다시 실행하지 않는다.
- 기본 출력은 짧은 요약, 자동 처리에는 JSON을 제공한다. 원 로그와 산출물은 파일로 보존하고 위치만 연결한다. 도구 생성 자체를 성능 개선으로 보고하지 않는다.
- 새 도구는 실제 실패를 막는 작은 검사로 검증하고 이 문서에 등록한다. 일회 작업을 위한 큰 프레임워크는 만들지 않는다.

문서 점검은 다음 도구로 묶는다.

```powershell
.venv/Scripts/python.exe -X utf8 tools/docs_check.py
.venv/Scripts/python.exe -X utf8 tools/docs_check.py --json
.venv/Scripts/python.exe -X utf8 tools/docs_check.py AGENTS.md HANDOFF.md
```

`docs_check.py`는 기본 진입 문서와 운영 문서의 UTF-8·LF 정규화 길이·기본 Markdown·필수 구획·단순 상대 링크의 파일 경로·폐지된 정기 인계 지시를 읽기 전용으로 검사한다. 정상은 exit 0, 문제는 exit 1이다. 웹 주소·앵커 이름·복잡한 Markdown·내용의 사실성은 검사하지 않는다. 문서 규격의 원본은 [작성·인계 규격](handoff-maintenance.md)이다.

## 로컬 에이전트 세션

전체 역할·권한·읽기 순서는 [AGENTS](../../AGENTS.md)가 원본이다. 상위 세션이 아래 도구를 실행하면 **같은 프로젝트에서 새 Codex 또는 Claude Code CLI 세션을 시작하고 결과를 회수**한다. 사용자는 현재 대화에 머물 수 있다. 조정자는 자원 실행·기록을 맡고 자식은 조사·구현·검토를 맡는다. 단순 작업에 별도 세션을 항상 만들지는 않는다. 상위 세션에 셸 실행 권한이 필요하며, 제품 간 대화 기록을 공유하는 방식은 아니다. 필요한 맥락은 파일과 작업 지시로 전달한다.

```powershell
.venv/Scripts/python.exe -X utf8 tools/agent_session.py doctor
.venv/Scripts/python.exe -X utf8 tools/agent_session.py start --provider claude --role reviewer --task-file docs/handoffs/evidence/2026-10/2026-10-04-document-roles-agent-cli/review-task.txt --timeout 240
.venv/Scripts/python.exe -X utf8 tools/agent_session.py status RUN_ID
.venv/Scripts/python.exe -X utf8 tools/agent_session.py result RUN_ID
```

예시 task는 이번 도구의 정적 검토용이다. 새 작업은 목적·사전/사후 검토 단계·관련 근거·허용 수정 파일·종료 조건을 적은 UTF-8 task 파일(작업 공간 안, 64KiB 이하)을 만든다. `--provider codex`도 같은 형식이다. 역할은 `researcher|reviewer|implementer`; 기본 읽기 전용이며 구현 변경은 `--role implementer --write`를 명시한다. `--dry-run`은 실행할 명령만 보여준다.

모델/effort를 덮어쓰지 않고 기존 CLI 설정·인증을 사용한다. CLI와 계정을 자동 설치하거나 설정하지 않으며 해당 계정 사용량을 소모할 수 있다. 로그인/중첩 세션 제한 등으로 실행이 거부되면 실패를 반환한다. Claude Code 안에서 다시 Claude를 시작하는 경우의 제한을 우회하지 않는다.

- `start`는 숨겨진 백그라운드 worker를 시작하고 실행ID를 반환한다. `status`는 조회 1회, `result`는 종료 후 응답 회수다. 예약·중앙 기록은 조정자가 관리한다.
- `.local/agent-sessions/RUN_ID/`에 `job.json`, `launch.json`, `prompt.txt`, `events.jsonl`, `stderr.log`, `worker.log`, `result.txt`를 저장한다. provider·역할·명령·task hash·PID·native session ID·시각·종료 상태를 추적할 수 있다.
- timeout은 기본 600초, 10~3600초 범위다. Claude는 추가로 최대 16턴이다. 기한/턴 상한/사용량 제한에 도달해도 자동 재시도하거나 provider를 전환하지 않는다. 부분 응답은 실패 상태와 함께 보존한다.
- `cancel RUN_ID`는 worker에 취소를 요청한다. 정상 worker는 자신이 시작한 자식에 Windows `taskkill /T` 또는 POSIX process-group 종료를 적용한다. 다른 외부 CLI/GPU 작업을 찾아 종료하지 않는다. Windows Job Object를 사용하지 않아 worker 강제 종료나 부모가 이미 종료된 후손까지 정리를 보장하지는 않는다.
- 작업 공간당 이 도구의 child 1개만 실행한다. `list`는 자체 작업 목록이다. worker가 중단되면 `status`로 PID/로그를 확인하고 `recover RUN_ID`로 남은 잠금을 회수한다. 기록된 PID가 살아 있으면 거부한다. PID 재사용을 판별하지 않으므로 거부 시 실제 명령·시작 시각을 대조해야 한다. 이 잠금은 직접 시작한 외부 CLI를 통제하지 않는다.
- `job.json` 생성 전 비정상 종료·살아 있는 고아 child 등은 자동 복구 대상이 아니다. 조정자가 `active.lock`과 로그의 PID·명령·시작 시각으로 소유권과 종료를 확인한 뒤 해당 작업만 정리한다. 확인 없이 잠금을 지우거나 새 실행을 겹치지 않는다.
- 자식은 중앙 문서 변경안을 응답으로 반환한다. 루트 5개 문서(AGENTS/CLAUDE/HANDOFF/plan/discoveries)의 hash 변화는 `needs_attention`으로 표시한다. 부모의 상태 변경 기록도 포함되며 위반 확정 판정은 아니다. STATE/색인·할당 밖 파일은 이 검사에 포함되지 않는다. 조정자가 변경 파일과 범위를 검토하며 자동 rollback은 하지 않는다.

**권한 범위:** Codex에는 `exec --sandbox read-only`(구현은 `workspace-write`)와 승인 요청 거부 정책을 지정한다. Claude는 Read/Glob/Grep만 제공하고 구현 때 Edit/Write를 추가한다. Claude 구현의 테스트 명령은 상위 세션이 실행한다. 읽기 전용은 프로젝트 밖 파일의 읽기 차단을 뜻하지 않는다. Codex의 명령 실행·GPU/원격 사용과 할당 파일 제한은 작업 지시로 통제하며 이 도구가 별도 자원 격리를 제공하지 않는다. 비밀 파일 조회·출력은 금지한다. Claude 도구 제한도 OS 격리가 아니며 기존 hooks까지 격리하지 않는다. Windows에서의 모든 샌드박스 경계를 검증한 것은 아니다. 승인/샌드박스 우회 flag를 사용하지 않는다.

exit 0만으로 완료 처리하지 않는다. 정상 종료 event와 최종 응답이 있어야 `completed`, 권한 거부/중앙 문서 변화는 `needs_attention`, 오류/응답 부재는 `failed`, 기한/취소는 별도 상태다. `completed`는 응답 회수 성공이며 코드 정확성·연구 성능의 증명이 아니다. CLI session ID는 로컬 CLI ID로, 앱 채팅 첨부나 자동 화면 전환을 뜻하지 않는다.

공식 근거: [Codex 비대화형 실행](https://learn.chatgpt.com/docs/non-interactive-mode), [Claude headless](https://code.claude.com/docs/en/headless), [Claude CLI](https://code.claude.com/docs/en/cli-reference). [관측 원문/hash](../handoffs/evidence/2026-10/2026-10-04-document-roles-agent-cli).

## 실행 환경

| 용도 | 인터프리터 |
|---|---|
| 대시보드·수집·Kaggle CLI | `.venv/Scripts/python -X utf8`, `.venv/Scripts/kaggle` |
| 학습·추론·corpus·원격 번들 | GPU Python `python`(system 3.12, torch 2.5.1+cu121, timm 1.0.19). `.venv`에는 torch·pandas가 없다 |

Kaggle 인증은 `~/.kaggle/access_token`(신형 API 토큰). 값을 출력·기록·커밋하지 않는다. Kaggle 작업 전에는 kaggle 스킬을 읽는다. Git Bash에서는 `export PYTHONIOENCODING=utf-8`을 먼저 둔다(Kaggle CLI 출력 인코딩 오류 방지).

## CPU 워커 기준 (2026-10-04 사용자 지시)

- 새 CPU 병렬 작업은 **12워커를 기본값**으로 한다. 처리량이 떨어지면 **10→8워커**로 낮추며, 메모리·디스크 I/O·실제 CPU 할당이 더 작으면 그 이하도 사용한다. 작은 순차 작업은 프로세스 시작 비용이 이득보다 크면 순차로 처리한다.
- 워커 수와 BLAS/OpenMP/PyTorch 내부 스레드는 구분한다. 프로세스별 12스레드를 겹쳐 12×12로 과다 실행하지 않는다. 여러 작업·에이전트가 동시에 돌면 전체 CPU·RAM·I/O 예산을 합산하고, 원격은 실제 할당을 상한으로 한다.
- 실제 처리량(건/초), 경과시간, peak RAM, I/O 대기를 보고 조정한다. 12워커가 항상 빠르다고 가정하지 않는다. 새 작업 로그에 요청/실효 워커 수와 감축 이유를 기록하며, 워커 비교만을 위해 완료 작업을 반복하지 않는다.
- 이미 동결한 실험 코드·프로토콜·실행 중인 작업은 이 선호만으로 변경/재시작하지 않는다. 적용이 필요하면 새 실행 계약에서 명시한다. 아래 corpus 예시는 향후 실행 기본값이며 과거 실측의 변경이 아니다.

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
| corpus | `training/src/raptor_corpus.py --contract raptor64_336_c140_s02 --workers 12` | 재실행은 이어서, `--verify` 재해시 |
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
| 새 RN 고정 대조 평가 | `python tools/exp079_complement_report.py --help` | EXP079 신규2fit·EXP078 B/A와 원판정/hash 계약을 확인한 후 고정90/10·12소견/P/N/그룹/조건부CI 계산. F2 후속에 그대로 사용하지 않음 |
| R2 두T4 분할 실측 | `python tools/d01_r2_shard_benchmark.py build --help` | 기존runtime worker·legacy reader를 고정100의50+50으로 나눠 측정. stage만생성, 원격push는별도. 입력/확률·실효CPU·프로세스종료·cold시간·메모리 검사, full9h PASS와구분 |
| 저장OOF 오류감사 | `python tools/oof_error_audit.py --out-dir <new-dir> --bootstrap 500 --seed 20261003` | CPU만 사용. 기존R1/ResNet34/R2 best-only·원라벨 기준12소견/그룹/조건부CI·source hash 기록. 학습/추론/비율튜닝 안 함 |
| 라벨 혼합 | `training/src/label_mix.py --oof artifacts/predictions/<P>_oof_best.csv --alpha 0.5 --mode blend\|silence\|target --out storage/labels/derived_mix/<name>.csv` | UID중복·Gold혼입·95% cover 검사. teacher/ancestor fold 누출 검사는 미구현 |
| 스캐너 메타 | `training/src/study_meta.py` | 결과 `storage/labels/derived_meta/study_meta.csv` |
| hash 재계산 | `tools/rehash_summary.py TAG` | 10-02 이전 summary를 `lf-v2`로 |

ensemble manifest의 `members`에는 각 `name/path/sha256`을 넣고 path는 manifest 기준으로 해석한다. corpus·12target 순서·res·k_eval·norm·window 계약이 맞아야 한다. 원본ZIP은 `--dicom-root` 대신 `--archive <zip> --archive-index <index>`를 함께 쓴다. 출력은 새CSV·동일stem의 `.members.npz`·`.json`이며 ranking·혼합·제출은 하지 않는다. `resident`는 모든weights를GPU에 유지하고 `sequential`은batch마다 모델을 전송한다. 실제5fold 메모리/처리량은 실측하며 acquired DICOM 오류·member 실패·nonfinite는 중단한다.

오류감사는 기존 라벨·fold·metadata·corpus와 저장된15개fold 예측을 읽는다. 출력 경로가 이미 있으면 중단한다. weak rank 오류를 임상FP/FN으로 부르지 않으며 R2의 teacher누출·혼합라벨epoch선택은 재채점으로 제거되지 않는다. Gold58은 별도 노출 진단이다. [첫 감사와 해석](../research/oof-error-contribution-2026-10-03.md).

## D01 패키지와 실제T4 확인

- 로컬stage: `python tools/d01_package.py build --out <new-stage-dir> --wheels <offline-wheel-dir>`. 기존R2 F0~4·공개anchor·로컬F0 FP16 sample3 receipt/CSV가 필요하다. 참조 경로가 다르면 `--expected-receipt <json> --expected-csv <csv>`를 명시한다. timm1.0.19/pydicom3.0.2 wheels와 고정OpenCV dataset을 사용한다. 기존stage를 덮어쓰거나 Kaggle에 자동 업로드하지 않는다.
- 출력 `assets/`·`preflight/`·`candidate/`·`package-review.json`의 hash/계약을 확인한 뒤 원격 절차를 따른다. generated preflight는 실제Kaggle T4에서 `python tools/d01_package.py preflight --assets <assets> --competition <competition-dir> --work <new-work-dir>`를 실행한다. 로컬GPU나CPU 결과로 대체하지 않는다.
- `candidate-r2`는 같은 assets hash의 PASS preflight를 요구하며 `blend`는 완성R2 receipt·anchor를 검사한다. 후보는 `0.9×rank(anchor)+0.1×rank(mean5(raw R2))`, average ties·최종재rank 없음이다. 공개anchor30셀은 그대로 두고 앞/뒤2셀만 붙인다. 당시 제출은 별도승인 대상이었고, 10-04 이후 준비된 후보 상시승인을 적용한다. D01은 이미 실패 완료되어 새근거 없는 재제출은 금지한다.
- [14:11 KST 로컬stage 검사](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-package-verification.json): 자산약1.469GB, CPU ensemble16+serving10+package9 통과. [14:15 비공개 업로드 시작](../handoffs/evidence/2026-10/2026-10-03-continuous-research/upload-launch.json) 이후 실제T4 검증은 대기 상태다. sample3 PASS만으로 대표표본/hidden 전체runtime이나 성능 향상을 보장하지 않는다. [원격 절차](remote-gpu.md).

## GPU 작업 관리

EXP088의 새 자원 감시는 `tools/exp088_resource_v2.py`를 통해 Windows private commit 합계와 호스트 가용 RAM을 구분한다. 100ms 표본이며 tree RSS는 진단값이다. 정확 argv/시간·두 메모리 한도/환경/출력 폴더를 받아 종료 receipt와 30초 telemetry를 남긴다. 측정/child/cleanup 실패는 FAIL, 재시도 없음. `tools/exp088_raptor384_v4.py prepare`로 새 protocol과 실행 계획만 만들고 root의 정확 phase 승인으로 실행한다. epoch 복구 상태는 원자적으로 저장하며 자동 resume/중간 V 선택은 없다. 원 실패/완료 pilot은 재사용한다. [새 계약](../../plan.md#exp088_raptor384_g1).

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
- 실행이 시작되면 Codex `automation_update`의 **현재 채팅 heartbeat**로 **예상 완료 후30분**에 첫 확인/재개를 예약한다(10-07 사용자 지시,10-08 재확인). 확인 때 미완료면 실제 진행량/처리속도와 전체 미실행 예약으로 ETA를 갱신해 **갱신된 완료 시각+30분**에 재예약한다. 예상 완료 뒤 짧은 종료처리만 남고 새 ETA가 불명확할 때30분 간격으로 확인하며, 긴 작업 내내30분마다 조회하지 않는다. 6시간 방향성검토 예약은 별도로 유지한다. 짧은 작업은 실행 규모에 맞는 한 번의 완료 확인으로 처리하고 과거 시각·즉시 반복 예약을 만들지 않는다. 기존 같은 작업 예약을 갱신하고 중복 생성하지 않는다. 새 채팅·OS 예약 작업으로 우회하지 않으며 ID·ETA 없는 일반 방침만으로 예약하지 않는다.
- 예약 prompt에는 상태 확인 방법, 성공/실패 표식, 산출물 검사, 허가된 후속, 완료 시 예약 해제, 판단 필요 시 보고할 내용을 명시한다. 시간 경과만으로 완료 처리하지 않는다. 알림 설정은 prompt와 분리하고 완료·실패·사용자 판단 필요 시 알리며 상태 불변에는 조용히 유지한다.
- 깨어나면 실제 프로세스/원격 상태/로그를 한 번 확인한다. 미완료면 ETA를 갱신해 다음 확인을 예약하고 반환한다. 긴 sleep·촘촘한 polling·채점 대기 반복문을 사용하지 않는다.
- 성공이면 파일/지표/계약 검사 후 이미 허가된 다음 단계를 이어간다. 실행 중인 학습/큐를 먼저 검사해 중복 실행을 피한다. 실패/중단이면 원인을 기록하고 기존 허가 범위 밖의 재시도·유료 사용·새 제출을 시작하지 않는다. 사용자 중단 지시가 우선한다. 후속이 끝났거나 더 진행할 허가가 없으면 예약을 중지/해제한다.
- 로컬 예약은 PC와 앱이 실행 중이어야 한다. 정확한 완료 순간의 이벤트 알림이 아니라 예약 시각에 실제 완료 여부를 확인해 재개한다. 예약 도구가 없거나 생성에 실패하면 자동 재개 불가를 명시하고 ID/로그·수동 재개 명령을 남긴다. 자식은 조정자에게 예약 정보를 반환한다. 다른 앱/OS 예약으로 우회하거나 예약된 것처럼 보고하지 않는다.
- [공식 OpenAI Docs](https://learn.chatgpt.com/docs/automations?surface=app): 기존 채팅 장시간 작업 follow-up, 로컬 PC·앱 실행 조건을 확인했다. 실제 예약ID와 갱신 ETA는 해당 작업 handoff에 남긴다.

## Kaggle 쓰기 작업

노트북 push 전 실제 SDK 전송 형식(outputs 제거·source 문자열 병합·JSON 직렬화)의 bytes를 검사한다. 로컬 보수 예산은1,000,000bytes 미만이다. [Kaggle 직원의1MB 제한 설명](https://www.kaggle.com/discussions/product-feedback/431257)과10-03의1.90MB/1.52MB SaveKernel400 사례를 참고한다. 큰 검증자료는 무손실 묶음압축이나 비공개 자산으로 분리하며, 기존 payload/hash/gate를 줄이거나 버리지 않는다. 통합133은동일17파일 tar+LZMA/base85 포장으로실제전송707,520bytes가되어 version1저장에성공했다. HTTP400을runtime실패로혼동하지않고, 실패stage·원인근거·새포장SHA를보존한다.

Notebook 산출물 하나를 회수할 때는 `.venv/Scripts/python.exe -X utf8 tools/kaggle_output_file.py --kernel owner/slug --version N --file relative/result.json --out <new-local-file>`을 쓴다. 정확한 버전/파일만 다운로드하고 SHA·시각을 기록한다. 기본16MiB 상한·기존파일 보존, 필요하면 명시적 `--max-bytes`/`--expected-sha256`를 준다. `kernels output` 전체회수는 설치된 dependency나 proxy 파일까지 열거해429/불필요한 다운로드를 만들 수 있다. 로그가 필요하면 SDK `ListKernelSessionOutput` 1페이지로 log만 회수하고, latest 관측과 버전고정 파일회수를 구분한다.

비공개 학습용 데이터셋·노트북 생성은 원격 학습 범위로 허용됐다(2026-10-02). 대회 제출은 2026-10-04 [준비된 후보 상시승인](../research/submission-authorization-2026-10-04.md)을 적용한다. 공개 게시·공개 범위 변경·유료 사용은 별도승인이다.

2026-10-03 후속 지시: 서로 다른 사전 고정 가설·검사된 후보가 있으면 하루 최대5회 기회를 활용한다. live 잔여 한도·pending을 확인하고 완성된 후보는 10-04 상시 제출 승인 범위에서 정확버전·현한도·중복을 확인한다. 횟수 소진이나 public 결과에 맞춘 비율/소견 반복튜닝은 하지 않는다. [검증 규칙](../decisions/0004-validation-protocol.md).

**제출 후 점수를 기다리지 않는다**(2026-10-03 사용자 지시). 채점은 숨겨진 테스트 재실행으로 최대 약9시간 걸린다. PENDING이면 ref·시각·예상 완료시각을 handoff에 남기고 위 heartbeat 확인을 예약한 뒤 반환한다. 재개 때 `kaggle competitions submissions rsna-knee-abnormality-detection --csv` 한 번으로 실제 상태를 확인한다. 미완료면 다음 확인을 예약하며 대기 반복문은 쓰지 않는다. 점수 확인이 새 제출의 허가는 아니다. 이전 ‘예약 감시 없음’ 지침은 후속 사용자 지시로 대체했다.

## 검사

| 대상 | 명령 |
|---|---|
| 학습 도구 | `python -m pytest tests/test_training_tools.py -q -p no:cacheprovider --basetemp=.local/pytest-tmp` |
| 앱 Python | `.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp=.local/pytest-tmp` |
| 웹 | `monitoring/web`에서 `npm run build`, `npm run test:ui` |

샌드박스에서는 기본 pytest 임시 폴더가 권한 오류를 낸다. `--basetemp`를 쓰고 끝나면 그 폴더를 지운다. 문서만 바꿨으면 링크·분량만 확인한다.

## B/RN 입력·완료 연결 (2026-10-04)

기존6weights serving은 전체testUID의 fold별90/10 rank 후3fold평균이다. `tools/b_rn3f_parity.py` V1의 CPU1thread 수치PASS와 실제12thread producer 해시는 미세한 bilinear 연산차로100건 모두 달랐다. V1 실패를 보존하고 `tools/b_rn3f_thread_input_v2.py` 별도계약으로 완료producer의 실제 thread를 재생한다. 부모/worker SHA·exact100 UID/volume/mask/center/normalized해시·원float64참조한도를 모두 요구한다. 로컬100 연결과 T4 원DICOM100 수치/종료PASS를 확인했다. 최종test 경로/hidden 전체·정확도보장은 별도다. [T4 결과](../handoffs/2026-10-04-0330-b-rn3f-t4-result.md).

- [V2 계약·정확 명령](../handoffs/evidence/2026-10/2026-10-04-exp084-result-and-serving/thread-input-v2/execution-plan.json): 완료된 모델추론 재실행 없이 CPU입력재구성만1회. math helper/허용치/producer는불변.
- [T4 한session 계약](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-t4-trial-v1/execution-plan.json): producer→V2input→`b_rn3f_acceptance.py evaluate`→외부resource/cleanup을확인. exit0·kernelCOMPLETE만으로PASS하지않는다. 결과회수는정확version의result.json→hash일치evidence.zip만사용한다.
