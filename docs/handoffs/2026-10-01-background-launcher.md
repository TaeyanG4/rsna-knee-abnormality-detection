# 2026-10-01 콘솔과 분리한 KneeOps 실행

## 요청과 결과

사용자가 PC 종료 후 실행 방법을 물었고 이어서 CMD 창을 닫아도 계속 작동하도록 요청했다. 확인 당시 localhost 앱과 관련 Python 프로세스는 없었다. 기존 실행 파일을 백그라운드 실행으로 변경하고 실제 앱을 켰다. Docker/Windows 서비스/부팅 자동 실행/절전 방지는 추가하지 않았다.

- 시작: 루트 `KneeOps.cmd`. 준비를 확인한 뒤 시작 콘솔이 종료되어도 앱은 유지된다. 실행 중에는 같은 앱을 연다.
- 종료: 루트 `KneeOps-Stop.cmd`. 진행 중 전송은 부분 파일을 보존하고 종료한다. 수집 작업이 실행 중이면 정상 종료에 시간이 걸릴 수 있다.
- 로그: `.local/kneeops/logs/background.log`.
- 상태: `.venv/Scripts/python -X utf8 -m monitoring.kneeops.desktop status`.
- PC 종료·절전·로그아웃 중에는 실행되지 않는다. 재부팅 후 직접 앱 시작과 다운로드 재시도가 필요하다.
- 기존 `python -m monitoring.kneeops open/serve`는 콘솔 실행을 유지한다. 실행 중인 콘솔 서버를 자동 중지하거나 이동하지 않는다.

## 구현

`monitoring/kneeops/desktop.py`는 start/status/stop/run 명령을 제공한다. Windows는 CREATE_NO_WINDOW와 분리된 stdin/stdout/stderr, 다른 OS는 start_new_session으로 실행한다. 시작 확인/상태 조회는 DB를 초기화하지 않는다. 실제 서버는 state 디렉터리 OS 잠금과 localhost 소켓을 **DB/worker 초기화 전에** 얻어 중복 실행으로 downloads/jobs를 interrupted로 바꾸지 않도록 한다.

실행별 UUID와 `/api/desktop`의 root/state/run_id를 대조해 해당 실행의 로컬 stop 파일만 만든다. 파일 경로로 임의 PID를 종료하지 않는다. watcher가 uvicorn.should_exit를 설정하여 기존 lifespan의 다운로드 취소/임시 파일 보존·수집 종료를 사용한다. 요청자가 제시한 임의 파일 경로로 shutdown하지 않는다. 종료 대기는 HTTP가 사라진 시점이 아니라 state 잠금이 해제된 시점을 기준으로 한다. 로그는 append하며 access log를 끈다.

`KneeOps.cmd`, 새 `KneeOps-Stop.cmd`, README, STATE, dashboard runbook, 구현 아키텍처를 갱신했다. UI/캐시/원본/기존 실험 기록은 변경하지 않았다.

## 검증

- 임시 한글/공백 경로의 독립 workspace에서 실제 서버 프로세스를 띄웠다. 시작 프로세스가 끝난 뒤 HTTP 생존, 재시작 요청 시 같은 PID/실행 ID, 기존 queued job 유지, 정상 종료와 stop 파일 정리를 확인했다.
- 점유된 포트로 실행하면 DB를 만들지 않는 것, 다른 workspace 및 콘솔 서버를 stop이 건드리지 않는 것을 검사했다.
- `.venv/Scripts/python -m pytest monitoring/tests tests -q --basetemp .local/tests/desktop-regression`: **54 passed**, 기존 FastAPI httpx deprecation 경고 1개.
- 실제 start 프로세스 종료 후 localhost 응답을 확인했다. 실제 앱은 백그라운드 PID `12956`, run ID `e2bae0a6e52b41f7b65eb2a1244276d8`로 관측했다. 향후 PID 값 대신 status 명령으로 현재 실행을 확인한다.

## 전체 다운로드 이어받기

기존 전체 ZIP 작업 `f80a6b01-15ac-43d7-837c-718505001392`은 PC 종료 후 `interrupted`, 수신 기록 172,567,298,048 bytes였다. 앞선 명시적 전체 다운로드 승인에 따라 인증을 확인하고 **같은 작업 ID의 start API**로 재시도를 요청했다. 새로운 계획/중복 전송을 만들지 않았다. 완료 여부는 API에서 재확인한다. 서버 Range/객체 검증 조건에 따라 이어받기 여부가 결정된다.

경로/원본 구조/완료 후 안전한 압축 해제 및 대시보드 연결 계획은 [이전 다운로드 인계](2026-10-01-download-in-progress.md)를 이어받는다. 사용자가 다운로드 완료를 알릴 때까지 현재 앱/전송을 유지한다. 실제 DB를 사용한 별도 create_app/DataManager 초기화나 서버 재시작으로 전송을 방해하지 않는다.
