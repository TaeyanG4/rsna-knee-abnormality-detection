# 새 CLI 검토와 수정 근거

작업 DOCS_AGENT_CLI_V1, 2026-10-04. [전체 적용 기록](../../../2026-10-04-2227-document-roles-agent-cli.md).

Claude Code 새 읽기 전용 세션 `20261004T134452-d5d69457`은 정적 검토만 수행했다. native ID `34a465cb-490c-4440-9686-135e3321d7aa`, 22:44:53–22:46:44 KST, exit 0·정상 result·권한 거부 없음. 검토 원문은 [claude-review.txt](claude-review.txt), 당시 실행 상태는 [claude-job.json](claude-job.json)에 보존한다. 후속 수정은 조정자가 수행했다.

| 검토 사항 | 처리와 검증 범위 |
|---|---|
| Windows의 열린 상태 파일에 os.replace 실패 가능 | 실제 Windows 열린 파일에서 PermissionError를 재현했다. 제한된 재시도를 추가하고 reader 종료 후 갱신 성공을 검사했다. 단순 정적 추측으로만 남기지 않았다. |
| worker 최초/마지막 상태 기록 실패 시 잠금 잔존 | 최초 쓰기를 try 안으로 이동, 마지막 쓰기와 잠금 해제를 분리했다. 살아 있는 child는 잠금을 유지한다. 시작/최종 쓰기 실패 주입 테스트 통과. |
| worker 시작 후 launch.json 저장 실패 시 잠금 조기 해제 | 실행 이후에는 잠금을 유지하고 실행ID/PID를 포함해 오류를 반환한다. 실패 주입으로 확인. |
| null 응답 필드에서 파싱 예외·메타데이터 소실 | null item/text/result를 실패로 판정하면서 session ID·permission denial을 보존한다. 성공 terminal이 있어도 최종 응답이 비면 실패. 테스트 통과. |
| Claude의 16턴 상한 | 사용법에 명시. 상한/기한 실패는 자동 재시도하지 않으며 마지막 부분 응답을 실패 상태와 함께 보존하도록 수정·검사. |
| 자식 프로세스 정리 | POSIX의 종료 경쟁과 process-group 강제 종료를 보완. Windows timeout/cancel의 직접 child 종료는 테스트했다. Windows Job Object·이미 분리된 후손·worker 강제 종료는 보장 범위에서 제외하고 문서화. POSIX 실환경은 미검증. |
| PID 재사용·job 생성 전 충돌 | 자동 복구의 한계를 명시했다. 살아 있는 PID의 잠금 회수 거부와 죽은 작업의 회수는 테스트. 실제 소유권 확인 없이 강제 회수하지 않는다. |
| 읽기 범위·자원 실행·중앙 파일 검사 범위 | read-only가 비밀 파일 접근 차단이나 GPU 격리를 뜻하지 않음을 명시. 루트 5개 파일만 hash 검사하며 부모 수정도 알림 대상이다. 모든 Windows 샌드박스 경계를 검증한 것은 아니다. |
| Claude 안에서 다시 Claude 실행 | 기존 중첩 실행 제한을 우회하지 않는다. doctor 관측과 실패 보고를 사용한다. 이번 실제 호출의 상위는 Codex였다. |
| 문서 표·중복·진입점 | discoveries 표의 빈 줄 제거, 도구 문서 서문 위치 수정, plan에서 과거 ETA/진행 제거, HANDOFF는 결과·진행과 교훈 링크 중심으로 정리. 이전 제안 문서는 역사 자료임을 표시. |

Codex 새 읽기 전용 세션 `20261004T135523-986da3fe`, native ID `01a10732-ac86-7591-ac8c-53370e91533f`는 22:55:23–22:55:50 KST에 네 루트 문서를 읽고 `SESSION_READ_OK`를 반환했다. exit 0·turn.completed·중앙 파일 변화 없음. [응답](codex-smoke.txt), [실행 상태](codex-job.json). 이는 같은 상위 대화에서 새 세션 호출·파일 읽기·결과 회수가 동작한 근거이며 구현 쓰기나 연구 성능 검증은 아니다.

최종 검사: `.venv/Scripts/python.exe -X utf8 -m pytest tests/test_agent_session.py -q --basetemp .local/test-agent-session-v3 --tb=short` → **22 passed in 3.61s**. 최초 실행은 기본 Windows 임시 폴더 권한 문제로 fixture 생성에 실패했고 workspace 내 새 basetemp에서 해결했다. v1 13개, v2 22개 통과 이력도 덮어쓰지 않았다. 무관한 학습/추론/제출/예약은 실행하지 않았다.
