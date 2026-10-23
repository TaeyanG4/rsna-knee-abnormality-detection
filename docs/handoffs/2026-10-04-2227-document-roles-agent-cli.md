# 문서 역할 분리와 로컬 에이전트 CLI

2026-10-04 22:27 시작, 2026-10-04 23:00 KST 완료. 작업 DOCS_AGENT_CLI_V1. 조정자 `01a106e1-8308-7510-ad2b-a13c2574dd8f`.

- 요청 반영: CLAUDE는 AGENTS 안내 한 줄. HANDOFF=실제 진행, plan=다음 가설, discoveries=교훈/closed ideas, AGENTS=공통 지시. 역할은 조정자·실행자·검토자로 일반화했다.
- [변경 전 원문 16개/hash](archive/snapshots/2026-10-04-agent-docs-v1/manifest.json). M01–M11 승계·이전 경로 호환 링크 유지. RN3 보류와 기존 승인 유지, 연구 재개 없음.
- [agent_session.py](../../tools/agent_session.py)로 같은 상위 대화에서 새 CLI 세션 호출·결과 회수. 기본 순차/읽기 전용, 명시적 구현 쓰기, 기한·취소·자체 잠금·종료 event 검증. 모델/effort는 기존 CLI 설정 사용. [사용법·권한 한계](../runbooks/tools.md#로컬-에이전트-세션).
- 실제 Claude 검토 `20261004T134452-d5d69457`: 22:46 정상 종료. Codex 읽기 확인 `20261004T135523-986da3fe`: 22:55 정상 종료·SESSION_READ_OK. [원문·수정 판단](evidence/2026-10/2026-10-04-document-roles-agent-cli/review-disposition.md). job/응답 복사본과 원 로그 hash 보존.
- Windows 파일 교체 충돌을 재현·수정하고 수명주기/잠금/파싱 검사를 보완. 최종 테스트 22개 PASS/3.61초. 최초 기본 temp 권한 실패는 workspace basetemp로 해결. 읽기 전용 호출 검증이며 실제 구현 쓰기·전체 OS 격리·연구 성능을 증명하지 않는다.
- 문서 링크/길이·snapshot hash·M행 보존·앱 최신 revision/added_at 정합은 [검증 기록](evidence/2026-10/2026-10-04-document-roles-agent-cli/final-validation.json) 참조. 도구 사용 중 핵심 문서 변화는 자동 위반 판정이 아닌 확인 대상이다.
- GPU/추론/제출/새 예약 0. 두 자식 완료·자체 잠금 해제. 이 작업의 남은 일 없음. 연구 후속은 [plan](../../plan.md)과 [HANDOFF](../../HANDOFF.md).
