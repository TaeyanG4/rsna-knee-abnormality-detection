# 2026-10-02 11:04 KST · 자료 수집 범위 개편·정기 수집 토글

- 상태: 완료. 기록자: Claude.
- 요청·결정: 자료 라이브러리에서 정기 수집 on/off, 직전 수집 재확인, 진단한 수집 갭 전부 해결, Aneurysm 2025 등 과거 대회 확인. Kaggle 작업은 kaggle 스킬을 먼저 사용.
- 원인: CLI `topics show`는 원글 없이 댓글만 주고 키가 `content`(HTML)라 파싱 0건이었다. 기존 토론·요추 은 축하 댓글뿐이었다. 노트북은 점수순 20개만 발견했다(Kaggle 963). Aneurysm은 기본 목록에 없었다. 스케줄은 앱 시작 시각 기준이었다.
- 결과·파일: SDK GetTopic+ListComments로 원글+댓글 트리를 받는 `topic_fetch.py`, HTML→텍스트 `textutil.py`, [collectors.py](../../monitoring/kneeops/collectors.py)·`history.py` 개편. 노트북 전체 카탈로그와 관련 소스, 댓글 증가·실행 변경 감지, 전체 LB 내 순위·메달 참고선을 추가했다. `service.py` 스케줄은 마지막 수집 기준이며 `/api/schedule`이 생겼다. `SyncControl.tsx`는 라이브러리·설정에서 즉시 적용된다. 기본 과거 대회에 Aneurysm 2025와 Brain Tumor 2021을 추가했다. [수집 범위](../runbooks/dashboard.md#수집-범위).
- 일괄 보충(02:01~02:30Z, 오류 0): notebook 963개 중 관련 217개, 소스 234개(API 버전 218). 토론 143개 중 원글+댓글 137개(모집 글 6개 제외). LB 4,809팀 중 내 1,357위(0.941), 참고선 금 0.957·은 0.944·동 0.943. 과거 대회 솔루션 원문: Aneurysm 16/16, 요추 25/25, 복부 32/32, 경추 24/24, 뇌종양 23/23, 각 인기 토론 10개. 로그는 `.local/kneeops/logs/backfill-*`.
- Codex 병행: 10:49~10:50 KST에 이전 수집기로 상위 notebook 21개를 받았다. 새 수집기는 이미 받은 것을 건너뛴다. 노트북 숫자 점수는 공식 API에 없다(web 관측값은 미검증).
- 스킬 확인: leaderboard writeup 목록 15팀은 토론 솔루션과 일치하고 URL은 비어 있었다. 내부 `/api/i` 경로는 수집기에 넣지 않았다. 스킬의 credential 검사가 access_token 권한을 600으로 조정했다.
- 검증: pytest 84 통과, Playwright 30 통과(data.spec의 기존 fixture 누락 보완), tsc/Vite build, 원글 포함 SQL 확인, 화면 캡처 확인.
- 남은 일: 서버 종료 대기 정지가 3회 재현됐다(PID 41432, 11828; 로그가 "Shutting down"에서 멈춤). 명령줄 확인 후 해당 PID만 종료했다. 현재 PID는 32140(관측값). 원인 조사는 별도 작업 제안. 정기 수집은 꺼진 상태 그대로다.
