# 2026-10-01 구현 인계

사용자 요청 순서: 경로 의존성 해소 → 폴더 개선 → 문서화 → 대회 대시보드 구현.

경로 모듈 knee_workspace를 만들고 기존 실행 코드의 root/cwd/H: 의존을 변경했다. 23개 폴더를 training/storage/artifacts/research/kaggle로 이동했다. config/workspace.toml의 layout_version은 2다. 원본 코드 backup과 이동 journal은 .local/migrations/layout-v2에 있다.

15개 경로·라벨·제출 테스트, compileall, PowerShell parsing을 통과했다. 큰 데이터 전체를 해시하거나 GPU 학습을 재실행한 것은 아니다. 자세한 새 구조와 복구 절차는 [작업 공간 runbook](../runbooks/workspace-layout.md)을 읽는다.

## 현재 제품

개인용 KneeOps를 구현했다. 실행은 KneeOps.cmd, 주소는 http://127.0.0.1:8765. Setup-KneeOps.ps1을 실제 실행해 독립 .venv와 웹 빌드를 검증했다. Docker는 선택 사항이며 독립 빈 DB의 health/HTML 200과 native와 다른 instance_id를 확인했다. 테스트 컨테이너 kneeops-smoke-20261001은 종료·제거했다.

사용자가 중간에 범위를 수정했다. 아이디어 보드·팀 공유는 제외하고 제출 이력을 유지한다. 왼쪽 메뉴를 종류별로 분리하고 목록 기본/카드 옵션, 처음 추가한 날짜, 한국 시간 당일 NEW를 표시한다. 일반 본문을 14–16px로 키웠고 데스크톱·모바일을 확인했다.

유사 과거 대회 수집과 전략 원문 인용을 추가했다. 현재 대회 6개 connector의 live sync가 성공했다. 요추 MRI 대회는 공식 페이지 8, notebook 목록 20, 토론 60, 솔루션 본문 8을 성공적으로 조회했다. 로그 손실 대회는 낮은 점수순, AUC는 높은 점수순, 방향 미확인 시 추천순을 쓴다. 토론 제목 순위는 참가자 주장, LB 반환 순서는 최종 Private 입상 검증과 구분한다.

노트북 source는 owner/slug, revision은 version+content hash. CLI pull metadata에는 version이 없어 read-only SDK helper에서 최신 번호를 받는다. 실제 vibhojoshi/rsna-2024-lumbar-spine-prediction 최신 v3 조회를 확인했다. v1 요청은 403으로 거절되었다. 과거 전체 버전 접근을 보장하지 않는다. 이미 수집한 원문과 출력 제외 셀 본문, 출처를 보존한다.

수집 간격은 0.25~24시간, 과거 대회 최대 5개 선택 가능. 앱 실행 중에만 동작하며 기본 꺼짐이다. 한 프로세스에서 단일 queue. 별도 Codex 예약 작업이나 OS 서비스는 생성하지 않았다.

## 검증

초기 15개 경로/기존 라벨·제출 테스트를 통과했다. 대시보드 테스트는 버전 보존·added_at 불변·리터럴 검색·대회 분리·metric contract·쓰기 보안·설정·DB 분리와 backup·실패 복구·수집 주기·전략 원문·지표 방향을 확인한다. 독립 .venv에서 경로+대시보드 15개 통과. FastAPI TestClient가 httpx 대신 향후 httpx2 사용을 권장하는 deprecation warning 1개가 있으나 실패는 아니다.

Playwright 3개 통과: 메뉴/목록/NEW/원문창, 설정 가독성/과거 대회, 모바일 메뉴와 페이지 overflow. 화면은 .local/ui-library-desktop.png, ui-settings-desktop.png, ui-library-mobile.png. 초기 실패는 label locator를 실제 select 접근성 label로 바꿔 해결했다. 화면에 따라 요청 응답 순서가 뒤집히는 문제도 request sequence로 막았다.

TypeScript/Vite build, Python compileall, 모든 PowerShell runner/setup의 parser 오류 0, 현재 문서 내부 링크 검사 통과. npm audit는 설치 당시 0 vulnerabilities. GPU 학습/실제 제출/full test inference는 실행하지 않았다.

최종 합산 pytest 25개 통과. 최종 UI build와 Playwright 3개도 통과했다. 실제 DB integrity_check=ok, online backup 파일 생성 확인. 최종 관측 자료 337개·실험 46개·제출 7개이며 수집 connector는 모두 ok였다. 화면별 자료 수에는 과거 대회와 로컬 문서도 포함한다.

## 다음 작업 진입점과 한계

현재 코드는 monitoring/kneeops, UI는 monitoring/web/src. docs/runbooks/dashboard.md의 CLI로 먼저 검색한다. README/AGENTS/CLAUDE와 실제 아키텍처를 우선하며 초기 상세 설계의 팀 bundle/아이디어 등은 현재 기능이 아니다.

DB는 schema3. added_at 컬럼 도입 전 자료는 migration 당일 시각으로 채웠다. legacy 빈 hypotheses/imports 테이블은 파괴적으로 삭제하지 않았지만 API/UI에서 쓰지 않는다. 원문 원본 파일은 DB 밖 .local/kneeops/notebooks에도 저장되어 전체 백업 시 state 폴더를 함께 보관한다.

네트워크 일시 SSL 실패는 재조회로 해결됐고 CLI는 transient 오류에 최대 3회 재시도한다. 전체 자료/댓글/과거 버전 완전성, 실제 입상자 계정 자동 매칭, LLM 요약/RAG/MCP는 구현하지 않았다. 더 강한 실험 비교에는 기존 산출물에 label/split/mask/study/transform hash를 추가 기록해야 한다. 정보 없는 점수를 억지로 비교 가능 처리하지 않는다.
