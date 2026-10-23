> 초기 설계 기록입니다. 2026-10-01 이후 개인용으로 범위를 조정했습니다. 현재 구현은 [실제 아키텍처](../architecture/implemented-dashboard.md), [추가 결정](../decisions/0002-personal-research.md), [STATE](../STATE.md)를 우선합니다. 팀 공유·아이디어 보드 등 아래 초기 제안은 현재 제품 기능이 아닙니다.

# 구현 backlog와 합격 기준

2026-10-01 기준. 지금 완료한 것은 조사·설계·문서 진입점이다. 아래 코드는 아직 구현하지 않았다. 새 프로그램은 monitoring/에서 시작하며 대회 데이터 경로는 사용자별 config로 주입한다.

## P0 기존 작업을 잃지 않는 기반

1. 현재 src/scripts의 경로 의존성과 주요 입출력 contract를 inventory한다. folder-migration.csv를 보강하되 원본을 옮기지 않는다.
2. workspace config loader와 논리 URI resolver를 만든다. cwd에 따른 경로 변화와 Windows/container 경로 매핑을 검증한다.
3. 원문 revision, source, claim, run, metric, submission, job, bundle의 최소 schema를 정의한다. origin_instance_id와 UUID를 사용한다.
4. 기존 summary/manifest importer의 preview를 구현한다. EXP056의 4-target weak OOF, EXP059의 leaky pseudo-OOF, live submission을 서로 다른 type으로 가져온다.
5. JSON CLI에 briefing, experiments list/compare, sources show를 먼저 구현한다.

완료 조건: 기존 자료를 두 번 읽어도 중복이 없고, 과거 '세션 제출 0건'이 live 7건을 덮어쓰지 않는다. metric 범위가 다른 비교는 이유를 표시한다. 기존 training script를 실행하지 않고 importer만으로 결과를 읽는다.

## P1 대회 운영에 바로 쓰는 웹 MVP

1. FastAPI + React + SQLite/FTS5 기반 localhost 앱을 만든다.
2. 오늘의 briefing, 우리 제출/LB, 공개 연구 목록·원문, 수집 상태의 4개 화면부터 만든다.
3. Kaggle CLI/SDK capability probe, 페이지 처리, 재시도와 last_success를 구현한다. 인증 실패는 빈 데이터가 아니다.
4. version/hash가 바뀐 노트북·토론만 다시 구조화한다. 기본 요약은 메타데이터 diff와 사용자/AI가 등록한 근거 메모로도 작동하게 한다. 유료 LLM 호출은 필수 의존성이 아니다.
5. 개인별 저장 필터·관심 target·표 컬럼·언어·시간대를 지원한다.
6. Docker image/Compose, native 실행 문서, persistent volume과 backup을 구현한다.

완료 조건: PC 한 대에서 수집→원문→우리 실험→제출을 연결하고, 앱을 껐다 켜도 상태가 남는다. 네트워크를 끊어도 cache를 조회하고 stale 표시가 보인다. 실제 검증한 설치 명령만 README에 추가한다.

## P2 독립 설치끼리 협업

1. bundles manifest/schema와 export/inspect/import-preview/import를 구현한다.
2. 중복, 동일 origin의 revision 충돌, 누락 artifact, schema 불일치, path traversal을 처리한다.
3. 두 개의 격리된 앱 인스턴스에서 동일 bundle을 왕복/반복 import하고 개인 설정이 섞이지 않는지 검증한다. 실제 다른 PC 검증 전에는 그 한계를 표시한다.
4. 자체 실험 결과 exporter, 가설 보드, 결정 사유, per-target 비교를 연결한다.

완료 조건: 공용 서버나 공유 DB 없이 양쪽이 같은 선택 자료를 읽을 수 있고 서로의 credential·preferences·원본 data를 전달하지 않는다. 팀원 출처의 결과는 imported observation이며 로컬 재현 완료로 표시되지 않는다.

## P3 필요한 구조 개선과 제품 안정화

1. 새로운 run은 표준 runs/<uuid>로 출력하도록 wrapper부터 도입한다.
2. 기존 코드 묶음의 path resolver를 도입하고 대표 입출력을 검증한 뒤 이동한다. 대용량 데이터는 옮기는 이득이 명확할 때만 작업한다.
3. 수집 parsing fixture, metric scope, UID/split 검증, 두 UI 탭과 CLI 동시 수정, backup 복구를 확인한다.
4. Codex/Claude 재개 drill: 새 세션에서 STATE/handoff를 읽고 정해진 다음 작업을 수행하는 데 필요한 정보를 찾는다.

## 대회 실험과의 우선순위

첫 웹 MVP는 약 3일 개발 범위로 제한한다. 연구 플랫폼 전체를 완성할 때까지 실험을 기다리지 않는다. 기존 specialist fold4 집계, label confidence contract 확인, 최신 공개 모델의 차이 분석은 P0/P1과 함께 진행할 수 있다. 실제 GPU 학습/제출은 사용자 요청과 비용·실행 계약에 따라 별도 실행한다.

진척은 코드 줄 수나 수집 건수보다 '다음 실험을 고르는 시간', '근거를 다시 찾는 시간', '비교 불가능한 점수 혼동 건수', '재현 가능한 run 비율'로 평가한다.

## 보류

RAG/embedding DB, MCP server, multi-agent orchestration, 공용 서버, 자동 DB 동기화, 대형 외부 모델 자동 실행, 자동 Kaggle 제출, Kubernetes, Redis/Celery는 초기 구축 대상이 아니다. 각각 실제 병목이 확인되면 작은 ADR로 채택 이유를 기록한다.
