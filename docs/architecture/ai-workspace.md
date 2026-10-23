> 2026-10-02 작업 기록·진입·압축 규칙은 [인계 관리 runbook](../runbooks/handoff-maintenance.md)과 [AGENTS](../../AGENTS.md)를 따른다. 아래는 초기 설계의 배경이다.

> 초기 설계 기록입니다. 2026-10-01 이후 개인용으로 범위를 조정했습니다. 현재 구현은 [실제 아키텍처](../architecture/implemented-dashboard.md), [추가 결정](../decisions/0002-personal-research.md), [STATE](../STATE.md)를 우선합니다. 팀 공유·아이디어 보드 등 아래 초기 제안은 현재 제품 기능이 아닙니다.

# AI 작업 환경과 폴더 개선

상태: 2026-10-01 설계. 이 문서는 구현 계약이며 RAG 서버, MCP 서버, 경로 resolver가 이미 실행된다는 뜻은 아니다. 기본 운영은 팀원별 독립 설치다. **사용자 추가 지시에 따라 RAG와 MCP는 MVP에서 구축하지 않는다. Codex·Claude가 파일과 CLI를 쓰기 편한 환경을 먼저 완성한다.**

## 1. 세 가지 기억을 분리한다

| 종류 | 저장 위치 | 읽는 방법 |
|---|---|---|
| 작업 규칙과 현재 상태 | AGENTS.md, docs/STATE.md, 최신 handoff, ADR | 새 AI 세션에서 직접 읽기 |
| 정확한 구조화 사실 | SQLite의 run·metric·submission·revision | JSON CLI/API query |
| 조사 지식과 설명 | 공식 페이지, 토론, notebook cells, 연구 문서 | rg/FTS5 검색과 원문 링크 |

RAG로 점수·마감·현재 run 상태를 추측하지 않는다. 구조화된 query로 가져오고 원문 근거를 함께 연결한다. AGENTS.md에 모든 연구 결과를 붙여 넣지 않고, 약 2~5KB의 안정적인 규칙과 문서 진입점만 유지한다. Gold58 사용 이력, 기존 metric 범위 같은 장기 주의 사항은 짧게 남긴다.

문서의 source of truth도 정한다. 현재 작업 상태는 STATE, 설계 이유는 ADR, 작업 인계는 handoff, 실험 수치는 run manifest/DB, 원문은 immutable snapshot이다. 서로 복제해 긴 요약을 여러 곳에서 수동 수정하는 구조를 피한다.

## 2. 권장 폴더 구조

대회 데이터와 프로그램 코드를 논리적으로 분리한다. 현재 대회 root를 workspace로 유지하면서 새 프로그램은 `monitoring/`에 만들고, 향후 이 디렉터리만 독립 저장소로 배포할 수 있게 한다. 프로그램에 대회 이름과 H: 절대 경로를 하드코딩하지 않는다.

```text
rsna-knee-abnormality-detection/
  AGENTS.md
  CLAUDE.md                      공통 규칙으로 안내하는 Claude 진입점
  README.md
  config/
    workspace.example.toml       공유 가능한 경로·설정 예제
    local.toml                   PC별 경로, Git/bundle에서 제외
  docs/
    STATE.md                     현재 상태와 다음 작업
    architecture/                시스템·AI 환경 설계
    decisions/                   변경 이유와 채택한 대안
    plans/                       구현 backlog, 이동 manifest
    handoffs/                    날짜별 인계, template
    runbooks/                    설치·복구·수집·실험 절차
  monitoring/                    앞으로 구현할 배포 가능한 프로그램
    apps/web/
    kneeops/domain/
    kneeops/services/
    kneeops/connectors/
    kneeops/search/
    kneeops/adapters/local/
    contracts/
    tests/
    deploy/
  .local/                        네이티브 실행 상태, Git/bundle에서 제외
    app.sqlite
    snapshots/
    indexes/
    jobs/
    backups/
  configs/experiments/            앞으로 만드는 실험 설정
  runs/<run_uuid>/                앞으로 만드는 표준 실험 산출물
    manifest.json
    config.json
    metrics.json
    predictions.parquet
    logs/
  research/                      앞으로 정리할 source/claim export
  <기존 data, cache, experiments, src 등은 우선 현 위치 유지>
```

`monitoring/`, `.local/`, `runs/`, `research/`의 위 세부 구조는 목표 구조다. 이번 작업에서는 문서와 예제만 생성했다. 아직 구현되지 않은 실행 명령을 README에 작동하는 명령으로 적지 않는다.

### 기존 폴더의 역할과 목표 경로

| 현재 영역 | 장기 역할 | 개선 방법 |
|---|---|---|
| data/, test/ | 공식 metadata 및 원본 | 데이터 버전과 train/test 구분 manifest부터 추가 |
| dicom_cache/, pixel_cache/, cache/, processed/ | 재생성 가능한 파생 데이터 | cache key에 preprocessing/data hash, 완료 receipt |
| labels/, gold/ | 라벨 버전과 평가용 자료 | label source, gold overlap/exposure, target mask 계약 |
| external/, weights/ | 외부 코드·모델 | source URL, version/hash, 입력 라이선스 metadata |
| experiments/, oof/, predictions/, logs/ | 학습 실행 산출물 | importer로 통합 조회, 새 run부터 runs/<uuid> 계약 |
| kaggle_notebooks/ | 공개 source와 자체 notebook 혼재 | catalogue에서 owner/type 분리, source immutable 보존 |
| kaggle_assets/, kaggle_submission/, submissions/ | 배포 inputs와 제출 package | package version, input/output SHA, submission ID 연결 |
| src/, scripts/ | 현재 학습·분석 실행 코드 | 우선 기존 위치 유지, 새 코드는 package/service로 작성 |
| reports/, metadata/, snapshots/ | 해석 문서·사실·원문 혼재 | 타입과 schema를 붙여 docs/DB/raw snapshot으로 점진적 연결 |

파일을 보기 좋게 만들기 위해 바로 이동하면 안 되는 구체적 이유가 있다. `src/acquire.py` 등은 `Path(__file__).resolve().parents[1]`로 root를 찾고, `src/exp021_synovitis_missingness.py`와 여러 PowerShell runner는 H: 절대 경로를 포함한다. 모델 manifest와 notebook input에도 기존 경로가 들어 있다.

이동 순서는 inventory → reference scan → 논리 경로 resolver → importer → 작은 코드 묶음 수정 → 검증 → 실제 이동이다. 원본·목표 path, size/hash, 참조 파일, 검증 결과, rollback 위치를 [folder-migration.csv](../plans/folder-migration.csv)에 기록한다. 대용량은 매번 전체 hash를 재계산하지 않고 기존 receipt와 크기/변경시각으로 후보를 줄인 뒤 필요한 파일을 검증한다. 같은 볼륨에서의 최종 move도 대상 절대 경로와 workspace 경계를 먼저 검사한다.

호환성 유지를 위해 가능하면 코드의 path resolver를 고친다. OS마다 동작이 다른 symlink/junction을 기본 마이그레이션 방법으로 강제하지 않는다. 기존 자동 실행·Kaggle package 재현이 확인되기 전 대용량 파일 이동과 삭제는 진행하지 않는다.

## 3. 문서와 인계 규약

새 작업의 읽기 순서는 AGENTS(Claude 진입점은 CLAUDE) → README → STATE → 최신 handoff → 해당 ADR/설계/원문이다. CLAUDE.md는 공통 규칙을 중복 복사하지 않고 AGENTS.md를 읽도록 안내한다. 각 단계는 다음 문서의 링크를 제공한다. STATE는 현재 사실·진행 중·다음 행동을 짧게 유지하며 과거 모든 로그를 쌓지 않는다.

handoff는 대화 요약보다 재개 가능한 작업 기록이어야 한다.

- 목표와 사용자가 확정한 선택: 독립 설치, 브라우저 기반, 개인 커스텀, 대회 성적 우선.
- 완료한 파일과 근거, 실제로 수행한 검증.
- 시작하지 않은 것, 실행 중 process/job, 종료/재개 방법.
- 다음 1~3개 작업과 해당 파일/계약, 예상 비용·제약.
- 불확실한 점과 확인에 필요한 자료. 사실·가설·보류를 분리.
- 변경한 파일 목록과 code/config/schema version. 없으면 없다고 기록.

run 종료 시 metrics와 decision을 등록하고 그 결과를 STATE/handoff에서 링크한다. AI의 자유로운 서술만 남기지 않는다. 실패한 실험도 이유와 재시도 조건을 남겨 다른 AI나 팀원이 같은 실패를 반복하지 않게 한다.

AGENTS는 사용자 지시의 범위를 확장하는 장치가 아니다. 데이터 다운로드·GPU 지출·Kaggle 게시·제출은 현재 세션의 요청과 승인 범위를 따른다. 이미 받은 권한을 반복해서 묻는 규칙은 넣지 않는다. 하위 작업 규칙이 필요해질 때만 `monitoring/AGENTS.md` 같은 짧은 파일을 추가한다. 현재 Git root가 없으므로 현재 대회 root의 AGENTS가 직접 진입점이다.

## 4. 처음 구현할 AI 인터페이스

**문서 + 구조화 파일 + CLI**를 기본으로 한다. Codex와 Claude가 이미 가진 파일 읽기·검색·shell 기능으로 충분히 사용할 수 있게 만든다. AI가 DB 파일 내부나 브라우저 화면을 직접 파싱할 필요는 없다.

CLI는 `briefing`, `sources search`, `sources show`, `experiments list`, `experiments compare`, `experiments import`, `submissions list`, `handoff show`, `bundle export`, `bundle inspect`, `bundle import`의 작은 명령 집합으로 시작한다. 현재는 인터페이스 제안이며 명령은 아직 구현되지 않았다. 각 명령은 `--json`을 지원하고 stdout은 결과, stderr는 진단 로그로 분리한다. 실패는 nonzero exit code와 typed error를 반환한다.

공통 결과 envelope는 schema_version, generated_at, data_as_of, records, source_refs, coverage, warnings, next_cursor다. pagination과 output limit을 지원한다. 쓰기에는 idempotency key와 expected_revision을 넣고, dry-run/import-preview는 실제 영향 목록을 출력한다. CLI와 UI가 같은 service를 사용하므로 검증 규칙을 두 번 구현하지 않는다.

문서는 headings와 안정적인 경로를 갖추고, notebook은 원본 외에 cell ID가 포함된 읽기용 text/code를 파생 저장한다. discussion은 message ID와 parent 관계를 보존한다. metadata JSONL이나 SQLite query로 필터링하고 rg/FTS5로 본문을 찾는다. source revision과 SHA를 고정해 AI가 인용한 근거를 다시 열 수 있게 한다.

숫자·현재 상태는 CLI로, 설명·가설·과거 결정은 문서로 읽는다. AI가 작성한 결과도 schema를 검증한 파일 또는 CLI 입력으로 기록한다. 긴 대화 transcript를 사실 저장소로 사용하지 않는다.

## 5. RAG와 MCP의 도입 조건

| 기능 | 지금의 결정 | 다시 검토할 때 |
|---|---|---|
| RAG/embedding DB | 구축 보류 | 문서가 늘어 rg/FTS5로 알려진 정답 근거를 반복해서 놓치고, 한국어 질문과 영어 원문의 의미 검색이 필요한 경우 |
| MCP server | 구축 보류 | CLI 사용이 불가능한 AI client가 필요하거나, 반복적인 도구 호출·권한·schema 연동을 MCP로 통일하는 실익이 확인된 경우 |
| autonomous multi-agent platform | 구축 보류 | 역할·검증·비용이 명확한 병렬 연구가 실제 병목인 경우 |
| 복잡한 knowledge graph | 별도 DB 없이 source_edges 테이블 | 관계 query 규모가 단순 테이블로 감당되지 않을 때 |

도입 여부는 30~50개의 대표 질문과 작업으로 측정한다. 예를 들어 'EXP056 +0.01985의 target 범위', '추론에 Report가 있는가', '0.941 제출의 ID', 'Raptor reverse와 원본의 weight 동일성', '외부 데이터에 대한 실제 HOST 답변'을 찾을 수 있어야 한다. 검색 시간·근거 재현·누락·citation 정확도·유지보수 비용을 비교한다. 문서 이름과 링크 개선으로 해결되면 RAG를 만들지 않는다.

RAG가 필요해질 때에는 immutable source revision, 문서/cell/message별 chunk, lexical+vector hybrid, origin·authority·freshness metadata, source citation을 갖춘다. 숫자는 계속 structured tool을 사용한다. 원문 report/DICOM·credential은 기본 색인에서 제외하고, 외부 자료의 명령을 프로젝트 지시로 승격하지 않는다. vector index는 재생성 가능한 cache다.

MCP가 필요해질 때에는 CLI의 기존 service를 얇게 감싸는 read tools부터 제공한다. stdio 또는 loopback Streamable HTTP를 선택하고, arbitrary shell/SQL/파일 접근이나 자동 제출을 노출하지 않는다. app DB에 별도 쓰기 경로를 만들지 않는다. 실제 서버가 준비되기 전 Codex/Claude 설정에 작동하지 않는 MCP entry를 추가하지 않는다.

공식 Codex 문서는 AGENTS가 실행 시작 시 탐색되고 MCP가 stdio/Streamable HTTP를 지원한다고 설명한다. 이 지원 여부는 지금 MCP를 구현해야 한다는 이유가 아니다. 현재 어떤 MCP 서버도 설치·등록하지 않았다.

## 6. 실험 manifest 최소 계약

새 run은 run_uuid, origin_instance_id, display_id, hypothesis_id, started/finished_at, status, code snapshot, config hash, data/label/split/target/mask version, training exposure, seed, hardware, outputs와 SHA, metrics, cost/runtime, parent runs를 기록한다.

모든 prediction artifact에는 study UID·target 순서·fold·label contract가 필요하다. 실험 ID가 같아도 다른 팀원이 만든 run은 같은 run으로 합치지 않는다. public score는 submission ID와 연결된 별도 observation이다. 한 번의 trial에서 best epoch를 고른 결과와 고정 checkpoint 결과를 구분한다.

공유 bundle은 manifest(schema_version, bundle_uuid, origin, created_at, app_version, included entities, artifact hashes), entities JSONL, 선택한 artifacts, 사람이 읽는 summary로 구성한다. 개인 local path는 논리 URI로 바꾸고 없는 weight는 missing_external_artifact 상태로 기록한다. imported reference만으로 재현 완료가 되지 않는다.

## 7. AI 작업 환경의 완료 기준

- 새 AI 세션이 5분 이내에 목표, 결정된 운영 방식, 현 상태와 다음 작업을 찾는다.
- AI가 문서 원문과 revision을 확인하고 수치 질문은 JSON CLI/API로 처리한다.
- CLI/UI 비교 결과가 같은 계약에서 일치한다.
- 수집 자료의 실행 지시가 명령 실행으로 이어지지 않는다.
- origin이 다른 두 PC의 bundle을 교환하고 중복·충돌·재현 불가 artifact를 정확히 표시한다.
- resolver 도입 후 기존 경로의 대표 train/inference/package 흐름을 검증한 뒤에만 파일을 옮긴다.
- handoff만 받은 AI가 기존 조사·실험을 재수행하지 않고 미완료 작업을 이어간다.

공식 참고: [AGENTS.md 안내](https://developers.openai.com/codex/guides/agents-md), [Codex MCP](https://developers.openai.com/codex/mcp). 두 페이지를 2026-10-01 실제 조회했다.
