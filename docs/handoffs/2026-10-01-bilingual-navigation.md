# 2026-10-01 공식 문서 번역·메뉴 통합 인계

## 요청과 구현

개요·규칙을 영어 원문과 한국어 번역으로 읽고, 대시보드의 관련 메뉴를 통합해 달라는 요청을 구현했다. 개인 PC 독립 설치, localhost 실행, 기존 자료와 날짜·저장본 보존 결정을 유지한다. 이전 [데이터 기능 인계](2026-10-01-data-management.md)의 다운로드·캐시 기능도 유지한다.

| 왼쪽 메뉴 | 화면 안 분류 |
|---|---|
| 대회 안내 | 대회 현황, 개요·규칙 |
| 자료 라이브러리 | 전체 자료, 공개 노트북, 토론, 내 노트북, 로컬 문서 |
| 과거 대회 | 과거 대회 연구, 과거 솔루션 |
| 학습 데이터 | 다운로드·캐시 관리 |
| 실험·제출 | 실험 비교, 제출 이력 |
| 내 작업 환경 | 개인 설정·수집 간격 |

기존 hash 주소는 유지한다. 공식 안내만 전용 읽기 화면으로 바꾸고 검색 라이브러리에서 여는 공식 문서에도 같은 언어 전환을 제공한다. 원문/한국어/나란히 보기, 마지막 언어 기억, 16px 본문, Markdown 표·수식, 문서/저장본 선택, 원문 링크, 수집 시각·추가 날짜·NEW·번역 출처를 표시한다. 모바일은 문서 dropdown과 세로 비교 화면을 사용한다.

## 번역 범위와 업데이트 원칙

현재 대회의 Description, rules, Evaluation, Timeline, data-description, Code Requirements, Efficiency Prize Evaluation, Prizes, abstract, Acknowledgements 등 10개 공식 본문을 한국어로 번역했다. 기존 로컬 수집본을 사용했고 이번 작업에서 Kaggle를 다시 수집하거나 외부 번역 API를 호출하지 않았다. 인명·기관명은 원문 표기를 유지한다.

`monitoring/kneeops/translations/catalog.json`은 대회, 페이지, 영어 원문 SHA256, 수집일, 번역일, 번역자, 참고를 담는다. `en/`은 해당 영어 snapshot, `ko/`는 한국어 전문이다. 로딩 시 원문 checksum을 검증하고, 공식 자료의 대회·페이지·선택한 revision SHA가 모두 일치해야 번역을 반환한다. 원문이 갱신되면 `outdated`로 영어만 표시하며, 과거 번역은 해당 과거 저장본에서 계속 볼 수 있다. 번역이 없는 대회/페이지는 `unavailable`이다. DB migration이나 기존 원문/added_at 변경은 없다.

한국어판은 AI 참고 번역이다. 규칙 원문에 함께 나타나는 CC-BY-NC 4.0, OSI 승인·상업적 이용 관련 문구의 충돌, 불일치하는 조항 참조·URL은 번역에서 임의 수정하지 않았으며 별도 참고로 안내한다. 영어 원문이 규칙 해석의 기준이다.

새 원문 번역을 추가할 때에는 영어 snapshot을 보존하고 한국어 본문을 검토한 후 catalog 항목을 새 SHA로 추가한다. 기존 SHA의 번역 항목은 남겨 과거 저장본을 계속 지원한다. 배포 환경은 재시작하여 catalog 캐시를 갱신한다. 임의의 수집 문서/과거 대회를 실시간 번역하는 기능은 아니다.

## 주요 변경 파일

- `monitoring/web/src/main.tsx`: 6개 메뉴 그룹과 내부 탭, 기존 주소 유지, 공식 reader lazy import.
- `monitoring/web/src/OfficialPage.tsx`, `official.css`: 공식 문서·언어·revision UI와 반응형 읽기 화면.
- `monitoring/kneeops/translation.py`, `translations/`: SHA로 고정한 번역 catalog와 영어·한국어 본문.
- `monitoring/kneeops/api.py`: `/api/source` 응답의 translation 정보.
- `pyproject.toml`: 번역 파일 package data.
- `monitoring/web/package.json`, `package-lock.json`: Markdown/GFM/math/KaTeX 의존성.
- Python 번역 검사 및 브라우저 navigation/official 회귀 검사, README·STATE·ADR0002·아키텍처·운영 안내·이 인계.

외부 수집 본문의 raw HTML, 외부 이미지, 비HTTP(S) 링크를 실행하거나 표시하지 않는다. 화면의 제목/수식 서식 보정은 저장된 영어 본문과 hash를 변경하지 않는다. 번역 reader는 분리 로딩해 초기 화면에서 Markdown/KaTeX를 내려받지 않는다.

## 검증

- `.venv/Scripts/python -m pytest monitoring/tests tests --basetemp .local/tests/official-20261001 -q`: **39 passed**. 번역 hash, 규칙 전문 항목, 원문 변경 시 이전 번역 차단, 과거 revision 지원, added_at 보존, 다른 대회/종류 분리, API 포함. 기존 FastAPI TestClient httpx deprecation 경고 1개.
- `npm run test:ui` (monitoring/web): **10 passed**. 기존 데이터/목록 기능, 메뉴 6개·기존 주소·뒤로가기, 원문/번역/비교, 규칙 전문·표·수식, 변경 원문 fallback, HTML/링크 안전 처리, 모바일 글자·overflow 확인.
- `npm run build`: TypeScript/Vite 성공. 초기 JS 약 294 kB, 분리된 공식 reader 약 442 kB.
- `compileall monitoring/kneeops knee_workspace` 성공.
- `docker build -q -f monitoring/Dockerfile -t kneeops-local:0.1 .` 성공. `docker run --rm kneeops-local:0.1 python -c "from monitoring.kneeops.translation import catalog; print(len(catalog()))"` 결과 **10**으로 번역 파일 포함 확인.
- desktop `.local/ui-official-bilingual.png`, mobile `.local/ui-official-mobile.png` 시각 검토 완료.
- 실제 in-app 브라우저에서 한국어 본문 확인. 사이드바의 긴 자료 라이브러리 제목을 줄바꿈하던 중복 총계 배지는 제거했다. 자료 수는 라이브러리 화면에서 확인할 수 있다.
- 문서 6개의 로컬 링크 누락 없음. 현재 DB `integrity_check=ok`, online backup `.local/kneeops/backups/2026-10-01T05-58-10.338199_00-00.sqlite` 생성.

브라우저 데이터 검사의 실제 API 미리보기는 전송하지 않은 planned 다운로드 이력을 추가한다. 이번 작업에서는 실제 다운로드·GPU 학습·Kaggle 제출을 실행하지 않았다. 현재 작업 폴더에는 Git 저장소가 없어 commit/diff는 생성하지 않았다.

## 실행과 다음 단계

루트 `KneeOps.cmd`로 실행한다. 공식 문서는 `http://127.0.0.1:8765/#official`에서 볼 수 있다. 백엔드 번역 코드를 변경했으므로 실행 중 서버도 재시작했다.

향후 새 규칙이 수집되면 SHA별 번역 검토를 진행한다. 대회 성능 관련 다음 단계는 [학습 데이터 운영 안내](../runbooks/training-data.md)의 캐시 영상·라벨 점검과 grouped fold 실험이다. Gold58은 노출된 진단셋으로 취급하며 독립 holdout이라고 주장하지 않는다.
