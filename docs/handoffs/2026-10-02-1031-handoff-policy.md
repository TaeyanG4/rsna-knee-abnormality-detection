# 2026-10-02 10:31 KST · 모든 작업 기록·압축 규칙

- 상태: 완료. 기록자: Codex. 요청은 모든 작업의 handoff 기록, AGENTS 최적화, 긴 기록 압축/오래된 기록 분리.
- 변경: AGENTS/CLAUDE에 공통 진입·매 작업 기록·크기/보관 기준. [STATE](../STATE.md)는 현재 사실만, [INDEX](INDEX.md)는 최신 링크/최근10건만, [월별 목록](monthly/2026-10.md)은 전체 이력. README 최신 링크도 INDEX로 통일.
- 규칙: 새 인계 기본600~1,200자/최대2,000자·50줄. AGENTS/STATE 각2,500자. 매 작업 길이 점검,7일/신규10건마다 보관 검토,30일 지난 완료 기록 월별 분리. 최근30일·최신5건·미완료 보호. [세부 절차](../runbooks/handoff-maintenance.md).
- 보존: 변경 전6문서 bytes+SHA256 snapshot. STATE/긴 최신 EDA를 요약하고 나머지 최근12개 인계는 그대로 유지. [원문 명세](archive/snapshots/2026-10-02/manifest.json). 예약 작업/앱 코드/데이터/DB 변경 없음.
- 검증: `.venv/Scripts/python -X utf8 -`로 문서12개/로컬 링크92개·원문SHA256 6개·월별14건·새 분량 상한 검사, 오류0. AGENTS 약43%/STATE74%/긴 EDA68% 축약. 결과: `.local/handoff-maintenance-20261002/verification.json`. 앱/학습 검사는 문서 변경이라 재실행하지 않음.
- 남은 일: 이번 요청 없음. 다음 작업부터 이 규칙 적용. 다음 보관 점검은10-09 이후 첫 작업 또는 신규10건. 작업 없는 동안 무인 실행하지 않음.
