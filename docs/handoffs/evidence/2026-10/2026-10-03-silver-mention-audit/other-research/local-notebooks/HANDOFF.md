# 2026-10-03 23:09 KST · 공개 노트북 Silver 처리 소스 감사

- 상태: 완료. 기록자: Codex / precision_code_review. 범위: 지정된 Pilkwang·Flight 로컬 snapshot의 source만; EXP055 receipt는 provenance만 확인. 중앙 문서 미수정.
- 결과: 두 notebook의 35개 대응 cell 중 32개 동일(코드 18/21). 추출기·loader 동일. uncertain-only=.626/conf=.55/loss weight=.6625, silence=.28/.05/.2875로 unknown masking 없음. PF OA의 공식 extent/depth 기준도 구현되지 않음. [보고서](report.md).
- 검증·근거: 표준 JSON parser로 source만 추출·SHA256·cell 대응 비교. [manifest](manifest.json), [cell/line 인용](source-excerpts.txt), [차이](lineage-source.diff). 수식은 독립 검토와 일치. Notebook 실행·import·outputs·실제 Report/label/Gold/prediction 조회·GPU·다운로드 없음.
- 한계: mounted label table이 target/conf를 덮어쓰며 weights 발견 시 inference로 조기 반환. 실제 public run의 label 출처/학습 여부 및 점수 효과는 이 감사로 확정 불가. EXP055에서 PF는 Flight 유지라는 receipt 주장만 확인.
- 다음: 부모 작업의 blind text audit에서 cue scope·PF severity·coverage를 확인. 현재 결과만으로 relabel/재학습하지 않음. 중앙 INDEX/STATE 반영은 부모 담당.
