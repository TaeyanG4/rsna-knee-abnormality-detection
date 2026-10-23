# 2026-10-03 20:32 KST · DINO 후보 검색 v2

- 상태: 완료. 기록자: Codex 하위 검토 에이전트.
- 요청·결정: v1보존, 기존특징/마스크/UID/fold만으로 고정4343건 후보검색1회. CPU6·900초·RSS1GiB 상한. 원본/캐시픽셀·label/prediction/Gold값·Report텍스트 미사용.
- 결과·파일: [등록본](registration.json)과 source/test SHA를 먼저 동결. [결과](run/result.json): 정확특징 교차fold그룹1개(2건), 상호최근접692쌍 중 top20 동결. 분석0.486초·peak173.55MiB. 환자중복을 확정하지 않으며 split/학습 변경 없음.
- 검증·근거: [unittest23통과](tests-result.txt), [출력무결성/해시](verification.json). 모든4343건의 교차fold/같은슬롯패턴·상호성·순서 확인. 실패·재시도 없음. 등록 요약은 float32 mean이라 썼지만 실제 동결코드는 mean/slot L2 중간연산 float64: [산술 정정과 해석](report.md), source/등록/결과는 보존.
- 남은 일·다음: 정확특징2건의 UID→cache연결 및 제한된 pixel/header 확인을 별도 범위로 판단. 고정top20 blinded 검토 후에만 미래split 필요성 판단. 다른 획득 패턴·teacher노출·LB과적합·T4/local일치는 이 검사로 해결되지 않음. 중앙 STATE/INDEX는 루트 담당.
