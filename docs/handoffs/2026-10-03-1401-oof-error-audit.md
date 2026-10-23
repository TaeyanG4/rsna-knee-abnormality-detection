# 2026-10-03 14:01 KST · 기존 OOF 소견·그룹 오류 기여

- 상태: 완료. 기록자: Codex subagent. 요청: EXP078 학습 중 기존 R1/R2/ResNet34 예측으로 CPU 오류 분해·다음 가설 도출. 현재학습·queue·manifest 미변경.
- 결과: [분석](../research/oof-error-contribution-2026-10-03.md), [전체근거](evidence/2026-10/2026-10-03-error-audit/audit.json), [12소견/P/N/CI](evidence/2026-10/2026-10-03-error-audit/targets.csv). 고정원라벨4,345·best15개: R1 .878263, ResNet .866471, R2 .892580. R2는간접누출·혼합라벨epoch선택, 다른둘도선택된개발OOF.
- 핵심: MCL/PF OA/외측반월판/외측OA가 R1 weak deficit47.6%. ResNet PF OA +.012005,5fold양수; 전면교체 근거 아님. 4slot군 점수가5slot보다 높아 결측=저성능 단정 불가. scanner·희귀양성·report-group≠환자독립 한계 기록. Gold58은 별도 노출진단.
- 도구·검증: [oof_error_audit.py](../../tools/oof_error_audit.py) 신규(기존metric adapter 재사용, 덮어쓰기 거부). tie/가중AUC·pair기여합·paired bootstrap CPU검사3통과, 감사9.8초. [54개source hash/검사](evidence/2026-10/2026-10-03-error-audit/verification.json). GPU추론·학습·제출 없음.
- 전략점검→조치: macro만 보기 보완. EXP078종료 때12소견/그룹 검토, PF OA 표현/plane·MCL정의 점검을 구조/감독 후보로 구체화. 작은Δ/라벨가설 영구폐기 금지. 추가확대 전 환자/영상중복 확인범위를 명시. 인계색인·STATE·runbook 통합은 root 담당.
