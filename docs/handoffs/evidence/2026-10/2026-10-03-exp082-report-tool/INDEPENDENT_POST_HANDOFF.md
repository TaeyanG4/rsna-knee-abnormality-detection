# 2026-10-03 21:41 KST · EXP082 결과 독립 정적 검토

- 상태: 완료. 기록자: Codex precision_code_review. 약4분,15분상한이내.
- 요청·결정: 평가코드·기존 report/analysis_plan·의존helper의 primary/rank/paired bootstrap/UID·label계약과 해석한계를 검토. 평가 재실행·GPU·Gold·원격·중앙문서수정 없음.
- 결과·파일: [독립검토](independent-post-review.json). 치명적 구현오류 발견없음. 고정B를공유하는90/10혼합의 RNones−RNclip이primary. Full868 target별average-rank 후고정,840report-group 공통weights로paired500draw. 12target/P+N·보고수치산술·등록/plan/report/source해시정합확인. 원래라벨·예측에서AUC/CI는재계산하지않음.
- 검증·근거: primary +0.000534467,조건부95%CI[-0.000268009,+0.001293080]; RN단독 +0.006828056,CI[-0.000695851,+0.014910356]. 둘다0포함. 작은양의점추정은보존하되개선입증/동일성입증으로해석불가.
- 남은 일·다음: 24개scanner/field/manufacturer/mask그룹은점추정만있고그룹별CI/interaction/heldout검증없음.500bootstrap은학습seed500회가아니며seed42한번·노출F2·weak라벨·중복/선택/private불확실성유지. 그룹이득/재현성·D01개선주장금지. 이번검토로평가기재실행필요성없음; 향후투자는사전질문/예산에따름.
