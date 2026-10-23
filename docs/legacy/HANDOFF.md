# 진행·결과·재개

갱신: 2026-10-05 12:44 KST. 담당: root / `01a0ff6f-f8dd-7fa0-ab39-2463d7d090d0`.

## 현재 작업

- **EXP088 완료·고정 평가 완료.** 07:29→10:54:44, 12epoch/5184updates/skip 0, parent SUCCESS/child 0/resource PASS, cleanup·실제 잔존 0. [수용](docs/handoffs/evidence/2026-10/2026-10-05-exp088-raptor384-v1/full-v4-root-acceptance.json).
- V888 raw AUC B 0.893397 → 384 0.895355, **+0.001958**, 조건부 CI **[-0.003816,+0.007675]**. 8소견 상승/4하락, 골절 이득·MCL 손실/그룹 차이 보존. 작은 양의 불확실 후보이며 교체/5fold 자동 확대하지 않는다. [결과·투자](docs/handoffs/evidence/2026-10/2026-10-05-exp088-raptor384-v1/evaluation-v4-root-result-decision.json).
- **EXP089 T4 v2 완료·root 수용.** 두 모델 worker PASS/supervisor COMPLETE/child 0/cleanup 정상. 각 입력100 byte proof exact·V8 확률 비교 PASS, 새 학습/평가0. B224 감독652.503초/R384 190.704초이며 두 번째 파일 캐시 warming 가능성 때문에 속도 우위로 해석하지 않는다. [29조건·비용·한계](docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-v2-root-acceptance.json).

## 결과·보류

- 원 T4 v1은 kernelspec 누락으로 셀 실행 전 ERROR/모델·입력 검사0였다. metadata만 교정해 별도 검토 PASS 후 v2를 시작했다. 원실패/source/protocol/dataset/완료 local V8 보존·재사용. [변경·비용](docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/root-t4-metadata-fix-execution-decision.json).
- EXP088은 336 cache→384의 레시피 묶음·노출 weak 개발군·한 seed다. .943/private/임상 개선 미확정. 기준 .943/B·RN .913/D01 오류/EXP087 실패/RN3 보류 유지. [일일 이력](docs/handoffs/2026-10-05-0320-exp088.md).

## 재개·예약

- 완료 예약 **exp089-t4 삭제 확인**. active 앱 작업/다운로드0. root는 두 고정 단독 후보의 test-only 제출 경로를 CPU20분 이내 준비한다. 원dataset v1/완료 결과 재사용, 아직 Public3/production/대회 제출0. source·출력/종료·예산 별도 검토 후 실제 무료quota를 확인해 다음 1단계를 판정한다.
- 완료exp088/토큰 복구/5시간/기타 주기 예약은 삭제 상태다. 실제 종료 후속만 유지한다. [계획](plan.md).
