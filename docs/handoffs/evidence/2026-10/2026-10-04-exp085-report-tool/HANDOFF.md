# 2026-10-04 03:12 KST · EXP085 평가 준비

- 상태: 완료(준비만). 담당 `/root/anchor_inputs`; 실제평가·pilot/full판정은 root 소유. 중앙문서는 수정하지 않음.
- 요청: 상대위치 실험의 독립 정적검토와 최종 1회 평가도구. [검토](static-review.md), [도구](../../../../../tools/exp085_relative_position_report.py), [검사](../../../../../tests/test_exp085_relative_position_report.py).
- 결과: V1 FP32 blocker를 보존하고 explicit autocast-disabled V2 수정 확인. source/protocol/ancestor/input/RNG/종료 gate를 점수 접근 전에 검사한다. 추가 blocker 없음; cleanup의 개별 hard-timeout 한계는 검토에 기록.
- 검증: 새 합성32검사 PASS(1.82초), metadata-only prepare PASS. 03:11:56 KST source17개 hash 일치·평가폴더 없음. [상세](verification.json), [prepare log](prepare.log).
- 동결: [registration](registration.json) SHA `3251ab5b0be7f60b0d9d0e01ebc57ac1b88609d181738a33430b913c9bf9b954`. 평가도구/검사 hash도 연결. 수정 후 같은 registration 재사용 금지.
- 다음: root가 full 종료를 확인한 뒤 [정확한 실행계획](execution-plan.json)의 command를 1회 실행하고 report/SUCCESS를 검증한다. 실패·기존출력 보존. 현재 평가실행 없음.
- 이전결정 M02–05/M09–10 적용: weak·선택노출·미확인 환자연결을 명시하고 EXP084 원B/RN을 계약 일치 시 재사용. paired primary, 양arm−원B mandatory, RN90/10 secondary,500 component bootstrap 고정. EXP082 양의 불확실 근거·RN 작은이득·골절손실을 함께 해석하며 자동채택/ratio/routing 금지.
- 제한: Gold·새예측·라벨값·model state·GPU 접근/실행0, EXP084 재평가0. owner보고52검사는 재실행하지 않음. V888 노출개발/한seed/조건부CI이며 private·0.943 개선을 입증하지 않는다. root가 중앙 HANDOFF에 이 문서를 연결한다.
