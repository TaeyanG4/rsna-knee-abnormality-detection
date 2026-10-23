# 예정 작업과 실행 조건

2026-10-04 00:45 KST 갱신. 현재 진행/담당자는 [루트 HANDOFF](../../HANDOFF.md)를 따른다. 사용자 질문에 대한 현재 우선순위. 시간은 기존 실측을 참고한 계획 추정이며 완료 약속이 아니다. 신규 제출 승인은 포함하지 않는다.

|순서|작업|시간·시점|시작/종료 조건|
|---|---|---|---|
|1|기존 B+RN 제출 경로 감사 완료|10-04 00:17|6가중치 계약 확인. rank/fold 집계·패키지·T4 검증은 미완료. [감사](../handoffs/evidence/2026-10/2026-10-04-candidate-readiness/report.md)를 재사용해 fold별90/10rank의3fold평균을 고정. CPU패키지45–75분 준비중, 원격/추론 미시작|
|2|새 group 대조 제안 완료·등록 전 검토|10-04 00:20|[제안](../handoffs/evidence/2026-10/2026-10-04-fresh-group-plan/report.md)에서 원래 blend 질문은 새B 포함3fit 권고. 3fit 선택·CPU adapter/평가기 준비중. 초기값 일치 agent보고/root검토 전, 새split/학습 미시작|
|3|D01 승인제출 결과 확인|10-04 05:02 첫조회|완료시점수/실행상태확인,미완료면갱신ETA재예약. 같은가설재제출·LB비율튜닝없음|
|4|소규모 새group GPU 대조|선택:RNclip20/RNones20/B8,66–82분 계획|등록/CPU gate/root검토 후1회 순차실행, 전체120분 상한·자동재시도0. 현재미시작. 추가fold/seed/384자동확대아님|
|5|선택된후보1개의최소T4검증·출력완성|조건부:경로1확인후실측/ETA산정|개발근거/골절·group손실/비용검토,무료quota확인. 입력수치·offline·메모리·시간검사;기존D01gate를다른후보에전용금지|
|6|검증된구체후보제시·승인후제출|후보준비후|매회사용자승인. 일5회는상한/09시리셋. 서로다른가설을준비하며같은비율변형으로횟수채우지않음|

Silver는 PFOA20 및 Synovitis16 대조까지 완료했다. 현재 표본에서 실제 Dread를 자동수정할 명확한 근거는 부족하다. 미언급/proxy를 임상 음성으로 바꾸지 않는다. 따라서 Baker추가표본·PFOA80확대·Silver대량교정/재학습은 조건부 보류다. 새로운Dread정의충돌 또는독립영상근거가확보되면재검토한다.

[Synovitis 결과](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/synovitis-pilot16/comparison-v1/report.md) · [Gold노출후검증계획](gold-exposure-validation-plan-2026-10-03.md) · [현재상태](../STATE.md).
