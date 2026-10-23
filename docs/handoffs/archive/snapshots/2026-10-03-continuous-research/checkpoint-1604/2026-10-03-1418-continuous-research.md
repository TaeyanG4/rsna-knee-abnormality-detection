# 2026-10-03 14:18 KST · 지속 연구 재개·D01 원격 준비

- 상태: 진행,15:36갱신. Codex. 중단전연구·연구실활용·일5회기회 유지. 제출/공개매회승인·유료별도승인. [절차](../runbooks/continuous-research.md).
- EXP078: A_F0/B_F0완료,A_F1 epoch8후최종평가,B_F1대기. 네fit 전효과판정 없음. 러너20764,[로그](../../artifacts/logs/queue/queue.log)·[명세](../../artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/manifest.json)·[실측](evidence/2026-10/2026-10-03-continuous-research/progress-checkpoint.json). 전체ETA16:20~16:40. 학습4파일동결. 완료후 input_ab_report.py→input_ab_diagnostics.py(P/N·소견CI·장비/결측;CPU3검사)와Biohub점검.
- D01고정90/10·원본30셀보존. [ID·결과·실패](evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json). v1입력hash FAIL유지. 새reference계약은15:04사전고정,로컬/KaggleCPU100+합성7 모두PASS,정규화max오차7.73e−7. [원격검증](evidence/2026-10/2026-10-03-input-gate-v2/remote-verification.json). codec6syntax/3압축fixture PASS,대회압축전수보증아님.
- R2 T4 100건603.86초·5모델finite·실패0. anchor50실제완료1,082.18초·49+순차4CoAt경로. 단순1,300건합산9.785h로공동9h gate보류;고정비용중복 외삽이며full실패증명아님. releasev2 CPU14검사·receipt결합/child격리/공동deadline구현,최종T4와시간gate 전후보생성 안 함. 이번제출0회.
- Header-only100건volume/mask/receipt동일·읽기72.05%감소,로컬85.29→87.39초로느려미채택. 비공개CPU knee-d01-header-io100 v1을15:31시작,최대1h·ETA15:45~16:00. 교차순서100건비교 후채택판단. [근거](evidence/2026-10/2026-10-03-continuous-research/dicom-header-io-review.json).
- [MCL120 동결후대조](evidence/2026-10/2026-10-03-mcl-review/unblind-v1/report.md):음성후보55/unknown56/미언급9,임상양성판정아님. 음성55에서Dread+0,다른소스+31~36. MCL2fit은새수정근거/독립평가기준부족으로보류,라벨전략전체기각안함. [5가설·비용비교](../research/execution-priorities-2026-10-03.md)에서입력대조/PFOA표현우선. 자동재라벨없음.
- 수집실패:동명dataset409→노트북명수정. CLI전체output의dependency수집/페이지429중단→SDK정확version/file회수. [명세](evidence/2026-10/2026-10-03-continuous-research/remote-receipt-fetch.json). frozen4/링크검사PASS·앱API revision확인.
- 같은채팅 exp078-a-b ACTIVE,다음15:45확인;미완료면갱신ETA재예약. [축약전보존](archive/snapshots/2026-10-03-continuous-research/checkpoint-1536/manifest.json). INDEX36건유지,무인보관없음.
