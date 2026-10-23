# 2026-10-03 17:40 KST · 6시간 방향·검증 재점검

- 상태: 연구 진행, 10-04 00:25 갱신. 중앙기록 담당 Codex root / 세션 `01a0ff6f-f8dd-7fa0-ab39-2463d7d090d0`. 사용자 요청: 다른 세션 재개·중복 방지·주기적 문서 갱신.
- 운영 변경 완료: AGENTS 2,485자. 시작/완료/실패/판정/범위변경/예약·반환 즉시, 활성 작업 최대30분마다 같은 기록 갱신. 담당/ID/산출물/승인 확인→동일 계약 완료 결과 재사용. 대기 확인루프·기록전용 예약 없음. [절차](../runbooks/handoff-maintenance.md)·[검증](evidence/2026-10/2026-10-04-session-resume/verification.json)·[원문](archive/snapshots/2026-10-04-session-resume/manifest.json).
- CPU 기본12워커, 효율 저하 시10→8/필요시 더 낮춤. 실제 할당·RAM/I/O·동시작업·내부스레드 합산. 동결실험/기존 job 변경 없음.
- 후보경로 CPU 감사 완료 00:17(dicom_header_io). B/RN 6weights·hash/입력 계약 확인. 공용 추론은 raw 평균이며90/10 rank 및 test fold 집계 미구현/미고정. 후보·T4 PASS 아님. [감사/다음 비용](evidence/2026-10/2026-10-04-candidate-readiness/report.md). root 최종 채택 검토 전.
- 새group 제안 완료 00:20(anchor_inputs): 원래 blend 질문은 새 RNclip20/RNones20/B8 총3fit, 계획66–82분. RN2fit26–36분은 단독 loss 질문. source24 hash 확인, 새split/학습/Gold·예측 조회 없음. 새adapter·B초기자산 계보 확인 필요. [제안](evidence/2026-10/2026-10-04-fresh-group-plan/report.md). 아직 등록/큐/실행 아님.
- 다음00:30: 위 두 결과를 재감사 반복 없이 회수·root 검토→정확 arm/집계식/비용 결정→선택한 후속 최소구현. [예정목록](../research/planned-work-2026-10-03.md). 기존B를새V에재사용 금지, weak/연구자노출/환자독립 미확인 유지. 허가된 로컬 소규모 연구 범위.
- Silver PFOA20/Synovitis16 가림대조 완료·재실행 금지. Dread 자동교정/추가표본/학습 근거 부족, 미언급/proxy≠임상음성. [종합](../research/silver-other-research-and-pfoa-2026-10-03.md). source 차이를 정확도 차이로 전용 금지.
- EXP080 RN 작은 효과·B 골절 손실, EXP082 작은 양의불확실 효과 보존. EXP081/083 현설정 미채택·전략 전체 기각 아님. 추가fold/seed/384 자동확대 없음. 내부개발 결과는0.943/정확D01 개선 증거 아님.
- D01 **56796964**, candidate-v3 실제v1. 10-03 20:46:11 승인 사용·제출,20:46:32 PENDING/점수 미확인. 첫조회10-04 **05:02**, 보수ETA05:22. 조기조회/재제출 금지. 133PASS이나1300추정8.594h/9h·29.541/30GiB로full미보장. 원hashFAIL/경로NO_GO 보존. [상태](evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json).
- 예약 `exp078-a-b`00:30 후속 검토. 00:13 앱activejobs/downloads0·00:07 로컬학습0 관측. 신규학습/원격/제출0. 제출/공개 매회승인·유료별도. Biohub: 작은효과/반대근거 보존·도구PASS≠성능·끝없는 감사 방지.
