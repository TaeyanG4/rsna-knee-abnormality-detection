# 2026-10-03 11:46 KST · 상위권 방향과 public LB 과적합 방지

- 상태: 완료(검토·문서). 기록자: Codex.
- 요청·결정: 현재 작업/기록을 종합해 최종 상위권 방향 결정. 사용자 추가 요구인 public LB 과적합 방지를 AGENTS·ADR0004에 반영. **0.943 기준본 유지 + 사전 고정 후보 + 작은 clean 대조**.0.943도 private 안정성 입증은 아님.
- 새 관측: 11:34 직접 수집0.943·1,129/4,943위(상위22.84%)·선두0.961, 제출8건. 대상 Raptor/큐 프로세스 없음, 앱 active job/download0. 최근 summary21개·코드4개 hash는 오전 감사와 동일, R2 best checkpoint5개 존재.
- 결과: [현재 실행 계획](../research/route-to-first-2026-10-03.md)을 단일 계획으로 재정리. 첫 탐색안은 EXP072 5fold best 확률평균→rank를 anchor와90/10 혼합(미실행·최적값 아님). hidden-DICOM serving/parity/runtime 후 검토. nested 라벨 outer2개를 별도로 검증, 신규 학습 확대는 gate 통과 후.
- 정정: R2 CV 누출과 제출 활용을 구분. ADR0003/0004의 public routing 전면금지·flat0.60 자동최종선택·CI포함0=동일·3건runtime 외삽·공유 anchor precision 일괄변경을 정리. 약한 라벨CV/Gold/LB/독립검증을 구분하고 사후 비율·소견 반복탐색 금지. 학습 runbook의 현행 코드 한계 명시.
- 근거·검증: [LB](evidence/2026-10/2026-10-03-direction/leaderboard.json)·[hash 재확인](evidence/2026-10/2026-10-03-direction/local-verification.json)·[검사/등록](evidence/2026-10/2026-10-03-direction/validation.json). 문서6개 앱 등록·API 내용 일치, 링크/분량/보존본 hash 검사. [압축 전 원문9개](archive/snapshots/2026-10-03-direction/manifest.json) 유지.
- 미실행/다음: 학습기·추론 코드 수정, 새 GPU 실행·다운로드·제출 없음. 큐 재개 안 함. 다음은①실험 manifest/hidden-DICOM adapter②기존 R2 고정 후보 검증③nested teacher·eval labels/fixed epoch 보완. 제출은 매번 승인, 채점 대기 금지.
- 후속 요구 반영: [handoff 타당성 감사](../research/progress-audit-2026-10-03.md). 기반/0.943/소규모 탐색은 타당, R2 채택 확정·증류 확대와 ‘라벨만 병목/모델상한’ 단정은 수정.224/384 비교의8/12epoch·variant·best/SWA 차이를 확인해 해상도 단독 효과로 해석하지 않음.
- 운영 변경: 사용자 지시로 AGENTS·tools runbook에 장시간 작업 ID/로그/ETA→같은 채팅 heartbeat→실제 종료/산출물 확인→완료 알림·허가된 후속 재개·예약 해제 절차 반영. 기존 ‘예약 감시 없음’ 대체. PC/앱 실행 필요. active job이 없어 새 예약은 생성하지 않음. [공식 문서 확인](evidence/2026-10/2026-10-03-direction/automation-docs.json). 보존 원문은 tools를 포함9개.
- 독립 검토2건 반영: label_mix의 ancestor 검사·report-group bootstrap은 미구현, 현재 study CI. Gold AUC 유사=parity 완료 과장 제거. 총 제출예산6~10회 통일, D01은 실전연결 탐색임을 명시. 확대 gate는LB 전에 고정.
