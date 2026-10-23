# 2026-10-04 01:04 KST · EXP084 실행과 serving 준비

- 담당 root / 세션 `01a0ff6f-f8dd-7fa0-ab39-2463d7d090d0`. 주요 기록은 [루트 HANDOFF](../../HANDOFF.md)에서 재개. [이전 문서전환](2026-10-04-0035-main-handoff.md)과 원문/hash 보존.
- EXP084 CPU 등록/검토 완료: T3455/V888, component3323/873·교차0. RNclip20/RNones20/새B8 세fit을 고정. [학습 구현/명령](evidence/2026-10/2026-10-04-exp084-implementation/HANDOFF.md) · [평가 사전등록/후속명령](evidence/2026-10/2026-10-04-exp084-report-tool/HANDOFF.md). synthetic9+23/static14·root binding 검토 결과 재사용, 등록sha18baee07… 확인.
- 01:03:49 순차큐 실행: runner39536, 첫 RN wrapper30860/supervisor34460/train31936. [root 실행 결정](evidence/2026-10/2026-10-04-next-investment/root-launch-decision.json) · [launch receipt](evidence/2026-10/2026-10-04-next-investment/queue-launch-receipt.json). 원queue와 byte동일 사본 사용해 `artifacts/logs/queue/exp084_queue.log`로 분리. 실패 후 중단/retry0·총7200초, RN각1800/B3600초.
- 01:05 첫epoch51초·VRAM1.9GiB, 실제 초기값/UID/FP16/Gold·metric차단 일치. 전체ETA02:10–02:30은 RN초기속도+과거B 근거이며 현재B 미측정. [실행관측](evidence/2026-10/2026-10-04-next-investment/running-checkpoint.json). 세fit 완료전 예측/점수 개봉금지; 모두SUCCESS와계약 확인후 전용평가기1회.
- B_RN3F 기존6weights CPUstage 완료/37합성PASS. [계획](evidence/2026-10/2026-10-04-b-rn3f-serving/PLAN.md). anchor_inputs 독립리뷰01:15상한, dicom_header_io 별도 B수치gate준비01:25상한. source/stage불변·GPUreference/T4 미실행. local reference는 학습과 겹치지 않게 후속 결정.
- Biohub: 작은 RN/loss 효과를 보존하며 새B로 원blend질문 유지. 새group는 weak·과거연구자노출·환자독립/Dread계보 불명확이며 정확D01/0.943/임상 성능 증거 아님. EXP084는 inputA 부재로 기존골절회복 입증불가.
- D01 제출56796964 재제출/조기조회 금지,05:02 첫조회 유지. 앱jobs/downloads0은01:04관측. 같은예약 exp078-a-b 01:15 CPU검토확인, 이후 가장 이른실행ETA로 갱신. 새제출/공개매회승인·유료별도.


- 사용자01:11: 남은기간 성과위해 연구우선순위 자율판단. [운영반영](evidence/2026-10/2026-10-04-next-investment/autonomous-priority-clarification.json).

- 01:31 RN두fit완료/skip0·B실행, local입력100+합성6 PASS. [실측/담당/후속](evidence/2026-10/2026-10-04-next-investment/checkpoint-0131.json). 다음01:42; 상세리뷰·T4/producer연결은별도. [압축전원문](archive/snapshots/2026-10-04-exp084-progress/manifest.json).
