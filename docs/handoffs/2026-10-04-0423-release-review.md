# B/RN 최종 게시 검토 · EXP085 진행

2026-10-04 04:23 KST · root. M07–M10 적용: 기존 T4 수치PASS를 재사용하고 새 게시/시간 계약만 검토했다. [이전 T4 결과](2026-10-04-0330-b-rn3f-t4-result.md).

- **새 결함:** CPU stage v2의 CSV rename 이후 metadata 실패가 FAILED 상태에 CSV를 남길 수 있었다. 최종 ack clock·elapsed 공백도 확인했다. 실제 원격 실패가 아닌 정적 검토다. [독립 검토](evidence/2026-10/2026-10-04-b-rn3f-release-v1/independent-code-review.json).
- **교정:** dicom_header_io가 publication/ack rollback·최종 clock 최소수정, 새 v3와 반례 준비 중. anchor_inputs가 변경분만 검토한다. 목표04:28–04:31/상한04:35. 원 v1/v2·46검사·100건 결과 보존, 중복 추론0. [root 범위](evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-pre-dispatch-repair.json).
- **시간 판정:** 설치 CLI의900은 실제 SaveKernel session timeout이다. hidden scoring 상속/override 근거는 없다. 이번 public3는 platform900/내부600·2T4 최대0.5GPUh·retry0의 기능검증만 유지한다. full 내부32100/플랫폼32400 설계가 현재900 버전의 실행가능성을 입증하지 않는다. 통과해도 제출 준비 HOLD, 별도 최종설정 판정 필요. [결정](evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-timeout-decision.json).
- **실행 전:** v3 정확source/manifest·변경반례·독립검토/root 검사 후 exact execution-plan만 사용. 아직 새 remote/push/제출0. 04:14 CLI GPU27.70h잔여,04:21 앱 active jobs/downloads0. 무조건32400으로 늘리고0.5h cap이라고 부르지 않는다.
- **EXP085:** 04:12 control완료/treatment4/8·1724updates·skip0·epoch290.109초. wrapper35936/helper40684/train7820. 현속도 ETA04:32–04:36(미완료),기존계획04:36–04:57. 원source/총7200초 불변. 양arm/phase/resource/SUCCESS 후 [고정평가기](evidence/2026-10/2026-10-04-exp085-report-tool/execution-plan.json)1회. 중간점수/Gold 없음.
- **재개:** 같은 예약 exp078-a-b를04:33으로 갱신. 기존 agents 결과부터 회수,상한 도달 시 산출물보존·재판정. D01 56796964 첫조회05:02 유지/승인사용완료·재제출금지. 새 제출/공개/유료는 매번승인.
