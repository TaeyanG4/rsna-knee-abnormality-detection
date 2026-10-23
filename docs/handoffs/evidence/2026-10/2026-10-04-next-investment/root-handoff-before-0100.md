# 통합 인계 · RSNA Knee

갱신: 2026-10-04 00:48 KST. 중앙기록 담당: Codex root, 세션 `01a0ff6f-f8dd-7fa0-ab39-2463d7d090d0`.

모든 작업·인계·과정은 여기에 요약한다. 긴 설명/명령/로그는 별도 문서에 연결하고 과거 기록은 보존한다.

## 이어받는 순서

1. [AGENTS](AGENTS.md) → 이 파일 → [STATE](docs/STATE.md) → 현재 작업의 필요한 문서만 읽는다.
2. 담당 세션/agent·실행ID/로그/산출물·기존 예약을 실제 상태와 대조한다. 완료된 동일 계약은 재사용하고 진행 중 작업은 회수한다. 오래된 기록만으로 종료를 추정하지 않는다.
3. 범위/비용/승인 확인 후 실행한다. 주요 변화 즉시·활성 작업 최대30분마다 갱신한다. root가 병합하며 대기 확인루프는 금한다.

[상세 인계 절차](docs/runbooks/handoff-maintenance.md) · [작성 틀](docs/handoffs/TEMPLATE.md) · [도구/재개 명령](docs/runbooks/tools.md) · [지속 연구](docs/runbooks/continuous-research.md)

## 현재 작업과 재개

- **EXP084 준비 중**: anchor_inputs가 새 그룹 분할/no-Gold 학습 adapter·등록, precision_code_review가 평가기 구현. RNclip20/RNones20/새B8 총3fit 선택. CPU 목표01:10, adapter 상한01:15/평가기01:20. 아직 GPU 학습/큐 없음. 학습 계획66–82분·전체상한120분, 자동재시도0.
- **B_RN3F 준비 중**: dicom_header_io가 기존6가중치의 CPU serving 구현. 전체 UID에서 fold별90/10 rank 후3fold 평균, anchor 혼합 없음. 목표01:20–01:35/상한01:50. 로컬GPU reference·T4·제출 미실행.
- 두 작업의 사전 선택·예산·범위는 [root 결정](docs/handoffs/evidence/2026-10/2026-10-04-next-investment/root-decision.json). 원본4학습소스/기존protocol 불변. agent의 코드/commands/검사를 회수해 root가 검토한다. 미완성 명령을 추정해 실행하지 않는다.
- **D01 제출56796964**: candidate-v3 실제v1,10-03 20:46:11 승인사용/제출. 마지막 실제 조회20:46:32 PENDING. 첫 결과조회10-04 **05:02**, 보수ETA05:22. 조기조회/재제출 금지. 133건 실행PASS이나 hidden시간/메모리·정확도 개선 미확인.
- 예약 **exp078-a-b**: 다음 **00:50** 준비상태 확인 예약. 미완료면 실제 진행량으로 ETA 갱신. 00:31 앱activejobs/downloads0·학습프로세스0 관측. agent CPU 준비와 모델 학습을 구분한다.

## 완료·보류와 다음 행동

- 완료 감사는 재사용한다. [작업별 근거·이번 변경](docs/handoffs/2026-10-04-0035-main-handoff.md).
- Silver PFOA20/Synovitis16 가림대조 완료·자동교정/확대 보류. RN 작은 효과/골절 손실 보존. [직전 연구 요약](docs/handoffs/2026-10-03-1740-six-hour-reassessment.md).
- 다음: 기존 agent 결과 회수 → 계약/한도 검토 → EXP084 한 번의 순차 GPU 대조 여부 결정. B_RN3F는 별도 수치/T4 검증 전 후보 완료로 부르지 않는다. 새 제출·공개 매회승인, 유료별도. 기존 D01 승인은 사용 완료.

[예정 목록](docs/research/planned-work-2026-10-03.md) · [과거 기록·월별 이력](docs/handoffs/INDEX.md)
