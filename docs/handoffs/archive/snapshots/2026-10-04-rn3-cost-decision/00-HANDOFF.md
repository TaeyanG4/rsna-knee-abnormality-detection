# 통합 인계 · RSNA Knee

갱신: 2026-10-04 16:09 KST. 중앙기록: root / `01a0ff6f-f8dd-7fa0-ab39-2463d7d090d0`.
작업·인계 통합 진입점. 상세는 링크에 보존. 연구 자율진행·검증후보 제출 상시승인(10-04), 공개/유료 별도승인.

## 이전 기록·반복 방지 — 고정 유지

**새 세션은 [누적 결정·실패·재검토 조건](docs/research/decision-memory.md)을 먼저 읽고 관련 근거를 확인한다.**

- **기준본/검증:** 0.943 실제 스택 보존. R2 teacher 누출·Gold 선택 노출·새 ImageNet 모델을 구분. 기존 모델 재분할만으로 독립검증이 되지 않는다.
- **EXP078–083:** 작은 RN/loss 효과와 반복된 골절 손실을 함께 보존. max-span/slot-bias 특정 설정 미채택을 입력·표현 전략 전체 기각으로 확대하지 않는다.
- **Silver:** PFOA20/Synovitis16 감사 완료·자동교정 근거 부족. 미언급은 임상음성이 아니며 동일 가림검토 반복 금지.
- **실행 실패:** 원 hash FAIL/경로 NO_GO·철회된 의심을 구분해 보존. kernel COMPLETE·exit0·sample PASS는 전체 실행/성능 PASS가 아니다. D01 재제출금지·검증/한도/중복확인 유지.

[최근 이력](docs/handoffs/INDEX.md) · [월별 전체 기록](docs/handoffs/monthly/2026-10.md) · [원문/hash 보관](docs/handoffs/archive/INDEX.md)

## 현재 작업·재개

- B/RN56809595 actualv2=.913,기준.943유지. D01 56796964는anchor+옛R2결합hiddenerror/점수없음·원인미확정/재제출없음.
- EXP087진단104초·attempt74/완료73·gradient136비유한/평가0。원FAIL보존·scale1미실행보류. EXP086현recipe−.064575미채택·DINO전체기각아님. EXP085양의불확실·EXP084/RN작은효과와골절손실보존.
- **root재개/CPU:** 학습0。dicom_header_io가anchor+RN3固定90:10실행성읽기검토·16:20목표/16:25상한. D01공통위험/새예산/실패감지확인후판정;원인완전규명/사용자재승인대기아님. 新GPU0.
- [현재근거/정정](docs/handoffs/2026-10-04-1606-feedback-priority.md). evidence/2026-10/2026-10-04-feedback-scale-control/anchor-rn-feasibility.*부터회수·중복dispatch금지. EXP079/080실결합OOF·기존최종재학습계획은존재.

예약: exp078-a-b16:20갱신예정·도구영수증우선. rsna-2는5시간토큰중단복구전용.

## 실행 전·종료 시

[AGENTS](AGENTS.md)→이 파일→누적 결정→[STATE](docs/STATE.md)→관련 상세. 담당자·ID/로그/산출물·예약 대조 후 완료계약 재사용. 새 실행은 M번호·달라진 근거·범위/비용 기록.
중앙문서root병합, 주요변화 즉시·활성최대30분 갱신. [절차](docs/runbooks/handoff-maintenance.md) · [템플릿](docs/handoffs/TEMPLATE.md) · [예정 목록](docs/research/planned-work-2026-10-03.md)

