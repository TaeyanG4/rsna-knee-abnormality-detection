# 통합 인계 · RSNA Knee

갱신: 2026-10-04 05:11 KST. 중앙기록: root / `01a0ff6f-f8dd-7fa0-ab39-2463d7d090d0`.
작업·인계 통합 진입점. 상세는 링크에 보존. 연구 자율진행·제출/공개/유료 승인.

## 이전 기록·반복 방지 — 고정 유지

**새 세션은 [누적 결정·실패·재검토 조건](docs/research/decision-memory.md)을 먼저 읽고 관련 근거를 확인한다.**

- **기준본/검증:** 0.943 실제 스택 보존. R2 teacher 누출·Gold 선택 노출·새 ImageNet 모델을 구분. 기존 모델 재분할만으로 독립검증이 되지 않는다.
- **EXP078–083:** 작은 RN/loss 효과와 반복된 골절 손실을 함께 보존. max-span/slot-bias 특정 설정 미채택을 입력·표현 전략 전체 기각으로 확대하지 않는다.
- **Silver:** PFOA20/Synovitis16 감사 완료·자동교정 근거 부족. 미언급은 임상음성이 아니며 동일 가림검토 반복 금지.
- **실행 실패:** 원 hash FAIL/경로 NO_GO·철회된 의심을 구분해 보존. kernel COMPLETE·exit0·sample PASS는 전체 실행/성능 PASS가 아니다. 사용한 제출승인 재사용 금지.

[최근 이력](docs/handoffs/INDEX.md) · [월별 전체 기록](docs/handoffs/monthly/2026-10.md) · [원문/hash 보관](docs/handoffs/archive/INDEX.md)

## 현재 작업·재개

- **EXP084:** 3fit02:11완료/skip0·고정평가1회. ones−clip −.000096(CI0포함)로 전환미채택. clip RN보완+.000997·골절−.003111(P29) 함께보존. [결과·다음투자](docs/research/exp084-group-loss-result-2026-10-04.md).
- **B_RN3F:** actualv1 공개3 postack99.768초/원10artifact SHA·root14gatePASS. 독립36값exact. platform900은fullHOLD·코드불변32400설정stage CPU준비(05:25). [실결과·후속](docs/handoffs/2026-10-04-0506-d01-error-exp086.md).
- **EXP085:** 처리−대조+.001852/원B+.002154,CI0포함·Baker/group편차. 후보보존/교체보류·재평가금지. [결과](docs/research/exp085-relative-position-result-2026-10-04.md).
- **EXP086:** genericDINOv2-S 전체8epoch CPU준비선택. dicom_header_io구현/precision계약·평가기,목표05:45–06:00·최대06:15,현재GPU0. 기존B예측재사용·Gold없음.
- **D01 56796964:** 05:03 hidden재실행 unhandled error/점수없음 확인. COMPLETE표시≠성공,원인미확정. 승인사용완료·재제출/채점반복조회금지. [상세](docs/handoffs/2026-10-04-0506-d01-error-exp086.md).
- 예약 **exp078-a-b**: 다음05:25 최종설정CPU/EXP086첫ETA20분전. D01채점대기종료.

## 실행 전·종료 시

[AGENTS](AGENTS.md)→이 파일→누적 결정→[STATE](docs/STATE.md)→관련 상세. 담당자·ID/로그/산출물·예약 대조 후 완료계약 재사용. 새 실행은 M번호·달라진 근거·범위/비용 기록.
중앙문서root병합, 주요변화 즉시·활성최대30분 갱신. [절차](docs/runbooks/handoff-maintenance.md) · [템플릿](docs/handoffs/TEMPLATE.md) · [예정 목록](docs/research/planned-work-2026-10-03.md)

