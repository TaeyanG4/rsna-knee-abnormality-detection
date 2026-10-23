# 인계 보관 색인

기본 진입은 [루트 HANDOFF](../../../HANDOFF.md)와 [누적 결정](../../research/decision-memory.md)다. 보관 기록은 그 시점의 사실이며 현재 구현의 증명이 아니다.

## 월별 완료 기록

아직 없음. 10-03 점검한36건 모두 최근30일 기록이다. [월별 목록](../monthly/2026-10.md)에서 찾는다. 30일·최신5건·미완료 보호 규칙은 [관리 절차](../../runbooks/handoff-maintenance.md)를 따른다.

## 압축 전 원문

| 보관일 KST | 대상 | 검증 명세 |
|---|---|---|
| 2026-10-02 | AGENTS, CLAUDE, README, STATE, TEMPLATE, viewer EDA 인계의 변경 전 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-02/manifest.json) |
| 2026-10-03 | 자원 방침 추가 후 AGENTS·재감사 반영 전 STATE, 축약 전 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-replan/manifest.json) |
| 2026-10-03 | 노트북 재선정 전 INDEX·STATE·전체 계획·ADR0003·월별 목록 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-notebook-selection/manifest.json) |
| 2026-10-03 | 방향 재정리 전 AGENTS·STATE·ADR0003/0004·전체 계획·학습/tools runbook·INDEX·월별 목록 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-direction/manifest.json) |
| 2026-10-03 | 작업 과정 재검토 전 누적감사·전체 계획·STATE·INDEX·월별 목록 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-process-review/manifest.json) |
| 2026-10-03 | 기준본 직접 학습 반영 전 STATE·INDEX·월별 목록 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-anchor-study/manifest.json) |
| 2026-10-03 | T4·4070 호환성 감사 반영 전 STATE bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-gpu-parity/manifest.json) |
| 2026-10-03 | 우선순위 실행 전 AGENTS·STATE·tools/training runbook·전체계획 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-execution/manifest.json) |

| 2026-10-03 | Biohub 교훈 반영 전 AGENTS·STATE·INDEX·월별·실행우선순위 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-biohub-review/manifest.json) |

| 2026-10-03 | 지속연구 정책 반영 전 AGENTS·STATE·tools/remote·ADR0004·계획 bytes | [원래 경로·SHA256·크기](snapshots/2026-10-03-continuous-research/manifest.json) |

원문은 `.md.txt`. Markdown 안 상대 링크를 해석할 때는 manifest의 원래 경로를 기준으로 한다. 일반 작업에서 전체 원문을 읽을 필요는 없다. 복원 시 현재 요약과 비교하고 덮어쓰기 전 별도 보존한다.

| 2026-10-04 | 루트 통합 인계 전환 전 원문10개 | [원래 경로·SHA256·크기](snapshots/2026-10-04-main-handoff/manifest.json) |
| 2026-10-04 | 과거 결정 진입점 보완 전 HANDOFF/AGENTS/INDEX/TEMPLATE/절차5개 | [원래 경로·SHA256·크기](snapshots/2026-10-04-history-entry/manifest.json) |

- [2026-10-04 EXP084결과 갱신 전 6문서](snapshots/2026-10-04-exp084-result/manifest.json): 원문·bytes·SHA보존.

- [10-04 release 검토 전 중앙기록](snapshots/2026-10-04-release-review/manifest.json): 원문/해시 보존.

- [10-04 EXP085結果 이전기록](snapshots/2026-10-04-exp085-result/manifest.json): 중앙기록/예정목록 원문과해시.

- [10-04 D01실패확인 전](snapshots/2026-10-04-d01-failure/manifest.json): 중앙기록/예정목록 원문보존.

- [10-04 EXP086 pilot 이전 중앙기록](snapshots/2026-10-04-0534-launch/manifest.json): 원문/bytes/SHA 보존.

- [10-04 V2pilot 이전 중앙기록](snapshots/2026-10-04-0603-v2pilot/manifest.json): 원문/bytes/SHA보존.

- [10-04 EXP086 결과 전 중앙기록](snapshots/2026-10-04-exp086-result/manifest.json): 원문/bytes/SHA보존.

- [10-04 EXP087 CPU준비 전 기록](snapshots/2026-10-04-exp087-cpu/manifest.json): 원문/해시·1936표현 정정전 보존.

- [10-04 EXP087 최종검토 전](snapshots/2026-10-04-exp087-review/manifest.json): 원문/해시보존.

- [10-04 EXP087 v3 CPU시험 전](snapshots/2026-10-04-exp087-v3-pilot/manifest.json): 중앙기록 원문/bytes/SHA 보존.

- [10-04 EXP087 315 최종검토 전](snapshots/2026-10-04-exp087-scale-review/manifest.json): 중앙기록 원문/bytes/SHA 보존.

- [10-04 EXP087 315 실행 전](snapshots/2026-10-04-exp087-stage-start/manifest.json): 중앙기록 원문/bytes/SHA 보존.

- [10-04 EXP087 학습 준비 전](snapshots/2026-10-04-exp087-training-start/manifest.json): 중앙기록 원문/bytes/SHA 보존.

- [10-04 상시제출승인 적용전](snapshots/2026-10-04-standing-submission-approval/manifest.json): 이전 승인규칙과 중앙기록 원문/bytes/SHA보존.

- [10-04 EXP087 GPUpilot시작전](snapshots/2026-10-04-exp087-pilot-start/manifest.json): 중앙기록 원문/bytes/SHA보존.

- [10-04 EXP087 full시작전](snapshots/2026-10-04-exp087-full-start/manifest.json): 중앙기록 원문/bytes/SHA보존.

- [10-04 EXP087 full실패이전](snapshots/2026-10-04-exp087-full-failure/manifest.json): 직전인계원문/hash보존.

- 2026-10-04 토큰복구·재개 갱신 전 원문: [중앙문서 snapshot](snapshots/2026-10-04-token-recovery-resume/manifest.json).

- 2026-10-04 11:14 사용자 일정정정 전 원문: [정상ETA·토큰복구 구분](snapshots/2026-10-04-token-cadence-correction/manifest.json).

- 10-04 16:09 [피드백대조 전 원문6건](snapshots/2026-10-04-feedback-priority/manifest.json).

- 10-04 16:28 [RN3비용준비 결정 전6건](snapshots/2026-10-04-rn3-cost-decision/manifest.json).

- 2026-10-04 문서역할·CLI변경전: [원문 16개/hash](snapshots/2026-10-04-agent-docs-v1/manifest.json). M01–M11은루트discoveries로승계.

- 2026-10-05 규칙·문서 규격 변경 전: [원문 14개/hash](snapshots/2026-10-05-agent-rules-v2/manifest.json). 정기 기록 폐지·조건별 읽기·도구화 적용.
