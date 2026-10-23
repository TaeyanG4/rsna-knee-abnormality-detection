# 2026-10-02 18:30 KST · 작업 목록·도구 사용법 정리

- 상태: 완료(문서). 학습은 큐로 진행 중. 기록자: Claude.
- 요청: handoff에 작업 목록, AGENTS에 도구 사용법, Colab 토큰 저장 위치 안내.
- 결과: [도구 runbook](../runbooks/tools.md) 신설, AGENTS에 도구 절 추가(원문 snapshot `archive/snapshots/2026-10-02/AGENTS.md.1820.txt`, manifest 기록). `remote_bundle.py`가 `storage/labels/derived_mix/` 라벨을 번들에 넣고 라벨 외 열은 거부. 학습 도구 테스트 10개 통과.
- 토큰: 로컬은 `~/.kaggle/access_token`(신형 토큰)으로 이미 동작. Colab은 파일이 아니라 보안 비밀 `KAGGLE_API_TOKEN`, 사용자가 직접 입력.

## 작업 목록 (ADR 0003 순서, gate 통과 시 다음)

| # | 작업 | 조건·방법 | 완료 기준 |
|---|---|---|---|
| 1 | EXP060 fold 0(384 px) | **완료** 18:58 | OOF 0.896 |
| 2 | 큐: hash 재계산 → Gold58 fp16 → R1 F0~F4 | 자동, 러너 pid 16136 | `tools/gpu_job.py --list` 전부 SUCCESS |
| 3 | Colab 첫 실행 | **보류**(10-02 사용자 결정). 다시 쓸 때 토큰 등록 후 L4 짧은 실행 | `fetch-results` 성공 |
| 4 | R0 판정 | **통과**(Gold58 fp16 0.9026) | weak OOF ≥ 0.88, Gold58 ≥ 0.90. 미달 시 maxspan-v5 fine-tune |
| 5 | R1 판정 | **미채택**(F0 OOF 0.8806 vs 0.8963). 384 fold 1~4 큐 추가 | fold 0이 EXP060 대비 −0.01 이내, 스캐너 격차 확인 |
| 6 | R2 라벨 | **효과 확인** +0.016(3 fold). 균일 0.5 채택. 2차: 이미지 0.7·두 teacher·반복 distillation | 기준 라벨 재채점 +0.005, Gold58 비하락, bootstrap 구간 > 0 |
| 7 | R3 기하 | 80~96장·384 px 새 corpus 계약 | +0.003 이상 재현 |
| 8 | R4 최종 학습 | 384 px는 여기서만(10-15 무렵). 전체 데이터 단일 + seed 1개 | OOF·seed 차 기록 |
| 9 | R5 앙상블 | nested OOF 가중치, flat-0.60 anchor 유지 | Gold58 튜닝 없음 |
| 10 | R6 제출 노트북 | Kaggle fp16, 공개 테스트 3 study parity | 차 ≤ 0.001, 9 h의 70% 이내 |
| 11 | 제출·최종 2개 선택 | 사용자 승인 후 | 10-21~22 |

- 일정: 10-02~04 R0·R1, 10-05~10 R2, 10-11~15 R3·R4, 10-16~20 R5·R6.
- 제출 전 공개 가중치·외부 데이터 규칙 재확인. 선택 정리: 비공개 테스트 데이터셋 `knee-raptor-res-remote-smoke-kaggle` 삭제.
- R1 fold를 원격으로 옮기면 큐 계획에서 그 fold를 빼고 러너를 재시작한다.
