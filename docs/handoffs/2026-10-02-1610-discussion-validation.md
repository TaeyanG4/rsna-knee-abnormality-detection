# 2026-10-02 16:10 KST · 토론 유효성 판단·검증 프로토콜·개선 작업 착수

- 상태: 완료(판단·프로토콜), 학습은 진행 중. 기록자: Claude.
- 요청: 토론 수집 상태와 유효/무효 판단. 이어서 검증 방법을 세우고 private에 유효하도록 1등 목표 개선 작업 진행.
- 토론 수집: 라이브 143 = DB 143, 원글·댓글 트리 완전, 잘림·빈 원글·HTML 잔여 0. 제목 모집 필터가 기술 글 1개(743374 빠른 추론)를 건너뛴 것을 수집하고 필터를 으로 변경(test 갱신). 판단표: [토론 판단](../research/discussion-triage-2026-10-02.md) — 유효 23건(라벨 lever, 기하·창 수, 스캐너 CV, fp16, 빠른 추론, 단일 fold 전체 학습), 참고 20건, 무효 약 100건(모집·운영·음수 추천).
- 검증 프로토콜: [ADR 0004](../decisions/0004-validation-protocol.md). 보고서 그룹 5-fold 고정, Gold58은 gate만, 선택은 weak OOF, 스캐너 그룹별 OOF 격차 감시, nested 블렌드, 제출 규칙, fp16 parity.
- 도구: `training/src/study_meta.py`(ZIP 헤더→`storage/labels/derived_meta/study_meta.csv`, 4,410 study, 스캐너 13그룹·성별·좌우. 첫 구현은 경로/오프셋 짝 오류로 3,560 unknown → 수정), `raptor_report.py`(스캐너 그룹 분석·`--compare` paired bootstrap), `label_mix.py`(OOF 혼합 라벨: blend/silence/target, Gold58 혼입·중복·커버리지 검사), `tools/gpu_queue.py`(순차 실행 큐). `raptor_corpus.py`의 uids.csv 문자열 구문 오류 수정.
- 실행 중: EXP060 fold 0(epoch 1 OOF 0.776·gold 0.751, 상승 중). 큐 `artifacts/logs/queue-r1-20261002.json`(pid 37736): EXP060 완료 → gold fp16 추론 → EXP061_R1_F0~F4(224 px, k 12, bs 8, 8 epoch) 순차. 로그 `artifacts/logs/queue/queue-r1-20261002.log`.
- 남은 일: (1) R0 gate 판정(`EXP060_GOLD_FP16` 결과, 기대 Gold58 ≥ 0.90). (2) R1 5 fold 후 `raptor_report.py --prefix EXP061_R1`로 OOF·스캐너 격차 확인, 384 대비 차 ≤ 0.01이면 fast loop 채택. (3) `label_mix.py`로 OOF 혼합 라벨(alpha 0.5, silence) 생성 → EXP062 학습, `--compare`로 판정. (4) 주최 미응답 규칙(공개 가중치·외부 데이터)은 제출 전 재확인.
