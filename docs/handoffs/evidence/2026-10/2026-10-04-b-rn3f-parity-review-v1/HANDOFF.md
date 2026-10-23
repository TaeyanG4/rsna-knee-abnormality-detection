# 2026-10-04 01:35 KST · B_RN3F parity V1 독립 리뷰

- 상태: 완료. 기록자/담당: Codex `/root/anchor_inputs`. 01:27 시작, 01:35 최종 관측. 출력은 이 evidence만; 중앙문서는 root 담당.
- 요청·결정: frozen V1 입력/수치 gate 독립 검토. 수학·입력 계약의 확인된 blocker는 없음. 자동 완료 판정에 운영 공백2건이 있어 명시적 acceptance 보강 필요. V1·임계값은 보존.
- 결과·파일: [상세 판정·hash](review.json), [상위 실패 반례](synthetic-counterexample/parent-failed-paired.json), [수치 실패 반례](synthetic-counterexample/numeric-failed-paired.json).
- P2/R1: `evaluate`는 완전한 atomic result만 검증하고 outer `run-receipt.json` 성공을 확인하지 않음. outer FAILED2개인데 `PASS_NUMERICAL_ONLY` 재현. 배열 수치비교 자체의 오류는 아니며 completed-work 사전조건을 호출자에게 맡긴 상태.
- P2/R2: raw1원소 +0.01로 허용치0.002 위반 시 실제 `main()`이 FAILED 보고서를 쓴 뒤 정상 반환/exit0. exit만 보는 supervisor는 오인 가능. JSON 상태를 확인하는 소비자는 거부 가능.
- 검증·근거: CPU1 합성 counterexample2개만 실행.100가상UID·6×100×12 배열 사용; weights/영상/model/torch/GPU/원격0. 기존20/37 tests 재실행0. 실제100 입력 재실행0. helper·tests·producer·실제protocol·stage manifest SHA5개 이전값과 일치.
- 타당한 부분: 독립 정수center/float64 bilinear·blank normalization, producer 실제입력 hash 연결,72개 소견별 확률 gate, 보수적 rank envelope. rank bound는 밀집분포에서 느슨하며 AUC 보장 아님. host간6e-6은 triangle bound이며 직접 실측차이가 아님.
- 남은 일·다음: ① root acceptance에서 outer COMPLETE+worker receipt hash+종료코드 확인 ② input `pass:true`/평가 `PASS_NUMERICAL_ONLY` 명시 확인 ③ helper 변경 시 새 버전으로 수정·반례검증. runtime/codec/실제양host/품질 gate는 별개.
- 재사용·제한: 합성 evidence의 protocol/hash는 반례 전용이며 실제 계약에 승계 금지. root가 별도 실행한01:30:37 로컬 CPU PASS는 리뷰에서 재실행·host간PASS로 주장하지 않음. 학습·producer·원격 실행 허가 사용0, 대기작업 없음.
