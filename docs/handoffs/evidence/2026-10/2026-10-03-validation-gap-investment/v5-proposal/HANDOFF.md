# 2026-10-03 20:54 KST · RN pos_weight 제안 저장

- 상태: 완료(설계 제안만). 기록자: Codex anchor_inputs.
- 요청·결정: EXP080 F2 RN20 기준+ones RN20 한fit을 최소 개발 screen으로 권고. F0/F1 현재입력 exact중복 반영, F2 독립holdout 주장/전수중복감사 선행조건 없음.
- 결과·파일: [제안](report.md): 고정B+RN90/10 간 macro12 차이 주비교, 작은효과 보존·alltargets양수 gate 없음. 학습/표현/감독3가설의 구현·GPU비용 비교. 최소Gold차단 adapter·UID순서/RNG/초기값/optimizer/FP16 보존 위험 기록.
- 검증·근거: [기록](evidence.json). 기존 EXP080 protocol/provenance·v3결과·동결4소스 text만 대조. train/eval sorted hash만으로 shuffle순서 보장 안 됨, Gold는시작시읽기+마지막평가 경로 모두존재. 큰checkpoint/OOF는직전hash관측 재사용, 이번재읽기없음. 학습/import/GPU/원격/점수계산 없음.
- 남은 일·다음: 원본7쌍pixel확인은별도담당; 결과를이문서가선언하지않음. root가구체계약/adapter준비범위판단. 중앙STATE/INDEX/기존파일 변경없음.
