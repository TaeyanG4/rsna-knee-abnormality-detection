# 2026-10-03 23:15 KST · Dread 감독 출처 확인

- 상태: 완료. 기록자: Codex / precision_code_review. 지정된 TXT 2개와 dataset metadata만 정적 확인.
- 결과: calibrated probability·soft label 우위는 제작자 주장. 생성 prompt/LLM 식별·보정 절차/검증셋·별도 판독검증·통제 비교 근거는 지정 자료에서 미공개/미확인. 틀렸다고 단정하지 않음. [보고서](report.md).
- 구현: 외부 12 soft target→class prevalence pos_weight BCE. per-target unknown/confidence mask 없음; slice mask와 별개. Qwen/Flight 규칙을 Dread에 귀속할 근거 없음.
- 검증: [정확 snapshot/hash](manifest.json), [line 인용](source-excerpts.txt). 모델 실행·CSV/Report/Gold/예측/outputs/packet 조회·중앙문서 수정 없음. JSON·hash·링크 확인.
- 다음: 원생성 prompt/model/version·calibration·soft/hard 비교 자료는 향후 출처 확인 항목. 현재 로컬 학습 실행과 라벨 개별 품질은 이번 범위 밖.
