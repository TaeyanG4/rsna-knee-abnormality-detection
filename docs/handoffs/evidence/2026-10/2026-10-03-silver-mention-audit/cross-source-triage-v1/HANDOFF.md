# 2026-10-03 23:25 KST · train3475 Dread↔Qwen 교차집계
- 상태: 완료. 기록자: Codex 하위 에이전트. 중앙기록 root 담당.
- 요청·결정: 실제 학습 Dread와 Qwen raw12소견을 train3475에서 각 CSV1회만. PFOA20 원라벨/Report 재개봉, Gold/F2값/예측/학습 없음. 등록→집계 순서, 원본 불변.
- 결과·파일: [summary.json](summary.json), [receipt.json](receipt.json), [집계표](aggregate-table.md). 양방향 불일치·Qwen mentioned0/negated1에서 Dread양성·mentioned1&negated0에서 Dread음성과 분모, 고정구간 soft분포 저장. Dread float32→np.rint, .5 ties는12target 모두0.
- 관측: Synovitis D+872/Q+881이나 D+Q−495·D−Q+504; Q미언급2453 중 D+467. Effusion D+912/Q+1814, D+Q−14·D−Q+916. PFOA D+1154/Q+1300, D+Q−141·D−Q+287, Q미언급1388 중 D+111. 우열·진실·단일순위 결론 아님.
- 검증·근거: [registration.json](registration.json) 고정 후 [triage.py](triage.py) run0.312초. Dread4349/Qwen4407 UID검사 후3475씩만 값 접근, 나머지874/932 폐기. 양쪽 중복·누락·finite/range오류0. 합성 histogram경계·float32tie·UID정렬 검사와 전체교차합·분모·hist 합계 PASS([verification.json](verification.json)); torch 미import.
- 해석·다음: Qwen 상태는 extractor 주장, 일치도≠정확도. Qwen 문제를 Dread 오류로 전용 금지. root가 이 집계와 정의감사를 결합해 다음 저비용 가림검토 가치만 판단. 라벨교체/마스킹/추가표본은 수행 안함.
