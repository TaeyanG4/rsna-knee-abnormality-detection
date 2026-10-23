# 2026-10-03 21:43 KST · 다음 연구 선택 설계

- 상태: 완료(설계만). 기록자: Codex anchor_inputs.
- 요청·결정: 새group RN2fit(A)와 고정B 두head(B)의 정보/비용 대조. 다른표현가설을분리하는B를우선권고, A는후속group확인으로보존. 원RN작은효과를기각하지않음.
- 결과·파일: [최소계약](report.md). 같은K24cache·기존Bhead동일초기값으로두head재학습, treatment에만zero-init5×12 slotattentionbias. 창선택/empty처리/clip loss고정. 실제center에서slot매핑·encoder불변·F2노출한계명시.
- 검증·근거: [출처](evidence.json). v5/v6/group계약·head/data 소스text·기존EXP082 보고서해석만. 코드import/학습/데이터실행·점수계산·새split배정없음.
- 남은 일·다음: root가bounded B구현판단. A에서기존B를새V에재사용하면노출문제가생기므로blend검증은새B포함66~82GPU분추정. 중앙문서·기존source변경없음.
