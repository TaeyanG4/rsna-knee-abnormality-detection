# 2026-10-03 · 원격 헤더 검사 실패 원인·새 비교계약

- 상태: 진단/로컬검사 완료,원격 v2 준비. 기록자: Codex 하위 작업. 루트의 인계/INDEX 통합 대상.
- 원격 v1 첫건 실패는5slot의 `ordered_sources_sha256`만 다름. volume/mask·현재baseline/candidate 전체receipt·과거 selected raw SHA/나머지receipt 동일. 기존 FAILED·산출물 보존.
- 같은 로컬 원본에서 float64 `np.dot`/FMA체인은 과거T4 해시5개를, 비융합 좌결합 곱셈·덧셈은 현재CPU 해시5개를 모두 정확히 재현. 위치Δ최대2.84e−14mm,실제 전체정렬/spacing은 동일. codec/경로차이 아님; 원격BLAS 내부실행은 직접관측하지 않음. [상세](dicom-header-order-v2-diagnosis.json).
- 별도 v2: 현재 두 구현의 전체receipt exact 유지. 과거와는 원본geometry·실제 전체ordered path/spacing SHA 및 모든 다른receipt항목 exact를 요구. 위치float해시 차이만 관측항목으로 보존하며 proof가 각각의 원래해시와 결합돼야 함. 픽셀/순서/spacing 변경은 실패. 렌더·정렬 수학과 adapter 소스 변경 없음.
- 로컬100건의 새reference가 원래T4 해시를 모두 재현,CPU20검사 및 v2첫1건통합 통과. `workspace://.local/d01-header-io-staging/20261003-v2`에734,699bytes notebook/AST·payloadSHA검증,계정쓰기 미실행. 원격v2 PASS·속도개선·전체9h 검증으로 주장하지 않음. 원격 결과 후만 채택판정.
