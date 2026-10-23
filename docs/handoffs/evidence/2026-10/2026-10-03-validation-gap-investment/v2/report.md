# 2026-10-03 20:32 KST · 고정 DINO 교차 fold 후보 검색

- 결과: 허용된 4,343건 전체에 최근접 기록 생성. 정확 특징 중복 1그룹(2건, fold1↔0), 상호 최근접 692쌍, 사전 지정 상위20쌍 동결.
- 시간·자원: 분석 0.4862초, 프로세스 도구 관측 0.9068초, peak RSS 181,977,088 bytes(173.55MiB), native BLAS 6 threads, GPU 미사용. 구현·검사·기록은 약12분(20:20~20:32); 초기 분석 예상1~5분보다 짧았다.
- 근거: [등록본](registration.json), [실행 결과](run/result.json), [정확 그룹](run/exact-groups.json), [top20](run/top20-reciprocal-pairs.json), [최근접 전수 기록](run/nearest-neighbor-ranks.csv), [검사/해시](verification.json). 등록 SHA256 `649bedd3c09e7331953cebe8b7f608d2a2514e31b39a617c72f04690dc1f6f90`.
- 정확 특징 그룹 UID는 `<study-uid>`과 `<study-uid>`. 두 번째 쌍 cosine0.982035, top20 마지막0.976501. 동일 특징 쌍의1.000000119는 float32 내적 반올림 오차이며 원본 값을 보존했다.
- 검증: 합성 unittest23개 통과. 출력4343행의 UID 유일성, 모든 최근접의 교차fold/동일 슬롯패턴, 상호성, top20 유일성/순서 확인. actual cohort 실행은1회; 재시도 없음. 원본/캐시 픽셀·label/prediction/Gold값·Report텍스트 접근 없음. 입력 open allowlist 적용, Python 라이브러리 코드 읽기만 추가 허용.

## 해석과 다음 판단

정확 float16 특징 일치는 동일 환자나 동일 원본 픽셀의 증명이 아니다. 과거 캐시/UID 매핑 오류나 추출 산출물 재사용도 가능하므로, 첫2건은 허가된 별도 범위에서 UID→캐시 연결과 제한된 픽셀/헤더를 대조할 우선순위가 높다. 이후 고정 top20을 blinded 검토할 수 있다. 이번 실행에서 사례 제거·fold 변경·학습·제출은 하지 않았다.

높은 cosine은 정상 해부학·scanner·획득법 유사성일 수 있다. 같은 슬롯패턴만 비교했으므로 다른 획득 구성, crop/mirror, 종적 변화는 놓친다. 후보가 적다는 이유로 환자 독립성을 주장할 수 없다. historical6-slot/224/longest-series 특징은 현재 Raptor5-slot/336/first-row 입력과 다르며, checkpoint/생성코드/runtime의 과거 결합은 불완전하다. Teacher 노출·검증오염·publicLB 과적합도 별도 문제다.

## 등록 설명의 산술 정정(재실행 없음)

등록본은 representation을 `float32 mean`으로 요약했지만, 동결 source의 실제 NumPy2 연산은 float32 valid 합 → int64 count로 나누며 **float64로 승격** → slot L2도 float64 → concatenate 때 float32 → 최종 L2/내적 float32다. 실행 후 합성 dtype 확인으로 발견한 설명상 차이다. 등록본/source/결과는 수정하거나 재실행하지 않았다. 비교 대상·mean/L2 수식·후보 선택은 사전 등록된 동결 코드와 동일하지만, 엄격한 all-float32 재현으로 부르면 안 된다. 재현 시 등록 source hash와 NumPy2.0.2를 함께 사용해야 한다.

장치 간 점수 일치 검증은 이 CPU 검색 범위에 포함되지 않는다. T4↔4070 Ti Super 비교 결과로 해석할 수 없다.
