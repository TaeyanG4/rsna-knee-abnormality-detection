# 2026-10-04 06:52 KST · 추가 시리즈 독립 계약 검토

- 상태: 완료·`/root/anchor_inputs`. **제한된 CPU pilot 진행을 조건부 권고**하며 실행 허가는 아니다. [상세 JSON](coverage-independent-review.json). source/pixel/label/OOF/GPU/census 재실행·중앙문서 변경0.
- B8 판정: producer의 **기존 center 목록 유지→slot2 출현순번1·3·5만 치환**안이면, 원초기화/G1/8epoch/loss/optimizer/RNG 및 평가batch4·K24가 같고 no-op이 정확할 때 **완료 EXP084 B8 예측을 재사용할 수 있다**. 후보 입력이 의도대로 다르다는 이유만으로 대조 재학습은 필요하지 않다.
- 대조 재학습 조건: 공통sampler·전처리·구조·증강·학습량을 함께 바꾸거나 B8에서 추가 적응하면, 같은 새조건의 first-only 대조가 필요하다. 평가batch/정밀도만 바뀌면 원가중치의 동일조건 재추론/동등성 검증 문제이며 자동 재학습 사유는 아니다.
- G1: T3455/V888·component3323/873, 기존 known-group 교차0 기록. 모든 추가series는 study의 split·제외를 상속하고 study당 label/loss1개를 유지한다. 새T↔V중복 발견시 기존B/ancestor 노출까지 재검토한다. report그룹·SeriesUID·PatientID는 환자 독립성 증명이 아니다.
- 고정K: train12/eval24는 triplet 수다. 다른slot/미치환window·mask·정규화·intensity RNG가 그대로여야 한다. q=(c−33)/9는 cache 상대순서일 뿐 해부학 정렬이 아니다. 각triplet은 한series 내부; m0/1·대안없음은 원본유지. 최종proposal의round-half-to-even·빈목록fallback·예상외decode실패중단·lexical첫후보 이후search금지를 확인했다.
- 실제로 모르는 것: metadata상 다른series≠다른pixel≠유용한정보.1936/4407은 원정의출처 미확인이고 v2의broader1985와 혼용금지. 현재G1 엄격가용률·정보이득은 미계산. EXP081 음성은 추가series 가설 전체 기각이 아니다. cross-series FoR가 다르다고 독립triplet을 모두 배제하거나 좌표정렬을 가정하지 않는다.
- producer가 쓸 경로: `EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json`의train과 `EXP081_CORONAL_SPAN_FIXED_F2/metadata-map.json`의plans를 UID로 연결한다. 정확선택된current/alternate 둘의 `geometry.json` records가 있는T표본만 고른다. EXP081의oldF2 cohort나max-span selected를새선택으로대체하지 않는다.4–8개 교집합수/UID는 producer가 한정확인하며 여기서중복census하지 않았다.
- 남은 gate: exactT UID·source·header지도와 별도출력경로 동결, root의별도decode승인. 최종proposal의8pair/192plane/10min/1GiBpixel/4GiBRSS cap과SOP·render동일성gate는 타당하며 구현검증은미완료. 전수cache/학습등록/GPU는미허가. M02/M03/M05/M07/M10·노출weakV/ancestor/private전이한계유지.
