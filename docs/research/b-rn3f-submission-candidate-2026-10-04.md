# B/RN 3fold 고정 후보 · 제출 검토

2026-10-04 11:26 KST 결과 확인. **actualv2 / 56809595: COMPLETE, Public 0.913**, 오류문구없음·Private공백. 기준본0.943보다−0.030으로현후보미채택/기준본유지. 사용자완료알림으로예정13:01보다일찍 CLI1회/정확ref1회확인했다. [결과·판정](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-submission-result-decision.json). 채점예약종료·같은후보재제출/사후비율·소견튜닝없음.

## 고정 가설과 정확한 후보

`taeyangg4/knee-b-rn3f-release-v1` **actual version2**, output `submission.csv`. 기존 B8/RNclip20 F0/F1/F2의6weights를 쓴다. 각 fold에서 전체 testUID의 B/RN average-tie percentile rank를90:10으로 섞고3fold 평균한다. 최종재rank/anchor/EXP084·085·086weights 대체는 없다. 제출명은 `B_RN3F fixed 3fold within-slot-valid rank90:10 v2`다.

이는 공개0.943 스택 또는 D01(anchor+옛R2)과 다른 후보다. 사전 고정한 입력·자체학습계열+RN 보완의 전이를1회 탐색한다. 같은계열의 비율 변형을 LB로 반복 탐색하지 않는다.

## 개선 근거와 한계

EXP080 F2 RN 보완+.001243, EXP084 새group fixedclip RN 보완+.00099742(조건부CI[+.000101,+.001865])는 작지만 보존할 내부 근거다. **이 수치는 정확 제출6weights의 독립 임상성적도,0.943 대비 개선량도 아니다.** EXP080 B−A 골절−.022337 반복·EXP084 RN보완 골절−.003111(P29)의 반대근거도 유지한다. 새group은weak·연구자노출·환자/Dread계보불확실·한seed이며 기존가중치를독립검증으로소급하지않는다.

공개LB는 새후보의 제한된 전이를 확인한다. 상승해도private/임상개선증명이 아니고, 하락하면 이번고정후보의 결과이며 Silver/입력/표현전략전체기각은 아니다. 점수후90:10이나소견routing을튜닝하지않는다.

## 실행 근거

- 원DICOM100의T4/로컬수치·종료검증 PASS를 재사용했다. 72확률조합max차.0015984<=.002, 최종rank max차.006. 실제6모델/원input정책검증이며새encoder검증아님.
- 최종설정v2 공개3 실행95.117249초/postack<600, input/parent/worker/resource/codec·10artifact SHA·notebooksource exact. 결과raw6/CSV는기존검증v1과exact. 표본RSS2.717GiB/sharedcgroup5.099/30GiB·OOM0/cleanup정상. .70/.90/.91실decode·.57availability만.
- 실제SDK platform32400 요청, 내부full32100/public600. 1300건+30%시간가산+300초는약4.202시간 **시나리오**이며hidden성공보장아님. 샘플메모리와실제hidden분포차이한계가남는다.
- manifest `05238751bc969f49e2a3d794a97c0fdb48081409aa1871c4d98a4e3e631d6750`; notebook `d9a617ea1b6cb88f1fdb2bdd8208163aa250e2b868f438beb27d04b18e2986fc`; source `4bca01aa59eac625fd8b85707b63e44f845873c0b61aa67425dad446e09356ae`.

05:43 CLI:당일1사용/4잔여(lifetime9),09시리셋. 05:41무료GPU27.63h. 새제출1회는잔여1slot을사용하고유료/공개없음. 제출전exactversion/새제출유무/현재한도재확인. [root실결과](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-final-config-result-review.json),[실행계획](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1/full-config-execution-plan.json).

[독립 실제v2검토](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1/full-config-result-independent-review.json) PASS:36값원계산exact·정상종료/codec·실제버전binding확인. 공개3PASS와hidden/성능미확정을구분한다.

## 실제 공개 결과의 해석

0.913은 이번 고정6weights/90:10/3fold후보의 실제 공개 성적이다. 기준0.943 스택과는 가중치·학습계열이 달라, −0.030을 RN혼합 자체의 인과효과로 볼 수 없다. 내부 B 대비 RN의 작은 효과는 보존하되 공개기준본 개선 증거로 확대하지 않는다. 채점성공은 정확제출 버전의 hidden실행완료 근거이며 실제hidden건수·시간·RAM로그까지 확인한 것은 아니다. Private/임상효과는 미확정이다.

## 검증 비교 정정 · 11:38

사용자 지적 후 원manifest의 `anchor_included=false`와 저장평가를대조했다. 내부EXP080은B.885460→혼합.886703(+.001243), EXP084는새B.89339653→혼합.89439394(+.000997)이며둘다0.943와다른비교다. EXP084 weights는이번제출에없다. 따라서는결론은미입증이다. 정확B-only3fold의같은공개성적이없다. 정확제출의독립성능및anchor대비경쟁력을연결하지못한검증/판단공백을인정한다. 완료parity/소스/계산검증을성능검증으로전용하지않는다. [대조근거·후속규칙](../handoffs/evidence/2026-10/2026-10-04-b-rn3f-release-v1/root-validation-comparator-reconciliation.json).
