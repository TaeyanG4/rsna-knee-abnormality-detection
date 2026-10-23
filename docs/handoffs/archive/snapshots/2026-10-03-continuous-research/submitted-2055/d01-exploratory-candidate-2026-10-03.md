# D01 고정 가설 — 탐색 제출 준비

2026-10-03 20:25 KST. CPU45검사·독립검토 후 비공개 `taeyangg4/knee-d01-candidate-v3` 실제version1을20:24실행했고RUNNING을확인했다. 결과/출력은아직미검증·대회제출0이다. 통합시험PASS와최종후보실행PASS를구분한다.

가설은 기존0.943 anchor의 순위90%에 기존R2 다섯모델의 평균확률 순위10%를 더하면 오류를 일부 보완할 수 있는지 확인하는 것이다. 전체 평가 cohort에서 각 소견을 한 번 순위화하며 동률은 평균순위, 마지막 재순위화는 없다. 소견별 routing·비율 탐색은 하지 않는다. EXP079/080의 새B+RN 보완과 다른 모델·가설이므로 그 개선량을 이 후보의 예상점수로 쓰지 않는다. 기존R2의teacher노출·라벨/epoch선택 문제는 그대로 남아 있어 독립적인 정확도 개선은 미입증이다.

기준본 원래30셀을 유지하며 실제test root에서 anchor를 실행하고, 자식 프로세스 종료를 확인한 뒤 R2를 실행한다. 전체 UID·12소견·다섯모델·유한확률·원래가중치·mean-before-rank를 확인한 혼합만 최종submission.csv로 내보낸다. 제어 가능한 실패에서는 원래anchor가 잠시 생성한 submission.csv를 제거한다. VM강제종료/SIGKILL 때Python정리를 보장하는 것은 아니다.

통합시험 `taeyangg4/knee-d01-integrated133` version2는 정확버전 결과와하위영수증을회수해 검증했다.133건생산경로2399.726초,원100입력/정규화와5모델확률exact·추가33volume/maskexact·DINOmemmap실제경로·후행public3수치gate통과. 추가33의별도GPU확률기준은없다. 원D01v1입력hashFAIL과통합version1경로NO_GO는보존한다.

| 실행 판단 | 실제 근거와 한계 |
|---|---|
| 시간 | 보수1300건 추정30938.468초=8.594시간.9시간에서24.36분여유로최소5분기준통과,30분/30%목표미달 |
| 메모리 | 133건 kernel peak29.541/30GiB,여유469.7MiB·OOM0. full1300 및Rad669/892캐시전환은미측정 |
| 자산 | 기존시험과같은기록된refs/docker/model. 원래코드의fingerprint/checkpoint확인범위이외 모든서버resolvedrevision을입증하지않음 |
| 정확도 | 실행가능성근거이며private/public개선보장아님. about1300도정확한최대건수보장아님 |

다음은최종후보그대로공개3건을비공개2T4에서한번실행하는확인이다. 추가133/200벤치마크는하지않는다. 실제N≤3인경우내부30분상한을두고,플랫폼상한은9시간/전체N>3공동deadline8시간55분을유지하는운영분리를검토했다. 시험예상10~20분은실측전계획값이며2T4약1GPU시간의내부시험예산·유료0·제출slot0이다. 최종adapter와stage hash/명령/실제결과는아래상태자료가우선한다.

실제노트북버전·COMPLETE영수증·3UID/12소견CSV·출력hash를확인한뒤탐색제출1회의승인을요청한다. 제출전live잔여횟수/pending을확인한다. 당일5회를채우는것이목표는아니며,고정가설의전이를한번확인한다. 결과상승/동점/하락은모두보존하고public점수를보고비율을바꾸지않는다. 상승도최종모델채택증명이아니고,하락도라벨/학습형모델전체기각증명이아니다.

Biohub점검:실행검증을성능향상으로표시하지않고추가benchmark루프를닫는다. 작은효과·실패유형·검증독립성과기존기준본보존을함께본다. 20:40에제출가능여부를판정하며시험미완료라면HOLD로기록한다. 이미시작한30분제한실행의완료만예약확인하고추가시도·벤치마크는붙이지않는다.

[통합 결과·후속 상태](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-serving-status.json) · [하위 영수증 검토](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-integrated-dependent-receipts-review.json) · [CPU 후보 검토](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-candidate-v3-cpu-preparation.json)

현재stage는`.local/d01-candidate-v3/20261003-v2`,notebook SHA `75ae6a56d5e7bb0993e6d026648169240b1f41f918ebb1b261d92f0e08ed9e2e`다. `kernels push --timeout 32400`으로저장했으며내부N≤3상한1800초를분리했다. 원v1stage보존·모델/입력/혼합/기존gate불변. ETA20:34~20:44,20:35첫후속확인예약이다.
