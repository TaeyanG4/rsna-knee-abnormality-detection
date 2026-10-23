# EXP083 · 시리즈별 attention bias 고정 대조 결과

2026-10-03 22:44 KST. **이 설정은 미채택한다.** 주 비교의 macro12 차이는 −0.000024565, 조건부95%CI [−0.000215275,+0.000170508]이다. CI에0이 포함된 것을 동등성이나 모든 시리즈 정보 활용의 무효로 해석하지 않는다.

재학습 control은 같은 캐시의 기존 B8 head보다 −0.011549423(CI[−0.016382214,−0.006989687]), treatment는 −0.011573988(CI[−0.016435223,−0.007007776])였다. 현재 fixed-feature 재학습 레시피 자체의 손실이 더 크다. 원인(학습률·학습량·고정K24변환·encoder/head결합)은 이번 대조로 분리하지 않았다. 과거 저장 B OOF에 대한 재현 parity를 통과했다고 주장하지 않는다.

|대조|macro12 AUC|같은 캐시 frozen B 대비|
|---|---:|---:|
|frozen_b8|0.885459959|+0.000000000|
|control|0.873910535|-0.011549423|
|treatment|0.873885971|-0.011573988|

## 12소견: treatment − control

|소견|P|N|AUC 차이|조건부95%CI|
|---|---:|---:|---:|---|
|ACL|113|755|-0.000058606|[-0.000751278, +0.000431397]|
|MCL|37|831|-0.000130094|[-0.001894958, +0.001428137]|
|Medial Meniscus|319|549|-0.000222690|[-0.000529515, +0.000119719]|
|Lateral Meniscus|125|743|+0.000430686|[-0.000305242, +0.001172152]|
|Medial OA|201|667|+0.000223769|[-0.000251433, +0.000687658]|
|Lateral OA|134|734|-0.000162674|[-0.000931595, +0.000662359]|
|PF OA|305|563|+0.000151414|[-0.000254738, +0.000599136]|
|Effusion|200|668|-0.000112275|[-0.000447708, +0.000204478]|
|Synovitis|193|675|-0.000069085|[-0.000356113, +0.000214968]|
|Baker's|99|769|-0.000065676|[-0.000430363, +0.000219057]|
|Contusion|133|735|-0.000051148|[-0.000453790, +0.000350715]|
|Fracture|26|842|-0.000228394|[-0.000766098, +0.000092178]|

## 실행과 검증 범위

원V1은 native FP16 가정 오류로6.984초에실패했고 그대로보존했다. 별도V2는 실제 nativeFP32를변환없이저장한다. pilot16건9.687초PASS 이후full215.422초(추출198.234초),두head각1100update/skip0,encoder불변·초기logit/train+eval·native/cache exact·paired dropout RNG exact를통과했다. 전체4343건 특징캐시는426,934,272bytes+NPYheader다. 실패·pilot·full합계232.093초이며45분상한안이다. 치료arm의60개bias모두실제로변했고control은zero를유지했다.

사전 protocol/코드와 최종 evaluator registration을 점수 개봉 전에 고정하고 단1회 CPU평가했다. 동일868UID/840component·고정원Silver이진화·12target·동일500draw를사용했고500회모두유효했다. source/hash·receipt·runtime·label·UID·성공종료·예산·skip 검사를 모두 거쳤다. 평가소스 독립검토에서도 치명오류를발견하지못했다.

장비/결측24그룹은 CI 없는 탐색적 점추정이다. 일부 작은그룹은12소견양/음성이없어macro를계산하지않았다. all_nonzero625건의 주차이−0.000104369, some_zero_mask243건+0.000019470이며 촬영/임상정상 여부나 그룹별 개선을입증하지않는다.

노출weakF2·한seed·선택된라벨/모델조건부CI다. 알려지지않은환자연결·보고서truth오류·과거Gold/연구자노출·모델선택·seed변동은 이CI에포함되지않는다. 정확D01/0.943/LB/private 개선근거가아니다.

## 판정과 다음 조치

- 현재fixed20 head/bias조합은미채택한다. 이F2점수를보고학습률·epoch·routing을반복탐색하지않는다. 시리즈정보가설전체는닫지않고,train내별도선택과공정한refitcontrol에서재검토할근거가생길때다룬다.
- Biohub점검:실행PASS와성능을분리했고음수와작은변화·CI를모두보존했다. 더큰공통재학습손실을숨기지않았으며특정실패를모든표현가설로확대하지않는다.
- 사용자선택에따라Silver오류·누락감사를우선한다. 기존3475train만의raw mentioned/negated/confidence 필드의의미/coverage를먼저확인한다. 가능성이큰소견을근거로고르며자동라벨교체·lossmask·새학습은아직없다.

[평가 원본](../handoffs/evidence/2026-10/2026-10-03-exp083-report-tool/evaluation-v1/report.json) · [root 판정/체크포인트](../handoffs/evidence/2026-10/2026-10-03-exp083-report-tool/root-result-decision.json) · [평가기 검토](../handoffs/evidence/2026-10/2026-10-03-exp083-report-tool/independent-review.json) · [V2 실행 계약](../../artifacts/experiments/EXP083_SLOT_BIAS_F2_PROTOCOL_V2/HANDOFF.md).
