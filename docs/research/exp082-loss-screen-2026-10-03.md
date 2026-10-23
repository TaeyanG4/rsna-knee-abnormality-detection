# EXP082 RN loss 대조 결과

2026-10-03 21:40 KST 기록. 21:27:52 학습 완료,21:29 고정 평가1회 완료. **작은 양의 개발 후보로 보존하며 즉시 채택·확대·제출하지 않는다.**

|고정 비교|macro12 AUC 차이|조건부95% CI|
|---|---:|---:|
|90:10(B,RNones) − 90:10(B,RNclip), 주 비교|+0.000534467|[−0.000268009,+0.001293080]|
|RNones − RNclip, 보조 비교|+0.006828056|[−0.000695851,+0.014910356]|

CI0포함은 효과없음/동등 증명이 아니라 현재 표본에서 불확실하다는 뜻이다. 혼합 AUC는0.886703057→0.887237524이며, 이것은 weak 개발 라벨 점수다. D01(anchor90%+옛R2 10%)와 다른 모델 조합이므로0.943 개선량이나 private 효과로 전용하지 않는다.

학습은691초/20고정epoch/8680updates/skip0. 동일 train3475/eval868,ImageNet 초기값,입력B,224/k12/eval24,seed42,batch8/eval4,optimizer/scheduler/strictFP16에서 BCE pos_weight만12ones로 바꿨다. no-Gold entrypoint의 초기값·UID·runtime gate를 통과했고 원4학습 소스는 보존했다. CPU준비10검사·독립16AST·root24binding,평가기8합성검사 후 실행했다. 평가를 반복하지 않았다.

## 소견별 주 비교

|소견|P|N|혼합 차이|
|---|---:|---:|---:|
|ACL|113|755|+0.001898845|
|MCL|37|831|+0.003366182|
|Medial Meniscus|319|549|+0.001187682|
|Lateral Meniscus|125|743|+0.002002692|
|Medial OA|201|667|-0.000346842|
|Lateral OA|134|734|-0.000528692|
|PF OA|305|563|-0.000017471|
|Effusion|200|668|-0.000729790|
|Synovitis|193|675|-0.001078488|
|Baker's|99|769|-0.000978576|
|Contusion|133|735|+0.001478185|
|Fracture|26|842|+0.000159876|

골절 혼합+0.000160(P26/N842,CI[−0.003100,+0.003620])은 기존B의 골절하락을 해결한 증거가 아니다. RN단독 골절−0.001370·Synovitis−0.011790·Baker’s−0.011887도 보존한다. MCL은P37이며 소견별 탐색 결과에 다중비교 보정은 없다.

결측mask군243건의 주 차이+0.001344,모두 nonzero625건+0.000154. Philips1.5T−0.000499,Siemens3T−0.000311,3T전체−0.000369 등 손실이 남는다. unknown field45건+0.003761은 작은 군이다. 그룹CI는 계산하지 않았고 단일class target/전체macro는 null로 보존했다. mask는 영상결측이며 임상음성 라벨이 아니다.

868건/840report-group의500회paired bootstrap은 전체fold에서 고정한 rank/예측을 재표집했다. subgroup별 재rank는 없다. CI는 한seed·현재weak라벨·고정예측에 조건부이며 연구자 노출,라벨오류,미확인환자중복,seed변동,private분포를 포괄하지 않는다.

그룹계약은4343건/4196components,확인중복쌍의 oldfold0↔1연결1개를 포함한다. 새split은 아직 없다. 전체cache/원본7표본 일치는 확인했지만 동일환자/전체study 동일성은 미확정이다. 알려진쌍은F2에서 둘다train이며F2 독립성 증명은 아니다.

Biohub 점검: 작은 이득을 폐기하지 않고 손실·불확실성을 함께 보존했다. 실행PASS를 성능PASS로 바꾸지 않았다. 다음은 새그룹에서 양RN 재학습(약26~36GPU분;혼합까지 비교하면 공통B포함66~82분)과 slot-aware head 대조의 정보가치를 비교한다. 자동fold/seed/비율탐색은 없다.

[원시 보고서](../../artifacts/predictions/EXP082_R34_LOSS_F2_REPORT/report.json) · [평가기 검증](../handoffs/evidence/2026-10/2026-10-03-exp082-report-tool/verification.json) · [판정과SHA](../handoffs/evidence/2026-10/2026-10-03-continuous-research/exp082-completed-decision.json) · [그룹계약](../../artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/HANDOFF.md).
