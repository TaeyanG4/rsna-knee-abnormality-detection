# EXP082 RN 가중치 대조 진행

2026-10-03 21:26 KST. D01 채점과 별개로 로컬 소규모 연구를 진행한다. EXP082_R34_ONES_F2는21:16시작했으며원본학습4파일을보존한별도no-Gold entrypoint에서학습한다. CPU10검사,독립16개AST동일성검토,root24개binding검사뒤시작했고실제초기state/UID/runtime검사를통과했다. 학습완료와성능향상은아직판정하지않았다.

학습개입은RN의BCE양성항weight12개를1로바꾸는것이다. train3475/eval868,ImageNet초기값,입력B,224/k12/eval24,fixed20,seed42,학습batch8/평가batch4,optimizer/scheduler와strictFP16은기존EXP080RN과같다. 공식Gold파일과Gold평가는제거했고UID원순서/원래제외집합을보존한다.

사전주비교는동일868건에서90%rank(B8)+10%rank(RNones)와90%rank(B8)+10%rank(기존RNclip)의macro12 AUC차이다. RN단독차이와12소견P/N/골절/MCL/PFOA/다른11/장비/mask를보조로본다. 전체fold에서rank를고정한후report-group500회paired bootstrap을적용한다. 조건부CI이며라벨/seed/반복선택/private불확실성을포함하지않는다. 작은양수·음수·0포함을모두보존하고자동추가학습이나비율탐색은없다.

전용평가기 tools/exp082_loss_report.py는합성8검사를통과했고후보학습이완료되기전에소스해시와정확명령을고정했다. SUCCESS만으로성능PASS로보지않고summary/provenance/OOF/8680attempts/skip/기존B,RN해시/Gold없음까지확인후한번평가한다. 현재새예측값/성능점수는열지않았다.

검증그룹계약은4343건을4196components로정리했다. 보고서그룹4197개에확인된fold0↔1중복edge를더한결과이며원분할은보존했다. 해당두사례의전체모델입력과원본7표본은같으나환자/전체study동일성은미확정이다. F2에서는둘다train이지만F2의weak라벨/과거노출/한seed문제는남는다. 이실험의좋은결과도정확D01이나private개선으로대신할수없다.

[학습명세](../../artifacts/experiments/EXP082_R34_LOSS_F2_PROTOCOL/README.md) · [평가기검증/명령](../handoffs/evidence/2026-10/2026-10-03-exp082-report-tool/verification.json) · [그룹계약](../../artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/HANDOFF.md) · [검증계획](../handoffs/evidence/2026-10/2026-10-03-validation-gap-investment/v6-validation-plan/memo.md).
