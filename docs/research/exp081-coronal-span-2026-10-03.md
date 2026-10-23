# EXP081 시리즈 선택 대조 — 고정 가중치 결과

2026-10-03 20:12 KST 갱신. **기존 B8 가중치에 최대 물리범위 시리즈를 바로 교체하는 안은 채택하지 않는다.** 전체 AUC 차이는 −0.000684, 조건부95%CI [−0.001967,+0.000278]로 불확실한 음의 추정이다. 넓은 커버리지·추가 시리즈·재학습 가설 전체의 반증은 아니다.

동일868건·동일 fixed-final8 checkpoint·strictFP16·batch8·K24로 두 군을 새로 GPU 추론했다. 학습 당시 OOF batch4를 대조로 재사용하지 않았다. 34건 입력 변경,834건 입력 및 예측 byte-exact(maxdiff0),양군 유한값·모델/소스/런타임 계약 통과 후 라벨을 열었다. 골절 손실을 해결한 결과로 부르지 않는다.

GPU는4070 Ti SUPER이며 기준군65.908초·대안군 약65초,평가 포함148.874초였다.19:59:57완료. CPU는시리즈헤더·기하·패치검사에사용했고129series/3305headers4.33초·전체준비21.68초,63 no-op exact였다. 구현/검토 시간은 추론 시간과 구분한다.

기준 macro0.885462 → 대안0.884778. PFOA +0.000693, 다른11 −0.000810, Fracture +0.000959이다. 변경34건은 골절 P1/N33, Baker P0이므로 이 subset의12소견macro는 계산하지 않았다. 관측된 all-nonzero mask625건 Δ−0.001208, some-zero243건 Δ0이며 mask는 임상 정상/촬영 결측 라벨이 아니다.

| 소견 | P/N | ΔAUC |
|---|---|---|
| ACL | 113/755 | -0.000234 |
| MCL | 37/831 | -0.003968 |
| Medial Meniscus | 319/549 | +0.000936 |
| Lateral Meniscus | 125/743 | -0.003273 |
| Medial OA | 201/667 | -0.000231 |
| Lateral OA | 134/734 | -0.000590 |
| PF OA | 305/563 | +0.000693 |
| Effusion | 200/668 | -0.000037 |
| Synovitis | 193/675 | +0.000008 |
| Baker's | 99/769 | +0.000053 |
| Contusion | 133/735 | -0.002527 |
| Fracture | 26/842 | +0.000959 |

변경34건의 다른4슬롯 pixel bytes는 유지했지만1건의mask/K24중심이바뀌므로 타슬롯의실제평가창까지고정됐다는주장은하지않는다. 소견별 양의변화는보존하며 결과를보고target routing이나비율을선택하지않는다.

F2는EXP080결과를본뒤가설을정한노출개발fold이며 weak보고서라벨/한가중치/환자독립미확인이다. 840report-group bootstrap500회의CI는고정예측조건부이며seed·라벨·반복선택·private분포·다중비교불확실성을포함하지않는다. 로컬대조는T4/4070 동등성증거가아니다. 0.943또는D01 anchor+기존R2의개선량으로전용하지않는다.

Biohub 점검: 작은PFOA/골절이득과MCL/meniscus손실을함께보존한다. CI0포함을동등으로해석하지않고,현재개입미채택과커버리지전략전체기각을분리한다. 추가학습·fold/seed·자동제출없음. 재검토는명확한정보누락/학습분포불일치기전과별도비용이확보될때한다.

[결과](../../artifacts/predictions/EXP081_CORONAL_SPAN_FIXED_F2/report.json) · [고정계약](../../artifacts/experiments/EXP081_CORONAL_SPAN_GPU_PROTOCOL/launch-contract.json) · [판정](../../artifacts/experiments/EXP081_CORONAL_SPAN_GPU_PROTOCOL/decision.json) · [원CPU준비](../../artifacts/logs/EXP081_CORONAL_SPAN_FIXED_F2/HANDOFF.md)

독립 결과 검토(anchor_inputs)에서도 source/input40개 hash, 계약→예측→gate→report→SUCCESS 연결과 라벨 개봉 순서를 확인했으며 결과를 무효화할 실행·평가 오류를 찾지 못했다. 같은12장에더넓은범위를담으면서세부정보·촬영조건·기존가중치와의분포차가함께포함된개입임을보존한다.
