# EXP084 새 그룹 loss 대조 결과

2026-10-04 02:28 KST. 세 fit이 02:11:51 완료됐고, 사전등록 평가를 02:22:59에 한 번 실행했다. **pos_weight=1 전환은 미채택하고 clip을 유지한다.** 두 방식의 동등성이나 loss 전략 전체의 무효를 입증한 것은 아니다.

| 사전 고정 비교 | macro12 차이 | 조건부 95% 구간 |
|---|---:|---|
| 주 비교: B/RN ones − B/RN clip | −0.00009579 | [−0.00100886, +0.00075570] |
| RN ones − RN clip | −0.00390390 | [−0.01164946, +0.00399352] |
| B/RN clip − B | +0.00099742 | [+0.00010133, +0.00186504] |
| B/RN ones − B | +0.00090163 | [−0.00014915, +0.00183245] |

B의 macro는 .89339653, clip 혼합은 .89439394다. RN의 작은 보완 효과는 보존한다. 그러나 clip 혼합의 골절은 B보다 −.003111(P29/N859, 구간[−.008792,+.001175])이며, input A가 없어 이전 B−A 골절 저하가 회복됐다고 말할 수 없다. PFOA는 +.004340(P298/N590)이다. 소견별 이득을 보고 혼합 비율이나 routing을 바꾸지 않는다.

EXP082의 이전 ones 혼합 +.000534(CI에 0 포함)도 반대 근거로 보존한다. 두 실험은 분할·학습 계보가 달라 수치를 단순 평균하지 않는다. 추가 seed/fold·loss 주변값 탐색은 이번 결과로 자동 확대하지 않는다.

T3455/V888, 알려진 component3323/873 사이 교차0. RN 두 fit은 각8620 updates·skip0, B는3448·skip0였다. 소요는726/724/2610초로 총 약67.7분이다. 세 모델을 모두 새 T에서 학습하고 fixed-final 뒤에만 평가했다. Gold 접근은 차단했다. 동일 component bootstrap500회 모두12소견 유효였다.

그룹 결과는 탐색적이다. ones−clip은1.5T535건 −.000455,3T315건 +.000091, zero-mask234건 +.000224/all-nonzero654건 −.000238로 방향이 섞인다. 작은 장비군은 표본 부족/단일 class를 표시했고 CI나 다중비교 보정을 주장하지 않는다. mask는 임상 정상 여부가 아니다.

새 image-model 계보의 검증 제외는 개선했으나 weak 라벨·기존 연구자 노출·외부 Dread 계보·환자/근접 중복 미확정은 남는다. CI는 고정된 모델·라벨·한 seed에 조건부이며 선택/학습 변동/임상 정답/Private 분포의 불확실성을 포함하지 않는다. 정확 D01 또는 공개0.943 기준본의 개선 검증이 아니다.

## 다음 투자와 Biohub 점검

M02/03/04/05/10을 적용했다. 작은 RN 이득과 골절의 반대 증거를 함께 보존하고, loss 한 설정의 보류를 감독 전략 전체 기각으로 확대하지 않는다. [비교한 3안](../handoffs/evidence/2026-10/2026-10-04-post084-priority/brief.md) 중 **EXP085 상대 위치정보 없음/있음 대조**를 다음 CPU 준비 대상으로 선택했다. 동일한 새 residual attention·파라미터 수·픽셀·마스크·학습량을 사용하고, T-only B8에서 전체 backbone을 두 arm 모두8epoch 추가 적응한다. weights-only 복원과 optimizer 재시작을 양쪽에 동일하게 적용한다.

예상 GPU80–92분은 attention overhead 미측정 계획이다. pilot 포함7200초, 각 full3600초 상한·자동재시도0. CPU 계약과 pilot 이후 root가 실제 실행을 결정한다. 두 arm 모두 기존 B보다 낮을 가능성도 평가하며, 같은 개발V에 대한 후속 선택이라는 한계를 유지한다. 독립 영상 근거가 없는 Silver 자동교정은 계속 보류한다.

## 근거

- [고정 평가 보고서](../handoffs/evidence/2026-10/2026-10-04-exp084-report-tool/evaluation-v1/report.json), [12소견 P/N·Δ·macro 기여](../handoffs/evidence/2026-10/2026-10-04-exp084-report-tool/evaluation-v1/targets.csv), [등록](../handoffs/evidence/2026-10/2026-10-04-exp084-report-tool/registration.json).
- [재계산 없는 독립 해석](../handoffs/evidence/2026-10/2026-10-04-exp084-result-and-serving/independent-result-review.md), [root 투자 결정](../handoffs/evidence/2026-10/2026-10-04-exp084-result-and-serving/post-result-decision.json).
- 보고서 SHA: `0051229e543564b8cebaec27277c300954c9f1e0ba3b376d933460c3686f1f10`.
