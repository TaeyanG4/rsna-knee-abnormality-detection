# 기존 OOF의 소견·그룹 오류 기여 감사 — 2026-10-03

**현재 weak-label 오류는 MCL·PF OA·Lateral Meniscus·Lateral OA에 많이 남는다.** R1에서 이 네 소견은 `(1−AUC)/12` 합계의47.6%다. 실제 회복 가능한 이득이나 대회 정답 기준의 병목 비율은 아니다. 학습 중인 EXP078은 변경하지 않았다.

고정 원보고서 라벨·동일4,345 study·15개 `oof_best.csv`를 CPU로 재채점했다. 모든 모델은 best만 사용해 이전 ResNet SWA 합산과 구분한다. 기존 도구의 수치 계약을 재사용하되 덮어쓰기·묵시적 UID 교집합·SWA fallback 없이 검사한 [반복 도구](../../tools/oof_error_audit.py)를 추가했다. [전체 근거/54개 파일 hash](../handoffs/evidence/2026-10/2026-10-03-error-audit/audit.json) · [12소견 표/CI](../handoffs/evidence/2026-10/2026-10-03-error-audit/targets.csv) · [weak 순위오류 사례](../handoffs/evidence/2026-10/2026-10-03-error-audit/weak_rank_error_cases.csv).

## 비교 범위

R1·ResNet34는 이미 노출된 개발fold에서 epoch를 고른 weak OOF다. R2는 teacher 간접누출과 혼합라벨 epoch선택이 있어 **아래 상승을 독립 개선으로 해석하지 않는다.** GPU·AMP·학습조건도 서로 달라 구조/라벨 효과의 통제대조가 아니다. 보고서그룹4,199개는 환자 독립성의 보증이 아니다.

이진 reference는 기존 `macro_auc`와 같은 float32→`numpy.rint`다. 정확히0.5인 cell은0개다. label hash=`71be1189…afade4`, split hash=`1ff65747…8b1`. Gold58과 교집합0. 두 손상 study도 이전 모델이 학습/평가한 기존4,345 정의를 보존해 포함했다. EXP078의 공통제외 기준과 직접 동일시하지 않는다.

| 소견 | P / N | R1 AUC | ResNet34 AUC | R2 AUC* | R1 macro deficit | R2−R1* |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 565 / 3780 | .90435 | .89664 | .92113 | .00797 | +.01678 |
| MCL | 214 / 4131 | .80969 | .79492 | .83255 | .01586 | +.02287 |
| Medial Meniscus | 1657 / 2688 | .90388 | .87643 | .91516 | .00801 | +.01127 |
| Lateral Meniscus | 585 / 3760 | .82970 | .80718 | .84957 | .01419 | +.01987 |
| Medial OA | 1072 / 3273 | .87747 | .87294 | .89625 | .01021 | +.01878 |
| Lateral OA | 649 / 3696 | .84388 | .83785 | .86228 | .01301 | +.01840 |
| PF OA | 1459 / 2886 | .82199 | .83399 | .84994 | .01483 | +.02795 |
| Effusion | 1112 / 3233 | .92267 | .91785 | .92657 | .00644 | +.00390 |
| Synovitis | 1065 / 3280 | .90868 | .90448 | .91374 | .00761 | +.00506 |
| Baker's | 532 / 3813 | .94456 | .93165 | .95470 | .00462 | +.01014 |
| Contusion | 661 / 3684 | .86387 | .83901 | .87210 | .01134 | +.00823 |
| Fracture | 159 / 4186 | .90842 | .88471 | .91698 | .00763 | +.00855 |
| 12-target pooled macro | 4345 study | **.878263** | **.866471** | **.892580** | **.121737** | **+.014317** |

*R2는 누출된 개발 관측. target Δ의 macro 기여는Δ/12. `deficit`는 weak 정답과 다른 양성/음성 쌍의 순위량이다. 한study가 여러 쌍에 관여하므로 이를 오류 건수나 곧 얻을 점수로 해석하지 않는다. 사례 CSV의 기여는 각 discordant pair를 양 끝study에 절반씩 배분한 값이다. 임상 FP/FN 목록이 아니다.

## 차이와 불확실성

- pooled와 fold평균은 다르다. fold평균 R1=.878702, ResNet=.868515, R2=.893018. 전부5fold 결과를 보존했다.
- **ResNet PF OA는 전체 평균보다 눈여겨볼 차이**다: R1 대비+.012005, 조건부95% 구간[+.003651,+.020636],5fold 모두 양수(+.000012~+.022515). 다른11소견 pooled는 낮고 총macro는−.011792다. 따라서 ResNet 전면교체가 아니라 보완 신호/해부학적 표현 가설이다. 특정 소견 routing의 효과를 증명하지 않는다.
- R2 PF OA는+.027952로 가장 큰 차이, MCL+.022869다. R2 Fracture 차이의 구간은[−.001504,+.018346]이고fold2/3은 음수다. bootstrap 양수여도 R2 누출은 사라지지 않는다.
- R1 MCL의 조건부 AUC 구간[.781927,.840502]은 넓다. Fracture도 양성159개다. 작은 음수/양수와CI 포함0은 자동 기각/동등의 증거가 아니다.
- 구간은 report-group paired500회, 동일 저장예측의 **pooled target AUC** 조건부 구간이다. 다중비교·선택·label오류·seed·환경·private 변화의 불확실성을 포함하지 않는다.

## 관측 가능한 그룹

35개 그룹 항목을 기록했고40study 미만은 수치가 불안정해 크기만 표기했다. 비교 가능한 소견 수/P/N을 모두 보존했으며 일부 소견 평균을12-target 점수로 표시하지 않는다.

| 조건 | study | R1 / R2* macro | 해석 |
|---|---:|---:|---|
| GE 1.5T | 695 | .88992 / .90673 | 비교적 큰 그룹 |
| Philips 1.5T | 619 | .86384 / .87954 | R1 PF OA .77320(P221), MCL .77705(P31) |
| Philips 3T | 663 | .87532 / .89037 | MCL .71179이나 P25뿐, site 원인 단정 불가 |
| Siemens 1.5T | 1148 | .87491 / .89028 | Lateral OA .81063, PF OA .81828 |
| Siemens 3T | 781 | .86926 / .88272 | PF OA .80889, Medial OA .82258 |
| cache 4slot 존재 | 1224 | .89144 / .90408 | 5slot보다 점수가 높음: 결측 자체가 원인이라는 단순설명 불가 |
| cache 5slot 존재 | 3118 | .87308 / .88785 | cohort/label/protocol 혼재 가능 |
| 같은 plane/FS 후보 여러 개 | 1898 | .87228 / .88654 | 원래자료의 후보 다양성, 선택이 틀렸다는 증거 아님 |
| 같은 plane/FS 후보 하나 | 2447 | .88073 / .89510 | 관찰 비교, 개입효과 아님 |

모든4,345 study에 메타데이터상 Sagittal/Coronal/Axial이 있다. cache0~3slot은3study이며 그중1개는 전부0 mask다. mask상 어떤 결측이라도 있는study는1,229개(28.3%). mask는 nonzero pixel 지표이며 acquisition 보증이 아니다. laterality unknown2,169개(49.9%). scanner fingerprint134개는 scanner 구성이고 환자ID가 아니다. scanner holdout·환자 중복 검증은 수행하지 않았다.

## Gold58은 별도 노출 진단

5fold best확률 평균의 Gold macro는 R1=.898203, ResNet=.889029, R2=.901589다. 이는 OOF가 아니고 같은58건에 반복 노출된 진단이다. 모든fold/target/P/N은 JSON의 별도 `gold58_diagnostic`에 있다. weak label을 잘 예측하는 것과 임상 정답을 잘 맞히는 것이 다르다는 경고용이며, 이 수치로 모델·라벨·routing을 선택하지 않는다.

## 다음 가설과 점검 시점

1. **EXP078 종료 때 우선 네 소견과 subgroup 결과를 함께 판단한다.** 입력 변경의12target/P/N/Δ, PF OA·MCL·외측 소견, candidate중복/slot/scanner를 확인한다. 보고서group와 환자독립은 구분한다. gate미달 양의효과는 보존/불확실로 기록한다.
2. **PF OA의 표현/plane 정보를 다음 구조 후보로 구체화한다.** ResNet의 반복된PF차이는 encoder 또는axial/구획 정보 활용을 점검할 근거다. 먼저 기존 예측의 약한 순위오류 사례를 영상/보고서와 대조해 projection·소견정의·scanner 혼재를 분류한다. 바로 public target weight를 튜닝하지 않는다.
3. **MCL은 감독 정의와 희귀양성 표본부터 점검한다.** 주최 중증도 기준과 추출라벨 의미를 양성/음성·scanner 층화 표본으로 확인한다. 낮은CV를 라벨오류로 단정하지 않고 anatomy/입력 문제와 나눈다.
4. **라벨혼합은 살아있는 가설로 유지한다.** R2 상승·다른참가자 실패만으로 채택/폐기하지 않는다. EXP078 이후 입력/구현 문제가 정리되면 fixed reference·fixed epoch·outer 제외teacher의 작은대조 비용과 다른구조 후보 비용을 비교한다.
5. **확대 전 독립성 점검을 병행한다.** 보고서중복만으로 동일환자·영상근접중복을 배제할 수 없다. 기존metadata/fingerprint로 검사 가능한 범위를 명확히 하고, 실제환자/영상중복 근거가 없으면 독립성을 주장하지 않는다. 모든데이터 전수검사를 끝낼 때까지 성능연구를 멈추는 뜻은 아니다.

검증: `tests/test_oof_error_audit.py`3개 통과(가중/tie AUC=명시적 재표집, pair기여합=deficit, paired동일모델Δ=0). CPU감사는9.8초. 기존예측/가중치/학습4파일/현재queue/manifest 변경 없음. GPU·새학습·제출 없음.
