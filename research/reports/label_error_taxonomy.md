# Label quality audit and error taxonomy

## 직접 확인한 데이터 품질 문제

21개 공개 CSV table을 공식 Gold58과 UID로 결합해 계산했다. 이 중 8개는 gold 58행의 모든 target이 공식 정답과 완전히 같았다. Gold를 학습 target에 덮어쓰는 것은 정상적인 학습용 Dataset 설계일 수 있지만, 그런 행을 teacher 추출 정확도로 채점하면 AUC 1.0이 나오므로 teacher 비교에서 제외했다.

Pilkwang LLM v2는 전체 4,406행이고 gold UID 한 개가 없어 57개 평가만 가능했다. 전체 58개 점수와 공통 57개 점수는 따로 보존한다. Lixin의 두 파일은 target matrix SHA가 같아 독립 teacher 둘로 세지 않는다.

Qwen v2의 `__confidence`는 P(present) 의미이며 predicted-class confidence로 뒤집어 해석하지 않았다. Barun은 원래 gold target 열이 아니라 `pseudo_` 열을 평가했지만, 그 열도 Gold58 전체가 정답과 같아 제외했다. 확장자 없는 공개 label 파일 하나는 HTML/CAPTCHA여서 label로 사용하지 않았다.

## Teacher 성능과 image-model 성능의 구분

현재 비교 가능한 최고 공개 teacher는 Flight hybrid, Gold58 macro AUC 0.899130946이다. Steven v4를 11개 target에 사용하고 Fracture를 Pilkwang 계열로 바꾼 derived blend다. 새로 독립 생성된 teacher 또는 untouched-gold 평가라고 부르지 않는다.

Report-to-gold AUC는 MRI student의 성능 상한이 아니다. Student가 report noise를 평균화하거나 MRI에서 더 직접적인 신호를 학습할 수 있으므로 teacher AUC가 낮다는 이유만으로 image AUC의 상한을 선언하지 않는다.

## 실제 생성한 train-only review queue

`label_error_review_queue_train_only.csv`는 Flight teacher의 0.5 threshold 기준 FP/FN 126개 study-target 쌍을 담고 있다. 공식 gold, teacher 확률, train report 원문, review_status, error_taxonomy 필드를 보존했다.

이 126개 항목을 모두 임상적으로 검토·분류한 것은 아니다. 현재 review_status는 UNREVIEWED, taxonomy는 UNASSIGNED다. 0.5 threshold는 오류 탐색용일 뿐 AUC 최적화 threshold를 의미하지 않는다.

검토할 원인 분류는 negation, severity, historical finding, postoperative state, suspected finding, omission, translation, bilateral ambiguity, OA/chondropathy, contusion/marrow edema, fracture subtype다. 이 범주별 빈도나 주원인을 이미 확정한 것처럼 기록하지 않는다.

## Unknown과 confidence

Probability, 명시적 unknown annotation, source annotation coverage, teacher 누락, source 간 variance는 별개 필드다. Lixin의 확률 또는 confidence만으로 not-addressed 여부를 추정하지 않는다.

Pilk gold57의 명시적 unknown을 0으로 바꾸면 macro 0.866009959, 0.5로 바꾸면 0.870039590이었다. 이것은 teacher diagnostic이며 새 student의 개선 결과가 아니다.

## 실제 실패한 consensus 실험

Gold 성능을 보기 전에 고정한 Steven raw / Pilkwang LLM / Lixin raw / Qwen raw 네 family의 동일 비중 평균은 0.880066925였다. Public Flight control 대비 paired 5,000 bootstrap delta 95% CI는 [-0.031704637, -0.007473768]이었다. 단순히 teacher 수를 늘리는 설정은 채택하지 않았다.

명시적 unknown만 0.5로 바꾼 변형은 0.887101696으로 여전히 control보다 낮았다. 이미 공개 community에서 gold를 활용한 선택 편향과 다중 비교는 bootstrap으로 해소되지 않는다.

4,349개 non-gold study만 포함하는 training target 및 uncertainty features를 출력했다. Confidence-weighted BCE의 student 효과는 아직 학습으로 검증하지 않았다.
