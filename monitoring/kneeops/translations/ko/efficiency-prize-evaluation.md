## 효율성 상

정확도가 높은 모델은 계산 비용도 큰 경우가 많으므로, 모델 효율성에 초점을 맞춘 두 번째 트랙을 운영합니다.

효율성 상에서는 실행 시간과 예측 성능을 함께 평가합니다.

대상이 되려면 제출물은 다음 조건을 충족해야 합니다.

- 팀이 리더보드 상을 위해 선택한 제출물 중 하나이거나, [My Submissions](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/submissions) 탭에 설명된 조건에 따라 자동 선택된 제출물 중 하나여야 합니다.
- Private Leaderboard에서 `sample_submission.csv` 기준 제출물보다 높은 순위여야 합니다.

이 조건을 충족하는 모든 제출물이 효율성 상 심사 대상입니다. 하나의 제출물이 리더보드 상과 효율성 상 양쪽의 대상이 될 수 있습니다.

비공개 테스트 데이터에서 아래 지표로 매긴 순위에 따라 적격 제출물에 상을 수여합니다. 순위별 상금은 [Prizes](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/overview/prizes) 탭을 참고하세요. 추가 내용은 토론 포럼 업데이트로 게시될 수 있습니다.

## 효율성 점수

제출물의 **효율성 점수**는 다음과 같이 계산합니다.

$$
\text{Efficiency} = \frac{\text{AUC}}{\text{Benchmark} - \max\text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}
$$

`AUC`는 [메인 대회 지표](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/overview/evaluation)에 따른 제출물 점수, `Benchmark`는 기준 제출물 `sample_submission.csv`의 점수, `max AUC`는 Private Leaderboard의 모든 제출물 중 최대 AUC입니다. `RuntimeSeconds`는 제출물을 평가하는 데 걸린 초 단위 시간입니다. **효율성 점수를 최소화하는 것이 목표입니다.**

대회 학습 기간에는 매일 갱신되는 [Efficiency Leaderboard 노트북](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)에서 **공개** 테스트 데이터 기준 리더보드를 확인할 수 있습니다. 대회가 끝나면 **비공개** 데이터의 효율성 점수로 갱신합니다. 학습 기간 중에는 팀별 순위만 표시하고 전체 점수는 표시하지 않습니다.
