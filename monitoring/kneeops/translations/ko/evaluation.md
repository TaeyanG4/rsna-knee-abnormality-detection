제출물은 12개 타깃 각각의 예측 신뢰도 점수와 관측된 정답 사이에서 계산한 [ROC 곡선 아래 면적(AUC)](http://en.wikipedia.org/wiki/Receiver_operating_characteristic)의 평균으로 평가합니다.

$$
\text{Final Score} = \frac{1}{12}\sum_{i=0}^{11} AUC_i
$$

즉, 최종 점수는 ROC AUC의 매크로 평균입니다.

## 제출 파일

테스트셋의 각 행에 대해 12개 타깃 라벨 각각의 신뢰도 점수를 예측해야 합니다. 파일은 헤더를 포함하고 다음 형식을 따라야 합니다.

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
...
```
