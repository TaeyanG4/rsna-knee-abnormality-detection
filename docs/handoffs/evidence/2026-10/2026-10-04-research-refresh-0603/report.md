# 2026-10-04 06:09 KST · 제한된 최신 자료 재확인

- 상태: 완료. 담당 `/root/precision_code_review`. 최신 실행 노트북100건·최근/고정 토론20건 목록에서 노트북2·토론2의 실제 본문만 수집·비교했다. 상세 ID/revision/SHA/시각·명령은 [receipt](receipt.json). 전체 sync·LB/제출 조회·GPU·가중치/라벨/픽셀 접근 없음.
- 결정: **EXP086 V2 조건 유지. 새 성능 개선 증거는 없으며 실험 중 변경·재시작 근거도 없다.** 과거 토론16건 재확인 및 Silver 감사를 반복하지 않았다.

| 새 관측 | 코드에서 확인한 변화와 판단 |
|---|---|
| [DINOsaur train](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5-train) v10→15 | 실제 모델은 ConvNeXt-Tiny. uint8 adaptation memmap·geometry cache·연속 float16 feature bank로 I/O를 줄였다. 속도개선은 미실측이며 우리 기존 corpus cache와 상당부분 겹친다. |
| [Plane-Aware](https://www.kaggle.com/code/lucifer19/rsna-knee-plane-aware-2-5d-mil-gpu-baseline) v10→13 | 본문 이름은 KneeFusion V11. 연결그룹·inner/outer 분리·EMA·유효slot 인코딩 추가. 기본224→252, 중심.38/.62→.35/.65 등 여러 조건이 함께 변함. 원문도 새 성능을 미검증으로 명시한다. |
| [외부 MRI](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416) 댓글12→13 | 새 답글3532465는 과거 Deepfake 실격 사례를 인용한 참가자 의견. RSNA host의 새 허용/금지 결정이 아니다. |
| [AUC 개선 질문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745576) 신규 | 불확실 라벨 제외가 도움되지 않았다는 자기보고. 비교코드·분할·수치·답변이 없어 Silver 변경 근거 부족. |

DINOsaur의 중요한 불일치: 소개는 “family-balanced”지만 v15는 provider-family 묶음을 제거하고 CSV별 median을 낸다. 같은 제공자 변형CSV가 중복 영향력을 가질 수 있다. adaptation은 Gold를 제외하나, head epoch 선택과 소견별 blend 권고는 Gold58을 사용한다. 독립 검증본으로 채택할 수 없다([변경분](dinosaur-diff.txt)).

Plane-Aware의 인접3장과 uint8 memmap은 v10에도 있었다. 새 성공사례로 계산하지 않는다. depth sampling은 다음 입력 가설의 참고자료로만 보존하며 EXP081/085 판정을 뒤집지 않는다. source의 decode 대체/fallback도 우리 성공 계약에 그대로 옮기지 않는다.

기존3자료의 added_at은 보존, 신규745576만 첫 등록. 새 소스는 localhost 연구실에 저장했다. 전역 수집시각·과거 public_score/last_run은 최신화하지 않았으므로 새버전 점수로 해석하지 않는다. 첫 목록페이지만 확인했으며 모든 변경을 포괄하지 않는다.

- 이전 결정 적용: M01/02/03/05/06/08/10. 다음: 현 DINO 결과 후 비용이 고정된 입력 가설만 재판정. 중앙 문서·실행계약 변경 없음.
