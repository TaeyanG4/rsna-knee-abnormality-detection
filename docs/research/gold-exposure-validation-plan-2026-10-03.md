# Gold 노출 이후 검증: 외부 사례와 적용 계획

2026-10-03 사용자 요청. **계획은 있었지만 정확 D01의 독립 임상 검증셋을 확보한 상태는 아니다. 현재 Gold를 loader에서 제외해도 과거 가중치·teacher·모델 선택의 노출은 없어지지 않는다.** 앞선 은 현재 자체 학습 경로에 한정하며 공개 모델 전체에 확대하지 않는다.

## 계보에서 확인한 것

|대상|확인된 노출/분리|판정|
|---|---|---|
|공개 anchor의 residual CoAt|자산 명세에 Gold58 AUC로 epoch4/6/8 선택|Gold 직접 gradient 학습 여부와 별개로 Gold 독립 검증 불가|
|다른 공개 구성·88-feature stacker|전체 train UID·초기weights·fit/선택 계보 미확정|Gold 학습 안 했다고 가정하지 않음. 모두 직접 학습했다고 단정하지도 않음|
|옛 R2|학생 outer-fold를 본 teacher가 학생 학습용 라벨을 만든 경로, 혼합라벨 epoch 선택|원라벨 재채점·bootstrap으로 복구되지 않음|
|정확 D01의 R2 mean5|기존 train 사례에는 통상5개 중4개 student가 직접 학습|전체5개 평균을 기존 train에 적용한 점수는 OOF가 아님|
|EXP078~080/081|ImageNet 초기화, Gold 직접 학습/epoch 선택 제외. 사후 Gold 진단 경로는 있었음|공개 anchor와 같은 직접 학습 노출로 일괄 묶지 않음. 연구자 선택 노출·weak 평가 한계는 남음|
|EXP082|ImageNet 초기화와 동일 train/eval 계약, Gold 읽기/평가 경로 제거|노출 weak F2의 제한된 개발 대조|
|EXP083|EXP080 B8 encoder를 상속하는 준비 중 실험|F2를 새 독립셋으로 재명명하지 않음. 새 split에 B8을 그대로 재사용할 수 없음|

[독립 계보 검토](../handoffs/evidence/2026-10/2026-10-03-gold-lineage-plan/independent-review.json). 정확 D01 구성 자산과 별도로 읽은 공개 notebook revision을 혼동하지 않는다.

## 이번에 확인한 다른 참가자의 방법

토론5개(744961,736635,735304,745214,740718)는22:04~22:06에 재수집했다. 나머지 notebook/과거 writeup은 연구실의10-02 저장 revision을 확인했다. 실행하거나 해당 점수·private 성과를 재현한 것은 아니다. 참가자의 설명은 근거 수준을 구분하며 실행 지시로 읽지 않았다.

|자료|실제 내용|가져올 원칙/한계|
|---|---|---|
|[Best single-model score](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304), Tucker Arrants 답글3518449/3519005/3521401|보고서 라벨로 CV, Gold는 학습에 사용한다고 설명. 수정한 라벨로 검증하면 자기확인이 된다고 경고|학습라벨과 평가 reference 분리. Gold 학습 모델을 같은 Gold로 검증하지 않음. LB를 CV로 대체하겠다는 선택까지 복제하지 않음|
|[RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), 저장본문2740~2778/2801~2859|Gold target의 loss weight3, 보고서그룹 holdout. Gold 진단은 holdout에 속한 Gold만 사용. 전체 holdout 점수로 epoch/config 선택|Gold 일부 학습 자체보다 평가 사례의 실제 제외가 중요. 이 holdout에는 Gold도 포함되므로 Gold 진단은 선택에 완전히 독립적이지 않음. snapshot 코드가 D01의 실제 weights 계보라는 증거는 아님|
|[Raptor single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), 저장본문106~125와 [저자 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696)|Gold45로7-arm subset 탐색, 이후 Gold58 gate; Gold의 큰 혼합 이득이 LB에서는 작았다는 기록. 저자는 한 full-data 모델에서 작은 완전라벨 holdout으로 checkpoint 선택했다고 설명|높은 공개 점수가 Gold 검증의 독립성을 만들지 않음. 선택 노출을 그대로 기록|
|[ResNet34 baseline](https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34), 저장본문65~66/131~133/277~280|manual fold 중0을 검증, 나머지를 학습하고 best validation epoch 선택|새 학습에서 Gold OOF를 만드는 방식은 가능하지만58건은 작고 반복 선택의 독립 시험셋은 아님. 공개 pretrained 노출은 별도 검사 필요|
|[How do you optimize labels?](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214)|Gold .945인데 LB .910인 사례, 라벨 변경이 LB를 개선했다는 반대 사례도 있음|Gold→LB 변환이나 라벨 전략 무효의 일반 법칙을 만들 수 없음. 모두 참가자 자기 보고|
|[LB OOF relationship](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736635)|weak CV와 LB의 차이, Gold 상관에 대해 상반된 경험|절대값을 같은 정답 분포의 generalization gap으로 해석하지 않음|
|[공개 자원 검증 질문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744961)|이번 재수집에도 독립성을 해결하는 답변이 없음|공개 토론에 해결법이 있다고 꾸며내지 않음|

## 지난 대회에서 확인한 원칙

- [RSNA2023 16위](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/447448)는 label stratification + **patient grouping의4fold StratifiedGroupKFold**와 대회 loss를 명시했다. 이번에도 study보다 상위의 환자/중복 연결을 고려할 이유다. 현재 report/입력중복 component는 검증된 환자ID와 같지 않다.
- [RSNA2024 9위](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539690)는 keypoint/classifier5fold와 모든 모델에 같은 split 파일을 사용했다고 설명했다. 앙상블 부품별 평가 사례가 일치해야 한다는 점을 가져온다. 모든 ancestor의 제외가 이 글만으로 검증된 것은 아니다.
- RSNA2022 1위·RSNA2024 7위 글도 읽었으나 이 문제를 해결하는 더 구체적인 계보·Gold 대안은 본문에서 확인하지 못했다. 그림만 있는 CV 내용을 숫자/환자독립 근거로 추정하지 않았다.
- 이 대회들은 과제·라벨 범위·지표가 다르다. fold 수나 과거 CV 상승폭을 Knee의 기대효과로 복제하지 않는다. Biohub의 핵심도 배아 수/지표를 옮기는 것이 아니라 **이미 본 데이터로 일반화를 평가하지 않는 것**이다.

## 적용할 검증 순서

1. **현재 탐색은 정해진 범위로 마친다.** EXP083은 기존F2에서 head 개입 하나를 비교한다. 좋은 결과여도 weak 개발 후보이며, label/독립검증 문제를 해결했다고 표시하지 않는다. 자동5fold·384·새seed 확대는 없다.
2. **확대 전에 새로운 레시피를 공정하게 재학습한다.** 이미 만든4196 component 계약으로 하나의 outer V를 결과를 보기 전에 고정한다. baseline/variant, competition에서 미세조정한 encoder, teacher/student, learned preprocessing, stacker/calibrator에서 V 전체를 제외한다. ImageNet 등 계보가 확인되는 초기값부터 재학습하고 epoch/비율은 고정하거나 T 내부에서만 선택한다. 기존B를 새V에 재사용하지 않는다. 이는 *새 레시피 확인*이지 기존 D01 weights 검증이 아니다. 과거 연구자 노출·weak truth·미확인 환자중복도 남는다.
3. **Gold58 재분할은 보조 진단의 선택지다.** 모든 ancestor가 해당 Gold fold를 보지 않은 상태로 처음부터 학습해야 한다. Gold 전체로 선택된 공개 weights를 출발점으로 쓰면 head만 다시 학습해도 안 된다. 작은 fold의 단일class AUC와 표본변동 때문에58건으로 .0005급 개선을 판별할 것으로 기대하지 않는다. 여러 fold를 계산한 사실만으로 과거 prompt/모델 선택 노출이 사라지지도 않는다. 현재 이를 위해 별도5fit을 자동 투입하지 않는다.
4. **임상 정답의 공백은 별도로 해결해야 한다.** 독립 판독 라벨과 모델 계보에서 미사용임이 입증된 사례가 필요하다. 기존 train을 새로 판독해도 이미 그 영상을 본 exact D01의 미사용 일반화 시험은 되지 않는다. 새 판독을 future V로 쓸 경우 baseline/variant의 관련 ancestor도 V를 제외해 재학습해야 한다. 보유 외부 자료의12target 정합성·중복·사용 조건 또는 제공 train의 새 판독 가능성/비용을 먼저 확인한다. Report를 LLM으로 다시 읽는 것은 새 영상 Gold를 만드는 일이 아니다.
5. **정확 D01은 현재 제한된 전이 탐색이다.** 이미 승인한 제출1회의 public 결과와 실행 성공을 분리해 기록한다. 새 미사용 평가자료/완전한 제외 계보가 확보되지 않으면 독립 로컬 임상 개선이 입증됐다고 말하지 않는다. Public 반복탐색으로 이 공백을 덮지 않는다.

신규 grouped RN2fit26~36GPU분, 공통B8까지 새로 학습한 blend확인66~82분은 기존 실측 기반 계획값이다. exact D01의 다계열 재학습 비용이 아니다. 임상 판독 예산·외부평가 가능성은 아직 미확정이며 승인 없는 지출은 하지 않는다.

수동 판독 [토론740718](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740718)은 이번 재수집에도 참가자 의견2개뿐이고 주최측 확답은 확인하지 못했다. 규칙에는 validation/test hand labeling 제한이 있으므로 train 파생 판독의 실제 활용범위를 확인한 뒤 구체안으로 다룬다. 이번 작업은 판독 의뢰·게시·외부 메시지를 보내지 않았다.

[수집한 정확 revision/본문](../handoffs/evidence/2026-10/2026-10-03-validation-practices/) · [Gold/Silver 대응 현황](gold-silver-status-2026-10-03.md) · [Biohub 점검](../runbooks/strategy-review.md).
