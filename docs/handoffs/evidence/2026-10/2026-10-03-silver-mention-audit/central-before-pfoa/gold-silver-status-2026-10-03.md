# Gold·Silver 차이와 실제 대응 상태

2026-10-03 사용자 재점검. **일부 방어와 감사는 적용했지만 Silver 라벨 오류를 해결했다고 말할 단계는 아니다.** 연구실의 기존 공식 설명·주최측 답변·현재 학습 코드를 대조했으며 새 Gold 평가/모델 점수 계산·라벨 수정·학습·제출은 하지 않았다.

|구분|Gold58|Silver/weak supervision|
|---|---|---|
|출처|대회 정의에 따라 MRI 영상을 직접 재판독한 제공 라벨|대부분의 나머지 train 보고서를 LLM 등으로 읽어 만든 파생 라벨; 소스별 정의가 다름|
|절차|MSK 전문의2명 독립 판독, 불일치 제3판독자 조정이라는 주최측 설명|보고서의 임상 서술·번역/추출·미언급·중증도 판단을 거치므로 영상 재판독 정답과 다를 수 있음|
|핵심 한계|58건으로 작고 이미 여러 선택에 노출됨; 현재 독립 holdout 아님|누락이 무작위가 아니며 보고서의 미언급은 영상 음성 아님; soft 수치도 보정된 질환확률로 입증되지 않음|
|현재 용도|새 모델 학습·epoch/혼합 선택에서 제외. EXP082는 loader/평가 경로도 제거|Dread soft target으로 학습하고 고정 이진화 reference에서 개발 비교. 개선은 임상/private 개선의 직접 증거가 아님|

주최측은 영상 기반 라벨이 보고서와 독립적으로 생성됐고, 불일치 시 영상 라벨을 기준으로 삼는다고 답했다. 임상 보고서와 대회 판독의 중증도 기준이 달라 불일치가 예상된다는 설명도 있다. 테스트에는 Report가 제공되지 않으므로 텍스트는 학습 감독에만 쓴다.

## 적용된 것과 남은 것

- **적용:** soft target 학습, Gold 및 알려진 보고서 중복 격리, 고정 라벨/UID/epoch의 paired 비교, 원라벨과 모델 계보 추적, Gold/LB 사후 선택 제한. 새 그룹 계약에 확인된 입력 중복을 연결했으나 환자 독립성은 미확정이다.
- **실행한 감사:** MCL120 가린 텍스트 검토에서 음성후보55/unknown56/미언급9를 분리했다. 음성후보55건의 Dread 이진양성은0건이었다. 이로부터 Dread 전체가 정확하거나 unknown이 음성이라고 결론 내리지 않았다. 기존 source를 자동교체하지 않았고 라벨연구 전체를 기각하지 않았다.
- **미적용:** 기존 Silver 모든 셀의 언급상태·불확실성에 따른 loss mask/신뢰도 가중치, 소견별 보고서→영상 기준 차이 교정, 새 독립 영상 판독 평가. 현재 BCE는 이 구분 없이 기존 soft 값을 사용한다. 감사에서 unknown을 보존한 것과 학습에서 unknown을 제외한 것은 다르다.
- **코드 확인:** 라벨 loader는 NaN이 있으면 행 전체를 제외하며 소견별 missing-loss mask는 아니다. EXP080은 학습/epoch 선택에서 Gold를 제외했지만 선택 후 진단 평가가 남았고, EXP082는 Gold 읽기/평가 경로를 제거했다. EXP083은22:36 V2 full에서noGold 경로·고정paired 대조를완료했다. B8의과거노출을없애는것은아니다. [독립 코드 감사·줄 근거](../handoffs/evidence/2026-10/2026-10-03-silver-label-review/code-audit.json).
- **혼동 금지:** EXP082 pos_weight=1은 class imbalance/최적화 대조이며 Silver의 틀린 라벨을 고친 것이 아니다. EXP083은 고정 Silver 아래 표현 효과를 분리하는 실험으로 라벨 문제를 해결하지 않는다. 낮은 soft 확률을 미언급으로 분류하거나 다른 LLM의 높은 confidence를 정답 보증으로 쓰지 않는다.

## 다음 감독 신호 단계

EXP083은고정대조종료후현설정미채택이다. 사용자22:41선택에따라다음우선순위는 **현재 보유 raw 추출의 mentioned/negated/uncertain 필드가 실제 어떤 의미인지, train-only UID에 얼마나 연결되는지 확인하는 작은 감사**를 다음 감독 가설의 준비로 둔다. Synovitis/Baker's/PFOA 등에서 대회 정의와 보고서 누락/중증도 차이를 비교해 하나를 고른다. 기존 MCL120을 전체 소견의 대표 오류율로 쓰지 않는다. 새로운 영상 정답이나 실제 감독 수정 근거가 없으면 즉시 추가 학습하지 않는다.

감독을 바꿨는데 원 Silver AUC가 낮아질 수 있으므로 원라벨 AUC만으로 교정안을 기각하지 않는다. 원라벨 paired 점수·가린 정의 감사·별도 영상 평가의 역할을 분리한다. 영상 판독이 필요하면 구체 표본·비용·사용 가능 평가를 먼저 제안한다. LLM 보고서 재검토를 영상 Gold 생성으로 부르지 않는다.

Biohub의 교훈은 여기서 **미주석/미언급을 음성으로 착각하지 않기, 공개모델 계보와 데이터 중복을 확인하기, 후처리·도구 검사에만 시간을 쓰지 않기, 한 번의 보류로 감독/재학습 전략을 닫지 않기**로 적용한다. Knee는 macro AUC·대회 판독 기준·보고서 누락 구조가 다르므로 Biohub의 수치·mask 규칙을 그대로 복제하지 않는다.

[주최측 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826) · [판독 절차/정의](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) · [MCL 동결 후 대조](../handoffs/evidence/2026-10/2026-10-03-mcl-review/unblind-v1/report.md) · [Biohub 적용 점검](../runbooks/strategy-review.md).

출처는 연구실의 저장 revision을 사용했으며 이번에 Kaggle 최신 원문을 재수집한 것은 아니다. 토론을 읽는 과정에서 과거 노출 Gold 사례를 포함한 글이 보였으나 새 Gold 파일/점수를 계산하거나 규칙 선택에 사용하지 않았다. [원문 revision 보존](../handoffs/evidence/2026-10/2026-10-03-silver-label-review/).
