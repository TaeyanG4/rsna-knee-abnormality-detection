# Silver 처리의 외부 사례와 PFOA 점검

2026-10-03 KST. 사용자 요청: 다른 연구의 처리 방법을 확인하고, 개선 가능성이 큰 Silver 오류·누락부터 조사한다. **외부 사례는 상태를 구분하고 검증하는 방법의 근거다. 우리 대회 점수 상승의 근거는 아니다.**

## 다른 연구에서 실제로 한 것

|자료|확인한 처리|이번 대회에 적용할 부분과 한계|
|---|---|---|
|[CheXpert, Irvin 등, 2019](https://arxiv.org/html/1901.07031v1)|보고서 추출에서 미언급(blank), 음성(0), 불확실(u), 양성(1)을 구분. 불확실한 셀의 loss를 제외하거나, 0/1로 매핑하거나, self-training의 soft 예측으로 바꾸거나, 별도 class로 학습하는 5가지 접근을 비교했다.|불확실/미언급/명시적 음성을 먼저 보존한다. U-Ignore는 불확실한 **셀**을 제외하는 방식이지 행 전체 제외가 아니다. 보고서 누락을 전부 mask하는 것이 항상 좋다는 결론은 없다. 논문도 무작위가 아닌 결측에서 제외 편향을 지적한다.|
|[CheXpert 공식 설명](https://stanfordmlgroup.github.io/competitions/chexpert/)|3명 판독 합의의 200-study 영상 검증에서 소견별 uncertainty 방식을 선택하고, 별도 500-study/500 unseen-patient test에서 평가했다. uncertainty 방식의 우열이 소견마다 달랐다.|우리의 Gold58은 이미 선택에 노출됐다. 기존 Silver AUC만으로 교정법을 선택하거나, 노출 Gold58로 여러 mask/가중치를 반복 선택하는 절차로 옮기지 않는다. 공개 모델의 전체 ancestor를 제외한 새 검증은 여전히 필요하다.|
|[MOAKS, Hunter 등, 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435/)|연골 손상을 부위별로 평가하며, 전체 손상 면적과 전층 손상 면적을 나누어 0/1/2/3으로 기록한다. 면적 구간은 없음/<10%/10–75%/>75%다.|연골 이상 유무만으로 PFOA 양성을 정하기보다 부위·깊이·범위를 분리해 기록한다. MOAKS 면적 비율은 대회의 대략 1cm 기준과 같은 단위가 아니며, 점수 번호를 대회 이진 라벨로 바로 변환하지 않는다.|
|[sMOAKS, Altahawi 등, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12372977/)|기존 MOAKS가 얕은 부분층과 깊은 부분층 손상을 구분하지 못하는 점을 보완해, >50% 두께 손상에 대한 별도 범위 점수를 추가했다.|대회 PFOA의 >50% 깊이 기준과 관련된 참고다. 연구의 목적은 판독 신뢰도·표현 세분화이며, 우리 classifier/LB의 개선 실험은 아니다. 논문의 등급을 새로운 정답으로 사용하지 않는다.|

CheXpert에서 검증 결과가 좋았던 소견별 처리법을 Knee에 그대로 복사하지 않는다. 흉부 X-ray와 무릎 MRI의 병변 정의, 보고서 작성 관행, 누락 구조 및 평가 정답이 다르다. 특히 CheXpert의 불확실(u)은 우리 Qwen의 mentioned=0 또는 낮은 P(present)와 같은 변수가 아니다. 이 문서는 CheXpert의 blank에 대한 모든 학습 구현까지 검증했다고 주장하지 않는다.

## 공개 Knee 노트북의 처리

저장된 [Pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)과 [Flight grouped-fold baseline](https://www.kaggle.com/code/flight0234/rsna-knee-baseline-dual-grouped-folds)의 추출→loader→loss를 실행 없이 읽었다. 두 snapshot은 35개 대응 cell 중32개가 같으며, 추출기와 loader는 동일하다. 두 성공 사례로 중복 계산하지 않는다.

해당 regex 경로에서 uncertain-only는 soft target .626, confidence .55, loss weight .6625가 된다. 미언급은 target .28, weight .2875로 남는다. 둘 다 **mask되지 않고 학습에 기여**한다. 한 개의 경증 양성 표현은 기본 계산상 .719가 되며, 공식 PFOA의 깊이와 범위 조건을 함께 강제하지 않는다. 이 수치는 실제 표본의 평가값이 아닌 코드 산술이다. 가중 BCE 적용과 검증의 `Y>.5` 이진화를 확인했으므로, soft 수치가 AUC에 무관하다는 코드 주석만 믿을 수 없다.

외부 라벨 테이블이 있으면 이 regex 결과를 덮어쓰며, 가중치가 발견되면 학습 함수가 추론으로 바로 반환한다. 따라서 **소스의 구현과 실제 공개 점수를 낸 실행 경로는 구분**한다. 현재 Dread 라벨에 같은 오류가 있다고 단정하지 않는다. [셀/줄·SHA·EXP055 계보 한계](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/other-research/local-notebooks/report.md).

실제 학습 소스인 Dread도 별도로 확인했다. dataset metadata의 “calibrated probability”, “soft labels train better”는 제작자 주장이다. 지정한 metadata·학습/추론 source에는 생성 prompt/LLM, 보정 절차·검증셋·지표, soft/hard 통제 실험 또는 독립 판독 검증이 제시돼 있지 않았다. 공개 학습 source는 soft target을 직접 읽어 class prevalence 기반 pos_weight BCE를 사용하며, per-target uncertainty mask나 confidence 가중치는 없다. Gold를 checkpoint 선택에 쓰는 것은 추출 확률의 독립 보정 검증이 아니다. 이 자료 부족은 Dread 라벨이 틀렸다는 판정이 아니며, 저장 source와 현재 로컬 실행을 혼동하지 않는다. [Dread 출처 감사](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/other-research/dread-provenance/report.md).

## 우리 자료에서 확인된 것

- train 3,475건에서 12소견×4필드의 형식과 UID coverage는 유효했다. 그러나 **형식 유효는 의미 정확성이 아니다.** Qwen의 `confidence`는 P(present) 용도이며 보정된 라벨 신뢰도가 아니다. 별도 uncertain 필드는 없다.
- raw JSON의 3,474건에는 12소견 필드가 모두 명시돼 있었고, 1건은 Baker target 누락/매핑 불가였다. 전체 3,475건에서 제한된 정규화 결과는 CSV와 같았다. 광범위한 parser 누락을 현재 최우선 원인으로 볼 근거는 줄었지만, extractor의 미언급 판단 정확성은 아직 검토해야 한다.
- 현재 Dread 학습 CSV에는 mentioned/negated가 없다. Qwen의 mentioned를 Dread의 진실이나 loss mask로 바로 옮기지 않는다. 폴더 이름 Qwen3-8B와 저장 생성 코드 Qwen2.5-32B가 다르며, 이 CSV를 생성한 정확 실행 revision은 미확인이다.
- [주최측 기준](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)은 PFOA에 **대략 1cm 이상의 중등도/넓은 범위, 연골 두께 >50% 소실**을 요구한다. 저장 Qwen prompt는 chondrosis·골극 등을 넓게 받아들이므로 정의 차이의 실제 영향을 조사할 가치가 있다. 이것만으로 Dread 오류가 확인된 것은 아니다.

## 결정과 검증

PFOA train-only 최대20건을 고정 규칙으로 뽑아 원라벨·예측·Gold를 가리고 보고서만 검토한다. 경증/고등급/중증도 불명/어휘 미검출 층을 포함하되, 층화 표본이므로 전체 오류율은 추정하지 않는다. 보고서의 부위·깊이·범위와 직접 인용을 남기고, 명시적으로 판단할 수 없는 것은 indeterminate 또는 unmentioned로 보존한다. 이 검토는 영상 Gold가 아니다.

가림 판단을 hash로 고정한 뒤 해당20건의 Dread/Qwen을 한 번 대조한다. 실제 수정 근거가 확인되면 그 범위만 대상으로 새 감독 가설을 구체화한다. broad masking, 자동 재라벨, 추가 모델 확대를 먼저 하지 않는다. 손상 깊이는 명확하지만 범위가 없는 사례는 음성으로 강제하지 않는다. 순수 텍스트 규칙만으로 영상 참값을 보장할 수 없는 한계도 유지한다.

Biohub 점검: 작은 효과를 보존하고, 특정 가림 감사에서 오류가 적다고 Silver 전략 전체를 폐기하지 않는다. 필요한 입력·계약 검사를 성능 증거와 구분하며, 다음 투자는 도구 확장보다 실제 감독 오류의 확인에 둔다.

## 23:18 결과와 후속 결정

20건/20개 component를 준비했고 두 LLM 검토를 각각 가림 상태에서 동결했다. 서로의 판단과 source labels를 보지 않은 검토지만 임상의 독립 판독은 아니다. 19건의 범주가 같았고, 해부학적 위치 해석이 갈린 PFOA-013은 합의시키지 않고 unresolved로 보존했다. 합의 음성후보9, 판단불가8, 미언급2, 불일치1이다. 엄격한 깊이+범위를 모두 명시한 양성후보는0이므로 양성 검출 민감도는 평가하지 못했다.

두 판정과 비교 규칙을 고정한 뒤 해당20건의 PFOA source fields만 한 번 개봉했다.

|사전 고정한 합의 음성 유형|건수|Dread 이진양성|Qwen 저장 양성|
|---|---:|---:|---:|
|PF 연골 소실 <50% 명시|2|0|2|
|정상/보존된 연골 명시|4|0|0|
|경증·low-grade의 정성적 표현|3|0|2|
|합계|9|0|4|

Dread soft 값은 이9건에서 .05 또는 .2였고 기존 float32→rint 평가 규칙으로 모두 음성이었다. **현재 학습용 Dread에 이 PFOA 중증도 규칙을 덮어쓸 직접 근거는 나오지 않았다.** Qwen4건은 가림 텍스트 정의와의 불일치 후보이며 영상 참값 오류로 확정한 것은 아니다. 두 명시적 <50% 사례는 Qwen prompt의 광범위한 OA 정의 문제와 일관된다. 판단불가·미언급11건(검토 불일치 포함)은 source 값과 달라도 오류로 집계하지 않는다. 표적20건의 수치는 전체 오류율이 아니다.

원 source/학습/가중치는 변경하지 않았다. 자동80건 확대와 PFOA 교정 재학습을 보류한다. 재검토 조건은 새로운 Dread의 명시적 정의 충돌, 또는 독립 영상 기준에서 확인된 문제다. 이 결과로 PFOA 감독·누락 처리 전체를 닫지 않는다.

train3475 source 교차집계도 소스별1회로 완료했다(0.312초, UID/형식 오류0). Synovitis는 Dread 양성872/Qwen 양성881로 총수는 비슷하지만 D+Q−495, D−Q+504로 대상이 크게 달랐다. Qwen 미언급2453 중 Dread 양성467이다. [12소견 전체 표](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/cross-source-triage-v1/aggregate-table.md). 두 추출기 중 어느 쪽이 맞는지, 임상 오류율이 얼마인지는 이 집계로 알 수 없다.

이를 근거로 다음 소규모 검토는 **Synovitis16건**을 선택했다. 네 source 조합에서 각4건/component당1을 고정 hash로 선정하고, source값·UID·층을 숨긴 full Report packet을 준비 중이다(23:28 시작,23:38 목표/23:43 상한). 활막염의 직접 서술과 effusion·plica·Hoffa 대리표현이 어떻게 다르게 처리됐는지, 한 source가 명시적 소견을 놓쳤는지 구분한다. 가림검토 예산30분이며 source값을 열기 전에 판단과 인용을 고정한다. 자동 재라벨/학습/표본확대는 하지 않는다. [사전 범위와 비용 비교](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/root-synovitis-decision.json).

Baker 중증도 작은 감사(약20–30분)는 대안으로 유지한다. Synovitis 감사(약30–60분 전체예상)는 양방향 차이가 크고 미언급/대리표현을 점검할 가치가 있어 우선했다. PFOA80 확대(약90–150분)는 현재 Dread 수정 근거가 약해 보류한다. 시간은 계획이며 점수 상승량은 추정하지 않는다.

[가림 packet/등록](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/pfoa-pilot20/HANDOFF.md) · [root 판정](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/pfoa-pilot20/root-blind-v1/freeze.json) · [별도 판정](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/pfoa-pilot20/reviewer2/freeze-receipt.json) · [불일치 보존](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/pfoa-pilot20/comparison-annotations.json) · [단일 source 대조](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/pfoa-pilot20/unblind-v1/result.json).

출처 원문·HTTP/시각/hash: [외부 연구 evidence](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/other-research/). [Qwen 의미·공식 정의](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/source-semantics/report.md), [필드 감사](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/field-inventory/result.json), [raw 구조 감사](../handoffs/evidence/2026-10/2026-10-03-silver-mention-audit/raw-presence/result.json). 저장된 Kaggle 자료의 snapshot과 현재 실제 revision은 구분한다.
