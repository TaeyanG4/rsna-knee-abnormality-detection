# ADR 0004 — 검증과 공개 LB 사용 규칙

2026-10-03 재정리. 사용자 요구: 공개 LB 과적합까지 고려해 최종 상위권을 노린다. 이 문서가 이전 규칙을 대체한다. [감사·실행 계획](../research/route-to-first-2026-10-03.md), [변경 전 원문](../handoffs/archive/snapshots/2026-10-03-direction/manifest.json).

2026-10-03 후속 지시 반영: 남은 기간 총6~10회 제한을 아래의 일5회 기회 활용 원칙으로 대체한다. 매회 제출 승인·가설 사전 고정·반복 LB 튜닝 금지는 유지한다. [이번 변경 전 원문](../handoffs/archive/snapshots/2026-10-03-continuous-research/manifest.json), [지속 연구](../runbooks/continuous-research.md).

## 무엇을 검증했다고 부를 것인가

- weak CV는 고정한 보고서 기반 라벨의 예측력이다. 대회 영상 재판독 정답의 성능이나 private LB를 직접 측정하지 않는다. Gold58은 여러 실험에 노출된 진단셋이며 epoch·가중치·채택 기준으로 사용하지 않는다.
- 현재 Jiweiliu0.943는 **public 재현 기준본**이다. 그 안의 공개 라우팅/보정이 private에서 안정적이라는 증거는 없다. 기준본 유지와 private 성능 보장을 구분한다.
- 공개 competition checkpoint가 validation study를 학습했다면 그 예측은 독립 OOF가 아니다. 이 checkpoint와 자기 모델 OOF를 섞은 점수를 clean ensemble OOF라고 부르거나 그 결과로 비율을 최적화하지 않는다. 제출 활용 자체는 별개다.
- R2/EXP072·073은 teacher→student 간접 fold 누출이 있는 탐색 결과다. 기존 모델을 제출 후보로 검토할 수는 있으나 +0.0143을 독립 개선으로 보고하지 않는다. 누출이 테스트 데이터 사용을 뜻하는 것은 아니다.

## 분할·학습·선택

- 기준 split은 workspace://experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv의 보고서 그룹5fold(869×5). Gold58와 중복4건은 제외한다. 환자ID가 없어 보고서 그룹은 환자 독립성의 대리일 뿐이다.
- 라벨 혼합 검증은 outer V를 모든 teacher/student 및 competition 사전학습 ancestor에서 제외한다. T 안의 inner OOF로 혼합라벨을 만들고 V는 고정 reference labels로 평가한다. 기존 global OOF 파일 재사용은 이 조건을 충족하지 않는다.
- 최종 판정용 outer V로 epoch를 고르지 않는다. inner validation에서 정한 epoch 또는 사전 고정 epoch를 사용한다. 현재 학습기의 OOF checkpoint 선택 결과는 개발 CV로 표시한다. full-data fold -1의 Gold fallback은 수정 전 사용하지 않는다.
- label intervention도 평가라벨·UID·12target·split·입력·정밀도를 고정한다. 바꾼 학습라벨로 계산한 점수끼리는 비교하지 않는다. mentioned/uncertain/negative는 별도 의미이며 단순 teacher≤0.1을 silence로 해석하지 않는다.
- 이미 본 fold를 새로 이름 붙여 untouched holdout이라고 부르지 않는다. 이후 실험에 고정할 holdout의 과거 노출과 ancestor 제외 여부를 명시한다.

## 안정성과 통계

- mean AUC뿐 아니라12소견별·fold별·스캐너/자장·결측 그룹 결과와 seed 변동을 본다. 그룹 격차만으로 site 암기를 단정하지 않는다. group-sliced OOF는 해당 스캐너를 학습에서 제외한 검증과 다르다.
- 최종 유망 레시피에만 추가 seed와 사전 지정 scanner-group holdout을 적용한다. 표본수·양성/음성 수·평가 가능한 target 수를 같이 기록하며, 단일 class target의 AUC는 임의로0.5로 채우지 않는다. 부분 target 지표는12-target macro와 구분한다.
- 고정된 paired 예측에 report-group bootstrap을 적용한다. 95% 구간이0을 포함하면 **이득 입증 부족**이지 ‘두 모델이 동일’이라는 증명은 아니다. CI는 라벨 오류·반복 탐색·학습 seed·private 분포 불확실성을 제거하지 않는다.
- 깨끗한 자체 OOF끼리의 혼합 비율은 outer 평가를 남기는 nested 방식으로 결정한다. 낮은 예측 상관만으로 앙상블 개선을 주장하지 않는다.

## 공개 LB 과적합 방지

1. 실행 전 실험ID, 코드/가중치 hash, 바꿀 요소1개, checkpoint 선택법, 가중치, 평가 지표, runtime 중단 조건을 기록한다. 제출 결과를 본 뒤 이전 가설을 바꾸지 않는다.
2. public LB는 사전 고정 후보의 전이 확인에 쓴다. 서로 다른 근거 있는 가설·완성 후보가 있으면 하루 최대5회 기회를 적극 활용하되, 매번 잔여 한도와 pending을 확인하고 사용자 승인을 받는다. 한 가설당1~2개 후보가 기본이며 점수가 오를 때까지 비율·seed·소견별 처리를 반복 제출하지 않는다. 횟수 소진이나 public 최고점 추적은 목표가 아니다.
3. 표시0.001 차이는 단독 채택 근거가 아니다. 반대로 상승폭이 크더라도 자동 채택하지 않는다. 내부 안정성·실행 품질·같은 수정의 반복 가능성을 함께 판단한다. public 최고점을 고른 사실 자체의 선택 편향도 기록한다.
4. 공개 테스트3건의 UID/행별 수정, Gold58·LB로 소견별 비율 탐색, 특정 모델을 좋게 보이게 할 split 변경은 하지 않는다. 기존 기준본은 그대로 보존하되 그 공개 튜닝 계보를 숨기지 않는다.
5. 심각한 LB 하락은 계약/분포/라벨/구현 문제를 재감사하는 신호다. 약한 라벨CV와 LB의 숫자 차이를 같은 정답의 generalization gap으로 해석하지 않는다.
6. 최종2개는 재현 가능한 주력 조합과 안정성·오류 구조가 다른 근거 있는 후보를 고른다. ‘public 최고’와 ‘자동 flat-0.60’로 미리 고정하지 않는다. 검증이 부족하면 기존 기준본을 예비안으로 남긴다. 어느 후보도 private 안정성이 보장되지는 않는다.

## 추론·배포

- 검증된 공개 기준본의 정밀도·전처리를 일괄 변경하지 않는다. 자체 Raptor T4 경로는 fp16을 기준으로 fp32/local parity를 검증한다. bf16 전면 금지 규칙을 기존 공개 파이프라인에 소급 적용하지 않는다.
- raptor_infer.py는 고정 로컬 corpus를 읽는다. 그 도구의 sample test 성공만으로 Kaggle 숨겨진 DICOM 추론이 구현됐다고 보고하지 않는다. 신규 test.csv UID에서 직접 전처리하는 serving adapter와 학습 계약의 일치가 필요하다.
- 공개3건은 schema/parity 확인용이다. decode·전처리·추론·I/O와 cold start를 포함한 대표100~200 train study, 큰 study의 꼬리·메모리·실패율로 실행시간을 검증한다. 9h 한도 대비 약30% 여유는 계획 목표이며 실측 전 완료 주장 금지.
- 성공 파일 존재와 모든 reader 성공을 구분한다. 누락 study·중복 UID·NaN·상수열·fallback·실제 참여 모델·hash를 기록한다. 기존 fallback을 조용히 새 모델 성공으로 표시하지 않는다.

이 규칙으로 과적합 위험을 줄일 수 있으나 private 순위는 공개되지 않아 최종 결과를 보장할 수 없다.
