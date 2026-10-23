# EXP086 이후 독립 반론

- 완료·`/root/precision_code_review`·2026-10-04 06:38 KST. 읽기전용 검토, 중앙문서 변경 없음. 상세 근거·SHA는 [receipt](independent-critique-receipt.json).
- 결론: **새 입력의 구체적 차이를 기존 metadata로 먼저 확정하고, CLS→patchmean 즉시 재학습은 후순위로 둔다.** 저렴한 GPU 시간이 단독 선택 근거는 아니다. root가 최종 선택한다.

## 관측과 해석

EXP086 V2는 B 대비 −.064575, 조건부 CI[−.073531,−.053989], 12소견 모두 하락했다. 3,448update/skip0, loss1.031→.898, full635.438초다. 이 전이 레시피 미채택은 타당하다. loss 감소는 학습 진행이며 수렴·최적 recipe의 증명은 아니다. DINO 전체 실패나 실행 결함으로 일반화하지 않는다.

현재 CLS→triplet MIL은 의도된 구현이다. CLS도 patch 정보를 집계하며 patch 평균은 작은 병변을 희석할 수 있다. 공유 B 학습률·8epoch가 DINO 최적이라는 근거는 없다. pooling이 이번 열세의 원인이라는 직접 근거도 없다. EXP026/027의 완료 frozen probe는 다른 입력·라벨·split이며 맞춘 CLS/mean 대조가 아니다. 재실행하지 않는다.

**구현 함정:** 설치 timm1.0.19에서 생성 인자 `global_pool=avg`만 바꾸면 기본 `fc_norm`도 바뀌어 최종 정규화가 token별→pool 후로 이동한다. parameter key도 달라진다. 순수 pooling 질문이면 기존 최종 token norm·가중치·head 구조를 유지하고 prefix 제외 평균만 바꿔야 한다. CLS로 학습된 head에 추론 때만 mean을 넣은 점수는 대조 실험이 아니다.

## 다음 최소 단계·시간

1. 기존 metadata만으로 추가 series의 plane/FS/범위와 현재 선택본 차이, native 인접성과 cache 인접성 차이를 명시한다. 설계30–60분은 추정이다. 다중후보1936/4407은 가능 범위이지 정보손실률이 아니다. EXP081 교체 실패도 추가 series 가설을 끝내지 않았다. 실제 추가정보가 불명확하면 새 cache/GPU에 진입하지 않는다.
2. 입력 후보는 고정 총K 재분배 **한 가지**를 먼저 정의한다. K증가·해상도·학습량을 동시에 바꾸지 않는다. 새 pixel decode/I/O 비용은 미측정. 원B 결과 재사용 가능한 계약이면 후보 B급40–50분, loader 변경으로 대조 재학습이 필요하면80–100분은 과거 실측 기반 추정이다.
3. root가 DINO pooling을 택하면 원generic 초기값에서 동일8epoch 한 후보만, 기존 CLS와 B 결과 재사용. 주비교mean−CLS와 B 대비 절대 열세를 함께 기록한다. GPU10–15분 추정은635초 기반이며 준비시간 별도다. 음성은 이 pooling 수정의 근거 부족, 양성도 즉시B교체 근거는 아니다. LR·epoch·해상도 탐색으로 자동 연장하지 않는다.

## 제한·재개

M02/05/07/08/10 적용. weak 개발V 반복노출·label/환자계보 공백·.943 전이공백 유지. patchmean도 유실한 원영상 정보를 복원하지 못한다. T4/4070 serving parity는 후보별 별도 계약이며 이번 로컬 score로 증명되지 않는다. 완료 report/소스만 읽었고 새 자료·GPU·모델load·라벨/픽셀·Gold평가·등록0. 남은 일은 root의 단일가설 선택; 별도 실행/예약 없음.
