# 2026-10-03 21:09 KST · 대기 중 검증 준비와 향상/악화 판단

**오늘은 그룹·라벨·노출 계보를 고정하고, root가 진행하는 EXP082 소규모 개발 screen을 마치는 것이 적절하다. 새 split이나 반복 CV만으로 D01/private 성능이 입증되지는 않는다.** 실제 점수 계산·학습·픽셀·원격 실행·기존 split 수정은 이번 검토에서 하지 않았다.

|구분|지금 판단 가능한 것|판단할 수 없는 것|
|---|---|---|
|정확 D01: 90%rank(anchor)+10%rank(mean5(옛R2))|고정 코드/자산의 실행·parity·runtime 근거, 승인된 한 번의 public 전이 결과|anchor의 competition 학습 노출, R2 teacher 경유 노출을 제거한 독립 OOF. 재채점/새 split 명명으로 없어지지 않음|
|EXP078~080 B+새RN|ImageNet 초기화·고정 epoch·동일 weak 라벨에서 입력/보완 효과의 개발 대조|정확 D01의 개선, 임상 재판독 정답/private 개선|
|EXP082 RN pos_weight=1|기존 F2의 동일 조건 RN loss 비교와 고정 B+RN 조합 비교|F2를 untouched로 해석하거나 새 grouped 검증으로 간주하는 것|

EXP082 등록 파일은 21:05 표적 조회에서 아직 보이지 않았다. 당시 STATE는 **한 fit CPU 준비·GPU 미시작**이었다. root가 소유한 진행 작업을 중복 구현하거나 실행하지 않았다. 새 레시피의 고정 RN 대조와 정확 D01을 서로 대체되는 검증으로 부르면 안 된다.

## 오늘 가능한 최소 준비: 약20~40분, GPU 불필요(추정)

1. **새 그룹 계약만 작성한다.** 기존 eligible4,343건의 report hash4,197그룹에 확인된 duplicate edge를 union한다. 두 endpoint는 각1건의 서로 다른 report 그룹이므로4,196components가 된다. 현재 fold1↔0를 잇는 component1개다. UID·원 report hash·old fold·v3/v4 근거 SHA를 보존한다. 기존 folds.csv는 변경하지 않는다. 이는 확인된 입력 중복 처리이며 환자 독립성 검증 완료가 아니다.
2. **평가라벨을 훈련 loss와 분리해 동결한다.** EXP080의 reference `labels_llm_soft.csv`, 파일SHA `8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1`, semantic label hash `71be1189ddbce243ea91a26b924e597d1e4fc392eed58237669f8d6000afade4`, float32→numpy.rint·12target순서·평가UID·유효 target처리까지 고정한다. 이번 검토는 manifest의 계약만 읽었고 라벨값은 읽지 않았다.
3. **학습 계보 제외표를 만든다.** 미래 outer V의 component 전체를 direct train뿐 아니라 competition-finetuned 초기weights·teacher·pseudo-label·inner OOF의 모든 학습 ancestor에서 제외해야 한다. 확인 불가능한 공개 anchor/옛R2는 `exposure unknown/known`으로 표시한다. ImageNet 초기화의 새 B/RN은 이 조건으로 다시 학습할 수 있다. 새 split에서 기존 B가 V를 본 경우 재사용 금지다.
4. **노출·선택 규칙을 남긴다.** F0/F1/F2의 기존 실험/점수·epoch선택·연구자 열람 이력을 표시한다. V로 epoch/비율을 고르지 않고 기존8/20epoch와90/10을 사전 고정하거나 T내 inner선택을 사용한다. 새 그룹할당 역시 이미 본 데이터를 재배치한 개발 검증이며 untouched holdout이 아니다.

## 향상·악화 판단 순서

먼저 같은 UID·12target·reference labels·정밀도·입력·checkpoint선택 계약과 finite/fallback/누락 검사를 통과해야 한다. 주 비교는 **새RN−동일조건 원RN**과 **동일B를 공유하는 고정90/10(B,새RN)−90/10(B,원RN)**로 사전 지정한다. B단독 비교는 별도로 표시한다. 원래 D01과 수치를 섞지 않는다.

paired macro12 차이와 사전 지정 소견/양성음성수·fold·scanner·결측 진단을 함께 본다. bootstrap 단위는 report+확인중복 component다. CI가0을 포함하면 이득 입증 부족이며 동일성 증명이 아니다. 양수여도 고정 weak 예측의 조건부 근거로만 해석한다. 라벨오류·연구자 반복선택·학습seed·미확인 환자중복·private분포 불확실성을 포함한 보증이 아니다. 소견별 우연한 하락을 전부 veto하거나 결과를 보고 routing/weight를 찾지 않는다. 반대로 반복된 골절 손실은 macro 기여·표본수와 함께 명시한다.

## 다음 재학습의 비용과 한계

|작업|예상 로컬 GPU 시간|해석|
|---|---:|---|
|기존 F2 screen의 신규 RN 한 fit|약13~18분|기존 RN/B를 같은 계약에서 재사용. 노출된 개발 screen|
|새 grouped split, RN 원loss+pos_weight1 두 fit|약26~36분|새 split에서 RN loss만 공정 대조|
|위 두 RN과 공통 새B8 학습|추가40~46분, 합66~82분|양 B+RN recipe를 같은 새split/동일B에서 비교|
|등록·계보/무결성·결과검토|별도20~40분|준비/분석 추정; 위 GPU시간과 구분|

근거는 EXP080 실측 A8=2,457초, B8=2,502초, RN20=772초다. 새split 실행시간은 아직 실측하지 않았다. 한 seed·한 fold의 weak 개발 대조라는 한계는 남는다. **이 비용은 exact D01 전체를 독립 조건으로 재현하는 비용이 아니다.** anchor와 옛R2의 전체 ancestor를 다시 만드는 비용/가능성은 미확정이며 자동 확대하지 않는다.

공개 LB는 D01 사전 고정 가설의 전이 확인1회로 기록한다. 상승·동점·하락 모두 보존하고 그 결과로 비율·seed·소견 처리를 반복 탐색하지 않는다. 0.001 표시차이나 private를 추정한 값을 단독 채택 기준으로 삼지 않는다. D01 결과가 새 B+RN을 입증하거나 기각하지도 않는다.

[관측값·source SHA·group inventory](evidence.json). 준비부터 기록까지 약5분. 검토 중 EXP080 summary의 진단 metric 필드가 부수적으로 포함된 광범위 읽기가1회 있었으나 Gold 파일을 열거나 해당값을 판단에 사용하지 않았다. 이후 추출은 runtime/config로 제한했다.
