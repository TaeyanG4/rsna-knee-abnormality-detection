# EXP084·serving 이후 다음 투자 비교

2026-10-04 01:36 KST. 읽기전용 제안이며 새 실험 등록·실행 지시가 아니다. EXP084의 고정3fit, 새 B/RN serving stage 및 현재 제출 계약은 그대로 둔다. 결과를 아직 읽지 않았으며 예상 LB 점수를 만들지 않는다. 이번에는 연구 문서4개와 연구실에 저장된 문서1개만 읽고, 이미 읽은 root 결정은 재사용했다.

**다음 성능 가설로는 아래 B의 ‘series 내부 상대위치·순서 정보를 학습하는 대조’를 우선 제안한다.** A의 학습량 가설은 구현 부담이 낮은 대안이고, C의 감독 정의 가설은 독립 영상 근거를 확보할 수 있을 때 우선순위가 높아진다. 어느 가설도 현재 자료로 성공 확률을 수치화할 수 없다. 선택 기준은 기존 실패와 겹치지 않는 질문, 사용할 수 있는 초기값, 검증 가능한 차이, 비용이다. EXP084가 끝났다는 이유만으로 새 fit을 자동 시작하지 않는다.

## 현재 집중이 지나칠 위험

EXP079의 고정90/10 RN 보완은 작은 양의 개발 효과였고 F2에도 남았으므로 폐기할 이유가 없다. 그러나 EXP084의 pos_weight 대조는 그 작은 감독 가중치 효과의 재확인이다. 결과가 양성이어도 backbone이 보는 정보나 표현 능력이 충분하다는 결론은 나오지 않는다. 반대로 음성이어도 영상·라벨 전략 전체의 기각이 아니다.

서빙은 필요한 연결 작업이지만 numerical PASS는 정확히 같은 모델을 실행한다는 증거다. 6시간 재평가는 이미 점수 정보 없이 실행 검증을 계속 확장할 위험을 지적했다. B parity·외부 RSS·한 번의 실제 T4 판정을 현재 범위에서 마친 뒤, 새 오류가 없다면 engineering100→추가표본→다른 wrapper의 반복을 다음 성과로 삼지 않는다. 초기 연구 배분50% 성능 가설/30% serving/20% 라벨 감사는 강제 할당량이 아니지만 최근 편중을 판단하는 기준으로 유효하다.

## 세 가설

|가설|구체 개입과 기존 자산 재사용|검증 가능한 대조와 재개 근거|예산·한계|
|---|---|---|---|
|**A. 더 충분한 대회 적응 학습이 현재 B8의 표현을 개선한다**|224px·동일창·동일라벨·동일모델을 유지하고 고정8epoch 이후 사전 고정된 추가8epoch 적응을 비교한다. 추가 epoch에서 backbone도 학습한다. generic ImageNet 초기값은 재사용 가능. EXP084의 T-only B8은 V 제외 계보가 유지되면 시작점 후보지만, optimizer/scheduler 상태 보존 여부는 아직 확인하지 않았다.|B8 대 B8+고정 continuation8. prefix·optimizer 상태를 이어받으면 학습량 효과에 가깝고, weights만 복원해 optimizer를 재시작하면 **추가적응+재시작 레시피 효과**라고 명명한다. 기존224/384 비교는8/12epoch도 달라 해상도나 학습량의 단독 효과를 검증하지 못했다. B8이 수렴했다거나 underfit이라고 확인된 것도 아니다. 새로운16epoch cosine가 첫8epoch LR까지 바꾸면 기존B8과 ‘학습량만’ 다르다고 하지 않는다.|CPU15–30분으로 기존 학습곡선·update/skip·state 가용성만 판정 후 종료. 추가B8은 과거B8 실측을 참조한 **약40–46GPU분 추정**. 새 공통trajectory가 필요하면16epoch 약80–92분 추정, 실제 속도로cap설정. 곡선·수치가 추가적응 근거를 주지 않으면 자동확대하지 않는다. 부정 결과가 DINO/의료사전학습 등 모든 표현 전략을 부정하지는 않는다.|
|**B. series 안의 상대위치·순서 정보가 현재 MIL이 놓치는 관계를 준다 — 우선 제안**|각triplet의 slot 내부 상대중심위치(0–1)를 이용하는 작은 relative-position attention을 소견별 pooling 앞에 추가한다. 이는 **실제 mm 좌표가 아니라 캐시에서 복원 가능한 상대index**다. 비교군과 처리군 모두 같은 새 attention 구조·파라미터 수·slot mask를 사용하고, 비교군의 위치입력만 상수로 고정한다. 픽셀 집합·정규화·학습 창수·평가K24·224px를 유지하고, 두 arm에서 backbone 전체를 같은8epoch fine-tune한다. EXP084 T-only B checkpoint를 두 arm 공통 시작점으로 재사용할 수 있다. 새head 초기값·optimizer 재시작도 두arm에서 일치시킨다.|고정 **위치정보 없음 대 있음**. 한 group V·원 reference라벨·고정epoch/seed·같은90/10을 쓰되 primary는 전체12-target macro; 양성수·골절·group·조건부CI도 함께 기록한다. 기존 B8 성적을 함께 보존해 둘 다 B8보다 낮은데 처리−대조만 양성이라는 이유로 채택하지 않는다. EXP083의 실패는 frozen encoder의 특정 slot-bias head/refit이고, 위치에 따라 달라지는 관계를 end-to-end 학습한 시험이 아니다. EXP081은 고정weights의 한coronal-slot max-span 직접교체였으며 순서정보 학습도 아니다. 따라서 이 가설은 두 음성결과를 지우지 않고도 별도로 물을 수 있다.|CPU 구현·계약검사45–90분, 두B8 규모 fine-tune은 **80–92GPU분 추정**(변경head overhead 미실측). K/해상도/encoder 크기를 동시에 늘리지 않고 한 비교만. 위치가 scanner/protocol의 대리변수가 될 수 있어 group 편차를 본다. 새head가 정보보다 최적화 부담만 더할 수 있다. B8 초기표현에 이미 정보가 소실됐을 가능성도 남는다. 이 대조 실패가 full3D·다른 pretrained representation의 실패는 아니다.|
|**C. 대회 정의에 맞는 감독 신호가 추출기 불일치보다 중요하다**|Synovitis 직접서술 대 effusion/plica/Hoffa 대리표현, 또는 PFOA 깊이·범위 정의처럼 이미 특정된 문제 중 **영상으로 확인 가능한 하나**를 고른다. 기존 Dread/Qwen은 후보 층화용이며 어느쪽도 정답으로 간주하지 않는다. 전문가 영상 판독을 source/예측 가림으로 얻고, 확인된 범위에만 train target/셀 loss 개입을 제안한다. mislabel과 unknown/unmentioned를 분리한다. 기존 V-excluded image 초기값과 평가 코드를 재사용할 수 있지만, V를 본 backbone은 재사용 불가다.|고정 원감독 대 근거가 있는 제한적 개입, 평가는 **개입한 label과 다른 frozen reference**. 임상 참값을 주장하려면 판독된 V뿐 아니라 관련 ancestor의 V 제외도 필요하다. PFOA20의 합의음성9건에서 Dread 이진양성0인 결과는 자동교정 근거 부족이고, Qwen4건 불일치가 Dread 오류를 뜻하지 않는다. MCL120·Synovitis16의 직접교정 근거 부족(root 결정)을 유지한다. 새로운 명시적 Dread 정의충돌 또는 독립영상근거가 재개 조건이며, 단순텍스트 표본 확대는 아니다.|먼저 사용범위/계보/판독 프로토콜 정리 CPU30–60분. 전문가 가용시간·표본수·비용은 **미확정**이므로 임의 견적/유료사용 없음. 공식 규칙의 train 파생판독 허용범위 확인이 선행이다. 근거가 생긴 뒤 작은 RN2fit은 과거계획26–36GPU분, 새B 포함 blend대조는66–82분 정도의 계획값이며 판독비용과 별개. 소견당 양성수가 적으면 AUC 불가/큰불확실성;58건급 표본으로 .0005 효과를 판별할 것으로 기대하지 않는다.|

## 비교를 해석할 경계

1. **0.943는 공개 스택의 성과다.** DINOv2 20member, A5 DINOv3 5fold, Rad 계열, Raptor와 추가CoAt, learned stacker가 섞인다. B/RN은 ImageNet 초기화 후 대회 Silver로 학습한 작은 자체 조합이다. 복잡성이나 member 수만으로 어느쪽이 우월하다고 결론낼 수 없지만, pos_weight 하나의 변화가 이 표현·학습량·입력 다양성 차이를 해소한다고 보지도 않는다. 기존 공개 DINO/CoAt competition weights를 새V에 적용한 예측은 계보 확인 없이 독립 baseline이 아니다.
2. **새 grouped 대조는 정확한 anchor 검증이 아니다.** anchor residual CoAt은 Gold58로 epoch4/6/8을 선택했고 다른 arm/88-feature stacker 계보도 완전히 확정되지 않았다. R2는 teacher 경유 노출·혼합라벨 epoch선택 문제가 별도다. 반면 EXP078–080의 ImageNet 초기화 모델을 R2와 동일하게 취급하지 않는다. 앞으로 EXP084 T-only B를 시작점으로 재사용한다면 그V는 gradient상 미사용이지만, 결과를 본 후의 다음 가설은 같은 개발V에 대한 후속선택이므로 새 독립 임상 확인이 아니다.
3. **학습량·정보량·정답 품질은 서로 다른 질문이다.** A는 같은 영상의 적응량, B는 같은 영상창 사이 관계, C는 감독 의미와 임상 참값을 묻는다. A/B의 weak AUC 개선이 C를 해결하지 않으며, C의 텍스트정의 일치가 영상표현의 충분성을 보장하지 않는다. Report는 학습감독에만 쓰고 테스트추론 입력으로 요구하지 않는다.
4. **anchor로의 전이는 마지막까지 남는다.** standalone B+RN의 실행 성공/weak 향상은 anchor와 보완적이라는 증거가 아니다. 실제 고정후보의 제한된 제출은 전이질문을 하나 확인하는 용도이며 0.943에 개발delta를 더하지 않는다. 사후12소견 routing·비율 탐색으로 검증 공백을 덮지 않는다. 하루5회는 준비된 서로 다른 가설의 기회이지 소진 목표가 아니다.

## root가 다음 하나를 고를 때

EXP084의 frozen 평가와 serving 결과를 먼저 판정한다. 큰 구현오류가 없다면 그 결과에 맞춰 pos_weight 주변값을 다시 훑지 말고 **B의 paired sequence-position 대조를 다음 성능 투자로 검토**한다. A는 B 구현비용이 예산을 넘거나 기존 학습로그가 명백한 미수렴/중단을 드러낼 때 더 싼 대안이다. C는 독립 판독·명시적 정의충돌이라는 새 근거가 생기면 A/B보다 먼저 갈 수 있다. 새학습을 시작하기 전에 root가 개입·T/V 계보·primary·고정epoch·최대시간을 하나의 계약으로 선택하며, 이 문서는 그 등록을 수행하지 않았다.

Biohub 누락 점검은 ‘큰모델/더많은data가 정답’이라는 결론이 아니다. 작은 양의 효과를 버리지 않으면서, 특정 head/selector 실패를 넓은 표현/sequence 전략의 실패로 확대하지 않고, 완료 가능한 작은질문만 반복하다 큰정보공백을 놓치지 않는 데 적용했다.

## 읽은 근거와 기록

본문의 수치·사례는 과거 기록이며 현재LB/미완료EXP084 결과로 업데이트하지 않았다. 아래5개가 이번에 본 문서 전부다. root의00:35 투자결정은 앞서 읽은 SHA `4cf8f39ea164b9cc5f3e04c752c0a659b3f2c6074d8a916cdecae46de23514ed` 내용을 재사용했으며 이번에 다시 열지 않았다.

- [route-to-first](../../../../research/route-to-first-2026-10-03.md), SHA `e1d59ccfd3f35742abc39a7b55395597d373a94421331cf3badb4fa4662943bc`.
- [6시간 재평가](../../../../research/six-hour-reassessment-2026-10-03.md), SHA `57c8dcc5e67615d1259471d7617471eb81064b4e8bfc7b79284cf9f02b19eb09`.
- [anchor 분석](../../../../research/anchor-notebook-study-2026-10-03.md), SHA `578807af1808cd0c9584777fdb220cb6ecd752c3b8eab5b1cc9708a517f3c3bb`.
- [Silver 외부연구·PFOA](../../../../research/silver-other-research-and-pfoa-2026-10-03.md), SHA `6b2110349292f04b779892fa28b03889ce54dfb6ac112202f76c9af2aa7dd17d`.
- 연구실API `source id=local:workspace://docs/research/gold-exposure-validation-plan-2026-10-03.md`, 저장revision `0522e7bad2d9429bf69ca1716339d7d6da5b79437d6f6e3c4ebeed4fead7ff08`, content SHA `ea086a1a61dd9f46d9e1e28452b60d307f5e79fb2d0b044b65fbbdda495d212d`, observed_at `2026-10-03T13:21:43.429182+00:00`. 이번01:31 KST 로컬API read이며 Kaggle 재수집/최신 revision 확인이 아니다. 그 문서의 참가자 토론도 자기보고로 구분했다.

운영: API 첫 검색의 `body=false`는 허용 enum이 아니어서422였고, 인수를 제거해 읽었다. 이후UTF8 출력으로 동일저장revision 본문을 확인했다. DB 생성/변경·외부수집·다운로드·모델/원DICOM/라벨원자자료 로드·코드/중앙문서 변경·실험등록/학습/제출/지출0. 산출물은 이brief 하나이며 별도장시간작업·heartbeat 없음. root가 중앙handoff와 다음선택을 담당한다.
