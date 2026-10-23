# 2026-10-03 12:47 KST · 검증용 추가 모델의 효율성

- 상태: 완료(검토). 기록자: Codex.
- 요청·판단: 다른 세션의 자체 학습이 현재 효율적인지 재검토. 최초 Raptor는 통제 가능한 학습법/arm 개선 기반이라는 목적에서 타당했다. 그러나 새 모델 자체는 새 정답·독립 검증셋을 만들지 않으므로 ‘anchor 검증 불가’만을 이유로 기본 모델을 추가하는 것은 낮은 우선순위다.
- 근거: [최초 학습](2026-10-02-1310-raptor-baseline.md)·[선정 목적](2026-10-02-1555-baseline-decision.md)·[ResNet/R2 전환](2026-10-02-2100-anchor-swap-submit.md)·[R2 과정 감사](2026-10-03-1212-process-review.md). 실제 R1·ResNet34 각5fold의 summary·best·OOF 파일10세트 확인: timm 초기화, 동일 split, fold당869행 기록, OOF epoch 선택. [관측/hash](evidence/2026-10/2026-10-03-training-efficiency/existing-models.json). GPU 재평가는 하지 않았다.
- 결과: [계획](../research/route-to-first-2026-10-03.md)에 목적·비용 구분 추가. 기존 OOF는 개발 진단에 재사용하되 독립 확증용으로 취급하지 않는다. nested P2는 R2 학습법의 누출 수리이며0.943/혼합 앙상블·임상 정답 검증이 아니다. fixed/inner epoch 계약에 기존 baseline이 맞지 않으면 대조군 재학습도 필요하다.
- 다음: ①기존 후보1개의 hidden-DICOM serving·계약·runtime 연결 ②라벨 혼합 또는 영상 입력 중 가설1개 선택 ③필요한 작은 대조만 학습, 이득 근거 후 확대. 보고서 weak AUC gate만으로 임상 교정 성패를 정하지 않는다.
- 실행 범위: 12:45 로컬 raptor_train/gpu_queue/queue_resume 대상 Python 프로세스 없음. 최신 다른 세션 기록은10/03 06:00이며 이후 새 학습은 확인 못 함; 원격/다른 이름의 전체 실행 부재를 뜻하지 않는다. briefing·문서 링크/분량 확인, 수정 계획 Store 반영·앱 API 내용 일치. 신규 학습·중단·제출·메시지 전송 없음.
- 12:51 시간 문의: summary17개 실측 재확인([시간/hash](evidence/2026-10/2026-10-03-training-efficiency/time-estimates.json)). 로컬224 8epoch 약41~42분/fold,384 12epoch 3.15~3.69h/fold. ResNet T4 40~50분, R2 T4 2.21~2.56h이므로 원격 병렬이 자동으로 더 빠른 것은 아니다.
- 계획 예산(실측 아님): 기존 OOF 분석1~2h·CPU5~20분; 후보 serving 구현3~6h+소표본/parity/runtime1~3h; 입력 A/B 구현3~6h+GPU3~5h(2outer×2fit); nested 구현4~8h+GPU6~10h(teacher6+student4fit, fixed epoch). ResNet 로컬 시간 미측정. 전체0.943의2T4 실행은 별도 계측, 업로드·원격 대기·채점 제외. 연결/분석+두 대조 중 하나까지1~3일 예상, 최초 계측 후 갱신. 학습은 백그라운드·완료 알림 방식이며 이번 실행 없음.
