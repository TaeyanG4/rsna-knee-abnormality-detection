# 2026-10-03 13:34 KST · 우선순위 실행·입력 A/B 시작

- 상태: 진행. 기록자: Codex. 로컬 소규모 대조 승인. 제출·공개·유료는 별도 승인. 첫 ETA20분 전 확인, 미완료면 갱신 ETA 재예약.
- 판단: [토론16개·우선순위](../research/execution-priorities-2026-10-03.md). 당일 수집본 대비 변경0.0.943 보존,nested/384 확대 보류. weak/Gold 개선의 LB 전이 반례를 반영해 비교 조건→실전 입력→영상A/B 순서.
- 구현: strict AMP/백엔드·nativeBF16·finite/UID/hash,norm/window,stable RNG. 학습/eval라벨 분리·fixed epoch·Gold선택 금지·optimizer/skip 기록. DICOM adapter/serve,큐 실패 뒤 무한대기 수정. [사용법](../runbooks/training-raptor.md).
- 검사: CPU44+평가도구13=57통과. 첫FP16 smoke overflow실패; init-scale1024·skip/scheduler 수정 후 A/B wiring35/34초 성공(24건·1epoch,각11update/1skip). 성능근거 아님. [검사/hash](evidence/2026-10/2026-10-03-priority-execution/verification.json)·[smoke](evidence/2026-10/2026-10-03-priority-execution/smoke-verification.json).
- 실데이터: test3 DICOM/cache 입력hash·예측 완전일치. train100 중99 volume/mask일치·1 CRC실패; 별도zero study는header오류. 표준ZIP 재현,원본보존·두arm 공통격리. [증거](../../artifacts/predictions/serving-20261003/cpu-parity100.json).100전체통과 아님.
- 실행: EXP078 A_F0→B_F0→A_F1→B_F1 총4fit.224px/global-valid vs within-slot-valid,원라벨·seed42·strictFP16·fixed8epoch. train3,474/eval869씩. [명세](../../artifacts/experiments/EXP078_INPUT_AB_PROTOCOL/manifest.json). 첫provenance 사전code/초기weights/UID/hash 일치. 점수 아직 없음.
- 13:27 시작,queue PID20764·학습19936·wrapper21180. 13:36 ep0완료402초/skip0. 첫fit ETA14:24,전체17:15 전후(3~5h범위). [로그](../../artifacts/logs/EXP078_INPUT_A_F0/run.log)·[실행/예약](evidence/2026-10/2026-10-03-priority-execution/launch.json).
- heartbeat `exp078-a-b` ACTIVE,같은채팅14:04 재개. 미완료면 갱신ETA 예약 후 반환.4fit SUCCESS 후 input_ab_report로 계약·fold/target/group CI·skip 판정,기록 후 예약해제. 실패 시 자동재학습 금지. 중단은 실제PID/명령줄 재확인 후 큐러너부터,학습은 별도.
- 남음: 신규실제T4/전체스택 parity,5fold decode공유·hidden전체시간·부분손상 정책. 개발fold/보고서라벨/노출Gold로 임상·private 향상 주장 금지.4fit 이후 확대 판단.
- 문서/AGENTS 갱신,보고서4개 앱등록·API/revision/added_at 확인. [등록](evidence/2026-10/2026-10-03-priority-execution/app-registration.json). PC/앱 실행 필요. [변경전](archive/snapshots/2026-10-03-execution/manifest.json).
