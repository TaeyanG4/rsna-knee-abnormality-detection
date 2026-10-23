# 2026-10-03 20:18 KST · 환자/유사영상 검증 공백·투자 감사

- 상태: 완료(읽기 전용 검토). 기록자: Codex / precision_code_review.
- 요청·결정: 추가 성능연구 전 기존 환자·near-duplicate 근거와15분 이내 검사1개, 감독/표현/학습 후보 비용 비교. 기존 연구·점수 재실행 금지.
- 결과·파일: report.md, inventory.json, candidate-protocol.json, research-api.json. report/Series UID 중복0은 환자 독립성이 아니다. scanner fingerprint133개도 환자ID가 아니다. 기존 study_meta는 PatientID를 수집하지 않아 원본 필드 존재 여부가 남는다.
- 검증·근거: 외부 DINO 특징60.92MB/4,407행의 UID순서가fold index와 일치. 제외2건 후4,343건 모두3개 이상slot 유효. 원본/전체픽셀 없이 후보 검색 가능. 특징 manifest의 과거 checkpoint/code SHA 누락은 한계. localhost API 수집자료revision·파일hash 보존, 학습/추론/라벨/점수/픽셀 읽기 없음.
- 다음: 기존특징 cross-fold exact/nearest 후보검색10~15분 상한 초안(미실행). 유사도를 환자/픽셀 중복으로 확정하지 않고 후보가 있으면 별도 가림 확인. GPU 첫 후보는 RN pos_weight=1만 바꾸는2fit25~35분; slot-aware head는그다음, MCL unknown 감독은변경근거부터. 비용은추정과과거실측을구분. 공통STATE/인계 변경 없음.
