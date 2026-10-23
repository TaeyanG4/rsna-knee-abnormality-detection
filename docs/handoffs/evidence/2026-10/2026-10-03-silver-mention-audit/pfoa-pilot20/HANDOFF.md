# 2026-10-03 23:07 KST · PFOA 가림20 packet 준비
- 상태: 완료. 기록자: Codex 하위 에이전트. 리뷰·라벨unblind 미시작, 중앙기록 root 담당.
- 요청·결정: 원 pfoa-pilot-scope 불변, 새 registration에23:02시작/23:10목표/23:12상한 기록. train3475만, seed·6/6/4/4·highgrade우선 고정, component당1.
- 결과·파일: [blinded-packet.json](blinded-packet.json)은 case_id와 full Report만20건. UID/stratum/component는 sealed/case-map.json에 분리. 라벨·예측 없음. highgrade6/mild6/unspecified4/absent4, uniquecomponent20.
- 검증·근거: [registration.json](registration.json) 후 [prepare_packet.py](prepare_packet.py) build 단1회0.203초. 사전 합성10문장+결정성/그룹/우선순위3검사. 사후 packet 스키마·원문hash·UID/subset/exclusion·component 검사 PASS([verification.json](verification.json)). 원 train.csv4407건 UID필터 후3475 Report만; 제외932 Report/label 접근없음. 후보 highgrade221/mild189/unspecified1296/absent1769.
- 한계: strata는 같은 구두점구간의 PF+cartilage+severity 어휘선택이며 임상판정 아님. 부정 highgrade도 어휘bucket에 포함, patellar tendon은unspecified 가능. 표적표본으로 유병률 추정불가. Gold제외는 기존 immutable train계약 계승, 환자독립성 증명 아님.
- packet SHA: d1898d9d51e6c2df0693095b574d53a8443de5673657f17a29b6ff3ad80c564e.
- 남은 일·다음: root가 hash/계약 확인 후 별도 가림검토. reviewer는 annotation hash 동결 전 sealed 접근금지. 이 builder는 Report 의미판독·라벨비교·학습 수행 안함.
