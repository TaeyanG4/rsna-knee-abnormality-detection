# 2026-10-03 22:50 KST · Silver 저장 필드 감사
- 상태: 완료. 기록자: Codex 하위 에이전트. 중앙 문서는 root 담당.
- 요청·결정: root-scope 고정, EXP083 train3475 UID만. 원본 불변, CPU 1회 집계. Report/qwen_raw 판독·출력, Gold/F2값/예측/LB, 학습·GPU·다운로드 없음.
- 결과·파일: [registration.json](registration.json)→[result.json](result.json). raw CSV 4407건 UID 검사 후 train3475만 필드 접근, 나머지932 폐기. 커버리지3475/3475, 중복·parse오류·필드결측/unknown0, status ok3475. 12×4 필드 모두 유효형식.
- 핵심: mentioned0은 Synovitis2453(70.59%), Fracture2371(68.23%), LateralOA1888(54.33%), Contusion1606(46.22%), PFOA1388(39.94%). mentioned0/negated1+label1은 모두0. Effusion label1+P≤.5는647/1814. 이는 의미검토 후보이며 정답 오류 확정 아님.
- inventory: [inventory-review.json](inventory-review.json), 23파일 헤더/스키마만. Qwen raw/filled는4필드, zaidaliiq는mentioned/conf/sev, pilkwangv2는conf/verdict. 현재 Dread는UID+12soft만. juanicarlos 무확장 파일은 CAPTCHA HTML이며 라벨 아님.
- 검증·근거: [audit.py](audit.py) prepare 후 analyze 단1회(0.390초). 사전 AST+합성9검사, 사후 모든 target 합계·hash검사 PASS([verification.json](verification.json)). CSV tokenizer와 SHA는 전체바이트를 통과하나 Report 열을 선택·판독·출력하지 않음.
- 출처 SHA: d2740bc88904ba1e604c74e66a0439d37228fdd2f9bd40bb98b9b4ce7e90456f.
- 남은 일·다음: root의 notebook 의미감사와 결합해 Effusion/Synovitis 중 작은 가림검토 대상을 결정. Qwen 상태를 현재 Dread 라벨의 정답·mask로 직접 이전 금지. 의미정확도·보정품질 미검증.
