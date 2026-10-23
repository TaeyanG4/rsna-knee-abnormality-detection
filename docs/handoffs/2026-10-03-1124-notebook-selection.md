# 2026-10-03 11:24 KST · 출발 노트북 재선정

- 상태: 완료(조사·문서). 기록자: Codex.
- 요청·결정: 공유0.943 노트북이 최선의 출발점인지 재조사. **Jiweiliu v11/우리0.943 제출본 유지**, 후보 추가·특정 arm 교체로 개선. Raptor 전면교체는 전제하지 않는다.
- 결과: [비교 보고서](../research/notebook-starting-point-2026-10-03.md). 앱 노트북242개 중 점수상위200·최근40 목록 확인, 후보12개 최신 소스 재수집 성공. 로컬 anchor와 최신 Jiweiliu 코드23/23 일치. 공개0.946 D4·0.945 Lite/S75는 별도 모델 자산 접근 미확인/실패. Skarin은 Fracture routing 한 줄 차이, Roman은 기존 DINO/CoAt 신호 재활용 후보. 최신2개는 공개테스트3건 고정 예측/행교환이라 제외.
- 검증·근거: [메타/hash](evidence/2026-10/2026-10-03-notebook-selection/sources.json)·[코드 비교](evidence/2026-10/2026-10-03-notebook-selection/code-comparisons.json)·[자산 API](evidence/2026-10/2026-10-03-notebook-selection/asset-access.json). 숫자best public score와 최신 코드 점수, sample runtime과 full runtime을 구분. 문서3개 앱 등록·API 내용 일치, 링크/분량/hash 검사. Kaggle 검색 MCP 인증 실패는 기존 앱·로컬 인증 adapter로 보완. tzdata 부재는 KST UTC+9로 처리, 설치 없음.
- 변경: 전체 계획·ADR의 기준본과 보완 모델 우선순위를 명시. [변경 전 원문](archive/snapshots/2026-10-03-notebook-selection/manifest.json) 보존. INDEX·월별 목록 갱신. 새 학습·GPU추론·원격 실행·대용량 다운로드·제출 없음.
- 다음: ①기준본과 단계별 출력·reader 수 고정 ②보유 자체 후보의 계약·오류 보완성 확인 ③후보1개 고정 blend/단일 routing ablation 패키지. 전체 계획의 nested 평가 보완은 유지.
