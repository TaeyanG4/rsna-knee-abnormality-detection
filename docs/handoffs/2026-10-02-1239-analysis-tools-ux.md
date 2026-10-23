# 2026-10-02 12:39 KST · 종료 멈춤 수정·분석 도구·UI/UX

- 상태: 완료. 기록자: Claude.
- 요청: 서버 종료 멈춤과 다른 문제·부족한 분석 도구 확인 후 개선, 전체 UI/UX 개선.
- 종료 멈춤 원인: CPython 3.12.6 Windows Proactor가 이미 reset된 소켓을 닫을 때 `sock.shutdown()` 예외로 `Server._detach()`를 건너뜀 → uvicorn 0.54 `wait_closed()` 무한 대기, 프로세스가 lock 유지. 로그에 ConnectionResetError가 있던 3회(PID 3128·41432·11828)만 멈췄다. 수정: [serving.py](../../monitoring/kneeops/serving.py) selector loop + graceful 10초 상한, 오래된 stop 파일 정리. 재발 시: active 작업 확인 → 명령줄 확인 → 해당 PID만 종료.
- 발견·수정: CSP가 KaTeX data: 폰트 차단(font-src 추가). 라이브러리가 과거 대회 자료를 맨 위에 보여 현재 대회 기본·정렬·본문 필터·추천/댓글/실행일/상태 열로 변경. 탭 수를 현재 대회 기준으로. 모바일 표 가로 넘침·헤더 버튼 공간.
- 새 도구: `predictions.py`+예측 비교 화면(소견별 AUC·Spearman·동일 가중 blend·bootstrap, Gold58/teacher 경고). `source_meta` 테이블, 노트북 첨부 입력(Kaggle 모델 포함 146개) 계보·같은 입력 필터, 두 자료 입력/diff 비교. 실험 검색·필터·정렬, 오늘 남은 제출(CLI submission-limits).
- 검증: pytest 92, Playwright 33, tsc/Vite. 재현 스크립트 80회 reset 후 정지 0.4초. 누수 강제 테스트는 상한 제거 시 실패 확인. live PID 7676 정지 0초 후 15196 재시작, API sync 오류 0(LB 1,364/4,819위, 남은 제출 5). 입력 재수집 247개 오류 0.
- 남은 일: 예측 비교는 진단용이며 가중치 최적화는 넣지 않았다(nested 검증 필요). 노트북 숫자 점수는 공식 API에 없다.
