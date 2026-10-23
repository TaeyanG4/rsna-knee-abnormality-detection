# 2026-10-02 · 폴더 탐색·EDA·슬라이스 버퍼

상태: 구현 완료. 기록자: Codex. 아래는 같은 날 작성된 상세 인계의 압축본이다.

- 요청: 왼쪽 폴더 탐색, 전체 현황·영상 품질·비교 도구, 슬라이스 로딩 끊김 개선.
- 결과: 원본/캐시 study→시리즈→파일 트리·같은 study 세션 재사용. 라벨/촬영/결측/UID 범위 EDA, 최대64장 품질·기하 검사, 같은 UID 원본↔캐시 비교, crop/해상도/정규화 PNG·JSON. 전체 픽셀 스캔·정합/자동 진단 아님.
- 전환: 기존 frame=null 초기화와 응답 후250ms 대기를 제거. 인접6장 prefetch+PNG decode,24장/추정32MiB/15초TTL. 기존 영상/표시# 유지, 준비 후 교체,700ms 뒤 작은 안내. 재생250ms tick. 파일 변경 감지에는 TTL 한계가 있으며 새 읽기/검토 저장은 서버 서명 검사 유지.
- 변경: `monitoring/kneeops/viewer.py`, `viewer_api.py`, `viewer_eda.py`; `monitoring/web/src/ViewerPage.tsx`, `ViewerTree.tsx`, `ViewerEda.tsx`, `ViewerCompare.tsx`, `viewer-eda.css`, `viewer-frame-cache.ts`, `main.tsx`; 관련 테스트와 사용 문서.
- 검증: 전체 Python68, 브라우저/버퍼15+960px 추가1 통과, TypeScript/Vite 성공. 300ms 지연에도 canvas 유지. 960px 표시 간격307/254/239/259/255/238/257ms. 모바일 overflow와 오류fallback 수정 후 통과. 최초960px 검사는 짧게 지나간 특정#를 놓쳐 관측 방식 수정 후 통과.
- 실제 데이터: API약0.50초, train4407/test3/라벨58/캐시2, 준비5study815DICOM. 사용자 탭에서도 원본MRI와 #19 전환 확인. 단일 관측이며 전체 원본 무결성/학습 성능 증명 아님.
- 운영: active수집/다운로드 없음 확인 후 stop. PID3128 종료대기 정지로 조기start 실패1회. 명령줄 확인 후 해당PID 종료, PID41432로 복구(당시값). launcher 자체 수정 없음. 재현 시 Windows uvicorn/proactor 조사.
- 미실행: 전체MRI CRC/픽셀 감사·환자 단위 누출 검증·GPU학습·제출·새 다운로드·Docker 재빌드. 다음:20~50study 시각검토 후 동일 계약의 입력 ablation 설계.

[사용법·API](../runbooks/image-viewer.md) · [압축 전 원문: 명령·실패·화면·운영 상세](archive/snapshots/2026-10-02/docs__handoffs__2026-10-02-viewer-eda-buffering.md.txt) · [원문 SHA256](archive/snapshots/2026-10-02/manifest.json). 화면 증거는 원문에 기재된 `.local/ui-*.png`. 원문 hash는 보존본의 값이며 현재 요약의 hash가 아니다.
