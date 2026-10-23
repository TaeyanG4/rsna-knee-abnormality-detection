# 2026-10-03 15:25 KST · DICOM 헤더 I/O 대조

- 상태: 로컬 검증 완료·원격 CPU 검사 준비. 기록자: Codex 하위 작업. 루트가 지속연구 인계/INDEX에 통합한다.
- 변경: `DirectoryDicomSource(header_only=True)` 명시 opt-in. 기본값·ZIP full-read·정렬/spacing·선택 픽셀 SHA/실패처리 유지. `header_io`는 pydicom에 반환한 헤더 바이트/시간이며 물리 디스크 I/O가 아니다. 기존 모델자산·원본·동결 학습4파일 수정 없음.
- 검증: CPU 36검사 통과 후 참조 UID 누락 방어 추가, 해당11검사 통과. 고정100건의 volume/mask/전체receipt가 구버전 및 실제 T4 runtime100과 모두 동일. [증거](dicom-header-io-review.json). 실행 helper 원문은 local 결과 폴더 보존.
- 결과: 논리 읽기13.343→3.730GB(72.05% 감소),헤더15,382개23.57MB. 로컬 warm-cache 렌더85.29→87.39초(2.47% 느림). 원격 속도 개선으로 주장하지 않는다. 로컬 작업PID23308은15:21:01완료,임시 staging9.64GB 보존.
- 다음: `workspace://.local/d01-header-io-staging/20261003-v1` 비공개 CPU notebook628,887bytes/AST·payload SHA검사 완료,계정 쓰기 미실행. 기존1.469GB자산+codec 재사용. 루트가 실행·회수 후 I/O 투자 판단; 실제5모델T4/anchor9h공동시간은 별도 확인.
- 누락점검: 실측 decode 병목에 한정한 비용절감 가설이며 AUC 개선 근거 아님. warm-cache 음의 결과도 보존하고, 성능 연구(MCL/PFOA/EXP078)를 밀어내는 추가 도구작업은 자동확대하지 않는다.
