# EXP087 scale 결과 독립 검토 · 2026-10-04 08:40 KST

- **PASS: 동결315 입력 stage 완료 수용. 학습·효능GO는 아니다.** [상세](result-independent-review.json), [신규307 배열검사](result-independent-new-array-checks.json), [root17개검사](root-stage-result-review.json).
- 독립담당: 신규307NPY 전부uint8[12,336,336],fileSHA·arrayhash·재계산mask가manifest와일치한다. 파일집합·경로·native12index·유효triplet·entry/fallback연결에문제0. 기존8배열은다시읽지않고완료PASS를재사용했다.
- manifest/fixture/receipt의315 UID집합·순서·G1소속·reason/reuse표시가같다. 기존8entry는동결내용과동일하다. manifest실제SHA는root검증을재사용했고receipt가그SHA를참조함을확인했다.
- 새uint8payload415,908,864B가실행receipt/cap과정확히같다. NPY총415,948,160B중39,296B는형식header다. 신규307과재사용8 모두fallback없음은고정map의관측이며광범위가용률이나임상정보증거가아니다.
- root담당종료·자원검사17/17 PASS를재사용했다:57.797초/RSS704,397,312B,child0/cleanup0/FAIL0/stderr2개0. parent35780/child1472는08:37:04부재. headers15,774/parser24,300,756B/pixels3,684/member3,046,922,358B.
- root분할집계: T252의epoch별예정치환[218,226,214,218,218,198,215,235]합1,742, V63의eval126. 독립검토는각row의9schedule구조와범위만확인했다. 집계는T/train·V/eval만취하며다른mode는가상일정이다.
- 예정치환수는실제resized tensor차이·예측·drop_last후노출·성능이득이아니다. 원307 DICOM replay·전체환자독립·Windows학습통합·B8 GPU정밀도·private/T4는검증범위밖이다.
- 원cache/ZIP/header/decoder/model/GPU/label/예측·원8/source/hashsweep·census/테스트재실행0. 진행중인B8연동계획검토도중복하지않았다. 다음연동/학습범위는root가별도결정한다.
