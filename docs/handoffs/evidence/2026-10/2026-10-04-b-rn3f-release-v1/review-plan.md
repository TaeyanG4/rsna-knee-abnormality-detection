# B/RN release 독립 acceptance 검토계획

2026-10-04 03:39 KST · `/root/anchor_inputs`. 상태: 계약 검토 완료, 새 구현 검토 대기. [root CPU 결정](root-cpu-decision.json), [next-route](../2026-10-04-b-rn3f-t4-trial-v1/next-route.md), [실제100 검토](../2026-10-04-b-rn3f-t4-trial-v1/root-result-review.json) 재사용. 새 GPU·원격조회·테스트·데이터/모델실행 없음. M04/M07–10 적용; 정확도·hidden PASS 아님.

## 구현 인수 조건

1. **출처:** 새 driver·release manifest가 원본 asset marker·exact6·decoder/runtime/ensemble/집계 소스와 신규 파일/fixture hash를 묶는다. frozen CLI의 public3/100 제한을 monkeypatch하지 않는다. root가 허용한 test-only driver만 추가한다. 이전 engineering PASS·FAIL은 불변이다.
2. **test 계약:** root 발견은 `test.csv`, `test_series.csv`, `test_series/` 및 제공된 공식 sample만 요구한다. train/Report 요구 금지. 문자열 UID의 안전성·중복·누락·extra를 검사한다. sample의 공식 13열 및 UID집합/순서를 대조하고 필요한 순서 정렬을 명시한다. sample 값을 예측으로 사용하지 않는다. N은 실제 입력에서 얻는다.
3. **집계:** 모든 N·6member raw float32가 완료된 뒤 frozen average-tie rank→fold별90/10→3fold 평균. shard/local/소견별 재rank나 비율변경 없음. final의 finite/range/12target/UID순서와 raw 재구성 일치·hash를 확인한다. public3 rank를100 rank값에 대조하지 않는다.
4. **시간:** 첫 notebook cell monotonic부터 public N3 600초/full32100초. 공개3 외부세션900초는 별도다. 준비·offline설치·fixture·worker·cleanup·검증·publish를 포함하고 단계마다 clock을 새로 시작하지 않는다. `process_resource_receipt.py:234`는 exit0 확인이 timeout보다 먼저이므로 반환 후 absolute deadline/elapsed도 검사한다. 늦은 COMPLETE·만료 후 publish를 거부한다.
5. **비용:** 고정 `1.3*(877.354403408*N/100)+300 <=30600`.1800초는 readiness 여유,32100초는9h에서300초를 뺀 실행 deadline이다. CPU replay 포함1107.604초로 투영하거나 RSS100을 N배 하지 않는다. 실제 public3 비용은 별도 기록하며 사후 계수 완화 없음.
6. **종료/게시:** worker는 작업폴더만 쓴다. parent가 resource COMPLETE/exit0/clean cleanup/실측 telemetry/OOM 없음·원본/출력 SHA·전체 raw/CSV 계약을 확인한 뒤 같은 volume 임시파일을 final로 atomic rename한다. final존재/hash와 receipt 상태를 함께 확인해 rename 전 COMPLETE만으로 성공하지 않는다. 실패/timeout에 최종파일이 없어야 하며 사전존재 파일은 덮어쓰거나 삭제하지 않는다. 실패근거 보존, retry0.

## 기존 작은 codec 자료 확인

- source: `C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/Lib/site-packages/pydicom/data/test_files/`. `SC_rgb_jpeg_gdcm.dcm`5204B, `MR_small_jp2klossless.dcm`6008B, `MR_small.dcm`9830B, `JPEG2000.dcm`3308B, 합24,350B. 이번에는 bytes/hash만 확인했다.
- 압축3개 file SHA는 [기존 codec 보고](../../../../../.local/d01-codec-staging/20261003-v1/preflight-v2/output/d01-codec-check.json)와 모두 일치. 보고 SHA `eb26515f89b9be3d7f6f33c9667cbe6e079227b4f7b67ee078a3581f087ac8d1`. MR_small SHA `3f27d1c22f1a66e80d7bb7c911e8610fd0bb70325a76746a7adb1c0ddefcf2bb`.
- 최종환경에서 pinned pydicom3.0.2/gdcm3.2.6/OpenCV4.12.0 및 offline wheel hash를 검사한다. producer와 같은 기본 `pixel_array` 경로로 fixture syntax/shape/dtype/pixel SHA, MR lossless↔uncompressed exact를 확인한다. import/decoder.available만으로 actual decode PASS 금지.
- 기존 fixture는 .70/.90/.91이며 .57은 availability만 확인한 범위다. RGB JPEG lossless 한 건이 모든 compressed MRI를 대표하지 않는다. broad header 조사/다운로드 없이 이 한계를 남긴다.

## 새 코드가 준비된 뒤 한 번 확인

focused CPU 반례: N1/3/100/>100·ties·UID재정렬/누락/중복·sample schema 오류·부분member·test-only mount·시간 경계/늦은exit0·cleanup실패/telemetry부재·rename실패와 사전final 보존. 기존31/37검사·local100·T4100 반복 없음. 코드 검토는 builder 알림 후, polling 없음. 이번 범위에서 계획 자체의 추가 blocker는 발견하지 못했다. builder 구현·최종환경 디코딩·public3는 아직 검증하지 않았다.
