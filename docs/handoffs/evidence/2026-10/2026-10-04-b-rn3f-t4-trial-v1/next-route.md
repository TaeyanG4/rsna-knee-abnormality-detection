# B/RN engineering100 이후 최소 제출 경로 — 읽기 검토

2026-10-04 03:23 KST, `/root/dicom_header_io`. root의 03:15 RUNNING 관측을 전달받았고 원격 상태/결과를 재조회하지 않았다. **현재 T4 PASS·성능 향상·후보 준비 완료를 주장하지 않는다.** [00:17 감사](../2026-10-04-candidate-readiness/report.md), [완료 CPU 패키지](verification.json)와 동결 소스만 읽었다. 감사/학습/추론/Gold/점수/원격 실행 0. 다음 구현은 정확한 T4 결과·receipt 확인 이후다.

## engineering100은 실제 raw DICOM 경로다

- [producer](../../../../../../tools/b_rn3f_serving.py:459)의 T4 분기는 `train_series.csv`와 `DirectoryDicomSource(train_series)`를 사용한다(:489–498). 로컬 분기만 기존 `Corpus(cache_root)`다. trial이 T4에 `--cache-root`를 넘기지 않으며 producer도 이를 거부한다(:527).
- 동결 `raptor_dicom.py`의 Directory source(:48)는 실제 `.dcm` bytes를 읽고, `_decode`(:206)는 `dataset.pixel_array`·modality LUT·MONOCHROME1 처리를 한다. `prepare_study`(:222)는 정렬/slot 선택→percentile/crop/336 resize→uint8 volume/mask를 만든다. DicomCorpus(:303)는 해당 실행에서 생성한 한 study만 잠시 보관한다. 미리 만든 로컬 픽셀 캐시를 T4에서 불러오는 경로가 아니다.
- volume/mask→within-slot-valid K24→224 resize/ImageNet→동일 입력을 공유하는 exact6 FP16 추론→전체100 rank90/10을 거친다. 이후 [V2 bridge](../../../../../../tools/b_rn3f_thread_input_v2.py:95)가 별도 CPU 프로세스에서 raw DICOM100을 다시 읽고 실제 producer threads로 normalized hash를 연결한다. [paired gate](../../../../../../tools/b_rn3f_parity.py:291)는 로컬 캐시와 T4의 volume/mask/centers exact, host별 analytical 한도, raw6×100×12와 rank를 검사한다. **통과하면 이 고정100의 end-to-end 연결 근거이며 모든 DICOM/hidden 보장은 아니다.**
- 압축 구문은 별도 범위다. 이전 selection의 bounded header 기록은2,354개 모두 Explicit VR Little Endian, compressed 추가UID0이었다. 새 producer receipt는 selected source hash를 남기지만 TransferSyntaxUID를 남기지 않는다. 따라서 현재100 통과만으로 JPEG Lossless/JPEG2000 검증을 주장할 수 없다. 기존 D01 codec fixture/고정 wheel 근거를 재사용하되 새 최종 환경에서의 짧은 fixture 확인을 public3 세션에 합치는 편이 최소 비용이다. 새 대량 header 감사는 필요하지 않다.

## 이미 닫힌 공백과 남은 변경

|범위|재사용할 것|아직 필요한 것|
|---|---|---|
|모델·계산|exact6 weights/기존 private dataset, frozen decoder/runtime/ensemble, 고정 fold별 rank90/10·평균과 모든 raw provenance|새 학습·가중치 교체·anchor 혼합·비율 탐색 없음|
|public3|producer의 `--cohort public3`가 이미 `test.csv`3UID→test_series를 지원(:420,491). `validate_result`/`validate_frame`/집계는 전달받은 N을 검사|최종 full용 driver와 같은 test 진입점으로 1회 실행, 공식 제출열/UID/최종 `submission.csv`·자원 receipt를 검사|
|hidden 전체|한 study 픽셀 보관, batch1/single GPU/resident6, 임의N의 raw/rank pure functions|동적 test UID·별도 full budget·test-only bootstrap·최종 atomic 제출 파일과 새 release manifest/완료 gate|

현재 frozen CLI는 public3에서 `len(ids)==3`, engineering 작업은 최대1,800초다. trial `find_competition`은 train 경로를 요구하고 workflow·parity·thread receipt는100개에 고정돼 있다. **이 검증 notebook을 그대로 제출하지 않는다.** frozen 검사에 예외/monkeypatch를 넣지 않고 작은 새 test driver가 원본 manifest와 함수 SHA를 참조하도록 설계한다. 원본 assets/실패/PASS 계약은 불변이다. 별도 driver가 test.csv의 실제 전체UID와 test_series만 읽고 동일 frozen activation/model/input/aggregate/output 검증 함수를 재사용하면 가중치1.15GB 재업로드도 필요 없다.

최소 변경은 ① 중복/누락 없는 동적 test UID 계약, ② test-only 경로와 새 driver·원본자산 SHA 결합, ③ 전체 N의 raw를 모두 모은 뒤 rank하고 공식13열/UID 순서·finite/range를 검증해 `/kaggle/working/submission.csv`를 atomic 생성, ④ 실제 종료/worker hash/RSS/cgroup와 full budget을 판단하는 별도 gate다. `sample_submission.csv`가 제공되면 공식 열 계약과 대조한다. 기존 engineering receipt를 hidden PASS로 이름만 바꾸지 않는다.

새 focused CPU 검사는 N=1/3/100/>100, 동순위, 임의UID순서·중복/누락·부분member, test-only mount, timeout/실패 시 최종CSV 부재만 다룬다. 이미 완료된 합성31/37검사·로컬GPU100·V2 CPU100을 재실행하지 않는다. public3는 N=3 rank를 자체 raw와 확인하며100건 rank값이나 D01 예측과 대조하지 않는다. 필요하면3건 CPU analytical/input hash 확인만 같은 세션에 넣고 새4070 추론은 추가하지 않는다.

## 단1회 public3와 hidden 비용 판정

1. **선행:** root가 현재 engineering100 `result.json`/압축증거를 회수해 explicit numerical·resource·parent/worker·source SHA PASS를 확인한다. FAIL이면 이 계획의 구현/실행을 자동 시작하지 않고 정확 원인만 처리한다.
2. **그 뒤 CPU 구현·검토:** 위4개 변경과 focused 검사에 약30–45분, hard cap60분을 제안한다. 이전45–75분 전체패키지 감사를 다시 시작하는 계획이 아니다. 구현 소스·공개3 final schema·고정 계산을 root가 검토한다.
3. **새 test driver로 private T4 public3 1회:** 세션 예상3–8분·상한15분·retry0를 계획 예산으로 제안한다. 실측 ETA가 아니며 현재100 완료시간으로 시작 전에 갱신한다. 할당1T4이면≤0.25 GPUh,2T4이면≤0.5 GPUh; 실제quota/billing은 root 확인. offline cold 설치, 기존 작은 codec fixture, test3 raw decode/exact6, 최종CSV/receipt를 같은 세션에서 확인한다. asset 재업로드·별도 codec GPU 세션·두번째 비교추론은 계획에 없다. 처리 대기/컨테이너 startup은 별도다.
4. **hidden 예산:** 새 시험의 전체 workflow 시간에는 CPU DICOM100 재처리가 있으므로 이를 그대로 N배 하지 않는다. producer parent elapsed와 `inference.seconds`/50·100 진행 로그로 cold·생산경로 처리량을 분리하고, 실제 N에 대한 보수 시나리오와9h reserve를 계산한다.100은 per-study tail 시간/모든 압축 구문/환자 분포를 대표하지 않는다. VRAM/모델과 한 study 버퍼는 대체로고정, raw/rank/metadata/receipt는O(N)이므로 RSS100×N도 틀린 계산이다. raw만288N bytes이며 이것만으로 전체RAM을 주장하지 않는다. 공개3 성공도 hidden 전체 성공을 증명하지 않는다.

예산 여유가 불충분하면 자동 two-T4/추가 cohort/모델 제외로 바꾸지 않고 그 공백만 새 투자 판단으로 올린다. 모든 구현·test3·비용 검토 뒤 사용자에게 정확한 notebook revision과 가설을 제시해 **해당 제출을 새로 승인**받는다. 공개0.943 대체 가치와 독립 정확도는 이번 engineering 작업으로 확정되지 않는다. M07–M10 및 Biohub 누락 원칙에 따라 작은 RN 이득·골절 손실·노출weak 한계를 그대로 유지한다.

재사용 식별: serving `bd0fcaf4…c7b5b`, trial `e358076f…7f234`, parity `298f1185…754b`, thread bridge `1d6bc7c2…9e21e`, legacy DICOM `e478626b…4867e`. 전체 SHA는 기존 [verification](verification.json)/동결 assets manifest에 있다. 이 문서의 시간·비용은 **다음 단계 제안**이며 실행승인·진행상태·PASS가 아니다.
