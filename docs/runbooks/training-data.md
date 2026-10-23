# 학습 데이터 관리와 570 GB 활용 판단

관측일: 2026-10-02 KST. 전체 ZIP 다운로드는 10월 2일 00:11 KST에 완료되었다. 실제 파일 265,018,939,620 bytes(265.0 GB / 246.82 GiB), ZIP 목록 819,640개·비압축 569,764,475,800 bytes를 확인했다. 전체 DICOM CRC 검사는 아직 하지 않았다. 최신 기능은 전체 목록 탐색과 선택 study만 원본으로 해제하는 방식이며, 실제 학습2건·테스트3건을 검증했다. [완료 확인 인계](../handoffs/2026-10-02-download-status.md).

## 결론

현재는 **기존 Adjacent 캐시 → 작은 기준 실험 → 필요한 study 원본만 추가 → 필요성이 입증되면 전체 다운로드** 순서를 권장한다. 원본 전체를 받는 것 자체가 성능 개선은 아니다. 먼저 라벨·검증 분할·추론 변환을 고정한 비교가 필요하다.

| 구분 | 확인된 규모 | 용도 / 한계 |
| --- | --- | --- |
| 공식 전체 | 819,640개, 569,764,475,800 bytes = 약 569.8 GB / 530.63 GiB | DICOM 819,635개와 CSV 5개. 압축 ZIP의 전송 크기는 이 값과 다르다. |
| 받은 전체 ZIP | 265,018,939,620 bytes = 약 265.0 GB / 246.82 GiB | 학습 4,407 study·819,078 DICOM, 공개 테스트 3 study·557 DICOM, CSV 5개 포함. 비공개 채점용 테스트는 포함되지 않는다. |
| 이 PC의 원본 폴더 | CSV 5개, 9,151,736 bytes | 전체 DICOM은 내려받지 않은 상태였다. |
| Adjacent 캐시 | 4,407 study, 약 12.26 GiB | uint8, 4 slots × 9 slices × 288 × 288. 첫 기준 실험에 추천. |
| Orthodiffusion 캐시 | 4,407 study, 각 약 12.91 GiB | uint8, 3 planes × 16 slices × 256 × 256. 동일 shard 해시 명세를 가진 두 폴더가 있다. |
| 캐시 폴더 전체 | 약 38.08 GiB | 중복 후보를 포함한 물리 파일 합계. 자동 삭제하지 않았다. |
| 공개 테스트 원본 | 3 study, 557 DICOM, 599,962,984 bytes | 약 600.0 MB / 572.2 MiB. 별도 부분 다운로드는 57개에서 실패했지만, 완료된 전체 ZIP에는 557개 모두 목록에 포함된다. 해제된 로컬 목록은 아직 57개다. 숨겨진 평가 데이터는 포함되지 않는다. |

공식 크기는 Kaggle `get_competition_data_files_summary` 결과다. CSV와 공개 테스트 수치는 인증된 로컬 SDK의 `list_data_tree_files`로 확인했다. 일부 별도 connector는 규칙 동의 오류를 반환했지만 로컬 SDK 조회·소형 다운로드는 정상 동작했다. 대회 신청을 다시 할 필요가 있다는 근거로 해석하지 않는다.

## 브라우저에서 사용

왼쪽 **학습 데이터** 메뉴: <http://127.0.0.1:8765/#data>.

1. 다운로드 폴더를 저장한다. 기본값은 `<workspace>/storage/downloads/rsna-knee-abnormality-detection`이다. 이 PC의 DB에만 저장되며 `config/workspace.toml`의 기존 학습 원본 경로를 바꾸지 않는다. 절대 경로로 다른 드라이브를 선택할 수 있다. 기존 원본 폴더와 분리해 보존한다.
2. CSV / 공개 테스트 / 학습 study 선택 / 전체 ZIP 중 범위를 고른다. **다운로드 준비**는 파일 목록·용량·기존 파일 충돌을 확인하는 미리보기다. 실제 전송은 **다운로드 시작**에서 수행한다.
3. 선택 학습 study는 UID 검색으로 한 번에 최대 5개를 고른다. 각 study의 모든 시리즈와 DICOM을 열거한다. SOP UID 문자열의 정렬 순서로 슬라이스를 일부만 고르지 않는다. CSV가 없는 새 PC는 CSV부터 받는다.
4. 실행 전 여유 공간과 쓰기 권한을 다시 확인한다. 개별 파일은 원본 크기 × 2.1 + 1 GiB, 전체 ZIP은 원본 크기 × 1.1 + 1 GiB를 보수적 여유 공간으로 사용한다. 전체 ZIP의 실제 압축 크기를 아는 것은 아니다. ZIP을 풀려면 원본 크기만큼 추가 공간이 필요하다.
5. 이력에서 완료 파일 수·저장 바이트·현재 파일의 수신 바이트를 확인한다. **중지**는 partial을 보존하고 **재시도 / 이어받기**는 같은 계획·경로를 사용한다. 서버의 Range/ETag/Last-Modified 지원과 원격 객체 동일성이 확인되어야 이어받으며 그렇지 않으면 해당 미완료 파일을 다시 받는다.

다운로드는 자료 수집과 별도 worker에서 한 번에 하나씩 수행된다. 앱을 종료하거나 PC를 끄면 계속 전송하지 않는다. 재시작 후 사용자가 재시도한다. 경로를 변경해도 이미 만든 계획의 경로는 바뀌지 않으므로 이력의 저장 경로를 확인한다. 계획과 완료 기록은 JSON으로 내보낼 수 있다.

기존 동명 파일은 크기만 보고 완료로 간주하거나 덮어쓰지 않는다. 해당 계획에서 완료 기록이 있는 파일만 크기를 재확인하고 건너뛴다. 외부에서 미리 받은 파일과 충돌하면 새 다운로드 전용 폴더를 사용한다. 임시 파일은 저장 경로의 `.kneeops/<plan id>/`에 보존한다. 다운로드 도중 이 폴더를 옮기거나 수정하지 않는다. 완료 후 원격 디렉터리 구조 그대로 배치한다. 전체 ZIP은 자동 압축 해제하지 않는다.

현재 최종 파일 배치는 같은 볼륨의 hard link를 사용해 기존 파일을 원자적으로 보호한다. NTFS/ext4 같은 hard link 지원 파일시스템을 사용한다. FAT/exFAT 또는 일부 네트워크 파일시스템에서 실패하면 지원되는 로컬 드라이브를 지정한다. 원격 목록은 계획 시점의 관측이며 Kaggle 서버의 버전 고정 다운로드를 보장하지 않는다. 개별 파일의 크기가 계획과 달라지면 완료 처리하지 않는다.

## 로컬 원본 목록과 영상 연결

**로컬 원본 데이터 → 로컬 목록 새로고침**은 설정한 다운로드 폴더와 기존 원본 폴더 아래 `train_series/<StudyUID>/<SeriesUID>/*.dcm`, `test_series/...`를 읽는다. ZIP과 임시 전송 파일은 제외한다. 다운로드 계획의 파일명/크기와 공식 series CSV를 대조해 미다운로드·일부만 있음·목록/크기 일치·완전성 미확인을 표시한다. 기존 데이터는 복사하거나 수정하지 않는다.

**영상 열기**로 해당 study를 연다. 테스트는 학습에 쓰지 않으며 불완전한 학습 study도 학습 입력에서 제외해야 한다. `training_allowed`는 train 구분과 계획의 파일 목록/크기 충족만 나타낸다. 라벨·분할 검증이나 DICOM 픽셀/CRC 검증 완료를 의미하지 않는다.

**로컬 데이터 명세 JSON**에는 원본 경로·파일 수·기대 크기·상태와 캐시 manifest/UID hash를 기록한다. workspace 안에서는 논리 URI를 사용한다. 학습 launcher가 아니며 외부 드라이브 경로는 각 PC에서 다시 연결해야 한다.

```powershell
.venv/Scripts/python -X utf8 -m monitoring.kneeops data index --json
.venv/Scripts/python -X utf8 -m monitoring.kneeops data library --json
```

API는 `GET /api/data/library`, `GET /api/data/library/descriptor`, `POST /api/jobs`의 `{ "kind": "data_index" }`다. POST에는 세션 token이 필요하다. 전체 ZIP 압축 해제와 80만 개 규모 명세의 성능 검증은 다운로드 완료 후 진행한다.

## 기존 캐시 연결

### 생성 출처

이 프로젝트의 Adjacent와 Orthodiffusion 캐시는 **공식 배포 캐시가 아니다**. 공식 대회 DICOM을 `taeyangg4` 계정의 자체 전처리 노트북에서 변환한 출력이다. 로컬 패키지 메타데이터, 생성 코드의 contract, 출력 로그와 manifest를 대조했다. 개별 사람/AI의 코드 작성자와 정확한 Kaggle 실행 버전 번호까지 확인된 것은 아니다.

- [Adjacent 생성 노트북](https://www.kaggle.com/code/taeyangg4/rsna-knee-exp034-adjacent-cache-4x9x288)
- [Orthodiffusion 생성 노트북](https://www.kaggle.com/code/taeyangg4/rsna-knee-exp032-orthodiffusion-cache-3x16x256): 동일 명세의 두 로컬 폴더 모두 이 노트북에 연결한다. 폴더 이름과 비슷한 과거 `rsna-knee-exp032-orthodiff-cache` 패키지는 현재 파일 명세와 생성 코드가 달라 출처로 사용하지 않는다.
- [OrthoDiffusion 참고 연구 코드](https://github.com/lt-0123/OrthoDiffusion/tree/99d15186bb8face1728263023d8414d9405c8f7e): `lt-0123` 연구 프로젝트이며 캐시 제작 계정과 구분한다.
- [공식 원본 데이터](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

생성 노트북은 로컬 메타데이터에 비공개로 기록되어 있어 해당 계정 또는 공유 권한이 필요하다. 링크는 노트북 페이지이며 과거 실행 버전 고정 링크는 아니다. 현재 원격 공개 범위/접근 성공을 재검증했다고 주장하지 않는다.

대시보드 각 캐시 카드에 생성 계정·비공식 표시·생성 노트북·원본/참고 링크를 제공한다. `monitoring/kneeops/cache_origins.json`은 정확한 manifest SHA256에 출처와 근거 파일 SHA256/관측일을 연결한다. 폴더명/contract만 같다는 이유로 출처를 붙이지 않으며 모르는 캐시는 출처 미확인으로 표시한다. 기존 manifest 자체는 수정하지 않았다.

우선 연결할 파일:

`storage/cache/derived/EXP034_ADJACENT_4X9X288_TEST/adjacent_cache_manifest.json`

이름에 TEST가 있지만 manifest와 UID 목록에는 **4,407개 학습 study**가 있다. 파일명만 보고 작은 시험 캐시라고 판단하지 않는다.

**이 캐시 연결**은 manifest를 개인 설정에 기록하고 **학습 입력 JSON**을 제공한다. 데이터 복사·학습 실행·기존 실험 설정 변경은 하지 않는다. JSON은 manifest 위치, 명세 hash, UID 파일 hash, 변환 contract, shape를 포함하므로 AI와 코드가 같은 입력을 참조할 수 있다. 외부 드라이브의 캐시는 절대 경로이므로 다른 PC에서는 다시 연결한다.

검사 범위: NPY 헤더와 크기, 연속 shard 구간, study 수, 중복 없는 UID 순서, mask shape와 기록된 mask SHA256. **13 GB 영상 shard 전체 해시와 실제 영상 품질은 검사하지 않았다.** 동일 해시 명세의 캐시 두 개를 발견했다는 것은 현재 두 파일 세트의 바이트 동일성을 직접 증명한 것이 아니다.

기존 `training/src/train_adjacent_2p5d_clean.py`의 `Shards`는 `numpy.load(..., mmap_mode='r')`를 사용한다. 12.26 GiB 캐시를 RAM/GPU에 한꺼번에 올리지 않고 Dataset이 요구하는 study를 읽는다. 학습용 Python/GPU 환경과 대시보드 `.venv`는 별도다. 영상 분석 기능을 위해 대시보드에는 NumPy·Pillow·pydicom과 압축 DICOM decoder를 설치했다. PyTorch/GPU 학습 환경은 포함하지 않는다.

AI가 읽는 실제 CLI:

```powershell
.venv/Scripts/python -X utf8 -m monitoring.kneeops data status --json
.venv/Scripts/python -X utf8 -m monitoring.kneeops data cache-descriptor --manifest storage/cache/derived/EXP034_ADJACENT_4X9X288_TEST/adjacent_cache_manifest.json --json
```

학습 코드가 사용하는 입력 옵션은 `--cache-dir storage/cache/derived/EXP034_ADJACENT_4X9X288_TEST`다. 내보낸 JSON을 자동으로 실행하는 학습 launcher는 아니다. `--out`, fold, 라벨 파일, 모델 가중치의 학습 노출을 확인한 다음 기존 학습 환경에서 실행해야 한다. 이 작업에서는 GPU 학습을 시작하지 않았다.

## 성능을 위한 단계별 판단

1. **입력·검증 계약 확인.** [영상 분석 화면](image-viewer.md)에서 캐시 20~50 study를 시각 검토하여 좌우/방향, 결측, 잘린 구조, 라벨 연결을 확인한다. 뷰어의 실제 영상 읽기와 도구 검증은 완료했지만 이 규모의 내용 검토는 아직 수행하지 않았다. report 유사도/중복을 고려한 grouped fold를 고정하고 Gold58 노출·backbone 학습 노출·weak label 평가를 기록한다. Gold58은 독립 holdout이 아니다.
2. **캐시 기준 실험.** Adjacent는 AX_FS/SAG_FS/COR_FS/SAG_STRUCT의 4 slots를 사용한다. 물리 위치 정렬 뒤 가운데 연속 9장, 140 mm crop, pooled percentile 2–98 정규화다. 기존 2.5D/MIL 학습을 batch 2·AMP부터 검증하고 peak VRAM, epoch 시간, target별 weak OOF를 기록한다. 16 GB GPU에서 가능한 배치는 실제 모델/입력에 따라 측정한다. 약한 라벨 평가 개선을 Private LB 개선으로 단정하지 않는다.
3. **원본을 추가할 실험만 선택.** 가운데 9장/crop 때문에 놓치는 병변·구조, 결측 slot 557개를 우선 조사한다. 해상도·슬라이스 수·full FOV를 한 번에 바꾸지 말고 같은 split에서 비교한다. Orthodiffusion은 center16, 3 fluid planes, full FOV 256px, volume min-max 정규화이며 plane fallback 416개·short slot 22개가 기록되어 있다. 같은 study 수여도 변환 계약이 달라 단순 교체 성능 비교에 주의한다.
4. **확장 판단.** 정보 손실이 병목이면 전체 원본 또는 Kaggle 측 전처리로 더 넓은 캐시를 만든다. CPU DICOM decode를 매 epoch 반복하지 않도록 uint8 shard/NPZ로 한 번 변환한다. 검증이 끝난 변환을 제출 추론에도 동일하게 적용하고 공개 테스트 3개에서 I/O를 점검한 뒤 숨겨진 약 1,300 study 규모의 시간/메모리를 별도로 추정한다. 테스트에는 Report가 없다.

## 참가자 코드 근거와 Kaggle 측 캐시 생성 대안

- [Multi-plane DINOv2](https://www.kaggle.com/code/abisheksrivastav/rsna-knee-multi-plane-dinov2): 저장본 SHA256 `31b0dccdcab7b29cb18784a53c694cd5dcf5e2128626cefd8dc76b607a49e447`. 6 protocol slots **각각 12 slices**를 캐시에 저장한다(결측이 없으면 study당 72장). 336px, 130mm crop, 깊이 20~80% 범위에서 균등 샘플링, 물리 DICOM 순서, uint8 NPZ `CompressedStudyCache`다. 3장씩 4그룹으로 나누므로 캐시 전체 장수와 한 번의 모델 입력 장수는 다르다. 과거 문서의 표현을 2026-10-02 코드 재확인 후 정정했다. 실제 Kaggle 버전 번호는 이 저장본에서 미확인이다.
- [Read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee): 저장본 SHA256 `223aac4eb1f7da195c82156f40b9f9a837b4087b9c9aa90d0022494bfb1687a2`. 336px 캐시, 남은 메모리 비율 0.45·최대 24 GB를 기준으로 슬라이스 수를 조정한다. 노트북의 점수 주장은 독립 검증한 결과가 아니다.

이를 근거로 전체 DICOM을 매번 읽는 방식보다, **필요한 해부학적 정보를 선택해 고정 캐시로 반복 학습**하는 흐름이 현실적이라고 판단했다. 참가자 코드는 수집된 참고 자료이며 자동 실행하지 않았다.

로컬 570 GB를 피하려면 Kaggle Notebook에 대회 데이터를 입력으로 붙이고 `/kaggle/input`에서 DICOM을 읽어 캐시 shard를 만들 수 있다. 정렬·slot 선택·crop·정규화 버전을 고정하고 UID/mask/shape/checksum 명세를 출력한다. 임시 cache와 보존되는 notebook output 위치를 구분하고, 출력 용량·실행 시간 한도를 확인해 분할한다. 이후 캐시 산출물만 PC에 다운로드하고 연결한다. Kaggle에 이미 원본이 있다는 장점이 있지만 전처리 연산과 저장 한도는 필요하다. 이 앱은 Kaggle Notebook 생성/실행/게시 또는 남의 캐시 재배포를 수행하지 않는다.

## 입력 크기와 성능 해석 (2026-10-02)

현재 캐시 모두 4,407 study의 UID를 포함한다. 전체 원본 확보는 환자 수 증가가 아니라 study당 해상도·슬라이스·시리즈 선택 범위를 넓히는 것이다. 전체 환자 사용과 모든 픽셀/슬라이스 사용을 구분한다. 결측 slot과 학습/검증 분할도 별도로 유지한다.

해상도 256/288/336/384/512, 시리즈당 9/16/24/32장 또는 가변 전체, 중앙 연속/넓은 범위 균등/학습 중 무작위 샘플링, 물리 crop/full FOV/국소+전체 입력 등을 비교할 수 있다. 이는 실험 후보이지 각 단계의 성능 향상이 검증된 목록이 아니다. uint8 캐시는 저장 형식이며 좋은 전처리를 보존하면 원본을 매 epoch 직접 읽을 필요가 없다. 이미 줄인 캐시를 확대해도 소실된 세부 정보가 복구되지 않는다.

같은 시리즈 수·dtype일 때 288px×9장→384px×24장은 입력 픽셀 수 약 4.74배다. 실측 GPU 메모리/학습 시간 배수와 같다는 뜻은 아니다. 큰 입력으로 batch/학습 횟수/앙상블 수가 줄거나 사전학습 변환과 어긋나면 점수가 하락할 수도 있다. 이라는 이 프로젝트의 검증 근거는 없다.

기존 EXP028 방향 정리 paired audit는 같은 체크포인트/헤드에서 Gold58 진단 macro AUC +0.0027785, 기록된 bootstrap 95% 구간 [-0.0053193, +0.0115253]이다. 상승·하락 target이 섞이고 구간에 0이 포함된다. Gold58은 이미 선택에 노출되었으므로 독립 검증이나 원본 전체 사용 효과의 증거가 아니다. 해상도/장수만 바꾼 통제 실험 결과로도 사용하지 않는다.

권장 비교 순서: baseline 재현 → 같은 9장으로 sampling 범위 비교 → 같은 범위에서 16/24장 → 같은 sampling에서 336/384px → crop 범위/전체+국소 → 유망 후보에만 추가 시리즈·3D. 환자 분할·라벨·모델·선택 규칙을 고정하고 12 target OOF와 불확실성, decode 실패/결측, 시간·VRAM을 비교한다. 계산량 공정 비교도 별도로 기록한다. 이번 질의에서는 새 학습을 실행하지 않았다.

## Docker

Windows의 큰 H: 데이터와 기존 GPU 환경에는 네이티브 실행을 계속 권장한다. Compose는 `/workspace`를 읽기 전용으로 유지하고 별도 `kneeops-downloads` 볼륨을 `/downloads`에 쓰기 가능하게 마운트한다. 기본 다운로드 경로도 `/downloads`다. Docker Desktop 내부 디스크 용량이 기준이므로 H: 여유 공간과 같다고 가정하지 않는다.

호스트 폴더로 받으려면 Compose의 `/downloads` 볼륨 항목을 원하는 호스트 전용 폴더의 bind mount로 변경한다. UI에는 컨테이너 경로 `/downloads`를 입력한다. `/workspace`의 읽기 전용 경로를 선택하면 쓰기 검사에서 차단된다. 볼륨 내용은 `docker compose down`에 보존되지만 `down -v`는 볼륨을 삭제하므로 사용하지 않는다.

## 전체 원본을 탐색하고 열기 (2026-10-02)

1. 학습 데이터 → 로컬 원본 데이터에서 전체 4,410 study를 검색한다. 학습/공개 테스트 필터와 20건씩 페이지 이동을 사용한다.
2. **전체 데이터 구조·설명**에서 train_series/test_series의 study → series → DICOM 구조, CSV 5개의 역할과 Kaggle 공식 데이터 설명 링크를 본다.
3. study의 **시리즈·파일 구조 및 촬영 설명**을 펼치면 공식 CSV의 촬영 방향·수분 민감도·지방억제와 시리즈별 파일 수·크기를 본다. 시리즈를 선택하면 30개씩 파일명을 본다. 이때 압축 해제는 없다.
4. **원본 준비 후 열기**를 누르면 해당 study의 모든 시리즈를 원본 그대로 해제하고 CRC·SHA256을 기록해 뷰어로 연다. 선택한 study 용량 외에 1 GiB의 여유 공간이 필요하다. 첫 열기는 수초 이상 걸릴 수 있으며 다음 열기는 크기/mtime receipt로 재사용한다.
5. 파일은 `<download>/prepared/<archive-id>/<split>_series/<study>/<series>/*.dcm`에 남아 후속 학습에 쓸 수 있다. 전체 ZIP은 보존되며 570 GB 전체 자동 해제·GPU 학습은 실행하지 않는다.
6. 데이터 명세 JSON은 준비된 study에 한해 파일 크기/mtime를 확인한다. CRC/SHA는 해제 시점 검사이며 전체 ZIP SHA 검사는 아니다. 테스트는 training_allowed=false다. 학습 입력의 라벨·분할·마스크와 Gold58 노출은 별도로 확인한다.

```powershell
.venv/Scripts/python -X utf8 -m monitoring.kneeops data library --json
.venv/Scripts/python -X utf8 -m monitoring.kneeops data structure --raw c6cbac7f5974259d656301e1 --json
.venv/Scripts/python -X utf8 -m monitoring.kneeops data prepare --raw c6cbac7f5974259d656301e1 --json
.venv/Scripts/python -X utf8 -m monitoring.kneeops data descriptor --json
```

예시 ID는 이 PC의 연결된 ZIP에 해당한다. 다른 PC에서는 library 결과의 ID를 사용한다. 전체 ZIP 목록 SQLite는 약 757 MB이며 현재 4,410 study의 요약만 UI로 전달한다. 준비 study는 자동 삭제하지 않고 총 준비 용량을 표시한다. ZIP이 변경되면 목록을 새로고침하며 변경된 ZIP에 기존 receipt를 적용하지 않는다.

중복 정리: Orthodiffusion의 완전히 같은 17개 복사본을 SHA256 확인 후 hardlink로 대체해 약 13.9 GB를 회수했다. 과거 경로는 남아 있지만 같은 파일 실물을 공유한다. 캐시는 읽기 전용 입력으로 사용하고 새 전처리는 새 폴더에 쓴다. 별도로 받은 테스트 57파일도 전체 ZIP에서 해제한 원본과 같음을 확인한 후 삭제했다. 정리 ledger는 `.local/kneeops/cleanup/`이며 ZIP·Adjacent·실험 산출물은 보존했다.
