# 2026-10-01 로컬 MRI 영상 분석 인계

## 요청과 결과

사용자가 다른 대회에서 쓰던 영상 분석 도구를 예시로 제공하며 대시보드에서 직접 데이터 파일을 열어 영상과 분석 도구를 사용하도록 요청했다. 이어서 중단 지점부터 마무리하도록 지시했다. **학습 데이터 → 영상 분석** (`#viewer`)을 추가했고 다운로드·캐시 화면의 바로가기와 기존 6개 메뉴를 유지했다.

지원 입력은 Adjacent/Orthodiffusion cache manifest, 숫자형 단독 NPY `[H,W]`/`[Z,H,W]`, 한 study의 단일 프레임 흑백 DICOM 파일/폴더다. 기존 캐시를 바로 사용할 수 있고, 경로 입력·등록 데이터 폴더 탐색으로 파일을 연다. 슬라이스·재생·휠·키보드, 대비·반전·확대/이동, ROI 통계/히스토그램, 길이 측정, 최대 12장 모아보기, 표시 PNG 저장을 제공한다.

study UID 검색·40개씩 페이지 이동·결측/공식 라벨/양성 소견 필터, 캐시 전체 품질/라벨 요약, 공식 train/test CSV 연결과 학습 Report, 검토 flag/메모·JSON 내보내기를 함께 제공한다. 미표기를 음성으로 바꾸지 않는다. 결측 마스크는 경고하고 ROI 통계를 제공하지 않는다. Gold58 노출과 테스트 Report 부재를 명시한다.

## 실제 데이터 관측

- Adjacent 캐시 4,407 study, 4×9×288. 557 study에 총 5,013개 결측 슬라이스가 있다. 공식 라벨이 있는 study는 58개이며 ACL은 양성 24, 음성 34, 미표기 4,349개다.
- 캐시 3개 폴더 모두 실제로 한 프레임을 읽고 유한 픽셀 수/해상도를 확인했다. Adjacent 288×288, Orthodiffusion 두 폴더 256×256. 단일 관측의 프레임 읽기/인코딩 시간은 각각 약 29/9/8 ms였으며 일반 성능 보장은 아니다.
- 대회 원본 DICOM·전체 원본 데이터는 다운로드하지 않았다. 실제 캐시 시각 확인과 도구 동작 검증을 수행했으며 20~50 study의 체계적 해부학/라벨 검토는 아직 하지 않았다.

## 구조와 보존

- `monitoring/kneeops/viewer.py`: catalog/경계 있는 폴더 탐색, 캐시 index, NPY mmap, DICOM 정렬/해독, frame/ROI 통계, 검토 기록.
- `monitoring/kneeops/viewer_api.py`, `api.py`: `/api/viewer/catalog|browse|studies|open|frame|contact|reviews|export`, Pydantic 및 기존 localhost/origin/CSRF 정책.
- `monitoring/web/src/ViewerPage.tsx`, `viewer.css`: 반응형 viewer와 canvas 도구, `main.tsx` 분류 탭/lazy loading, `DataPage.tsx` 바로가기.
- `pyproject.toml`: NumPy, Pillow, pydicom, pylibjpeg, libjpeg/openjpeg decoder. 기존 학습 환경·GPU/PyTorch와 분리.
- `monitoring/tests/test_viewer.py`, `monitoring/web/tests/viewer.spec.ts`: 신규 회귀 검사.
- README, STATE, 구현 아키텍처, dashboard/training-data runbook, 새 [영상 분석 runbook](../runbooks/image-viewer.md), 이 인계.

기존 원본·캐시·라벨·source revision·added_at을 변경하지 않는다. 개인 DB에 `viewer_reviews` 테이블만 추가한다. source/study/파일 버전/slot/slice별 마지막 검토를 저장하고 이전 파일 버전 기록을 유지한다. 원문 URI와 경로/크기/mtime 서명, manifest SHA, 선택 픽셀 SHA를 기록하며 전체 cache hash 검증이라고 주장하지 않는다. JSON에는 영상 본문/Report가 없고 DB 백업에는 검토 기록이 포함된다.

파일 열기 세션은 메모리의 최근 24개이고 앱 재시작 후 다시 열어야 한다. 캐시는 한 장만 mmap에서 복사한다. DICOM은 헤더로 목록을 구성하고 현재 한 장만 decode한다. 이미지 4M 픽셀, DICOM 하나 64 MiB, 한 study 폴더 300개·파일 2,500개, 단독 NPY 4,096장, 모아보기 최대 12장으로 제한한다. 직접 네트워크 공유 경로·pickle·RGB·다중 프레임은 지원하지 않는다.

## 검증 결과

- `.venv/Scripts/python -m pytest monitoring/tests tests --basetemp .local/tests/viewer-regression -q`: **48 passed**. 기존 39개와 신규 9개. FastAPI TestClient httpx deprecation 경고 1개.
- 신규 검사는 캐시 라벨/결측/페이지 필터, mmap/PNG/반전/ROI, 메모 재시작 보존과 변경 원본 분리, DICOM IOP/IPP 정렬·rescale·MONOCHROME1·RLE 압축 decode·mm², fallback·복수 study/다중 프레임 거절, 중복 위치·간격 불일치, NPY NaN/Inf·pickle/shape 거절, 폴더 traversal·잘못된 mask 값, API token/origin/JSON export를 포함한다.
- `npm run test:ui`: **14 passed**. 기존 10개와 viewer 4개. 실제 캐시 프레임·슬라이스·ROI·모아보기·필터·결측·폴더 탐색·빈 설치·오류·모바일 overflow/다크 테마 검사. 검토 저장 UI는 격리된 route fixture를 사용해 실제 연구 메모를 만들지 않는다. 실제 저장/재시작/내보내기는 pytest 임시 DB에서 검사한다.
- 마무리에서 0인 자동 대비 상한값의 보존과 메모 입력 시 재생 중지를 보완했다. `.venv/Scripts/python -m pytest monitoring/tests/test_viewer.py -q --basetemp .local/tests/viewer-final`: **9 passed**, `npm run test:ui -- tests/viewer.spec.ts`: **4 passed** 재확인.
- `npm run build`: TypeScript/Vite 성공. viewer lazy JS 약 28.7 kB, 기본 JS 약 294.8 kB.
- Docker image build 성공. 비root `docker run --rm`에서 NumPy/pydicom/Pillow import와 JPEG2000/JPEG-LS decoder available 확인. 별도 상시 컨테이너는 실행하지 않는다.
- desktop/mobile/dark 화면은 `.local/ui-viewer-desktop.png`, `ui-viewer-mobile.png`, `ui-viewer-dark.png`.
- `compileall monitoring/kneeops knee_workspace` 성공, 관련 문서 7개의 로컬 링크 누락 없음. 현재 DB `integrity_check=ok`, online backup `.local/kneeops/backups/2026-10-01T08-19-20.910835_00-00.sqlite` 생성. 실제 연구 DB의 검토 메모는 0개로 테스트 메모가 섞이지 않았음을 확인했다.

브라우저 전체 검사 중 기존 data 테스트는 전송하지 않는 planned 다운로드 미리보기 기록을 만든다. 실제 다운로드·수집·학습·Kaggle 제출은 이 작업에서 실행하지 않았다. Git 저장소가 없는 작업 폴더이므로 commit/PR은 만들지 않았다.

## 실행과 다음 작업

루트 `KneeOps.cmd` 또는 `.venv/Scripts/python -X utf8 -m monitoring.kneeops serve --port 8765`. 완료 시 실제 localhost 서버에 새 backend/static을 반영했다. 브라우저에 구버전이 남으면 새로고침한다. Docker 입력은 호스트 H: 경로가 아닌 `/workspace/...` 또는 `/downloads/...`를 사용한다.

사용법과 API 필드는 [영상 분석 runbook](../runbooks/image-viewer.md)에 있다. 다음 성능 작업은 캐시 20~50 study를 결측/라벨 소견별로 골라 시각 검토하고, 필요한 study의 원본과 crop·슬라이스 범위를 비교하는 것이다. 실제 대회 DICOM의 decoder/방향/라벨 연결은 원본을 받은 뒤 검증해야 한다. 병변/예측 overlay, 3D/MPR, NPZ/NIfTI/Zarr 지원은 현재 기능이 아니다.
