# 2026-10-02 전체 원본 탐색·파일 설명·중복 정리

사용자 요청: 중복을 정리하고 다운로드한 전체 원본을 대시보드에서 확인. 추가 요청으로 전체 데이터 구조와 설명을 표시한다. 전체 ZIP을 보존하면서 필요한 study만 lossless 해제하는 방식을 적용했다. 새 다운로드·GPU 학습·제출은 없다.

## 구현과 실제 결과

- 전체 ZIP 인덱스: `9175e113e4a00fe679c4088b`, 819,640항목(819,635 DICOM+5 CSV), 학습 4,407·공개 테스트 3 study. 중앙 목록 SHA256 `18bc75eba31d03a12f35e646a337980db06c3cb48b974402396fec9dfe5d7412`. 약31초에 생성했고 SQLite 파일은 약757MB다. 전체 ZIP SHA256은 계산하지 않았다.
- `data_archive.py`: 안전한 중앙 목록 검사, SQLite 영속 인덱스, ZipInfo/header offset 기반 선택 읽기. study당 OS 파일 잠금, 1GiB reserve, 링크/ADS/경로탈출/중복 항목 차단, CRC+SHA256, 임시 파일과 atomic receipt, 기존 파일 충돌 시 보존, receipt 기반 재개. Python zipfile의 `_RealGetContents`/`_end_offset` 사용 부분은 Python 업그레이드 때 ZIP64/압축 회귀 확인이 필요하다.
- `DataStructure.tsx`, `RawDataPanel.tsx`: 전체 구조와 CSV 설명, UID 검색·train/test 필터·20 study씩 목록, 시리즈별 방향/수분 민감도/지방억제·용량·30개씩 파일 탐색. 공식 CSV에는 SeriesDescription이 없으므로 DICOM 촬영명과 혼동하지 않는다. 자세한 촬영명은 뷰어의 DICOM 시리즈에서 표시한다. 공식 Kaggle 데이터 설명 바로가기도 있다.
- `data_library.py`, `viewer.py`, `api.py`: ZIP 원본을 raw 목록에 통합. 같은 study의 loose copy는 중복 행으로 나열하지 않으며 과거 snapshot은 보존. raw 열기는 전체 시리즈를 준비한 후 기존 뷰어로 연결한다. 최대632파일/14시리즈이므로 현재 뷰어의 study 한도 안에 들어간다.
- `data structure|prepare|descriptor` JSON CLI. `GET /api/data/structure`는 파일을 풀지 않는다. descriptor export는 준비된 파일의 크기/mtime receipt도 확인하며 미준비 ZIP study는 training_allowed=false다. 테스트는 항상 false다. 라벨/분할/마스크 계약을 보장하는 학습 launcher는 아니다.
- 실제 준비: 학습2·공개 테스트3 study, 총815 DICOM, 702,286,056 bytes. 위치 `storage/downloads/rsna-knee-abnormality-detection/prepared/9175e113e4a00fe679c4088b/`. 학습 원본 MRI 512×512, 테스트 전체5시리즈, 슬라이스 모아보기를 실제 브라우저로 검증했다. 전체570GB를 한꺼번에 해제하지 않았고 나머지는 첫 열기에 준비한다.

## 중복 정리

- 전체 SHA256을 두 번 실제로 읽어 비교한 Orthodiffusion 복사본17파일을 NTFS hardlink로 대체했다. 제거한 중복 물리 bytes: **13,865,230,576**. canonical `storage/cache/derived/EXP032_orthodiff`; 호환 경로 `EXP032_ORTHODIFF_3X16X256`은 같은 파일 실물을 가리킨다. 고유한 파일·실험 경로는 없애지 않았다. 향후 캐시는 불변 입력으로 사용하고 새 전처리를 이 경로에 덮어쓰지 않는다.
- 앱은 manifest 내용과 UID/mask/shard의 실제 device/inode가 모두 같은 캐시만 합쳐 보인다. Adjacent와 Orthodiffusion 2개가 표시된다. 내용 hash가 같다는 이유만으로 검사하지 않은 다른 캐시를 자동 통합하지 않는다.
- 별도로 받은 공개 테스트57개는 ZIP에서 CRC 검사하며 준비한 영상과 SHA256이 모두 일치해 삭제했다. 제거 **92,193,282 bytes**. 빈 시리즈·study 폴더만 rmdir했고 재귀 삭제는 하지 않았다. ZIP·준비 원본·공식 CSV·Adjacent·실험 산출물은 보존했다.
- 합계 중복 **13,957,423,858 bytes** 제거. 별도로 생성한 원본 준비702MB와 인덱스757MB 때문에 드라이브의 순증 여유와 이 값은 다르다.
- ledger: `.local/kneeops/cleanup/cache-bbe0f07c1f404ccbaadb5d137eb18b78.json`과 `loose-*.json`. DB online backup `2026-10-01T16-19-12.224378_00-00.sqlite`. `original-preparation-2026-10-02.json`은 초기4study만 기록하며 브라우저 테스트가 학습1study를 추가해 최종5study다.
- 다운로드의 과거 실패 이력은 보존. 전체 ZIP이 연결되어 있으면 공개 테스트에 재다운로드 불필요 안내를 표시하고 재시도를 감춘다. 미실행 planned 미리보기는 이력 화면에서 숨긴다.

## 검증

실제 실행 명령(프로젝트 루트, npm 명령만 `monitoring/web`):

```powershell
.venv/Scripts/python -X utf8 -m pytest monitoring/tests -q --basetemp=.local/test-archive-full-01 --tb=short
# 59 passed, 14.99s
.venv/Scripts/python -X utf8 -m pytest monitoring/tests/test_data_archive.py monitoring/tests/test_data_library.py -q --basetemp=.local/test-archive-export-01 --tb=short
# export의 누락 파일 차단 보강 후 12 passed
npm run build
# TypeScript/Vite 성공, 마지막 변경 후 재실행
npx playwright test tests/archive.spec.ts tests/raw-data.spec.ts tests/cache-origin.spec.ts tests/viewer.spec.ts --reporter=line
# 실제 localhost 데이터에서 8 passed, 19.7s
```

처음 pytest 기본 임시 폴더는 Windows 접근 거부여서 workspace `.local`의 새 basetemp로 재실행했다. 테스트 코드 문제로 포장하지 않는다. 예상된 duplicate ZIP fixture warning과 기존 Starlette deprecation warning이 있다. 최종 export API에서도 raw4410/학습준비2/receipt확인5, 캐시2개를 확인했다.

스크린샷: `.local/ui-archive-structure.png`, `ui-archive-mobile.png`, `ui-archive-train-viewer.png`. 실제 이미지를 열어 구조 UI와 MRI 렌더링을 확인했다. 기존 ROI/대비/검토메모 fixture/모바일 검증도 통과했다. Docker 재빌드는 하지 않았다.

현재 백그라운드 앱 PID **33184**, run ID `aff419b4d4d54dfc9b648fbcaf33cd4a` (관측값). 다운로드/수집 active가 없는 것을 확인 후 정상 재시작했다. 이후 상태는 API로 다시 확인한다.

## 남은 한계와 다음 단계

- 모든 영상의 CRC를 아직 검증하지 않았다. 준비한 study만 CRC+SHA256을 검사했다. 재사용/명세 export의 크기·mtime 확인은 전체 재해시와 다르다.
- study별 준비는 동기 API 요청이며 첫 열기에 수초 이상 걸릴 수 있다. 브라우저를 닫아도 진행 중 요청은 서버에서 끝날 수 있다. 앱/PC가 종료되면 같은 study 열기로 파일별 재개한다. 자동 제거/LRU/전체 일괄 해제 버튼은 없다.
- 전체 ZIP index는 약757MB이고 첫 목록 생성은 약31초. 이후 목록·파일 탐색은 이 DB를 사용한다. 더 큰 규모가 필요해지면 API study pagination과 인덱스 압축을 고려한다.
- JSON CLI prepare는 뷰어 없이 study를 준비할 수 있다. 후속 학습은 적절한 label/fold/mask 계약과 함께 descriptor의 준비된 train 경로를 소비해야 한다. 자동 GPU 학습을 시작하지 않는다.
- 기존 캐시 hardlink는 두 경로에서 내용을 수정하면 같은 실물이 바뀐다. 새 캐시 버전은 새 디렉터리에 기록한다. 고유 산출물 삭제/캐시 재생성은 이번 범위가 아니다.

최신 사용법: `docs/runbooks/training-data.md`; 상태: `docs/STATE.md`; 원본 열기: `http://127.0.0.1:8765/#data`.
