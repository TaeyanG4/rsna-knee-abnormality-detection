# 2026-10-01 학습 데이터 기능 인계

## 사용자 요청과 완료 범위

500 GB 이상 대회 데이터를 어떤 방식으로 학습에 이용할지 판단하고, 먼저 다운로드 버튼·개인 저장 경로·기본 경로와 일부/캐시 활용 방법을 추가했다. 전체 다운로드나 GPU 학습은 시작하지 않았다. 앞선 [전체 구현 인계](2026-10-01-implementation.md)의 기능과 개인 PC 독립 설치 결정을 유지한다.

왼쪽 **학습 데이터** `#data`에서 캐시 현황과 구조 검사, 캐시 연결·입력 JSON, 저장 폴더, CSV·공개 테스트·학습 study 1~5개·전체 ZIP 다운로드를 제공한다. 미리보기 후 명시적 시작, 진행량, 중지, 재시도/조건부 이어받기, JSON 파일 명세와 이력이 있다. 별도 다운로드 worker라 수집 queue와 독립적이다. UI는 desktop/좁은 앱 패널/mobile에서 확인했다.

## 근거와 권장 판단

공식 파일 summary: 819,640개, 569,764,475,800 bytes(약 569.8 GB/530.63 GiB). 이 PC의 원본 디렉터리는 CSV 5개뿐이었다. 기존 캐시는 약 38.08 GiB, 3개 폴더이며 Orthodiffusion 두 폴더는 동일 shard hash 명세를 갖는 중복 후보다. 어떤 기존 파일도 삭제하지 않았다.

권장 Adjacent 캐시는 4,407 study, 약 12.26 GiB, 4×9×288 uint8. 헤더·size·UID·mask 검사는 통과했다. 기존 학습 코드가 mmap을 사용한다. 영상 shard 전체 hash·실제 영상 시각 검토·라벨 성능 검증은 수행하지 않았다.

이 PC의 초기 빈 캐시 설정에는 검증된 Adjacent manifest를 연결했다. 실제 API의 연결·입력 JSON 응답도 확인했다. 데이터 복사나 학습 실행은 없다.

인증된 로컬 SDK directory tree를 이용해 공개 테스트 3 study의 557 DICOM/599,962,984 bytes 목록을 조회했다. 다운로드하지 않았다. MCP 쪽 동일 tree 조회의 규칙 동의 오류와 로컬 SDK 정상 동작을 구분한다. 참가자 공개 노트북의 12-slice/uint8/NPZ 및 메모리 예산 사용을 확인한 근거와 hash는 [데이터 runbook](../runbooks/training-data.md)에 있다. 외부 노트북을 실행하지 않았다.

## 변경 파일

- `monitoring/kneeops/data_remote.py`: 격리 SDK catalog/download. signed URL/인증정보를 로그로 전달하지 않음.
- `monitoring/kneeops/data_manager.py`: 경로/공간·동명 파일 보호, 불변 plan, 별도 worker, staging·hard link 완료 배치, 중지/재시도.
- `monitoring/kneeops/data_cache.py`: stdlib NPY header 검사, manifest/UID/mask 검증, 개인 캐시 연결 명세.
- `api.py`, `cli.py`: localhost 쓰기 보안 적용한 data API와 읽기 전용 JSON CLI.
- `monitoring/web/src/DataPage.tsx`, `data.css`, `main.tsx`: 새 메뉴·반응형 데이터 화면.
- `monitoring/tests/test_data.py`, `monitoring/web/tests/data.spec.ts`: 회귀/브라우저 검사.
- `compose.yaml`, `monitoring/Dockerfile`: 읽기 전용 workspace와 별개인 `/downloads` 쓰기 볼륨/권한.
- README, STATE, 아키텍처, 운영/학습 데이터 runbook, 이 handoff.

DB schema3 기존 테이블을 유지하고 `downloads`를 추가한다. `data_destination`, `data_cache_manifest`는 개인 meta 설정. 다운로드 파일/partial은 선택 경로에 있어 DB 백업만으로 보존되지 않는다.

## 실제 검증

- `.venv/Scripts/python -m pytest monitoring/tests tests --basetemp .local/tests/data-resume-final -q`: **36 passed**. 경로/동명 보호, 다운로드 계획 경로 고정, 공간 재확인, queue 중복 거절, partial 보존·재시도, subprocess 종료, 재시작 interrupted, 잘못된 크기의 완료 차단, tree pagination/scope, cache 누락/중복/traversal, CSRF/미리보기 분리를 포함한다. 이미 받은 ZIP partial의 공간을 제외하고 재개 필요 공간을 계산하는 회귀 검사도 통과했다. FastAPI TestClient의 httpx deprecation 경고 1개.
- `npm run build`: TypeScript/Vite 성공. `npm run test:ui`: **6 passed**, 기존 3개와 데이터 3개. 실제 대용량 전송 버튼은 누르지 않았다. 시작/중지 UI는 격리된 route fixture에서 검증했다. 실제 API 전체 미리보기 두 건은 전송하지 않은 planned 이력으로 남는다.
- `compileall monitoring/kneeops knee_workspace` 성공. `data status`와 `data cache-descriptor --manifest ...` 실제 실행 확인.
- Kaggle 실제 전송: 470-byte `sample_submission.csv`를 helper 단독 1회, 별도 state의 worker 전체 경로로 1회 확인. 후자는 complete 상태와 최종 파일을 확인했다. 합계 940 bytes이며 원본 영상은 받지 않았다. 진단 폴더 `.local/data-download-smoke-20261001`, `.local/data-worker-smoke-20261001` 보존.
- `docker build -f monitoring/Dockerfile -t kneeops-local:0.1 .` 성공. 비root 컨테이너의 `/downloads` 쓰기·삭제 검사 성공, `docker compose config --quiet` 성공. smoke 컨테이너는 `--rm`으로 제거되었다. 실서비스 컨테이너를 별도로 띄우지 않았다.
- 현재 DB integrity_check=ok와 online backup 생성 확인.
- 화면 `.local/ui-data-preview.png`, `.local/ui-data-mobile.png`; 실제 in-app 브라우저에서도 가독성 확인.

## 운영과 다음 작업

`KneeOps.cmd` 또는 `.venv/Scripts/python -X utf8 -m monitoring.kneeops serve --port 8765`. 기본 로컬 주소 `http://127.0.0.1:8765/#data`.

다음 성능 작업은 캐시의 20~50 study 시각/라벨 점검과 grouped fold 기준 실험이다. Gold58은 노출된 진단셋으로 취급한다. 공개 코드 점수 주장과 검증한 제출 점수를 구분한다. 테스트의 Report 부재와 전처리/추론 계약 일치를 유지한다.

한계: 다운로드는 앱 실행 중만 가능. 원격 ETag/Range 조건이 안 맞으면 partial부터 이어받지 않고 다시 받을 수 있다. 전체 ZIP의 실제 크기는 미확인이고 자동 unzip은 하지 않는다. 동명 외부 파일의 자동 재사용/자동 삭제는 없다. hard link 지원 파일시스템(NTFS/ext4)을 사용한다. UI에서 Kaggle 원격 전처리 Notebook을 생성·실행하거나 직접 GPU 학습을 시작하지 않는다. 캐시 연결은 학습 코드 변경/실행을 대신하지 않는다.
