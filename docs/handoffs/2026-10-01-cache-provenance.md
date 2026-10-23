# 2026-10-01 캐시 제작 출처와 직접 링크

사용자가 캐시의 공식 여부/제작자를 물으며 직접 링크를 요청했다. 캐시 카드에 출처와 링크를 추가했고 다운로드 서버를 재시작하지 않고 정적 웹 빌드로 반영했다.

## 확인된 출처

- Adjacent manifest SHA256 `7b74403d1696eb3e1376ace935903b72f962128d07b987f86afadec95c952c71` → `taeyangg4/rsna-knee-exp034-adjacent-cache-4x9x288`.
- Orthodiffusion 두 폴더의 동일 manifest SHA256 `6ec26d6b5295133ad9f77cf5a543641b901bccbd4f22427a9c014e00cf0f8020` → `taeyangg4/rsna-knee-exp032-orthodiffusion-cache-3x16x256`.
- 둘 다 공식 대회 DICOM으로 만든 자체 전처리 캐시다. 원래 Kaggle 패키지는 비공개로 기록되어 있고 제작/실행 계정은 taeyangg4다. 개별 사람/AI 작성자는 미확인이다. 공개 연구 OrthoDiffusion의 `lt-0123`를 캐시 제작자로 표시하지 않는다.
- 과거 `EXP032_ORTHODIFF_3X16X256_CACHE` 패키지는 manifest contract/shard 파일명/구성이 현재 캐시와 다르다. 폴더 이름만으로 이전 노트북 링크를 붙이지 않았다. 현재 두 폴더에는 같은 최신 출력 로그/명세가 있다.
- 로컬 `kernel-metadata.json`, 생성 코드, 로그의 contract, manifest를 대조했다. 원격 노트북의 현재 공개 범위나 실행 버전 번호는 조회하지 않았다. 공식 원본 링크와 참고 GitHub 커밋 링크도 명확히 구분했다. 원본 캐시·패키지·로그는 읽기만 했다.

## 구현

- `monitoring/kneeops/cache_origins.json`: manifest SHA별 출처, 직접 URL, 관측일, 생성 코드/metadata/log 경로와 SHA. workspace URI 사용.
- `data_cache.py`: 캐시 검사와 학습 입력 descriptor에 provenance 추가. SHA가 등록되지 않은 캐시에 계정/출처를 추정하지 않는다.
- `monitoring/web/src/CacheOrigin.tsx`, DataPage/data.css: 비공식 자체 생성 표시·생성 계정·노트북/원본/참고 링크·비공개 접근 설명. 같은 registry를 웹 번들에 포함하므로 실행 중인 이전 API가 provenance 필드를 아직 반환하지 않아도 UI에 표시된다.
- tsconfig JSON import, pyproject registry 패키지 포함. 기존 manifest/영상/캐시 연결 설정은 수정하지 않았다. 중복 캐시의 삭제·병합·전면 해시는 하지 않았다.
- README/STATE/training-data runbook 갱신.

## 검증과 다음 작업

- `.venv/Scripts/python -m pytest monitoring/tests/test_data.py -q --basetemp .local/tests/cache-origin`: **22 passed**, 기존 httpx deprecation 경고 1개.
- `npm run build`: TypeScript/Vite 성공.
- `npm run test:ui -- tests/cache-origin.spec.ts`: **2 passed**. 실제 세 카드의 계정·직접 링크, 같은 Orthodiffusion 생성 노트북, 비공개 설명, 모바일 overflow, 같은 이름이지만 미등록 SHA인 캐시의 출처 미확인을 검증했다.
- `.local/ui-cache-origin.png`, `.local/ui-cache-origin-mobile.png` 시각 검토. 모바일 캡처는 sidebar 전환 완료 후 수행한다.
- 대용량 다운로드는 같은 작업 `f80a6b01-15ac-43d7-837c-718505001392`의 running을 확인했다(관측 약 192.8 GB). 중단/재시작하지 않았다. 이후 상태는 API에서 확인한다.

서버 재시작을 하지 않았으므로 현재 프로세스의 API/학습 입력 JSON에는 새 provenance 필드가 아직 없을 수 있다. 다음 정상 시작 시 반영된다. 현재 CLI 프로세스와 웹 화면에는 반영된다. 다운로드 완료 전에 이를 위해 서버를 재시작하지 않는다. Docker 재빌드는 하지 않았다.

사용자에게는 공식 배포 캐시가 아니라 taeyangg4의 자체 노트북 생성물이라는 결론과 직접 링크를 설명한다. 다운로드 완료 후 압축 해제/원본 연결 작업은 [다운로드 인계](2026-10-01-download-in-progress.md)를 계속 따른다.
