# EXP087 CPU 입력 검증 구현 — 실행 전 기록

2026-10-04 07:16 KST, producer `dicom_header_io`. **구현·메타데이터 freeze 완료, 실제 영상 디코딩은 HOLD**다. root의 독립 소스 검토 회수와 별도 실행 결정이 남았다. 중앙 HANDOFF/STATE는 수정하지 않았다.

- 소스: `workspace://tools/exp087_series_coverage.py`
- focused tests: `workspace://tests/test_exp087_series_coverage.py`
- 최종 fixture/plan/정확 argv: `freeze-v2/fixture.json`, `execution-plan.json`, `launch.json`
- 최종 테스트: **20 passed, 1.59s**, `focused-tests-final.xml`. 전부 합성 데이터이며 실제 DICOM·cache pixels·모델·라벨/예측 파일 접근은 0이다.

## 검증 범위

기존 `StudyWindows`와 train epochs0/3/7 및 eval의 원본 입력을 byte 단위로 비교했다. enabled/disabled/forced-original과 absent는 원 C·intensity·window RNG를 보존하며, 선언된 slot2 홀수 출현 위치 외 tensor가 동일하다. Python/NumPy/Torch 전역 RNG도 소비하지 않는다. MONOCHROME1/2, modality slope/intercept가 있는 메모리 내 합성 DICOM은 기존 passive renderer와 volume/mask가 정확히 일치했다. Geometry/중복후보/사용중 series/FS불일치/empty·mask hole·native 반복/byte cap/무승인 실행 차단을 검사했다. 실제 archive codec/픽셀 동등성은 아직 미검증이다.

## 메타데이터로 고정한 표본

G1 T3455 digest 및 V888 digest를 기존 계약과 확인하고, 옛 EXP081의 complete metadata alternate list에서 **lexical 첫 strict alternate**를 먼저 고른 뒤 정확한 두 series의 기존 감사 header가 있는 T study만 사용했다. 최대8개를 얻으면 탐색을 끝냈다. train plan88행을 방문했으며 전수 treatment census가 아니다. v2는 v1의 동일8 UID/series/header를 그대로 재사용했다.

- 8 study / 16 series, 원본+대안 각12 planes
- freeze된 header members392, 예정 unique pixel reads192
- 저장 ZIP member 크기 기준 예정 pixel payload **152,746,838B**; 실제 읽은 양이나 처리시간이 아니다.
- rendered payload 상한21,676,032B; 실제 원본 cache와의 replay/no-op는 후속 CPU 실행에서 판정한다.
- fixture UID digest: `5a11521f8e79e0a3355b7ccf84a875e1ada700cbdca8369ef5478000a74d16d3`

## 실행 계획과 안전장치

`freeze-v2/launch.json`에 interpreter·cwd·placeholder 없는 argv·환경·plan SHA가 고정됐다. 아직 해당 argv는 실행하지 않았다. 출력은 새 `workspace://.local/exp087-coverage/pilot-v1`이며 현재 존재하지 않는다. 이미 존재하면 중단한다. 원본·기존 cache는 읽기 전용이다.

부모 supervisor는 source/fixture/기존 IDs·masks hash와 source stat/pick subset을 확인한다. 실제 child도 exact plan hash·명시 pixel flag·supervisor 전달값을 요구한다. CPU 1 worker / internal1을 사용한다. 요청 기본12 대비 감소 이유는 16개 작은 series의 순차 ZIP 접근과 RSS 상한이며 불필요한 process pool을 만들지 않는다.

시간600초, supervisor+child+descendant RSS4GiB를 100ms 간격으로 감시하고 초과 시 종료한다. RSS는 샘플 사이 일시 초과 가능성이 있는 감시 제한이며 OS hard quota라고 주장하지 않는다. Header2048/64MiB, pixel192/1GiB, rendered21,676,032B를 코드에서 제한한다. 예상 파일 크기를 읽기 전에 charge하고 성공 후 실제 길이 일치를 확인한다. 압축/block I/O와 parser/API bytes를 구분한다. 자동 재시도는 없다.

실행 시 새로운 SOP·header와 기존 감사 metadata를 비교하고 ZIP CRC/offset/size를 검증한다. 원 slot2를 다시 렌더링하여 cache와 byte/mask 동일성을 확인한다. 원 C를 재계산하는 전역 mask 변경은 하지 않는다. 대안이 geometry·SOP 중복·sampled rendered duplicate·유효 triplet 조건을 만족하지 않으면 원 입력을 보존하고 reason을 기록한다. 파일/parse/decode/source/budget 오류는 FAIL이다. 후보 교체 탐색은 없다.

기존 overlay는 최대길이 선택과 이번 lexical 선택이 같다고 가정하지 않았다. 이 pilot은 새 runner의 원본 replay와 raw/decoded provenance·SOP/처리비용을 확인하므로 지정192장 이내에서 재생성하도록 명시했다. header와 원 cache 계약·기존 no-op 근거는 재사용했다. full ZIP/shard 재해시는 하지 않는다.

## 버전·남은 판단

`freeze-v1`은 최초 미실행 계획으로 보존했다. 원 IDs/masks 런타임 hash gate, 결과 범위 표기, global RNG 검사 보강 후 source를 `freeze-v2`에 새로 결박했다. UID 표본·정책·예산은 바꾸지 않았다. source나 tests가 바뀌면 v2도 그대로 실행할 수 없으며 변경 근거와 새 binding이 필요하다.

초기 pytest의 시스템 Temp 권한 오류는 새 workspace temp를 지정해 해결했다. `git status`는 이 경로가 Git checkout이 아니어서 사용할 수 없었으며 파일 SHA로 결박했다. 기존 B config.json 부재는 실제 protocol manifest/provenance의 관련 항목을 읽어 해결했다.

**남은 것:** 독립 소스 검토 → root의 구체 실행 결정 → 최대10분 CPU pilot 1회 → 원본 replay/no-op·실제 대안창 사용·중복·비용 판정. 완료 exit code는 임상 추가정보/현재 G1 prevalence/성능 개선·T4 parity·학습 GO를 뜻하지 않는다. 표본 선택은 T-only지만 G1 자체는 노출된 weak 개발검증이다. full cache·학습·모델·GPU·remote·제출은 이 구현 범위에 없다.
