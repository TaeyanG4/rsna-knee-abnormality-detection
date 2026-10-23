# EXP087 같은 정책 입력 staging·B8 adapter — 실행 전

2026-10-04 08:09 KST, dicom_header_io. **CPU 구현 완료, 새로운 실제 header/pixel·model·labels/OOF·GPU 접근0.** 새 cache 실행과 학습은 root의 별도 결정까지 HOLD다. 중앙 HANDOFF/STATE는 수정하지 않았다.

소스는 `workspace://tools/exp087_coverage_stage_v1.py`, adapter는 `workspace://training/src/exp087_coverage_data.py`, 검사는 `workspace://tests/test_exp087_coverage_stage_v1.py`다. 최종 원본·도구·fixture SHA는 `final-bindings.json`, 정확한 명령은 **`freeze-v2/launch.json`**에 있다. 원 v3 helper SHA `b91d32…87a3a8`는 그대로이며 원 pilot/cache 파일을 수정하지 않았다.

## 완료 근거 재사용과 고정 범위

저장된 `policy-count-v1/potential-studies.json` **315건(T252/V63)**을 입력으로 사용했다. 새로운 eligibility census나 다른 대안 선택은 하지 않았다. 원 파일의 raw CRLF SHA `31ded410…b169017`을 결박했다. 기존 result의 LF payload SHA `4a325b16…33c265`와 구분한다.

완료한8 study는 원 `pilot-v1` alternate NPY·mask·native indices를 참조한다. 독립 저장결과 PASS와 16개 기존 NPY 검사 근거를 재사용하고 source binding에 필요한 기존 alternate 파일 hash를 확인했다. **원8 재디코딩 없음.** 나머지307개에 대해서는 저장된 두 series의 metadata를 조회하고 읽기전용 archive index에서 정확한 member path/CRC/offset/size를 고정했다. 원본·대안 metadata의 plane/fluid/FS, 원5-slot picks와 unused 조건을 확인했으며 새로운 후보 탐색을 하지 않았다.

## 제안 cap의 적정성

|항목|고정 메타데이터 관측/계획|제안 실행 cap|
|---|---:|---:|
|새 header member|15,774|65,536|
|새 대안 plane|최대307×12=3,684|3,684|
|대안 selected payload|상한3,046,923,720B|8GiB|
|새 rendered uint8|최대415,908,864B|415,908,864B|
|header parser bytes|아직 미측정|512MiB|
|process tree RSS|아직 미측정|8GiB|
|wall time|아직 미측정|900초|

payload 상한은 각 대안 series의 **크기가 가장 큰12 member** 합이며 실제 native pick이 아직 없을 때의 안전한 상한이다. 실제 압축 디스크 I/O 계측값이 아니다. 기존 pilot의392 headers/192reads/152.7MB/11.281초에 단순 비례하면 약225–454초 규모이나 codec·크기·cache 상태가 달라 예측이나 완료 보장으로 쓰지 않는다. 900초를 넘으면 결과가 있어도 FAIL이며 자동 재시도하지 않는다.

요청 기본12 workers에서 **실제1 worker/internal1**을 제안한다. 성공한 순차 pilot의 처리량이 있고 원 ZIP 한 곳의 읽기·메모리 동시 증가를 피할 수 있다. CPU staging에서는 모델이나 Torch inference가 없다. v3 supervisor 함수 본문을 변경 없이 별도 driver에 재사용했고, 제안 cap만 별도 module에 결박했다. 예외 후 child/descendant 정리와 최종 elapsed 검사는 유지된다. RSS는100ms 샘플 감시로 OS hard quota가 아니다.

## 실제 실행 시 하는 일

exact plan/source·original IDs/masks·원 cache shard stat·315 picks subset을 먼저 검증한다. 새307건은 원본과 대안의 header만 읽어 내부 geometry/SOP를 확인하고 **대안12 planes만 디코딩**한다. 원본 DICOM pixel을 다시 디코딩하지 않는다. source/index/header/parse/decode/budget 오류는 FAIL이다. 정상 geometry 부적격·같은 study의 shared SOP·최종12 planes sampled duplicate·유효 triplet 부재는 사전 reason으로 원 입력을 유지한다. 후보를 다음 series로 바꾸지 않는다.

추가로 읽게 되는 SOP가 현재 G1의 T/V를 연결하면 즉시 FAIL하여 root가 ancestry를 검토하도록 한다. 이는 새로운 exhaustive duplicate census가 아니다. 원본과 sampled duplicate를 비교할 때는 기존 immutable cache slot2만 참조한다. **새307의 원본 DICOM replay exact는 측정하지 않으며**, 원8의 성공을 전체 원본 replay 성공으로 확대하지 않는다.

staging 완료 manifest는8재사용+307결과, fallback reason, NPY/hash/mask/native indices, 전체 G1 membership를 기록한다. train8 epochs/eval K24의 `schedule_counts`는 원 C와 실제 대안mask로 계산한 **예정 치환 수**이며 모든 resized tensor의 실제 차이 수나 성능효과는 아니다. drop_last 후 실제 train fraction도 아니다. 완료 상태 `STAGED_INPUTS_ONLY_NO_TRAINING`은 학습 허가를 뜻하지 않는다.

## 최소 B8 adapter와 comparator

`CoverageStudyWindows`는 기존 `StudyWindows`를 감싼다. 기존 dataset이 원 normalized X·Y·UID를 만들고, 원 C의 slot2 출현 j=1,3,5,…만 대안의 q/half-even center로 치환한다. 레이블 객체·sample당 loss·study order·global RNG·나머지 window를 보존한다. 원 K12/K24·224·ImageNet norm·jitter.20·seed42·within-slot-valid·원 corpus 경로와 정확 G1 순서를 검사한다. store는 immutable manifest와 adapter/helper/원 corpus/IDs/masks/대안파일 hash를 확인하며 Windows DataLoader spawn에 module 또는 parent memmap을 pickle하지 않는다.

합성검사에서 frozen v3 helper의 전체 tensor와 일치했고 train epochs0/3/7, eval, force-original·disabled·fallback, 원 라벨과 RNG, **batch4×K24 collate**, 잘못된 recipe/G1/corpus 거부, pickle 및 cap을 확인했다. 최종 synthetic15개1.83초 PASS이며 변경 없는 저장8 artifact 검사는 이전 PASS를 재사용했다. 최초14개 검사1.82초에는 저장8 검사가 포함됐고, 새 cross-G1 guard도 별도로 PASS했다.

후속 후보는 **기존 B8과 동일 generic ImageNet CoAt initial_state `e4ea3a…5b38e`, G1 T3455/V888, seed42, 8epochs/3448updates, architecture/loss/optimizer/schedule/augmentation/train order, eval batch4/K24/precision/pooling**를 유지해야 한다. 입력 내용만 이번 정책으로 바뀌는 경우 기존 B8 comparator를 재사용할 수 있다. 이 때문에 자동 control 재학습을 시작하지 않는다. 공유 recipe를 바꾸면 그때 matched control이 필요하고, eval plumbing만 바뀌면 같은 가중치 재추론/동등성 문제다. 실제 training runner 통합·학습 등록·모델 로드·GPU 실행은 이번 범위 밖이다.

## 다음과 버전

root가 독립 source/fixture/예산 검토를 회수한 뒤 `freeze-v2/launch.json`의 단1회 CPU 실행 여부를 결정한다. 새 출력 `workspace://.local/exp087-coverage/stage-v1`은 아직 없다. 완료/실패·비용·실제 유효315 map을 판정한 뒤에만 학습 투자 결정을 한다. whole315 입력 tensor와 B8 GPU/eval runtime 동등성이 이미 검증됐다고 주장하지 않는다.

미실행 `freeze-v1`은 보존했다. Windows pickle·adapter/corpus 결박을 보강하고 v2를 만들었으며 **315 rows/source-member maps를 그대로 재사용**했다. index/eligibility를 재열거하지 않았다. 작은 입력 비중(<1% 예정 windows)은 작은 효과의 보장도 큰 효과의 근거도 아니다. G1은 노출된 weak 개발검증이고 임상/private/.943/D01 개선이나 T4 parity를 증명하지 않는다. M02/M03/M04/M05/M07/M09/M10/M11/Biohub의 재사용·누락·비용·실패 보존을 적용했다.
