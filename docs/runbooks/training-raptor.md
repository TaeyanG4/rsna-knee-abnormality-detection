# Raptor baseline 로컬 학습

2026-10-03 갱신. 이 문서는 dreaddevelopment의 Raptor 레시피(CoAtNet, 2.5D 3-slice 창, 소견별 attention pooling, soft 보고서 라벨)의 도구 사용법이다. 전체 전략은 [ADR0003](../decisions/0003-baseline-and-route-to-first.md), 검증은 [ADR0004](../decisions/0004-validation-protocol.md)를 따른다. Raptor만 학습하거나 공개 스택 전체를 교체하는 결정은 아니다. 모든 명령은 워크스페이스 루트에서 **GPU Python(system 3.12, torch 2.5.1+cu121)** 으로 실행한다. 대시보드 `.venv`가 아니다.

현재 구현: strict runtime·입력 seed·checkpoint norm/window 계약·학습/평가 라벨 분리·고정 최종epoch·DICOM serving adapter를 추가했다. [실행 우선순위와 A/B 사전 조건](../research/execution-priorities-2026-10-03.md)을 함께 읽는다. global OOF 라벨 혼합의 간접 누출은 이 코드 수정으로 해소되지 않는다. outer/inner teacher 계보 검증은 미완료이며 새 평가라벨도 독립적인 임상 정답은 아니다. 일반 `raptor_report.py`는 study bootstrap, 이번 사전 고정 A/B용 `tools/input_ab_report.py`는 report-group bootstrap을 제공한다. 변경 전 문서는 [원문/hash](../handoffs/archive/snapshots/2026-10-03-execution/manifest.json)에 보존했다.

## 계약(contract)

corpus는 study당 고정 slot 스택이다. 체크포인트는 자기 corpus 계약에서만 의미가 있고 모든 산출물에 계약 이름을 기록한다.

| 계약 | slot(평면, Fluid, 장수) | 크기 | crop | span | 공개 arm |
|---|---|---|---|---|---|
| `raptor64_336_c140_s02` | Sag Fluid1 18 · Sag Fluid0 14 · Cor Fluid1 12 · Cor Fluid0 8 · Ax any 12 | 336 | 140 mm | 2~98% | maxspan-v5 (학습 baseline) |
| `raptor64_384_c140_s02` | 같음 | 384 | 140 mm | 2~98% | native384dense-v10 |
| `raptor44_384_c140_s06` | 12·10·8·6·8 | 384 | 140 mm | 6~94% | native384-v8 |

정규화: 시리즈의 선택 슬라이스 전체 2~98 백분위로 clip → uint8. 반복 선택된 slice도 백분위 모집단에 반복 포함한다. 정렬: IOP/IPP 법선 투영, 없으면 InstanceNumber. 시리즈 선택: 평면 일치 후 Fluid_Sensitive 일치 우선, 없으면 CSV 첫 행. Fluid_Sensitive와 Fat_Suppression은 서로 다른 필드다. 한 시리즈는 한 slot에만 쓴다. 창은 연속된 cache3장이며 원본 DICOM의 물리적 인접3장이라는 보장은 없다.

`legacy`는 기존 창 선택을 유지한다. `global-valid`는 mask가 모두 양수인3장, `within-slot-valid`는 그중 같은slot 안의3장만 허용한다. mask는 정규화 후 픽셀합>0 지표이므로 acquisition 존재와 동일하지 않다. 정책은 checkpoint에 저장하고 추론에서 읽는다. 과거 정책 필드가 없는 checkpoint는 `legacy`로 해석하며, 공개 anchor 노트북 자체는 변경하지 않는다.

## 도구

| 도구 | 역할 |
|---|---|
| `training/src/raptor_corpus.py` | ZIP에서 직접 corpus 생성(압축 해제 없음). shard별 SHA256, `picks.jsonl` 출처, 재개 가능. `--verify`로 재해시. 4,410 study ≈ 9분(16코어, NVMe) |
| `training/src/raptor_infer.py` | 고정 corpus의 공개 arm(`--arm`) 또는 자체 checkpoint(`--ckpt`)를 gold/oof/test/all에 추론. norm/window·UID·finite 검사, 실제 batch·입력 tensor/hash 기록. CPU 확인은 `--device cpu --amp none` |
| `training/src/raptor_runtime.py` | strict/throughput 정책·native BF16 검사·공통 정규화·배치 추론. 실패 시 정밀도 변경/중립값 대체 없음 |
| `training/src/raptor_train.py`·`raptor_data.py` | grouped 학습·공유epoch를 읽는 dataset. `--fixed-epoch`이면 최종epoch만 채택, 미지정이면 development OOF 선택. Gold 선택 금지. 산출물 `model_best.pt`, `oof_best.csv`, `summary.json`, `provenance.json`; SWA는 development 모드에만 생성 |
| `training/src/raptor_serve.py`·`raptor_dicom.py` | 신규UID+test_series+DICOM을 한 study씩 처리. directory 또는 명시한 read-only ZIP/index. 단일checkpoint raw확률·입력/실패 영수증; 제출·blend 없음 |
| `tools/raptor_serving_check.py` | 선택 study의 DICOM→volume/mask를 불변 corpus와 CPU 비교. 불일치/실패는 영수증 후 실패 종료 |
| `training/src/raptor_report.py` | fold별 OOF를 하나로 합쳐 weak/gold AUC 계산, 예측 비교용 CSV |
| `tools/input_ab_report.py` | 사전등록한 A/B의 코드·초기weight·UID·라벨·설정/epoch 계약 검사, target별 Δ·report-group bootstrap·FP16 skip 비율. 자동 학습 확대 없음 |
| `tools/gpu_job.py` | 로그·환경 기록 백그라운드 실행. `--status TAG`, `--list` |
| `tools/gpu_queue.py` | JSON 계획대로 순차 실행(선행 작업 SUCCESS 대기). 계획은 시작 시 메모리에 읽으므로 수정 후 러너 재시작 |
| `tools/rehash_summary.py` | 2026-10-02 이전 summary의 label/split hash를 줄바꿈 독립 방식(`lf-v2`)으로 재계산 |
| 원격 GPU | Colab·Kaggle 노트북 실행은 [remote-gpu.md](remote-gpu.md) |

```powershell
python training/src/raptor_corpus.py --contract raptor64_336_c140_s02 --workers 10
python training/src/raptor_infer.py --arm maxspan-v5 --subset gold --amp fp16
python training/src/raptor_train.py --help
python training/src/raptor_serve.py --help
```

새 학습은 중복 실행하지 않고 사전등록 manifest/큐의 새tag·제외명세를 사용한다. 이번 대조의 핵심 옵션은 `--select oof --epochs 8 --fixed-epoch 8 --amp fp16 --eval-amp fp16 --runtime-policy strict`,224용 arch와 `--res 224`다. `--labels`는 학습 target·pos_weight, `--eval-labels`는 고정 OOF reference이며 별도hash를 기록한다. `--exclude-ids`에는 두arm 공통 품질 격리 CSV를 준다. 자세한 run명세·상태는 [인계 색인](../handoffs/INDEX.md)을 따른다.

`--fixed-epoch`은1부터 세며 `--epochs`와 같아야 한다. 이 모드의 `model_best.pt`는 이름과 달리 AUC 최고epoch가 아니라 사전 고정한 마지막epoch다. 매epoch OOF/Gold 채점과 SWA 선택을 하지 않고, 마지막에 OOF 및 Gold 진단을 출력한다. `--select gold`는 거부하며 `--fold -1`은 fixed epoch 없이 거부한다. 고정epoch가 없으면 OOF를 사용한 개발 선택이라는 한계가 남는다.

장시간 GPU 작업은 `gpu_job.py`/승인된 큐로 백그라운드 실행하고, ID·로그·ETA와 같은 채팅 완료확인 예약을 남긴다. 완료를 기다리는 반복문을 만들지 않는다. [도구 운영](tools.md)을 따른다.

## 검증 규칙

- **parity 먼저.** 공개 maxspan-v5의 과거 Gold58은 fp16 0.9198, fp32 0.9199, 헤더0.9214였다(2026-10-02). 큰 이탈은 점검 신호이나 aggregate AUC가 비슷하다고 corpus가 동일한 것은 아니다. 입력·study별 예측도 비교한다. 기존 compact cache의 과거값은0.864였다.
- **정밀도.** 이번 자체 A/B는 로컬도 학습/평가 FP16으로 맞춘다. CLI 학습 기본값은 여전히 BF16이므로 `--amp fp16 --eval-amp fp16`을 명시한다. 과거 동일checkpoint의 로컬fp16↔fp32 최대차0.0007은 참고 관측이며 새 후보의 허용오차가 아니다. 공개 anchor의 정밀도는 일괄 변경하지 않는다.
- **분할.** `EXP026_GROUPED_FROZEN_PROBE/folds.csv`(보고서 hash 그룹, 5×869). fold −1은 Gold58 보고서 중복 4건이며 학습·평가에서 제외한다. Gold58은 학습 금지·epoch 선택 금지.
- **비교.** 같은 평가라벨 hash·split/UID·target12개·평가 mask/변환 계약에서 비교한다. 학습라벨이 달라도 같은 고정reference로 평가할 수 있으나 teacher 계보와 checkpoint 선택 독립성은 별도로 확인한다. 라벨/split hash는 LF 고정(`lf-v2`)이며 새 결과는 `eval_label_hash`도 기록한다. 기존 report 기본값은 학습라벨이므로 `raptor_report.py --labels`를 명시하거나 사전등록 A/B 전용 report를 쓴다.
- 16 GB에서 384px는 `--bs 4 --accum 2 --grad-ckpt`(약 15 GB). OOM이면 `--bs 2 --accum 4`. 224 px R1(bs 8, k 12)도 `--grad-ckpt` 필수(T4 실측 최대 11.9 GiB, 끄면 bs 8·k 8도 T4 OOM).
- CoAtNet rmlp는 해상도 고정: 384 px는 `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 224 px는 `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`. 학습 시작 시 preflight가 불일치를 막는다.

## 현재 구현과 T4·4070 확인 범위

13시 이후 보완한 자체 도구의 현재 동작은 다음과 같다. 코드 구현은 실제 T4 재검증·성능 개선 완료와 구분한다.

- `--runtime-policy strict`는 cuDNN benchmark를 끄고 deterministic 알고리즘을 요구하며 TF32를 끈다. `throughput`은 benchmark를 허용하지만 TF32는 끈다. requested/effective AMP와 실제 backend 설정을 기록하고 미지원 native BF16을 거부한다. CPU는 `--amp none`을 명시한다.
- 입력 seed는 SHA256(seed,epoch,UID,stream)이며 창과 intensity RNG를 분리했다. 공유epoch tensor로 persistent worker에 epoch를 전달한다. 같은 입력 샘플을 재현하기 위한 것이며 서로 다른 GPU/라이브러리의 비트 일치를 보장하지 않는다.
- infer/serve는 checkpoint norm·window policy를 전달하고 입력 UID별 tensor SHA를 기록한다. RuntimeError·NaN/Inf를 FP32 재시도 또는0.5로 숨기지 않는다. norm이 없는 과거checkpoint만 `legacy_imagenet_assumption`을 기록한다.
- 학습 loss·평가 출력의 nonfinite는 중단한다. FP16 overflow에 따른 optimizer skip과 실제 update 수를 기록하고 scheduler도 skip한다.10회 연속skip은 중단하며, 누적비율1% 기준은 A/B 결과검토 gate다.
- 원격 생성기는 `--amp bf16`과 `--amp=bf16` 모두 거부한다. 원격라이브러리 설치 전체를 고정한 것은 아니므로 실제 버전 영수증과 T4 parity가 계속 필요하다.

다음은 **2026-10-03 12:58 KST 수정 전 감사의 당시 관측**이다. [실측·환경·코드 hash](../handoffs/evidence/2026-10/2026-10-03-gpu-parity/verification.json), [재계산 코드](../handoffs/evidence/2026-10/2026-10-03-gpu-parity/audit.py).

| 확인 | 결과 | 한계 |
|---|---|---|
| 과거 공개 maxspan의 FP16/BF16/FP32 | 로컬4070의 정밀도 검사 | 실제 T4 비교가 아님 |
| 과거 T4 smoke→로컬 CPU FP32,58건 | 최대 확률차0.004960·평균0.000363 | 200study·1epoch 시험 모델 |
| 이번 R2 F0 T4 보존 예측→로컬4070 FP16,869건×12 | 최대0.0004814·평균0.00004228·p99 0.00020643, 최소 소견별Spearman0.99999854. 고정 원라벨 AUC0.89704585→0.89703992 | 동일 run의 보존 checkpoint. T4 평가batch4/로컬infer batch1·라이브러리 차이 포함, 순수 GPU 효과 분리 아님 |

이번 로컬 추론66.8초, fallback0·모든 확률finite·UID/12열 일치. T4는 새로 실행하지 않았고 기존 결과를 사용했다. 원격 당시 checkpoint SHA·study별 tensor SHA 영수증이 없어 연결 근거는 보존 run과 메타데이터다. 새 로컬 checkpoint SHA는 기록했다. 이 검사는 모델 품질 개선이나 독립 OOF를 입증하지 않으며 다른fold·hidden DICOM·전체0.943 스택의 검증을 대신하지 않는다. 사전 허용오차가 없었으므로 관측값을 새 합격선으로 소급 사용하지 않는다.

현재 R2 F0/F1은 T4·Torch2.10/timm1.0.26, F2는4070·Torch2.5.1/timm1.0.19, F3/F4는T4·Torch2.11/timm1.0.29다. R1은 로컬 BF16, R2 원격은 FP16이다. fold 점수 차이에 데이터·라벨·장비·정밀도·라이브러리 효과가 함께 들어갈 수 있다.

당시 확인한 time_ns RNG·norm 미전달·FP32 fallback·backend 미고정·BF16 옵션검사 공백은 위의 자체 도구에서 보완했다. 과거 산출물은 수정하지 않았다. 당시17개 summary의 norm은 모두 imagenet이므로 norm 결함이 기존 점수를 바꾸었다고 단정하지 않는다. 기존 R2의 간접누출·라벨/epoch선택 문제와 당시 실행환경 차이는 그대로 남는다.

**후속 후보의 제출 전 절차:**

1. checkpoint·코드·입력 UID/순서·실제 전처리 tensor hash·resize/norm/창·batch·TTA·정밀도와 라이브러리/TF32/cuDNN 정책을 명세한다. 공통 입력을 양쪽에서 확인해 데이터 차이를 먼저 분리한다.
2. 재학습 없이 같은 checkpoint의4070 FP16과 실제 T4 FP16 예측을 비교한다. 로컬 BF16/FP32는 보조 진단이다. 원본 출력과 별도로 결과를 저장하고, UID누락/중복·NaN·fallback은 먼저 해결한다.
3. 확률 최대/평균/p99 차이, 소견별 rank/tie·AUC 변화를 함께 기록한다. 허용오차·중단 기준은 결과를 보기 전에 정하며, 기준 초과 시 입력→라이브러리→정밀도/커널을 분리해 조사한다. Gold 점수로 정밀도를 골라 성능을 최적화하지 않는다.
4. 자체 후보의 최종 평가를 제출 FP16 경로로 다시 산출한다. 학습 BF16을 일괄 금지할 필요는 없으나 A/B는 우선 동일 GPU·환경·학습 정밀도로 맞춘다. 학습법의 차이는 추가 seed로 확인하고 추론 수치 호환성과 구분한다.
5. 공개0.943의 정밀도·TF32 설정은 보존한다. 전체 스택은 기존 T4 실행 경로에서 arm별 입력·출력·최종순위를 확인한다. 이번 R2 검사를 공개 스택 검증으로 확대 해석하지 않는다.

새 strict 경로로 실제 T4를 실행한 검증, R2 전체5fold와0.943 전체스택의 수치 호환성, hidden 전체실행시간 검증은 아직 완료되지 않았다.

## 신규 DICOM serving: 실측과 다음 확인

`raptor_serve.py --ckpt <checkpoint> --test-csv <test.csv> --series-csv <test_series.csv> --dicom-root <test_series> --out <새 raw.csv> --amp fp16`으로 사용한다. `<…>`는 실제 경로로 바꾼다. ZIP 비교는 `--dicom-root` 대신 `--archive <ZIP> --archive-index <SQLite>`를 사용한다. 학습study 비교는 명시한 UID/series와 `--split train`을 사용하며 train 라벨·Report를 읽지 않는다. 기존출력은 거부한다.5fold 통합decode·blend·제출파일 생성 기능은 아직 없다.

adapter는 기존 fixed64 corpus 전처리를 재구현한 것으로 공개 anchor의 dense96 전처리와 같다는 주장이 아니다. metadata에 해당slot이 없으면 계약에 따른0 padding을 유지한다. acquisition이 명시됐는데 파일/header/pixel 읽기에 실패하면 해당study와 원인을 기록하고 중단한다. acquisition 부재와 실패를 같은0.5로 바꾸지 않는다. 한study의 픽셀만 캐시하며 원본/corpus를 수정하지 않는다.

| 확인 범위 | 결과 | 한계/근거 |
|---|---|---|
| 공개test3, R2F0, 로컬FP16·batch1 | DICOM/cache 입력tensor hash 및 checkpoint hash 동일, 예측 최대차0·fallback0. 추론구간 DICOM4.077초/cache0.646초 | [비교](../../artifacts/predictions/serving-20261003/sample3-comparison.json)·[DICOM영수증](../../artifacts/predictions/serving-20261003/R2F0_test3_dicom.json)·[cache영수증](../../artifacts/predictions/serving-20261003/R2F0_test3_cache.json).3건의 AUC/전체runtime 검증 아님 |
| 개발train100, CPU 전처리 |99건 volume/mask byte일치,1건 ZIP CRC오류로 strict 실패. 총83.71초 | [대표100건](../../artifacts/predictions/serving-20261003/cpu-parity100.json).series 다수10건·mask 적은10건·고정seed 나머지 표본이며 전체분포 보증 아님 |
| 별도 기존 zero-volume study | ZIP local-header 오류로 실패 | [표준ZipFile 재현](../handoffs/evidence/2026-10/2026-10-03-priority-execution/archive-failures.json). CRC건과 구분 |

100건 전체통과나 hidden-serving 완료로 표현하지 않는다. 신규UID 경로의 정상입력 연결은 부분 검증됐으며, 손상 입력을 부분적으로 복구할 정책·실제 Kaggle T4·5fold decode공유·전체시간/메모리 검증은 후속이다.

## 알려진 데이터 사실

study `<study-uid>`는 기존 corpus에서 전부0인 volume이며 이번 명시적 ZIP 읽기에서 local-header 오류를 확인했다. `<study-uid>`은 별도 CRC오류다. 첫 A/B는 두study를 양쪽arm에서 동일하게 사전 격리한다. 두 오류 모두 표준ZipFile에서 재현됐고 원본은 보존한다. 기존 `legacy`는 빈mask에 앞3장 fallback이 있으나 새 valid 정책은 유효triplet이 없으면 거부한다. 이 corpus의 mask 장수 과거 집계는64장3,163·56장1,122·50장117 study이며 전체 원본 무결성 검사로 해석하지 않는다.
