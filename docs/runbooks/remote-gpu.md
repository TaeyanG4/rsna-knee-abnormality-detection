# 원격 GPU 사용 (Colab · Kaggle 노트북)

2026-10-03 사용자 방침: 로컬 우선, 필요시 무료 Kaggle·보유 Google Plus Colab 사용, 더 필요하면 유료 비용 제안. 아래 Colab 보류는 당시 기록이며 현재는 조건부 활용 가능하다. 표의 쿼터/CU는 현재 잔여량이 아니므로 실행 시 확인한다.

10-03 후속 지시: 중단 전 [지속 연구](continuous-research.md)를 이어가되 EXP078의4fit·gate를 유지하고 결과 뒤 다음 실험을 등록한다. 근거 있는 서로 다른 가설에 하루 최대5회 제출 기회를 활용하며 제출·공개는 매회 승인이다. 횟수 소진·public 비율 반복튜닝은 하지 않는다. [변경 전 원문](../handoffs/archive/snapshots/2026-10-03-continuous-research/manifest.json).

2026-10-02 검증 기록. 로컬 4070 Ti SUPER는 한 번에 학습 하나만 돌린다. 같은 학습 도구를 원격 GPU에서도 사용할 수 있다. 당시의 라벨 병렬 비교·최종5fold×2seed는 사용 예시이며 자동 실행 계획이 아니다. 현재는 사전등록한 비교나 실제T4 serving처럼 목적·비용이 분명한 작업에만 원격을 쓰고 실제 할당을 먼저 확인한다.

## 자원과 실측

| 자원 | 한도 | R1 fold 하나(224 px, 8 epoch) | 맞는 용도 |
|---|---|---|---|
| Kaggle 노트북 T4 | 주 약 30 GPU시간, 세션 12 h | 약 3.3 h(0.38초/study 실측, 최대 11.9 GiB) | 주 최대 약 9 fold, 토큰 불필요 |
| Colab L4 (Google AI Plus 50 CU/월) | 약 2 CU/h → 약 25 h | 미측정. **2026-10-02 사용자 결정으로 보류** | 병렬 라벨 비교 |
| Colab A100 / T4 | 약 15 CU/h / 약 1.2 CU/h | | A100은 급할 때만 |
| 로컬 4070 Ti SUPER | 상시 | | 384 px, corpus 생성, 평가·보고 |

T4·P100에는 bf16이 없다. Kaggle 노트북 생성기는 `--amp fp16`을 강제하고, Colab 노트북은 GPU를 보고 bf16/fp16을 고른다. 224 px R1 설정(bs 8, k 12)은 `--grad-ckpt`가 필수다. 끄면 T4에서 bs 8·k 8도 OOM이었다.

## 데이터 (전부 비공개 Kaggle 데이터셋)

| 데이터셋 | 내용 | 갱신 |
|---|---|---|
| `taeyangg4/knee-raptor-tools` (1.2 MB) | 학습 스크립트, 라벨 CSV, folds, study_meta, train.csv(UID+12 라벨, 판독문 제외), `run_remote.py` | 코드·라벨이 바뀌면 `python tools/remote_bundle.py build --kind tools` 후 `publish --kind tools` |
| `taeyangg4/knee-raptor-corpus-336` (31.9 GB) | 로컬 Raptor corpus 그대로(shard 12개+manifest) | corpus 계약이 바뀔 때만 |
| `taeyangg4/knee-raptor-res-<tag>` | Colab 실행 하나의 결과(summary·history·OOF/gold CSV·model_best.pt) | 실행이 자동 생성 |

판독문(Report)은 원격으로 보내지 않는다. 업로드 실측: corpus 1,100초, 결과 293 MB 32초. 새 데이터셋·버전은 Kaggle 처리에 약 3분 걸린다.

## Kaggle 노트북 (토큰 불필요)

```powershell
python tools/remote_bundle.py kernel --tag EXP071_KGL_F1 --train "--fold 1 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --k 12 --k-eval 24 --bs 8 --accum 1 --grad-ckpt --epochs 8 --oof-subsample 200 --select oof --workers 4" --push
python tools/remote_bundle.py fetch-kernel --tag EXP071_KGL_F1
```

`kernel`은 `kaggle/packages/KNEE_RAPTOR_RUN_<TAG>`에 비공개 GPU 노트북을 만들고 두 데이터셋을 입력으로 붙인다. `fetch-kernel`은 상태를 보여 주고, 끝났으면 `results/<TAG>`를 `artifacts/experiments/<TAG>`로 가져오며 노트북 로그를 `kernel.log`로 남긴다. 형식 예시는 `kaggle/packages/KNEE_RAPTOR_REMOTE_SMOKE`.

## D01 serving 확인 (2026-10-03 추가)

[14:11 로컬stage 검사](../handoffs/evidence/2026-10/2026-10-03-continuous-research/d01-package-verification.json)에서 약1.469GB 자산·anchor30셀 불변·CPU ensemble16+serving10+package9 통과를 확인했다. [14:15 비공개 업로드 시작](../handoffs/evidence/2026-10/2026-10-03-continuous-research/upload-launch.json)은 성공 완료가 아니다. 이 관측에서 실제T4 preflight·대표표본/전체runtime은 아직 검증되지 않았다.

`tools/d01_package.py build`가 만드는 비공개 assets·preflight·candidate를 사용한다([명령/전제조건](tools.md)). 업로드 완료·자산hash·실제T4·남은 할당을 확인한 뒤 preflight를1회 실행한다. preflight는FP16 strict·5fold decode공유·resident·batch1이며 sample3 입력hash와F0 수치 parity를 검사한다. 같은assets hash의PASS 출력이 candidate의 전제조건이다. 공개anchor는 바꾸지 않고 R2만 별도 프로세스에서 실행한다. sample3 성공을 전체hidden 실행시간 또는 독립 성능 검증으로 해석하지 않는다.

장시간 업로드/원격실행은 ID·로그·ETA와 같은채팅 heartbeat를 남기고 반환한다. 미완료이면 갱신ETA에 재예약하고 실제 종료/산출물을 확인한 뒤 후속을 이어간다. 비공개 preflight 실행과 대회 제출은 별개이며 제출·공개는 매번 승인을 받는다.

## Colab

1. `tools/colab/knee_raptor_colab.ipynb`를 Colab에서 연다(파일 → 노트북 업로드). 런타임 유형 **L4 GPU**.
2. 보안 비밀(열쇠 아이콘)에 `KAGGLE_API_TOKEN` 추가, 노트북 액세스 허용. 토큰 입력은 사용자가 직접 한다.
3. 셀 1(도구)·셀 2(corpus, 파일 단위로 받아 끊겨도 이어받음)는 세션마다 한 번. 셀 3은 `TAG`·`FOLD`만 바꿔 반복.
4. 로컬에서 `python tools/remote_bundle.py fetch-results`(새 실행 전부) 또는 `--tag <TAG>`.

## 규칙

- 원격 실행도 ADR 0004를 따른다(같은 folds·라벨, Gold58 선택 금지).
- label/split hash는 줄바꿈을 LF로 고정해 계산한다(`hash_scheme: lf-v2`). 그 전 로컬 결과는 `python tools/rehash_summary.py <TAG>`로 맞춘다.
- summary.json `environment`에 GPU·torch·timm이 남는다. 2026-10-02 Kaggle은 torch 2.10.0/timm 1.0.26, 로컬은 2.5.1/1.0.19다. 정밀도·버전이 다른 결과를 섞을 때 확인한다.
- 같은 TAG가 로컬에 있으면 가져오지 않고 로컬 것을 유지한다.
- 모든 노트북·데이터셋은 비공개다. 대회 제출과 공개 게시는 매번 사용자 승인 후에만 한다. 점수 확인은 다음 제출 허가가 아니며 채점은 대기 없이 같은채팅 heartbeat로 확인한다.

### 10-03 실제 실행에서 확인한 packaging 조건

- ipynb의 `metadata.kernelspec.name=python3`와 language 설정을 검사한다. `kernel-metadata.json`의 language만으로 papermill 실행이 보장되지 않는다.
- 상세 헤더 감사는 원본/hash로 보존하고 notebook에는 실행 UID·필수 계약만 넣는다.1.9MB 본문400은43KB축소 후 접수됐으며 서버가 정확한 용량한도를 설명한 것은 아니다.
- 실제T4 정규화tensor 바이트차이를 발견하면 volume/mask→center→resize→normalize 단계별로 진단한다. 기존FAILED를 지우거나 소급PASS로 바꾸지 않는다. 디코더 capability와 실제압축fixture/competition표본 coverage도 구분한다.
