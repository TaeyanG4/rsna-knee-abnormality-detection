# Public solution / artifact lineage

## 조사 범위와 증거 수준

공개 Notebook 24개를 로컬에 확보했다. 원본 Notebook, Kaggle metadata, 추출한 Python, Markdown 설명, 설정 관련 행을 보존했다. 숫자 Notebook version을 확인하지 못한 항목은 원본 파일 SHA256으로 고정한다. 소스 확보, 코드 검토, 모델 실행, Public LB 재현은 별도 상태다.

`public_solutions.csv`는 공개 Notebook 목록이며 `experiments/notebook_manifest.csv`에는 자체 진단 Notebook이 포함될 수 있다. 원래 22개 소스에 대한 `metadata/notebook_similarity.csv`에는 231개 쌍의 비교가 있다. 뒤에 추가한 Multi-plane DINO와 Read the report Notebook 두 개는 그 유사도 파일에 아직 포함되지 않는다.

## Raptor: 네 개 arm이 네 개 독립 모델은 아니다

현재 확보한 Mattia Dinosaurs 소스의 내부 Raptor 구성:

| Arm | 실제 checkpoint | 비중 | 저장/encoder 해상도 | Study slice | Windows | Source-depth span |
|---|---|---:|---|---:|---:|---|
| MaxSpan v5 normal | raptor_ft_coatnet_v5_full_swa.pt | 0.60 | 336 / 384 | 64 | 62 | 0.02–0.98 |
| MaxSpan v5 reverse | 같은 v5 checkpoint | 0.10 | 336 / 384 | 64 | 62 | 0.02–0.98 |
| Native384dense v10 | raptor_ft_coatnet_v10_full.pt | 0.10 | 384 / 384 | 64 | 62 | 0.02–0.98 |
| Native384 v8 | raptor_ft_coatnet_v8_full_swa.pt | 0.20 | 384 / 384 | 44 | 42 | 0.06–0.94 |

모든 arm의 nominal crop은 140 mm다. v5 normal/reverse는 같은 weight를 공유하므로 모델 다양성이 아니라 inference TTA다. v8과 v10은 실제 SHA256이 다른 checkpoint이며, v8의 44-slice 계약을 v10의 64-slice 계약과 혼동하지 않는다.

코드 근거: `kaggle_notebooks/mattiaangeli__bend-the-knee-to-the-dinosaurs/embedded_26__KE_SRC.py`, ARMS, `_eval_centers`, `eval_windows`, `_make_reader`.

실제 weight hash:

| Checkpoint | SHA256 |
|---|---|
| Raptor v5 | 20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a |
| Raptor v8 | 3db77388662d516bb0ce790dafd24e36f4991f6089af26daf546817fe1c21862 |
| Raptor v10 | 5add8816a47e040efca89bb44480e1cbd01f92db1253831ff7ceb6c65ee02b1d |

## CoAt residual-gated branch

동일 학습 run의 epoch 4, 6, 8 세 checkpoint다. 세 개 독립 fold 또는 서로 독립적인 training run으로 세지 않는다. `training_history_through_epoch10.json`은 Gold58 macro AUC를 early-stop/selection 지표로 사용했음을 명시한다.

| Epoch | SHA256 |
|---|---|
| 4 | cefaf007415b5e45ce2a720d5e211a183a4e90f160710ba1f72b1ce7c46239a2 |
| 6 | 56eca1f61651362ecfae8aecdd9327125713b95653344b8ad62ef8cff17d08f4 |
| 8 | 8d3159102046c127dc0d084fb69c9304ffcbd4b98d1333f9811d4d613fea1b12 |

세 파일 모두 게시자 manifest와 직접 계산한 SHA256이 일치한다. 구조는 CoAtNet 384 + bounded spatial residual + finding-specific MIL이다. Inference는 130 mm, within-series triplet, 6개 slot의 budget 20/12/12/8/16/8, 최대 76 windows다.

학습 label SHA는 `3b360a6a029ee1864324fba0e895dd84065351495ec6bcff109cd5d2888c888f`이며 공개 label CSV 중 어느 생성 버전인지 완전히 대응시키지는 못했다. loss는 confidence-normalized BCE와 macro pairwise ranking 결합으로 기록되어 있다. Gold58로 checkpoint를 골랐으므로 gold 점수와 bootstrap은 독립 holdout 검증이 아니다.

최신 Mattia 소스는 이 CoAt3 branch와 Raptor family를 결합한 뒤 DINO/legacy/Rad 계열과 추가 blend한다. 전체 결합식은 원본 Notebook에 고정했다. 전체 original pipeline을 한 GPU에서 일부 arm을 생략한 코드로 대체하지 않았다.

## DINO / medical pretraining / task-specific components

Pilkwang 20-member manifest와 OOF는 확보했다. 20개 checkpoint 전부의 로컬 weight SHA 확인 또는 자체 DINO inference는 아직 하지 않았다.

RadImageNet encoder와 여러 head는 backbone 공유 가능성이 있으므로 이름이 다른 Notebook 수를 독립 모델 수로 세지 않는다. 연결된 변환 encoder Dataset은 CC-BY-NC-SA4 표기여서 대회 및 winners 공개 조건과의 적합성을 확인하기 전 실행 보류했다.

별도로 코드 검토한 구성요소는 Raptor finding MIL, CoAt spatial residual, DINO CLS+patch pooling, RadImageNet image features/heads, Multi-plane pathology query fusion, Meniscus residual specialist, multilingual report-teacher pipeline이다. 이것이 독립적인 완성형 강자 7개를 모두 실행했다는 뜻은 아니다.

## Multi-plane DINO의 중요한 예외

`abisheksrivastav/rsna-knee-multi-plane-dinov2`는 336 px, 130 mm, 3-slice group, 6 canonical slots, learned target queries를 사용한다. DICOM SeriesDescription/SequenceName/ScanOptions/TR/TE를 읽어 fluid weighting과 FS를 복구한다.

공식 CSV에서 Fluid_Sensitive와 Fat_Suppression이 전 행 같다는 사실만으로 이 Notebook의 fluid/non-FS slot이 항상 비었다고 단정하면 안 된다. `series_protocol`의 header 기반 override를 확인했다.

## Source similarity와 방향성

ask812 / prvsiyan head-and-shoulders, renta weak-meniscus / hyakuman / Kelvin, renta hybrid / vaihero 일부 쌍은 추출 코드 line-set Jaccard 1.0이었다. 이는 코드 유사성이지 input Dataset 버전, weights, 실행 결과까지 동일하다는 증명은 아니다. copied-from 방향은 명시적 근거가 없는 경우 미확인으로 남긴다.

## 다음 geometry audit 후보: cross-slot triplet

Raptor의 `_eval_centers`는 concatenated study stack의 전체 valid min/max 범위에서 center를 만든다. 각 center의 세 slice가 같은 series/slot인지 확인하는 분기는 없다. 모든 64 slice가 유효하고 전체 62 center를 평가하면 네 slot 경계 각각의 양쪽 center, 즉 8/62 windows가 경계를 걸친다.

이는 코드에서 유도한 조건부 관찰이다. 실제 training sampler가 같은 계약을 사용했을 수 있어, 바로 제거하면 train/inference mismatch가 생길 수 있다. 아직 수정하지 않았고 AUC 개선을 주장하지 않는다. 원본 재현 후 mask/slot-boundary별 예측과 runtime을 분리하는 후속 실험 후보로만 둔다.

## Public score의 의미

0.941이 포함된 제목은 직접 확인한 Public LB 숫자로 취급하지 않는다. indexed Raptor V1의 0.933과 이전 Full4arm V56의 0.936은 외부 공개 score 증거이며 우리 계정의 재현 결과가 아니다. 136–173초의 공개 example run 시간을 hidden 1,300 study 실행 시간으로 사용하지 않는다.
