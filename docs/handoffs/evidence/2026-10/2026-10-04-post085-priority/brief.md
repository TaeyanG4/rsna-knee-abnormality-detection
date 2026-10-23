# EXP085 이후 다음 성능 투자 — 읽기전용 초안

2026-10-04 04:45 KST. 담당 `/root/dicom_header_io`. **다음 최소 실험은 로컬 generic DINOv2-S/14를 현재 B와 같은 영상·8epoch 계약으로 전체 fine-tune하는 단일 후보를 제안한다.** 기존 B 결과를 대조로 재사용한다. 이는 다른 encoder/사전학습 표현의 전이를 묻는 비교이며, 의료 사전학습 효과만 분리하는 시험은 아니다. root 결정 전 구현·실험등록·실행을 하지 않았다.

## 이미 끝난 질문과 해석

EXP085의 treatment−control +.001852, CI[−.002560,+.006766], treatment−원B +.002154, CI[−.003634,+.006973], 고정blend +.002131/CI0포함은 **작은 후보 보존·미채택**이다. 원B 대비 Baker −.009430, CI[−.017828,−.002247]를 같이 보존한다. 상대위치를 다시 제안하거나 seed/fold/384로 자동 확대하지 않는다. control−B도 +.000302, CI[−.005069,+.005357]이므로 단순 추가8epoch가 큰 병목 해법이라는 근거는 없다. 새head+optimizer 재시작이 섞인 대조라 연속B16의 효과로도 부르지 않는다.

post084의 이전 우선안 B는 이 결과로 완료됐다. M02/04/05/06/10 및 Biohub 누락점검을 적용해, 작은효과를 버리지 않되 현재 CoAt 표현·제한된 입력을 고정한 주변 변경만 계속하지 않는다. 앞으로 결과가 음성이어도 해당 레시피와 encoder 계열 전체를 구분한다.

## 서로 다른 네 가설

|우선·가설|근거와 최소 개입|예산·한계 및 다음 조건|
|---|---|---|
|**1. 다른 사전학습 표현이 현재 CoAt의 오류와 다른 정보를 학습한다**|generic DINOv2-S/14 전체 fine-tune 8epoch, 기존 B8과 동일 split/영상/label/학습량·MIL pooling. 로컬 pretrained88MB가 있어 다운로드 없음. 기존 DINO frozen sparse probe와 1epoch280 시험은 이 비교를 끝내지 않았다. 의료 RadioDINO는 같은 계열의 다음 대안으로 보존하되 동시에 돌리지 않는다.|CPU 준비 **45–90분 추정**, candidate GPU **25–70분의 거친 계획값**, 실측 전 보장 아님. 한 T-only pilot로 비용을 고정한 뒤 full≤90분 cap 제안. 아래 단일후보 계약을 먼저 결정.|
|**2. 고정 첫series/제한창에서 빠지는 영상 정보가 병목이다**|동일 plane/FS 다중후보 study1936/4407(43.93%)는 정보손실 가능 범위이며 오류율은 아니다. 후보2개를 예측·Report를 보지 않는 사전 metadata 규칙으로 고르고, 학습 때 series별 균등추출/추론 때 둘 다 집계한다. K 총량을 유지한 coverage 질문 또는 K 증가 질문 중 하나만 먼저 고정한다.|기존 EXP081 max-span 직접교체(변경34/868, Δ−.000684)는 같은 가설이 아니다. 원캐시에 없는 두번째series는 새 파생입력이 필요. header129series/3305개4.33초·준비21.68초는 과거 실측이지만 전체 pixel cache 비용은 **미측정**. CPU설계1–3h+paired B급학습80–100분은 추정, decode/I/O 별도. 지금은1번보다 준비공백이 큼.|
|**3. MRI 자체에 맞춘 사전 적응 레시피가 weak label만으로 학습할 때 잃는 표현을 보존한다**|generic encoder에서 T-only MRI masked-image/동일study view의 자기지도 적응을 한 뒤, 동일한 fixed supervised8epoch로 전이한다. teacher를 쓴다면 teacher와 student 모두 V/연결group 제외. 기존 추가지도8epoch와 달리 label 없이 영상통계에 적응하는 질문이다.|현재 완료된 해당 SSL 결과/구현/실측은 확인하지 못함. 최소 CPU설계1–3h, GPU1–4h **예산가설**이며 projection 아님. 심한 crop이나 서로 다른 병변 view에 강한 일치loss를 강제하면 병변정보를 지울 수 있다.1번 결과와 T-only 짧은 throughput 계약이 생긴 뒤 판단; 지금 구현하지 않음.|
|**4. 실제 정의충돌이 확인된 제한적 감독 개입이 표현개선보다 중요할 수 있다**|보고서 미언급·불확실·임상음성을 분리하고, 새 독립영상근거 또는 명시적 Dread 정의충돌이 있는 target/cell만 원감독 대 제한개입으로 비교. 원 reference 평가는 유지하되 교정된 label을 같은 평가정답으로 쓰지 않는다.|PFOA20·Synovitis16 감사는 완료·자동Dread교정 근거부족. Qwen confidence는 보정 신뢰도가 아니며 source불일치만으로 mask/수정하지 않음. 같은 가림검토·표본확대 제안 없음. 전문가/독립영상 시간·비용은 **미확정**. 새 근거가 생기면 우선순위 상향; 지금 학습보다선행해야 한다는 주장은 하지 않음.|

## 1번 최소 실험의 구체안 — 등록 전 제안

- 초기값은 HF `timm/vit_small_patch14_dinov2.lvd142m`, 로컬 snapshot `4610ca143709d58a633b6397a74412c2c3842454`. 실제 safetensors88,240,510bytes, 이번 파일SHA `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081`. generic external 초기값이며 대회 fine-tuned 공개weights/EXP042 checkpoint를 쓰지 않는다. HF 최신revision/모델카드는 이번에 외부조회하지 않았고 local snapshot에는 config.json이 없었다. timm 명시 arch 및 source lineage를 계약에 고정한다.
- 데이터/대조는 `EXP084_B_G1`의 T3455/V888·group contract·Dread 원soft label·제외UID·캐시 `RAPTOR64_336_C140_S02`·224px·within-slot-valid·학습K12/평가K24를 그대로 제안한다. seed42·fixed8epoch·3448 optimizer updates·원augmentation/학습률/정규화·head형식을 유지하고 encoder+그에 필요한 feature dimension만 바꾼다. 이는 동일 파라미터수/동일FLOPs 실험이 아니고 **encoder-family transfer package** 비교다. ViT는224가patch14로 나누어지고 현재 공용build_backbone이 dynamic_img_size/token pooling을 명시한다. 실제호환성/gradient는 아직 실행 확인하지 않았다.
- 원B의 완료 예측/receipt를 재사용하여 원B GPU rerun0으로 제한한다. primary는 candidate−B의12target macro, 동일V의P/N·Baker/Fracture를 포함한 소견별Δ·group·skip·조건부CI를 mandatory로 둔다. 최종8epoch만 평가하고 중간V/Gold로epoch를 고르지 않는다. 새혼합/12target routing/비율탐색을 primary에 섞지 않는다. RN은 기존 결과 보존이며 DINO 혼합은 별도 질문이다.
- **비용 측정만을 위한 T-only32update pilot 1회**를 제안한다. 평가label/Gold는 열지 않고 초기 state로 재시작할 full과 구분한다. finite/skip·peakmemory·실제 update time·추론4행/종료를 확인한다. `1.3×(정상update초×3448)+초기화/평가예비300초 ≤5400초`일 때만 root가 single full8epoch를 판단한다. pilot≤180초/full≤5400초·retry0를 제안하며, 이 숫자는 아직 등록/실행 승인이 아니다. 상한초과나 overflow는 별도 수정결정 없이 자동batch/precision/epoch 변경하지 않는다.
- 완료 B8은 총2610초(43.50분),EXP085 control2451.563초/treatment2380.094초, supervised전체4839.25초+pilot101.953초 실측을 비용 기준으로 썼다. DINO full시간은 새encoder라 이 값과 같다고 보장하지 않는다. 기존 DINO feature 추출35.514초(64,002triplets),RadioDINO88.828초(192,006독립slices),EXP02732.034초는 **frozen extraction/probe 실측**이며 backbone backward8epoch 시간으로 외삽하지 않는다.
- 동일 G1을 쓰면 gradient상 V-excluded 비교지만 연구자가 이미 본 개발 V다. **새 V를 만든다면 B·DINO뿐 아니라 teacher/student/stacker/calibrator와 대회 적응 encoder의 모든 ancestor가 그 V/연결 group를 제외해야 하며, 기존 B가 그 V를 봤다면 비교군도 재학습**한다. 이번 저비용안은 새 독립검증을 확보했다고 주장하지 않는다. 외부 generic corpus의 patient overlap까지 입증한 것도 아니다. weak 라벨·보고서 group의 환자 독립 미확인·조건부 CI·연구자 노출·0.943 전이 공백은 유지한다.
- Δ>0이 작거나 CI0포함이면 후보/불확실로 보존하고 자동 확대하지 않는다. 기존 B보다 낮으면 이 8epoch 전이 레시피 미채택이며 DINO/의료 표현 전체를 기각하지 않는다. 큰 개선이라도 .943에 더하거나 T4/private 성능으로 옮기지 않는다. 결과와 시간을 보고 root가 다음 한 가설을 결정한다.

## 기존 자산·부정 근거·공개 코드 계보

DINO를 ‘새로운 미시도 이름’으로 제안하지 않는다. EXP026 grouped frozen Ridge(alpha100)는 당시 sparse18triplet/teacher high-confidence 평가에서 weak .780454, EXP027 RadioDINO는 같은 cache-adjacent 입력에서 .750447이었다. 당시 동일계약 비교의 반대 근거는 보존하되 현재 Dread/G1/밀집K24 대조와 직접 점수 비교하지 않는다. EXP042 DINO는280px/1epoch/다른4slot 입력·옛fold/Gold 노출 결과 .687542weak였다. 이 checkpoint/선택 정보를 새 generic 초기값으로 혼입하지 않는다. 같은 frozen probe를 다시 돌리지 않고 이 결과를 재사용한다.

RadioDINO-S16 로컬 자산은86,676,336bytes, revision `773d28d873455153495d39179fc42ea816aa1212`, 보존된 receipt SHA `6f88c66e9155a5d223d8d08a78aebc294e76deede181c4eb8b1895a70af49a01`. 카드가 RadImageNet1.35M CT/MRI/US의 DINO 사전학습·CC-BY-4.0을 선언한다. 이번엔 weight 재hash/모델 load 없음; 카드 주장이 Knee 우월성이나 competition 조건 전체 확인의 증거는 아니다. 공용 runner의 `hf_hub:` 이름만 넘기면 ViT 분류 분기가 맞지 않을 수 있으므로 향후 명시적 구조/strict weight adapter가 필요하다. 이미 있는88.8초 feature와32초 triplet 결과를 버리고 새로운 frozen screening을 반복하지 않는다.

0.943 공개 기준본은 DINOv2 20member/A5 DINOv3-16channel5fold/RadImageNet/CoAt 등의 복합 계보다. generic DINOv2-S 한 모델이 동일 baseline도 아니고 같은 성능을 기대할 근거도 없다. 공개 대회 weight는 새 V 계보가 증명되지 않으면 독립 대조 초기값에서 제외한다. 로컬 ConvNeXt pretrained도 존재하지만 모델 이름을 넓혀 동시 screen/HP 주변 탐색을 하는 안은 제안하지 않는다.

연구실 API에서 저장 토론 [745214](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745214)의 고정 revision을 읽었다. SpeedSci는 DINOv2 .942LB와 label 변경+.011을, Tucker는 단순 Qwen과 여러 모델 low.95를 자기보고했다. 서로 다른 경험은 표현과 감독 중 하나만 병목이라고 단정하지 않을 근거이며, 검증된 recipe/개선 예측치가 아니다. 저장 revision `a717d83c7e91df747c4f235e872bc68158f1fcf45f6a3a4ac61d9dd05f0804d0`, content SHA `045228d4e75ca920e11944389ee37cc44c41400390dfb27c010f20865cdb9a05`, observed_at `2026-10-03T13:05:53.449925+00:00`; 이번04:41–04:42KST localhost 읽기이며 Kaggle 최신 재수집은 아니다. 공개 LB 사후 비율 탐색의 권고로 사용하지 않는다.

## 기록·범위

산출물은 이 문서 하나다. 다운로드/외부 원격조회·GPU·모델 load·신규 구현·임상 판독·label 재감사·실험 등록·제출0, 중앙문서 변경0. 로컬 연구실은 sources 검색2회+기존 revision 읽기1회만 했다. 일부 추정 경로(exp085 연구md/raptor_model/oldsmoke 로그/캐시config 등)는 없어 기존 실제 파일 목록과 완료 summary로 교정했고 누락을 부재 증거로 확대하지 않았다. 처음 파일명 검색에 pytest 산출물도 포함되어 출력이 잘렸으므로 이후 정확 경로만 읽었다. acquisition/training 소스는 텍스트만 읽고 import하지 않았다. 별도 장시간 작업/예약 없음; root가 선택과 중앙 handoff를 담당한다.

아래 SHA는 이번 읽기 시점의 로컬 파일이며, 외부 최신 revision 확인을 대신하지 않는다.

|자료(workspace URI)|SHA256|
|---|---|
|`workspace://docs/handoffs/evidence/2026-10/2026-10-04-post084-priority/brief.md`|`c190c3686561ed9250494e07ca026acf7c5ef8bbfef6c976a377fd5cd82cbd5d`|
|`workspace://docs/handoffs/evidence/2026-10/2026-10-04-exp085-report-tool/evaluation-v1/report.json`|`26b24c8823ea8e089a43f8678fbe0f229d2b04949ddbdb4378caf4b183e0a151`|
|`workspace://artifacts/experiments/EXP084_B_G1/summary.json`|`7773a07eab4043185d752e234d132fa60bd9af28c294dfd8e4581088abcbc7d5`|
|`workspace://artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/config.json`|`fd1205526a916937bb85164fd1f27e11cd0d8d39f362505110e66a2318563c67`|
|`workspace://artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/result.json`|`42c0f723b00e02e3317a5812ed4f1a04740a0c85c2ca6ae12f71cf1fcf040d76`|
|`workspace://artifacts/experiments/EXP027_RADIODINO_FROZEN_LOCAL/result.json`|`1f03a7780ca7ec06038d99f7858cf2e3abf03268b96588fded989fe32f544f29`|
|`workspace://artifacts/experiments/EXP042_SMOKE_DINOV2S_FIXED/summary.json`|`fbebafe03c8d0b9f2d529a50c83cf6ce4954d392ca56946398a7a7b26bc5bfb4`|
|`workspace://artifacts/experiments/exp024_dinov2_features/manifest.json`|`8480aa196b9475b3074d1e47ea12805779d3335c5b4e49d1b2903a5d451ac47b`|
|`workspace://artifacts/experiments/EXP026_RADIODINO_54_FEATURES/manifest.json`|`c1ea3fae82724ab11b75dc36fca5c1ae87aa973a492453cc987ac14e634007be`|
|`workspace://storage/external/research_20260913/asset_receipt.json`|`ffae033070fd7088f5b436274957113685bb7fd97ab7dcc098ddcffe8bf72f25`|
|`workspace://storage/external/research_20260913/Snarcy__RadioDino-s16/README.md`|`2d9f7abf8ba88231130cb840eb05fbf65d8a0191ec795c1c0cd7bb96c710802a`|
|`workspace://docs/research/anchor-notebook-study-2026-10-03.md`|`578807af1808cd0c9584777fdb220cb6ecd752c3b8eab5b1cc9708a517f3c3bb`|
|`workspace://docs/research/exp081-coronal-span-2026-10-03.md`|`3da794dc699d41e0bf5764953e1cf648db49c8542ab85bf8a18bcfd788361060`|
|`workspace://docs/research/silver-other-research-and-pfoa-2026-10-03.md`|`6b2110349292f04b779892fa28b03889ce54dfb6ac112202f76c9af2aa7dd17d`|
|`workspace://docs/runbooks/strategy-review.md`|`ad08fafd5d768d57347082eee882c30caad11a87cbc1436987b0921bf6f9e8cb`|
|`workspace://docs/research/decision-memory.md`|`55f91ccad386004a2e9a4703b0d52c31b41029733022ec486f3e2667972cfaef`|
|`workspace://training/src/raptor_common.py`|`bc6db5b3e8cd13b52fbff8c039d3674cdf573f86c1959e22fe9b2fd94255d7dc`|
|`workspace://training/src/raptor_train.py`|`484abfd364163a330241a0aedb332f64287700b50d38438d2efb773777d8afd8`|
|`workspace://training/src/exp024_dinov2_features.py`|`0ca367d39de4e9b8367d13e16a995dd0e17bd737a96b472fd9e942e46984de56`|
|`workspace://training/src/exp026_radiodino_54_features.py`|`68ebab8fae4b628a5b102281239c2de13c8d447baeaec28b102e23415a1e8326`|
