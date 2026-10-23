# B+RN 기존 개발 후보의 제출 경로 CPU 감사

2026-10-04 KST. 상태는 **자산 식별 완료 / serving 미검증 / 제출 후보 미완성**이다. 원본·기존 실험·D01을 수정하지 않았고 GPU·원격·다운로드·학습·제출을 실행하지 않았다. 정확한 파일별 SHA256·크기·메타데이터·코드 위치는 [artifact-audit.json](artifact-audit.json)에 있다.

가중치가 없어서 다시 학습할 필요는 없다. F0/F1/F2의 CoAtNet B 3개와 ResNet34 B 3개가 존재하며, 6개 모두 같은 입력 계약이다. 그러나 현재 공용 serving CLI의 출력은 6개 확률의 동일가중평균이다. 이를 그대로 제출하면 개발에서 평가한 B+RN 90/10과 다른 가설이다. **가장 작은 다음 조치는 새 학습이 아니라, 한 가지 제출 집계법을 사전 고정한 별도 B+RN 패키지와 B 입력의 100건 수치·실행 검증을 준비하는 것이다.**

## 정확한 가중치와 학습 계약

모든 경로는 `workspace://artifacts/experiments/<tag>/model_best.pt`. 파일명은 best지만 이번 실험에서 선택은 **fixed-final**이며 B는 8번째, RN은 20번째 epoch다. `summary`·`provenance`와 checkpoint 내부의 arch/res/contract/window/epoch/tag/provenance가 일치함을 확인했다.

|tag|가족·fold|checkpoint SHA256|
|---|---|---|
|EXP078_INPUT_B_F0|CoAtNet B F0|74a70e8f4a684a53ad3b3489144eac16e5c5c500a602440cf8f8756ac03c9b17|
|EXP079_R34_B_F0|ResNet34 F0|59304302e4d0bfb084f8e47c73cc91f7e290c2faef6b62f58124bdd05287e1d4|
|EXP078_INPUT_B_F1|CoAtNet B F1|b2ebdc06490174048efaa5b559aa99258e013cfe075e60d62c89f326c72b58f9|
|EXP079_R34_B_F1|ResNet34 F1|a3cf3f22eeade5115270df81e47c673420ee7ce77416ade716e7620a2e07cfdb|
|EXP080_INPUT_B_F2|CoAtNet B F2|5e3399bb3f44bb43f81b58c4048cabad1c4657f78a4952d30b15a957f022992f|
|EXP080_R34_B_F2|ResNet34 F2|d198282aa26b2e06a26412942ef61ef318d789c16891801ffe363feb44ca9abe|

6파일 총 **1,136,009,670 bytes**(약1.058 GiB), state tensor 합1,135,315,032 bytes다. 이는 저장값의 크기이며 peak VRAM/RAM 수치가 아니다. `torch.load(weights_only=True,mmap=True,map_location='meta')`로 메타데이터/shape만 확인했다. 실제 모델 생성·strict state loading·finite weights 검사·추론은 하지 않았다. checkpoint 전체 객체가 역직렬화되지만 Gold/OOF metric 필드는 검사·선택·출력하지 않았다. CUDA는 초기화되지 않았다.

공통 입력: `raptor64_336_c140_s02`; 5slot 깊이 `[18,14,12,8,12]`, 원 volume64×336×336 uint8, 140mm crop, span0.02–0.98, **within-slot-valid K24 triplet**, 224×224 bilinear/align_corners=False, ImageNet mean/std, 12target 고정순서. 마스크는 정규화 후 nonzero pixel 합이며 실제 촬영 유무나 임상 정상의 판정값이 아니다. 유효triplet이 없으면 center=-1/정규화 전 zero window가 유지된다.

B arch=`coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`, RN=`resnet34.a1_in1k`. ImageNet 초기화가족별 SHA는 B `e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e`, RN `093be508bda4f14b886c7ca28bf504bfdd2bffc95043252a60be10755b927224`. 두 가족 모두 trainK12/evalK24, seed42, batch8, AdamW wd.02/headLR.001, strictFP16. B bbLR3e-5/8epoch/grad checkpoint/workers6, RN bbLR5e-4/20epoch/no grad checkpoint/workers4. 기존 worker 계약은 변경하지 않았다. 네 학습 소스의 실제 현재 SHA가 모든 provenance와 일치한다.

## 현재 코드가 하는 일과 빠진 일

- `raptor_ensemble.py:87`는 각 checkpoint를 개별 arch로 생성하고 공통 input contract를 검사하므로 서로 다른 두 backbone 자체는 설계상 수용한다. `:140`의 `predict_ensemble`은 study당 입력을 한 번 만들고 member별 raw `[M,N,12]`와 float32 동일평균을 반환한다. 6모델 실제 로드는 아직 확인하지 않았다.
- `raptor_ensemble_serve.py:14`는 모델 manifest→DICOM→raw CSV/memberNPZ/receipt까지 연결한다. final weighting/rank/submission/schema gate는 없다. 모든 모델을 한 manifest에 넣었을 때 나온 평균 CSV를 90/10이라고 부를 수 없다.
- `exp079_complement_report.py:137`의 평가식은 **각 fold·각 target 전체 행의 average-tie percentile rank**에 `0.9×rank(B_f)+0.1×rank(RN_f)`를 적용한다. `exp080_f2_report.py`도 같은 함수를 사용한다. 최종 재rank·group별 재rank·bootstrap 재rank가 없다.
- **3fold를 test에서 어떻게 합칠지는 기존 protocol에 고정돼 있지 않다.** `rank(mean3 probabilities)`와 `mean3(rank probabilities)`는 일반적으로 다르다. 여섯 모델의 raw를 모은 뒤 root가 한 번 고정해야 한다. 제안은 `mean_f[.9·rank(B_f)+.1·rank(RN_f)]`로, 기존 fold내 식을 유지한다. 이것도 3fold 공동 test 예측이라는 새 조합이며 성능이 이미 검증됐다는 뜻은 아니다. `rank(mean3(B))`·`rank(mean3(RN))`라는 대안과 사후 비교·비율탐색을 자동 실행하지 않는다.
- shard를 쓰면 **모든 test UID의 raw를 합친 다음** target별 rank한다. shard별 rank를 합치면 표본분포에 따라 다른 출력이 된다. ties·dtype·target 순서·UID completeness·shape·finite/range 및 raw member SHA를 receipt에 고정해야 한다.
- B+RN용 private asset manifest, offline notebook, candidate-specific CPU/GPU references, actual T4 gate, output receipt는 확인되지 않았다. `kaggle/packages`·`tools`·`training`에서 정확 tag 검색도 평가/연구 helper만 찾았다. 이것은 조회 범위 안의 결론이며 모든 외부저장소 부재 주장은 아니다.

## D01와 다른 점

|항목|기존 B+RN 후보|D01 제출 경로|
|---|---|---|
|모델|신규 ImageNet 시작 B8+RN20, F0–2 총6|공개 anchor49+CoAt branch와 기존 EXP072 R2MIX RN F0–4|
|입력|within-slot-valid,224,K24|R2 legacy,224,K24 및 anchor 여러 별도 입력|
|혼합|fold별 B/RN rank90/10 검증, test fold집계 미고정|.9 rank(anchor)+.1 rank(mean5 raw R2), 이미 고정|
|검증|노출 weak 개발 fold의 작은 효과|실행/수치/표본runtime gate; 성능향상 근거와 별개|
|패키지|없음|asset/helper/codec/anchor/source hash 강결합|

`d01_package.py`·`d01_release_v3.py`는 `ASSETS_SHA=e62893…036b`, R2 member_count5, legacy centers, anchor 및 기존 reference에 고정돼 있다. 새 여섯 weights로 바꿔 같은 D01 gate를 PASS 처리하면 안 된다. decode·raw member inference·offline wheel 설치·프로세스 감시의 **패턴과 검증된 원파일**은 재사용할 수 있지만, 새 manifest/receipt 이름과 근거가 필요하다.

D01 asset에 묶인 `raptor_common.py`/`raptor_runtime.py`/`raptor_ensemble.py`는 현재 소스와 byte동일하다. DICOM adapter는 packaged `e478626b…4867e`, 현재 `497a41f8…5634b`로 다르다. 현재 `DirectoryDicomSource(header_only=False)`가 기본값이라 자동으로 header 최적화를 쓰지는 않지만, 이번 최소 경로에서는 이미 측정한 **packaged legacy reader**를 고정하는 편이 변수를 줄인다. 이것은 B window center 검증을 대신하지 않는다.

## 최소 T4 경로와 게이트

1. **CPU 새 패키지 설계/구현**: 위6member 순서·SHA, 입력·3fold 집계식 하나·rank 범위·실행환경·오류중단을 사전등록. current 공통/런타임/ensemble와 pinned legacy DICOM adapter를 명시 복사하고 새 bundle에 hash. D01 원본은 보존. offline timm1.0.19/pydicom3.0.2/python-gdcm3.2.6 wheels는 로컬 실파일을 확인했고 OpenCV4.12.0.88은 고정 dataset+SHA가 있으나 이 감사에서는 새 환경 설치를 하지 않았다. 새 test에는 Report 입력이 없다.
2. **새 B input 기준**: 기존 고정 engineering100 UID를 재사용하되 B의 within-slot-valid centers를 새로 계산해 volume/mask/centers/norm reference를 생성한다. boundary·유효창부족·allzero도 합성으로 포함한다. D01 numerical reference는 legacy centers 및 c>=1만 허용하므로 B의 -1 empty case까지 그대로 통과시킬 수 없다. 기존 v1 exact hashFAIL은 보존하고 새 버전 계약으로 허용오차를 근거화한다. 이미 근거가 있는336→224 resize 한계(5e-7 windows/3e-6 normalized,rtol0)는 새 입력에도 적용 여부를 검증해야 한다.
3. **실제 exact6 로컬↔T4 확률 검사**: 소규모 고정 train-only engineering subset과 public3에서 member별 확률·유한성·row order·strictFP16·zero fallback·offline dependency를 비교한다. T4 compute7.5/로컬8.9·torch환경 차이로 bit-identical 확률을 가정하지 않는다. 허용오차는 실행 전 새 gate에 등록; R2F0의 과거 실제 max차이는 새 B/RN의 PASS 증거가 아니다. 로컬GPU baseline은 별도 허가된 실행에서만 만든다.
4. **동일 notebook에서 exact6 대표100 처리량·peak**: 최소 경로는 single-device resident/batch1로 코드 분기를 줄이고, 실제 CUDA allocation/RSS/VRAM/decoder/전체cold시간을 기록한다. 실패 시 stop, 임의FP32 fallback/모델제외 없음. two-T4 shard 최적화는 이 비용이 필요하다고 나온 뒤 별도 버전으로만 검토한다. 작은 표본 성공을 hidden 전체 보장으로 쓰지 않는다.
5. **동일 코드로 public3 final 출력 검사**: UID중복·누락·순서,12target순서,finite/range,전체 cohort rank/ties,모든member 완료,최종CSV hash 확인. hidden runtime은 실제 N과 study복잡도·cold/RSS 근거로 갱신한다. 모든 gate와 개발가치 검토 후에만 root가 구체 후보를 사용자에게 제출 승인 요청한다.

## 비용과 ETA

이번 실제 자산검사는 6checkpoint 총1.136GB를 순차 hash하고 meta로 읽는 데 **2.390초**였으며 전체 CPU 감사는 약17분이다. 신규 병렬 CPU는 사용자 선호12worker로 시작하되 실할당/처리량/RAM/I/O/중첩thread에 따라10→8→더 낮게 조절한다. 이처럼 작은 순차read에는 pool을 만들지 않았다. Kaggle가 과거처럼 effectiveCPU4라면12worker를 강제하지 않는다.

|남은 일|계획 예산|실측 여부/종료 조건|
|---|---|---|
|새6member manifest·fold집계·offline wrapper·출력 gate·CPU 합성검사|CPU45–75분|아직 미구현. 성능수정 없이 serving 계약만 완성|
|고정100 B input reference와 소규모 exact6 로컬GPU reference|CPU5–15분 + 로컬GPU10–20분 상한 제안|새실측 없음. 필요 자산 확인 후 timebox 등록, 지연이면 원인/부분완료 보고|
|최소 T4 한 세션: 작은 gate→대표100→public3|예상10–25분, 세션 hard cap30분 제안|B+RN 실측이 아니므로 ETA 신뢰 낮음. 할당1T4이면≤0.5 GPUh,2T4 과금이면≤1 GPUh 계획; 실제quota확인 필요. 통과 보장/자동 재시도 없음|
|hidden전체 후보 실행|현재 산정 불가|T4100 cold+per-study·peak 결과와 실제N으로 산정. 가상0.94x 점수예측 없음|

비교 근거로 D01 R2-only5개 T4 100건은 cold603.857초/추론541.776초, peak allocated2.269GB/reserved2.494GB였다. two-T4100 cold348.839초. 그러나 B의3CoAtNet은 R2의5ResNet과 계산량이 달라 이 시간을 그대로 대입할 수 없다. D01 anchor+R2의133건 production2399.726초, 보수1300 projection30938.468초(8.594시간), 30분reserve 미달은 **anchor 포함 별도 경로**다. B+RN-only는 anchor 비용을 없애지만, 예상 빠름이 같은 정확도·0.943 대체 가치를 보장하지 않는다.

## 개발 이득과 제출 가치를 구분한 판단

EXP079 F0/F1의 고정 RN 보완 평균 +0.002118, EXP080 F2 +0.001243/조건부CI[+0.000339,+0.002101]은 작은 개발 후보 근거로 보존한다. F2 B−A 골절−0.022337, 혼합−A−0.020464와 F1 반복손실, F2 LateralOA−0.014092를 같이 보존한다. 이득의 CI는 고정예측/노출weak label에 조건부이며 seed·교사오류·환자독립·private분포·연구자선택을 포함하지 않는다.

**공개0.943에 +0.001243을 더해0.944243이라고 예측할 수 없다.** 개발 B는 공개 anchor가 아니고, RN도 D01 R2가 아니며, 학습 라벨·checkpoint선택·입력·앙상블규모·참조정답이 다르다. 정확 public anchor의 독립 개선 근거도 없다. B+RN-only를 anchor 대신 내는 것과 anchor에 B+RN을 추가하는 것은 별도의 두 가설이다. 후자는 추가 혼합비와 runtime이 미등록이므로 이번 최소 경로에 끼워 넣지 않는다.

Biohub 누락 점검: 작은 효과를 버리지 않되 실행완료를 성능으로 바꾸지 않았고, 손실소견·label/ancestry 한계를 보존했다. 추가학습보다 기존자산의 단일 검증 가능한 경로를 먼저 제안한다. root의 다음 구체 행동은 **새 패키지45–75분 및 최대30분 T4 session 투자 여부와 3fold 집계식 하나를 고정**하는 것이다. 이 감사만으로 후보 채택·serving PASS·제출 승인은 발생하지 않는다.
