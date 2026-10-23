# EXP086 pilot 실패: 결정론 유지 교정안

관측 2026-10-04 05:40 KST. 담당 dicom_header_io. 범위: 설치 source·실패 로그 읽기와 산술만. 구현/모델 load/GPU/학습/다운로드 0, 원 v1/source/protocol/FAIL·중앙문서 변경 0. 원 실패를 재실행하지 않았다.

**권고: 원 가설 보존을 우선하면 ①을 먼저 CPU 준비한다.** ②는 timm 공식 API로 구현이 짧고 실행이 단순하지만, 224 위치 임베딩의 독립 학습이라는 새로운 parameterization을 명시해야 한다. 두 안을 동시에 학습하는 제안이 아니며, 어느 안도 아직 구현·실측·GPU 허가되지 않았다.

## 확인한 원인과 범위

- 원 프로토콜 SHA `0de375945192a050e7ca22944cc1990e3a67d0c125d333e83bf4a796719e5401`.
- `workspace://artifacts/experiments/EXP086_DINOV2S_G1_PILOT_RESOURCE/receipt.json`: FAILED, child/wrapper exit1, 83.235초, timeout180, retry0, cleanup actions/errors/survivors 모두 없음. stderr는 `scaler.scale(loss).backward()`에서 `upsample_bicubic2d_aa_backward_out_cuda`에 deterministic 구현이 없다고 보고했다. 실패 전 성공 optimizer update를 증명하는 receipt는 없고 최종 산출물도 없다.
- 설치 timm1.0.19 `VisionTransformer._pos_embed`(vision_transformer.py:743–778)는 dynamic 입력의 37×37+CLS 위치표를 매 forward 16×16+CLS로 resample한다. `resample_abs_pos_embed`(pos_embed.py:19–60)는 FP32, bicubic, antialias=True이며 prefix는 보간에서 제외한다.
- 설치 torch2.5.1 functional.py:4588–4595는 bicubic+AA를 CUDA/CPU별 native op로 dispatch한다. torch/__init__.py:1258–1264는 CUDA interpolate backward의 결정론 제한을 명시한다. CPU와 CUDA backward entrypoint는 설치 native header에 각각 존재한다. CPU path의 실제 결정론·copy-gradient 동작은 새 검증 대상이며 source만으로 PASS라 하지 않는다.
- 원 CPU probe는 CUDA backward를 검사하지 않았으므로 CPU 통과와 모순되지 않는다. 이번 83.235초는 startup·Windows worker spawn·부분 forward/backward·정리의 합이다. steady update 시간이나 전체8epoch ETA로 환산하면 안 된다.

## ① canonical 518 trainable table 보존 + 미분가능 CPU FP32 보간

**변경:** strict=True로 원 asset을 먼저 읽는다. CUDA에 있는 `pos_embed`를 autograd가 연결된 `.to(cpu, float32)`로 옮겨 **같은 timm resample 함수**를 호출하고, 결과를 원 장치/원 위치표 dtype으로 돌려보낸다. `.detach()`, `no_grad()`, NumPy 변환, 고정 캐시를 쓰면 위치표의 학습이 끊기므로 금지한다. activation dtype(FP16)으로 강제 변환하면 기존 FP32 위치 더하기와 달라지므로 피한다. prefix/old_size37×37/new_size16×16/AA/align_corners 기본값도 보존한다.

**보존:** 원 state tensor shape `[1,1370,384]`, 초기state SHA, 526,080개 위치 파라미터, backbone/head optimizer partition, AdamW state 크기, canonical 위치표에 대한 gradient 경로와 full fine-tune 의미를 유지할 수 있다. 수학적으로 출력은 `I(P)`, 원 위치표 gradient는 `Iᵀ(dL/dI(P))`이다.

**차이:** CPU/CUDA 보간의 부동소수점 누산이 다를 수 있다. 원 CPU reference와 함수/값 연결을 확인하되 CUDA bitwise 동등성을 주장하지 않는다. 초기 파라미터 SHA가 같아도 실행 연산 정책은 달라졌으므로 새 source/protocol/receipt가 필요하다. CPU 내부1thread 고정, CPU autocast 비활성·FP32를 명시하고 runtime strict/deterministic=True/TF32=False는 그대로 둔다.

**비용:** 보수적으로 prefix까지 이동하면 forward 원표2,104,320B→CPU, 결과394,752B→GPU이고 backward가 역방향을 더해 약4,998,144B(4.77MiB)/update이다. 3448update는 약17.23GB 누적전송이며 복사 동기화와 CPU 보간/역전파 비용은 미측정이다. `raptor_common.encode`는 B×K를 한 번에 backbone 호출하고 위치표를 broadcast하므로 이 비용을96windows 배로 곱하지 않는다. timm `forward_features`의 pos embed는 checkpointed transformer blocks 밖이라 blocks 재계산 횟수만큼 보간하지도 않는다. 작은 전송량만으로 시간 gate 통과를 보장할 수 없다.

**구현/배포 주의:** 전역 timm 함수를 monkeypatch하지 않고 후보 모델에 한정한 adapter로 train/eval에 같은 정책을 사용한다. 원 shape라 state_dict는 기존 factory와 맞지만, 기본 GPU 보간으로 추론하면 CPU 보간 정책과 달라진다. checkpoint metadata·추론 factory·향후 T4 bundle에 positional policy를 명시해야 한다. 이번 수정만으로 T4 parity/serving 준비 완료가 아니다.

## ② strict canonical load 후 CPU에서224 위치표로1회변환, 변환표 학습

**설치된 공식 최소 경로:** CPU에서 원518 model 생성→exact asset strict load→원 초기state 검증→`backbone.set_input_size(img_size=(224,224))`→변환된 초기state 별도 hash. 해당 timm API(vision_transformer.py:713–741)는 patch grid를 바꾸고 `nn.Parameter(resample_abs_pos_embed(...))`로 위치표를 교체한다. patch_embed.py:71–95는 patch_size 인자를 바꾸지 않으면 projection 교체분기를 건너뛰고 grid만 갱신한다. 이후 dynamic forward는 tokens 수가 같아 pos_embed.py:31–32의 early return을 타며 보간 backward 자체가 없어지는 것이 source상 기대 경로다.

**보존/변경:** strict 원자산 검증은 유지한다. 원표를 무손실로 보존한 별도 asset과 변환내역을 남기되 실제 trainable표는 `[1,257,384]`, 98,688개이다. 원표보다427,392개 적다. 공유head/입력/라벨/224/window/optimizer schedule은 유지 가능하지만 초기state SHA·parameter count·AdamW state·학습 parameterization이 바뀐다. `Q=I(P0)`가 첫 forward에서 같아도 이후 `Q`를 직접 AdamW로 갱신하는 것은 `P`를 갱신한 뒤 `I(P)`를 얻는 것과 동등하지 않다. **단순 runtime 교정으로 숨기지 않는다.**

**비용/실용성:** 한번의CPU 변환만 필요하며 반복 CPU 왕복이 없으므로 ①보다 runtime가 단순할 것으로 예상한다. 실제초기화/전처리/훈련 속도는 미측정이다. SGD/AdamW 궤적차이가 점수에 미치는 방향도 예측할 수 없다. 함수가 새 Parameter를 만들므로 optimizer 구성은 반드시 변환 뒤에 한다.

**checkpoint 위험:** 기존 `raptor_common.load_checkpoint`는 arch 기본518 shape를 만든 뒤 strict state load한다. 변환224 checkpoint는 이 경로에서 size mismatch가 난다. factory가 저장된 table/grid 계약을 먼저 적용한 뒤 strict load하도록 별도 adapter·metadata를 준비해야 하며, 기존공용 load 함수를 조용히 바꾸면 안 된다. 기본 `arch` 문자열만으로 새shape를 표현할 수 없다.

## 더 작은 대안과 비추천 우회

- **pos_embed만 freeze:** `.requires_grad_(False)`로 원GPU 보간의 위치표 backward를 제거하는 것은 소스 변경이 가장 작다. 초기값/shape는 보존하고 대부분encoder는 학습하지만526,080개 위치파라미터는 학습하지 않는다. 원 full-encoder 가설을 바꾸고 동결표 가설이 되므로 자동교정으로 권하지 않는다. 채택시 새trainable mask·protocol·pilot 필요.
- **고정 보간행렬을 통한 결정론 matmul:** 선형변환I와Iᵀ를 미리 구성하면 canonical표를 유지하고 CPU왕복을 줄일 가능성이 있다. 그러나 AA kernel/boundary/연산순서·gradient 동등성 검증과별도행렬자산이 필요해 이번두안보다 작은 교정이 아니다.
- **antialias=False**는 kernel을 바꾸고 CUDA bicubic backward 결정론 지원도 보장하지 않는다. nearest/bilinear로변경도 다른보간계약이다. `warn_only`, deterministic=False, detach/no_grad로오류은폐, 이전실패를성공으로변경, 같은v1재시도는 모두 제외한다.

## 새 검증의 최소 범위와 예산 제안

1. root가 한 안을 선택한 뒤 **CPU 준비만 별도 결정**. 기존 model probe 전체를 반복하기보다 바뀐 위치연산에 대한 작은 synthetic forward/backward·gradient finite/nonzero·strict 설정/정책·shape/state/parameter binding을 검사한다. ①은 CPU copy 이후 gradient가 원canonical parameter에 도달하는지, ②는공식변환값·새shape·optimizer 이후구성이 맞는지 확인한다. 구현후1회의 좁은합성 모델연결 검사는 필요할 수 있으나 현재 미승인/미실행이다.
2. 원v1/FAIL/프로토콜·evaluation registration을 그대로보존하고 v2 source snapshot·protocol/output/log/resource/registration을 새위치로 만든다. ①도 source실행정책변경이므로 원v1 SHA를 재사용하지 않는다. 원 B를 다시학습할 이유는 없다.
3. **root의 새 pilot 승인 이후에만** strictTrue/FP16/32T-updates·T4study inference·full3448 schedule/worker계약을 그대로두고 새GPU pilot1회. 원retries0승인은 소진됐으므로 원pilot재시도로취급하지 않는다. startup여유 포함 기존180초cap 유지가능성은 선택시명시하며 실패83.235초로완료가능성을단정하지 않는다.
4. 8warmup제외24update의 실제dataloader+forward+backward+copy시간과전체resource receipt를 확인. full 비용식 `1.3*(steady*3448)+300<=5400`·합5580을 통과하고 root가 별도full결정해야 한다. full8epoch5400초는검증전견적상한이며예측실행시간이아니다.
5. CPU준비 예상15–30분(변경안·checkpoint계약에따라달라지는 작업견적), GPU pilot최대3분+확인, full승인시최대90분. ①속도가 gate를넘으면자동으로②를실행하지않고 비용/계약변경을기록해재결정한다. 단순작동교정으로효과가입증된것이아니며 원weakV 한계/노출·Baker/Fracture/소견별검사는그대로다.

## 파일 근거와 SHA256

설치경로 접두사: `C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/Lib/site-packages/`. source의 줄번호는 위 관측버전 기준이다. C++ implementation 전체가아닌설치 header선언도사용했으므로 CPU성공/성능은실측예정으로분리했다.

| 파일 | SHA256 |
|---|---|
| timm/layers/pos_embed.py | `8e5beda55289ff7a4a91fadee6ea4134939b860642723f8fb5dc82039fcc4d37` |
| timm/layers/patch_embed.py | `78f2d6afba3bf71a0b8a51cfb06e034b08af0708f4a49509511e4b419e448cda` |
| timm/models/vision_transformer.py | `d362a7ad3f5676833e04d28d5e02220e57122bdffff30fbcb1d71180b4de4407` |
| torch/nn/functional.py | `afb8598085c2c4bc55fdf9e00d7cf6c13df08c7c2cabfbbd34b0b60b42e5ab9d` |
| torch/__init__.py | `b6639a0441d7c404443ba90901caaff93f07790d9357e65abe715cad1e897b2d` |
| torch/include/ATen/ops/_upsample_bicubic2d_aa_backward_native.h | `2461402bccb605d0a65ed1a35b4846c27188ab6062ca06271a63a27320cdc7b3` |
| torch/include/torch/csrc/autograd/generated/Functions.h (ToCopyBackward0) | `c3e4f4139ca569256cd88700954450aae7f3d139cb23a3d6878d13e5b1841358` |
| EXP086_DINOV2S_G1_PILOT_RESOURCE/stderr.log | `624895b44897926a41e0adcff15dde7e8574e956fe4e888ada8b9a1dc7a1c3a3` |
| EXP086_DINOV2S_G1_PILOT_RESOURCE/receipt.json | `d2050828ad437a53d1c1a20650ab1bd4bb6c7c3adf28c0fa3353f61ec1a7828f` |
