# EXP085 상대위치 대조 · CPU 준비

## 질문과 기준

같은 슬롯끼리 섞는 작은 attention에 **슬롯 내부 상대좌표**를 추가하면 없는 경우보다 전체12소견 예측이 나아지는지를 묻는다. 좌표는 `(center-slot_start)/(slot_size-1)`이며 실제 mm·해부학 방향이나 원본 series의 물리적 위치가 아니다. 절대 위치를 직접 주지 않으므로 공통 slot grouping 위에 상대거리·방향 정보를 추가하는 시험이다.

M02/M03: EXP084의 T3,455/V888·알려진 component 그룹을 그대로 쓰고 T-only B8을 두 arm 공통 초기값으로 쓴다. 알려지지 않은 환자연결·weak label·이미 본 개발V의 노출은 남는다. 새 임상 holdout/정확D01 검증이라고 부르지 않는다.

M04/M05: primary는 treatment−control macro12이고 control−원B와 treatment−원B도 반드시 함께 보고한다. EXP083처럼 두 refit 모두 원B보다 낮은 사실을 상대차이로 숨기지 않는다.12소견 P/N·골절·그룹·동일 component bootstrap 조건부CI와 비용을 함께 판정한다. 이번 실패로 모든 sequence/3D/pretraining 전략을 기각하지 않는다.

M10: 공개LB·Gold·비율·소견 routing·epoch 탐색이 없다. 본 시험은 두 arm만이며 serving parity/제출 판단과 구분한다.

## 정확한 구조

원 B의 `backbone → LayerNorm → 12소견 attention pooling → clsW/clsb`를 모두 보존한다. raw feature F=1024에 한 attention residual을 더한 뒤 원 B.head를 그대로 호출한다.

- `qkv: Linear(1024,192,bias=False)`를 q/k/v 각64로 나눈다. 새 attention dropout은 없다.
- `bias(delta)=Linear(16,1,bias=False)(tanh(Linear(2,16)([delta,abs(delta)])))`, `delta=p_key-p_query`.
- `attention=softmax(q*kᵀ/sqrt(64)+bias)`는 유효한 같은 슬롯의 key만 허용한다. logits/softmax/value 합산은 FP32로 계산한다.
- context를 `Linear(64,1024,bias=False)`로 투영하고 원 feature dtype으로 복원해 더한다.
- control은 p를 모두0으로 바꾸며 treatment는 실제 좌표를 쓴다. 구조·262,208개 추가 파라미터·학습 가능성·mask는 같다.
- 출력 projection과 relative MLP 마지막 weight만0이다. 별도의0 scalar gate는 없다. 시작 시 원B/두arm logits가 같고 projection이 먼저 학습한 뒤 위치 경로 gradient가 생긴다.
- invalid query에는 자기 자신 한 칸을 열어 softmax를 유한하게 만들고 residual을0으로 만든다. 원래 blank영상의 encoding/pooling은 제거하지 않는다.

공동 token/slot/position 순열에는 equivariant하고 최종 pooling은 invariant하다. 픽셀만 고정하고 좌표만 재배정하면 treatment는 달라질 수 있고 control은 달라지지 않아야 한다. 서로 다른 슬롯의 특징을 바꿔도 다른 슬롯 residual에는 전달되지 않는다.

## 학습과 입출력

원 `StudyWindows`를 한 번 호출한 결과 픽셀을 그대로 사용한다. 같은 UID·epoch·seed·stream으로 center만 재현하고 읽는 동안 공유epoch가 바뀌면 실패한다. 64-slot `[18,14,12,8,12]`, within-slot-valid·trainK12/evalK24·224px·ImageNet 정규화·intensity jitter를 유지한다. 중복 center와 선택 순서를 유지한다.

EXP084 checkpoint는 weights/provenance만 저장한다. 따라서 두arm은 **B8 초기값에서 옵티마이저를 재시작한8epoch 추가적응**이며 uninterrupted B16이 아니다. 각arm은 새 AdamW/OneCycleLR/GradScaler, 같은 seed42, 새 attention의 같은 초기값, 전체 backbone 학습을 사용한다. 원 bb_lr3e-5/head_lr1e-3, bs8, clipped BCE, strict fp16을 유지한다. 학습 sampling epoch는8–15로 고정한다. optimizer 그룹은 이름 접두어가 아니라 backbone parameter identity로 구분한다.

각arm3448 updates를 요구하고 첫 overflow/skip·시간초과·비유한값·대조 RNG 불일치에 실패한다. 모든 minibatch UID/center/좌표·dropout RNG 전후를 대조하고 매epoch 첫batch 픽셀hash도 대조한다. 전체 캐시 shard의 재해시는 하지 않으며 입력의 불변성 계약은 기존 저장소 규칙을 따른다.

본학습은 고정 마지막8epoch weights와 V888의 예측만 저장한다. 중간 V평가·SWA·best-epoch 선택은 없다. runner는 T행의 target만 float로 변환한다. CSV의 V행 텍스트를 UID 식별용으로 지나가지만 Vtarget을 수치화하거나 점수화하지 않는다. 최종 평가는 별도 담당자가 두arm 성공 후 한 번 수행한다. RNclip 고정90/10 rank blend는 사전 고정 secondary이고 primary를 대체하지 않는다.

## 단계와 비용

1. root가 새 phase decision을 기록한 뒤 paired pilot을1회 실행한다. 각arm32updates·T16 K24 inference만 실행하고 weights/prediction은 저장하지 않는다. 검증 지표로 선택하지 않는다.
2. 첫8 train update와 첫eval batch의 시작비용은 한 번만 반영한다. 나머지 정상 처리 속도로3448 updates/V888을 추정하고 전체에1.25를 곱한다. 이는 실측 full runtime 보장이 아니다.
3. 각fit 보수추정≤3600초, pilot실제시간+두fit 보수추정≤7200초일 때 root가 full을 배정한다. pilot600초 cap, 두full fit 각각3600초 cap, 전체 active supervised 실행7200초 cap, retry0. 사람의 검토 대기시간은 GPU작업 예산에서 제외한다고 명시했다.

과거 EXP084 B 실측2610초(43.5분)로 두fit은87분 수준이 출발 추정이다. 새attention·검사·I/O overhead는 GPU pilot 전 미검증이다. 준비된 [명령](execution-plan.json)은 실제로 실행하지 않았다. root는 기존 GPU작업 종료·새경로·heartbeat를 확인한다.

## 현재 증거와 한계

[합성47검사](verification.json)는 초기logits 일치, 두arm 파라미터/초기값, optimizer 분리, gradient 경로, 마스킹·빈입력·순열·픽셀보존·시간값 거부를 확인했다. 실제 backbone/GPU/학습은 실행하지 않았다. 원B checkpoint는 hash만 읽었고 model state를 CPU에 로드하지 않았다. 원source와 EXP084 산출물은 수정하지 않았다.

phase receipt PASS, 외부resource receipt COMPLETE/exit0/정리오류0, gpu_job SUCCESS를 함께 확인해야 한다. 최종평가도 원B와 같은 UID/target/label/split/정규화 계약을 검사한다. 이 CPU 결과는 T4/4070 parity·대회점수·제출준비 완료를 뜻하지 않는다.
