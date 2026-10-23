# Frozen anchor + RN3 실행 가능성 검토

2026-10-04 KST · 담당 dicom_header_io · CPU 소스/기록 읽기 전용. **현재 판정은 HOLD: 새 구현·원격 실행을 시작할 근거가 충분하지 않다.** 원본과 정확 RN 자산은 존재하지만, RN3만의 처리 시간·메모리 실측이 없고 D01과 공유하는 anchor 실행 위험도 남아 있다. Anchor+B/RN6는 기존 성분 실측의 1,300건 시나리오부터 9시간을 넘는다.

## D01 실패와 이번 안의 관계

**Anchor 결합은 처음 시도하는 전략이 아니다.** 10-03 출발 노트북 검토에서 이미 ‘0.943 전체 스택 보존 + 작은 보완 모델’을 선택했고, D01은 그대로 anchor 90% + 기존 R2 다섯 모델의 평균확률을 rank한 값 10%를 제출했다. 제출56796964는 exact-ref05:03 관측에서 hidden 재실행 중 unhandled error·점수 없음이었다. 실패 단계/원인은 UNKNOWN이다. 원100/133/public3 engineering PASS나 COMPLETE 표시로 이 실패를 소급 PASS로 바꾸지 않는다. D01 재실행·재제출·gate 완화는 제안하지 않는다.

| D01과 공유하는 부분 | RN3로 바뀌는 부분 | 판단 |
|---|---|---|
| 원 anchor 30셀·원 weights·2T4 모델군, scratch/cache 및 source refs | 옛 R2 5개 대신 기존 ResNet34 F0/F1/F2 3개 | tail이 가벼울 가능성은 있으나 실측 이득/실패 원인 해결은 미입증 |
| 실제 root 탐색·`train.csv` 요구·test N·순서·12target·전체 cohort rank | RN은 qualified B/RN serving 입력 224/K24/within-slot-valid·strict FP16 | B/RN6의 채점 성공은 다른 경로의 성공이며 결합 경로 보장은 아님 |
| anchor 종료→자손 정리→보완 모델→검증→최종 파일 공개 | R2 `rank(mean probability)`에서 RN `mean(per-fold rank)`로 변경 | 새 고정 가설이며 D01의 동일 재제출은 아님. 성능 근거가 생긴 것은 아님 |
| near-30GiB anchor 메모리, N>48·133·Rad 전환, 공통 deadline·출력 인계 | RN3 전용 manifest/검증기 및 새 외부 receipt chain 필요 | 공통 위험을 관측할 구체안과 공동 예산 없이 교체만으로 GO 불가 |

`train.csv` 부재, N-gate, 시간, 메모리는 **정적 위험 조건**이지 D01의 확정 실패 원인이 아니다. 원인 불명만으로 영구 차단하지도 않는다. 다음 관측이 어느 위험을 줄이는지와 비용을 먼저 제시해야 한다.

## 정확한 후보와 소스

고정 후보는 다음 하나다. (A)는 원0.943 anchor가 출력한 값, (P_f)는 RN fold별 확률이다.

`Y[u,t] = 0.90 * A[u,t] + 0.10 * mean_f(rank_N(P_f[:,t])[u])`

- `rank_N`: 전체 supplied cohort의 각 target/fold에 average-tie percentile rank/N. RN 확률을 먼저 평균해서 rank하지 않는다. float64 rank/평균, 12target 공통, 최종 재rank·소견 routing·비율 탐색 없음.
- 원 anchor cell27은 이미 마지막 `rank(method='average', pct=True)`를 하고 cell29가 최종 파일을 만든다. **원 최종 A를 그대로 90% 사용**하며 새 wrapper가 내부 계수·원 가중치·원 셀을 바꾸지 않는다. 내부 모델별 비중도 그대로다.
- 비교용 anchor+B/RN6는 원 standalone의 `mean_f(.9 rank B_f + .1 rank RN_f)`를 실행하는 비용만 비교했다. 바깥 10%에 이 출력을 넣으면 RN의 유효 비중은 1%가 되어 RN10%와 다른 가설이다. 두 안을 LB에서 연속 비교하자는 제안이 아니다.

| 자산 | 정확 위치·기록된 식별자 |
|---|---|
| 원 anchor | `kaggle/packages/KNEE_ANCHOR_0943/rsna-knee-fast-2xt4-inference.ipynb`; Jiweiliu v11, 30셀/23코드셀, SHA `cd5b8b69f89bcfaa6b6ee065f7b18f7134dc71968be3827c87ca3e5672007cac`; 우리 anchor v1 제출56773226=0.943 |
| RN_F0 | `artifacts/experiments/EXP079_R34_B_F0/model_best.pt`; staged `weights/RN_F0.pt`; SHA `59304302e4d0bfb084f8e47c73cc91f7e290c2faef6b62f58124bdd05287e1d4` |
| RN_F1 | `artifacts/experiments/EXP079_R34_B_F1/model_best.pt`; staged `weights/RN_F1.pt`; SHA `a3cf3f22eeade5115270df81e47c673420ee7ce77416ade716e7620a2e07cfdb` |
| RN_F2 | `artifacts/experiments/EXP080_R34_B_F2/model_best.pt`; staged `weights/RN_F2.pt`; SHA `d198282aa26b2e06a26412942ef61ef318d789c16891801ffe363feb44ca9abe` |
| RN serving | `.local/b-rn3f-staging/20261004-v1/assets/b-rn3f-assets.json`, manifest SHA `40f5a04dd0f770f05e9af5cf9a3827ccf5748c7989847c0778ba189fde40f80e`; dataset `taeyangg4/knee-b-rn3f-v1-assets`. RN은 `resnet34.a1_in1k`, 20epoch, 기존 체크포인트다. |
| 수식/입력 코드 | 같은 assets의 `b_rn3f_serving.py` L107–136·L271+, `training/src/raptor_ensemble.py` L87/140/279. 원 wrapper는 **6개를 강제**하므로 3개로 몰래 바꾸면 안 된다. |
| 최종 B/RN release | `.local/b-rn3f-full-config-staging/20261004-v1/bundle/b_rn3f_release.py`; source SHA `4bca01aa59eac625fd8b85707b63e44f845873c0b61aa67425dad446e09356ae`; actualv2 release manifest `05238751bc969f49e2a3d794a97c0fdb48081409aa1871c4d98a4e3e631d6750` |

위 SHA는 완료된 manifest/receipt에서 재사용했다. 이번에 weights를 열거나 전체 hash 검사를 반복하지 않았다. 현재 소스 파일은 텍스트로 확인했으며, 새 실행 직전의 exact binding 검증은 별도 root 계약이 필요하다. 원 anchor 모든 upstream 자산의 resolved revision 계보가 완전히 확정된 것은 아니다.

## 실측과 계산 시나리오

| 완료된 측정 | 값 | 범위 |
|---|---:|---|
| 원 anchor100 | 2,031.043초 | 2T4, 원 스택; 50건은1,082.179초 |
| 통합133 중 anchor | process2,219.337초 / whole phase2,229.387초 | D01 engineering133의 부분 단계; hidden 실행 성공 아님 |
| anchor133 메모리 | kernel peak29.541/30GiB, 여유469.7MiB | cgroup/page cache 포함. sample RSS16.853GiB와 구별 |
| B/RN6 T4 원DICOM100 | producer877.354초, inference837.037초 | **1개 visible T4**, 실제 CPU4threads, all6+공통 DICOM 입력 포함 |
| 같은100 자원 | sample RSS3,013,132,288B; GPU allocated1,926,045,184B | session cgroup14,937,755,648B는 전체 검증 session. OOM0/정리 정상 |
| B/RN6 actualv2 공개3 | post-ack95.117초; RSS2.717GiB / session cgroup5.099GiB | 새 실행 없이 기존 receipt 재사용 |
| RN3만 | 시간·peak 미측정 | 기존 receipt에 RN/B별 시간이 없어 분리할 수 없음 |

**1,300은 계산 가정일 뿐 hidden N을 확인한 숫자가 아니다.** cold start를 빼거나 더 빠른133 관측만 선택하지 않고 기존100 전체 시간을 비례 확장했다.

| 1,300건 시나리오 | 초 / 시간 | 판단 |
|---|---:|---|
| anchor만 | 26,403.557초 /7.334h | 새로운 full 측정 아님 |
| B/RN6만 | 11,405.607초 /3.168h | RN3만의 비용으로 단정 불가 |
| 순차 anchor+B/RN6 | 37,809.164초 /10.503h | reserve 전부터9h 초과 |
| 같은 합 +300초 | 38,109.164초 /10.586h | 불충족 |
| 같은 합 ×1.3 +300초 | 49,451.914초 /13.737h | 불충족 |

RN3의 100건 환산 시간이 **438.188초 이하**여야 anchor100 기반1,300건+최소5분 reserve가 계산상 맞는다. 30분 reserve면 **322.803초 이하**가 필요하다. 이는 요구 속도이며 RN3 실측값이 아니다. 6개→3개라는 이유로877.354를 절반으로 줄이거나, 두 T4 할당을 이유로 다시 절반으로 줄이면 안 된다. 입력 decode가 공유되어 있고 anchor는 이미 두 GPU를 사용한다. 30%를 전체 합에 적용하면 anchor만도34,624.624초(+300 포함)여서1,300건 gate를 충족하지 않는다.

RN3에 all6 시간을 임시 proxy로 쓰는 보수적 산술 screen도1,300에서 FAIL이다. 이 proxy는 RN3의 입증된 상한이 아니다. 같은 합으로 최소5분 reserve만 두면 N≤1,103, `1.3*sum+300≤30,600`이면 N≤801이라는 **산술 조건**이 나온다. 이를 실제 hidden N 또는 새로운 허가 gate로 채택하지 않았다. 원 standalone6의 N-gate는 anchor 시간을 전혀 포함하지 않아 그대로 가져올 수 없다.

## 메모리·2T4·입력·정리 조건

- anchor 먼저, 자손 전체 reap와 GPU/메모리 상태 기록 뒤 RN을 별도 프로세스에서 실행하는 순차 구성이 최소 변경이다. RN source는 현재 **visible CUDA1개**를 요구한다. 2GPU sharding/anchor와 동시 실행은 아직 검증하지 않은 별도 최적화다. CPU도 요청12가 아니라 실제 할당 상한(완료 T4에서는4)을 지켜야 한다.
- 메모리는 순차 phase이므로 두 peak를 더하지 않는다. 반대로 원 anchor peak가29.541GiB였으므로 RN을 줄여도 anchor의 near-limit 위험이 사라지지 않는다. RN3 peak를 절반으로 추정하지 않는다.
- anchor N>48은 CoAt 순차 실행, N≥133은1GiB DINO RAM threshold를 넘어 memmap. 기존133에서 memmap 생성/정리를 통과했지만1,300 backing약9.84GiB와 Rad 약669/892 전환은 실측하지 않았다. scratch 여유·dirty/file pages·flush/unlink·phase 후 잔존을 관측해야 한다.
- 원 anchor는 cell6에서 `len(pd.read_csv(ROOT/'train.csv'))`를 cache 계획에 사용한다. 따라서 RN의 test-only 경로와 달리 결합 실행은 train.csv 존재와 읽기를 요구한다. 이 파일을 비우거나 proxy로 바꾸면 원 조건을 보존한 실행이 아니다. 이 정적 사실이 D01 원인이라는 증거는 없다.
- D01 suffix는 원 anchor의 절대 `submission.csv`를 감사 후 격리 경로로 이동한다. controlled failure/후속 실패 시 최종 파일 제거, 마지막 atomic publication까지 함께 검증해야 한다. 외부 SIGKILL/VM 종료에서 Python cleanup을 보장하지 않는다.
- 9h platform request·첫 cell부터의 공통32100초 internal deadline·reserve·단계별 telemetry를 새 결합에 맞게 고정해야 한다. 최악의2T4 ×9h는18GPUh 계획량이다. 예전27.63h 잔량을 현재 할당으로 사용하지 않는다.

## 최소 변경과 재사용/새 검사의 경계

원 anchor 셀/weights 및 여섯 모델 wrapper는 불변으로 둔다. 새 버전의 RN3 manifest가 정확 세 개만 `load_members`에 제공하고, 동일 input/precision 경로에서 raw `[3,N,12]`와 완료 scope를 확인해야 한다. RN rank 평균과 고정 anchor90/10 결합, 새 전체 deadline/resource/final-publication receipt만 추가한다. D01 R2 budget이나 원 six-member guard를 느슨하게 만들어 통과시키지 않는다. anchor와 RN의 dependency 설치 환경도 별도 child로 분리해 원 anchor 환경을 바꾸지 않는다.

재사용: 원 anchor source/member/assets 및100/133 완료 engineering 검사, RN3를 포함한 B/RN6 T4/4070100 입력·수치 gate, codec·원 release atomic rollback/cleanup, 기존 공개3 receipt. **원100/133/public3 재추론은 하지 않는다.** 새 wrapper는 새 계약이므로 기존 PASS를 전체 결합 PASS로 확대하지 않는다.

새롭게 필요한 범위는 (1) RN3 subset·ties·whole-cohort/UID/12target·중복rank·누락member·실패시파일/timeout의 작은 합성 CPU 검사, (2) 원source/의존성/root/train.csv/asset 계약의 제한된 차이 검토, (3) root가 투자할 가치가 있다고 판단할 경우에만 **새로 고정한 다른49건**으로 변경된 anchor→RN3 경계를 한 번 관측하는 것이다. 마지막 항목은 실행 승인이 아니다. 기존100/133/public3의 재측정도 아니다.

그 관측에서는 anchor/RN setup·decode·model load·inference·serialize·cleanup 시간을 분리하고, 단계 진입/종료와 오류를 원자적으로 저장한다. cgroup memory.stat/events·scratch·각GPU before/after·자손 잔존·원본 마지막 출력과 새 최종 출력의 구분을 보존한다. 재현 실패 시 ‘어느 단계에서’인지 알아낼 수 있어야 한다. N49는CoAt >48 경계를 거치지만 full cache/hidden 메모리 보장은 아니다.

비용을 살피기 위한 산술은 anchor50 실측1,082.179 + all6의49/100 proxy429.904 =1,512.082초, 여기에30%+300초면 **37.762분**이다. 따라서 향후45분 platform request/2T4 약1.5GPUh 예산을 root가 검토할 수는 있다. RN3의 입증된 상한이나 플랫폼 강제종료 보장은 아니며, 오늘은 구현·SDK·명령도 만들지 않았다. 관측49 PASS만으로1,300 release PASS를 선언하면 안 된다.

## 사전 판정 규칙

**GO 재검토:** 원source/weights와 수식 불변, 구체적인 stage 실패관측과 실제 RN3 비용으로 D01 공통 위험에 대한 정보가 늘고, 실제 supplied N에 대한 공동 예산/사전 reserve를 충족하며, OOM0·자손정리·원자적 최종출력까지 root가 확인한 경우다. root가 하나의 새 계약과 비용을 정한 뒤에만 다음 단계로 간다.

**HOLD:** RN3 시간 근거 부재, jointN-gate 실패, source/train.csv/asset 불확실, 메모리/scratch 여유 미확인, 공통경로 위험에 대한 차별적 관측계획 부재, fallback/누락모델/불완전 receipt. 원인 불명을 RN 교체로 해결했다고 하거나 실패 gate를 완화해야만 가능한 경우도 보류한다. 현재는 앞의 비용·공통위험 조건 때문에 HOLD다.

정확한 anchor와 후보가 모두 보지 않은 common unused V가 없으므로 독립 개선은 입증되지 않았다. **0.913은 여섯 모델 standalone 후보의 성적이지 RN standalone 점수나 RN 보완의 인과효과가 아니다.** 예측 상관을 읽거나 성능 근거로 사용하지 않았다. 결국 한 번의 고정 탐색 제출까지 간다면 하락/표시동점은0.943 유지, 상승은 잠정 public 후보로만 보존한다. 상승도 private/임상 개선 증명이 아니며 원0.943를 계속 보존하고 비율·소견·B/RN 조합의 후속 LB 탐색은 하지 않는다.

상세 경로·기록된 SHA·실측/시나리오 수치는 같은 폴더 `anchor-rn-feasibility.json`에 있다. 이번 작업: 구현0·모델/weights/라벨/OOF/배열/pixel0·hash sweep0·원격0·추가agent0·중앙문서수정0.
