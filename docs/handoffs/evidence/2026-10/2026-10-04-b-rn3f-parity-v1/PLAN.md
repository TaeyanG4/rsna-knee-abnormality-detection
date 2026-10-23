# B 전용 입력·수치 gate V1

이 작업은 기존 `B_RN3F_SERVING_V1` producer와 stage를 수정하지 않는다. 새 `tools/b_rn3f_parity.py`가 사전 계약을 저장하고, 앞으로 각 host에서 입력을 CPU로 재구성하며, local/T4 산출물을 비교한다. 지금은 합성 입력만 실행했고 실제100 cache/DICOM·GPU·원격 작업은 하지 않았다.

## 입력의 연결

각 host에서 같은 pinned common/runtime/DICOM reader로 동일100 UID의 volume/mask/K24 windows를 재구성한다. 독립 선택기는 `[18,14,12,8,12]` 슬롯 안에서 binary mask가 연속3개인 중심만 찾는다. K24의 인덱스는 정수 산술로 독립 구현하며 유효 중심이 없으면24개 `-1`이다. `-1`은 normalization 전0입력이고 ImageNet normalization 후에는 음의 채널 상수가 된다.

독립 float64 reference는336→224의 `(j+.5)*1.5-.5` 좌표와 정확한1/4·3/4 계수로 bilinear를 직접 계산한다. 학습·추론이 사용한 float32 mean/std를 float64로 승격해 정규화한다. torch interpolate/OpenCV를 reference 계산에 호출하지 않는다. 각host의 실제 float32 windows와 normalized를 원소별 비교하고 max/p99/mean을 기록한다.

그 재구성 normalized SHA가 기존 producer의 `inference.input_tensor_hashes`와 같아야 한다. volume/mask SHA·선택center·empty count도 producer의 `runtime.input_evidence`와 모두 같아야 한다. 이후 host간 정수입력/마스크/center를 exact 비교한다. 따라서 다른 입력으로 별도 gate를 통과시키고 실제 추론을 검증했다고 할 수 없다. normalized host간 hash가 달라도 지우지 않고 mismatch수를 보존한다. 각host가 같은 독립reference의3e-6 이내이면 triangle inequality로 host간6e-6이지만 실제원소간 차이를 측정했다고 표현하지 않는다.

## 사전 허용치와 근거

`protocol.json`은 helperSHA·serving manifestSHA·6member/12target·고정100 UID SHA·허용치·synthetic목록을 묶으며 모든 실제100 입력/예측 관측 전에 생성된다. 평가 시 외부에서 넘긴 protocolSHA도 검사한다. 수정은 새 버전으로만 하며 관측 후 허용치를 늘리지 않는다.

|대상|max|p99|mean|근거와 한계|
|---|---:|---:|---:|---|
|각host windows|5e-7|5e-7|5e-7|표준 float32 bilinear 연산 envelope4.7684e-7; compiler proof 아님|
|각host normalized|3e-6|3e-6|3e-6|같은 연산·mean/std envelope2.4435e-6; 새K24/empty 합성검사 적용|
|raw 확률: 각각6×12|0.002|0.002|0.0005|기존D01 prospective engineering budget을 새로운 member/target 각각에 고정. B 실측오차나 임상오차 근거가 아님|

p99는 max 한계를 그대로 쓰므로 별도의 경험적 percentile 기준을 만들어낸 것이 아니다. 분포 진단을 보존하며 mean gate를 별도로 쓴다. 과거D01 normalized7.15e-7 관측은 원인진단 배경이며 이번B 결과가 아니고, 그FAIL/PASS를 이번 결과로 승계하지 않는다.

순위 허용치는 고정 raw epsilon0.002로부터 유도한다. reference 값에서0.004 밖인 다른 값은 두 host에서각각±0.002 움직여도 순서가 뒤집힐 수 없다. 각UID/target의 가능한 평균동순위 rank 구간을 계산해 실제rank차이가 그 구간 이내인지 원소별로 검사한다. 이 bound의 max/p99/mean도 보고한다. blend는 같은 고정90/10·3fold 평균으로 bound를 전파한다. 밀집한 확률에서는 큰rank차이를 허용할 수 있으므로 rank지표는 수치계약의 결과이지 성능보장이 아니다. reference분포를 사용하는 고정수학식은 사후 비율/허용치 탐색과 구분한다.

합성suite는 allzero, 슬롯경계, 고립sparse, 유효sparse, highcontrast, ramp다. tests는 독립center·blank normalization·raw범위/shape/dtype·각각72개 소견gate·평균gate·rank bound·protocol변조·actual producer입력연결·합성종단간 성공/실패를 검사한다.

## 이후 명령과 운영

`execution-plan.json`에 실제 protocolSHA와 로컬 명령을 저장한다. `input`은 CPU 전용이고 torch import 전 CUDA를 숨긴다. T4 쪽 CPU입력gate는 producer가 이미 설치한 pinned offline environment가 Python path에 있어야 하며, 설치나 다운로드를 자동 수행하지 않는다. mount 경로는 실제원격 환경에서 확인한 뒤 명령을 고정한다. 각 run은 새 output을 요구하고 실패 시 FAILED 영수증을 쓴다. kernel COMPLETE나 stdout만으로 PASS하지 않는다.

평가기는 완성된 producer result의 파일hash/UID/target/rawmember/정확rank출력 검사를 먼저 재사용한다. weights/header 전체감사는 반복하지 않는다. 그 뒤 두host 입력report를 연결하고 raw6×100×12를 비교한다. 성공명칭은 `PASS_NUMERICAL_ONLY`이고 `submission_ready=false`다. decoder 압축fixture, whole-runtime/RSS, hidden 전체코호트, 경쟁품질은 별도다.

## producer를 고치지 않는 RSS 최소 방안

`tools/gpu_job.py`는 로그·PID·환경·SUCCESS/FAILED를 기록하지만 RSS peak계측은 없다. 그것만으로 이공백이 해결됐다고 하지 않는다. 다음 실행 전 별도 작은 외부 supervisor에서 producer `run`을 child로 시작하고 parent+recursive descendants의 RSS를100ms마다 샘플링한다. psutil이 없으면 계측을 누락하지 않고 시작 전 실패시킨다. PID+creation time으로 process재사용을 방지하고 접근실패·sample수·interval·timeout·exit를 기록한다. 샘플 peak는 순간최대의 하한이며 shared memory중복 때문에 시스템사용량과도 다를 수 있다.

Linux에서는 실제 cgroup 경로의 `memory.current`, `memory.peak`, `memory.events`, `memory.max`를 시작/종료시 함께 기록한다. 공유cgroup이면 session context 지표로 표시하며 producer단독peak로 주장하지 않는다. 기존peak를reset하거나 다른작업에 영향을 주지 않는다. 로컬Windows는 같은psutil tree 측정과 가용RAM을 기록한다. 이wrapper는 아직 구현하지 않았으며 root가 GPU reference를 배정하기 전10분 안팎의 별도작업으로 준비한다. sampling은 실행프로세스 내계측이며 채팅대기loop가 아니다.

GPU reference는 EXP084 학습 완료/활성작업 확인 뒤 root가 결정한다. 100입력gate 예상CPU5–15분(실측전 추정), 로컬6member추론10–20분/30분cap, T4세션30분cap은 기존계획을 유지한다. 비용·quota 확인과 실행완료ETA/heartbeat는 실제시작 담당root가 기록한다.
