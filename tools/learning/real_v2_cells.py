"""Cell sources of docs/learning/colab_knee_real_v2.0.1.ipynb (built by tools/learning/make_colab_real.py).

Placeholders @@NAME@@ are filled by the generator: embedded data (hashes, fold vectors, EXP107 OOF), the input file list and the
verbatim library pieces it extracts from the EXP107 training sources. Markdown and code comments are Korean (learning notebook style of
docs/learning/colab_knee_ablation_v1.2.0.ipynb). Nothing here runs on import.
"""

VERSION = 'v2.0.1'

MD_TITLE = r'''# 무릎 MRI 12개 소견 — RadImageNet ResNet50 실전 학습: K16 학습 창 screen → 5-fold · v2.0.1

> 버전 v2.0.1 (2026-10-08) — v2.0.0 대비: 미완료 screen은 판정하지 않음(INCOMPLETE), 마지막 epoch 뒤 한도 도달 시 내보내기·평가 전에 멈춤, timm 1.0.19 고정 + 환경 지문(다르면 이어하기·screen 짝 비교 거부), 토큰은 Colab Secret 우선·업로드는 비공개 임시 폴더, 비공개 내장 데이터 경고 강화, k_eval은 작업 시작 전에만 변경 (astra 출시 전 검토 반영).
>
> 프로젝트의 EXP107 레시피(RadImageNet ResNet50, 가운데 슬라이스×3, 224 px, 20 epoch 고정)에서
> **학습 창 개수 K12 → K16 하나만** 바꿉니다. 생성기: `tools/learning/make_colab_real.py` (손으로 고치지 말고 생성기를 고치세요).

**두 가지 실행 방식 (`CFG['mode']`)**

| 방식 | 하는 일 | 시간(T4 추정) |
|---|---|---|
| `'screen'` (기본) | fold 0만 4번: **K12 s42 → K16 s42 → K12 s43 → K16 s43** (대조군을 먼저 돌려, 끊겨도 변형만 남는 일이 없음). 평가 = 같은 Colab 플랫폼에서 K16 vs K12 짝 component bootstrap (seed 평균, K12 seed 흔들림 보고) | 약 4~5시간 |
| `'production'` | K16 5-fold (+ 선택 K12 fold 0 대조군). **screen에서 K16이 관문을 통과했을 때만** 바꾸세요 | 약 6.5~7.5시간 (+1시간) |

> 저장된 Kaggle EXP107 OOF와의 차이는 **채택 근거가 아닙니다**: 같은 EXP107이 로컬과 Kaggle에서 fold별로 0.001–0.004 Dread 달랐고
> (관문 +0.004와 같은 크기), 그래서 비교는 같은 Colab 플랫폼의 짝 대조군으로 합니다. EXP107은 "cross-platform reference, not a control" 한 줄로만 표시합니다.

무료 Colab T4에서 **끊겨도 이어서** 돌아가도록 만든 "실전" 노트북입니다. epoch마다 Drive에 체크포인트를 저장하고,
다시 실행하면 마지막으로 검증된 체크포인트에서 정확히 같은 궤적으로 이어집니다(난수·샘플러·스케줄러·GradScaler 상태 포함).

> ⛔ **비공개 노트북 — 절대 공유·공개·업로드하지 마세요.** 이 파일 안에는 우리 팀의 **EXP107 OOF 예측값**(K36·K24, gzip+base64는 그대로 복원됨),
> **검사 순서·component 벡터**, **Gold58 UID의 sha256(솔트 없음 → 후보 UID를 넣어 보면 포함 여부를 확인할 수 있음)** 이 들어 있습니다.
> 이 파일은 출력 없이 배포되었습니다. Colab에서 실행한 뒤 출력이 저장된 사본(.ipynb)도 공유하지 마세요. 대회 데이터와 결과물은 팀 밖과 공유하면 안 됩니다.

---

## ⏱ 걸리는 시간 (Colab 무료 T4 추정, 보장 아님)

| 단계 | 시간 |
|---|---|
| 0~1. 런타임 준비, Drive 연결 | 1~2분 |
| 2~3. Kaggle 토큰(Colab Secret 또는 업로드) + 데이터 32 GB 받기·검사 (VM이 바뀔 때마다) | 약 15~40분 |
| 6. 작업 하나 학습 + 평가 K36·K24: K16 / K12 | 약 75~90분 / 약 60~70분 |
| 6. screen (작업 4개) | **약 4~5시간** (여러 세션에 나눠도 됨) |
| 6. production (K16 5개 fold, + 선택 K12 fold 0) | 약 6.5~7.5시간 (+1시간) |
| 7. 평가 (bootstrap 200회) | 1~2분 |

한도: **계산 시간 9.0시간**(방식별 계획마다 따로 — screen 4작업 합계, production 작업 합계; 여러 세션 합계, Drive에 기록) · **세션 벽시계 11.5시간**(VM 하나당). 한도에 닿기 전에 epoch이 끝나는 지점에서
깔끔하게 멈추고 이어하는 방법을 출력합니다. Colab은 그보다 일찍 끊을 수도 있습니다 — 그때도 마지막 epoch까지는 Drive에 남아 있습니다.

## 🔁 끊겼을 때 이어하기

1. 런타임 → 다시 연결 (T4 GPU 확인)
2. **0번 셀부터 차례로 모두 실행** (런타임 → 모두 실행). 새 VM이면 2번에서 토큰을 다시 올리고 3번이 데이터를 다시 받습니다.
3. 6번 셀이 Drive `knee_real/<run_name>/<작업 이름>/`의 마지막 체크포인트를 찾아 이어서 학습합니다. 끝난 작업은 건너뜁니다.
   작업 이름 = 방식·학습 창 수·seed·fold (예: `screen_k12_s42_f0`, `production_k16_s42_f3`) → screen과 production은 절대 섞이지 않습니다.
4. 레시피(CFG의 위쪽 값, `k_eval` 포함)를 바꾸면 "레시피 해시"가 달라져 이어하기를 **거부**합니다. `k_eval`(K24 빼기 등)은 **작업 시작 전에만** 바꾸세요.
   다른 레시피는 `run_name`을 바꿔 새로 시작하세요.
5. 환경(python·torch·cuda/cudnn·timm·numpy·GPU)이 처음 작업을 시작했을 때와 다르면 이어하기를 **거부**합니다 (timm은 1.0.19로 고정 설치).
   screen의 4개 작업은 모두 같은 환경이어야 하며, 다르면 짝 비교를 **무효(INVALID)** 로 표시합니다 → `run_name`을 바꿔 새로 시작하세요.

## 📦 준비물

- **Kaggle 토큰**: Colab Secret `KAGGLE_API_TOKEN`(왼쪽 열쇠 아이콘, '노트북 액세스' 켜기)을 먼저 씁니다 — 디스크에 쓰지 않음.
  없으면 `kaggle.json` 또는 `access_token` 파일을 2번 셀에서 본인이 직접 업로드합니다. 어느 쪽이든 화면에 출력하지 않습니다.
- 비공개 Kaggle 데이터셋 `taeyangg4/knee-raptor-corpus-336` 접근 권한 (본인 계정), 공개 데이터셋 `marwanmath/resnet-50-radimagenet-marwan`.
- Google Drive의 `knee_compact/meta.csv` (v1.x 학습 노트북에서 올린 파일: StudyInstanceUID, fold, Dread 12개, flight_*, relabel_*).
  노트북이 내장 해시로 라벨·fold를 검사하고, 다르면 학습을 거부합니다.

## 💾 Drive 저장과 용량 주의

- 위치: `MyDrive/knee_real/<run_name>/<작업 이름>/` — 작업마다 `model_best.pt`(약 97 MB), `oof_k36.csv`, `oof_k24.csv`, 기록 파일.
- 학습 중 체크포인트(모델+옵티마이저 등, 약 0.3 GB)를 epoch마다 저장하고 **검증된 최신 2세대**만 남깁니다.
- ⚠ Drive에서 지우거나 덮어쓴 파일은 **휴지통**으로 가서 30일 동안 용량을 차지합니다. fold 하나에 약 6 GB, 5 fold에 약 30 GB가
  휴지통을 거쳐 갑니다(screen은 4작업 약 24 GB). 무료 15 GB Drive라면 중간에 용량이 찰 수 있습니다. 노트북은 Drive 남은 공간이 `drive_min_free_gb`보다 작으면
  epoch 사이에서 멈춥니다. 휴지통 비우기는 **본인이 직접** 판단해서 하세요(노트북은 절대 지우지 않습니다).
  용량이 빠듯하면 `ckpt_every_epochs=2`로 바꾸면 휴지통 양이 절반이 됩니다(끊기면 최대 2 epoch을 다시 학습).

## 🎯 정직한 기대치

- 이 노트북 하나로 기대하는 public 점수는 **약 0.91~0.93 (중심 ≈0.92)** 입니다. **0.95는 이 노트북만으로 기대하기 어렵습니다.**
- 7번 셀의 관문(ΔDread ≥ +0.004이고 구간 > 0, Δconsensus ≥ 0, flight·relabel ≥ −0.002)은 **잠정 진급 기준**이지 public 향상의 증명이 아닙니다.
  screen에서는 같은 Colab 플랫폼의 K16 vs K12 (seed 2개 평균)에 적용합니다.
- 저장된 EXP107 OOF(Kaggle T4)는 **cross-platform reference, not a control**입니다. 짝 대조군은 같은 Colab 조건의 K12입니다.

## ⚖ 라이선스와 출처

- RadImageNet ResNet50 가중치(Kaggle `marwanmath/resnet-50-radimagenet-marwan`, sha256 `08629f7e…`)는 Kaggle 데이터셋 라이선스가
  **CC-BY-NC-SA-4.0**(비상업·동일조건)입니다. 원 저작물(github.com/BMEII-AI/RadImageNet)과 이 사본의 동일성은 확인하지 않았습니다.
  출처 기록을 유지하고, **상금 자격(prize eligibility)은 확인되지 않았습니다.** 가중치와 학습된 체크포인트를 공개 재배포하지 마세요.
'''

MD_0 = r'''## 0. 런타임 설정 · 버전 · 설정값(CFG)
Colab 메뉴 → **런타임 → 런타임 유형 변경 → T4 GPU**. `CFG` 위쪽(레시피)은 EXP107 값 그대로이고 `k_train=16`만 다릅니다.
아래쪽(실행 방식·설정)은 바꿔도 이어하기에 영향이 없습니다. 처음에는 `mode='screen'` 그대로 두세요.
'''

MD_1 = r'''## 1. Google Drive 연결 + 실행 계획
체크포인트와 결과는 Drive `knee_real/<작업 이름>/`에 저장됩니다. 실행할 작업 목록(순서)과 지난 세션의 진행 상황을 보여 줍니다.
'''

MD_2 = r'''## 2. Kaggle 토큰 — Colab Secret 우선, 없으면 본인이 직접 업로드
- ① Colab Secret `KAGGLE_API_TOKEN`이 있고 '노트북 액세스'가 켜져 있으면 환경 변수로만 씁니다 (디스크에 쓰지 않음).
- ② 없으면 `kaggle.json`(옛 형식) 또는 `access_token`(새 형식) 파일 **하나**를 이름 그대로 선택하세요. `/content` 아래 비공개 임시 폴더로 받은 뒤
  `/root/.kaggle`에 권한 600으로만 복사하고 임시 사본은 지웁니다 (Drive에는 쓰지 않음). 다른 이름·여러 파일이면 저장하지 않고 멈춥니다.
- 토큰 값은 **출력하지 않습니다**. 이 VM에 데이터가 이미 다 받아져 있으면 이 단계를 건너뜁니다.
'''

MD_3 = r'''## 3. 데이터 받기 — 파일별로 이어받기 + 체크섬 검사
- 코퍼스 `taeyangg4/knee-raptor-corpus-336` (4,410 검사 × 64 슬롯 × 336² uint8, 12개 shard, 31.9 GB) → Colab 로컬 디스크 `/content/corpus`.
- 가중치 `marwanmath/resnet-50-radimagenet-marwan` 의 `ResNet50.pt` → sha256 `08629f7e…` 확인.
- 이미 받은 파일은 크기·체크섬을 확인하고 건너뜁니다. 실패하면 4번까지 다시 받습니다. 받기 전에 디스크 여유 공간을 확인합니다.
- 학습은 Drive가 아니라 **로컬 디스크의 shard를 memory-map**으로 읽습니다 (Drive에서 직접 읽으면 매우 느림).
'''

MD_4 = r'''## 4. 라벨과 fold — 내장 해시로 검사
- 4a: 생성 시점에 프로젝트 파일에서 계산한 **검증용 해시·fold 벡터·EXP107 저장 OOF**(float16, gzip+base64). 수정하지 마세요.
- 4b: Drive `knee_compact/meta.csv`를 읽어 Dread 라벨(float32, EXP099 순서)의 sha256, fold 구성, component 분리(같은 component가 두 fold에
  걸치지 않음), Gold58 제외, 코퍼스 포함 여부를 확인합니다. Dread·fold가 다르면 **학습을 거부**합니다.
'''

MD_5 = r'''## 5. 인라인 라이브러리 — EXP107 학습 코드를 글자 그대로 옮김
코퍼스 읽기(memory-map), 창 중심 고르기(within-slot-valid), 가운데 슬라이스×3 + 리사이즈, 정규화 2x−1, RadImageNet 가중치 strict 로드,
EXP107 모델/헤드(슬라이스별 attention MIL), 샘플링 난수 흐름과 밝기 jitter가 모두 원본과 같습니다. 각 조각 위에 원본 파일과 함수 이름을 적었습니다.
'''

MD_6 = r'''## 6. 학습 실행 — 작업별, epoch마다 Drive 체크포인트, 끊기면 이어하기
- 1번 셀이 보여 준 순서로 작업을 학습합니다 (screen: K12 s42 → K16 s42 → K12 s43 → K16 s43 / production: K16 fold 0~4 → K12 fold 0).
  끝난 작업은 건너뛰고, 중간에 끊긴 작업은 마지막 체크포인트에서 이어집니다.
- 체크포인트: 모델·옵티마이저·스케줄러·GradScaler·난수(python/numpy/torch/cuda)·샘플러 상태·epoch·업데이트 수(시도/성공)·레시피 해시.
  임시 파일에 쓴 뒤 이름을 바꾸고(원자적 저장), 다시 읽어 sha256을 확인한 **최신 2세대**를 남깁니다.
- 첫 epoch에는 **데이터 대기 시간 vs 계산 시간**을 측정해 출력합니다(데이터 대기가 크면 CPU/디스크가 병목).
- production에서 `run_control=True`이면 5개 fold 뒤에 fold 0을 K12(EXP107 그대로)로 한 번 더 학습합니다 — 같은 Colab 조건의 짝 대조군.
  (screen의 `screen_k12_s42_f0`과 같은 레시피라, 평가 셀은 production 대조군이 없으면 그것을 씁니다.)
- 각 fold의 최종 체크포인트는 프로젝트 릴리스 도구(`tools/exp104_release_v5.py`/`exp111_release_v6.py`)가 읽는 형식 그대로입니다.
'''

MD_7 = r'''## 7. 평가 — 같은 플랫폼 짝 비교(component bootstrap), 잠정 관문
- 정답지: Dread(학습에 쓴 라벨, float32 → `numpy.rint`), flight, relabel, consensus(세 라벨이 모두 같은 칸만).
- screen: Δ = K16 − K12 (같은 Colab, seed 42·43 평균), K12 seed 흔들림도 보고. 구간은 **component 단위** 복원추출 200회(모든 정답지·K·seed에 같은 표본)의 95% 범위.
- 판정(PASS / NOT PASSED)은 서로 다른 seed 2개의 K12·K16 짝이 모두 끝나고(전체 검증 명단) 4개 작업의 환경 지문이 같을 때만 합니다.
  아니면 `INCOMPLETE — diagnostic only` 또는 `INVALID`(환경 불일치)로 표시하고 production 권고를 하지 않습니다.
- production: fold별·전체·소견별 AUC, fold 0 K12 대조군과의 짝 비교. 저장된 EXP107은 어느 방식에서나 **cross-platform reference, not a control**.
- 마지막에 나오는 **붙여넣기 표**를 복사해 대화창에 붙여 주세요(버전과 레시피 해시 포함).
'''

# ---------------------------------------------------------------------------------------------------------------- cell 0
CELL_0 = r'''# ===== [0] 버전 · 설정(CFG) · GPU 확인 · notebook v2.0.1 =====
print('notebook v2.0.1')
import os, sys, time, json, hashlib, importlib.util, subprocess, platform
# EXP107의 strict 런타임(결정론적 cuBLAS)은 CUDA가 초기화되기 전에 설정해야 합니다.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
NOTEBOOK_VERSION = 'v2.0.1'
ON_COLAB = 'google.colab' in sys.modules                     # 로컬 테스트 하네스에서는 False

CFG = dict(
    # ---- 레시피 (EXP107 그대로, k_train만 12 → 16). 이 값들이 '레시피 해시'가 되며, 학습 도중 바꾸면 이어하기를 거부합니다 ----
    arch='resnet50', init='radimagenet', res=224,
    k_train=16,                     # 학습 창 개수 (EXP107: 12) ← 이번 실험에서 바꾼 유일한 값
    k_eval=[36, 24],                # 평가 창 개수: K36이 주 지표, K24는 진단용. 레시피 해시에 포함 → 바꾸려면 작업 시작 전에만 (K24는 작업당 약 1분)
    epochs=20,                      # 고정 20 epoch, 마지막 epoch 모델 사용 (검증 점수로 고르지 않음)
    bs=8, accum=1, drop_last=True, eval_bs=4,
    bb_lr=5e-4, head_lr=1e-3, wd=0.02, pct_start=0.15, grad_clip=3.0,   # AdamW + OneCycle (나머지는 torch 기본값)
    amp='fp16', amp_init_scale=1024.0, eval_amp='fp16', runtime_policy='strict',
    norm='radimagenet', input_policy='center3', window_policy='within-slot-valid',
    jitter=0.20,                    # 학습 때 검사 단위 밝기 배율 0.9~1.1 (EXP107 StudyWindows 기본값)
    head_dropout=0.2,               # 헤드 attention dropout (build_model 기본값, 기록용)
    seed=42,                        # production seed (screen은 아래 screen_seeds를 씀; 작업마다 seed가 레시피 해시에 들어감)
    label_set='dread', loss_policy='clipped_positive_term', checkpoint_selection='fixed-final',
    ema=False, swa=False, affine=False, flip=False,             # v2.0.1은 모두 끔 (켜면 거부)
    # ---- 실행 방식 (해시에 들어가지 않음: 바꿔도 이어하기 가능) ----
    run_name='run1',                # 실행 이름: Drive knee_real/<run_name>/ 아래에 작업 폴더. 환경이 바뀌어 이어하기가 거부되면 새 이름으로
    mode='screen',                  # 'screen'(기본) = fold 0만: K12 s42 → K16 s42 → K12 s43 → K16 s43 (대조군을 먼저 → 끊겨도 짝이 남음, 약 4~5시간)
                                    # 'production' = K16 5-fold (+ 선택 K12 fold 0 대조군). screen에서 K16이 관문을 통과했을 때만 바꾸세요.
    screen_fold=0, screen_seeds=[42, 43],
    folds=[0, 1, 2, 3, 4], run_control=True, control_k_train=12, control_fold=0,   # production 전용
    train=True,                     # False면 학습 없이 7번 평가만
    device='cuda', num_workers=2, prefetch_factor=2, main_threads=1,
    compute_limit_h=9.0,            # 계산 한도: 방식별 계획마다 따로 (screen 작업 합계 / production 작업 합계)
    session_limit_h=11.5, reserve_min=15,
    prior_epoch_s_per_k=17.5,       # 측정 전 epoch 시간 추정 (Kaggle T4 K12 168초 ÷ 12 × 1.25)
    ckpt_every_epochs=1, keep_generations=2, delete_train_ckpt_after_fold=True, drive_min_free_gb=2.0,
    drive_root='/content/drive/MyDrive/knee_real', meta_path='/content/drive/MyDrive/knee_compact/meta.csv',
    corpus_dir='/content/corpus', weights_dir='/content/weights', verify_dir='/content/knee_verified',
    local_tmp='/content/knee_tmp', verify_shards='sha256',
    n_boot=200, boot_seed=0, print_every=200,
    trace_batches=2, trace_x_hash=False, heartbeat_s=300,
    subset=None, stop_after_epochs=None, crash_at=None,         # (로컬 테스트 전용) 일부 검사만 쓰기 / 강제 중단 시험
)
CFG.update(globals().get('CFG_OVERRIDE') or {})              # (로컬 테스트 전용) Colab에서는 아무것도 바꾸지 않습니다

# 세션 벽시계 시작 시각: 같은 VM이면 커널을 다시 시작해도 유지되도록 /content에 기록합니다 (VM이 바뀌면 새로 시작).
if ON_COLAB:
    _t0_file = '/content/.knee_v2_session_t0'
    if not os.path.exists(_t0_file):
        with open(_t0_file, 'w') as _f:
            _f.write(str(time.time()))
    SESSION_T0 = float(open(_t0_file).read())
else:
    SESSION_T0 = time.time()
SESSION_ID = time.strftime('%Y%m%d-%H%M%S') + '-' + os.urandom(4).hex()   # 전역 난수를 쓰지 않음

if ON_COLAB:
    _r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
    print('GPU:', _r.stdout.strip() or '(없음)')
    if CFG['device'] == 'cuda' and not _r.stdout.strip():
        raise SystemExit('GPU가 없습니다 → 런타임 → 런타임 유형 변경 → T4 GPU')
    if CFG['device'] == 'cuda' and 'T4' not in _r.stdout:
        print('⚠ T4가 아닙니다. 시간 추정과 FP16 동작이 문서와 다를 수 있습니다.')
# timm은 로컬 CPU 시험(하네스)과 같은 1.0.19로 고정합니다. timm 1.0.19의 torch 요구에는 상한이 없어 Colab torch와 충돌하지 않습니다.
TIMM_PIN = '1.0.19'
from importlib import metadata as _md
try:
    _timm_installed = _md.version('timm')
except _md.PackageNotFoundError:
    _timm_installed = None
if _timm_installed != TIMM_PIN:
    if 'timm' in sys.modules:
        raise SystemExit(f'timm {_timm_installed}이(가) 이미 불러와졌습니다 → 런타임 → 세션 다시 시작 후 0번 셀부터 다시 실행하세요')
    print(f'timm {_timm_installed} → {TIMM_PIN} 설치')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'timm=={TIMM_PIN}'], check=True)
import numpy as np, pandas as pd, torch, timm
if timm.__version__ != TIMM_PIN:
    raise SystemExit(f'timm {timm.__version__} ≠ {TIMM_PIN}: 런타임 → 세션 다시 시작 후 0번 셀부터 다시 실행하세요')
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | timm {timm.__version__} | numpy {np.__version__} | "
      f"pandas {pd.__version__} | 장치 {CFG['device']} | 세션 경과 {(time.time() - SESSION_T0) / 3600:.2f} h")
'''

# ---------------------------------------------------------------------------------------------------------------- cell 1
CELL_1 = r'''# ===== [1] Google Drive 연결 + 실행 계획(작업 폴더) =====
import shutil
if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')                            # 브라우저 팝업에서 Google 계정 권한을 허용
if CFG['mode'] not in ('screen', 'production'):
    raise SystemExit("CFG['mode']는 'screen' 또는 'production'이어야 합니다")

def job_name(mode, k_train, seed, fold):
    """작업 폴더 이름 = 방식·학습 창 수·seed·fold. screen과 production의 체크포인트가 절대 섞이지 않습니다."""
    return f'{mode}_k{int(k_train)}_s{int(seed)}_f{int(fold)}'

def plan_root():
    """Drive knee_real/<run_name>/ — 작업 폴더, 방식별 환경 지문, 평가 기록이 여기에 있습니다."""
    return os.path.join(CFG['drive_root'], CFG['run_name'])

def planned_jobs():
    """실행할 작업 (mode, k_train, seed, fold) 순서. screen은 같은 seed의 대조군(K12)을 변형(K16)보다 먼저 돌립니다."""
    if CFG['mode'] == 'screen':
        f = int(CFG['screen_fold'])
        return [('screen', k, int(s), f) for s in CFG['screen_seeds'] for k in (int(CFG['control_k_train']), int(CFG['k_train']))]
    jobs = [('production', int(CFG['k_train']), int(CFG['seed']), int(f)) for f in CFG['folds']]
    if CFG['run_control']:
        jobs.append(('production', int(CFG['control_k_train']), int(CFG['seed']), int(CFG['control_fold'])))
    return jobs

os.makedirs(plan_root(), exist_ok=True)
_du = shutil.disk_usage(CFG['drive_root'])
print(f"실행 폴더 {plan_root()} | Drive 남은 공간 {_du.free / 1e9:.1f} GB / 전체 {_du.total / 1e9:.0f} GB")
if _du.free / 1e9 < 8:
    print('⚠ Drive 여유 공간이 8 GB 미만입니다. 체크포인트가 휴지통에 쌓이면 학습 중 멈출 수 있습니다 (위 "Drive 용량 주의" 참고).')
print(f"방식 {CFG['mode']} | 작업 {len(planned_jobs())}개 (이 순서로 실행):")
for _j in planned_jobs():                                    # 지난 세션의 진행 상황 요약
    _p = os.path.join(plan_root(), job_name(*_j), 'run_state.json')
    if not os.path.exists(_p):
        print(f'  {job_name(*_j)}: 아직 시작 전')
        continue
    with open(_p, encoding='utf-8') as _f:
        _s = json.load(_f)
    _v = _s['folds'].get(str(_j[3]), {})
    _st = '완료' if _v.get('status') == 'done' else f"epoch {_v.get('completed_epochs', 0)}/{CFG['epochs']}까지 저장됨 → 이어서 학습"
    print(f"  {job_name(*_j)}: {_st} | 계산 누적 {_s['compute_seconds'] / 3600:.2f} h | 세션 {len(_s['sessions'])}회 | 레시피 해시 {_s['recipe_hash'][:12]}")
'''

# ---------------------------------------------------------------------------------------------------------------- cell 2
CELL_2 = r'''# ===== [2] Kaggle 토큰: Colab Secret 우선 → 없으면 비공개 임시 폴더로 업로드 (값은 출력하지 않음) =====
import shutil
# 받을 파일 목록 (생성 시점에 프로젝트 파일에서 계산한 크기와 sha256)
INPUTS = json.loads(r"""@@INPUTS_JSON@@""")
CORPUS_DIR, WEIGHTS_PATH = CFG['corpus_dir'], os.path.join(CFG['weights_dir'], INPUTS['weights']['name'])
os.makedirs(CFG['verify_dir'], exist_ok=True)

def _marker_path(name):
    return os.path.join(CFG['verify_dir'], name + '.json')

def marker_ok(path, name, level):
    """검사 기록이 있고 파일 크기·수정 시각이 그대로면 다시 해시하지 않습니다. level='sha256'이면 sha256 검사 기록만 인정."""
    m = _marker_path(name)
    if not (os.path.exists(path) and os.path.exists(m)):
        return False
    with open(m) as f:
        rec = json.load(f)
    st = os.stat(path)
    return rec.get('bytes') == st.st_size and rec.get('mtime_ns') == st.st_mtime_ns and (level == 'size' or rec.get('level') == 'sha256')

def inputs_already_complete():
    level = CFG['verify_shards']
    files = [(os.path.join(CORPUS_DIR, n), n, (level if n.startswith('vols_shard') else 'sha256')) for n, _, _ in INPUTS['corpus_files']]
    files.append((WEIGHTS_PATH, INPUTS['weights']['name'], 'sha256'))
    return all(marker_ok(p, n, lv) for p, n, lv in files)

KAGGLE_DIR = '/root/.kaggle'
TOKEN_NAMES = ('kaggle.json', 'access_token')
SECRET_HINT = ("Colab Secret KAGGLE_API_TOKEN을 쓸 수 없습니다. 왼쪽 열쇠 아이콘 → KAGGLE_API_TOKEN의 '노트북 액세스' 스위치를 켜세요. "
               '지금은 토큰 파일(이미 있으면 그 파일, 없으면 업로드)로 진행합니다.')

def token_from_secret(get_secret=None):
    """Colab Secrets의 KAGGLE_API_TOKEN을 환경 변수로만 설정합니다 (디스크에 쓰지 않고, 값은 출력하지 않음). 성공하면 True.
    kaggle CLI 2.x(kagglesdk)는 KAGGLE_API_TOKEN 환경 변수를 파일보다 먼저 읽고, subprocess는 os.environ을 물려받습니다."""
    try:
        if get_secret is None:
            from google.colab import userdata
            get_secret = userdata.get
        tok = get_secret('KAGGLE_API_TOKEN')
    except Exception as e:                                   # SecretNotFoundError, NotebookAccessError 등 (메시지·값은 출력하지 않음)
        print(f'{SECRET_HINT} ({type(e).__name__})')
        return False
    tok = (tok or '').strip()
    if not tok:
        print(SECRET_HINT + ' (비어 있음)')
        return False
    os.environ['KAGGLE_API_TOKEN'] = tok
    del tok
    return True

def install_token(upload, kaggle_dir=KAGGLE_DIR, base='/content'):
    """files.upload()는 받은 파일을 '현재 폴더'에도 씁니다. 그래서 /content 아래 비공개 임시 폴더(700)로 들어가서 올리고,
    정확히 한 파일(kaggle.json 또는 access_token)만 kaggle_dir에 권한 600으로 복사한 뒤 임시 사본과 폴더를 지웁니다. 내용은 출력하지 않습니다."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix='knee_token_', dir=base)
    os.chmod(tmp, 0o700)
    cwd = os.getcwd()
    up = None
    try:
        os.chdir(tmp)
        try:
            up = upload()
        finally:
            os.chdir(cwd)                                    # 업로드가 실패해도 원래 폴더로
        names = list(up or {})
        if len(names) != 1 or names[0] not in TOKEN_NAMES:
            raise SystemExit(f'토큰 파일 하나(kaggle.json 또는 access_token, 이름 그대로)만 올리세요 — 받은 파일 {len(names)}개, 아무것도 저장하지 않았습니다')
        name = names[0]
        os.makedirs(kaggle_dir, exist_ok=True)
        os.chmod(kaggle_dir, 0o700)
        dst = os.path.join(kaggle_dir, name)
        if os.path.exists(dst):
            os.remove(dst)
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(up[name])
        os.chmod(dst, 0o600)
        return dst
    finally:
        up = None
        shutil.rmtree(tmp, ignore_errors=True)               # 임시 사본 삭제 (Drive에는 아무것도 쓰지 않음)

def activate_file_token(kaggle_dir=KAGGLE_DIR):
    """새 형식 access_token 파일은 환경 변수로도 전달합니다 (kaggle.json은 CLI가 직접 읽음). 값은 출력하지 않습니다."""
    p = os.path.join(kaggle_dir, 'access_token')
    if os.path.exists(p):
        with open(p) as f:
            os.environ['KAGGLE_API_TOKEN'] = f.read().strip()

def setup_kaggle_token(get_secret=None, upload=None, kaggle_dir=KAGGLE_DIR, base='/content'):
    """① Colab Secret KAGGLE_API_TOKEN (디스크에 쓰지 않음) → ② 이미 있는 토큰 파일 → ③ 파일 업로드(비공개 임시 폴더). 사용한 경로 이름을 돌려줍니다."""
    if token_from_secret(get_secret):
        print('토큰 준비 완료 (Colab Secrets)')
        return 'secret'
    source = 'file'
    if not any(os.path.exists(os.path.join(kaggle_dir, n)) for n in TOKEN_NAMES):
        if upload is None:
            from google.colab import files
            upload = files.upload
        print('kaggle.json 또는 access_token 파일을 선택하세요 (Kaggle → Settings → API, 파일 이름을 바꾸지 마세요)')
        install_token(upload, kaggle_dir, base)
        source = 'upload'
    activate_file_token(kaggle_dir)
    print('토큰 준비 완료 (파일, 내용은 출력하지 않습니다)')
    return source

if not ON_COLAB:
    print('로컬 테스트: 토큰 단계 건너뜀')
elif inputs_already_complete():
    print('데이터가 이미 이 VM에 모두 있습니다 → 토큰 단계 건너뜀')
else:
    TOKEN_SOURCE = setup_kaggle_token()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'kaggle'], check=False)   # 2.x: KAGGLE_API_TOKEN 환경 변수 지원
'''

# ---------------------------------------------------------------------------------------------------------------- cell 3
CELL_3 = r'''# ===== [3] 데이터 받기: 파일별 이어받기 · 재시도 · 디스크 확인 · 체크섬 =====
import shutil, zipfile

def sha256_of(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def write_marker(path, name, level, digest):
    st = os.stat(path)
    tmp = _marker_path(name) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, level=level, sha256=digest), f)
    os.replace(tmp, _marker_path(name))

def kaggle_fetch(ref, name, dest):
    """Kaggle CLI로 파일 하나를 받습니다. 출력은 마지막 300자만 돌려줍니다(토큰은 CLI가 출력하지 않음)."""
    r = subprocess.run(['kaggle', 'datasets', 'download', '-d', ref, '-f', name, '-p', dest, '--unzip', '-q', '-o'],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-300:].replace('\n', ' ')

FETCH = globals().get('FETCH_OVERRIDE') or kaggle_fetch     # (로컬 테스트 전용) 하네스가 가짜 받기 함수를 넣을 수 있음
ALLOW_FETCH = ON_COLAB or 'FETCH_OVERRIDE' in globals()

def ensure_file(ref, name, size, sha, dest, level='sha256', retries=4):
    """dest/name이 크기(와 sha256)까지 맞으면 그대로 두고, 아니면 지우고 다시 받습니다. 결과 문자열을 돌려줍니다."""
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, name)
    if marker_ok(path, name, level):
        return 'ok(이전 검사)'
    t0 = time.time()
    for attempt in range(retries + 1):
        if os.path.exists(path):
            if os.path.getsize(path) == size:
                if level == 'size' or sha is None:
                    write_marker(path, name, 'size', None)
                    return f'ok(크기) {time.time() - t0:.0f}s'
                got = sha256_of(path)
                if got == sha:
                    write_marker(path, name, 'sha256', got)
                    return f'ok(sha256) {time.time() - t0:.0f}s'
                problem = 'sha256 불일치'
            else:
                problem = f'크기 {os.path.getsize(path)} ≠ {size} (덜 받아짐)'
            if not ALLOW_FETCH:                              # 로컬 테스트: 원본 파일을 절대 지우지 않음
                raise SystemExit(f'{name}: {problem} (로컬 테스트라 지우거나 받지 않음): {path}')
            print(f'  {name}: {problem} → 지우고 다시 받습니다')
            os.remove(path)
        if attempt == retries:
            break
        if not ALLOW_FETCH:
            raise SystemExit(f'{name}: 파일이 없습니다 (로컬 테스트라 받지 않음): {path}')
        need = 2 * size + 2e9                                # zip 압축본과 풀린 파일이 잠시 함께 있을 수 있음
        free = shutil.disk_usage(dest).free
        if free < need:
            raise SystemExit(f'디스크 부족: {name}에 {need / 1e9:.1f} GB 필요, 남은 공간 {free / 1e9:.1f} GB')
        rc, tail = FETCH(ref, name, dest)
        zp = path + '.zip'
        if not os.path.exists(path) and os.path.exists(zp):  # CLI가 압축본만 남긴 경우 직접 풉니다
            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
            os.remove(zp)
        if not os.path.exists(path):
            print(f'  {name}: 받기 실패 (시도 {attempt + 1}/{retries}, rc={rc}) {tail}')
            time.sleep(30 if ON_COLAB else 0)
    raise SystemExit(f'{name}: {retries}번 시도했지만 올바른 파일을 받지 못했습니다')

def ensure_inputs():
    t0 = time.time()
    w = INPUTS['weights']
    print('가중치', w['name'], ensure_file(w['ref'], w['name'], w['bytes'], w['sha256'], CFG['weights_dir']))
    for name, size, sha in INPUTS['corpus_files']:
        level = CFG['verify_shards'] if name.startswith('vols_shard') else 'sha256'
        print(f'코퍼스 {name} ({size / 1e9:.2f} GB)', ensure_file(INPUTS['corpus_ref'], name, size, sha, CORPUS_DIR, level), flush=True)
    with open(os.path.join(CORPUS_DIR, 'manifest.json'), encoding='utf-8') as f:
        man = json.load(f)
    listed = [[s['file'], s['bytes'], s['sha256']] for s in man['shards']]
    if listed != [x for x in INPUTS['corpus_files'] if x[0].startswith('vols_shard')]:
        raise SystemExit('코퍼스 manifest의 shard 목록이 내장 목록과 다릅니다')
    print(f"입력 준비 완료 {time.time() - t0:.0f}s | 코퍼스 계약 {man['contract']} | 로컬 디스크 남은 공간 "
          f"{shutil.disk_usage(CORPUS_DIR).free / 1e9:.0f} GB")
    return man

CORPUS_MANIFEST = ensure_inputs()
CORPUS_CONTRACT = CORPUS_MANIFEST['contract']
'''

# ---------------------------------------------------------------------------------------------------------------- cell 4a
CELL_4A_HEAD = r'''# ===== [4a] 내장 검증 데이터 (생성기가 프로젝트 파일에서 계산 · 수정하지 마세요) =====
# ⛔ 비공개: EXP107 OOF 예측값(복원 가능), 검사 순서·component 벡터, 솔트 없는 Gold58 UID sha256이 들어 있습니다. 이 노트북을 공유·공개하지 마세요.
import base64, gzip

def _unpack(lines):
    return gzip.decompress(base64.b64decode(''.join(lines)))

# 해시·fold 벡터·pos_weight·EXP107 기준값 (사람이 읽을 수 있는 JSON)
EMB = json.loads(r"""@@EMB_JSON@@""")
'''

# ---------------------------------------------------------------------------------------------------------------- cell 4
CELL_4 = r'''# ===== [4] 라벨과 fold: Drive의 knee_compact/meta.csv + 내장 해시 검사 =====
import glob
# ---- 라벨·fold 도구 (생성기가 원본에서 글자 그대로 복사) ----
@@LABEL_UTILS@@

# ---- meta.csv 찾기 ----
META_PATH = CFG['meta_path']
if not os.path.exists(META_PATH):
    _hits = sorted(glob.glob('/content/drive/MyDrive/**/knee_compact/meta.csv', recursive=True))[:3] if ON_COLAB else []
    if not _hits:
        raise SystemExit(f'meta.csv를 찾지 못했습니다: {META_PATH} (Drive의 knee_compact/ 폴더에 올렸는지 확인하세요)')
    META_PATH = _hits[0]
meta = pd.read_csv(META_PATH, dtype={UID: str})
_need = [UID, 'fold'] + LAB + [f'{p}_{t}' for p in ('flight', 'relabel') for t in LAB]
_miss = [c for c in _need if c not in meta.columns]
if _miss:
    raise SystemExit(f'meta.csv에 열이 없습니다: {_miss[:5]}')
if meta[UID].isna().any() or meta[UID].duplicated().any():
    raise SystemExit('meta.csv에 빈 UID 또는 중복 UID가 있습니다')
UIDS_SORTED = sorted(meta[UID].astype(str))
if len(UIDS_SORTED) != EMB['n_studies'] or hashlib.sha256('\n'.join(UIDS_SORTED).encode()).hexdigest() != EMB['sorted_uids_sha256']:
    raise SystemExit('meta.csv의 검사 목록이 EXP099의 4,343 G1 검사와 다릅니다 → 학습 거부')
meta = meta.set_index(UID).loc[UIDS_SORTED]
ROW_OF = {u: i for i, u in enumerate(UIDS_SORTED)}

# ---- EXP099 순서(canonical: EXP084 ordered train + val)와 Dread 라벨 sha256 ----
CANON = [UIDS_SORTED[i] for i in np.argsort(CANON_RANK, kind='stable')]
if uid_hash(CANON) != EMB['canonical_sha256']:
    raise SystemExit('내장 canonical 순서가 손상되었습니다')
DREAD_CANON = meta.loc[CANON, LAB].to_numpy(dtype=np.float32)
if hashlib.sha256(np.ascontiguousarray(DREAD_CANON, '<f4').tobytes()).hexdigest() != EMB['dread_f32_sha256']:
    raise SystemExit('Dread 라벨(float32, EXP099 순서)의 sha256이 EXP107과 다릅니다 → 학습 거부')
if not (np.isfinite(DREAD_CANON).all() and DREAD_CANON.min() >= 0 and DREAD_CANON.max() <= 1):
    raise SystemExit('Dread 라벨 값이 [0,1] 밖입니다')
DREAD_OF = dict(zip(CANON, DREAD_CANON))
DREAD_SORTED = meta[LAB].to_numpy(dtype=np.float32)

# ---- fold 구성: sha256 + fold별 순서 있는 train/val 목록 해시(EXP107 manifest) ----
FOLD_OF = dict(zip(UIDS_SORTED, meta['fold'].astype(int)))
if hashlib.sha256(np.asarray([FOLD_OF[u] for u in CANON], np.int8).tobytes()).hexdigest() != EMB['fold_sha256']:
    raise SystemExit('fold 구성이 EXP099와 다릅니다 → 학습 거부')
FOLD_IDS = {}
for _k in range(5):
    _f = EMB['folds'][str(_k)]
    _tr = [u for u in CANON if FOLD_OF[u] != _k]
    _va = [u for u in CANON if FOLD_OF[u] == _k]
    if (len(_tr), len(_va)) != (_f['n_train'], _f['n_val']) or uid_hash(_tr) != _f['train_sha256'] or uid_hash(_va) != _f['val_sha256']:
        raise SystemExit(f'fold {_k}의 순서 있는 train/val 목록이 EXP107과 다릅니다 → 학습 거부')
    if not np.array_equal(pos_weight(DREAD_OF, _tr), np.asarray(_f['pos_weight'], np.float32)):
        raise SystemExit(f'fold {_k}의 pos_weight가 EXP107 등록값과 다릅니다 → 학습 거부')
    FOLD_IDS[_k] = dict(train=_tr, val=_va)

# ---- component 분리(같은 component가 두 fold에 걸치지 않음), Gold58 제외, 코퍼스 포함 ----
if hashlib.sha256(np.ascontiguousarray(COMPONENT_IDX, '<i2').tobytes()).hexdigest() != EMB['components']['idx_sha256']:
    raise SystemExit('내장 component 표가 손상되었습니다')
_cf = pd.Series(meta['fold'].to_numpy()).groupby(COMPONENT_IDX).nunique()
if (_cf != 1).any():
    raise SystemExit(f'{int((_cf != 1).sum())}개 component가 두 fold 이상에 걸칩니다 → 학습 거부')
_gold = set(EMB['gold58_uid_sha256'])
if any(hashlib.sha256(u.encode()).hexdigest() in _gold for u in UIDS_SORTED):
    raise SystemExit('Gold58 검사가 meta.csv에 들어 있습니다 → 학습 거부')
_corpus_ids = set(np.load(os.path.join(CORPUS_DIR, 'ids.npy'), allow_pickle=True).astype(str).tolist())
if not set(UIDS_SORTED) <= _corpus_ids:
    raise SystemExit('코퍼스에 없는 검사가 있습니다')

# ---- 비교용 정답지 flight / relabel (학습에는 쓰지 않음). 다르면 경고만 하고 그 정답지는 평가에서 뺍니다 ----
REF = {}
REF_OK = {}
for _p in ('flight', 'relabel'):
    _a = meta[[f'{_p}_{t}' for t in LAB]].to_numpy(dtype=np.int8)
    REF[_p] = _a
    REF_OK[_p] = hashlib.sha256(np.ascontiguousarray(_a).tobytes()).hexdigest() == EMB[f'{_p}_sha256']
    if not REF_OK[_p]:
        print(f'⚠ {_p}_* 열이 생성 시점 값과 다릅니다 → 평가에서 {_p}와 consensus를 뺍니다')

# ---- 저장된 EXP107 OOF(비교 기준) 복원 + sha256 확인 ----
EXP107_OOF = {}
for _k in (36, 24):
    _b = EXP107_OOF16[_k]
    if hashlib.sha256(_b).hexdigest() != EMB['exp107']['oof_f16_sha256'][str(_k)]:
        raise SystemExit(f'내장 EXP107 K{_k} OOF가 손상되었습니다')
    EXP107_OOF[_k] = np.frombuffer(_b, '<f2').reshape(EMB['n_studies'], len(LAB)).astype(np.float32)
print(f"라벨·fold 검사 통과: {len(UIDS_SORTED)} 검사, fold 크기 {[len(FOLD_IDS[k]['val']) for k in range(5)]}, "
      f"component {EMB['components']['n']}개 모두 한 fold 안, Gold58 없음, flight {REF_OK['flight']}, relabel {REF_OK['relabel']} | {META_PATH}")
'''

# ---------------------------------------------------------------------------------------------------------------- cell 5
CELL_5_HEAD = r'''# ===== [5] 인라인 라이브러리 — EXP107 학습 코드를 글자 그대로 옮김 · notebook v2.0.1 =====
# 각 조각은 생성기(tools/learning/make_colab_real.py)가 원본 파일에서 그대로 복사했습니다. 원본 파일 sha256(= EXP107 provenance):
@@SOURCE_HASHES@@
# 이 셀은 정의만 합니다(실행되는 학습 없음). DataLoader 작업 프로세스도 이 정의를 씁니다.
import hashlib, json, os, random, re
from pathlib import Path
import numpy as np
'''

# ---------------------------------------------------------------------------------------------------------------- cell 6
CELL_6 = r'''# ===== [6] 학습 실행: fold별 · epoch마다 Drive 체크포인트 · 끊기면 이어하기 · notebook v2.0.1 =====
import contextlib, gc, io, math, random, shutil, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader

# ---- 레시피 해시: 이 값들 + 데이터 해시가 같아야 이어서 학습합니다 ----
RECIPE_KEYS = ('arch', 'init', 'res', 'k_eval', 'epochs', 'bs', 'accum', 'drop_last', 'eval_bs', 'bb_lr', 'head_lr', 'wd', 'pct_start',
               'grad_clip', 'amp', 'amp_init_scale', 'eval_amp', 'runtime_policy', 'norm', 'input_policy', 'window_policy', 'jitter',
               'head_dropout', 'seed', 'label_set', 'loss_policy', 'checkpoint_selection', 'ema', 'swa', 'affine', 'flip', 'device', 'subset')

def recipe_for(k_train, seed):
    r = {k: CFG[k] for k in RECIPE_KEYS}
    r.update(k_train=int(k_train), seed=int(seed), k_eval=[int(k) for k in CFG['k_eval']],
             data={k: EMB[k] for k in ('dread_f32_sha256', 'fold_sha256', 'canonical_sha256', 'corpus_manifest_sha256', 'weights_sha256')})
    return r

def recipe_hash(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

# ---- v2.0.1이 지원하는 레시피인지 확인 (지원하지 않는 값은 조용히 무시하지 않고 거부) ----
assert CFG['arch'] == ARCH and CFG['init'] == 'radimagenet' and CFG['norm'] == 'radimagenet' and CFG['input_policy'] == 'center3'
assert CFG['accum'] == 1 and CFG['drop_last'] and CFG['eval_bs'] == max(1, CFG['bs'] // 2) and CFG['head_dropout'] == 0.2
assert not (CFG['ema'] or CFG['swa'] or CFG['affine'] or CFG['flip']), 'v2.0.1은 EMA/SWA/affine/flip을 지원하지 않습니다'
assert CFG['loss_policy'] == 'clipped_positive_term' and CFG['checkpoint_selection'] == 'fixed-final' and CFG['label_set'] == 'dread'
assert CFG['k_eval'] and all(int(k) > 0 for k in CFG['k_eval']) and CFG['window_policy'] == 'within-slot-valid'
if CFG['device'] == 'cpu':
    assert CFG['amp'] == 'none' and CFG['eval_amp'] == 'none', 'CPU는 amp none'
else:
    assert CFG['amp'] == 'fp16' and CFG['eval_amp'] == 'fp16', 'T4는 FP16 (bf16 없음)'

# ---- 파일 도구: 임시 파일에 쓴 뒤 이름 바꾸기(원자적), sha256 확인 ----
def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def _fsync(f):
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass

def atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
        _fsync(f)
    os.replace(tmp, path)

def atomic_copy(src, dst):
    tmp = dst + '.tmp'
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def save_verified(obj, dst):
    """torch.save → 로컬 임시 파일 → Drive에 .tmp로 복사 → 이름 바꾸기 → Drive 파일을 다시 읽어 sha256 확인."""
    os.makedirs(CFG['local_tmp'], exist_ok=True)
    local = os.path.join(CFG['local_tmp'], os.path.basename(dst) + '.local')   # 대상과 같은 경로가 되지 않도록 접미사
    torch.save(obj, local)
    digest, nbytes = sha256_file(local), os.path.getsize(local)
    atomic_copy(local, dst)
    if sha256_file(dst) != digest or os.path.getsize(dst) != nbytes:
        raise RuntimeError(f'저장 확인 실패(sha256 불일치): {dst}')
    os.remove(local)
    return digest, nbytes

def copy_verified(local, dst):
    digest = sha256_file(local)
    atomic_copy(local, dst)
    if sha256_file(dst) != digest:
        raise RuntimeError(f'저장 확인 실패(sha256 불일치): {dst}')
    return digest, os.path.getsize(dst)

# ---- 학습 체크포인트 세대 관리: epoch_NNN.pt + epoch_NNN.json(sha256), 검증된 최신 keep개만 ----
def list_generations(ck_dir):
    out = []
    for f in os.listdir(ck_dir) if os.path.isdir(ck_dir) else []:
        m = re.fullmatch(r'epoch_(\d{3})\.json', f)
        if m:
            out.append((int(m.group(1)), os.path.join(ck_dir, f)))
    return sorted(out)

def save_generation(ck_dir, payload, completed, rh, log):
    name = f'epoch_{completed:03d}'
    pt, side = os.path.join(ck_dir, name + '.pt'), os.path.join(ck_dir, name + '.json')
    t0 = time.time()
    digest, nbytes = save_verified(payload, pt)
    atomic_write_json(side, dict(completed_epochs=completed, sha256=digest, bytes=nbytes, recipe_hash=rh,
                                 notebook_version=NOTEBOOK_VERSION, written_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    for c, s in list_generations(ck_dir)[:-max(1, int(CFG['keep_generations']))]:   # 오래된 세대 정리
        for p in (s[:-5] + '.pt', s):
            if os.path.exists(p):
                os.remove(p)
    for f in os.listdir(ck_dir):                             # 끊긴 저장이 남긴 임시 파일 정리
        if f.endswith('.tmp'):
            os.remove(os.path.join(ck_dir, f))
    log(f'  체크포인트 저장 epoch {completed} ({nbytes / 1e6:.0f} MB, {time.time() - t0:.1f}s, sha256 {digest[:12]} 확인)')

def load_latest_generation(ck_dir, rh, log):
    """최신 세대부터 sha256을 확인하며 읽습니다. 손상된 세대는 건너뛰고 이전 세대를 씁니다. 다른 레시피면 거부합니다."""
    for completed, side in reversed(list_generations(ck_dir)):
        with open(side, encoding='utf-8') as f:
            rec = json.load(f)
        if rec.get('recipe_hash') != rh:
            raise SystemExit(f'{side}: 다른 레시피(해시 {str(rec.get("recipe_hash"))[:12]})의 체크포인트입니다. drive_root를 바꾸거나 CFG를 되돌리세요.')
        pt = side[:-5] + '.pt'
        if not os.path.exists(pt) or os.path.getsize(pt) != rec['bytes']:
            log(f'  ⚠ 세대 epoch {completed}: 파일 없음/크기 다름 → 이전 세대 확인')
            continue
        with open(pt, 'rb') as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != rec['sha256']:
            log(f'  ⚠ 세대 epoch {completed}: sha256 불일치(손상) → 이전 세대 확인')
            continue
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        if payload.get('recipe_hash') != rh or payload.get('completed_epochs') != completed:
            raise SystemExit(f'{pt}: 체크포인트 내용이 기록과 다릅니다')
        return dict(path=pt, payload=payload)
    return None

# ---- 난수 상태 (python / numpy / torch CPU / CUDA) ----
def capture_rng(cuda):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if cuda else None)

def restore_rng(state, cuda):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if cuda:
        if state.get('torch_cuda') is None:
            raise SystemExit('CPU에서 만든 체크포인트는 CUDA에서 이어갈 수 없습니다')
        torch.cuda.set_rng_state_all(state['torch_cuda'])

# ---- 배치 순서: EXP107 DataLoader(shuffle=True, generator=Generator().manual_seed(seed), persistent workers)와 같은 순서 ----
class EpochSampler(torch.utils.data.Sampler):
    """torch DataLoader는 첫 iterator를 만들 때 generator에서 base seed를 한 번 뽑고(persistent workers라 이후 epoch에는 다시 뽑지 않음),
    RandomSampler는 epoch마다 randperm을 두 번 뽑습니다(두 번째는 num_samples % n = 0개를 내는 나머지 호출).
    같은 추출을 명시적으로 하므로 순서가 EXP107과 같고, generator 상태를 저장·복원해 이어하기가 정확합니다."""

    def __init__(self, n, seed):
        self.n = int(n)
        self.generator = torch.Generator().manual_seed(int(seed))
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)   # 첫 iterator의 base seed 추출
        self.epoch, self.order = None, None

    def set_epoch(self, epoch):
        self.order = torch.randperm(self.n, generator=self.generator).tolist()
        torch.randperm(self.n, generator=self.generator)     # RandomSampler의 나머지 호출 (0개) — 난수 흐름을 맞춤
        self.epoch = int(epoch)

    def __iter__(self):
        if self.order is None:
            raise RuntimeError('set_epoch()를 먼저 호출하세요')
        return iter(self.order)

    def __len__(self):
        return self.n

    def state_dict(self):
        return dict(generator=self.generator.get_state(), epoch=self.epoch, n=self.n)

    def load_state_dict(self, s):
        if s['n'] != self.n:
            raise SystemExit('샘플러 크기가 체크포인트와 다릅니다')
        self.generator.set_state(s['generator'])
        self.epoch, self.order = s['epoch'], None

# ---- 실행 상태(run_state.json): 레시피 해시, 환경 지문, 계산 시간 누적, 세션 기록, fold 상태 ----
def env_record():
    info = dict(python=sys.version.split()[0], torch=torch.__version__, timm=timm.__version__, numpy=np.__version__,
                platform=platform.platform(), device=CFG['device'])
    if CFG['device'] == 'cuda':
        p = torch.cuda.get_device_properties(0)
        info.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=p.name, gpu_memory_gib=round(p.total_memory / 2**30, 2))
    return info

def env_fingerprint():
    """수치 결과를 바꿀 수 있는 환경: python, torch, cuda/cudnn, timm, numpy, GPU 이름. 작업을 이어갈 때와 screen 4개 작업은 모두 같아야 합니다."""
    fp = dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__, numpy=np.__version__)
    if CFG['device'] == 'cuda':
        fp.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    else:
        fp.update(cuda=None, cudnn=None, gpu='cpu')
    return fp

def fp_diff(a, b):
    return {k: [(a or {}).get(k), (b or {}).get(k)] for k in sorted(set(a or {}) | set(b or {})) if (a or {}).get(k) != (b or {}).get(k)}

def job_paths(job):
    job_dir = os.path.join(plan_root(), job_name(*job))
    return job_dir, os.path.join(job_dir, 'run_state.json')

def plan_fp_path():
    return os.path.join(plan_root(), f"env_fingerprint_{CFG['mode']}.json")

def total_compute_s():
    """계산 한도는 '방식(mode)별 계획'마다 따로입니다: 지금 방식에서 계획한 작업들의 계산 시간 합 (Drive 기록 기준).
    screen → production으로 바꾸면 production 작업들의 새 9시간 한도가 시작됩니다 (노트북 전체 하나의 한도가 아님)."""
    tot = 0.0
    for job in planned_jobs():
        p = job_paths(job)[1]
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                tot += float(json.load(f)['compute_seconds'])
    return tot

def check_plan_env(fp):
    """방식별 계획의 환경 지문(첫 작업 시작 때 기록)과 비교. screen은 4개 작업이 같은 환경이어야 짝 비교가 유효하므로 다르면 거부,
    production은 fold마다 다를 수 있어 경고만 돌려줍니다(각 작업의 이어하기는 open_run에서 따로 강제)."""
    os.makedirs(plan_root(), exist_ok=True)                 # 1번 셀 뒤에 run_name·drive_root를 바꿔도 안전하게
    p = plan_fp_path()
    if not os.path.exists(p):
        atomic_write_json(p, dict(fingerprint=fp, mode=CFG['mode'], recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), session_id=SESSION_ID))
        return None
    with open(p, encoding='utf-8') as f:
        plan = json.load(f)['fingerprint']
    if plan == fp:
        return None
    d = fp_diff(plan, fp)
    if CFG['mode'] == 'screen':
        raise SystemExit(f"환경이 screen 계획의 첫 작업과 다릅니다 {d}. 같은 환경이 아니면 K16 vs K12 짝 비교가 무효이므로 이어가지 않습니다. "
                         f"0번 셀 CFG['run_name']을 새 이름으로 바꿔 screen 4개 작업을 처음부터 다시 시작하세요.")
    return f'⚠ production 계획의 첫 작업과 환경이 다릅니다 {d} (fold마다 환경이 섞임 — 평가 셀이 표시합니다)'

def open_run(job):
    mode, k_train, seed, fold = job
    run_dir, state_path = job_paths(job)
    recipe = recipe_for(k_train, seed)
    rh = recipe_hash(recipe)
    fp = env_fingerprint()
    if os.path.exists(state_path):
        with open(state_path, encoding='utf-8') as f:
            state = json.load(f)
        if state.get('recipe_hash') != rh:
            raise SystemExit(f'{run_dir}는 다른 레시피(해시 {state.get("recipe_hash", "")[:12]} ≠ {rh[:12]})로 시작된 작업 폴더입니다. '
                             "CFG['run_name']을 바꾸거나 CFG 위쪽(레시피) 값을 원래대로 되돌리세요.")
        if state.get('env_fingerprint') != fp:              # 다른 환경에서 옛 궤적을 조용히 이어가지 않음
            raise SystemExit(f"{job_name(*job)}: 이 작업을 시작한 환경과 지금 환경이 다릅니다 {fp_diff(state.get('env_fingerprint'), fp)} → 이어하기 거부. "
                             f"0번 셀 CFG['run_name']을 새 이름으로 바꿔 새로 시작하세요 (screen이면 4개 작업 모두).")
        warn = check_plan_env(fp)
    else:
        warn = check_plan_env(fp)                            # screen: 다른 환경이면 새 작업도 시작하지 않음
        os.makedirs(run_dir, exist_ok=True)
        state = dict(schema='knee-colab-v2-run', notebook_version=NOTEBOOK_VERSION, job=job_name(*job), mode=mode, k_train=int(k_train),
                     seed=int(seed), fold=int(fold), recipe=recipe, recipe_hash=rh, env_fingerprint=fp, warnings=[], compute_seconds=0.0,
                     lost_seconds=0.0, sessions=[], folds={})
    if warn:
        print(warn)
        if warn not in state.setdefault('warnings', []):
            state['warnings'].append(warn)
    hb_path = os.path.join(run_dir, 'heartbeat.json')                  # 지난 세션이 저장 없이 끊겼다면 그 계산 시간도 셉니다
    if os.path.exists(hb_path):
        with open(hb_path, encoding='utf-8') as f:
            hb = json.load(f)
        if hb.get('session_id') != SESSION_ID and not hb.get('accounted') and hb.get('pending_seconds', 0) > 0:
            state['lost_seconds'] += hb['pending_seconds']
            state['compute_seconds'] += hb['pending_seconds']
            hb['accounted'] = True
            atomic_write_json(hb_path, hb)
    if not any(s.get('session_id') == SESSION_ID for s in state['sessions']):
        state['sessions'].append(dict(session_id=SESSION_ID, started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), notebook_version=NOTEBOOK_VERSION,
                                      library_sha256=EMB['library_sha256'], environment=env_record()))
    atomic_write_json(state_path, state)
    return run_dir, state_path, state, recipe, rh

def tag_for(mode, k_train, seed, fold):
    return f'V2_RADR50_{mode.upper()}_K{int(k_train)}_S{int(seed)}_F{int(fold)}'

def fold_done(fold_dir):
    p = os.path.join(fold_dir, 'DONE.json')
    if not os.path.exists(p):
        return False
    with open(p, encoding='utf-8') as f:
        done = json.load(f)
    for name, rec in done['files'].items():
        q = os.path.join(fold_dir, name)
        if not os.path.exists(q) or os.path.getsize(q) != rec['bytes'] or sha256_file(q) != rec['sha256']:
            print(f'  ⚠ {q}: 완료 기록과 다름 → 이 fold를 다시 마무리합니다')
            return False
    return True

def epoch_estimate(history, k_train):
    secs = [h['epoch_seconds'] for h in history[-3:]]
    return 1.05 * max(secs) if secs else CFG['prior_epoch_s_per_k'] * k_train

def eval_estimate(n_val):
    return 0.005 * n_val * sum(int(k) for k in CFG['k_eval']) + 120   # EXP107 T4: 888 검사 K24 77초 + K36 61초의 약 2배 + 저장

def limit_reason(est_s, pending_s):
    used = total_compute_s() + pending_s
    wall = time.time() - SESSION_T0
    if used + est_s > CFG['compute_limit_h'] * 3600:
        return f"계산 한도 {CFG['compute_limit_h']} h: 누적 {used / 3600:.2f} h + 다음 작업 예상 {est_s / 60:.0f}분이 한도를 넘습니다"
    if wall + est_s > CFG['session_limit_h'] * 3600 - CFG['reserve_min'] * 60:
        return (f"세션 벽시계 한도 {CFG['session_limit_h']} h (여유 {CFG['reserve_min']}분): 경과 {wall / 3600:.2f} h + "
                f"다음 작업 예상 {est_s / 60:.0f}분")
    free = shutil.disk_usage(CFG['drive_root']).free / 1e9
    if free < CFG['drive_min_free_gb']:
        return f"Drive 남은 공간 {free:.1f} GB < {CFG['drive_min_free_gb']} GB (휴지통 정리는 본인이 직접 판단하세요)"
    return None

def stop_message(reason, tag, completed, ck_dir):
    print('\n' + '=' * 78)
    print(f'⏸ 여기서 멈춥니다: {reason}')
    print(f"   마지막 저장: {tag} epoch {completed}/{CFG['epochs']} ({ck_dir})")
    print('   이어하기: 런타임 → 런타임 연결 해제 및 삭제 → 다시 연결(T4) → 0번 셀부터 차례로 모두 실행.')
    print('            새 VM이면 2번 토큰 업로드와 3번 다운로드를 다시 하고, 학습은 Drive의 마지막 체크포인트에서 이어집니다.')
    print("   계산 한도 때문이면 0번 셀 CFG['compute_limit_h']를 올린 뒤 실행하세요 (레시피 해시는 바뀌지 않습니다).")
    print('=' * 78)

def autocast_ctx(dtype, dev):
    return torch.autocast(dev, dtype=dtype) if dtype is not None and dev == 'cuda' else contextlib.nullcontext()

@torch.no_grad()
def predict(model, ids, k_eval, dev, eval_dtype):
    """EXP107 evaluate(): 고정 창(균등 간격) K개, batch eval_bs, FP16 autocast(T4), sigmoid. 검사 순서를 그대로 돌려줍니다."""
    zero = {u: np.zeros(len(LAB), np.float32) for u in ids}
    dl = DataLoader(CenterSliceWindows(CORPUS_DIR, list(ids), zero, CFG['res'], int(k_eval), False, CFG['norm'], CFG['seed'],
                                       jitter=CFG['jitter'], window_policy=CFG['window_policy'], input_policy=CFG['input_policy']),
                    batch_size=CFG['eval_bs'], shuffle=False, num_workers=min(int(CFG['num_workers']), 2), collate_fn=collate,
                    persistent_workers=False, worker_init_fn=worker_init)
    model.eval()
    P, U = [], []
    for x, y, uids in dl:
        with autocast_ctx(eval_dtype, dev):
            logits = model(x.to(dev))
        if logits.shape != (len(uids), len(LAB)) or not torch.isfinite(logits).all():
            raise RuntimeError(f'invalid evaluation logits: {uids}')
        P.append(torch.sigmoid(logits.float()).cpu().numpy())
        U += uids
    return np.concatenate(P), U

def stop_loader(tl):
    it = getattr(tl, '_iterator', None)
    if it is not None and hasattr(it, '_shutdown_workers'):
        it._shutdown_workers()
    tl._iterator = None

def train_fold(job):
    mode, k_train, seed, fold = job
    tag = tag_for(*job)
    if fold_done(job_paths(job)[0]):                         # 끝난 작업은 환경과 상관없이 건너뜀 (screen 유효성은 7번 셀이 확인)
        print(f'{tag}: 이미 완료 → 건너뜀')
        return 'done'
    run_dir, state_path, state, recipe, rh = open_run(job)  # 레시피·환경 지문 확인 (다르면 거부)
    fold_dir = run_dir                                       # 작업 폴더 = 결과 폴더 (작업마다 따로)
    ck_dir = os.path.join(fold_dir, 'train_ckpt')
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(CFG['local_tmp'], exist_ok=True)
    log_path = os.path.join(fold_dir, 'train.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    acc = dict(t=time.time())                                # 계산 시간 회계: 마지막으로 기록한 시각

    def account(fold_status):
        now = time.time()
        state['compute_seconds'] += now - acc['t']
        acc['t'] = now
        state['folds'][str(fold)] = dict(fold_status, tag=tag, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        atomic_write_json(state_path, state)
        atomic_write_json(os.path.join(run_dir, 'heartbeat.json'), dict(session_id=SESSION_ID, pending_seconds=0.0, accounted=True))

    dev = CFG['device']
    cuda = dev == 'cuda'
    f_emb = EMB['folds'][str(fold)]
    train_ids, val_ids = list(FOLD_IDS[fold]['train']), list(FOLD_IDS[fold]['val'])
    pw_np = pos_weight(DREAD_OF, train_ids)                  # EXP099 규칙: 학습 fold의 Dread 평균 → clip
    if not np.array_equal(pw_np, np.asarray(f_emb['pos_weight'], np.float32)):
        raise SystemExit('pos_weight가 EXP107 등록값과 다릅니다')
    if len(train_ids) // CFG['bs'] != f_emb['len_train_loader'] or (CFG['epochs'] == 20 and len(train_ids) // CFG['bs'] * 20 != f_emb['steps']):
        raise SystemExit('fold 계약(배치 수·업데이트 수)이 EXP107과 다릅니다')
    if CFG['epochs'] != 20:
        print(f"⚠ epochs={CFG['epochs']} (EXP107은 20): 레시피 해시가 달라지는 별도 실행입니다")
    if CFG['subset']:                                        # (로컬 테스트 전용) fold의 앞쪽 일부 검사만
        train_ids, val_ids = train_ids[:CFG['subset']['train']], val_ids[:CFG['subset']['val']]
    epochs = int(CFG['epochs'])
    len_tl = len(train_ids) // CFG['bs']
    steps = max(len_tl * epochs, 1)
    max_skips = steps // 100

    # ---- 런타임 → 작업 스레드 → seed → 모델 (EXP107 runner와 같은 순서) ----
    runtime = configure_runtime(CFG['amp'], dev, policy=CFG['runtime_policy'])
    eval_runtime = configure_runtime(CFG['eval_amp'], dev, policy=CFG['runtime_policy'])
    worker_init(0)
    torch.set_num_threads(int(CFG['main_threads']))
    seed_all(seed)
    built, load_report = build_radimagenet_model(CFG['arch'], WEIGHTS_PATH, RADIMAGENET_R50_SHA256)
    init_sha, _ = state_hashes(built)
    model = built.to(dev)
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, CFG['res'], CFG['res'], device=dev))
    model.train()
    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    bb = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': bb, 'lr': CFG['bb_lr']}, {'params': head, 'lr': CFG['head_lr']}], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[CFG['bb_lr'], CFG['head_lr']], total_steps=steps, pct_start=CFG['pct_start'])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw_np, dtype=torch.float32, device=dev))
    amp_dtype = {'fp16': torch.float16, 'none': None}[CFG['amp']]
    eval_dtype = {'fp16': torch.float16, 'none': None}[CFG['eval_amp']]
    scaler = torch.amp.GradScaler('cuda', init_scale=CFG['amp_init_scale'], enabled=CFG['amp'] == 'fp16' and cuda)

    # ---- 데이터: EXP107 CenterSliceWindows(K=k_train, 학습 모드) + 명시적 샘플러 ----
    all_labels = {**{u: DREAD_OF[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in val_ids}}
    tds = CenterSliceWindows(CORPUS_DIR, train_ids, all_labels, CFG['res'], int(k_train), True, CFG['norm'], seed,
                             jitter=CFG['jitter'], window_policy=CFG['window_policy'], input_policy=CFG['input_policy'])
    sampler = EpochSampler(len(train_ids), seed)
    W = int(CFG['num_workers'])
    tl = DataLoader(tds, batch_size=CFG['bs'], sampler=sampler, drop_last=True, num_workers=W, collate_fn=collate, pin_memory=cuda,
                    persistent_workers=W > 0, prefetch_factor=int(CFG['prefetch_factor']) if W > 0 else None,
                    worker_init_fn=worker_init, generator=torch.Generator().manual_seed(seed))
    if len(tl) != len_tl:
        raise RuntimeError('DataLoader 길이가 계약과 다릅니다')
    trace_corpus = Corpus(CORPUS_DIR)                        # 기록용: 각 배치의 창 중심을 다시 계산

    provenance = dict(
        code_sha256=EMB['code_sha256'], radimagenet_module_sha256=EMB['radimagenet_module_sha256'],
        code_origin=f'notebook {NOTEBOOK_VERSION} cell 5 = verbatim copy of these EXP107 source files (sha256 of the source files)',
        library_sha256=EMB['library_sha256'], notebook_version=NOTEBOOK_VERSION, recipe=recipe, recipe_hash=rh,
        initial_state_sha256=init_sha, initial_state_matches_exp107_registered=init_sha == EMB['exp107']['initial_state_sha256'],
        runtime=runtime, eval_runtime=eval_runtime, train_ids_ordered_sha256=uid_hash(train_ids), eval_ids_ordered_sha256=uid_hash(val_ids),
        label_set='dread', label_f32_sha256=EMB['dread_f32_sha256'], fold_sha256=EMB['fold_sha256'],
        corpus_manifest_sha256=EMB['corpus_manifest_sha256'], sampling='sha256(seed,epoch,uid,stream)-v1 (EXP107)',
        batch_order='EpochSampler = EXP107 DataLoader(shuffle=True, Generator(seed), persistent workers)', checkpoint_selection='fixed-final',
        gold_access='disabled', loss_policy='clipped_positive_term', actual_pos_weight=pw_np.tolist(), fold=fold, seed=seed, mode=mode, job=job_name(*job),
        arch=CFG['arch'], k_train=int(k_train), k_eval=[int(k) for k in CFG['k_eval']], len_train_loader=len_tl, scheduler_total_steps=steps,
        max_fp16_skips=max_skips, init_weights=load_report, input_policy=CFG['input_policy'], norm=CFG['norm'], subset=CFG['subset'])

    # ---- 이어하기: 마지막 검증된 세대에서 모든 상태 복원 ----
    history, trace, attempted, skipped, start_epoch, fold_compute = [], [], 0, 0, 0, 0.0
    resumed = load_latest_generation(ck_dir, rh, log)
    if resumed is not None:
        p = resumed['payload']
        model.load_state_dict(p['model'])
        opt.load_state_dict(p['optimizer'])
        sched.load_state_dict(p['scheduler'])
        scaler.load_state_dict(p['scaler'])
        sampler.load_state_dict(p['sampler'])
        history, trace = list(p['history']), list(p['trace'])
        attempted, skipped, start_epoch = int(p['attempted_updates']), int(p['skipped']), int(p['completed_epochs'])
        fold_compute = float(p['compute_seconds_fold'])
        if p.get('env_fingerprint') not in (None, state['env_fingerprint']):
            raise SystemExit(f"체크포인트의 환경 지문이 작업 기록과 다릅니다 {fp_diff(p.get('env_fingerprint'), state['env_fingerprint'])} → 이어하기 거부")
        if p['initial_state_sha256'] != init_sha:
            raise SystemExit('같은 환경인데 초기 모델 해시가 처음 세션과 다릅니다 → 이어하기 거부 (가중치 파일·코드 확인)')
        restore_rng(p['rng'], cuda)                          # 마지막: 모델 생성 등이 쓴 난수를 덮어씀
        log(f"↻ {tag}: 이어하기 epoch {start_epoch}/{epochs} ({os.path.basename(resumed['path'])}) | 업데이트 {attempted} (건너뜀 {skipped})")
    else:
        log(f'▶ {tag}: 새로 시작 | {CFG["arch"]} RadImageNet | center3 2x-1 | K{k_train} 학습 / K{CFG["k_eval"]} 평가 | fold {fold} seed {seed} ({mode}) | '
            f'train {len(train_ids)} | val {len(val_ids)} | steps {steps} | 초기 해시 {init_sha[:12]}')
        if not provenance['initial_state_matches_exp107_registered']:
            w = (f"⚠ 초기 모델 해시 {init_sha[:12]} ≠ EXP107 등록값 {EMB['exp107']['initial_state_sha256'][:12]} "
                 f"(등록 환경 torch {EMB['exp107']['initial_state_env']['torch']} / timm {EMB['exp107']['initial_state_env']['timm']}, 지금 "
                 f"torch {torch.__version__} / timm {timm.__version__}). 헤드 초기값이 EXP107과 다를 수 있습니다 — 같은 노트북 안의 짝 비교는 영향 없음")
            log(w)
            if w not in state.setdefault('warnings', []):
                state['warnings'].append(w)
                atomic_write_json(state_path, state)
    log(f'  세션 {SESSION_ID} | {json.dumps(env_record(), ensure_ascii=False)}')

    def payload(completed):
        return dict(schema='knee-colab-v2-train-ckpt', notebook_version=NOTEBOOK_VERSION, recipe_hash=rh, job=job_name(*job), mode=mode, seed=seed, tag=tag,
                    fold=fold, k_train=int(k_train), model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(), rng=capture_rng(cuda),
                    sampler=sampler.state_dict(), dataset_epoch=completed, completed_epochs=completed, attempted_updates=attempted,
                    successful_updates=attempted - skipped, skipped=skipped, history=history, trace=trace, compute_seconds_fold=fold_compute,
                    initial_state_sha256=init_sha, env_fingerprint=state['env_fingerprint'], train_ids_sha256=uid_hash(train_ids),
                    val_ids_sha256=uid_hash(val_ids))

    def finish(status):
        stop_loader(tl)
        return status

    measure = True                                           # 이 세션의 첫 epoch: 데이터 대기 vs 계산 시간 측정
    hb_t = time.time()
    for ep in range(start_epoch, epochs):
        est = epoch_estimate(history, k_train) + (eval_estimate(len(val_ids)) if ep == epochs - 1 else 0)
        reason = limit_reason(est, time.time() - acc['t'])
        if reason:
            account(dict(status='training', completed_epochs=ep))
            stop_message(reason, tag, ep, ck_dir)
            return finish('stopped')
        t_ep = time.time()
        sampler.set_epoch(ep)
        tds.set_epoch(ep)
        model.train()
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        wait = comp = 0.0
        t_prev = time.time()
        for i, (x, y, batch_uids) in enumerate(tl):
            t_got = time.time()
            wait += t_got - t_prev
            if CFG['crash_at'] and [ep, i] == list(CFG['crash_at']):
                raise RuntimeError(f'(테스트) 강제 오류 epoch {ep} step {i}')   # VM이 갑자기 끊긴 상황 흉내
            rec = None
            if i < int(CFG['trace_batches']):
                rec = dict(epoch=ep, step=i, uids=list(batch_uids), lr=[float(g['lr']) for g in opt.param_groups],
                           centres=[[int(c) for c in select_centers(trace_corpus.mask(u), trace_corpus.depth, int(k_train), trace_corpus.spec,
                                                                    CFG['window_policy'], study_rng(seed, ep, u))] for u in batch_uids])
                if CFG['trace_x_hash']:
                    rec['x_sha256'] = hashlib.sha256(x.numpy().tobytes()).hexdigest()
            # ---- 아래 한 step은 EXP107 runner의 학습 루프 그대로 ----
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with autocast_ctx(amp_dtype, dev):
                logits = model(x)
                if logits.shape != (len(batch_uids), 12) or not torch.isfinite(logits).all():
                    raise RuntimeError('nonfinite or invalid training logits')
                loss = lossf(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite training loss at epoch {ep} step {i}')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'], error_if_nonfinite=CFG['amp'] != 'fp16')
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if CFG['amp'] == 'fp16' and cuda and scaler.get_scale() < scale_before:   # FP16 overflow로 건너뛴 step: 스케줄러를 진행하지 않음
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError(f'FP16 optimizer skips exceed 1% of {steps} ({skipped})')
            else:
                sched.step()
            attempted += 1
            lv = loss.item()
            tot += lv
            if rec is not None:
                rec['loss'] = lv
                trace.append(rec)
            if measure and cuda:
                torch.cuda.synchronize()
            t_prev = time.time()
            comp += t_prev - t_got
            if measure and i + 1 == min(50, len(tl)):
                log(f'  처음 {i + 1} step: 데이터 대기 {wait:.1f}s · 계산 {comp:.1f}s → 이 속도면 epoch 약 {(wait + comp) / (i + 1) * len(tl) / 60:.1f}분')
            if (i + 1) % int(CFG['print_every']) == 0:
                rate = (time.time() - t_ep) / (i + 1)
                log(f'  ep{ep} step {i + 1}/{len(tl)} loss {tot / (i + 1):.4f} | {rate:.2f}s/step, epoch ETA {rate * (len(tl) - i - 1) / 60:.0f} min')
            if time.time() - hb_t > CFG['heartbeat_s']:
                atomic_write_json(os.path.join(run_dir, 'heartbeat.json'),
                                  dict(session_id=SESSION_ID, fold=fold, epoch=ep, step=i, pending_seconds=time.time() - acc['t'], accounted=False))
                hb_t = time.time()
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0
        completed = ep + 1
        ep_seconds = time.time() - t_ep
        fold_compute += ep_seconds
        history.append(dict(epoch=ep, loss=tot / max(1, len(tl)), epoch_seconds=round(ep_seconds), peak_gpu_gib=round(peak, 2),
                            optimizer_updates=attempted, fp16_skips=skipped, load_wait_seconds=round(wait, 1), compute_seconds=round(comp, 1),
                            session_id=SESSION_ID))
        log(f"ep{ep} loss {history[-1]['loss']:.4f} | fixed-final | {round(ep_seconds)}s | peak {peak:.1f} GiB | skips {skipped} | "
            f"lr {opt.param_groups[0]['lr']:.2e}/{opt.param_groups[1]['lr']:.2e}")
        if measure:
            log(f'  [측정] 이 세션 첫 epoch: 데이터 대기 {wait:.0f}s / 계산 {comp:.0f}s '
                f"({'데이터 읽기가 병목 → num_workers·디스크 확인' if wait > 0.25 * comp else '계산이 병목: 정상'})")
            measure = False
        if completed % int(CFG['ckpt_every_epochs']) == 0 or completed == epochs:
            save_generation(ck_dir, payload(completed), completed, rh, log)
            account(dict(status='training', completed_epochs=completed))
        if CFG['stop_after_epochs'] and completed - start_epoch >= int(CFG['stop_after_epochs']) and completed < epochs:
            log(f'(테스트) epoch {completed} 뒤 강제 중단')
            return finish('interrupted')

    # ---- 마무리: 정확한 업데이트 수 확인 → 릴리스 형식 체크포인트 → K36·K24 OOF → 기록 → 완료 표시 ----
    if attempted != steps:
        raise RuntimeError('full exact update gate failed before inference')
    est = eval_estimate(len(val_ids))
    reason = limit_reason(est, time.time() - acc['t'])
    if reason:                                               # 마지막 학습 체크포인트는 이미 저장됨 → 내보내기·평가는 다음 세션에서
        account(dict(status='training', completed_epochs=epochs))
        stop_message(reason, tag, epochs, ck_dir)
        return finish('stopped')
    stop_loader(tl)
    provenance['environment'] = env_record()
    provenance['fp16_skipped_updates'] = skipped
    provenance['env_fingerprint'] = state['env_fingerprint']
    provenance['warnings'] = state.get('warnings', [])
    provenance['sessions'] = [dict(session_id=s['session_id'], started_at=s['started_at'], environment=s.get('environment')) for s in state['sessions']]
    best = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'arch': CFG['arch'], 'res': CFG['res'], 'lab': LAB,
            'contract': CORPUS_CONTRACT, 'k_eval': int(CFG['k_eval'][0]), 'norm': CFG['norm'], 'window_policy': CFG['window_policy'],
            'epoch': epochs - 1, 'tag': tag, 'amp_train': CFG['amp'], 'amp_eval': CFG['eval_amp'], 'fold': fold, 'label_set': 'dread',
            'provenance': provenance}
    best.update(mode=mode, seed=seed, input_policy=CFG['input_policy'], init='radimagenet', init_weights_sha256=RADIMAGENET_R50_SHA256, k_train=int(k_train),
                k_eval_all=[int(k) for k in CFG['k_eval']], notebook_version=NOTEBOOK_VERSION, recipe_hash=rh)
    files = {}
    files['model_best.pt'] = save_verified(best, os.path.join(fold_dir, 'model_best.pt'))
    t_eval = time.time()
    for k_eval in CFG['k_eval']:
        t0 = time.time()
        P, U = predict(model, val_ids, k_eval, dev, eval_dtype)
        if U != val_ids:
            raise RuntimeError(f'K{k_eval} prediction UID order changed')
        local = Path(CFG['local_tmp']) / f'{tag}_oof_k{k_eval}.csv'
        write_predictions(local, U, P)
        files[f'oof_k{k_eval}.csv'] = copy_verified(str(local), os.path.join(fold_dir, f'oof_k{k_eval}.csv'))
        os.remove(local)
        log(f'K{k_eval} OOF 저장 ({len(U)} 검사, {time.time() - t0:.0f}s)')
    if dev == 'cpu' and torch.cuda.is_initialized():
        raise RuntimeError('CPU run initialised CUDA')
    provenance['cuda_initialized'] = bool(torch.cuda.is_initialized())
    summary = dict(tag=tag, job=job_name(*job), mode=mode, seed=seed, notebook_version=NOTEBOOK_VERSION, recipe_hash=rh, contract=CORPUS_CONTRACT, backbone=CFG['arch'],
                   img=CFG['res'], k_train=int(k_train), k_eval=[int(k) for k in CFG['k_eval']], max_epochs=epochs, train=len(train_ids),
                   val=len(val_ids), rows=len(val_ids), selected_checkpoint='fixed-final', selected_epoch_1based=epochs, optimizer_updates=attempted,
                   successful_updates=attempted - skipped, skipped_updates=skipped, history=history, fold_compute_seconds=round(fold_compute),
                   eval_seconds=round(time.time() - t_eval), oof_files={f'k{k}': f'oof_k{k}.csv' for k in CFG['k_eval']},
                   gold_access='disabled', metric_access='deferred_single_fixed_evaluation (7번 셀)',
                   caveat='weak Dread labels; one seed; Colab T4 trajectory, independent of the Kaggle EXP107 runs (D-B-008)')
    for name, obj in (('history.json', history), ('provenance.json', provenance), ('summary.json', summary)):
        local = os.path.join(CFG['local_tmp'], f'{tag}_{name}')
        with open(local, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
        files[name] = copy_verified(local, os.path.join(fold_dir, name))
        os.remove(local)
    local = os.path.join(CFG['local_tmp'], f'{tag}_trace.jsonl')
    with open(local, 'w', encoding='utf-8') as f:
        for r in trace:
            f.write(json.dumps(r) + '\n')
    files['trace.jsonl'] = copy_verified(local, os.path.join(fold_dir, 'trace.jsonl'))
    os.remove(local)
    inventory = {n: dict(sha256=d, bytes=b) for n, (d, b) in files.items()}
    atomic_write_json(os.path.join(fold_dir, 'output-inventory.json'), inventory)  # 릴리스 v6 inventory_check 형식
    atomic_write_json(os.path.join(fold_dir, 'DONE.json'), dict(tag=tag, recipe_hash=rh, notebook_version=NOTEBOOK_VERSION, files=inventory,
                                                               env_fingerprint=state['env_fingerprint'], warnings=state.get('warnings', []),
                                                               finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    if CFG['delete_train_ckpt_after_fold']:
        shutil.rmtree(ck_dir, ignore_errors=True)           # 완료 표시 뒤에만 지웁니다 (Drive 휴지통으로 감)
    account(dict(status='done', completed_epochs=epochs))
    log(f"✅ {tag} 완료: 업데이트 {attempted} (건너뜀 {skipped}), fold 계산 {fold_compute / 60:.0f}분 → {fold_dir}")
    del model, opt, sched, scaler, tl, tds, built
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return 'done'

def run_all():
    for job in planned_jobs():                               # screen: K12 s42 → K16 s42 → K12 s43 → K16 s43
        status = train_fold(job)
        if status in ('stopped', 'interrupted'):
            return status
    print('✅ 계획한 학습이 모두 끝났습니다. 7번 셀에서 평가합니다.')
    return 'complete'

RUN_STATUS = run_all() if CFG['train'] else 'skipped'
print('학습 상태:', RUN_STATUS)
'''

# ---------------------------------------------------------------------------------------------------------------- cell 7
CELL_7 = r'''# ===== [7] 평가: 같은 Colab 플랫폼의 짝 비교(component bootstrap) · 잠정 관문 · notebook v2.0.1 =====
# screen     : fold 0에서 K16 vs K12 (seed 평균, 같은 플랫폼). 저장된 EXP107은 "cross-platform reference, not a control" 한 줄로만 보여 줍니다.
# production : K16 5-fold의 fold별·전체·소견별 AUC, fold 0 K12 대조군과 짝 비교, EXP107은 cross-platform 참고값.
from scipy.stats import rankdata

def auc_rank(y, s):
    """tools/exp099_kfold.auc와 같은 average-rank Mann-Whitney AUC (rankdata 'average' = avg_rank). 양성·음성 한쪽이 없으면 nan."""
    y = np.asarray(y).astype(bool)
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    r = rankdata(s, method='average')
    return (r[y].sum() - P * (P + 1) / 2) / (P * N)

def macro_score(Y, S, M=None):
    per = [auc_rank(Y[:, t] if M is None else Y[M[:, t], t], S[:, t] if M is None else S[M[:, t], t]) for t in range(len(LAB))]
    ok = [v for v in per if np.isfinite(v)]
    return (float(np.mean(ok)) if ok else float('nan')), per, len(ok)

def references(rows):
    """정답지: Dread(float32 → rint), flight, relabel, consensus(세 라벨이 같은 칸, 값은 Dread). 검증 실패한 정답지는 뺍니다."""
    D = np.rint(DREAD_SORTED[rows]).astype(int)
    refs = {'Dread': (D, None)}
    for p in ('flight', 'relabel'):
        if REF_OK[p]:
            refs[p] = (REF[p][rows].astype(int), None)
    if REF_OK['flight'] and REF_OK['relabel']:
        F, R = REF['flight'][rows].astype(int), REF['relabel'][rows].astype(int)
        refs['consensus'] = (D, (D == F) & (F == R))
    return refs

JOB_ENV = {}                                                 # 작업 이름 → 완료 기록의 환경 지문

def load_job_preds(mode, k_train, seed, fold):
    """완료된 작업의 OOF {K: (uids, P)} — DONE.json의 sha256과 레시피 해시를 확인합니다. 미완료면 None."""
    jd = os.path.join(plan_root(), job_name(mode, k_train, seed, fold))
    p = os.path.join(jd, 'DONE.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        done = json.load(f)
    if done.get('recipe_hash') != recipe_hash(recipe_for(k_train, seed)):
        raise SystemExit(f'{jd}: 완료 기록의 레시피 해시가 지금 CFG와 다릅니다')
    JOB_ENV[job_name(mode, k_train, seed, fold)] = done.get('env_fingerprint')
    expected = FOLD_IDS[fold]['val'] if not CFG['subset'] else FOLD_IDS[fold]['val'][:CFG['subset']['val']]
    out = {}
    for k in CFG['k_eval']:
        name = f'oof_k{int(k)}.csv'
        q = os.path.join(jd, name)
        if sha256_file(q) != done['files'][name]['sha256']:
            raise SystemExit(f'{q}: 완료 기록과 sha256이 다릅니다')
        df = pd.read_csv(q, dtype={UID: str})
        if list(df.columns) != [UID] + LAB or df[UID].tolist() != expected:
            raise SystemExit(f'{q}: 열 또는 검사 목록·순서가 fold {fold} 검증 목록과 다릅니다')
        out[int(k)] = (df[UID].tolist(), df[LAB].to_numpy(dtype=np.float64))
    return out

def to_matrix(preds_by_fold, k):
    """{fold: {K: (uids, P)}} → 정렬된 UID 순서의 [4343, 12] 행렬(없는 칸 nan)과 채워진 행 번호."""
    S = np.full((len(UIDS_SORTED), len(LAB)), np.nan)
    rows = []
    for fold in sorted(preds_by_fold):
        uids, P = preds_by_fold[fold][k]
        idx = [ROW_OF[u] for u in uids]
        S[idx] = P
        rows += idx
    return S, np.asarray(sorted(rows), dtype=int)

def component_draws(rows, n_boot, seed):
    """component 단위 복원추출: 같은 component의 검사는 함께 뽑힙니다. 모든 비교가 같은 표본을 씁니다(짝 bootstrap)."""
    comps = COMPONENT_IDX[rows]
    order = np.argsort(comps, kind='stable')
    uniq, start = np.unique(comps[order], return_index=True)
    groups = np.split(order, start[1:])
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]

def paired(A, B, rows, refs, draws):
    """Δ = mean(A의 macro) − mean(B의 macro). A, B는 행렬 목록(seed 평균). 같은 bootstrap 표본으로 95% 구간."""
    A, B = [S[rows] for S in A], [S[rows] for S in B]

    def score(Ms, Y, M, d=None):
        if d is None:
            return float(np.mean([macro_score(Y, S, M)[0] for S in Ms]))
        return float(np.mean([macro_score(Y[d], S[d], None if M is None else M[d])[0] for S in Ms]))
    out = {}
    for name, (Y, M) in refs.items():
        a, b = score(A, Y, M), score(B, Y, M)
        bd = [score(A, Y, M, d) - score(B, Y, M, d) for d in draws]
        lo, hi = np.nanpercentile(bd, [2.5, 97.5]) if draws else (np.nan, np.nan)
        out[name] = dict(a=a, b=b, delta=a - b, lo=float(lo), hi=float(hi))
    return out

def fmt(v, sign=False):
    return '   nan ' if v is None or not np.isfinite(v) else (f'{v:+.4f}' if sign else f'{v:.4f}')

def gate(say, d, status, label):
    """잠정 진급 관문 (사전 등록 값): ΔDread ≥ +0.004 & 구간 > 0, Δconsensus ≥ 0, Δflight·Δrelabel ≥ −0.002.
    status: 'complete'일 때만 PASS / NOT PASSED. 'incomplete' → INCOMPLETE — diagnostic only, 'invalid' → INVALID (환경 불일치). 둘 다 판정·권고 없음."""
    if not all(n in d for n in ('Dread', 'consensus', 'flight', 'relabel')):
        say(f'\n잠정 관문 [{label}]: 판정 불가 (flight/relabel 정답지 검증 실패)')
        return dict(verdict='NO VERDICT — references missing', status=status, conditions=None)
    cond = {'ΔDread ≥ +0.004 & 구간 > 0': bool(d['Dread']['delta'] >= 0.004 and d['Dread']['lo'] > 0),
            'Δconsensus ≥ 0': bool(d['consensus']['delta'] >= 0), 'Δflight ≥ −0.002': bool(d['flight']['delta'] >= -0.002),
            'Δrelabel ≥ −0.002': bool(d['relabel']['delta'] >= -0.002)}
    if status == 'invalid':
        verdict = 'INVALID — 환경 불일치 (짝 비교 무효)'
    elif status != 'complete':
        verdict = 'INCOMPLETE — diagnostic only'
    else:
        verdict = 'PASS (잠정)' if all(cond.values()) else 'NOT PASSED'
    say(f'\n잠정 관문 [{label}]: {verdict} — 조건 ' + ', '.join(f"{c} {'✔' if ok else '✘'}" for c, ok in cond.items()) +
        ('' if status == 'complete' else '  ※ 판정하지 않습니다 (참고용 숫자)'))
    return dict(verdict=verdict, status=status, conditions=cond)

def delta_line(d):
    return ' | '.join(f"{n} {fmt(v['delta'], True)} [{fmt(v['lo'], True)}, {fmt(v['hi'], True)}]" for n, v in d.items())

def evaluate_screen(say, report):
    kc, kt, f = int(CFG['control_k_train']), int(CFG['k_train']), int(CFG['screen_fold'])
    seeds = [int(s) for s in CFG['screen_seeds']]
    P = {(k, s): load_job_preds('screen', k, s, f) for s in seeds for k in (kc, kt)}
    say('작업: ' + ' | '.join(f"K{k} s{s} {'완료' if P[(k, s)] else '미완료'}" for s in seeds for k in (kc, kt)))
    pairs = [s for s in seeds if P[(kc, s)] and P[(kt, s)]]
    report['pairs'] = pairs
    if not pairs:
        say(f'같은 seed의 K{kc}·K{kt} 짝이 아직 없습니다 → 6번 셀을 이어서 실행하세요.')
        return
    for k in [int(x) for x in CFG['k_eval']]:
        mats = {key: to_matrix({f: v}, k) for key, v in P.items() if v}
        rows = mats[(kc, pairs[0])][1]
        if any(not np.array_equal(r, rows) for _, r in mats.values()):
            raise SystemExit('작업들의 검증 검사 목록이 다릅니다')
        refs = references(rows)
        keys = sorted(mats)
        say(f"\n[K{k}] fold {f} macro AUC (n={len(rows)}, {len(np.unique(COMPONENT_IDX[rows]))} components) " +
            ('주 지표' if k == int(CFG['k_eval'][0]) else '진단용'))
        say('| 정답지 | ' + ' | '.join(f'K{a} s{b}' for a, b in keys) + f' | K{kc} 평균 | K{kt} 평균 | Δ | K{kc} seed 흔들림 | K{kt} seed 흔들림 |')
        spread = {}
        for name, (Y, M) in refs.items():
            v = {key: macro_score(Y, mats[key][0][rows], M)[0] for key in keys}
            mc = float(np.mean([v[(kc, s)] for s in pairs]))
            mt = float(np.mean([v[(kt, s)] for s in pairs]))
            sc = [v[(kc, s)] for s in seeds if (kc, s) in v]
            st = [v[(kt, s)] for s in seeds if (kt, s) in v]
            spread[name] = dict(control=(max(sc) - min(sc)) if len(sc) > 1 else None, variant=(max(st) - min(st)) if len(st) > 1 else None)
            say(f'| {name} | ' + ' | '.join(fmt(v[key]) for key in keys) + f' | {fmt(mc)} | {fmt(mt)} | {fmt(mt - mc, True)} | '
                f"{fmt(spread[name]['control'])} | {fmt(spread[name]['variant'])} |")
        t0 = time.time()
        d = paired([mats[(kt, s)][0] for s in pairs], [mats[(kc, s)][0] for s in pairs], rows, refs,
                   component_draws(rows, int(CFG['n_boot']), int(CFG['boot_seed'])))
        report['deltas'][f'K{k}'] = d
        report['seed_spread'][f'K{k}'] = spread
        say(f"Δ K{kt} − K{kc} (같은 Colab 플랫폼, seed {pairs} 평균, component bootstrap {CFG['n_boot']}회, {time.time() - t0:.0f}s): " + delta_line(d))
        if k == int(CFG['k_eval'][0]):
            Y = refs['Dread'][0]
            pt = np.mean([macro_score(Y, mats[(kt, s)][0][rows])[1] for s in pairs], axis=0)
            pc = np.mean([macro_score(Y, mats[(kc, s)][0][rows])[1] for s in pairs], axis=0)
            say('소견별 Dread (K16 평균, Δ): ' + ' | '.join(f'{t} {fmt(a)} ({fmt(a - b, True)})' for t, a, b in zip(LAB, pt, pc)))
        x = paired([mats[(kc, s)][0] for s in pairs], [EXP107_OOF[k]], rows, refs, [])
        say(f'cross-platform reference, not a control — 저장된 EXP107 (Kaggle T4, K12 s42) fold {f}: ' +
            ' | '.join(f"{n} {fmt(v['b'])}" for n, v in x.items()) + f' · Colab K{kc} 평균 − EXP107: ' +
            ' | '.join(f"{n} {fmt(v['delta'], True)}" for n, v in x.items()))
    k0 = int(CFG['k_eval'][0])
    # ---- 판정 조건: 서로 다른 seed 2개 이상의 K12·K16 짝이 모두 완료 + 전체 검증 명단 + 모든 작업이 같은 환경
    full_roster = not CFG['subset'] and len(rows) == len(FOLD_IDS[f]['val'])
    complete = len(set(seeds)) >= 2 and len(pairs) == len(seeds) and full_roster
    plan_fp = None
    if os.path.exists(plan_fp_path()):
        with open(plan_fp_path(), encoding='utf-8') as fh:
            plan_fp = json.load(fh)['fingerprint']
    envs = {job_name('screen', k, s, f): JOB_ENV.get(job_name('screen', k, s, f)) for (k, s), v in P.items() if v}
    bad_env = sorted(n for n, e in envs.items() if e is None or e != plan_fp)
    report['environment'] = dict(plan=plan_fp, jobs=envs, mismatched=bad_env)
    if bad_env:
        say(f'⚠ 환경 지문이 screen 계획과 다른 작업: {bad_env} → 짝 비교 무효')
    status = 'invalid' if bad_env else ('complete' if complete else 'incomplete')
    report['gate'] = gate(say, report['deltas'][f'K{k0}'], status, f'K{k0}, K{kt} vs K{kc}, 같은 플랫폼')
    sp = report['seed_spread'][f'K{k0}']['Dread']['control']
    if sp is not None:
        say(f"참고: ΔDread {fmt(report['deltas'][f'K{k0}']['Dread']['delta'], True)} vs K{kc} seed 흔들림 {fmt(sp)} "
            f"({'흔들림보다 큼' if abs(report['deltas'][f'K{k0}']['Dread']['delta']) > sp else '흔들림 이내'})")
    v = report['gate']['verdict']
    if v.startswith('PASS'):
        say('→ 잠정 통과: production(K16 5-fold)으로 바꿔 한 번의 동결된 public 시험을 준비할 근거입니다 (public 향상의 증명은 아님).')
    elif v.startswith('NOT PASSED'):
        say('→ 통과하지 못함: K16을 채택하지 않습니다 (production으로 바꾸지 마세요).')
    elif v.startswith('INVALID'):
        say("→ 무효: 작업들의 환경이 달라 비교하지 않습니다. CFG['run_name']을 바꿔 4개 작업을 같은 환경에서 다시 돌리세요.")
    else:
        say(f'→ 미완료: seed {pairs}의 짝만 끝났거나 검증 명단이 일부입니다. 진단용 숫자일 뿐 판정·production 권고는 없습니다. 6번 셀을 이어서 실행하세요.')
    say('※ 채택 판단은 이 같은 플랫폼 짝 비교로만 합니다. 저장된 EXP107과의 차이는 플랫폼 차이(로컬 vs Kaggle fold별 0.001–0.004 Dread)와 '
        '섞여 있어 대조군이 아닙니다.')

def evaluate_production(say, report):
    kc, kt, seed, cf = int(CFG['control_k_train']), int(CFG['k_train']), int(CFG['seed']), int(CFG['control_fold'])
    preds = {f: p for f in [int(x) for x in CFG['folds']] for p in [load_job_preds('production', kt, seed, f)] if p}
    say(f"K{kt} 완료 fold {sorted(preds)} / {[int(x) for x in CFG['folds']]}")
    report['folds_done'] = sorted(preds)
    if not preds:
        say('완료된 fold가 없습니다 → 6번 셀을 먼저 끝내세요.')
        return
    for k in [int(x) for x in CFG['k_eval']]:
        S, rows = to_matrix(preds, k)
        B = EXP107_OOF[k]
        refs_all = references(rows)
        say(f'\n[K{k}] macro AUC — 이 실행 (괄호: 저장된 EXP107, cross-platform reference, not a control) ' +
            ('주 지표' if k == int(CFG['k_eval'][0]) else '진단용'))
        say('| scope | n | ' + ' | '.join(refs_all) + ' |')
        for fold in sorted(preds):
            fr = np.asarray(sorted(ROW_OF[u] for u in preds[fold][k][0]), dtype=int)
            cells = [f"{fmt(macro_score(Y, S[fr], M)[0])} ({fmt(macro_score(Y, B[fr], M)[0])})" for Y, M in references(fr).values()]
            say(f'| fold{fold} | {len(fr)} | ' + ' | '.join(cells) + ' |')
        cells = [f"{fmt(macro_score(Y, S[rows], M)[0])} ({fmt(macro_score(Y, B[rows], M)[0])})" for Y, M in refs_all.values()]
        say(f'| pooled | {len(rows)} | ' + ' | '.join(cells) + ' |')
        t0 = time.time()
        d = paired([S], [B], rows, refs_all, component_draws(rows, int(CFG['n_boot']), int(CFG['boot_seed'])))
        report['deltas'][f'K{k}_vs_exp107'] = d
        say(f"Δ vs 저장된 EXP107 (cross-platform reference, not a control; component bootstrap {CFG['n_boot']}회, {time.time() - t0:.0f}s): " + delta_line(d))
        if k == int(CFG['k_eval'][0]):
            Y = refs_all['Dread'][0]
            pa, pb = macro_score(Y, S[rows])[1], macro_score(Y, B[rows])[1]
            say('소견별 pooled Dread: ' + ' | '.join(f'{t} {fmt(a)} ({fmt(a - b, True)})' for t, a, b in zip(LAB, pa, pb)))
    k0 = int(CFG['k_eval'][0])
    complete = sorted(preds) == [0, 1, 2, 3, 4] and not CFG['subset']
    report['gate_cross_platform'] = gate(say, report['deltas'][f'K{k0}_vs_exp107'], 'complete' if complete else 'incomplete',
                                         f'K{k0} vs 저장된 EXP107 — cross-platform, 채택 근거 아님')
    envs = {n: e for n, e in JOB_ENV.items() if n.startswith('production_')}
    if len({json.dumps(e, sort_keys=True) for e in envs.values()}) > 1:
        say(f'⚠ production 작업들의 환경 지문이 섞여 있습니다: {envs}')
    report['environment'] = envs
    ctrl, src = load_job_preds('production', kc, seed, cf), f'production_k{kc}_s{seed}_f{cf}'
    if ctrl is None:                                         # 같은 레시피(K12 s42 fold 0)를 screen에서 이미 돌렸다면 그것을 씁니다
        ctrl, src = load_job_preds('screen', kc, seed, cf), f'screen_k{kc}_s{seed}_f{cf} (같은 레시피 해시)'
    if ctrl is not None and cf in preds:
        say(f'\n[같은 플랫폼 짝 대조군] fold {cf}: K{kt} − K{kc} ({src})')
        for k in [int(x) for x in CFG['k_eval']]:
            Sa, rows = to_matrix({cf: preds[cf]}, k)
            Sb, rows_b = to_matrix({cf: ctrl}, k)
            if not np.array_equal(rows, rows_b):
                say('  검사 목록이 달라 비교하지 않습니다')
                continue
            d = paired([Sa], [Sb], rows, references(rows), component_draws(rows, int(CFG['n_boot']), int(CFG['boot_seed'])))
            report['deltas'][f'control_K{k}'] = d
            say(f'  K{k}: ' + delta_line(d))
    else:
        say(f'\n[같은 플랫폼 짝 대조군] 아직 없음 (fold {cf} K{kc} 미완료)')

def evaluate_all():
    lines = []

    def say(s=''):
        print(s)
        lines.append(s)
    say(f"=== knee real {NOTEBOOK_VERSION} 결과 ({CFG['mode']}) — 이 블록 전체를 복사해서 붙여 주세요 ===")
    lost = sum(json.load(open(job_paths(j)[1], encoding='utf-8')).get('lost_seconds', 0) for j in planned_jobs() if os.path.exists(job_paths(j)[1]))
    hashes = sorted({recipe_hash(recipe_for(k, s))[:12] for _, k, s, _ in planned_jobs()})
    say(f"방식 {CFG['mode']} | 작업 {[job_name(*j) for j in planned_jobs()]} | 레시피 해시 {hashes} | K{CFG['k_train']} 변형 / K{CFG['control_k_train']} 대조 / "
        f"평가 K{CFG['k_eval']} | 계산 누적 {total_compute_s() / 3600:.2f} h (잃은 시간 {lost / 3600:.2f} h) | 장치 {CFG['device']} | subset {CFG['subset']}")
    report = dict(version=NOTEBOOK_VERSION, mode=CFG['mode'], deltas={}, seed_spread={})
    (evaluate_screen if CFG['mode'] == 'screen' else evaluate_production)(say, report)
    say(f"\nCFG: {json.dumps({k: CFG[k] for k in RECIPE_KEYS + ('k_train', 'run_name', 'mode', 'screen_fold', 'screen_seeds', 'folds', 'run_control', 'control_k_train', 'n_boot')}, ensure_ascii=False)}")
    say(f'=== 끝 (notebook {NOTEBOOK_VERSION}) ===')
    report['lines'] = lines
    out = os.path.join(plan_root(), f"eval_report_{CFG['mode']}_{NOTEBOOK_VERSION}.json")
    atomic_write_json(out, report)
    print('평가 기록 저장:', out)
    return report

EVAL_REPORT = evaluate_all()
'''
