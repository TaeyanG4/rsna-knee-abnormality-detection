"""Write docs/learning/colab_knee_baseline.ipynb — a personal, educational Colab notebook (Korean text and comments)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n').splitlines(keepends=True)})


md("""
# 무릎 MRI 12개 소견 분류 — 단순 CNN 기준선 (교육용) · v1.0.1

> 버전 v1.0.1 (2026-10-07) — v1.0.0 대비: 데이터 셀이 Drive 전체에서 images.npy를 찾고 파일 크기를 확인합니다.

RSNA Knee 대회 데이터를 **줄인 학습용 세트**로 단순 2.5D CNN을 처음부터 끝까지 학습해 보는 노트북입니다.

> 규칙: 대회 데이터는 팀 밖과 비공개로 공유하면 안 됩니다. 이 데이터와 노트북은 본인 계정에서만 쓰세요.

---

## ⏱ 걸리는 시간 (Colab 무료 T4 기준, 추정치)

| 단계 | 시간 |
|---|---|
| 0. 런타임 준비, 패키지 설치 | 1~2분 |
| 1. 데이터 받기 (2.6GB) + 로컬 디스크로 복사 | 2~5분 |
| 2. 데이터 살펴보기 | 1분 이내 |
| 5. 학습 — ResNet34, fold 1개, 8 epoch | **약 20~30분** (epoch당 2~3분) |
| 6. 평가 | 1분 이내 |
| (선택) 5개 fold 모두 | 약 2~2.5시간 |
| (선택) ResNet50으로 바꾸면 | 약 1.5~2배 |

무료 Colab은 오래 쓰지 않으면 세션이 끊길 수 있습니다. **fold 하나씩** 돌리고 결과를 저장하세요.

---

## 🧠 원리 — 이 노트북이 하는 일

**1) 문제:** 무릎 MRI 검사 하나를 보고 12개 소견(ACL 파열, 반월상연골 파열, 관절염, 삼출 등)이 **각각 있는지 확률**을 냅니다. 소견이 여러 개 동시에 있을 수 있어서 *다중 라벨 분류*입니다.

**2) 입력 — "2.5D":** MRI는 3D지만, 여기서는 검사마다 **2D 슬라이스 12장**을 씁니다. 원본 DICOM을 무릎 중심 140mm로 자르고, 실제 위치 순서대로 정렬하고, 밝기를 8비트로 맞춘 캐시에서 골라낸 장들입니다. 3D 모델보다 가볍고 빠르면서 여러 단면을 볼 수 있습니다.

**3) 전이학습:** ImageNet(일반 사진 128만 장)으로 미리 학습된 ResNet을 가져와 시작합니다. 테두리·질감 같은 기본 시각 특징을 이미 알고 있어서, 적은 의료영상으로도 빨리 배웁니다.

**4) 슬라이스 평균 (MIL pooling):** 슬라이스 12장 각각을 CNN에 넣어 특징 벡터를 얻고, **12개를 평균**내서 검사 하나의 특징으로 만듭니다. 그 위에 선형층 하나가 12개 점수를 냅니다. 어떤 장에 소견이 보이는지 몰라도 학습할 수 있는 단순한 방법입니다.

**5) 라벨과 손실:** 정답 라벨은 판독 보고서를 LLM이 읽고 만든 **확률(soft label)**입니다. 공식 정답(전문의가 영상을 다시 판독)과 다를 수 있는 **잡음 섞인 라벨**입니다. 손실은 이진 교차엔트로피(BCE)이고, 드문 소견(MCL, 골절 등)의 양성 비중을 키우는 `pos_weight`를 씁니다.

**6) 검증 — fold와 OOF:** 검사를 5묶음(fold)으로 나눠, 4개로 학습하고 나머지 1개로 평가합니다. 같은 환자의 검사는 같은 fold에 들어가게 나눠져 있어서 정보가 새지 않습니다. 5번 돌리면 모든 검사에 대해 "학습 때 못 본 예측(OOF)"이 생깁니다.

**7) 평가 지표 — AUC:** 양성 검사와 음성 검사를 하나씩 뽑았을 때 모델이 양성에 더 높은 점수를 줄 확률입니다. 1.0이 완벽, 0.5가 무작위. 12개 소견의 평균(macro AUC)이 대회 점수 방식입니다.
- 주의: **채점 정답지(라벨 세트)가 바뀌면 점수의 크기도 바뀝니다.** 같은 정답지 안에서만 모델끼리 비교하세요.

**8) 혼합 정밀도(AMP):** 계산 일부를 16비트로 해서 T4에서 약 2배 빠르게, 메모리는 적게 씁니다.
""")

md("""
## 0. 런타임 설정
Colab 메뉴 → **런타임 → 런타임 유형 변경 → T4 GPU**를 고르세요.
""")
code("""
# GPU가 잡혔는지 확인합니다. "Tesla T4, 15360 MiB" 비슷하게 나오면 정상입니다.
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# timm: 사전학습된 이미지 모델(ResNet, EfficientNet 등)을 한 줄로 불러오는 라이브러리
# scikit-learn: AUC 같은 평가 지표 계산용
!pip -q install timm scikit-learn
""")

md("""
## 1. 데이터 준비 (둘 중 하나)
- **A. Google Drive:** `images.npy`(약 2.6GB)와 `meta.csv`를 Drive의 `knee_compact/` 폴더에 올려 두었을 때.
- **B. 비공개 Kaggle 데이터셋:** 두 파일을 본인 계정의 비공개 데이터셋으로 올려 두었을 때. Kaggle 토큰 파일은 **본인이 직접** 업로드합니다.

어느 쪽이든 마지막에 Colab 로컬 디스크(`/content`)로 복사합니다. Drive에서 직접 읽으면 학습 중 데이터 읽기가 느려지기 때문입니다.
""")
code("""
import os, shutil

SOURCE = 'drive'                                              # 'drive' 또는 'kaggle'
KAGGLE_DATASET = 'YOUR_KAGGLE_ID/knee-compact-learning-v1'    # B 방법일 때 본인 데이터셋 이름으로 바꾸세요
DATA = '/content/knee_compact'                                # 학습 때 읽을 로컬 폴더
os.makedirs(DATA, exist_ok=True)

if SOURCE == 'drive':
    from google.colab import drive
    drive.mount('/content/drive')                             # 브라우저 팝업에서 Google 계정 권한을 허용
    import glob
    EXPECTED = 2614972544                                     # images.npy 정상 크기(바이트). 다르면 업로드가 덜 끝난 것
    # 폴더 이름이나 위치가 달라도 찾도록 MyDrive 전체에서 images.npy를 검색 (Drive가 크면 수십 초 걸릴 수 있음)
    hits = sorted(glob.glob('/content/drive/MyDrive/**/images.npy', recursive=True)[:5] +
                  glob.glob('/content/drive/MyDrive/images.npy'))
    if not hits:
        print('MyDrive 최상위 목록:', sorted(os.listdir('/content/drive/MyDrive'))[:50])
        raise FileNotFoundError('Drive에서 images.npy를 찾지 못했습니다. 업로드가 끝났는지(drive.google.com 우측 아래 진행 표시), '
                                '"공유 문서함"이 아니라 "내 드라이브"에 있는지 확인하세요. 업로드 직후라면 런타임 > 세션 다시 시작 후 다시 실행하세요.')
    src = os.path.dirname(hits[0])
    size = os.path.getsize(hits[0])
    print('찾은 위치:', src, '| images.npy 크기:', size)
    if size != EXPECTED:
        raise RuntimeError(f'images.npy 크기가 {size}바이트로 정상({EXPECTED})과 다릅니다. 업로드가 아직 진행 중이거나 중간에 끊긴 것입니다.')
    for name in ['images.npy', 'meta.csv']:                   # 로컬 디스크로 복사 (1~3분)
        shutil.copy(f'{src}/{name}', f'{DATA}/{name}')
else:
    from google.colab import files
    print('kaggle.json 또는 access_token 파일을 선택하세요')
    up = files.upload()                                       # 토큰 파일을 본인이 직접 선택
    name = list(up)[0]
    os.makedirs('/root/.kaggle', exist_ok=True)
    dst = '/root/.kaggle/kaggle.json' if name.endswith('.json') else '/root/.kaggle/access_token'
    open(dst, 'wb').write(up[name])
    os.chmod(dst, 0o600)                                      # 토큰 파일은 본인만 읽을 수 있게
    if not name.endswith('.json'):
        os.environ['KAGGLE_API_TOKEN'] = open(dst).read().strip()   # 새 형식 토큰은 환경 변수로 전달
    !pip -q install -U kaggle
    !kaggle datasets download -d {KAGGLE_DATASET} -p {DATA} --unzip -q

print(os.listdir(DATA))                                       # images.npy, meta.csv가 보이면 준비 완료
""")

md("""
## 2. 데이터 살펴보기
`images.npy`는 **메모리 매핑(mmap)**으로 엽니다. 2.6GB를 RAM에 한꺼번에 올리지 않고, 필요한 검사만 그때그때 디스크에서 읽는 방식입니다.
""")
code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# 12개 소견 이름 (meta.csv의 열 이름과 같은 순서)
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

X = np.load(f'{DATA}/images.npy', mmap_mode='r')   # 모양: [검사 수, 슬라이스 12, 224, 224], 값 0~255
meta = pd.read_csv(f'{DATA}/meta.csv')               # 검사 ID, fold 번호, Dread 라벨(학습용), 비교용 라벨들
print('이미지 배열:', X.shape, X.dtype, '| 검사 수:', len(meta))
print('fold별 검사 수:', meta['fold'].value_counts().sort_index().to_dict())

# 라벨 확률이 0.5 이상인 비율 = 대략적인 양성 비율. 드문 소견일수록 학습이 어렵습니다.
print('Dread 라벨 기준 양성 비율:')
print((meta[T] >= 0.5).mean().round(3).to_string())

# 검사 하나의 슬라이스 12장을 그려 봅니다.
i = 0
fig, ax = plt.subplots(2, 6, figsize=(15, 5))
for j in range(12):
    ax[j // 6, j % 6].imshow(X[i, j], cmap='gray')
    ax[j // 6, j % 6].set_title(f'slice {j}', fontsize=8)
    ax[j // 6, j % 6].axis('off')
plt.suptitle('검사 1건의 슬라이스 12장 (앞쪽은 시상면, 뒤쪽은 관상면·축상면)')
plt.show()
print('이 검사의 라벨:', meta.loc[i, T].round(2).to_dict())
""")

md("""
## 3. 데이터셋과 증강
- 모델에 넘기는 형태: 슬라이스 12장을 [12, 224, 224] 텐서로, 값은 0~1.
- **증강(augmentation)**은 학습 때만 이미지를 조금씩 바꿔 모델이 외우지 않고 일반화하도록 돕습니다. 여기서는 밝기·대비 조금, 작은 이동만 씁니다.
- 연습 과제: 증강을 **하나씩** 바꿔 보고 점수가 어떻게 변하는지 보세요.
""")
code("""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class KneeSet(Dataset):
    \"\"\"검사 하나 = (슬라이스 12장 텐서, 12개 soft label)\"\"\"
    def __init__(self, rows, train):
        self.rows = list(rows)                                   # 이 데이터셋에 쓸 meta.csv의 행 번호들
        self.train = train                                       # True면 증강을 적용
        self.y = meta.loc[self.rows, T].values.astype('float32') # Dread soft label (0~1 확률)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # 디스크(mmap)에서 이 검사의 12장만 읽고 0~1 범위 실수로 바꿉니다.
        x = torch.from_numpy(np.array(X[self.rows[i]], dtype=np.float32) / 255.0)   # [12, 224, 224]
        if self.train:
            # 대비를 ±5%, 밝기를 ±0.025만큼 무작위로 바꿉니다 (장비마다 밝기가 다른 것을 흉내).
            x = x * (1 + 0.1 * (torch.rand(1) - 0.5)) + 0.05 * (torch.rand(1) - 0.5)
            # 상하좌우로 최대 12픽셀 이동합니다 (무릎 위치가 조금씩 다른 것을 흉내).
            dx, dy = np.random.randint(-12, 13, 2)
            x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(1, 2))
        return x.clamp(0, 1), torch.from_numpy(self.y[i])
""")

md("""
## 4. 모델 — 슬라이스별 CNN 특징 → 평균 → 12개 출력
상위권 참가자의 조언대로 **어텐션 없이 단순 pooling**부터 시작합니다. `BACKBONE` 한 줄만 바꾸면 다른 모델을 시험할 수 있습니다 (`resnet34`, `resnet50`, `efficientnet_b0` 등).
""")
code("""
import timm

BACKBONE = 'resnet34'   # 바꿔 보기: 'resnet50', 'efficientnet_b0', 'convnext_tiny'

# ImageNet 사전학습 모델은 이 평균/표준편차로 정규화된 입력을 기대합니다.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class SliceMean(nn.Module):
    def __init__(self, name):
        super().__init__()
        # num_classes=0 → 분류층을 떼고 특징 벡터만 내보내는 로 씁니다.
        self.enc = timm.create_model(name, pretrained=True, num_classes=0)
        # 특징 벡터 → 12개 소견 점수(logit)
        self.head = nn.Linear(self.enc.num_features, 12)

    def forward(self, x):                              # x: [배치 B, 슬라이스 12, H, W]
        b, s, h, w = x.shape
        x = x.reshape(b * s, 1, h, w)                  # 슬라이스를 각각 독립된 이미지로 펼침
        x = x.repeat(1, 3, 1, 1)                       # 흑백 1채널 → 3채널 (사전학습 모델이 3채널을 기대)
        x = (x - MEAN.to(x.device)) / STD.to(x.device) # ImageNet 방식 정규화
        f = self.enc(x)                                # [B*12, 특징 차원]
        f = f.reshape(b, s, -1).mean(1)                # 슬라이스 12장의 특징을 평균 → 검사 하나의 특징
        return self.head(f)                            # [B, 12] — 시그모이드 전 점수(logit)
""")

md("""
## 5. 학습 (fold 하나)
- 손실: soft label BCE + `pos_weight`(드문 소견의 양성 오류를 더 크게 벌함).
- 학습률 스케줄: OneCycle (처음엔 천천히 올렸다가 점점 줄임).
- 혼합 정밀도(AMP fp16)로 빠르게 학습합니다.
- `FOLD`를 0~4로 바꾸면 다른 fold를 검증으로 씁니다.
""")
code("""
from sklearn.metrics import roc_auc_score
import time

FOLD, EPOCHS, BS, LR = 0, 8, 8, 3e-4    # 검증 fold, 학습 횟수, 배치(검사 수), 최대 학습률
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
dev = 'cuda'

# FOLD 번호인 검사는 검증, 나머지는 학습에 씁니다.
tr = meta.index[meta['fold'] != FOLD]
va = meta.index[meta['fold'] == FOLD]
print('학습', len(tr), '검사 / 검증', len(va), '검사')

# pos_weight = (음성 비율 / 양성 비율), 너무 극단적이지 않게 1~10으로 자릅니다.
prev = np.clip((meta.loc[tr, T].values >= 0.5).mean(0), 0.03, 0.7)
pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

dl_tr = DataLoader(KneeSet(tr, True), BS, shuffle=True, num_workers=2, drop_last=True)
dl_va = DataLoader(KneeSet(va, False), BS, shuffle=False, num_workers=2)

model = SliceMean(BACKBONE).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)          # 가중치 감쇠로 과적합 완화
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(dl_tr), pct_start=0.15)
scaler = torch.amp.GradScaler()                                                  # fp16에서 기울기 값이 너무 작아지는 것을 막음

def predict(dl):
    \"\"\"검증 세트 전체에 대해 확률(0~1)을 예측합니다.\"\"\"
    model.eval(); P = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for x, _ in dl:
            P.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
    return np.concatenate(P)

def macro_auc(Y, P):
    \"\"\"12개 소견의 AUC 평균. 양성이나 음성이 한쪽만 있는 소견은 계산에서 뺍니다.\"\"\"
    return np.mean([roc_auc_score(Y[:, t], P[:, t]) for t in range(12) if 0 < Y[:, t].sum() < len(Y)])

Yd = np.rint(meta.loc[va, T].values).astype(int)   # 검증용 정답: Dread 확률을 0/1로 반올림
for ep in range(EPOCHS):
    t0 = time.time(); model.train(); tot = 0
    for x, y in dl_tr:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast('cuda', dtype=torch.float16):                        # 순전파는 fp16으로
            loss = F.binary_cross_entropy_with_logits(model(x).float(), y, pos_weight=pos_weight)
        opt.zero_grad()
        scaler.scale(loss).backward()                                              # 역전파
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)                          # 기울기 폭주 방지
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
    P = predict(dl_va)
    print(f'epoch {ep + 1}/{EPOCHS}: train loss {tot / len(dl_tr):.4f} | '
          f'val macro AUC (Dread) {macro_auc(Yd, P):.4f} | {time.time() - t0:.0f}s')
""")

md("""
## 6. 평가 — 정답지를 바꿔 가며 채점하기
- **같은 정답지 안에서는 높을수록 좋습니다.** 정답지끼리는 난이도가 달라 직접 비교하면 안 됩니다.
  - 엄격한 라벨(relabel)은 확실한 사례만 양성이라 점수가 높게, 너그러운 라벨(flight)은 애매한 사례까지 양성이라 낮게 나옵니다.
- 학습에 쓴 라벨(Dread)로 채점하면 그 라벨에 유리합니다.
- **합의** 열은 세 라벨이 모두 동의한 검사-소견 쌍만 써서 어느 라벨에도 유리하지 않은, 더 깨끗한 기준입니다.
""")
code("""
# 세 가지 정답지 (모두 판독 보고서에서 LLM이 뽑은 라벨 — 공식 정답은 아님)
refs = {'Dread': np.rint(meta.loc[va, T].values).astype(int),
        'flight': meta.loc[va, [f'flight_{t}' for t in T]].values,
        'relabel': meta.loc[va, [f'relabel_{t}' for t in T]].values}
for k, Y in refs.items():
    print(f'{k:8s} 기준 macro AUC {macro_auc(Y, P):.4f}')

# 합의 기준: 세 정답지가 모두 같은 판정(모두 양성 또는 모두 음성)인 쌍만 남깁니다.
st = np.stack([refs['Dread'], refs['flight'], refs['relabel']])
cons = (st.min(0) == st.max(0))                       # [검증 검사 수, 12] — True면 세 라벨이 일치
aucs = [roc_auc_score(refs['Dread'][cons[:, t], t], P[cons[:, t], t]) for t in range(12)
        if 0 < refs['Dread'][cons[:, t], t].sum() < cons[:, t].sum()]
print(f'합의     기준 macro AUC {np.mean(aucs):.4f}  (합의된 쌍 비율 {cons.mean():.2f})')

# 소견별 AUC (Dread 기준) — 어떤 소견이 어려운지 봅니다.
per = pd.DataFrame({t: [roc_auc_score(refs['Dread'][:, j], P[:, j])] for j, t in enumerate(T)}).T
per.columns = ['AUC (Dread)']
print(per.round(3).to_string())
""")

md("""
## 7. 저장과 다음 실험 아이디어
- 같은 설정을 **seed만 바꿔 2번 이상** 돌려 보세요. 같은 조건에서도 점수가 0.005 안팎으로 흔들립니다. 그보다 큰 차이만 진짜 개선으로 보세요.
- 한 번에 **하나씩** 바꿔 보기: 증강 종류, epoch 수, 백본(`resnet50`, `efficientnet_b0`), 평균 대신 최댓값 pooling(`.amax(1)`).
- 프로젝트에서 가장 좋았던 단일 계열은 **RadImageNet(의료영상 135만 장으로 사전학습) ResNet50**입니다. 구조는 그대로 두고 가중치만 바꿔 비교해 볼 수 있습니다.
""")
code("""
# 모델 가중치와 검증 예측을 저장합니다. Colab 세션이 끝나면 /content는 지워지니, 필요하면 Drive로 복사하세요.
torch.save(model.state_dict(), f'/content/{BACKBONE}_fold{FOLD}.pt')
pd.DataFrame(P, columns=T, index=meta.loc[va, 'StudyInstanceUID']).to_csv(f'/content/oof_{BACKBONE}_fold{FOLD}.csv')
print('저장 완료:', f'/content/{BACKBONE}_fold{FOLD}.pt', f'/content/oof_{BACKBONE}_fold{FOLD}.csv')
""")

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'provenance': []}},
      'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'docs' / 'learning' / 'colab_knee_baseline_v1.0.1.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote', out, len(cells), 'cells')
