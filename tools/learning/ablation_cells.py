"""Code cells for the ablation version of docs/learning/colab_knee_baseline.ipynb (personal Colab study).

Cell A replaces "3. 데이터셋과 증강" (KneeSet), cell B replaces "5. 학습", cell C replaces "6. 평가",
cell D replaces "7. 저장" (the last code cell). make_colab_ablation.py assembles the notebook from these strings.
"""

CELL_A = r'''import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 증강 종류. 실험할 때는 한 번에 바꿔서 대조군('base')과 비교합니다.
#  'none'   : 증강 없음
#  'base'   : 밝기·대비 조금 + 상하좌우 이동 (원래 노트북과 같음 → 대조군)
#  'flip'   : base + 50% 확률로 좌우 뒤집기 (왼쪽 무릎 ↔ 오른쪽 무릎을 흉내)
#  'affine' : base + 작은 회전(±10°)과 확대·축소(±10%) (자세·화면 크기 차이를 흉내)
#  'sdrop'  : base + 슬라이스 12장 중 2장을 무작위로 지우기 (일부 시퀀스가 빠진 검사를 흉내)
AUGS = ['none', 'base', 'flip', 'affine', 'sdrop']

class KneeSet(Dataset):
    """검사 하나 = (슬라이스 12장 텐서 [12, 224, 224], 12개 soft label)"""
    def __init__(self, rows, aug='none'):
        assert aug in AUGS, aug
        self.rows = list(rows)                                   # 이 데이터셋에 쓸 meta.csv 행 번호
        self.aug = aug                                           # 검증용은 항상 'none'
        self.y = meta.loc[self.rows, T].values.astype('float32') # 학습 라벨: Dread soft label (0~1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(X[self.rows[i]], dtype=np.float32) / 255.0)   # [12, H, W], 0~1
        if self.aug != 'none':
            # 대비 ±5%, 밝기 ±0.025 (장비마다 밝기가 다른 것을 흉내)
            x = x * (1 + 0.1 * (torch.rand(1) - 0.5)) + 0.05 * (torch.rand(1) - 0.5)
            # 상하좌우 최대 12픽셀 이동 (무릎 위치 차이를 흉내)
            dx, dy = np.random.randint(-12, 13, 2)
            x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(1, 2))
        if self.aug == 'flip' and torch.rand(1) < 0.5:
            x = torch.flip(x, dims=(2,))                         # 가로 방향 뒤집기 (12장 모두 같이)
        if self.aug == 'affine':
            a = np.deg2rad(np.random.uniform(-10, 10)); s = np.random.uniform(0.9, 1.1)
            theta = torch.tensor([[np.cos(a) / s, -np.sin(a) / s, 0.0],
                                  [np.sin(a) / s, np.cos(a) / s, 0.0]], dtype=torch.float32)[None]
            grid = F.affine_grid(theta, (1, 1, x.shape[1], x.shape[2]), align_corners=False)
            x = F.grid_sample(x[:, None], grid.expand(x.shape[0], -1, -1, -1), align_corners=False).squeeze(1)
        if self.aug == 'sdrop':
            x[np.random.choice(x.shape[0], 2, replace=False)] = 0  # 2장을 검은 화면으로
        return x.clamp(0, 1), torch.from_numpy(self.y[i])
'''

CELL_B = r'''from sklearn.metrics import roc_auc_score
import os, time, random, json

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS, BS, LR = 8, 8, 3e-4                 # 학습 횟수, 배치(검사 수), 최대 학습률 — 실험 중에는 고정!
# 결과는 Drive에 저장합니다. Colab 연결이 끊겨도 이미 끝난 실험은 다시 돌리지 않고 건너뜁니다.
SAVE = '/content/drive/MyDrive/knee_runs' if os.path.isdir('/content/drive/MyDrive') else '/content/knee_runs'
os.makedirs(SAVE, exist_ok=True)
print('결과 저장 위치:', SAVE)

def seed_all(seed):
    """같은 seed면 같은 초기값·같은 데이터 순서·같은 증강이 나오도록 고정합니다."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def worker_init(worker_id):
    # DataLoader 일꾼마다 numpy 난수를 다르게, 그러나 재현 가능하게 (안 하면 일꾼들이 같은 증강을 반복)
    np.random.seed(torch.initial_seed() % 2**32)

def run_config(aug, seed, fold, backbone, epochs):
    """실험 설정 기록. 같은 이름의 결과가 이미 있을 때, 설정까지 같아야 재사용합니다 (v1.2.0)."""
    return dict(aug=aug, seed=seed, fold=fold, backbone=backbone, epochs=epochs, bs=BS, lr=LR,
                n_studies=len(meta), images_shape=list(X.shape))

def run_experiment(name, aug='base', seed=42, fold=0, backbone=None, epochs=None):
    """fold 하나를 학습하고 검증 예측(OOF)을 Drive에 저장합니다. 같은 설정의 결과가 이미 있으면 건너뜁니다."""
    backbone = backbone or BACKBONE; epochs = epochs or EPOCHS
    out = f'{SAVE}/{name}_seed{seed}_fold{fold}.csv'
    cfg = run_config(aug, seed, fold, backbone, epochs)
    cfg_path = out.replace('.csv', '.json')
    if os.path.exists(out):
        if not os.path.exists(cfg_path):
            # v1.1.x 결과에는 설정 파일이 없습니다. 이름이 같으면 같은 설정이라고 보고, 지금 설정을 기록해 둡니다.
            json.dump(dict(cfg, note='v1.1.x 결과: 설정 기록 없이 재사용'), open(cfg_path, 'w'), indent=1)
            print('이미 있음(이전 버전 결과) → 건너뜀:', out)
            return
        old = json.load(open(cfg_path))
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if diff:
            raise ValueError(f'{out} 은(는) 다른 설정으로 만든 결과입니다: {diff}. 실험 이름을 바꾸세요.')
        print('이미 있음 → 건너뜀:', out)
        return
    seed_all(seed)
    tr = meta.index[meta['fold'] != fold]; va = meta.index[meta['fold'] == fold]
    # pos_weight: 드문 소견의 양성 오류를 더 크게 벌함 (1~10으로 자름)
    prev = np.clip((meta.loc[tr, T].values >= 0.5).mean(0), 0.03, 0.7)
    pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    g = torch.Generator(); g.manual_seed(seed)
    dl_tr = DataLoader(KneeSet(tr, aug), BS, shuffle=True, num_workers=2, drop_last=True,
                       generator=g, worker_init_fn=worker_init)
    dl_va = DataLoader(KneeSet(va, 'none'), BS, shuffle=False, num_workers=2)
    model = SliceMean(backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * len(dl_tr), pct_start=0.15)
    use_amp = dev == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    Yd = np.rint(meta.loc[va, T].values).astype(int)
    print(f'== {name} | aug={aug} seed={seed} fold={fold} {backbone} | 학습 {len(tr)} / 검증 {len(va)}')
    for ep in range(epochs):
        t0 = time.time(); model.train(); tot = 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y, pos_weight=pos_weight)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        model.eval(); P = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            for x, _ in dl_va:
                P.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
        P = np.concatenate(P)
        print(f'  epoch {ep + 1}/{epochs}: loss {tot / len(dl_tr):.4f} | val AUC (Dread) {fast_macro(Yd, P):.4f} | {time.time() - t0:.0f}s')
    # 마지막 epoch의 예측을 저장합니다 (epoch을 점수 보고 고르면 검증 점수가 부풀려지므로 고르지 않음).
    # 임시 파일에 쓴 뒤 이름을 바꿉니다 (중간에 끊겨도 반쯤 쓴 결과가 남지 않게).
    pd.DataFrame(P, columns=T, index=meta.loc[va, 'StudyInstanceUID']).to_csv(out + '.tmp')
    json.dump(cfg, open(cfg_path, 'w'), indent=1)
    os.replace(out + '.tmp', out)
    print('  저장:', out)

def fast_auc(y, p):
    """순위 기반 AUC (roc_auc_score와 같은 값, 더 빠름). 양성이나 음성이 없으면 nan."""
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def fast_macro(Y, P, M=None):
    """12개 소견 AUC의 평균. M이 있으면 각 소견에서 M이 True인 검사만 씁니다(합의 기준)."""
    vals = []
    for t in range(Y.shape[1]):
        m = np.ones(len(Y), bool) if M is None else M[:, t]
        vals.append(fast_auc(Y[m, t], P[m, t]))
    return np.nanmean(vals)
'''

CELL_C = r'''import glob

def references(rows):
    """정답지 네 가지. 모두 판독문에서 뽑은 라벨이라 공식 정답은 아닙니다."""
    d = np.rint(meta.loc[rows, T].values).astype(int)
    f = meta.loc[rows, [f'flight_{t}' for t in T]].values.astype(int)
    r = meta.loc[rows, [f'relabel_{t}' for t in T]].values.astype(int)
    st = np.stack([d, f, r])
    cons = st.min(0) == st.max(0)                      # 세 라벨이 모두 같은 칸만 True
    return {'Dread': (d, None), 'flight': (f, None), 'relabel': (r, None), 'consensus': (d, cons)}

def load_run(name, seed, fold=0):
    p = pd.read_csv(f'{SAVE}/{name}_seed{seed}_fold{fold}.csv', index_col=0)
    rows = meta.set_index('StudyInstanceUID').loc[p.index, 'row'].values
    return rows, p[T].values

def compare(control, variants, fold=0, n_boot=200):
    """대조군 대비 각 변형의 점수 차이를, 정답지 4개 각각에 대해 보여줍니다.
    - 점수는 seed별 점수의 평균, 차이는 (변형 평균 − 대조군 평균).
    - 구간은 검증 검사를 복원추출해 200번 다시 계산한 95% 범위 (모든 실험에 같은 표본을 써서 짝 비교).
    - '흔들림'은 대조군 seed끼리의 차이: 이보다 작은 차이는 우연일 수 있습니다."""
    def seeds_of(name):
        return sorted(int(f.split('_seed')[1].split('_')[0]) for f in glob.glob(f'{SAVE}/{name}_seed*_fold{fold}.csv'))
    ctrl_seeds = seeds_of(control)
    ready = []
    for v in variants:
        if seeds_of(v) != ctrl_seeds:
            print(f'※ {v}: seed {seeds_of(v)} ≠ 대조군 seed {ctrl_seeds} → 아직 비교하지 않습니다 (실험이 덜 끝남)')
        else:
            ready.append(v)
    variants = ready
    runs = {n: [load_run(n, s, fold) for s in seeds_of(n)] for n in [control] + variants}
    rows = runs[control][0][0]
    for n, rs in runs.items():
        assert rs, f'{n}: 저장된 실험이 없습니다'
        assert all((r == rows).all() for r, _ in rs), '검증 검사 순서가 다릅니다'
    refs = references(rows)
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(rows), len(rows)) for _ in range(n_boot)]
    def mean_score(rs, Y, M, idx=None):
        idx = np.arange(len(rows)) if idx is None else idx
        return np.mean([fast_macro(Y[idx], P[idx], None if M is None else M[idx]) for _, P in rs])
    lines = []
    for ref, (Y, M) in refs.items():
        c = mean_score(runs[control], Y, M)
        cs = [fast_macro(Y, P, M) for _, P in runs[control]]
        noise = max(cs) - min(cs) if len(cs) > 1 else float('nan')
        lines.append(f'[{ref}] {control} {c:.4f} (seed {len(cs)}개, 흔들림 {noise:.4f})')
        for v in variants:
            d = mean_score(runs[v], Y, M) - c
            bd = [mean_score(runs[v], Y, M, b) - mean_score(runs[control], Y, M, b) for b in boots]
            lo, hi = np.nanpercentile(bd, [2.5, 97.5])
            lines.append(f'    {v:8s} Δ {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  (seed {len(runs[v])}개)')
    print('\n'.join(lines))
    return '\n'.join(lines)
'''

CELL_D = r'''# ===== 7번 셀 · v1.2.0 — 실험 계획: (이름, 증강, seed, 에폭) =====
# T4 기준 ResNet34 fold 0: 8 에폭 약 20~30분, 16 에폭 약 45~60분.
# 연결이 끊기면 이 셀을 다시 실행하세요. 같은 설정으로 끝난 실험은 Drive에 있어서 건너뜁니다.
PLAN = [
    ('ctrl',   'base', 42, 8),    # run 1: 대조군 (8 에폭) — seed 두 개의 차이 = 우연에 의한 흔들림
    ('ctrl',   'base', 43, 8),
    ('flip',   'flip', 42, 8),    # run 1: 바꾼 것 1개 (좌우 뒤집기)
    ('flip',   'flip', 43, 8),
    ('ctrl16', 'base', 42, 16),   # run 2: 같은 설정, 에폭만 16
    ('ctrl16', 'base', 43, 16),
]
for name, aug, seed, ep in PLAN:
    run_experiment(name, aug=aug, seed=seed, epochs=ep)

# 결과 비교 — 이 출력 전체를 복사해서 알려 주세요 (버전 v1.2.0도 같이).
summary = compare('ctrl', ['flip', 'ctrl16'])
'''
