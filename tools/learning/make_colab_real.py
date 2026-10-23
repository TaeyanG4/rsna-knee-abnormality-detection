"""Build docs/learning/colab_knee_real_<VERSION>.ipynb (VERSION in real_v2_cells.py; v2.0.1 now, the v2.0.0 file is kept as built): a self-contained, resumable Colab (free T4) notebook that trains the EXP107
RadImageNet ResNet50 five-fold recipe with K16 training windows (K36 primary / K24 diagnostic evaluation; optional fold-0 K12 control).

  python tools/learning/make_colab_real.py [--out PATH] [--force]

Cell texts live in tools/learning/real_v2_cells.py. This generator
  * copies the library pieces VERBATIM from the EXP107 training sources (ast source segments), after checking every source file's sha256
    against the EXP107 protocol manifest (training_code_sha256 / source_bindings), so the notebook code is the EXP107 code;
  * computes the embedded verification data from project files: canonical (EXP099) study order, sha256 of the float32 Dread label array
    in that order, the fold vector and per-fold ordered train/val hashes and pos_weight (re-derived and compared with the EXP107
    manifest), the uid -> component map (VALIDATION_GROUP_CONTRACT_V1/components.json, checked component-disjoint across folds), sha256 of
    the Gold58 uids (uids read from storage/competition/train.csv, never printed), the corpus/weights file sizes and sha256, and the
    stored EXP107 OOF at K36/K24 as float16 (gzip + base64, sha256 of the decoded bytes);
  * cross-checks the local compact meta.csv (.local/learning-compact-v1/meta.csv, the file the user uploaded to Drive) against them.
Reads local files only (no network, no GPU). Writes only the notebook (refuses to overwrite unless --force).
"""
from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_v2_cells as C  # noqa: E402

VERSION = C.VERSION
OUT = ROOT / 'docs' / 'learning' / f'colab_knee_real_{VERSION}.ipynb'
UID = 'StudyInstanceUID'
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
       'Contusion', 'Fracture']
PROTOCOL = 'artifacts/experiments/EXP107_KFOLD_RADR50_PROTOCOL_V1/manifest.json'
PROTOCOL_SHA256 = 'b51ce3b7e9aeff8bded1cadf1afe50d3cb926df4df139237e3cbaa778836cc81'   # = EXP107_RADR50_DREAD_OOF/result.json protocol_sha256
OOF_DIR = 'artifacts/experiments/EXP107_RADR50_DREAD_OOF'
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
WEIGHTS = 'storage/weights/marwanmath__resnet-50-radimagenet-marwan/ResNet50.pt'
META = '.local/learning-compact-v1/meta.csv'
COMPONENTS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
GOLD_SOURCE = 'storage/competition/train.csv'
CORPUS_REF = 'taeyangg4/knee-raptor-corpus-336'
WEIGHTS_REF = 'marwanmath/resnet-50-radimagenet-marwan'
SMALL_CORPUS_FILES = ['manifest.json', 'ids.npy', 'masks.npy', 'picks.jsonl', 'uids.csv']
CHUNK = 2000   # characters per embedded base64 line (generated notebook lines stay short)

# Verbatim library pieces: (source file, [names in definition order], Korean note per name)
LABEL_PIECES = [
    ('training/src/raptor_common.py', ['LAB', 'UID']),
    ('tools/exp099_kfold.py', ['T']),
    ('tools/exp084_group_loss.py', ['uid_hash']),
    ('tools/exp099_kfold.py', ['pos_weight', 'avg_rank', 'auc', 'macro12']),
]
LIB_PIECES = [
    ('training/src/raptor_common.py', ['LAB', 'UID', 'seed_all', 'Corpus', 'eval_centers', 'train_centers', 'study_rng', 'select_centers',
                                       'windows_from', 'build_backbone', 'build_model', 'write_predictions']),
    ('tools/exp084_group_loss.py', ['worker_init']),
    ('training/src/raptor_data.py', ['IMAGENET_MEAN', 'IMAGENET_STD', 'StudyWindows', 'collate']),
    ('training/src/raptor_radimagenet.py', ['RADIMAGENET_R50_SHA256', 'RADIMAGENET_R50_KEYS', 'RADIMAGENET_R50_PARAMS', 'STAGE_MAP', 'ARCH',
                                            'INPUT_POLICIES', 'NORMS', 'file_sha256', 'map_radimagenet_state', 'strict_load_backbone',
                                            'load_radimagenet_state', 'build_radimagenet_model', 'center3_windows_from', 'normalise',
                                            'CenterSliceWindows']),
    ('training/src/raptor_runtime.py', ['native_bf16_supported', 'runtime_snapshot', 'configure_runtime']),
    ('tools/exp101_kfold_arch.py', ['state_hashes']),
]
NOTES = {
    'LAB': '12개 소견 이름(제출 열 순서)', 'UID': '검사 ID 열 이름', 'T': 'EXP099 채점 코드의 소견 목록(LAB과 같음)',
    'uid_hash': '순서 있는 검사 목록의 sha256 (fold 목록 검사용)',
    'pos_weight': 'EXP099 규칙: 학습 fold Dread 평균을 [0.03, 0.7]로 자르고 (1-p)/p를 [1, 10]으로 자른 float32',
    'avg_rank': '동점 평균 순위', 'auc': 'average-rank Mann-Whitney AUC (tools/g1_compare.py와 같음)', 'macro12': '12개 소견 AUC 평균',
    'seed_all': 'python/numpy/torch(+cuda) seed', 'Corpus': '샤드 코퍼스를 memory-map으로 읽기 (로컬 디스크)',
    'eval_centers': '(legacy 정책용, select_centers가 참조)', 'train_centers': '(legacy 정책용, select_centers가 참조)',
    'study_rng': '검사·epoch·흐름별 sha256 난수 — 창 샘플링과 밝기 jitter가 wall time에 의존하지 않음',
    'select_centers': '창 중심 고르기: within-slot-valid = 슬롯 안에서 앞뒤 슬라이스가 모두 유효한 중심만 (학습=무작위, 평가=균등 간격)',
    'windows_from': '(triplet 입력, StudyWindows.__getitem__이 참조 — 이 노트북은 center3를 씀)',
    'build_backbone': 'timm 백본 (num_classes=0, avg pool)', 'build_model': 'EXP107 모델: 백본 + 소견별 attention MIL 헤드 (dropout 0.2)',
    'write_predictions': 'OOF CSV 쓰기 (UID + 12개 확률)', 'worker_init': 'DataLoader 작업 프로세스마다 수치 라이브러리 스레드 1개',
    'IMAGENET_MEAN': '(imagenet 정규화용 상수)', 'IMAGENET_STD': '(imagenet 정규화용 상수)',
    'StudyWindows': '검사 단위 데이터셋 기본 클래스 (공유 epoch 텐서: persistent worker에도 set_epoch 반영)', 'collate': '배치 묶기',
    'RADIMAGENET_R50_SHA256': 'RadImageNet ResNet50.pt sha256', 'RADIMAGENET_R50_KEYS': '가중치 텐서 수', 'RADIMAGENET_R50_PARAMS': '백본 파라미터 수',
    'STAGE_MAP': 'torchvision Sequential 번호 → timm 이름', 'ARCH': '백본 이름', 'INPUT_POLICIES': '입력 정책', 'NORMS': '정규화 종류',
    'file_sha256': '파일 sha256', 'map_radimagenet_state': 'RadImageNet 키 → timm resnet50 키',
    'strict_load_backbone': 'strict 로드 + 바이트 단위 재확인', 'load_radimagenet_state': '가중치 파일 sha256 확인 후 로드',
    'build_radimagenet_model': 'seed 후 모델 생성(헤드 초기값은 seed의 함수) → 백본을 RadImageNet으로 교체',
    'center3_windows_from': '가운데 슬라이스 하나를 3채널로 반복, 336→224 bilinear', 'normalise': "정규화 'radimagenet' = 2x − 1",
    'CenterSliceWindows': 'EXP107 데이터셋: 창 중심·샘플링 흐름·밝기 jitter(0.9~1.1)는 StudyWindows 그대로, 입력만 center3',
    'native_bf16_supported': '(T4는 bf16 없음)', 'runtime_snapshot': '런타임 설정 기록',
    'configure_runtime': "strict: 결정론적 알고리즘·cuDNN benchmark 끔·TF32 끔 (EXP107 'strict')", 'state_hashes': '모델 상태 sha256 (초기값 비교용)',
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def need(ok, msg):
    if not ok:
        raise SystemExit('generator check failed: ' + msg)


def extract(rel, names):
    """Top-level function/class/assignment source segments of `rel`, verbatim, in the order of `names`."""
    src = (ROOT / rel).read_text(encoding='utf-8')
    lines = src.splitlines()
    found = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            found[node.name] = '\n'.join(lines[start - 1:node.end_lineno])
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            found[node.targets[0].id] = '\n'.join(lines[node.lineno - 1:node.end_lineno])
    missing = [n for n in names if n not in found]
    need(not missing, f'{rel}: missing {missing}')
    return [found[n] for n in names]


def pieces_text(pieces, hashes):
    out = []
    for rel, names in pieces:
        for name, seg in zip(names, extract(rel, names)):
            out.append(f'# --- 원본: {rel} · {name} (sha256 {hashes[rel][:12]}…) — {NOTES[name]}\n{seg}\n')
    return '\n'.join(out)


def blob_lines(data):
    s = base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode('ascii')
    return [s[i:i + CHUNK] for i in range(0, len(s), CHUNK)]


def blob_assign(name, data, dtype=None):
    body = ',\n'.join(f"    '{x}'" for x in blob_lines(data))
    expr = f'_unpack([\n{body},\n])'
    if dtype:
        expr = f"np.frombuffer({expr}, '{dtype}').copy()"
    return f'{name} = {expr}\n'


def build_embedded():
    p_path = ROOT / PROTOCOL
    need(sha256_file(p_path) == PROTOCOL_SHA256, 'EXP107 protocol manifest changed')
    p = json.loads(p_path.read_text(encoding='utf-8'))
    # ---- source files of the verbatim copies = the frozen EXP107 versions
    hashes = {}
    for name, expected in p['training_code_sha256'].items():
        rel = 'training/src/' + name
        hashes[rel] = sha256_file(ROOT / rel)
        need(hashes[rel] == expected, f'{rel} differs from the EXP107 training_code_sha256')
    for rel in ('training/src/raptor_radimagenet.py', 'tools/exp084_group_loss.py', 'tools/exp099_kfold.py', 'tools/exp101_kfold_arch.py'):
        hashes[rel] = sha256_file(ROOT / rel)
        need(hashes[rel] == p['source_bindings'][rel], f'{rel} differs from the EXP107 source binding')
    # ---- canonical order, labels, folds (re-derived; must equal the EXP107 registration)
    base = json.loads((ROOT / p['paths']['base_ordered_ids']).read_text(encoding='utf-8'))
    canonical = list(base['train']) + list(base['val'])
    need(len(canonical) == len(set(canonical)) == 4343, 'canonical order is not 4,343 unique studies')
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from raptor_common import load_label_csv  # noqa: E402  (read-only import; no side effects besides locating the workspace)
    labels, semantic = load_label_csv(ROOT / p['label_set']['path'])
    need(semantic == p['label_set']['semantic_sha256'] and sha256_file(ROOT / p['label_set']['path']) == p['label_set']['sha256'],
         'Dread label file differs from the EXP107 registration')
    fold_of, folds = {}, {}
    for k in range(5):
        f = p['folds'][str(k)]
        need(sha256_file(ROOT / f['ordered_ids']['path']) == f['ordered_ids']['sha256'], f'fold {k} ordered ids file changed')
        ordered = json.loads((ROOT / f['ordered_ids']['path']).read_text(encoding='utf-8'))
        need(uid_hash(ordered['train']) == f['ordered_ids']['train_sha256'] and uid_hash(ordered['val']) == f['ordered_ids']['val_sha256'],
             f'fold {k} ordered ids hash')
        for u in ordered['val']:
            need(u not in fold_of, 'validation sets overlap')
            fold_of[u] = k
    need(set(fold_of) == set(canonical), 'folds do not partition the canonical studies')
    for k in range(5):
        f = p['folds'][str(k)]
        tr = [u for u in canonical if fold_of[u] != k]
        va = [u for u in canonical if fold_of[u] == k]
        need(uid_hash(tr) == f['ordered_ids']['train_sha256'] and uid_hash(va) == f['ordered_ids']['val_sha256'],
             f'fold {k}: canonical-order derivation differs from the registered ordered ids')
        prev = np.clip(np.stack([labels[u] for u in tr]).mean(0), 0.03, 0.7)
        pw = np.clip((1 - prev) / prev, 1, 10).astype(np.float32)
        need(np.array_equal(pw, np.asarray(f['pos_weight'], np.float32)), f'fold {k} pos_weight')
        need(len(tr) // 8 == f['len_train_loader'] and len(tr) // 8 * 20 == f['steps'], f'fold {k} contract')
        folds[str(k)] = dict(n_train=len(tr), n_val=len(va), train_sha256=f['ordered_ids']['train_sha256'], val_sha256=f['ordered_ids']['val_sha256'],
                             pos_weight=[float(x) for x in f['pos_weight']], len_train_loader=f['len_train_loader'], steps=f['steps'])
    sorted_uids = sorted(canonical)
    position = {u: i for i, u in enumerate(canonical)}
    canon_rank = np.asarray([position[u] for u in sorted_uids], dtype='<i2')
    need([sorted_uids[i] for i in np.argsort(canon_rank, kind='stable')] == canonical, 'canonical rank round trip')
    dread_canon = np.stack([labels[u] for u in canonical]).astype(np.float32)
    dread_sha = sha256_bytes(np.ascontiguousarray(dread_canon, '<f4').tobytes())
    fold_sha = sha256_bytes(np.asarray([fold_of[u] for u in canonical], np.int8).tobytes())
    # ---- the compact meta.csv the user uploaded (local copy): same labels and folds
    meta = pd.read_csv(ROOT / META, dtype={UID: str})
    need(sorted(meta[UID]) == sorted_uids, 'meta.csv studies differ from the 4,343 canonical studies')
    meta = meta.set_index(UID).loc[sorted_uids]
    need(np.array_equal(meta.loc[canonical, LAB].to_numpy(dtype=np.float32), dread_canon), 'meta.csv Dread differs from the label file')
    need((meta['fold'].astype(int).to_numpy() == np.asarray([fold_of[u] for u in sorted_uids])).all(), 'meta.csv fold differs')
    ref_sha = {pfx: sha256_bytes(np.ascontiguousarray(meta[[f'{pfx}_{t}' for t in LAB]].to_numpy(dtype=np.int8)).tobytes())
               for pfx in ('flight', 'relabel')}
    # ---- components (one component never spans two folds)
    comps = json.loads((ROOT / COMPONENTS).read_text(encoding='utf-8'))
    comp_of = {u: c['component_id'] for c in comps for u in c['members']}
    need(set(comp_of) == set(sorted_uids) and len(comp_of) == 4343, 'component map does not cover the studies')
    cids = sorted(set(comp_of.values()))
    cindex = {c: i for i, c in enumerate(cids)}
    comp_idx = np.asarray([cindex[comp_of[u]] for u in sorted_uids], dtype='<i2')
    cross = pd.Series([fold_of[u] for u in sorted_uids]).groupby(comp_idx).nunique()
    need((cross == 1).all(), f'{int((cross != 1).sum())} components span two folds')
    # ---- Gold58 (uids only, never printed): absent from the studies
    g = pd.read_csv(ROOT / GOLD_SOURCE, dtype={UID: str}, usecols=[UID, 'ACL'])
    gold = set(g.loc[g['ACL'].notna(), UID].astype(str))
    need(len(gold) == 58 and not gold & set(sorted_uids), 'Gold58 count or overlap')
    gold_hashes = sorted(sha256_bytes(u.encode()) for u in gold)
    del g, gold
    # ---- corpus and weights
    cdir = ROOT / CORPUS
    need(sha256_file(cdir / 'manifest.json') == p['corpus_manifest_sha256'], 'corpus manifest differs from EXP107')
    man = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    need(man['ids_sha256'] == sha256_file(cdir / 'ids.npy') and man['masks_sha256'] == sha256_file(cdir / 'masks.npy'), 'corpus ids/masks')
    corpus_files = [[n, (cdir / n).stat().st_size, sha256_file(cdir / n)] for n in SMALL_CORPUS_FILES]
    corpus_files += [[s['file'], s['bytes'], s['sha256']] for s in man['shards']]
    for n, size, _ in corpus_files:
        need((cdir / n).stat().st_size == size, f'corpus file size {n}')
    wsha = sha256_file(ROOT / WEIGHTS)
    need(wsha == p['init_weights']['sha256'] and (ROOT / WEIGHTS).stat().st_size == p['init_weights']['bytes'], 'RadImageNet weights')
    inputs = dict(corpus_ref=CORPUS_REF, corpus_files=corpus_files,
                  weights=dict(ref=WEIGHTS_REF, name='ResNet50.pt', bytes=p['init_weights']['bytes'], sha256=wsha))
    # ---- stored EXP107 OOF (benchmark) as float16
    res = json.loads((ROOT / OOF_DIR / 'result.json').read_text(encoding='utf-8'))
    need(res['protocol_sha256'] == PROTOCOL_SHA256 and res['rows'] == 4343, 'EXP107 result.json')
    oof16, oof_sha, oof_src = {}, {}, {}
    for k, fname in ((36, 'oof_k36.csv'), (24, 'oof_k24.csv')):
        path = ROOT / OOF_DIR / fname
        oof_src[str(k)] = sha256_file(path)
        need(oof_src[str(k)] == res['outputs'][fname], f'{fname} differs from result.json')
        df = pd.read_csv(path, dtype={UID: str})
        need(list(df.columns) == [UID] + LAB and sorted(df[UID]) == sorted_uids, f'{fname} rows/columns')
        arr = df.set_index(UID).loc[sorted_uids, LAB].to_numpy(dtype=np.float32)
        b = np.ascontiguousarray(arr.astype('<f2')).tobytes()
        oof16[k], oof_sha[str(k)] = b, sha256_bytes(b)
    metrics = {key: dict(pooled=round(m['pooled']['macro12'], 6), mean_of_folds=round(m['mean_of_folds'], 6),
                         folds={f[4:]: round(m[f]['macro12'], 6) for f in m if f.startswith('fold')}) for key, m in res['metrics'].items()}
    emb = dict(
        schema='knee-colab-v2-embedded', notebook_version=VERSION, generator='tools/learning/make_colab_real.py',
        generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), n_studies=4343,
        sorted_uids_sha256=sha256_bytes('\n'.join(sorted_uids).encode()), canonical_sha256=uid_hash(canonical),
        canonical_rule='EXP084 ordered-ids train + val (EXP099 canonical order)', dread_f32_sha256=dread_sha,
        dread_rule='storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv via load_label_csv, float32 [4343,12] in canonical order',
        fold_sha256=fold_sha, fold_rule='int8 EXP099 validation fold per study in canonical order', folds=folds,
        flight_sha256=ref_sha['flight'], relabel_sha256=ref_sha['relabel'], reference_rule='int8 [4343,12] in sorted-uid order from meta.csv',
        components=dict(source=COMPONENTS, file_sha256=sha256_file(ROOT / COMPONENTS), n=len(cids),
                        idx_sha256=sha256_bytes(comp_idx.tobytes()), rule='index into sorted component_id, int16, sorted-uid order'),
        gold58_uid_sha256=gold_hashes, corpus_manifest_sha256=p['corpus_manifest_sha256'], weights_sha256=wsha,
        code_sha256={n: hashes['training/src/' + n] for n in p['training_code_sha256']},
        radimagenet_module_sha256=hashes['training/src/raptor_radimagenet.py'],
        tool_sha256={r: hashes[r] for r in ('tools/exp084_group_loss.py', 'tools/exp099_kfold.py', 'tools/exp101_kfold_arch.py')},
        exp107=dict(protocol=PROTOCOL, protocol_sha256=PROTOCOL_SHA256, initial_state_sha256=p['initial_state_sha256'],
                    initial_state_env=dict(torch=p['initial_state']['torch'], timm=p['initial_state']['timm']),
                    oof_source_sha256=oof_src, oof_f16_sha256=oof_sha, metrics=metrics,
                    note='Kaggle T4 trajectories (KNEE_REMOTE_RELAX_ENV); a benchmark, not a matched Colab control'),
        library_sha256=None,
    )
    return emb, inputs, hashes, canon_rank, comp_idx, oof16


def code_cell(src, i):
    return dict(cell_type='code', execution_count=None, id=f'c{i:02d}', metadata={}, outputs=[], source=src.splitlines(True))


def md_cell(src, i):
    return dict(cell_type='markdown', id=f'm{i:02d}', metadata={}, source=src.splitlines(True))


def build(out, force=False):
    emb, inputs, hashes, canon_rank, comp_idx, oof16 = build_embedded()
    hash_lines = '\n'.join(f'#   {rel}: {h}' for rel, h in sorted(hashes.items()))
    label_utils = pieces_text(LABEL_PIECES, hashes)
    lib = C.CELL_5_HEAD.replace('@@SOURCE_HASHES@@', hash_lines) + '\n' + pieces_text(LIB_PIECES, hashes)
    emb['library_sha256'] = sha256_bytes((label_utils + '\n' + lib).encode('utf-8'))
    cell_4a = (C.CELL_4A_HEAD.replace('@@EMB_JSON@@', json.dumps(emb, indent=1, ensure_ascii=False))
               + '# EXP099 canonical 순서: 정렬된 UID i번의 canonical 위치 (int16)\n' + blob_assign('CANON_RANK', canon_rank.tobytes(), '<i2')
               + '# component 번호 (정렬된 UID 순서, int16) — 같은 번호는 같은 component\n' + blob_assign('COMPONENT_IDX', comp_idx.tobytes(), '<i2')
               + '# 저장된 EXP107 OOF (Kaggle T4, float16, 정렬된 UID 순서 [4343, 12]) — 비교 기준(benchmark)\n'
               + 'EXP107_OOF16 = {}\n' + blob_assign('EXP107_OOF16[36]', oof16[36]) + blob_assign('EXP107_OOF16[24]', oof16[24])
               + "print('내장 데이터:', EMB['n_studies'], '검사 |', EMB['components']['n'], 'components | EXP107 pooled Dread K36',\n"
               + "      EMB['exp107']['metrics']['k36']['pooled'], '/ K24', EMB['exp107']['metrics']['k24']['pooled'])\n")
    cell_2 = C.CELL_2.replace('@@INPUTS_JSON@@', json.dumps(inputs, indent=1))
    cell_4 = C.CELL_4.replace('@@LABEL_UTILS@@', label_utils)
    cells = [md_cell(C.MD_TITLE, 0), md_cell(C.MD_0, 1), code_cell(C.CELL_0, 0), md_cell(C.MD_1, 2), code_cell(C.CELL_1, 1),
             md_cell(C.MD_2, 3), code_cell(cell_2, 2), md_cell(C.MD_3, 4), code_cell(C.CELL_3, 3), md_cell(C.MD_4, 5),
             code_cell(cell_4a, 4), code_cell(cell_4, 5), md_cell(C.MD_5, 6), code_cell(lib, 6), md_cell(C.MD_6, 7),
             code_cell(C.CELL_6, 7), md_cell(C.MD_7, 8), code_cell(C.CELL_7, 8)]
    for c in cells:
        text = ''.join(c['source'])
        need('@@' not in text, 'unfilled placeholder')
        if c['cell_type'] == 'code':
            compile(text, f"cell {c['id']}", 'exec')      # syntax check of every code cell
    nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'provenance': []}},
              nbformat=4, nbformat_minor=5)
    text = json.dumps(nb, ensure_ascii=False, indent=1)
    longest = max(len(line) for c in cells for line in c['source'])
    need(longest <= 4000, f'a notebook line has {longest} characters')
    need(len(text.encode('utf-8')) < 3 * 2**20, 'notebook is not < 3 MB')
    out = Path(out)
    if out.exists() and not force:
        raise SystemExit(f'refusing to overwrite {out} (use --force)')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    report = dict(out=str(out), bytes=out.stat().st_size, sha256=sha256_file(out), cells=len(cells), longest_line=longest,
                  library_sha256=emb['library_sha256'], dread_f32_sha256=emb['dread_f32_sha256'], fold_sha256=emb['fold_sha256'],
                  components=emb['components']['n'], exp107_pooled=emb['exp107']['metrics'])
    print(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--force', action='store_true', help='overwrite an existing notebook at --out')
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='replace')
    build(a.out, a.force)


if __name__ == '__main__':
    main()
