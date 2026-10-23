"""VAST_PARITY_3090_V1 v1.0.2 — immutable staging and two-part parity probe.

Changelog 1.0.2 (2026-10-09, B): torchvision pin compares the base version only (local wheel
reports 0.20.1 for a +cpu build; torchvision is not on the model path — timm/torch are, and stay exact).
Changelog 1.0.1 (2026-10-09, B): `stage --link-corpus` records the corpus shards in the inventory
without copying them (≈30 GB) — the Vast disk already holds a hash-identical copy (manifest f3afc9ab…),
which B symlinks to storage/cache/derived/RAPTOR64_336_C140_S02 on the remote before admission; the
remote inventory check still hashes every shard. Upload ran at ≈1.9 MB/s, so re-uploading would cost ≈5 h.
Changelog 1.0.0 (2026-10-09): frozen FP16/seed-gap gates, target-only labels,
per-study integer/tensor receipts, portable runner shim, pinned remote script,
strict SSH upload plan, complete-output verification. No implicit GPU/network
work: only local-infer and explicitly invoked internal run commands use CUDA.
Use `python -m tools.vast_parity_3090 --help` from the project/staged root.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

VERSION = '1.0.2'
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.local/vast-parity-3090-v1'
REG = 'artifacts/experiments/VAST_PARITY_3090_V1'
PROTOCOL = 'artifacts/experiments/EXP127_AUG1_PROTOCOL_V1/s42/manifest.json'
RAD = 'artifacts/experiments/EXP127_AUG1_S42_F0/model_best.pt'
COAT = 'artifacts/experiments/EXP084_B_G1/model_best.pt'
COAT_SHA = '478bdc01f8f6ce4fdeada6275188b5ad526a4ebf4f860a758322ee719964ff8c'
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
LABELS = {
    'dread': 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv',
    'steven_v4': 'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
    'pilkwang_v2': 'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv',
    'relabel_v1': 'storage/labels/derived_relabel/relabel_v1.csv',
}
PINS = {'torch': '2.5.1+cu121', 'torchvision': '0.20.1+cu121', 'timm': '1.0.19', 'numpy': '2.0.2'}
KNOWN_HOSTS = '.local/hc06-illinois-stable-20261009-v10/known_hosts-54941539-v1'
SSH_KEY = 'C:/Users/Taeyang/.ssh/id_ed25519'  # PATH ONLY; never opened
REMOTE = '/workspace/vast-parity-3090-v1'
ARMS = ('radr50_plain_k36', 'radr50_tta_k36', 'coatnet_k24')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def relative(value):
    """Only non-secret, contained, relative package paths are admissible."""
    s = str(value).replace('\\', '/')
    p = PurePosixPath(s)
    lower = s.lower()
    if (p.is_absolute() or re.match(r'^[a-zA-Z]:', s) or '..' in p.parts
            or any(x in lower for x in ('api-keys', '.cache/huggingface', '.ndatools',
                                       '.ssh', '.local/relabel', 'watcher-config'))
            or lower == 'storage/competition/train.csv'):
        raise ValueError('unsafe package path')
    return p.as_posix()


def contained(root, rel):
    p = Path(root) / relative(rel)
    if not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('package link escapes its root')
    return p


def remap_tree(obj, source_root):
    """Normalize only root-owned absolute paths. No arbitrary drive stripping."""
    prefix = str(source_root).replace('\\', '/').rstrip('/') + '/'
    if isinstance(obj, dict):
        return {remap_tree(k, source_root): remap_tree(v, source_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_tree(v, source_root) for v in obj]
    if isinstance(obj, str):
        s = obj.replace('\\', '/')
        if s.lower().startswith(prefix.lower()):
            return relative(s[len(prefix):])
        if re.match(r'^[a-zA-Z]:[/\\]', obj):
            raise ValueError('unmapped absolute Windows path in manifest')
    return obj


def safe_table(path, ids=None, prediction=False):
    import numpy as np
    import pandas as pd
    wanted = {UID, *TARGETS, *(t + '_soft' for t in TARGETS)}
    d = pd.read_csv(path, dtype={UID: str}, usecols=lambda c: c in wanted)
    cols = [t if t in d.columns else t + '_soft' for t in TARGETS]
    if not set(cols).issubset(d.columns) or d[UID].isna().any() or d[UID].duplicated().any():
        raise ValueError('invalid target-only table')
    if ids is None:
        return d[[UID, *cols]]
    if not set(ids).issubset(d[UID]) or (prediction and d[UID].tolist() != list(ids)):
        raise ValueError('UID membership/order mismatch')
    a = d.set_index(UID).loc[list(ids), cols].to_numpy(np.float64)
    if prediction and (not np.isfinite(a).all() or ((a < 0) | (a > 1)).any()):
        raise ValueError('nonfinite/out-of-range probabilities')
    return a


def references(root, ids):
    import numpy as np
    soft = {k: safe_table(Path(root) / v, ids) for k, v in LABELS.items()}
    dread = soft['dread']
    if not np.isfinite(dread).all() or ((dread < 0) | (dread > 1)).any():
        raise ValueError('incomplete Dread reference')
    x = np.stack(list(soft.values()))
    if ((x[np.isfinite(x)] < 0) | (x[np.isfinite(x)] > 1)).any():
        raise ValueError('invalid consensus source')
    b = x >= .5
    mask = np.isfinite(x).all(0) & (b.all(0) | (~b).all(0))
    return {'dread': (np.rint(dread.astype(np.float32)), np.ones(dread.shape, bool)),
            'consensus': (b[0].astype(float), mask)}


def auc(probs, ref):
    import numpy as np
    from scipy.stats import rankdata
    y, mask = ref
    if probs.shape != y.shape or not np.isfinite(probs).all():
        raise ValueError('invalid metric arrays')
    values = []
    for j in range(y.shape[1]):
        yy, pp = y[mask[:, j], j], probs[mask[:, j], j]
        n1, n0 = int((yy == 1).sum()), int((yy == 0).sum())
        if n1 and n0:
            values.append(float((rankdata(pp)[yy == 1].sum() - n1*(n1+1)/2)/(n1*n0)))
        else:
            values.append(None)
    valid = [v for v in values if v is not None]
    if not valid:
        raise ValueError('no evaluable AUC targets')
    return {'macro': float(np.mean(valid)), 'per_target': values, 'n_targets': len(valid),
            'mask_counts': mask.sum(0).tolist()}


def baseline(root, ordered):
    refs = references(root, ordered['val'])
    result = {}
    for seed in (42, 43):
        path = Path(root) / f'artifacts/experiments/EXP127_AUG1_S{seed}_F0/oof_k36.csv'
        pp = safe_table(path, ordered['val'], prediction=True)
        result[str(seed)] = {name: auc(pp, rr) for name, rr in refs.items()}
    limits = {name: abs(result['42'][name]['macro'] - result['43'][name]['macro']) for name in refs}
    frozen = read(Path(root) / REG / 'baseline.json')
    for name in refs:
        for seed in ('42', '43'):
            if abs(result[seed][name]['macro'] - frozen[seed][name]) > 1e-12:
                raise ValueError('frozen baseline does not reproduce: ' + name)
    return {'metrics': result, 'limits': limits}


def semantic_label_hash(path):
    """The frozen runner's target-only CSV semantic hash, without its imports."""
    table = safe_table(path)
    cols = [t if t in table.columns else t + '_soft' for t in TARGETS]
    table = table.dropna(subset=cols)
    return hashlib.sha256(table[[UID, *cols]].to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest()


def training_verdict(scores, baseline_result):
    delta = {k: abs(scores[k]['macro'] - baseline_result['metrics']['42'][k]['macro'])
             for k in ('dread', 'consensus')}
    passed = all(delta[k] <= baseline_result['limits'][k] for k in delta)
    return ('EQUIVALENT' if passed else 'NOT_EQUIVALENT'), delta


def code_closure(root, seeds):
    """Static local import closure, no project modules executed by staging."""
    root = Path(root)
    todo, seen = list(seeds), set()
    while todo:
        rel = relative(todo.pop())
        if rel in seen:
            continue
        path = contained(root, rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        seen.add(rel)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [v.name for v in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ''
                if n.level:
                    pkg = list(PurePosixPath(rel).parent.parts)
                    pkg = pkg[:len(pkg) - n.level + 1]
                    base = '.'.join([*pkg, base]).strip('.')
                names = [base, *[base + '.' + v.name for v in n.names]]
            for name in names:
                stem = name.replace('.', '/')
                for candidate in (stem + '.py', stem + '/__init__.py', 'training/src/' + stem + '.py'):
                    if (root / candidate).is_file():
                        todo.append(candidate)
        for parent in PurePosixPath(rel).parents:
            init = (parent / '__init__.py').as_posix()
            if (root / init).is_file() and init not in seen:
                todo.append(init)
    return seen


def corpus_files(root, ordered):
    import numpy as np
    c = Path(root) / CORPUS
    m = read(c / 'manifest.json')
    ids = np.load(c / 'ids.npy', allow_pickle=True).astype(str).tolist()
    rows = {u: i for i, u in enumerate(ids)}
    required = set(ordered['train'] + ordered['val'])
    if len(required) != 4343 or not required.issubset(rows):
        raise ValueError('training + full evaluation corpus roster differs')
    indices = {rows[u] for u in required}
    files = ['manifest.json', 'ids.npy', 'masks.npy']
    for s in m['shards']:
        if any(s['start'] <= i < s['stop'] for i in indices):
            files.append(relative(s['file']))
    return [CORPUS + '/' + x for x in files]


def copy_new(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copyfile(src, dest)


def stage_package(root, dest, initial_state=None, link_corpus=False):
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest.exists():
        raise FileExistsError('stage is immutable; choose a fresh --stage path')
    p = read(root / PROTOCOL)
    ordered = read(root / p['folds']['0']['ordered_ids']['path'])
    if (len(ordered['val']), len(ordered['train'])) != (888, 3455):
        raise ValueError('not EXP127 fold0')
    if sha(root / COAT) != COAT_SHA:
        raise ValueError('wrong B224 checkpoint')
    # Reproduce the two frozen seed baselines BEFORE any GPU work.
    metric_baseline = baseline(root, ordered)
    if initial_state is not None:
        ip = Path(initial_state).resolve()
        relative(ip.relative_to(root).as_posix())  # never inspect an arbitrary secret path
        from tools.vast_parity_3090_training import verify_initial
        verify_initial(ip, root / p['initial_state']['keys_file'])
    code = code_closure(root, [
        'tools/vast_parity_3090.py', 'tools/vast_parity_3090_training.py',
        'tools/parity_runtime.py', 'tools/parity_compare_arrays.py',
        'tools/exp138_flip_tta.py', 'tools/gpu_queue.py', 'tools/gpu_job.py',
        'training/runners/exp127_radr50_single.py', 'training/src/raptor_exp127.py',
        'training/src/raptor_common.py', 'training/src/raptor_data.py',
        'training/src/raptor_radimagenet.py'])
    needed = set(code) | set(p['source_bindings'])
    needed |= {'training/src/' + n for n in p['training_code_sha256']}
    needed |= {PROTOCOL, RAD, COAT, 'config/workspace.toml', p['initial_state']['keys_file'],
               p['init_weights']['local'], *LABELS.values(), *corpus_files(root, ordered)}
    needed |= {v for k, v in p['paths'].items() if k != 'corpus'}
    needed |= {p['folds']['0'][k]['path'] for k in ('ordered_ids', 'groups')}
    needed |= {REG + '/' + n for n in ('PROTOCOL.md', 'tol.json', 'baseline.json')}
    needed |= {f'artifacts/experiments/EXP127_AUG1_S{s}_F0/{n}'
               for s in (42, 43) for n in ('oof_k36.csv', 'summary.json', 'provenance.json')}
    # Validate bound bytes before copying; never print label rows or report fields.
    bindings = {**p['source_bindings'], **{'training/src/' + k: v for k, v in p['training_code_sha256'].items()}}
    bindings.update({CORPUS + '/manifest.json': p['corpus_manifest_sha256'],
                     p['init_weights']['local']: p['init_weights']['sha256']})
    bindings.update({p['folds']['0'][k]['path']: p['folds']['0'][k]['sha256'] for k in ('ordered_ids', 'groups')})
    for rel, expected in bindings.items():
        if sha(contained(root, rel)) != expected:
            raise ValueError('frozen binding changed: ' + rel)
    dest.mkdir(parents=True)
    rewritten, entries = {}, {}
    linked = set(corpus_files(root, ordered)) if link_corpus else set()
    for rel in sorted(needed):
        src, dst = contained(root, rel), contained(dest, rel)
        if rel in linked:  # v1.0.1: recorded, not copied; the remote copy is symlinked and re-hashed there
            entries[rel] = record(src)
            continue
        if rel.startswith('storage/labels/') and rel.endswith('.csv'):
            table = safe_table(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(dst, index=False, lineterminator='\n')
            rewritten[rel] = {'original_sha256': sha(src), 'staged_sha256': sha(dst)}
        else:
            copy_new(src, dst)
        entries[rel] = record(dst)
    normalized = remap_tree(p, root)
    if semantic_label_hash(dest / p['paths']['labels']) != p['label_set']['semantic_sha256']:
        raise ValueError('sanitized training labels changed the original semantic hash')
    write(dest / 'normalized-protocol.json', normalized)
    write(dest / 'ordered-ids.json', ordered)
    write(dest / 'infer-ids.json', ordered['val'][:300])
    write(dest / 'baseline-reproduced.json', metric_baseline)
    baseline(dest, ordered)  # serialization must not change either scoring reference
    init_rel = None
    if initial_state:
        init_rel = 'inputs/initial-state' + Path(initial_state).suffix
        copy_new(initial_state, dest / init_rel)
    cfg = {'version': VERSION, 'protocol': PROTOCOL, 'normalized_protocol': 'normalized-protocol.json',
           'source_root': str(root), 'initial_state': init_rel, 'workers': 12,
           'training_ready': init_rel is not None, 'rewritten_bindings': rewritten,
           'rad_checkpoint': RAD, 'coat_checkpoint': COAT, 'corpus_linked': bool(link_corpus),
           'blocked_reason': None if init_rel else 'Exact saved EXP127 S42 initial state file missing; never regenerate RNG.'}
    write(dest / 'vast-config.json', cfg)
    for rel in ('normalized-protocol.json', 'ordered-ids.json', 'infer-ids.json',
                'baseline-reproduced.json', 'vast-config.json', *([init_rel] if init_rel else [])):
        entries[rel] = record(dest / rel)
    corpus = {k: entries[k] for k in corpus_files(root, ordered)}
    write(dest / 'corpus-files.json', {'files': corpus, 'total_bytes': sum(v['bytes'] for v in corpus.values()),
          'train_studies': 3455, 'eval_studies': 888, 'inference_studies': 300,
          'note': 'Whole touched shards retained; no crop/reserialize/subset of frozen corpus.'})
    entries['corpus-files.json'] = record(dest / 'corpus-files.json')
    write(dest / 'inventory.json', {'version': VERSION, 'files': entries,
          'total_bytes': sum(v['bytes'] for v in entries.values())})
    remote_script(dest)
    print(json.dumps({'stage': str(dest), 'training_ready': cfg['training_ready'],
                      'corpus_bytes': sum(v['bytes'] for v in corpus.values()),
                      'stage_bytes': tree_bytes(dest), 'inventory_sha256': sha(dest / 'inventory.json')}))


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def verify_inventory(stage):
    inv = read(Path(stage) / 'inventory.json')
    for rel, expected in inv['files'].items():
        if record(contained(stage, rel)) != expected:
            raise ValueError('stage hash/size mismatch: ' + rel)
    return inv


def receipt(out):
    out = Path(out)
    files = {p.relative_to(out).as_posix(): record(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p != out / 'receipt.json'}
    write(out / 'receipt.json', {'status': 'COMPLETE', 'files': files, 'version': VERSION})


def verify_receipt(out):
    out = Path(out)
    rr = read(out / 'receipt.json')
    if rr.get('status') != 'COMPLETE' or not rr.get('files'):
        raise ValueError('missing completed receipt')
    actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p != out / 'receipt.json'}
    if actual != set(rr['files']):
        raise ValueError('receipt output roster changed')
    for rel, rec in rr['files'].items():
        if record(contained(out, rel)) != rec:
            raise ValueError('output receipt mismatch: ' + rel)


def pins_ok():
    got = {}
    for name in PINS:
        try:
            got[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            got[name] = None
    base = lambda v: v.split('+')[0] if isinstance(v, str) else v
    if any((base(got[k]) != base(v)) if k == 'torchvision' else (got[k] != v) for k, v in PINS.items()):
        raise ValueError('required runtime package pins differ: ' + json.dumps(got))
    return got


def validate_runtime(info, inference=False):
    """Missing or identically wrong receipts must not yield a parity PASS."""
    required = {'torch': PINS['torch'], 'cuda': '12.1', 'cudnn': 90100,
                'cudnn_benchmark': False, 'cudnn_deterministic': True,
                'matmul_tf32': False, 'cudnn_tf32': False,
                'flash_sdp': False, 'mem_efficient_sdp': True, 'math_sdp': True,
                'deterministic_algorithms': True, 'deterministic_warn_only': False,
                'torch_threads': 1}
    if any(k not in info or info[k] != value for k,value in required.items()):
        raise ValueError('runtime receipt violates registered pins/backends')
    for k in ('numpy','timm','torchvision'):
        got = info.get('libs', {}).get(k)
        if (str(got).split('+')[0] != PINS[k].split('+')[0]) if k == 'torchvision' else (got != PINS[k]):  # v1.0.2
            raise ValueError('runtime library pin mismatch: ' + k)
    policy = dict(amp='fp16', sdpa='efficient', strict=True, threads=1, seed=42,
                  deterministic_warn_only=False)
    if inference:
        policy['batch'] = 1
    if any(info.get('policy', {}).get(k) != value for k,value in policy.items()):
        raise ValueError('runtime policy receipt differs')


def validate_diagnostic_metadata(stage, diagnostic_dir):
    stage, dd = Path(stage), Path(diagnostic_dir)
    p, cfg = read(stage / PROTOCOL), read(stage / 'vast-config.json')
    registered = read(stage / p['initial_state']['keys_file'])
    validate_runtime(read(dd / 'runtime.json'))
    ii = read(dd / 'initial-state-receipt.json')
    if ii['combined_sha256'] != p['initial_state_sha256'] or ii['keys'] != registered['keys']:
        raise ValueError('diagnostic initial-state receipt differs')
    binding = read(dd / 'binding.json')
    if (binding.get('inventory_sha256') != sha(stage / 'inventory.json')
            or binding.get('initial_state_file_sha256') != sha(stage / cfg['initial_state'])
            or binding.get('normalized_protocol_sha256') != sha(stage / 'normalized-protocol.json')):
        raise ValueError('diagnostic source/initial-state binding differs')


def tensor_hash(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def integer_stages(corpus, uid, k):
    import numpy as np
    from raptor_common import select_centers
    vol, mask = corpus.volume(uid), corpus.mask(uid)
    centers = np.asarray(select_centers(mask, corpus.depth, k, corpus.spec, 'within-slot-valid', None), dtype='<i8')
    if vol.dtype != np.uint8 or mask.dtype.kind not in 'biu':
        raise ValueError('corpus integer stages have unexpected types')
    return {n: tensor_hash(a) for n, a in (('volume_uint8', vol), ('mask', mask), ('centers', centers))}


def check_integer(left, right):
    if set(left) != set(right):
        raise ValueError('input study/arm roster mismatch')
    mismatches = [k for k in left if left[k]['integer'] != right[k]['integer']]
    if mismatches:
        raise ValueError(f'integer-stage hash mismatch in {len(mismatches)} studies/arms')


def infer(stage, out, reference=None):
    """Identical entry on both sides; inference never reads validation labels."""
    stage, out = Path(stage).resolve(), Path(out).resolve()
    if Path(__file__).resolve() != stage / 'tools/vast_parity_3090.py':
        raise ValueError('run the staged module from the staged working directory')
    verify_inventory(stage)
    pins_ok()
    if reference:
        verify_receipt(reference)
        expected = read(Path(reference) / 'inputs.json')
    else:
        expected = None
    out.mkdir(parents=True, exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(stage / 'training/src'))
    from tools.parity_runtime import configure_parity, runtime_info
    configure_parity(amp='fp16', sdpa='efficient', strict=True, threads=1, seed=42)
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.version.cuda != '12.1' or torch.backends.cudnn.version() != 90100:
        raise ValueError('CUDA/cuDNN pin mismatch')
    info = runtime_info()
    info['deterministic_warn_only'] = torch.is_deterministic_algorithms_warn_only_enabled()
    info['policy'] = dict(amp='fp16', sdpa='efficient', flash=False, strict=True,
                          deterministic_warn_only=False, threads=1, seed=42, batch=1)
    validate_runtime(info, inference=True)
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    info['driver'] = driver.stdout.strip()
    write(out / 'runtime.json', info)
    from raptor_common import Corpus, load_checkpoint
    from raptor_data import StudyWindows
    from raptor_radimagenet import CenterSliceWindows
    # exp138 delegates exactly these plane-aware primitives to exp124.
    # Its public CLI is hard-coded for five multi-head folds; no monkeypatch.
    from tools.exp138_flip_tta import flip
    cfg, ids = read(stage / 'vast-config.json'), read(stage / 'infer-ids.json')
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError('expected the frozen first 300 ordered studies')
    required_keys = {f'{family}/{i:04d}' for family in ('radr50', 'coatnet') for i in range(300)}
    if expected is not None and set(expected) != required_keys:
        raise ValueError('reference input roster differs')
    corpus = Corpus(stage / CORPUS)
    zero = {u: np.zeros(12, np.float32) for u in ids}
    inputs, timings = {}, {}
    for family, ck, k, norm in [('radr50', cfg['rad_checkpoint'], 36, 'radimagenet'),
                               ('coatnet', cfg['coat_checkpoint'], 24, 'imagenet')]:
        start = time.monotonic()
        model, meta = load_checkpoint(stage / ck)
        if meta['res'] != 224 or list(meta['lab']) != TARGETS:
            raise ValueError('checkpoint resolution/target order mismatch')
        if family == 'radr50' and (meta.get('tag') != 'EXP127_AUG1_S42_F0' or meta.get('input_policy') != 'center3'):
            raise ValueError('wrong RadR50 checkpoint')
        model = model.cuda().eval()
        ds = (CenterSliceWindows(stage / CORPUS, ids, zero, 224, k, False, norm, 42,
                                 window_policy='within-slot-valid', input_policy='center3') if family == 'radr50'
              else StudyWindows(stage / CORPUS, ids, zero, 224, k, False, norm, 42,
                                window_policy='within-slot-valid'))
        plain, mirrored, logits = [], [], []
        ranges = flip.plane_ranges(corpus.spec)
        with torch.inference_mode():
            for i, uid in enumerate(ids):
                key = f'{family}/{i:04d}'
                integer = integer_stages(corpus, uid, k)
                # Stop before model execution for this study if integer bytes differ.
                if expected is not None and (key not in expected or expected[key]['integer'] != integer):
                    raise ValueError('integer-stage mismatch before forward: ' + key)
                x, _, got = ds[i]
                if got != uid or not torch.isfinite(x).all():
                    raise ValueError('invalid inference input')
                data = x.numpy()
                compact = data[:, :1] if family == 'radr50' else data
                if family == 'radr50' and not np.array_equal(data, np.repeat(compact, 3, axis=1)):
                    raise ValueError('center3 channel compression is not lossless')
                path = out / 'float-inputs' / (key + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, x=compact)
                inputs[key] = {'uid': uid, 'integer': integer, 'tensor_sha256': tensor_hash(data),
                               'shape': list(data.shape), 'float_file': path.relative_to(out).as_posix(),
                               'compact_channels': family == 'radr50'}
                if expected is not None:
                    with np.load(contained(reference, expected[key]['float_file']), allow_pickle=False) as z:
                        inputs[key]['float_max_abs_vs_local'] = float(np.max(np.abs(compact.astype(np.float64) - z['x'])))
                with torch.autocast('cuda', dtype=torch.float16):
                    lg = model(x[None].cuda())
                if not torch.isfinite(lg).all():
                    raise ValueError('nonfinite inference logits')
                logits.append(lg.float().cpu().numpy()[0])
                plain.append(torch.sigmoid(lg.float()).cpu().numpy()[0])
                if family == 'radr50':
                    centers = flip.select_centers(corpus.mask(uid), corpus.depth, k, corpus.spec, 'within-slot-valid', None)
                    xm = flip.mirror_windows(x, flip.flip_mask(centers, ranges))
                    inputs[key]['tta_tensor_sha256'] = tensor_hash(xm.numpy())
                    with torch.autocast('cuda', dtype=torch.float16):
                        lgm = model(xm[None].cuda())
                    if not torch.isfinite(lgm).all():
                        raise ValueError('nonfinite TTA logits')
                    mirrored.append(torch.sigmoid(lgm.float()).cpu().numpy()[0])
        pp = np.stack(plain)
        arm = 'radr50_plain_k36' if family == 'radr50' else 'coatnet_k24'
        np.savez(out / (arm + '.npz'), probs=pp, logits=np.stack(logits), ids=np.asarray(ids), targets=np.asarray(TARGETS))
        if family == 'radr50':
            np.savez(out / 'radr50_tta_k36.npz', probs=(pp + np.stack(mirrored))/2,
                     ids=np.asarray(ids), targets=np.asarray(TARGETS))
        timings[family] = time.monotonic() - start
        del model
        torch.cuda.empty_cache()
    write(out / 'inputs.json', inputs)
    write(out / 'run.json', {'version': VERSION, 'inventory_sha256': sha(stage / 'inventory.json'),
          'tol_sha256': sha(stage / REG / 'tol.json'), 'checkpoint_sha256': {k: sha(stage/v) for k,v in [('rad',RAD),('coat',COAT)]},
          'timings_seconds': timings, 'ids': ids, 'labels_read': False})
    receipt(out)


def float_report(local, remote, a, b):
    import numpy as np
    check_integer(a, b)
    maximum, changed = 0., 0
    for key in a:
        if a[key]['uid'] != b[key]['uid'] or a[key]['shape'] != b[key]['shape']:
            raise ValueError('input identity/shape mismatch')
        with np.load(contained(local, a[key]['float_file']), allow_pickle=False) as x, np.load(contained(remote, b[key]['float_file']), allow_pickle=False) as y:
            if x['x'].shape != y['x'].shape or not np.isfinite(x['x']).all() or not np.isfinite(y['x']).all():
                raise ValueError('invalid saved input tensor')
            for z, rec in ((x['x'], a[key]), (y['x'], b[key])):
                whole = np.repeat(z, 3, axis=1) if rec['compact_channels'] else z
                if list(whole.shape) != rec['shape'] or tensor_hash(whole) != rec['tensor_sha256']:
                    raise ValueError('saved float-input tensor/hash does not match receipt')
            maximum = max(maximum, float(np.max(np.abs(x['x'].astype(np.float64) - y['x']))))
        changed += a[key]['tensor_sha256'] != b[key]['tensor_sha256']
    return {'integer_matched': len(a), 'integer_total': len(a), 'float_max_abs': maximum,
            'float_hash_mismatches': changed}


def compare_runs(stage, local, remote, out, inference_only=False, local_diagnostic=None):
    import numpy as np
    from tools.parity_compare_arrays import compare as compare_arrays
    from tools.parity_runtime import diff_runtime
    stage, local, remote, out = map(Path, (stage, local, remote, out))
    verify_inventory(stage)
    tol = read(stage / REG / 'tol.json')
    verify_receipt(local)
    remote_infer = remote if inference_only else remote / 'infer'
    verify_receipt(remote_infer)
    for d in (local, remote_infer):
        r = read(d / 'run.json')
        if r['inventory_sha256'] != sha(stage / 'inventory.json') or r['tol_sha256'] != sha(stage / REG / 'tol.json'):
            raise ValueError('runs do not share the frozen stage/tolerances')
    ids = read(stage / 'infer-ids.json')
    li, ri = read(local / 'inputs.json'), read(remote_infer / 'inputs.json')
    expected_roster = {f'{family}/{i:04d}': uid for family in ('radr50','coatnet') for i,uid in enumerate(ids)}
    for inp in (li, ri):
        if set(inp) != set(expected_roster) or any(inp[k]['uid'] != uid for k,uid in expected_roster.items()):
            raise ValueError('missing/changed input-study roster')
    inputs = float_report(local, remote_infer, li, ri)
    refs = references(stage, ids)
    results = {}
    for arm in ARMS:
        with np.load(local / (arm + '.npz'), allow_pickle=False) as l, np.load(remote_infer / (arm + '.npz'), allow_pickle=False) as r:
            for z in (l, r):
                if z['ids'].tolist() != ids or z['targets'].tolist() != TARGETS or z['probs'].shape != (300, 12):
                    raise ValueError('prediction identity/targets/shape mismatch')
                if not np.isfinite(z['probs']).all() or ((z['probs'] < 0) | (z['probs'] > 1)).any():
                    raise ValueError('invalid probability output')
            numeric = compare_arrays(l['probs'], r['probs'], tol['probs'])
            # Existing helper ignores undefined Spearman columns; require all 12.
            from scipy.stats import spearmanr
            rank = [float(spearmanr(l['probs'][:, j], r['probs'][:, j]).statistic) for j in range(12)]
            rank_ok = all(np.isfinite(rank)) and min(rank) >= tol['probs']['min_spearman']
            al, ar = auc(l['probs'], refs['dread']), auc(r['probs'], refs['dread'])
            if al['n_targets'] != 12 or ar['n_targets'] != 12:
                raise ValueError('inference macro12 has an unevaluable target')
            delta = abs(ar['macro'] - al['macro'])
            passed = (numeric.get('pass', False) and rank_ok and delta <= tol['macro_auc_abs']
                      and inputs['float_max_abs'] <= tol['float_input_max_abs'])
            results[arm] = {'status': 'PASS' if passed else 'FAIL', 'numeric': numeric,
                            'local_auc': al, 'remote_auc': ar, 'macro_auc_abs_delta': delta}
    runtime = {'local': read(local / 'runtime.json'), 'remote': read(remote_infer / 'runtime.json')}
    for info in runtime.values():
        validate_runtime(info, inference=True)
    diff = diff_runtime(runtime['local'], runtime['remote'])
    invariant_keys = ('torch', 'cuda', 'cudnn', 'policy', 'cudnn_benchmark', 'cudnn_deterministic',
                      'matmul_tf32', 'cudnn_tf32', 'flash_sdp', 'mem_efficient_sdp', 'math_sdp')
    invariant_ok = all(runtime['local'].get(k) == runtime['remote'].get(k) for k in invariant_keys)
    for k in ('numpy', 'timm', 'torchvision'):
        invariant_ok &= runtime['local']['libs'].get(k) == runtime['remote']['libs'].get(k)
    part1 = 'PASS' if invariant_ok and all(x['status'] == 'PASS' for x in results.values()) else 'FAIL'
    training = {'status': 'NOT_RUN', 'rule': 'both absolute macro deltas <= frozen S42-S43 gaps'}
    if not inference_only:
        train = remote / 'training'
        if (train / 'receipt.json').exists():
            verify_receipt(train)
            run = train / 'EXP127_AUG1_S42_F0'
            s, prov = read(run / 'summary.json'), read(run / 'provenance.json')
            p = read(stage / PROTOCOL)
            if (s['selected_epoch_1based'], s['optimizer_updates'], s['train'], s['val']) != (20, 8620, 3455, 888):
                raise ValueError('Part2 incomplete training schedule')
            if prov['initial_state_sha256'] != p['initial_state_sha256'] or not prov['initial_state_matches_registered']:
                raise ValueError('Part2 initial state changed')
            if any(s['args'].get(k) != v for k, v in p['common_args'].items()) or s['args'].get('seed') != 42:
                raise ValueError('Part2 recipe changed')
            if prov['eval_ids_ordered_sha256'] != p['folds']['0']['ordered_ids']['val_sha256']:
                raise ValueError('Part2 full validation roster changed')
            expected_provenance = {
                'protocol_sha256': sha(stage / 'normalized-protocol.json'),
                'runner_sha256': sha(stage / 'training/runners/exp127_radr50_single.py'),
                'guard_sha256': sha(stage / 'tools/exp127_radr50_single.py'),
                'label_semantic_sha256': p['label_set']['semantic_sha256'],
                'corpus_manifest_sha256': p['corpus_manifest_sha256'],
                'train_ids_ordered_sha256': p['folds']['0']['ordered_ids']['train_sha256'],
                'code_sha256': p['training_code_sha256'],
                'actual_pos_weight': p['folds']['0']['pos_weight'],
                'checkpoint_selection': 'fixed-final', 'fold': 0, 'seed': 42,
                'k_eval': 24, 'k_eval_extra': 36,
            }
            if any(prov.get(k) != val for k,val in expected_provenance.items()):
                raise ValueError('Part2 frozen source/input/initialization provenance mismatch')
            if not 0 <= s['skipped_updates'] <= 86 or s['selected_checkpoint'] != 'fixed-final':
                raise ValueError('Part2 completion/skip contract failed')
            train_ids = read(stage / 'ordered-ids.json')['val']
            pp = safe_table(run / 'oof_k36.csv', train_ids, prediction=True)
            bb = read(stage / 'baseline-reproduced.json')
            scores = {k: auc(pp, v) for k, v in references(stage, train_ids).items()}
            verdict, delta = training_verdict(scores, bb)
            if any(scores[k]['n_targets'] != bb['metrics']['42'][k]['n_targets'] for k in scores):
                raise ValueError('training comparison target set changed')
            training = {'status': verdict,
                        'scores': scores, 'absolute_deltas': delta, 'limits': bb['limits'],
                        'note': 'One seed gap is not a noise distribution or permission for cross-GPU paired screens.'}
            diag = Path(local_diagnostic) if local_diagnostic else local.parent / 'local-diagnostic'
            if not (diag / 'receipt.json').exists():
                training['status'] = 'FAIL_MISSING_LOCAL_DIAGNOSTIC'
            else:
                verify_receipt(diag)
                dl, dr = diag / 'diagnostic32', train / 'diagnostic32'
                for dd in (dl, dr):
                    validate_diagnostic_metadata(stage, dd)
                il, ir = read(dl / 'inputs.json'), read(dr / 'inputs.json')
                if len(il) != 32 or len(ir) != 32:
                    raise ValueError('expected 32 diagnostic updates on each side')
                for lstep, rstep in zip(il, ir):
                    for key in ('step', 'uids_sha256', 'y_sha256', 'integer_studies'):
                        if key not in lstep or lstep[key] != rstep.get(key):
                            raise ValueError('training diagnostic integer/order/label mismatch')
                with np.load(dl / 'train32.npz', allow_pickle=False) as l, np.load(dr / 'train32.npz', allow_pickle=False) as r:
                    training['diagnostic_report_only'] = {key: compare_arrays(l[key], r[key], {}) for key in ('loss','grad_norm','scale')}
    report = {'version': VERSION, 'part1': part1, 'part2': training, 'arms': results,
              'inputs': inputs, 'runtime': runtime, 'runtime_diff': diff,
              'runtime_invariants_match': bool(invariant_ok), 'tol_sha256': sha(stage / REG / 'tol.json')}
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'parity-report.json', report)
    lines = ['# Parity results: VAST_PARITY_3090_V1', '',
             f'- Environment: dev={runtime["local"].get("gpu")} / ref={runtime["remote"].get("gpu")}; runtime diff: see JSON.',
             '- Policy: fp16 autocast; SDPA efficient, flash off; hard strict; TF32 off; threads=1; batch=1.',
             '- Sample: first 300 EXP127 ordered V888 fold0 studies; no synthetic studies.',
             f'- Preprocessing: integer {inputs["integer_matched"]}/{inputs["integer_total"]}; float max {inputs["float_max_abs"]:.9g} (limit {tol["float_input_max_abs"]}).',
             f'- Frozen tolerance SHA256: {report["tol_sha256"]}', '']
    for arm, r in results.items():
        n = r['numeric']
        lines += [f'## {arm}', f'- Output: max {n.get("max_abs")} / p99 {n.get("p99_abs")} / mean {n.get("mean_abs")} / Spearman min {n.get("min_spearman")} -> {r["status"]}',
                  f'- Final: Dread macro12 dev {r["local_auc"]["macro"]:.12f} / ref {r["remote_auc"]["macro"]:.12f}; |delta| {r["macro_auc_abs_delta"]:.9g} (limit 1e-4).', '']
    lines += [f'- Decision: {"PASS_FINAL" if part1 == "PASS" else "FAIL"}; Part2 {training["status"]}.',
              '- Limits: weak, exposed validation labels; no hidden-test, clinical-quality or cross-device trajectory guarantee.',
              '- Failure policy: Vast only for complete same-GPU screens or standalone models; no ratio/LB retuning.']
    (out / 'parity-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(out)
    print(json.dumps({'part1': part1, 'part2': training['status'], 'report': str(out)}))
    return 0 if part1 == 'PASS' and (inference_only or training['status'] == 'EQUIVALENT') else 1


def remote_script(stage):
    stage = Path(stage)
    # Bootstrap is pure CPU. This script never starts/stops/rents an instance.
    script = '''#!/usr/bin/env bash
set -euo pipefail
cd /workspace/vast-parity-3090-v1
export PYTHONUTF8=1 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1
test "$#" -eq 1 || { echo "usage: bash remote_run.sh reviewed-admission.json" >&2; exit 2; }
test -f "$1"
test -f reference/receipt.json
test ! -e remote.once
python3 -m venv --system-site-packages .venv-parity
PY="$PWD/.venv-parity/bin/python"
if ! "$PY" -m tools.vast_parity_3090 _pins; then
  "$PY" -m pip install --upgrade --force-reinstall torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
  "$PY" -m pip install --upgrade timm==1.0.19 numpy==2.0.2
fi
"$PY" -m pip install scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 pydicom==3.0.2 opencv-python-headless==4.10.0.84
"$PY" -m tools.vast_parity_3090 _pins
"$PY" -m tools.vast_parity_3090 _admit --admission "$1"
"$PY" -m pip freeze > environment-pip-freeze.txt
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu --format=csv > environment-gpu.csv
"$PY" tools/gpu_queue.py remote-queue.json --foreground
test -f remote/receipt.json
sha256sum remote/receipt.json > remote-receipt.sha256
'''
    path = stage / 'remote_run.sh'
    if path.exists():
        if path.read_text(encoding='utf-8') != script:
            raise ValueError('existing remote script differs; use a fresh stage')
    else:
        path.write_text(script, encoding='utf-8', newline='\n')
    q = [{'tag': 'VAST_PARITY_3090_V1', 'continue_on_failure': False,
          'command': [REMOTE + '/.venv-parity/bin/python', '-m', 'tools.vast_parity_3090', '_remote-run']}]
    if not (stage / 'remote-queue.json').exists():
        write(stage / 'remote-queue.json', q)
    write_if_equal(stage / 'launch-files.json', {n: record(stage / n) for n in ('remote_run.sh', 'remote-queue.json')})
    return path


def write_if_equal(path, obj):
    if Path(path).exists():
        if read(path) != obj:
            raise ValueError('immutable generated file differs')
    else:
        write(path, obj)


def admit(stage, admission):
    stage = Path(stage)
    verify_inventory(stage)
    for rel, rec in read(stage / 'launch-files.json').items():
        if record(stage / rel) != rec:
            raise ValueError('launch file changed')
    a = read(admission)
    if (a.get('status') != 'ADMITTED' or a.get('instance_id') != 54941539
            or a.get('inventory_sha256') != sha(stage / 'inventory.json')
            or a.get('launch_files_sha256') != sha(stage / 'launch-files.json')
            or not a.get('owner') or a.get('max_gpu_minutes', 0) <= 0
            or not time.time() < a.get('expires_unix', 0)
            or not a.get('budget_within_existing_authorization')
            or not a.get('reviewed_endpoint') or not a.get('reviewed_trust_sha256')):
        raise ValueError('fresh reviewed source/input/environment/trust/budget admission required')
    cfg = read(stage / 'vast-config.json')
    if not cfg['initial_state']:
        raise ValueError('Part2 blocked: exact saved original initial state FILE is missing')
    from tools.vast_parity_3090_training import verify_initial
    p = read(stage / cfg['protocol'])
    verify_initial(contained(stage, cfg['initial_state']), contained(stage, p['initial_state']['keys_file']))
    verify_receipt(stage / 'reference')
    verify_receipt(stage / 'reference-diagnostic')
    validate_diagnostic_metadata(stage, stage / 'reference-diagnostic/diagnostic32')
    if read(stage / 'reference/run.json')['inventory_sha256'] != sha(stage / 'inventory.json'):
        raise ValueError('reference comes from another package')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, check=True)
    if gpu.stdout.strip().splitlines() != ['NVIDIA GeForce RTX 3090']:
        raise ValueError('expected exactly one RTX 3090')
    ps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, check=True)
    if ps.stdout.strip():
        raise ValueError('GPU is not idle')
    write(stage / 'remote.once', {'admission_sha256': sha(admission), 'accepted_unix': time.time(),
                                'max_gpu_minutes': a['max_gpu_minutes']})


def remote_run(stage):
    stage = Path(stage).resolve()
    once = read(stage / 'remote.once')
    # Admission must be fresh even if queue was delayed. No replays of receipts.
    if time.time() - once['accepted_unix'] > 300:
        raise ValueError('admission expired before queue launch')
    out = stage / 'remote'
    out.mkdir(exist_ok=False)
    for name in ('environment-pip-freeze.txt', 'environment-gpu.csv', 'remote.once'):
        copy_new(stage / name, out / name)
    deadline = time.monotonic() + once['max_gpu_minutes']*60
    commands = [['_infer', '--out', str(out / 'infer'), '--reference', str(stage / 'reference')],
                ['compare', '--local', str(stage / 'reference'), '--remote', str(out / 'infer'),
                 '--out', str(out / 'inference-comparison'), '--inference-only'],
                ['_train', '--out', str(out / 'training')]]
    try:
        for cmd in commands:
            bounded_run([sys.executable, '-m', 'tools.vast_parity_3090', *cmd, '--stage', str(stage)],
                        stage, max(1, deadline-time.monotonic()))
        write(out / 'complete.json', {'status': 'COMPLETED_NOT_YET_TRAINING_EQUIVALENT', 'version': VERSION})
    except BaseException as exc:
        write(out / 'failed.json', {'status': 'FAILED', 'reason': str(exc), 'version': VERSION})
        raise
    finally:
        receipt(out)


def bounded_run(command, cwd, seconds):
    """Linux remote timeout kills only this child process group, including loaders."""
    if os.name != 'posix':
        raise ValueError('bounded remote wrapper requires Linux')
    proc = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        result = proc.wait(timeout=seconds)
        if result:
            raise subprocess.CalledProcessError(result, command)
    except BaseException:
        # Workers can outlive a failed parent, so terminate the whole owned group.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise


def upload_plan(stage, host, port, known_hosts=KNOWN_HOSTS):
    stage = Path(stage).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host) or not 1 <= port <= 65535:
        raise ValueError('invalid SSH endpoint')
    trust = Path(known_hosts)
    if not trust.is_absolute():
        trust = ROOT / trust
    # Trust contents/private key never opened. B binds live endpoint to reviewed file.
    opts = f'-i "{SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="{trust.as_posix()}"'
    target = f'root@{host}'
    commands = [
        f'ssh {opts} -p {port} {target} "test ! -e {REMOTE} && mkdir -p {REMOTE}"',
        f'scp {opts} -P {port} -r "{stage.as_posix()}/." {target}:{REMOTE}/',
        f'scp {opts} -P {port} -r "{(WORK / "local").as_posix()}" {target}:{REMOTE}/reference',
        f'scp {opts} -P {port} -r "{(WORK / "local-diagnostic").as_posix()}" {target}:{REMOTE}/reference-diagnostic',
        f'scp {opts} -P {port} "{(WORK / "reviewed-admission.json").as_posix()}" {target}:{REMOTE}/reviewed-admission.json',
        f'ssh {opts} -p {port} {target} "cd {REMOTE} && bash remote_run.sh reviewed-admission.json"',
        f'scp {opts} -P {port} -r {target}:{REMOTE}/remote "{(WORK / "remote").as_posix()}"',
    ]
    # rsync is an alternative in Git Bash/WSL with an installed client, not extra upload.
    posix_opts = f'-i {shlex.quote(SSH_KEY)} -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile={shlex.quote(trust.as_posix())} -p {port}'
    rsync = f'rsync -rt --checksum --partial -e {shlex.quote("ssh " + posix_opts)} {shlex.quote(stage.as_posix()+"/")} {target}:{REMOTE}/'
    print('PRINT ONLY. After Start, B must re-read endpoint/host key, budget, ownership and GPU/PIDs.\n'
          'Changed endpoint/key requires a NEW reviewed trust file; never bypass checking or edit the frozen one.\n'
          'Create fresh reviewed-admission.json only after live checks; never reuse HC06 receipts.\n')
    print('\n'.join(commands))
    print('\n# Alternative to stage scp (Git Bash/WSL; adjust local drive spelling):\n' + rsync)
    print(json.dumps({'stage_upload_bytes': tree_bytes(stage),
                      'reference_upload_bytes': tree_bytes(WORK / 'local') if (WORK / 'local').exists() else None,
                      'diagnostic_upload_bytes': tree_bytes(WORK / 'local-diagnostic') if (WORK / 'local-diagnostic').exists() else None,
                      'note': 'Total additionally includes reviewed-admission.json; initial state and local reference may still be missing.'}))
    return commands


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('stage', 'local-infer', 'local-diagnostic', 'remote-script', 'upload-plan', 'compare', '_infer', '_train', '_admit', '_remote-run', '_pins'):
        p = sub.add_parser(name)
        p.add_argument('--stage', type=Path, default=WORK / 'stage' if name in ('stage','local-infer','local-diagnostic','remote-script','upload-plan','compare') else ROOT)
        if name == 'stage':
            p.add_argument('--initial-state', type=Path)
            p.add_argument('--link-corpus', action='store_true')
        if name in ('_infer', '_train'):
            p.add_argument('--out', type=Path, required=True)
        if name == '_infer':
            p.add_argument('--reference', type=Path)
        if name == '_train':
            p.add_argument('--diagnostic-only', action='store_true')
        if name == '_admit':
            p.add_argument('--admission', required=True)
        if name == 'upload-plan':
            p.add_argument('--host', required=True)
            p.add_argument('--port', required=True, type=int)
            p.add_argument('--known-hosts', default=KNOWN_HOSTS)
        if name == 'compare':
            p.add_argument('--local', type=Path, default=WORK / 'local')
            p.add_argument('--remote', type=Path, default=WORK / 'remote')
            p.add_argument('--out', type=Path, default=WORK / 'comparison')
            p.add_argument('--inference-only', action='store_true')
            p.add_argument('--local-diagnostic', type=Path, default=WORK / 'local-diagnostic')
    a = ap.parse_args(argv)
    if a.command == 'stage':
        stage_package(ROOT, a.stage, a.initial_state, a.link_corpus)
    elif a.command == 'remote-script':
        print(remote_script(a.stage))
    elif a.command == 'upload-plan':
        upload_plan(a.stage, a.host, a.port, a.known_hosts)
    elif a.command == 'local-infer':
        verify_inventory(a.stage)
        # B schedules this outer command through gpu_queue -> gpu_job.
        subprocess.run([sys.executable, '-m', 'tools.vast_parity_3090', '_infer', '--stage', str(a.stage.resolve()),
                        '--out', str(WORK / 'local')], cwd=a.stage, check=True)
    elif a.command == 'local-diagnostic':
        verify_inventory(a.stage)
        subprocess.run([sys.executable, '-m', 'tools.vast_parity_3090', '_train', '--stage', str(a.stage.resolve()),
                        '--out', str(WORK / 'local-diagnostic'), '--diagnostic-only'], cwd=a.stage, check=True)
    elif a.command == '_infer':
        infer(a.stage, a.out, a.reference)
    elif a.command == '_train':
        from tools.vast_parity_3090_training import run_training
        verify_inventory(a.stage)
        run_training(a.stage, a.out, diagnostic_only=a.diagnostic_only)
        receipt(a.out)
    elif a.command == 'compare':
        try:
            return compare_runs(a.stage, a.local, a.remote, a.out, a.inference_only, a.local_diagnostic)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            if not a.out.exists():
                a.out.mkdir(parents=True)
                write(a.out / 'parity-report.json', {'part1': 'FAIL', 'part2': {'status': 'NOT_VALIDATED'},
                      'structural_error': str(exc), 'version': VERSION})
                (a.out / 'parity-report.md').write_text('# Parity results: FAIL\n\nStructural validation failed: '+str(exc)+'\n', encoding='utf-8')
                receipt(a.out)
            raise
    elif a.command == '_pins':
        pins_ok()
    elif a.command == '_admit':
        admit(a.stage, a.admission)
    elif a.command == '_remote-run':
        remote_run(a.stage)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f'VAST_PARITY_3090_V1 blocked: {exc}', file=sys.stderr)
        raise SystemExit(2)
