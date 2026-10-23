"""Seven-lens completion v1.0.1 for H-B-155, H-B-158 and H-B-162 (2026-10-15).

Changelog 1.0.1: HB162 CPU readout may explicitly accept its authenticated
aggregate-only memory breach; preserve DO_NOT_SUBMIT and write both tables.
Changelog 1.0.0: CPU readouts on existing predictions (H-B-155 L6 fold-2 Q7S
S-replacement; H-B-158 L1-L6 from its authenticated GPU journals; prepared
H-B-162 readout), plus the panel-inference worker used by the single
FORENSICS_LENSES_1015 gpu_job (tools/forensics_1015_v1.py v1.1.0 controller):
Gold58 + OAI MOAKS panels through each item's own serving path.

Never prints label rows (aggregate metrics only). No training, release edits,
network or submission. Lenses are evidence, not gates (COMMON.md); every
ineligible lens carries a written reason. The historical H105 aligned reader
installs CPU tripwires on import, so it is imported only inside CPU readout
functions after every torch.load in that process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'training/src'))
from tools import hb145_output_io as io

VERSION = '1.0.1'
OUT = ROOT/'.local/scratch/forensics-1015/v110'
LWORK = OUT/'lenses'
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
MIL_MEMBERS = (('control', 42), ('control', 43), ('crop', 42), ('crop', 43))
HB162_KEYS = ('ADAMW42', 'SAM42', 'ADAMW43', 'SAM43')
PARENT_KEY = 'EXP181_CONTROL'
PANELS = ('gold58', 'oai557')
VIEWS = (320, 384)
PANEL_TARGETS = (0, 2, 3, 4, 5, 6, 7, 8, 9)  # MOAKS-mapped nine (hb122 convention)
POLICY = dict(
    l6_primary='Q7S deployed composition = H105 "replace" bundle (R50, C4, S); matching member replaced',
    l6_secondary='H105 "add" bundle (R50, C4, T, S): H-B-122/123 precedent, descriptive only',
    l6_coatnet='own/partner 50:50 rank(Q7S fold slice)/rank(own CoAtNet member), multi_lens coatnet recipe',
    l6_threshold=.0005, l7_oai_threshold=.002, draws=300, seed=20261003,
    mil_serving='hb155 fit evaluation path: crops.Windows(MILWindows K36) + evaluate_with tta=True, fp16 strict',
    release_serving='hb158 authenticated release definitions, K94, anatomical mirror, chunk16, fp16, TF32 on',
    hb155_oai='INELIGIBLE: no OAI native448/compartment-crop cache exists; crop serving path cannot run on OAI',
    gold58='EXPOSED descriptive (label development, epoch selection, prior research)',
    clean_oof=False, exposure='EXPOSED', training=False, submission_authorized=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


SOURCES = ('tools/lens_completion_1015_v1.py', 'tests/test_lens_completion_1015_v1.py',
           'tools/multi_lens_readout.py', 'tools/multi_lens_readout_l4_v2.py',
           'tools/hb105_aligned_oof_v1.py', 'tools/oai_moaks_panel.py', 'tools/exp130_gold58_compare.py',
           'tools/hb155_build_v1.py', 'tools/hb155_fit_v1.py', 'tools/hb155_readout_v1.py',
           'training/arms/hb155_crops_v1.py', 'training/arms/native_resolution_v2.py',
           'training/arms/mil_screen_hooks_v3.py', 'training/arms/mil.py',
           'tools/hb158_scale_contrast_v1.py', 'tools/hb162_sam_f0_v1.py',
           'training/arms/coatnet_history.py', 'tools/hb145_output_io.py')


def source_bindings():
    return {p: sha(ROOT/p) for p in SOURCES}


# ---------------------------------------------------------------- panels

def gold_panel():
    from tools.exp130_gold58_compare import official_gold
    import numpy as np
    ids, y = official_gold(ROOT)
    y = np.asarray(y, dtype=float)
    need(len(ids) == 58 == len(set(ids)) and y.shape == (58, 12) and np.isin(y, (0, 1)).all(),
         'official Gold58 contract')
    return list(ids), y


def oai_panel():
    from tools import oai_moaks_panel as p
    ids = p.validate()
    y = p.outcomes(ids)
    need(len(ids) == 557 and y.shape == (557, 12), 'OAI MOAKS panel contract')
    return list(ids), y


def counts(y):
    import numpy as np
    y = np.asarray(y, dtype=float)
    return {str(j): dict(positive=int((y[:, j] == 1).sum()), negative=int((y[:, j] == 0).sum()),
                         unknown=int(np.isnan(y[:, j]).sum())) for j in range(y.shape[1])}


# ---------------------------------------------------------------- H-B-155 members

def hb155_review():
    from tools import hb155_build_v1 as b
    return b.WORK/'review.json'


def mil_member(arm, seed):
    """Authenticated fold-2 fit, exactly the registered readout's checks."""
    from tools import hb155_build_v1 as b
    from tools import hb155_readout_v1 as r
    cp = b.config_path(arm, seed)
    c = b.require_fit(cp, hb155_review())
    d = b.WORK/'fits'/b.member(arm, seed)
    need(b.read(d/'heartbeat.json').get('status') == 'complete', 'H-B-155 fit incomplete')
    summary, receipt = b.read(d/'summary.json'), b.read(d/'val.receipt.json')
    b.require_runtime(cp, b.read(d/'runtime.json'))
    r.validate_records(c, summary, receipt, config_hash=b.sha(cp), csv_hash=b.sha(d/'val.csv'),
                       checkpoint_hash=b.sha(d/'final.pt'), prepare_hash=b.sha(b.WORK/'prepared.json'),
                       runtime_hash=b.sha(d/'runtime.json'))
    return dict(key=f'{arm}-s{seed}', arm=arm, seed=seed, config=c, config_path=rel(cp),
                checkpoint=rel(d/'final.pt'), checkpoint_sha256=b.sha(d/'final.pt'),
                prediction=rel(d/'val.csv'), prediction_sha256=b.sha(d/'val.csv'),
                corpus=c['corpus'], corpus_manifest_sha256=b.sha(ROOT/c['corpus']/'manifest.json'))


def mil_members():
    return {f'{a}-s{s}': mil_member(a, s) for a, s in MIL_MEMBERS}


def mil_predictions(member):
    """Registered readout loading (pandas float64), exact roster order."""
    import pandas as pd
    c = member['config']
    frame = pd.read_csv(ROOT/member['prediction'])
    need(frame['StudyInstanceUID'].astype(str).tolist() == c['val'], 'H-B-155 CSV order differs')
    need(sha(ROOT/member['prediction']) == member['prediction_sha256'], 'H-B-155 CSV changed')
    return frame[c['recipe']['targets']].to_numpy(dtype=float)


# ---------------------------------------------------------------- release models

def hb162_terminal_state():
    from tools.job_runtime import job_status
    return job_status(ROOT/'artifacts/logs', 'HB162_SAM_F0')['state']


def hb162_outputs():
    """Only after HB162_SAM_F0 is no longer running (never read live output dirs)."""
    state = hb162_terminal_state()
    if state != 'SUCCESS':
        return None, f'HB162_SAM_F0 job state {state}; continuation checkpoints unavailable'
    from tools import hb162_sam_f0_v1 as sam
    try:
        output = sam.completed('full')
    except (ValueError, OSError, KeyError) as exc:
        return None, f'HB162 completed-run authentication failed: {exc}'
    return output, None


def release_models(hb162_output=None):
    from tools import hb158_scale_contrast_v1 as s
    models = {PARENT_KEY: dict(key=PARENT_KEY, checkpoint=rel(s.CHECKPOINT), checkpoint_sha256=s.MODEL_SHA,
                               views=list(VIEWS), kind='parent')}
    if hb162_output is not None:
        from tools import hb162_sam_f0_v1 as sam
        for key in HB162_KEYS:
            path = sam.WORK/'gpu/full'/key/'model_best.pt'
            models[key] = dict(key=key, checkpoint=rel(path), views=[320], kind='continuation',
                               checkpoint_sha256=hb162_output['artifacts'][f'{key}/model_best.pt'])
    return models


# ---------------------------------------------------------------- registration

def oai_ancestry():
    """EXP181 CONTROL lineage: the panel's own OAI-free admission check."""
    from tools import oai_moaks_panel as p
    try:
        recipe = p.inspect_candidate(PARENT_KEY)
        return dict(eligible=True, ancestry=recipe['ancestry'], evidence=recipe['evidence'],
                    checkpoint_sha256=recipe['sha256'])
    except (ValueError, KeyError, OSError) as exc:
        return dict(eligible=False, ancestry=f'inspect_candidate failed: {exc}')


def registration():
    from tools import hb158_scale_contrast_v1 as s
    from tools import oai_moaks_panel as p
    gold, gy = gold_panel()
    oai, oy = oai_panel()
    members = mil_members()
    _, proof = s.serving(320)
    return dict(version=VERSION, policy=POLICY, sources=source_bindings(),
        gold58=dict(ids_sha256=uid_hash(gold), n=len(gold), label_counts=counts(gy)),
        oai557=dict(ids_sha256=uid_hash(oai), n=len(oai), label_counts=counts(oy),
                    registration_sha256=sha(p.OUT/'registration.json'),
                    cache96_manifest_sha256=sha(ROOT/p.CACHES['96']/'manifest.json')),
        hb155_members={k: {n: v for n, v in m.items() if n != 'config'} for k, m in members.items()},
        hb155_review_sha256=sha(hb155_review()),
        release=dict(parent=PARENT_KEY, checkpoint_sha256=s.MODEL_SHA, serving=proof,
                     corpus_manifest_sha256=sha(s.CORPUS/'manifest.json'),
                     hb162_tool_sha256=sha(ROOT/'tools/hb162_sam_f0_v1.py'),
                     hb162_registration_sha256=sha(ROOT/'.local/scratch/hb162/H-B-162/registration.json')),
        oai_ancestry=oai_ancestry())


def registered():
    saved = io.read_json(LWORK/'registration.json')
    need(saved == json.loads(json.dumps(registration())), 'lens registration stale; re-register and repeat smoke/review')
    return saved


# ---------------------------------------------------------------- GPU/CPU panel plan

def plan(phase, hb162_available):
    """Registered task list. Calibration = first two studies of each panel."""
    need(phase in ('mil-calibration', 'mil-full', 'release-calibration', 'release-full', 'mil-cpu', 'release-cpu'),
         'unknown lens phase')
    small = not phase.endswith('full')
    tasks = []
    if phase.startswith('mil'):
        for arm, seed in MIL_MEMBERS:
            tasks.append(dict(family='mil', key=f'{arm}-s{seed}', view=224, panel='gold58', n=2 if small else 58))
            if small:
                # Cross-GPU parity: two fold-2 validation studies vs the 4090 val.csv rows.
                tasks.append(dict(family='mil', key=f'{arm}-s{seed}', view=224, panel='fold2val2', n=2))
    else:
        for view in VIEWS:
            for panel in PANELS:
                tasks.append(dict(family='release', key=PARENT_KEY, view=view, panel=panel,
                                  n=2 if small else (58 if panel == 'gold58' else 557)))
        if hb162_available:
            keys = ('SAM42',) if small else HB162_KEYS
            for key in keys:
                for panel in PANELS:
                    tasks.append(dict(family='release', key=key, view=320, panel=panel,
                                      n=2 if small else (58 if panel == 'gold58' else 557)))
    return tasks


def task_name(task):
    return f"{task['key']}-{task['view']}-{task['panel']}.json"


class Status:
    """Committed-unit status for the supervisor's idle watchdog (status file only)."""
    def __init__(self, folder, total):
        self.folder, self.total, self.done = Path(folder), total, 0

    def tick(self, phase='inference'):
        self.done += 1
        try:
            io.receipt(self.folder/'status.json', dict(phase=phase, completed=self.done, total=self.total))
        except OSError:
            pass  # heartbeat failures are nonfatal (checklist item 1)

    # mil_screen_hooks progress interface
    def phase_start(self, phase):
        pass

    def completed(self, phase, detail=None, **_):
        self.tick(phase)


def _peak(device):
    import torch
    return torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0


def _reset(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()


def run_mil(tasks, device, folder, status):
    """H-B-155 fit evaluation path, verbatim objects (fit v1.2.0 evaluation block)."""
    import numpy as np
    import torch
    from tools import hb155_fit_v1 as fit
    h, crops, native = fit.h, fit.crops, fit.native
    amp = 'none' if device == 'cpu' else 'fp16'
    runtime = h.configure_runtime(amp, 'cpu' if device == 'cpu' else 'cuda', policy='strict')
    h.PROGRESS = status
    gold, _ = gold_panel()
    members = mil_members()
    verified = set()
    for key in dict.fromkeys(t['key'] for t in tasks):
        m = members[key]; c = m['config']
        need(sha(ROOT/m['checkpoint']) == m['checkpoint_sha256'], 'H-B-155 checkpoint changed')
        model = h.build_model(c['recipe']['model'], ROOT)
        model = model.to(memory_format=torch.channels_last).to(device)
        state = torch.load(ROOT/m['checkpoint'], map_location='cpu', weights_only=True)
        h.strict_load(model, state['model'])
        del state
        model.eval()
        corpus = h.common.Corpus(ROOT/c['corpus'])
        need(corpus.contract == c['corpus_contract'], 'H-B-155 corpus contract differs')
        for task in [t for t in tasks if t['key'] == key]:
            ids = (gold if task['panel'] == 'gold58' else c['val'])[:task['n']]
            if (c['corpus'], tuple(ids)) not in verified:
                # Accessed shards only (Gold58: 4 crop / 1 native448 shard, < 2 GiB): seconds.
                h.verify_shards(corpus, ids)
                verified.add((c['corpus'], tuple(ids)))
                status.tick('verify')
            labels = {uid: np.zeros(12, np.float32) for uid in ids}
            ds = crops.Windows(h.MILWindows(ROOT/c['corpus'], ids, labels, 36, False, c['seed']), c['resolution'])
            loader = h.DataLoader(ds, batch_size=1 if device == 'cpu' else 4, shuffle=False,
                                  num_workers=c['workers'], worker_init_fn=h.worker_init,
                                  collate_fn=native.NativeArm().collate_fn)
            _reset(device)
            start = time.perf_counter()
            values, ordered, _ = h.evaluate_with(native.NativeArm(), model, loader, device, amp, corpus, tta=True)
            seconds = time.perf_counter()-start
            need(ordered == ids and values.shape == (len(ids), 12) and np.isfinite(values).all()
                 and ((values >= 0) & (values <= 1)).all(), 'invalid H-B-155 panel predictions')
            row = dict(version=VERSION, family='mil', key=key, view=224, panel=task['panel'], ids=ids,
                       probabilities=values.tolist(), seconds=seconds, device=str(device), runtime=runtime,
                       checkpoint_sha256=m['checkpoint_sha256'], k=36, tta=True, amp=amp,
                       peak_reserved_bytes=_peak(device))
            if task['panel'] == 'fold2val2':
                reference = mil_predictions(m)[:len(ids)]
                row['parity_vs_4090_val_csv_max_abs'] = float(np.max(np.abs(values-reference)))
            io.journal(folder/'journal'/task_name(task), row, folder)
            status.tick('mil')
        del model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def run_release(tasks, device, folder, status, hb162_output=None):
    """H-B-158/H-B-162 serving: authenticated release definitions at 320/384."""
    import gc
    import numpy as np
    import torch
    from tools import hb158_scale_contrast_v1 as s
    from tools import oai_moaks_panel as p
    if str(device).startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    gold, _ = gold_panel()
    oai, _ = oai_panel()
    cache = p.Cache('96')
    models = release_models(hb162_output)
    for key, view in dict.fromkeys((t['key'], t['view']) for t in tasks):
        meta = models[key]
        ns, proof = s.serving(view)
        model = s.make_model(ns, device)
        if meta['kind'] == 'continuation':
            from tools import hb162_sam_f0_v1 as sam
            path = ROOT/meta['checkpoint']
            need(sha(path) == meta['checkpoint_sha256'], 'HB162 continuation checkpoint changed')
            payload = torch.load(path, map_location='cpu', weights_only=False)
            need(payload['registration_sha256'] == sha(sam.WORK/'registration.json')
                 and payload['parent_sha256'] == s.MODEL_SHA, 'continuation provenance mismatch')
            model.load_state_dict(payload['model'], strict=True)
            del payload
        for task in [t for t in tasks if (t['key'], t['view']) == (key, view)]:
            ids = (gold if task['panel'] == 'gold58' else oai)[:task['n']]
            values, inputs, times = [], [], []
            _reset(device)
            for uid in ids:
                if task['panel'] == 'gold58':
                    vol, mask, hashes = s.volume(uid)
                else:
                    raw, m = cache.get(uid)
                    m = np.asarray(m)
                    need(np.isin(m, (0, 1)).all(), 'OAI mask values')
                    vol, mask, hashes = np.array(raw), m.astype(np.uint8), dict(cache='96')
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                values.append(s.infer(ns, model, vol, mask, device))
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                inputs.append(hashes)
                status.tick('release')
            values = s.valid_probs(np.asarray(values), len(ids))
            row = dict(version=VERSION, family='release', key=key, view=view, panel=task['panel'], ids=ids,
                       probabilities=values.tolist(), seconds=times, inputs=inputs, device=str(device),
                       checkpoint_sha256=meta['checkpoint_sha256'], serving=proof,
                       precision='fp32' if device == 'cpu' else 'fp16', k=94, mirror=True,
                       peak_reserved_bytes=_peak(device))
            io.journal(folder/'journal'/task_name(task), row, folder)
        del model, ns
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def worker(phase, device, folder, hb162_output=None):
    """Executes one registered lens phase; caller owns supervision and memory admission."""
    folder = Path(folder)
    saved = io.read_json(folder/'plan.json')
    tasks = plan(phase, hb162_output is not None)
    need(saved['tasks'] == tasks, 'lens plan differs from controller registration')
    total = sum(t['n'] for t in tasks)
    status = Status(folder, total)
    if phase.startswith('mil'):
        run_mil(tasks, device, folder, status)
    else:
        run_release(tasks, device, folder, status, hb162_output)
    names = sorted(p.name for p in (folder/'journal').glob('*.json'))
    need(names == sorted(task_name(t) for t in tasks), 'lens phase journals incomplete')
    return dict(status='LENS_PHASE_COMPLETE', phase=phase, journals=len(names))


def cpu_smoke():
    """Two studies per panel through both serving paths on CPU (fp32/none amp)."""
    import torch
    reg = registered()
    result = dict(version=VERSION, status='CPU_SMOKE_PASS', registration_sha256=sha(LWORK/'registration.json'),
                  phases={}, hb162='not exercised on CPU: continuation checkpoints are produced by the running job;'
                  ' loader path unit-tested with a synthetic payload', cuda_validated=False)
    for phase in ('mil-cpu', 'release-cpu'):
        folder = LWORK/'cpu-smoke'/phase
        need(not (folder/'journal').exists() or not list((folder/'journal').glob('*.json')),
             'existing CPU smoke journals retained; remove deliberately before replay')
        folder.mkdir(parents=True, exist_ok=True)
        io.atomic_json(folder/'plan.json', dict(phase=phase, tasks=plan(phase, False)))
        started = time.perf_counter()
        worker(phase, 'cpu', folder)
        result['phases'][phase] = dict(seconds=time.perf_counter()-started,
            journals={q.name: sha(q) for q in sorted((folder/'journal').glob('*.json'))})
    parity = [io.read_json(q)['parity_vs_4090_val_csv_max_abs']
              for q in sorted((LWORK/'cpu-smoke/mil-cpu/journal').glob('*fold2val2.json'))]
    result['mil_cpu_fp32_vs_4090_fp16_val_csv_max_abs'] = max(parity)
    need(not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    need(reg == registered(), 'inputs changed during CPU smoke')
    io.atomic_json(LWORK/'cpu-smoke.json', result)
    return result


def smoke_binding():
    s = io.read_json(LWORK/'cpu-smoke.json')
    need(s.get('status') == 'CPU_SMOKE_PASS' and s.get('registration_sha256') == sha(LWORK/'registration.json'),
         'lens CPU smoke missing/stale')
    for phase, item in s['phases'].items():
        for name, pin in item['journals'].items():
            need(Path(name).name == name and sha(LWORK/'cpu-smoke'/phase/'journal'/name) == pin,
                 'lens smoke journal changed')
    return {'lenses/registration.json': sha(LWORK/'registration.json'),
            'lenses/cpu-smoke.json': sha(LWORK/'cpu-smoke.json')}


# ---------------------------------------------------------------- lens assembly (CPU)

def rank(p):
    from tools import multi_lens_readout as ml
    return ml.rank(p)


def image_item(ids, arms, controls, y, eligible, policy):
    """Seed-mean probabilities, participant/study singleton components (hb122 convention)."""
    import numpy as np
    y = np.asarray(y, dtype=float)
    need(len(ids) == len(set(ids)) and y.shape == (len(ids), 12) and not np.isinf(y).any()
         and np.isin(y[np.isfinite(y)], (0, 1)).all(), 'invalid binary panel outcomes')
    need(bool(arms) and bool(controls), 'panel members missing')  # seed means; member sets may differ (vs parent)
    for a in (*arms.values(), *controls.values()):
        a = np.asarray(a)
        need(a.shape == y.shape and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'invalid panel probabilities')
    return dict(ids=list(ids), arm=np.stack([np.asarray(arms[k], np.float32) for k in sorted(arms)]).mean(0, dtype=np.float32),
                control=np.stack([np.asarray(controls[k], np.float32) for k in sorted(controls)]).mean(0, dtype=np.float32),
                labels=np.nan_to_num(y), mask=np.isfinite(y), eligible=bool(eligible),
                components=[dict(component_id=u, members=[u]) for u in ids], component_policy=policy)


def mapped_nine(item):
    from tools import multi_lens_readout as ml
    w = ml.weights(item['ids'], item['components'])
    a = ml.auc_draws(item['arm'], item['labels'], item['mask'], w)
    c = ml.auc_draws(item['control'], item['labels'], item['mask'], w)
    ix = list(PANEL_TARGETS)
    value = ml.stat((a[:, ix]-c[:, ix]).mean(1), eligible=False, reason='descriptive mapped-nine macro')
    value.update(arm_auc=ml.clean(a[0, ix].mean()), control_auc=ml.clean(c[0, ix].mean()))
    return value


def evaluate(ctx, arms, controls, bundle=None, images=None, reasons=None):
    """Corrected-L4 evaluator (multi_lens_readout_l4_v2) plus written reasons."""
    from tools import multi_lens_readout as ml
    from tools import multi_lens_readout_l4_v2 as lens
    result = lens.evaluate(ctx, arms, controls, label_change=False, bundle=bundle, image_reads=images or None)
    ls = result['lenses']
    for name, reason in (reasons or {}).items():
        if not ls[name].get('eligible', True) or ls[name].get('value') is None:
            ls[name]['reason'] = reason
    if images:
        for name, item in images.items():
            entry = ls['L7']['entries'][name]
            entry['component_policy'] = item['component_policy']
            if name == 'oai_moaks':
                entry['mapped_nine_macro'] = mapped_nine(item)
            if not item['eligible']:
                entry['ineligible_reason'] = item.get('reason', 'descriptive panel')
    return ml.clean(result)


def aligned_bundle(fold, ids, arms, controls, variant):
    """Replace the S42/S43 leaves of one heldout fold; pooled family/bundle ranks fixed on all folds."""
    import numpy as np
    from tools import hb105_aligned_oof_v1 as aligned   # installs CPU tripwires: import last
    raw, rosters, auth = aligned.authenticate()
    need(rosters[fold]['val'] == list(ids), 'fold roster/order differs from the aligned surrogate')
    offset = sum(len(rosters[f]['val']) for f in range(fold))
    output = []
    for pair in (arms, controls):
        value = {f: dict(v) for f, v in raw.items()}
        for seed in ('42', '43'):
            a = np.asarray(pair[seed], dtype=np.float32)
            need(a.shape == raw[fold]['S'+seed].shape, 'replacement shape differs')
            value[fold]['S'+seed] = a
        output.append(aligned.variants(value)[variant][offset:offset+len(ids)])
    return output, auth


def q7s_slice(fold, ids):
    """Unchanged deployed-composition Q7S pooled ranks, aligned to the given UIDs."""
    import numpy as np
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, auth = aligned.authenticate()
    roster = rosters[fold]['val']
    need(set(roster) == set(ids) and len(roster) == len(ids), 'fold rows differ from the aligned surrogate')
    offset = sum(len(rosters[f]['val']) for f in range(fold))
    q = aligned.variants(raw)['replace'][offset:offset+len(roster)]
    index = {u: i for i, u in enumerate(roster)}
    return np.asarray(q)[[index[u] for u in ids]], auth


def coatnet_bundle(ctx, ids, arm, control):
    """multi_lens coatnet recipe: q = rank(mean fixed partners); Q = (q + rank(own))/2."""
    q, auth = q7s_slice(0, ids)
    q = rank(q)
    return dict(context=ctx, arm=(q+rank(arm))/2, control=(q+rank(control))/2,
                recipe=dict(scope='ALIGNED_OOF_SURROGATE_ONLY', fold=0, family_replaced='coatnet (own)',
                            fixed_partner='Q7S deployed composition (R50, C4, S) fold-0 slice, unchanged',
                            weights='50:50 rank(Q7S)/rank(own)', note=POLICY['l6_coatnet'])), auth


# ---------------------------------------------------------------- GPU journal access

def gpu_rows(phase):
    """Authenticated completed controller receipt of the FORENSICS_LENSES_1015 job."""
    from tools import forensics_1015_v1 as forensics
    receipt = forensics.completed_phase(phase, None)
    folder = forensics.WORK/'gpu'/phase
    return {name: io.read_json(folder/'journal'/name) for name in receipt['journals']}


def panel_values(rows, key, view, panel, ids):
    import numpy as np
    row = rows[f'{key}-{view}-{panel}.json']
    need(row['ids'] == list(ids) and row['key'] == key and row['view'] == view, 'panel journal identity')
    return np.asarray(row['probabilities'], dtype=float)


def l7_items(kind, arm_keys, control_keys, views=(320, 320), ancestry=None):
    """Seed-mean image reads for Gold58 (+OAI where servable)."""
    gold, gy = gold_panel()
    family_phase = 'mil-full' if kind == 'mil' else 'release-full'
    rows = gpu_rows(family_phase)
    items = {}
    va, vc = views
    items['gold58'] = image_item(gold, {k: panel_values(rows, k, va, 'gold58', gold) for k in arm_keys},
                                 {k: panel_values(rows, k, vc, 'gold58', gold) for k in control_keys},
                                 gy, False, 'Gold58 study singletons; EXPOSED descriptive')
    items['gold58']['reason'] = POLICY['gold58']
    if kind == 'release':
        oai, oy = oai_panel()
        eligible = bool(ancestry and ancestry.get('eligible'))
        items['oai_moaks'] = image_item(oai, {k: panel_values(rows, k, va, 'oai557', oai) for k in arm_keys},
                                        {k: panel_values(rows, k, vc, 'oai557', oai) for k in control_keys},
                                        oy, eligible, 'one knee per participant; participant singleton bootstrap')
        items['oai_moaks']['reason'] = ('EXP181 CONTROL lineage passes the panel OAI-free admission'
                                        if eligible else 'OAI-free ancestry not proven')
    return items, rows


def mean_views(rows, key, panel, ids):
    return (panel_values(rows, key, 320, panel, ids)+panel_values(rows, key, 384, panel, ids))/2


# ---------------------------------------------------------------- readouts

def lens_table(result):
    """Markdown seven-lens table: value, 95% CI, threshold, PASS, eligible, reason."""
    def fmt(x):
        if isinstance(x, dict):
            return ', '.join(f'{k} {fmt(v)}' for k, v in x.items())
        return 'n/a' if x is None else (f'{x:+.4f}' if isinstance(x, float) else str(x))
    def ci(x):
        if isinstance(x, dict):
            return '; '.join(f'{k} {ci(v)}' for k, v in x.items())
        return 'n/a' if not x else f'[{x[0]:+.4f}, {x[1]:+.4f}]'
    lines = ['| Lens | Value | 95% CI | Threshold | PASS | Eligible | Note |', '|---|---|---|---|---|---|---|']
    for name in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'):
        v = result['lenses'][name]
        note = v.get('reason', '')
        if name == 'L4' and v.get('selected_targets'):
            note = 'targets ' + ', '.join(v['selected_targets'])
        if name == 'L3' and 'rulers' in v:
            note = 'best ruler; ' + ', '.join(f"{k} {fmt(r['value'])}" for k, r in v['rulers'].items())
        if name == 'L6' and 'recipe' in v:
            note = (v['recipe'].get('fixed_partner') or v['recipe'].get('bundle', '')) + '; ' + note
        if name == 'L7' and 'entries' in v:
            note = '; '.join(f"{k}: eligible={e['eligible']}" + (f" ({e.get('ineligible_reason')})" if not e['eligible'] else '')
                             for k, e in v['entries'].items())
        lines.append(f"| {name} | {fmt(v.get('value'))} | {ci(v.get('ci95'))} | {v.get('threshold', 'n/a')} | "
                     f"{v.get('PASS')} | {v.get('eligible', True)} | {note} |")
    return '\n'.join(lines)


def write_result(name, result):
    from tools import multi_lens_readout as ml
    LWORK.mkdir(parents=True, exist_ok=True)
    io.atomic_json(LWORK/f'{name}.json', ml.clean(result))
    (LWORK/f'{name}.md').write_text(f'# {name}\n\n' + lens_table(result) + '\n', encoding='utf-8')
    return LWORK/f'{name}.json'


def readout_hb155(with_gpu=False):
    import numpy as np
    from tools import hb155_build_v1 as b
    from tools import multi_lens_readout as ml
    members = mil_members()
    c = members['crop-s42']['config']
    arms = {str(s): mil_predictions(members[f'crop-s{s}']) for s in (42, 43)}
    controls = {str(s): mil_predictions(members[f'control-s{s}']) for s in (42, 43)}
    images = None
    reasons = {'L7': 'Gold58 GPU panel inference pending (FORENSICS_LENSES_1015 mil phases); '
                     + POLICY['hb155_oai']}
    if with_gpu:
        images, _ = l7_items('mil', [f'crop-s{s}' for s in (42, 43)], [f'control-s{s}' for s in (42, 43)], (224, 224))
        reasons['L7'] = 'Gold58 descriptive (EXPOSED); ' + POLICY['hb155_oai']
    ctx = ml.make_context(c['val'], ROOT)
    (qa, qc), auth = aligned_bundle(2, c['val'], arms, controls, 'replace')
    bundle = dict(context=ctx, arm=qa, control=qc, recipe=dict(scope='ALIGNED_OOF_SURROGATE_ONLY', fold=2,
        family_replaced='S (ConvNeXt-S finding MIL): H-B-155 crop vs control replace S42/S43 fold-2 leaves',
        fixed_partner='Q7S deployed composition (R50, C4, S); folds 0,1,3,4 and other families unchanged',
        bundle='replace', arithmetic='FP32 seed mean; pooled family ranks on all 4343 rows; fold-2 slice'))
    result = evaluate(ctx, arms, controls, bundle, images, reasons)
    (sa, sc), _ = aligned_bundle(2, c['val'], arms, controls, 'add')
    secondary = ctx.delta(sa, sc)
    result['lenses']['L6']['secondary_add_bundle'] = ml.stat(secondary[:, :2].mean((1, 2)), .0005,
        eligible=False, reason=POLICY['l6_secondary'])
    result['lenses']['L6']['final_proxy'] = ml.unavailable(
        'No held-out own (CoAtNet) fold-2 predictions exist; the EXP181 own anchor is fold 0 only')
    result['lenses']['L6']['exposure'] = (
        'Fold-2 rows are heldout for every partner leaf (R50 s42/s43 F2, C4 F2, S F2 fits exclude fold 2) and for '
        'both H-B-155 arms; EXPOSED: L2 targets derive from the same report rulers, fold-2 selection shared by '
        'both arms, pooled ranks couple folds; clean_oof=false')
    saved = io.read_json(b.WORK/'readout.json')
    for lens_name in ('L1', 'L2', 'L3', 'L4', 'L5'):
        need(result['lenses'][lens_name]['value'] == ml.clean(saved['lenses'][lens_name]['value']),
             f'H-B-155 {lens_name} differs from the registered readout')
    result.update(version=VERSION, item='H-B-155', status='COMPLETE_L1_L7' if with_gpu else 'CPU_L1_L6',
        registered_readout_sha256=sha(b.WORK/'readout.json'), aligned_bindings=len(auth['bindings']),
        members={k: {n: v for n, v in m.items() if n != 'config'} for k, m in members.items()},
        exposure='EXPOSED', clean_oof=False, rows=len(c['val']), ordered_ids_sha256=uid_hash(c['val']))
    return write_result('hb155-lenses' + ('-final' if with_gpu else '-cpu'), result)


def hb158_predictions():
    import numpy as np
    from tools import hb158_scale_contrast_v1 as s
    reg = s.registered()
    receipt = s.completed_phase('full', 2*len(reg['ids']))
    ids = reg['ids']
    arms = {}
    for res in VIEWS:
        rows = [io.read_json(s.WORK/'gpu/full/journal'/f'{res}-{i:04d}.json') for i in range(len(ids))]
        need([r['uid'] for r in rows] == ids and all(r['view'] == res for r in rows), 'HB158 journal order')
        arms[res] = s.valid_probs([r['probabilities'] for r in rows], len(ids))
    return ids, arms, dict(registration_sha256=sha(s.WORK/'registration.json'),
                           controller_sha256=sha(s.WORK/'gpu/full/controller.json'),
                           result_sha256=sha(s.WORK/'result.json'), journals=len(receipt['journal_sha256']))


def readout_hb158(with_gpu=False):
    from tools import multi_lens_readout as ml
    ids, views, binding = hb158_predictions()   # torch.load inside authenticate: before aligned import
    ancestry = oai_ancestry()
    arm, control = (views[320]+views[384])/2, views[320]
    images = None
    reasons = {'L5': 'Single checkpoint (EXP181 CONTROL seed 42); no same-recipe second seed exists '
                     '(EXP188/190/192 are different recipes, all seed 42); a seed pair needs a new fit',
               'L7': 'GPU panel inference pending (FORENSICS_LENSES_1015 release phases)'}
    if with_gpu:
        gold, gy = gold_panel(); oai, oy = oai_panel()
        rows = gpu_rows('release-full')
        images = {'gold58': image_item(gold, {'42': mean_views(rows, PARENT_KEY, 'gold58', gold)},
                                       {'42': panel_values(rows, PARENT_KEY, 320, 'gold58', gold)}, gy, False,
                                       'Gold58 study singletons; EXPOSED descriptive'),
                  'oai_moaks': image_item(oai, {'42': mean_views(rows, PARENT_KEY, 'oai557', oai)},
                                          {'42': panel_values(rows, PARENT_KEY, 320, 'oai557', oai)}, oy,
                                          ancestry['eligible'], 'one knee per participant; participant singleton bootstrap')}
        images['gold58']['reason'] = POLICY['gold58']
        images['oai_moaks']['reason'] = ancestry['ancestry']
        reasons.pop('L7')
    ctx = ml.make_context(ids, ROOT)
    bundle, auth = coatnet_bundle(ctx, ids, arm, control)
    result = evaluate(ctx, {'42': arm}, {'42': control}, bundle, images, reasons)
    result['lenses']['L6']['final_proxy'] = ml.unavailable('The own member IS the EXP181 CONTROL anchor (coatnet family)')
    result.update(version=VERSION, item='H-B-158', contrast='mean(320,384) - crop320 on EXP181 CONTROL fold 0',
        status='COMPLETE_L1_L7' if with_gpu else 'CPU_L1_L6', inputs=binding, oai_ancestry=ancestry,
        aligned_bindings=len(auth['bindings']), exposure='EXPOSED_SCREEN', clean_oof=False, rows=len(ids),
        ordered_ids_sha256=uid_hash(ids))
    if with_gpu:
        result['oai_panel_parity_vs_existing_EXP181_CONTROL_csv'] = oai_parity(rows)
    return write_result('hb158-lenses' + ('-final' if with_gpu else '-cpu'), result)


def oai_parity(rows):
    """Descriptive: release-path 320 OAI predictions vs the panel's own coatnet path CSV."""
    import numpy as np
    from tools import oai_moaks_panel as p
    oai, _ = oai_panel()
    path = p.OUT/'full-v103'/'EXP181_CONTROL.csv'
    import csv as _csv
    with path.open(encoding='utf-8', newline='') as stream:
        table = {r['study_key']: [float(r[t]) for t in p.TARGETS] for r in _csv.DictReader(stream)}
    need(set(table) == set(oai), 'existing EXP181 OAI CSV roster differs')
    mine = panel_values(rows, PARENT_KEY, 320, 'oai557', oai)
    other = np.asarray([table[u] for u in oai])
    return dict(max_abs=float(np.max(np.abs(mine-other))), mean_abs=float(np.mean(np.abs(mine-other))),
                csv_sha256=sha(path), note='different implementations (release vs training.arms.coatnet); descriptive')


def readout_hb162(with_gpu=False, accept_memory_breach=False):
    """Prepared: SAM vs AdamW (matched seeds) primary; SAM seed-mean vs untouched parent secondary."""
    import numpy as np
    from tools import multi_lens_readout as ml
    from tools import hb162_sam_f0_v1 as sam
    state = hb162_terminal_state()
    need(state == 'SUCCESS' or (accept_memory_breach and state == 'FAILED'),
         'H-B-162 has not finished successfully; readout refused')
    # Same narrow admission as the registered evaluator; loads all model-related
    # inputs before the aligned helper installs CPU tripwires.
    loaded,parent,ids,_,_,identity,evidence = sam.evaluation_inputs(accept_memory_breach)
    arms = {s: loaded[f'SAM{s}'] for s in ('42', '43')}
    controls = {s: loaded[f'ADAMW{s}'] for s in ('42', '43')}
    registered_result = io.read_json(sam.WORK/'result.json')
    need(registered_result.get('binding') == sam.binding()
         and registered_result.get('full_output_sha256') == evidence['full_output_sha256']
         and registered_result.get('memory_gate') == evidence['memory_gate']
         and registered_result.get('evaluator_sources') == evidence['evaluator_sources'],
         'registered scientific result is stale or belongs to different inputs')
    ancestry = oai_ancestry()
    reasons = {'L7': 'No authenticated paired Gold58/OAI continuation predictions available in this CPU readout; '
                     '888 held-out studies are ruler predictions, not independent L7 image panels'}
    images = None
    if with_gpu:
        images, _ = l7_items('release', [f'SAM{s}' for s in ('42', '43')], [f'ADAMW{s}' for s in ('42', '43')],
                             (320, 320), ancestry)
        reasons.pop('L7')
    ctx = ml.make_context(ids, ROOT)
    sam_mean = np.mean(list(arms.values()), axis=0); adam_mean = np.mean(list(controls.values()), axis=0)
    bundle, auth = coatnet_bundle(ctx, ids, sam_mean, adam_mean)
    primary = evaluate(ctx, arms, controls, bundle, images, reasons)
    primary['lenses']['L6']['final_proxy'] = ml.unavailable('Own member is the replaced CoAtNet continuation')
    secondary_images = None
    if with_gpu:
        secondary_images, _ = l7_items('release', ['SAM42', 'SAM43'], [PARENT_KEY], (320, 320), ancestry)
    pbundle, _ = coatnet_bundle(ctx, ids, sam_mean, parent)
    secondary = evaluate(ctx, {'mean': sam_mean}, {'mean': parent}, pbundle, secondary_images,
                         dict(reasons, L5='Single untouched parent; seed pairs are read in the primary contrast'))
    result = dict(version=VERSION, item='H-B-162', status='COMPLETE_L1_L7' if with_gpu else 'L1_L6',
                  primary_SAM_vs_ADAMW=primary, secondary_SAM_mean_vs_parent=secondary,
                  registered_gate=registered_result['gate'], identity=identity,
                  oai_ancestry=ancestry, aligned_bindings=len(auth['bindings']), exposure='EXPOSED', clean_oof=False,
                  **evidence)
    suffix = ('-final' if with_gpu else '') + ('-memory-v101' if evidence['accepted_memory_breach'] else '')
    for contrast in (primary, secondary):
        contrast.update(version=VERSION, **evidence)
    path = write_result('hb162-lenses' + suffix, primary)
    write_result('hb162-parent-lenses' + suffix, secondary)
    io.atomic_json(LWORK/('hb162-lenses' + suffix + '-full.json'), ml.clean(result))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'validate', 'cpu-smoke', 'readout'])
    ap.add_argument('--item', choices=['HB155', 'HB158', 'HB162'])
    ap.add_argument('--with-gpu', action='store_true')
    ap.add_argument('--accept-memory-breach', action='store_true')
    args = ap.parse_args(argv)
    if args.accept_memory_breach and (args.command != 'readout' or args.item != 'HB162'):
        ap.error('--accept-memory-breach is restricted to readout --item HB162')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    LWORK.mkdir(parents=True, exist_ok=True)
    if args.command == 'register':
        io.atomic_json(LWORK/'registration.json', registration()); result = dict(status='LENS_REGISTERED_CPU_ONLY')
    elif args.command == 'validate':
        registered(); result = dict(status='LENS_VALIDATED_CPU_ONLY')
    elif args.command == 'cpu-smoke':
        r = cpu_smoke(); result = {k: v for k, v in r.items() if k != 'phases'}
    else:
        need(args.item is not None, '--item required')
        if args.item == 'HB162':
            path = readout_hb162(args.with_gpu, args.accept_memory_breach)
        else:
            path = {'HB155': readout_hb155, 'HB158': readout_hb158}[args.item](args.with_gpu)
        result = dict(status='READOUT_WRITTEN', path=rel(path))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
