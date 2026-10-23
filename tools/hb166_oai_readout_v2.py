"""H-B-166 OAI readout v2.0.0 (2026-10-16), CPU build / B-only GPU execution.

New version; frozen v1 inputs and receipts are never edited. Changelog: completed
eight-member identities, explicit ineligible ancestry, pooled descriptive and
conditional H166/H105 add proxy estimands, executable serial/memory/watchdog/review gates,
exact frozen native/crop completion-manifest reuse bindings, and measured-only
peak gating after reserve-only qualification headroom checks.
No labels are written or printed. GPU run fails closed while lineage is unproven.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools import hb145_output_io as io

VERSION = '2.0.0'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.local/scratch/hb166-oai/H-B-166'
WORK = BASE / 'readout-v200'
NATIVE = ROOT / '.local/scratch/hb147-lenses/H-B-147/oai448'
CROP = BASE / 'crop448'
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
KEYS = tuple(f'{arm}-s{s}' for s in (42, 43, 44, 45) for arm in ('multiview', 'control'))
RECIPE = dict(batch=4, microbatch=2, k_per_view=36, resolution=224, workers=0,
              amp='fp16', backend='strict', flip_tta=True, serial=True)
RULE = dict(weak_indices=[3, 5, 6, 8], threshold=.002, draws=300, seed=20261003,
    seed_mean='float32 probability arithmetic mean; no rank before own AUC',
    folds='fold2 s42/s43 and fold3 s44/s45 separately',
    pooled='NEW descriptive estimand: arithmetic mean of all four seeds per arm, then paired AUC delta',
    bootstrap='same singleton-participant draws for both arms; CI descriptive',
    proxy='(rank(mean(R50 s42/s43))+rank(C4 s42)+rank(mean(T s42/s43))+rank(own S))/4; H166/H105 add',
    h2='Contusion index10 ONLY; unavailable under frozen OAI mapping; no BML substitution',
    clean_oof=False, exposure='EXPOSED', submission_authorized=False)
EXTRA_CODE = ('tools/hb166_oai_readout_v2.py', 'tools/hb155_watchdog_v1.py',
    'tools/hb150_io_v1.py', 'tools/gpu_queue.py', 'tools/gpu_job.py',
    'tools/job_runtime.py', 'tools/hb147_progress_v1.py', 'tools/hb166_readout_v1.py',
    'tools/hb166_build_v1.py', 'tests/test_hb166_oai_readout_v2.py')
PARTNERS = ('r50-s42', 'r50-s43', 'c4-s42', 't-s42', 't-s43')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return io.read_json(Path(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def safe_path(path):
    p = (ROOT / path).resolve()
    need(p.is_relative_to(ROOT) and '.local/relabel' not in p.as_posix().lower(), 'unsafe project path')
    return p


def panel_ids():
    ids = np.load(NATIVE / 'ids.npy', allow_pickle=False).tolist()
    need(len(ids) == len(set(ids)) == 557 and len({u.split('_')[1] for u in ids}) == 557,
         'frozen557 participant/roster differs')
    return ids


def completed_members():
    old = read(BASE / 'registration.json')
    entries = []
    for stage, seeds, fold in (('pilot-v100', (42, 43), 2), ('pilot-fold3-v110', (44, 45), 3)):
        for seed in seeds:
            mp = ROOT / f'.local/scratch/slot-b/hb166/{stage}/configs/multiview-s{seed}.json'
            mv = read(mp)
            for arm, cp in (('multiview', mp), ('control', safe_path(mv['control_config']))):
                cfg = read(cp); directory = cp.parent.parent / 'fits' / f'{arm}-s{seed}'
                # Only heartbeat is read until it authenticates completion.
                need(read(directory / 'heartbeat.json').get('status') == 'complete', 'member not complete')
                summary = read(directory / 'summary.json'); identity = summary['identity']
                ck = directory / 'final.pt'; cs = sha(cp); ks = sha(ck)
                need(old['configs'][rel(cp)] == cs and cfg['arm'] == arm and cfg['seed'] == seed
                     and cfg['fold'] == fold and not cfg['engineering_only'], 'fit config identity drift')
                need(identity['config_sha256'] == cs and identity['arm'] == arm
                     and identity['seed'] == seed and identity['fold'] == fold
                     and not identity['engineering_only'] and summary['checkpoint_sha256'] == ks,
                     'completed member identity drift')
                if arm == 'control':
                    need(mv['control_fits'][rel(ck)] == ks, 'registered control differs')
                entries.append(dict(key=f'{arm}-s{seed}', arm=arm, seed=seed, fold=fold,
                    config=rel(cp), config_sha256=cs, checkpoint=rel(ck), checkpoint_sha256=ks,
                    summary=rel(directory / 'summary.json'), summary_sha256=sha(directory / 'summary.json')))
    need(tuple(x['key'] for x in entries) == KEYS, 'member ordering differs')
    return entries


def lineage_audit(entries):
    """Prove direct exposure narrowly; incomplete teacher ancestry is ineligible."""
    oai = set(panel_ids()); evidence = []
    for e in entries:
        c = read(safe_path(e['config'])); roots = c['recipe']['roots']; model = c['recipe']['model']
        canonical_path = safe_path(roots['canonical']); canonical = read(canonical_path)
        canonical_ids = set(canonical if isinstance(canonical, list) else canonical['train'] + canonical['val'])
        corpus = safe_path(c['corpus']); cache_ids = set(np.load(corpus / 'ids.npy', allow_pickle=False).tolist())
        manifest = read(corpus / 'manifest.json')
        need(sha(corpus / 'ids.npy') == manifest['ids_sha256'], 'training cache roster hash differs')
        datasets = {}; blockers = ['Complete independently verified initial-weight and Dread/Aquila teacher training populations are absent']
        for split in ('train', 'val'):
            ids = c[split]
            need(len(ids) == len(set(ids)) and set(ids) <= canonical_ids and set(ids) <= cache_ids,
                 'fit roster not in canonical competition/cache roster')
            overlap = len(set(ids) & oai); named = sum(str(u).startswith('OAI_') for u in ids)
            if overlap or named:
                blockers.append('Direct OAI exposure in ' + split)
            datasets[split] = dict(count=len(ids), ordered_sha256=digest(ids),
                dataset='RSNA competition canonical roster', canonical_sha256=sha(canonical_path),
                cache_ids_sha256=manifest['ids_sha256'], oai_overlap=overlap, oai_named=named)
        caches = {}
        for cache_name, cache_path in [('native', corpus), *([('crop', safe_path(c['crop_corpus']))]
                                                          if e['arm'] == 'multiview' else [])]:
            cm = read(cache_path / 'manifest.json')
            ci = np.load(cache_path / 'ids.npy', allow_pickle=False).tolist()
            need(len(ci) == len(set(ci)) and sha(cache_path / 'ids.npy') == cm['ids_sha256'],
                 'fit cache roster differs')
            need(set(c['train'] + c['val']) <= set(ci), 'fit roster absent from cache')
            caches[cache_name] = dict(path=rel(cache_path), manifest_sha256=sha(cache_path / 'manifest.json'),
                ids_sha256=cm['ids_sha256'], contract=cm['contract'], studies=len(ci),
                canonical_competition_members=len(set(ci) & canonical_ids),
                outside_canonical_members=len(set(ci) - canonical_ids),
                oai_overlap=len(set(ci) & oai), oai_named=sum(str(u).startswith('OAI_') for u in ci),
                fitted_roster_subset=True)
        sources = {}
        for name, hashkey in (('initial_state', 'initial_file_sha256'), ('weights', 'weights_sha256')):
            p = safe_path(model[name]); actual = sha(p)
            need(actual == model[hashkey], name + ' hash differs')
            sources[name] = dict(path=rel(p), sha256=actual, complete_training_population_proven=False)
        label_path = safe_path(c['label_path'])
        # Hashing the structured soft-label artifact does not print/read row contents.
        label_info = dict(path=rel(label_path), sha256=sha(label_path),
            semantic_sha256=c['label_semantic_sha256'], recipe=c['recipe']['labels'],
            complete_teacher_training_populations_proven=False)
        provenance = label_path.with_suffix('.provenance.json'); doc = read(provenance)
        need(doc['csv_sha256'] == label_info['sha256'], 'soft-label provenance artifact differs')
        teacher = label_path.parent / 'aquila_v1.provenance.json'; td = read(teacher)
        label_info.update(provenance=rel(provenance), provenance_sha256=sha(provenance),
            oai_declaration=doc.get('oai', 'not declared'),
            aquila_provenance=rel(teacher), aquila_provenance_sha256=sha(teacher),
            aquila_method=td['method'], aquila_ancestry='image-assisted; complete training populations unaudited')
        evidence.append(dict(key=e['key'], config_sha256=e['config_sha256'],
            checkpoint_sha256=e['checkpoint_sha256'], datasets=datasets, caches=caches, sources=sources,
            soft_labels=label_info, eligible=False, verdict='INELIGIBLE', blockers=blockers))
    return dict(schema='hb166-oai-lineage/v2', version=VERSION, verdict='INELIGIBLE', eligible=False,
                interpretation='Missing full ancestry proof; not a finding of OAI contamination', members=evidence)


def partner_contract():
    p = WORK / 'partner-inventory.json'
    if not p.exists():
        return dict(status='UNAVAILABLE', members={},
                    reason='No authenticated complete H166 R50/C4/T OAI557 partner bundle registered; R50/C4 missing')
    d = read(p)
    need(d.get('status') in ('AVAILABLE', 'UNAVAILABLE'), 'partner status differs')
    if d['status'] == 'UNAVAILABLE':
        need(not d.get('members') and not d.get('anchor'), 'partial partner bundle cannot be unavailable')
    else:
        need(set(d['members']) == set(PARTNERS) and not d.get('anchor'), 'complete H166 R50/C4/T partner bundle required')
        for x in d['members'].values():
            need(x.get('path') and x.get('sha256') and x.get('receipt') and x.get('receipt_sha256'),
                 'partner identity receipt required')
    return d


def frozen_cache_binding(old):
    completion_path = BASE / 'completion-evidence.json'
    completion = read(completion_path)
    native = sha(NATIVE / 'manifest.json'); crop = sha(CROP / 'manifest.json')
    need(native == old['native_manifest_sha256'] == completion['native_manifest_sha256'],
         'frozen native manifest binding differs')
    need(crop == completion['crop_manifest_sha256'], 'frozen crop manifest binding differs')
    return dict(native_manifest_sha256=native, crop_manifest_sha256=crop,
                original_completion_evidence_sha256=sha(completion_path))


def registration():
    old = read(BASE / 'registration.json'); entries = completed_members()
    for p, expected in old['code'].items():
        need(sha(safe_path(p)) == expected, 'frozen v1 code differs: ' + p)
    lineage = lineage_audit(entries); cache_binding = frozen_cache_binding(old)
    return dict(schema='hb166-oai-readout/v2', version=VERSION, members=entries, recipe=RECIPE, rule=RULE,
        original_registration_sha256=sha(BASE / 'registration.json'),
        **cache_binding,
        ordered_ids_sha256=digest(panel_ids()), studies=557,
        code={p: sha(safe_path(p)) for p in sorted(set(old['code']) | set(EXTRA_CODE))},
        lineage=lineage, partners=partner_contract(), no_admission=True)


def register():
    WORK.mkdir(parents=True, exist_ok=True); r = registration(); p = WORK / 'registration.json'
    if p.exists():
        need(read(p) == r, 'registration drift: use a reviewed new version')
    else:
        io.atomic_json(p, r)
        io.atomic_json(WORK / 'lineage.json', r['lineage'])
    return r


def authenticate(r):
    for p, expected in r['code'].items():
        need(sha(safe_path(p)) == expected, 'code binding differs: ' + p)
    need(sha(BASE / 'registration.json') == r['original_registration_sha256'], 'original registration differs')
    need(sha(BASE / 'completion-evidence.json') == r['original_completion_evidence_sha256'],
         'original completion evidence differs')
    need(digest(panel_ids()) == r['ordered_ids_sha256'], 'canonical roster differs')
    for directory, key in ((NATIVE, 'native_manifest_sha256'), (CROP, 'crop_manifest_sha256')):
        need(sha(directory / 'manifest.json') == r[key], 'input manifest differs')
    for e in r['members']:
        for key in ('config', 'checkpoint', 'summary'):
            need(sha(safe_path(e[key])) == e[key + '_sha256'], 'member identity differs: ' + e['key'])
    need(read(WORK / 'lineage.json') == r['lineage'], 'lineage binding differs')
    return r


def check_review(r):
    p = WORK / 'gpu-review.json'
    need(p.is_file(), 'independent GPU execution review missing')
    v = read(p); report = safe_path(v.get('report', 'missing'))
    need(v.get('verdict') == 'PASS' and v.get('reviewer') not in (None, '', 'builder')
         and v.get('scope') == 'H-B-166 OAI4070 executable readout'
         and v.get('registration_sha256') == digest(r), 'independent review binding differs')
    need(report.is_relative_to(WORK) and sha(report) == v.get('report_sha256'), 'review report differs')
    need(v.get('cpu_smoke_sha256') == sha(WORK / 'cpu-smoke.json'), 'smoke review binding differs')


def scientific_gate(r):
    lineage = r['lineage']
    need(lineage.get('eligible') is True and lineage.get('verdict') == 'ELIGIBLE'
         and len(lineage['members']) == 8 and all(x.get('eligible') is True for x in lineage['members']),
         'OAI lineage INELIGIBLE: GPU readout blocked')


def memory_gate(baseline, peak, total):
    need(all(isinstance(x, (int, float)) and np.isfinite(x) and x >= 0 for x in (baseline, peak, total))
         and total > 0 and baseline <= total, 'invalid measured memory')
    need(peak <= 12 * (1 << 30), 'real full-shape peak exceeds 12 GiB')
    need(baseline + peak + 2 * (1 << 30) <= total, 'measured desktop + peak + 2 GiB exceeds total')
    return dict(PASS=True, baseline_bytes=baseline, peak_reserved_bytes=peak, total_bytes=total)


def qualification_headroom(baseline, total):
    """Before measuring a peak, check only the actual desktop and fixed reserve."""
    need(all(isinstance(x, (int, float)) and np.isfinite(x) and x >= 0 for x in (baseline, total))
         and total > 0 and baseline <= total, 'invalid measured baseline')
    need(baseline + 2 * (1 << 30) <= total, 'measured desktop + 2 GiB reserve exceeds total')
    return dict(PASS=True, baseline_bytes=baseline, total_bytes=total)


def probability_array(value, ids):
    p = np.asarray(value, np.float32)
    need(p.shape == (len(ids), 12) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
         'invalid probabilities')
    return p


def aligned_predictions(path, ids):
    with np.load(path, allow_pickle=False) as z:
        uid = z['ids'].tolist(); p = probability_array(z['probabilities'], uid)
    need(len(uid) == len(set(uid)) and set(uid) == set(ids), 'prediction roster differs; no intersection')
    order = {u: i for i, u in enumerate(uid)}
    return p[[order[u] for u in ids]]


def load_partners(spec, ids):
    if spec['status'] == 'UNAVAILABLE':
        need(not spec.get('members') and not spec.get('anchor'), 'partial bundle is invalid')
        return None
    need(set(spec['members']) == set(PARTNERS) and not spec.get('anchor'), 'partial partner bundle')
    from tools.hb166_build_v1 import L6_PARTNERS
    def one(key, x):
        p = safe_path(x['path']); rp = safe_path(x['receipt'])
        need(sha(p) == x['sha256'] and sha(rp) == x['receipt_sha256'], 'partner hash differs')
        receipt = read(rp)
        family, seed = key.rsplit('-s', 1)
        family = {'r50': 'R50', 'c4': 'C4', 't': 'T'}[family]
        source_path, source_hash = L6_PARTNERS[family][seed]
        need(receipt.get('complete') is True and receipt.get('prediction_sha256') == x['sha256']
             and receipt.get('ordered_ids_sha256') == digest(ids) and receipt.get('oai_eligible') is True,
             'partner receipt/lineage differs')
        need(receipt.get('family') == family and str(receipt.get('seed')) == seed
             and receipt.get('source_oof') == source_path and receipt.get('source_oof_sha256') == source_hash
             and x.get('source_oof') == source_path and x.get('source_oof_sha256') == source_hash
             and len(str(x.get('checkpoint_sha256', ''))) == 64
             and receipt.get('checkpoint_sha256') == x['checkpoint_sha256'],
             'partner registered family/checkpoint identity differs')
        return aligned_predictions(p, ids)
    return dict(members={k: one(k, x) for k, x in spec['members'].items()})


def proxy_probabilities(a, c, partners):
    from tools.multi_lens_readout import rank
    need(set(partners['members']) == set(PARTNERS), 'partial partner bundle')
    p = partners['members']; r50 = np.stack([p['r50-s42'], p['r50-s43']]).mean(0, dtype=np.float32)
    tiny = np.stack([p['t-s42'], p['t-s43']]).mean(0, dtype=np.float32)
    fixed = [rank(r50), rank(p['c4-s42']), rank(tiny)]
    return dict(H166_add=(np.mean([rank(a), *fixed], axis=0), np.mean([rank(c), *fixed], axis=0)))


def comparison(ids, a, c, y, *, eligible, descriptive):
    from tools import multi_lens_readout as ml
    from tools import lens_completion_1015_v1 as lens
    item = lens.image_item(ids, {'mean': a}, {'mean': c}, y, eligible, 'one frozen knee per participant')
    w = ml.weights(ids, item['components'])
    delta = ml.auc_draws(item['arm'], item['labels'], item['mask'], w) - ml.auc_draws(item['control'], item['labels'], item['mask'], w)
    weak = ml.stat(delta[:, ml.WEAK].mean(1), .002, eligible=eligible and not descriptive)
    def gated(stat):
        stat['point_threshold_met'] = stat['PASS']
        stat['PASS'] = stat['scientific_PASS'] = bool(stat['PASS'] and stat['eligible'])
        return stat
    return dict(weak4=gated(weak), mapped_nine=gated(lens.mapped_nine(item)), descriptive=descriptive,
        targets={name: gated(ml.stat(delta[:, t], eligible=False, reason='descriptive')) for t, name in enumerate(ml.TARGETS)},
        H2=ml.unavailable('Contusion target10 is unmapped in frozen OAI outcomes; no BML substitute'))


def compute_readout(ids, predictions, y, *, eligible=False, partners=None):
    need(set(predictions) == set(KEYS), 'all eight paired members required')
    need(len(ids) == len(set(ids)), 'duplicate roster')
    y = np.asarray(y, float)
    need(y.shape == (len(ids), 12) and np.isnan(y[:, 10]).all(), 'frozen Contusion mapping differs')
    pp = {k: probability_array(v, ids) for k, v in predictions.items()}
    out = {}
    for name, seeds in (('fold2', (42, 43)), ('fold3', (44, 45)), ('pooled', (42, 43, 44, 45))):
        means = {arm: np.stack([pp[f'{arm}-s{s}'] for s in seeds]).mean(0, dtype=np.float32)
                 for arm in ('multiview', 'control')}
        a, c = means['multiview'], means['control']
        row = comparison(ids, a, c, y, eligible=eligible, descriptive=name == 'pooled')
        row['seeds'] = list(seeds); row['probability_mean_sha256'] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in means.items()}
        if partners is None:
            row['blend_proxy'] = dict(status='UNAVAILABLE', eligible=False, reason='H166 add partners unavailable: exact R50/C4 OAI predictions missing; T alone is insufficient')
        else:
            row['blend_proxy'] = {k: comparison(ids, *v, y, eligible=False, descriptive=True)
                                  for k, v in proxy_probabilities(a, c, partners).items()}
        out[name] = row
    return dict(version=VERSION, studies=len(ids), eligible=eligible, rule=RULE, views=out,
                labels_exported=False, no_submission_authorization=True)


def atomic_npz(path, ids, probabilities):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.tmp')
    def save():
        with temp.open('wb') as f:
            np.savez(f, ids=np.asarray(ids), probabilities=np.asarray(probabilities, np.float32)); f.flush(); os.fsync(f.fileno())
    io.retry(save); io.replace(temp, path)


def heartbeat(sequence, studies, event='complete'):
    hp = os.environ.get('HB155_HEARTBEAT'); token = os.environ.get('HB155_TOKEN')
    if hp and token:
        try:
            io.atomic_json(hp, dict(token=token, pid=os.getpid(), phase='panel_inference',
                sequence=sequence, completed_at=time.monotonic(), event=event,
                detail=dict(progress_kind='panel', cumulative_units=studies)))
        except OSError:
            pass


def inference(entry, ids, *, device, progress=True):
    """One shared path for real CPU smoke and full GPU K36-per-view flip."""
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from training.arms import hb166_multiview_v1 as mv
    from training.arms import mil_screen_hooks_v3 as h
    from training.arms import native_resolution_v2 as nr
    from training.arms import hb155_crops_v1 as crops
    need(device in ('cpu', 'cuda'), 'device differs')
    if device == 'cpu':
        need(not torch.cuda.is_initialized(), 'CPU isolation failed'); torch.set_num_threads(2)
    runtime = h.configure_runtime('none' if device == 'cpu' else 'fp16', device, policy='strict')
    cfg = read(safe_path(entry['config'])); ck = safe_path(entry['checkpoint'])
    need(sha(ck) == entry['checkpoint_sha256'] and sha(safe_path(entry['config'])) == entry['config_sha256'], 'member pin differs')
    model = h.build_model(cfg['recipe']['model'], ROOT)
    if entry['arm'] == 'multiview':
        model = mv.attach_views(model)
    state = torch.load(ck, map_location='cpu', weights_only=True); identity = state['identity']
    need(identity['config_sha256'] == entry['config_sha256'] and identity['seed'] == entry['seed']
         and identity['arm'] == entry['arm'] and identity['fold'] == entry['fold'] and state['epoch'] == 19,
         'checkpoint internal identity differs')
    h.strict_load(model, state['model']); del state
    model = model.eval().to(device=device, memory_format=torch.channels_last)
    full = h.common.Corpus(NATIVE); crop = h.common.Corpus(CROP)
    zeros = {u: np.zeros(12, np.float32) for u in ids}
    ds = mv.make_dataset(NATIVE, CROP, ids, zeros, 36, False, entry['seed'], 224)
    control = crops.Windows(h.MILWindows(NATIVE, ids, zeros, 36, False, entry['seed']), 224)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=mv.collate)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.monotonic(); arrays = []; ordered = []; batch_seconds = []
    heartbeat(0, 0, 'enter')
    with torch.inference_mode():
        for batch, (x, _, uids, views) in enumerate(loader, 1):
            need(x.shape[1:] == (72, 3, 224, 224), 'multi-view K36 shape differs')
            for j in range(len(uids)):
                need(torch.equal(x[j, :36], control[len(ordered) + j][0]), 'native/control window identity differs')
            then = time.monotonic(); flipped = mv.flip_views(x, views, uids, full, crop, 36)
            arm = mv.ARM if entry['arm'] == 'multiview' else nr.ARM
            if entry['arm'] == 'control':
                x = x[:, :36]; flipped = flipped[:, :36]
            with h.autocast(device, 'none' if device == 'cpu' else 'fp16'):
                a = arm.forward(model, x.to(device), views.to(device) if entry['arm'] == 'multiview' else None)
                b = arm.forward(model, flipped.to(device), views.to(device) if entry['arm'] == 'multiview' else None)
            p = ((a.float().sigmoid() + b.float().sigmoid()) * .5).cpu().numpy()
            arrays.append(probability_array(p, uids)); ordered.extend(uids)
            if device == 'cuda': torch.cuda.synchronize()
            batch_seconds.append(time.monotonic() - then)
            if progress: heartbeat(batch, len(ordered))
    need(ordered == ids, 'worker output order differs')
    peak = torch.cuda.max_memory_reserved() if device == 'cuda' else 0
    result = dict(studies=len(ids), seconds=time.monotonic() - started, peak_reserved_bytes=peak,
        recipe=RECIPE, runtime=runtime, checkpoint_sha256=entry['checkpoint_sha256'],
        config_sha256=entry['config_sha256'], full_control_identity=True,
        batch_seconds=batch_seconds, device=device, cuda_initialized=torch.cuda.is_initialized())
    del model
    if device == 'cuda': torch.cuda.empty_cache()
    else: need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU initialized CUDA')
    return np.concatenate(arrays), result


def cpu_smoke(r):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; ids = panel_ids()
    chosen = {next(u for u in ids if f'_{s}_' in u) for s in ('L', 'R')}
    ids = [u for u in ids if u in chosen]; results = {}; predictions = {}; started = time.monotonic()
    for entry in r['members']:
        p, receipt = inference(entry, ids, device='cpu'); results[entry['key']] = receipt
        predictions[entry['key']] = p
        atomic_npz(WORK / 'cpu-smoke' / (entry['key'] + '.npz'), ids, p)
    from tools import oai_moaks_panel as panel
    y = panel.outcomes(ids)
    actual = compute_readout(ids, predictions, y, eligible=False, partners=None)
    io.atomic_json(WORK / 'cpu-smoke' / 'actual-readout.json', actual)
    # Exercise conditional proxy wiring with explicitly synthetic partner scores.
    # These are NOT authenticated OAI partner model outputs or scientific evidence.
    synthetic = dict(members={k: predictions[KEYS[i]] for i, k in enumerate(PARTNERS)})
    proxy = compute_readout(ids, predictions, y, eligible=False, partners=synthetic)
    proxy.update(engineering_only=True, synthetic_partner_fixture=True)
    io.atomic_json(WORK / 'cpu-smoke' / 'synthetic-proxy-readout.json', proxy)
    out = dict(status='CPU_SMOKE_PASS', studies=2, arms=results, seconds=time.monotonic()-started,
        cuda_visible_devices='', cuda_initialized=False, engineering_only=True,
        registration_sha256=digest(r), labels_used_in_memory_only=True, labels_exported=False,
        actual_readout_sha256=sha(WORK / 'cpu-smoke' / 'actual-readout.json'),
        synthetic_proxy_readout_sha256=sha(WORK / 'cpu-smoke' / 'synthetic-proxy-readout.json'),
        views=['fold2', 'fold3', 'pooled'], H2='UNAVAILABLE', actual_proxy='UNAVAILABLE')
    io.atomic_json(WORK / 'cpu-smoke.json', out); return out


def queue_plan():
    return [dict(tag='hb166-oai-readout-v200', command=[PYTHON, '-B', '-m',
        'tools.hb166_oai_readout_v2', 'run'], continue_on_failure=False)]


def require_queue_context():
    """Authenticate actual ancestors and gpu_job reservation, not an env flag."""
    import psutil
    from tools import job_runtime as jr
    ancestors = psutil.Process().parents(); commands = [p.cmdline() for p in ancestors]
    need(any(any(Path(s).name.lower() == 'gpu_queue.py' for s in cmd) for cmd in commands), 'gpu_queue ancestor required')
    wrapper = next((p for p, cmd in zip(ancestors, commands) if '--_run-job' in cmd
                    and any(Path(s).name.lower() == 'gpu_job.py' for s in cmd)), None)
    need(wrapper is not None, 'gpu_job owned reservation required')
    cmd = wrapper.cmdline(); tag, runid = cmd[cmd.index('--_run-job')+1:cmd.index('--_run-job')+3]
    job = jr.read_json(jr.LOGS / jr.tag_name(tag) / 'job.json')
    need(job['run_id'] == runid and job['pid'] == wrapper.pid
         and abs(job['create_time'] - wrapper.create_time()) < .01
         and job['command'] == queue_plan()[0]['command']
         and job['command_sha256'] == jr.command_digest(job['command'], ROOT), 'gpu_job reservation differs')
    return dict(tag=tag, run_id=runid, wrapper_pid=wrapper.pid)


def device_baseline():
    p = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=name,memory.used,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    need(p.returncode == 0, 'measured GPU baseline unavailable')
    name, used, total = [v.strip() for v in p.stdout.strip().split(',')]
    need('4070 Ti SUPER' in name, 'registered 4070 Ti SUPER required')
    return dict(device=name, baseline_bytes=float(used)*(1<<20), total_bytes=float(total)*(1<<20))


def worker(r, key, mode, baseline_path):
    scientific_gate(r); check_review(r); reservation = require_queue_context()
    need(os.environ.get('HB155_TOKEN') and os.environ.get('HB155_HEARTBEAT'), 'owned watchdog required')
    baseline = read(baseline_path); entry = next(e for e in r['members'] if e['key'] == key)
    need(Path(baseline_path).resolve() == (WORK / 'baselines' / f'{mode}-{key}.json').resolve()
         and baseline['registration_sha256'] == digest(r) and baseline['reservation'] == reservation
         and baseline['key'] == key and baseline['mode'] == mode, 'baseline binding differs')
    ids = panel_ids()[:4] if mode == 'qualify' else panel_ids()
    if mode == 'full':
        enforce_memory_receipts(r, baseline=baseline)
    else:
        qualification_headroom(baseline['baseline_bytes'], baseline['total_bytes'])
    p, result = inference(entry, ids, device='cuda')
    result['memory_gate'] = memory_gate(baseline['baseline_bytes'], result['peak_reserved_bytes'], baseline['total_bytes'])
    result.update(registration_sha256=digest(r), complete=True, ordered_ids_sha256=digest(ids),
                  key=key, mode=mode, oai_eligible=True)
    directory = WORK / mode; directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + '.npz'); atomic_npz(path, ids, p)
    result['prediction_sha256'] = sha(path)
    io.atomic_json(directory / (key + '.json'), result)
    return dict(status='WORKER_COMPLETE', key=key, studies=len(ids))


def enforce_memory_receipts(r, *, baseline=None):
    rates = {}
    for arm in ('multiview', 'control'):
        v = read(WORK / 'qualify' / f'{arm}-s42.json')
        need(v['registration_sha256'] == digest(r) and v['recipe'] == RECIPE and v['studies'] == 4
             and v['mode'] == 'qualify' and v['complete'] is True and v['device'] == 'cuda', 'full-shape qualification differs')
        e = next(e for e in r['members'] if e['key'] == f'{arm}-s42')
        need(v['checkpoint_sha256'] == e['checkpoint_sha256'] and v['config_sha256'] == e['config_sha256'], 'qualification checkpoint differs')
        g = v['memory_gate']; memory_gate(g['baseline_bytes'], v['peak_reserved_bytes'], g['total_bytes'])
        if baseline is not None:
            memory_gate(baseline['baseline_bytes'], v['peak_reserved_bytes'], baseline['total_bytes'])
        need(v['seconds'] > 0 and len(v['batch_seconds']) == 1 and v['batch_seconds'][0] > 0, 'qualification throughput missing')
        rates[arm] = 4 / v['seconds']
    return rates


def serial_run(r):
    scientific_gate(r); check_review(r); reservation = require_queue_context()
    from tools import hb155_watchdog_v1 as watchdog
    from tools.job_runtime import file_lock
    with file_lock(WORK / 'serial.lock', blocking=False):
        need(not (WORK / 'execution.json').exists(), 'attempt exists; no implicit retry')
        io.atomic_json(WORK / 'execution.json', dict(status='IN_FLIGHT', registration_sha256=digest(r), reservation=reservation))
        for mode, entries in (('qualify', [e for e in r['members'] if e['seed'] == 42]), ('full', r['members'])):
            rates = enforce_memory_receipts(r) if mode == 'full' else {}
            for e in entries:
                baseline = device_baseline(); bp = WORK / 'baselines' / f'{mode}-{e["key"]}.json'
                if mode == 'full': enforce_memory_receipts(r, baseline=baseline)
                else: qualification_headroom(baseline['baseline_bytes'], baseline['total_bytes'])
                baseline.update(registration_sha256=digest(r), reservation=reservation, key=e['key'], mode=mode)
                io.atomic_json(bp, baseline)
                command = [PYTHON, '-B', '-m', 'tools.hb166_oai_readout_v2', '_worker',
                    '--member', e['key'], '--mode', mode, '--baseline', str(bp)]
                code = watchdog.supervise(command, WORK / 'attempts' / f'{mode}-{e["key"]}', total=None,
                    healthy_rates={'panel': rates[e['arm']]} if mode == 'full' else None)
                need(code == 0, f'serial queue stopped after failed {mode}/{e["key"]}: {code}')
        result = readout(r)
        io.atomic_json(WORK / 'execution.json', dict(status='COMPLETE', registration_sha256=digest(r), reservation=reservation))
        return result


def readout(r):
    ids = panel_ids(); pp = {}
    for e in r['members']:
        path = WORK / 'full' / (e['key'] + '.npz'); receipt = read(path.with_suffix('.json'))
        need(receipt.get('complete') is True and receipt.get('mode') == 'full'
             and receipt.get('registration_sha256') == digest(r) and receipt.get('checkpoint_sha256') == e['checkpoint_sha256']
             and receipt.get('config_sha256') == e['config_sha256'] and receipt.get('prediction_sha256') == sha(path)
             and receipt.get('ordered_ids_sha256') == digest(ids), 'member prediction receipt differs')
        pp[e['key']] = aligned_predictions(path, ids)
    from tools import oai_moaks_panel as panel
    need(panel.validate() == ids, 'frozen label roster differs')
    partners = load_partners(r['partners'], ids)
    out = compute_readout(ids, pp, panel.outcomes(ids), eligible=r['lineage']['eligible'], partners=partners)
    out.update(registration_sha256=digest(r), lineage_verdict=r['lineage']['verdict'])
    io.atomic_json(WORK / 'readout.json', out)
    return dict(status='READOUT_COMPLETE', views=list(out['views']), eligible=out['eligible'])


def prepare(r):
    from tools.gpu_queue import validate_plan
    plan = validate_plan(queue_plan()); io.atomic_json(WORK / 'gpu-queue.json', plan)
    return dict(status='PREPARED_NO_ADMISSION', queue_jobs=len(plan), serial_members=8, lineage=r['lineage']['verdict'])


def validate(r):
    for directory in (NATIVE, CROP):
        m = read(directory / 'manifest.json')
        need(m['finished'] and np.load(directory / 'ids.npy', allow_pickle=False).tolist() == panel_ids(),
             'cache roster/completion differs')
        for name in ('ids', 'masks', *(['side_unknown'] if directory == CROP else [])):
            need(sha(directory / (name + '.npy')) == m[name + '_sha256'], 'cache index hash differs')
        for s in m['shards']:
            need(sha(directory / s['file']) == s['sha256'], 'cache pixel hash differs')
            if 'qa_file' in s:
                need(sha(directory / s['qa_file']) == s['qa_sha256'], 'cache QA hash differs')
    load_partners(r['partners'], panel_ids())
    out = dict(status='VALIDATED', studies=557, members=len(r['members']), registration_sha256=digest(r), no_gpu=True)
    io.atomic_json(WORK / 'validation.json', out); return out


def dry_run(r):
    from tools import hb155_watchdog_v1 as watchdog
    checks = {}
    for name, fn in [('lineage', lambda: scientific_gate(r)), ('review', lambda: check_review(r)), ('queue', require_queue_context)]:
        try: fn(); checks[name] = 'PASS'
        except (ValueError, FileNotFoundError) as exc: checks[name] = 'BLOCKED: ' + str(exc)
    need(watchdog.Policy(0).check(900) == 'no_heartbeat_15m', 'idle simulation differs')
    memory_gate(1<<30, 12<<30, 16<<30)
    qualification_headroom(3<<30, 16<<30)
    memory_gate(3<<30, int(10.9*(1<<30)), 16<<30)
    blocked = False
    try: memory_gate(3<<30, 12<<30, 16<<30)
    except ValueError: blocked = True
    need(blocked, 'memory gate bypass')
    result = dict(status='DRY_RUN_NO_ADMISSION', gates=checks, idle_simulation='PASS', memory_simulation='PASS',
                  gpu_started=False, admitted=False, launch=f'{PYTHON} -B tools/gpu_queue.py {rel(WORK / "gpu-queue.json")} --foreground',
                  projected_minutes=[16, 33], estimate_not_measured=True)
    io.atomic_json(WORK / 'dry-run.json', result); return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['register', 'prepare', 'validate', 'cpu-smoke', 'dry-run', 'readout', 'run', '_worker'])
    p.add_argument('--member', choices=KEYS); p.add_argument('--mode', choices=['qualify', 'full'])
    p.add_argument('--baseline', type=Path)
    a = p.parse_args(argv)
    if a.command not in ('run', '_worker'): os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if a.command == 'register':
        r = register(); result = dict(status='REGISTERED', version=VERSION, members=8, lineage=r['lineage']['verdict'])
    else:
        r = authenticate(read(WORK / 'registration.json'))
        if a.command == 'prepare': result = prepare(r)
        elif a.command == 'validate': result = validate(r)
        elif a.command == 'cpu-smoke': result = cpu_smoke(r)
        elif a.command == 'dry-run': result = dry_run(r)
        elif a.command == 'readout': result = readout(r)
        elif a.command == 'run': result = serial_run(r)
        else:
            need(a.member and a.mode and a.baseline, 'worker arguments required')
            result = worker(r, a.member, a.mode, a.baseline)
    # Summary only: no predictions, labels or detailed member data go to stdout.
    print(json.dumps({k: v for k, v in result.items() if k not in ('arms',)}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
