"""EXP126: unpushed anchor + shared-own Stage 2, fixed registered 0.7/0.3 ranks.

No local command uses CUDA or Kaggle. build only writes a fresh private kernel.
Worker is a Kaggle subprocess after the unchanged EXP105 anchor has finished.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = 'exp126-anchor-blend-v8'
V7_SHA = 'b520dbd35b0f026c00c5891406279dec2a2d0364c11dc1b4be8c8caafd72a6a9'
EXP105_SHA = '4c0d227c3d0d66d0e40821994ff125f84e78b6f964c240c5e052d06df76c3fd2'
FINAL_WEIGHTS = (0.7, 0.3)
CHILDREN = ['resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3']
MANIFEST = 'exp126-own-manifest.json'
CONFIG = 'exp126-config.json'
STAGE1_NB = ROOT / '.local/hb15-combined/knee-exp105-anchor-own-v1/knee-exp105-anchor-own-v1.ipynb'
STAGE1_SHA = 'ba8acd5ac2a8f5619ea09ae90dc548c74999ce01ce7e3bf42d8157367af12cd3'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def need(ok, message):
    if not ok:
        raise ValueError(message)

def load(name, filename, digest=None):
    path = HERE / filename
    need(digest is None or sha(path) == digest, 'Pinned source changed: ' + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def modules():
    return load('exp105_for_exp126', 'exp105_combined_release.py', EXP105_SHA), load('exp119_for_exp126', 'exp119_release_v7.py', V7_SHA)

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')

def projection(studies=1300, own='rad5', tta=False, anchor_seconds=26404.0, fallback=0, rad_k=36):
    """D-B-018 anchor scenario plus corrected D-B-048 screen costs; no repeated decode.

    Own models share GPU0 sequentially after BOTH anchor GPUs have completed.
    GPU1 stays idle after anchor; splitting requires separate workers and is not claimed.
    Default anchor 26404 is a scenario, not a measured hidden-cohort upper bound.
    """
    need(studies > 0 and own in ('rad5', 'coat5rad5') and rad_k in (24, 36), 'Invalid projection inputs')
    # K24 uses the same checkpoints and fixed blend, with 2/3 as many windows.
    # This is a linear cost projection, not a measured hidden-cohort runtime.
    costs = {'rad5': 1.3 * 5 * 0.25 * (rad_k / 36) * studies * (2 if tta else 1)}
    if own == 'coat5rad5':
        costs['coat5'] = 1.3 * 5 * 0.35 * studies
    decode = 1.3 * 6.52503319258 * fallback
    # 600 seconds is conservative install/load/audit/final publication allowance.
    total = anchor_seconds + sum(costs.values()) + decode + 600.0
    margin = 32400.0 - total
    return dict(studies=studies, own=own, rad_k=rad_k, plane_flip_tta=tta, anchor_scenario_seconds=anchor_seconds,
                own_model_seconds=costs, fallback_decode_seconds=decode, fixed_seconds=600,
                total_seconds=total, margin_seconds=margin, minimum_margin_seconds=3600,
                fits_one_hour_margin=margin >= 3600,
                scope='Arithmetic screen scenario; actual launch checks elapsed anchor and usable store count',
                gpu_schedule='Anchor unchanged on GPU0+GPU1; then one v7 shared worker with all own arms on GPU0; GPU1 idle',
                cheapest_variant='rad5 K24, plane-flip-tta off; require actual elapsed+projection <=28800s')

def anchor_gate(record):
    need(record.get('ok') is True and record.get('members') == CHILDREN
         and record.get('audit_members') == 4 and record.get('audit_status') == 'COMPLETE'
         and record.get('family_reduction') == 'rank_of_member_probability_mean'
         and not record.get('failure_events'),
         'four-anchor-child hard gate failed')

def plane_flags(centers, spec):
    """EXP124 plane_ranges/flip_mask semantics, without importing its training modules."""
    slots = spec.get('slots'); ranges = []; start = 0
    need(isinstance(slots, list) and slots, 'Missing plane slots')
    for item in slots:
        need(isinstance(item, (list, tuple)) and len(item) == 3, 'Malformed plane slot')
        plane, _, count = item
        need(plane in ('Sagittal', 'Coronal', 'Axial') and type(count) is int and count >= 3, 'Ambiguous plane')
        ranges.append((plane, start, start + count)); start += count
    need(start == spec.get('depth') and {p for p, _, _ in ranges} == {'Sagittal', 'Coronal', 'Axial'}, 'Invalid plane coverage')
    flags = []
    for c in centers:
        if c == -1:
            flags.append(False); continue
        hits = [p for p, lo, hi in ranges if lo + 1 <= c < hi - 1]
        need(len(hits) == 1, 'Window crosses plane boundary')
        flags.append(hits[0] in ('Coronal', 'Axial'))
    return flags

class TTARuntime:
    """Proxy keeps v6's serving_input swap and frozen predict path for both passes.

    Rad center3 only. Plain and mirrored served probabilities are averaged in fp32.
    Mirror applies after normalisation; horizontal mirroring commutes with it.
    """
    def __init__(self, runtime, enabled):
        self.runtime, self.enabled = runtime, enabled
        self.prepare_input = runtime.prepare_input

    def predict(self, model, corpus, ids, k, res, amp, device, **kw):
        import numpy as np
        from raptor_common import select_centers
        original = self.runtime.prepare_input
        try:
            self.runtime.prepare_input = self.prepare_input
            plain, receipt = self.runtime.predict(model, corpus, ids, k, res, amp, device, **kw)
            # v6 center3 adapter is present only inside the center3 context.
            is_rad = getattr(self.prepare_input, '__name__', '') == 'center3_prepare_input'
            if not self.enabled or not is_rad:
                return plain, receipt
            def mirrored(view, uid, k_eval, resolution, norm, window_policy='legacy'):
                x = self.prepare_input(view, uid, k_eval, resolution, norm, window_policy)
                centers = select_centers(view.mask(uid), view.depth, k_eval, view.spec, policy=window_policy)
                flags = plane_flags(centers, dict(view.spec, depth=view.depth))
                out = x.copy()
                for i, yes in enumerate(flags):
                    if yes:
                        out[i] = out[i, ..., ::-1]
                return np.ascontiguousarray(out)
            self.runtime.prepare_input = mirrored
            flipped, second = self.runtime.predict(model, corpus, ids, k, res, amp, device, **kw)
            receipt = dict(receipt, plane_flip_tta=True, mirrored_pass=second,
                           seconds=receipt['seconds'] + second['seconds'])
            return (plain + flipped) / np.float32(2), receipt
        finally:
            self.runtime.prepare_input = original

def worker(a):
    """Explicit CPU mode refuses CUDA initialisation; production runs only inside kernel."""
    e105, v7 = modules(); v6 = v7.v6
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=SCHEMA + '-worker', status='FAIL', device=a.device, amp=a.amp,
                   release_manifest_sha256=a.sha256, unscored={}, arms={})
    try:
        if a.cpus and hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, {int(c) for c in a.cpus.split(',')})
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', KNEE_WORKSPACE_ROOT=a.assets)
        sys.path[:0] = [a.env, str(Path(a.assets) / 'training/src'), str(HERE)]
        release = v7.checked_manifest(a.manifest, a.sha256)
        roots = {}
        for entry in release['asset_sets']:
            d = entry['dataset']; inp = Path(a.kaggle_root) / 'input'
            roots[d] = str(e105.find_one([inp / 'datasets' / d, inp / d.split('/')[1]], v6.NEW_MANIFEST))
        v6.verify_old_assets(a.assets)
        v7.verify_asset_sets(release, roots, release['arms'])
        sys.path.insert(1, str(v7.code_root(release, roots) / v6.CODE_DIR))
        import torch
        if a.device == 'cpu':
            def refuse(*args, **kwargs):
                raise RuntimeError('CUDA forbidden for EXP126 CPU mode')
            torch.cuda._lazy_init = refuse
            torch.cuda.is_available = lambda: False
            need(a.amp == 'none', 'CPU requires amp none')
        else:
            need(sys.platform.startswith('linux') and torch.cuda.device_count() == 1
                 and 'T4' in torch.cuda.get_device_name(0), 'One visible T4 required')
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        import cv2; cv2.setNumThreads(1)
        import timm, pydicom
        if a.device == 'cuda':
            import gdcm
            need(sys.version_info[:2] == (3, 12) and
                 (torch.__version__, timm.__version__, cv2.__version__, pydicom.__version__, gdcm.Version.GetVersion())
                 == v6.VERSIONS, 'Pinned Linux CPython 3.12 serving versions required')
        import pandas as pd
        import raptor_runtime as rt
        from raptor_dicom import DirectoryDicomSource
        from raptor_shared_input import StoreCorpus
        rt.configure_runtime(a.amp, a.device, 'strict')
        ids = json.loads(Path(a.ids_json).read_text(encoding='utf-8'))
        need(isinstance(ids, list) and ids and len(ids) == len(set(ids)) and all(isinstance(u, str) and u for u in ids),
             'study IDs malformed')
        rows = pd.read_csv(a.rows_csv, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str}).to_dict('records')
        corpus = StoreCorpus(a.store, ids, rows, DirectoryDicomSource(a.dicom_root), v7.DECODE_CONTRACT, a.split, audit=2)
        served = v7.serve_shared(release['arms'], {x['name']: roots[release['arm_assets'][x['name']]] for x in release['arms']},
                                 corpus, ids, a.amp, a.device, runtime_module=TTARuntime(rt, a.plane_flip_tta),
                                 skip_failed_inputs=True)
        stats = corpus.stats()
        need(stats['trusted'] is True, 'Shared-store audit failure')
        for arm in release['arms']:
            name = arm['name']; path = out / (name + '.csv')
            e105.write_csv(v6, path, served['scored_ids'], served['probs'][name].T.tolist())
            receipt['arms'][name] = dict(arm=arm, checkpoints=served['checks'][name],
                                        timing=served['timing'][name], csv_sha256=sha(path), csv=path.name)
        receipt.update(status='PASS', scored_ids=served['scored_ids'], unscored=served['unscored'],
                       studies=len(ids), ids_sha256=e105.ids_sha256(ids), decode=served['decode'], store=stats,
                       plane_flip_tta=a.plane_flip_tta, training=0, scoring=0, label_reads=0, retries=0)
        code = 0
    except BaseException as exc:
        receipt['error'] = str(exc); code = 1
    write(out / 'receipt.json', receipt)
    return code

def checked_own(release, digest, directory, ids, exit_code, plane_flip_tta=False):
    e105, v7 = modules(); d = Path(directory)
    r = json.loads((d / 'receipt.json').read_text(encoding='utf-8'))
    need(exit_code == 0 and r.get('status') == 'PASS', 'own worker failed')
    need(r.get('plane_flip_tta') is plane_flip_tta, 'Worker TTA policy differs')
    need(r.get('release_manifest_sha256') == digest and r.get('ids_sha256') == e105.ids_sha256(ids)
         and r.get('studies') == len(ids), 'Worker release/cohort binding failed')
    scored, missed = r['scored_ids'], r['unscored']
    need(len(scored) == len(set(scored)) and set(scored).isdisjoint(missed)
         and set(scored) | set(missed) == set(ids), 'Worker study coverage failed')
    need(r['store']['trusted'] is True and bool(scored), 'Untrusted/empty own output')
    need(set(r['arms']) == {x['name'] for x in release['arms']}, 'Arm set differs')
    tables = []
    for arm in release['arms']:
        item = r['arms'][arm['name']]; path = d / item['csv']
        need(item['arm'] == arm and sha(path) == item['csv_sha256'], 'Arm manifest/CSV binding failed')
        checks = item['checkpoints']
        need(len(checks) == len(arm['checkpoints']) and
             all(c.get('sha256') == ck['sha256'] and c.get('name') == ck['name']
                 and c.get('k_eval_release') == arm['k_eval'] for c, ck in zip(checks, arm['checkpoints'])),
             'Checkpoint binding missing/different')
        rows = e105.read_anchor(path, scored)
        tables.append(rows)
    if len(tables) == 1:
        own = tables[0]
    else:
        cols = v7.v6.rank_mean(tables, [1] * len(tables))
        own = [list(x) for x in zip(*cols)]
    index = {u: i for i, u in enumerate(ids)}
    return {index[u]: row for u, row in zip(scored, own)}, r

def post_anchor(ns, environ=None):
    """Never raises; any hard-gate/binding/timeout failure preserves anchor bytes."""
    import subprocess
    env = os.environ if environ is None else environ
    e105, v7 = modules()
    work = Path(env.get('HB15_WORKING', '/kaggle/working')); out = work / 'exp126'
    out.mkdir(parents=True, exist_ok=True)
    final = work / 'submission.csv'; backup = out / 'anchor_submission.csv'
    r = dict(schema=SCHEMA, path='anchor_only', final_weights=list(FINAL_WEIGHTS), reason=None)
    anchor_sha = None
    try:
        anchor_sha = sha(final); shutil.copyfile(final, backup)
        r['anchor_sha256'] = anchor_sha
        r['anchor'] = e105.coat_family_record(ns, work)
        setup = ns.get('_HB15_SETUP') or {}; need(setup.get('ok') is True, 'setup failed')
        hook = ns.get('_HB15') or {}; need(callable(hook.get('finalize')), 'shared hook missing')
        summary = hook['finalize'](120); need(summary.get('thread_finished') is True, 'hook did not finish')
        r['hook'] = summary
        need(not summary.get('hook_errors') and not summary.get('anchor_raised') and not summary.get('dropped_ids'),
             'shared hook errors, dropped studies, or anchor exceptions')
        anchor_gate(r['anchor'])
        src = Path(env['HB15_SRC']); conf = json.loads((src / CONFIG).read_text(encoding='utf-8'))
        need(sha(src / MANIFEST) == conf['manifest_sha256'], 'own manifest changed')
        release = v7.checked_manifest(src / MANIFEST, conf['manifest_sha256'])
        comp = Path(env.get('RSNA_COMP_ROOT') or '/kaggle/input/competitions/rsna-knee-abnormality-detection')
        ids = e105.read_sample(comp); anchor = e105.read_anchor(final, ids)
        need(set(map(str, ns.get('_RSNA_TEST_IDS', []))) == set(ids), 'anchor cohort differs')
        store_ok = int(((summary.get('store') or {}).get('status_counts') or {}).get('OK', 0))
        elapsed = time.time() - float(setup['t0'])
        p = projection(len(ids), conf['own'], conf['plane_flip_tta'], elapsed,
                       max(0, len(ids)-store_ok), conf['rad_k'])
        r['runtime'] = p
        need(p['fits_one_hour_margin'], 'runtime margin below 60 min')
        hidden = projection(1300, conf['own'], conf['plane_flip_tta'],
                            float(conf.get('anchor_projection_seconds', 26404)), rad_k=conf['rad_k'])
        r['hidden_runtime'] = hidden
        if len(ids) > 3:
            need(hidden['fits_one_hour_margin'], '1300-study hidden runtime projection margin below 60 min')
        r['environment'] = e105.install_env(v7.v6, env['HB15_ASSETS'], env['HB15_OPENCV'], env['HB15_ENV'])
        write(out / 'ids.json', ids)
        cmd = [sys.executable, str(src / Path(__file__).name), 'worker', '--manifest', str(src / MANIFEST),
               '--sha256', conf['manifest_sha256'], '--assets', env['HB15_ASSETS'], '--env', env['HB15_ENV'],
               '--store', env['HB15_STORE'], '--ids-json', str(out / 'ids.json'), '--out', str(out / 'shared'),
               '--rows-csv', env.get('HB15_ROWS_CSV') or str(comp / 'test_series.csv'),
               '--dicom-root', env.get('HB15_DICOM_ROOT') or str(comp / 'test_series'),
               '--split', env.get('HB15_SPLIT', 'test'), '--device', 'cuda', '--amp', 'fp16']
        if conf['plane_flip_tta']: cmd.append('--plane-flip-tta')
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
        if allowed:
            cmd.extend(['--cpus', ','.join(map(str, allowed[:2]))])
        child = dict(env, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        # Reserve the same 600 s for receipt checks and final CSV publication.
        timeout = max(1, 28800 - p['fixed_seconds'] - (time.time() - float(setup['t0'])))
        with (out / 'worker.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(cmd, env=child, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        own, worker_r = checked_own(release, conf['manifest_sha256'], out / 'shared', ids, result.returncode,
                                   conf['plane_flip_tta'])
        need(len(ids)-len(own) <= math.floor(0.05*len(ids)), 'unscored >5%')
        cols = e105.final_blend(v7.v6, anchor, own, FINAL_WEIGHTS)
        digest = e105.publish_blend(v7.v6, final, ids, cols, anchor_sha, backup)
        r.update(path='blended', blended_sha256=digest, worker=worker_r,
                 release_manifest_sha256=conf['manifest_sha256'], own_weights=release['rank_weights'])
    except BaseException as exc:
        r['reason'] = str(exc)
        if anchor_sha is not None and final.exists() and sha(final) != anchor_sha:
            e105.restore_anchor(final, backup, anchor_sha)
    r['final_sha256'] = sha(final) if final.exists() else None
    r['final_is_anchor'] = r['final_sha256'] == anchor_sha
    write(out / 'receipt.json', r)
    return r

def build(a):
    e105, v7 = modules(); out = Path(a.out)
    rad_k = getattr(a, 'rad_k', 36)
    need(rad_k in (24, 36), 'RadR50 K must be 24 or 36')
    need(not out.exists(), 'Fresh output directory required')
    need(sha(STAGE1_NB) == STAGE1_SHA, 'Scored EXP105 stage-1 notebook changed')
    stage1 = json.loads(STAGE1_NB.read_text(encoding='utf-8'))
    base = ROOT / '.local/exp119-release-v7/coat5rad5/release-manifest.json'
    release = v7.checked_manifest(base, sha(base))
    release = copy.deepcopy(release)
    names = ['rad5'] if a.own == 'rad5' else ['coat5', 'rad5']
    release['arms'] = [x for x in release['arms'] if x['name'] in names]
    for arm in release['arms']:
        arm['weight'] = 1.0
        if arm['name'] == 'rad5':
            arm['k_eval'] = rad_k
            arm['model_seconds_per_study'] *= rad_k / 36
    release['arm_assets'] = {x['name']: release['arm_assets'][x['name']] for x in release['arms']}
    release['asset_sets'] = [x for x in release['asset_sets'] if x['dataset'] in set(release['arm_assets'].values())]
    release['asset_packages'] = {k: v for k, v in release.get('asset_packages', {}).items()
                                 if k in set(release['arm_assets'].values())}
    release['arm_specs'] = [x.replace('@36:', '@24:') if rad_k == 24 and x.startswith('rad5=') else x
                            for x in release.get('arm_specs', []) if x.split('=', 1)[0] in names]
    release['rank_weights'] = v7.v6.rank_weights(release['arms'])
    release['release'] = 'exp126-' + a.own + ('-k24' if rad_k == 24 else '') + ('-tta' if a.plane_flip_tta else '')
    release['combination'] = ('single arm: raw mean_prob probabilities' if len(names) == 1 else
                              'per-target mean of average-tie percentile ranks, equal arm weights (EXP098)')
    v7.validate_release(release)
    text = json.dumps(release, indent=2) + '\n'
    digest = hashlib.sha256(text.encode()).hexdigest()
    conf = dict(own=a.own, rad_k=rad_k, plane_flip_tta=a.plane_flip_tta, manifest_sha256=digest,
                anchor_projection_seconds=a.anchor_seconds, final_weights=list(FINAL_WEIGHTS))
    sources = dict(e105.SOURCES)
    for name in ('exp119_release_v7.py', 'exp111_release_v6.py', Path(__file__).name):
        sources[name] = (HERE / name, V7_SHA if name == 'exp119_release_v7.py' else None)
    out.mkdir(parents=True)
    (out / MANIFEST).write_text(text, encoding='utf-8', newline='\n'); write(out / CONFIG, conf)
    sources[MANIFEST] = (out / MANIFEST, digest); sources[CONFIG] = (out / CONFIG, None)
    e105.SOURCES = sources
    nb, meta, record = e105.build_notebook('full')
    need(len(nb['cells']) == len(stage1['cells']) and nb['cells'][1:-1] == stage1['cells'][1:-1],
         'EXP105 anchor/hook path differs from scored stage-1 notebook')
    post = "import importlib.util\nfrom pathlib import Path\ntry:\n    _spec126 = importlib.util.spec_from_file_location('exp126', Path(_HB15_SETUP['src']) / 'exp126_anchor_blend_v8.py')\n    _module126 = importlib.util.module_from_spec(_spec126)\n    _spec126.loader.exec_module(_module126)\n    _EXP126 = _module126.post_anchor(globals())\n    print(_EXP126['path'], _EXP126['reason'])\nexcept BaseException as _exc126:\n    print('EXP126 anchor-only:', type(_exc126).__name__, str(_exc126))\n"
    nb['cells'][-1] = e105.code_cell(post)
    slug = 'knee-exp126-' + a.own + ('-k24' if rad_k == 24 else '') + ('-tta' if a.plane_flip_tta else '') + '-v8'
    meta.update(id='taeyangg4/' + slug, title=slug, code_file='release.ipynb', is_private=True)
    meta['dataset_sources'] = list(dict.fromkeys(meta['dataset_sources'] + [x['dataset'] for x in release['asset_sets']]))
    write(out / 'release.ipynb', nb); write(out / 'kernel-metadata.json', meta)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code': compile(e105.cell_text(cell), 'cell%d' % i, 'exec')
    record.update(schema=SCHEMA, state='STAGED_ONLY_NOT_PUSHED', kernel=meta['id'],
                  final_weights=list(FINAL_WEIGHTS), own=conf, anchor_cells_unchanged=True,
                  runtime=projection(1300, a.own, a.plane_flip_tta, a.anchor_seconds, rad_k=rad_k),
                  scored_stage1_notebook_sha256=STAGE1_SHA,
                  notebook_sha256=sha(out / 'release.ipynb'), metadata_sha256=sha(out / 'kernel-metadata.json'),
                  post_sha256=hashlib.sha256(post.encode()).hexdigest(),
                  parity_scope='Existing real 24-study v7 plain-arm parity is reused; fresh EXP126/TTA real-model parity remains required',
                  push_argv_not_run=[str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / 'tools/kaggle_push.py'), str(out.resolve()), '--accelerator', 'NvidiaTeslaT4'])
    record.pop('submit_argv_template_not_run', None)
    write(out / 'execution-plan.json', record)
    print(json.dumps(dict(kernel=meta['id'], out=str(out), runtime=record['runtime'])))
    return record

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--own', choices=['rad5', 'coat5rad5'], default='rad5')
    b.add_argument('--plane-flip-tta', action='store_true'); b.add_argument('--anchor-seconds', type=float, default=26404)
    b.add_argument('--rad-k', type=int, choices=(24, 36), default=36)
    b.add_argument('--out', required=True)
    w = sub.add_parser('worker')
    for key in ('manifest', 'sha256', 'assets', 'env', 'store', 'ids-json', 'rows-csv', 'dicom-root', 'out'):
        w.add_argument('--'+key, required=True)
    w.add_argument('--device', choices=['cpu', 'cuda'], default='cpu'); w.add_argument('--amp', default='none')
    w.add_argument('--split', default='test'); w.add_argument('--kaggle-root', default='/kaggle')
    w.add_argument('--plane-flip-tta', action='store_true')
    w.add_argument('--cpus', default='')
    a = p.parse_args(argv)
    if a.command == 'worker': return worker(a)
    build(a); return 0

if __name__ == '__main__':
    sys.exit(main())
