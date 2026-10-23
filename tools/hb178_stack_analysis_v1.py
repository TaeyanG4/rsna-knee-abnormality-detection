"""H-B-178 v1.0.0: CPU-only paired saved-prediction factorial analysis.

New tool; no original, frozen stage or receipt edits. Reuses authenticated
HB147/HB158, HB105 aggregation, and registered multi-lens bootstrap helpers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / '.local/scratch/astra-b5/hb178'
VERSION = '1.0.0'
ARMS = ('baseline', 'native', 'scale', 'stack')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def probs(value, n):
    import numpy as np
    a = np.asarray(value)
    need(a.shape == (n, 12) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(),
         'invalid probability shape/range/finite')
    return a


def align(source, wanted, value):
    need(len(source) == len(set(source)) and len(wanted) == len(set(wanted))
         and set(source) == set(wanted), 'duplicate or unequal UID rosters')
    a = probs(value, len(source)); ix = {u: i for i, u in enumerate(source)}
    return a[[ix[u] for u in wanted]]


def common_folds(support):
    need(bool(support), 'no fold support')
    return sorted(set.intersection(*(set(v) for v in support.values())))


def factorial(q0, q1, c0, c1):
    from tools.multi_lens_readout import rank
    n = len(q0)
    q0, q1, c0, c1 = [rank(probs(x, n)) for x in (q0, q1, c0, c1)]
    return dict(baseline=(q0+c0)/2, native=(q1+c0)/2,
                scale=(q0+c1)/2, stack=(q1+c1)/2)


def contrasts(scores):
    need(set(scores) == set(ARMS), 'factorial arm set differs')
    b, n, c, s = [scores[k] for k in ARMS]
    return dict(native_minus_baseline=n-b, scale_minus_baseline=c-b,
                stack_minus_baseline=s-b, stack_minus_native=s-n,
                stack_minus_scale=s-c, interaction=s-n-c+b)


def statistic(values):
    from tools.multi_lens_readout import stat
    result = stat(values)
    # The inherited point-threshold PASS is not a CI decision.
    result.pop('PASS'); result.pop('threshold'); result.pop('eligible')
    return result


def summarize(scores, panel=False):
    import numpy as np
    from tools import multi_lens_readout as ml
    from tools.lens_completion_1015_v1 import PANEL_TARGETS
    output = {}
    ds = contrasts(scores)
    for name, d in ds.items():
        if panel:
            primary = d[:, ml.WEAK].mean(1)
            detail = dict(mapped_nine=statistic(d[:, PANEL_TARGETS].mean(1)),
                          target_point=ml.clean(d[0]))
        else:
            primary = d[:, :2].mean((1, 2))
            detail = dict(ruler_macros={r: statistic(d[:, i].mean(1)) for i, r in enumerate(ml.RULERS)},
                          ruler_target_point=ml.clean(d[0]))
        output[name] = dict(primary=statistic(primary), **detail)
    cells = ds['stack_minus_baseline'][0]
    if panel:
        cells = cells[list(PANEL_TARGETS)]
    complete = all(output[k]['primary']['valid_draws'] == 300 and
                   output[k]['primary']['value'] is not None for k in output)
    positive = all(output[k]['primary']['ci95'] is not None and
                   output[k]['primary']['ci95'][0] > 0
                   for k in ('stack_minus_native', 'stack_minus_scale'))
    floor = bool(np.isfinite(cells).all() and (cells >= -.003).all())
    return dict(contrasts=output, all_draws_valid=complete, floor_pass=floor,
                minimum_stack_baseline_cell=float(cells.min()),
                exploratory_consistency=bool(complete and positive and floor),
                release_authorized=False,
                arm_primary_auc={k: float((a[0, ml.WEAK] if panel else a[0, :2]).mean())
                                 for k, a in scores.items()})


def read_inputs():
    """Authenticate model identities before HB105 installs its no-load tripwire."""
    import numpy as np
    from tools import hb147_lenses_v1 as h, lens_completion_1015_v1 as L
    from tools import forensics_1015_v1 as F
    from tools import multi_lens_readout as ml
    from tools import hb145_output_io as io
    ids, views, scale_binding = L.hb158_predictions()
    need(len(ids) == len(set(ids)) == 888, 'expected unique fold0 roster')
    reg = h.registered()
    native = {}
    for r in (224, 320):
        native[r] = {}
        for s in (42, 43):
            m = reg['members'][h.key(0, r, s)]
            need(m['targets'] == list(ml.TARGETS) and m['oai_ancestry']['eligible'], 'native target/ancestry')
            native[r][s] = align(m['ids'], ids, h.predictions(m))
    h.completed('full')
    F.lens()  # select the actual completed v111 forensic job, not old v110
    panel_rows = L.gpu_rows('release-full')
    ancestry = L.oai_ancestry()
    need(ancestry['eligible'], 'OAI EXP181 ancestry not eligible')
    oids, oy = L.oai_panel()
    tables = {}
    for r in (224, 320):
        values = {}
        for s in (42, 43):
            m = reg['members'][h.key(0, r, s)]
            row = io.read_json(h.WORK/'gpu/full/journal'/f'{m["key"]}-oai557.json')
            need(row['ids'] == oids and row['checkpoint_sha256'] == m['checkpoint_sha256']
                 and row['registration_sha256'] == sha(h.WORK/'registration.json'), 'native panel identity')
            values[str(s)] = probs(row['probabilities'], len(oids))
        tables[r] = values
    oi = L.image_item(oids, tables[320], tables[224], oy, True, 'participant singletons')
    cv = {}
    for view in (320, 384):
        row = panel_rows[f'{L.PARENT_KEY}-{view}-oai557.json']
        need(row['checkpoint_sha256'] == ancestry['checkpoint_sha256'], 'scale OAI checkpoint differs')
        cv[view] = probs(L.panel_values(panel_rows, L.PARENT_KEY, view, 'oai557', oids), len(oids))
    pins = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (
        h.WORK/'registration.json', h.WORK/'gpu/full/controller.json',
        F.WORK/'gpu/release-full/controller.json', F.WORK/'lenses/registration.json',
        ROOT/'artifacts/experiments/OAI_MOAKS_PANEL_V1/registration.json')}
    return ids, views, native, oi, cv, dict(pins=pins, scale=scale_binding, oai_ancestry=ancestry)


def execute():
    import numpy as np
    import torch
    from tools import multi_lens_readout as ml, hb145_output_io as io
    pre = io.read_json(WORK/'pre-review.json')
    need(pre.get('verdict') in ('PASS', 'PASS-WITH-NOTES') and pre.get('independent') is True
         and pre.get('spec_sha256') == sha(WORK/'SPEC.md'), 'bound independent prereview required')
    review_path = (ROOT/pre['evidence_path']).resolve()
    need(review_path.is_relative_to(WORK) and sha(review_path) == pre['evidence_sha256'], 'prereview evidence drift')
    ids, views, native, oi, cv, inputs = read_inputs()
    from tools import hb105_aligned_oof_v1 as A
    raw, rosters, auth = A.authenticate()
    q = {}
    for r in (224, 320, 'original'):
        changed = {f: dict(v) for f, v in raw.items()}
        if r != 'original':
            for s in (42, 43):
                changed[0]['S'+str(s)] = align(ids, rosters[0]['val'], native[r][s]).astype(np.float32)
        pooled = A.variants(changed)['replace'][:len(ids)]
        q[r] = align(rosters[0]['val'], ids, pooled)
    ctx = ml.make_context(ids, ROOT)
    result = dict(version=VERSION, status='RESULT_REVIEW_REQUIRED', clean_oof=False,
        exposure='EXPOSED_SCREEN', submission_authorized=False,
        full_stack_common_folds=common_folds(dict(native=(0,1), scale=(0,), multiview=(2,3))),
        partial_stack_common_folds=[0], rows=len(ids), ordered_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        policy='HB158 L6 50:50 proxy; matched native224 primary; original HB105 sensitivity; not deployed Q_A5q',
        bootstrap=dict(draws=300, seed=20261003, unit='connected components', paired=True),
        inputs=inputs, aligned_bindings=auth['bindings'],
        sources=A.source_pins(), spec_sha256=sha(WORK/'SPEC.md'), prereview_sha256=sha(WORK/'pre-review.json'))
    for name, base in (('l6_matched',224), ('l6_original_sensitivity','original')):
        arrays = factorial(q[base], q[320], views[320], (views[320]+views[384])/2)
        result[name] = summarize({k:ctx.score(v) for k,v in arrays.items()})
    arrays = factorial(oi['control'], oi['arm'], cv[320], (cv[320]+cv[384])/2)
    w = ml.weights(oi['ids'], oi['components'])
    result['oai_two_family_diagnostic'] = dict(rows=len(oi['ids']),
        policy='50:50 rank(native fold0 seed mean)/rank(EXP181); weak-four; NOT L6/full O',
        **summarize({k:ml.auc_draws(v,oi['labels'],oi['mask'],w) for k,v in arrays.items()},panel=True))
    need(not torch.cuda.is_initialized(), 'CUDA initialized')
    result['cuda_initialized'] = False
    io.atomic_json(WORK/'result.json',ml.clean(result))
    return dict(status=result['status'],rows=len(ids),oai_rows=len(oi['ids']),cuda_initialized=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['analyze'])
    p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    print(json.dumps(execute()))


if __name__ == '__main__':
    main()
