"""H-B-179 real two-study composition replay v1.0.0; no CUDA/notebook execution.

Reuses authenticated completed H174 R6 CPU inference; independently executes the
candidate mirror pair and H170 plain/mirror oracle on the same full studies.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r224r6mirfq_v1132 as b
from tools import c85r7_r6_cpu_parity as rp
from tools import c85r7_r224_timing as t
from tools import hb145_output_io as io

R6_PREDICTION_SHA = '7e6ab3677c38dcbc9985fcd1414f7924db86bfe305ce0c4af778cd33c425a373'
DATA = ROOT / '.local/exp189-cpu-v120/input/visible_data/test_series'


def namespace(raw):
    ns = t.public_namespace()
    tree = ast.parse(raw)
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in t.CONSTANTS}
    b.old.need(constants == t.source_parts()[0], 'mirror CPU recipe drift')
    allowed = set(t.FUNCTIONS) | {'_slot_index', '_mirror_tri', 'infer_pair_probs'}
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in allowed
        or isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in ('SLOT_SIZES', 'SAGITTAL_SLOTS') for x in n.targets)]
    b.old.need({n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} == allowed, 'AST mirror allowlist')
    exec(compile(tree, '<hb179-mirror-allowlist>', 'exec'), ns)
    return ns


def predictions(path, ids, labels):
    import numpy as np
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        b.old.need(reader.fieldnames == ['StudyInstanceUID', *labels], 'prediction target order')
        rows = list(reader)
    found = [r['StudyInstanceUID'] for r in rows]
    b.old.need(len(found) == len(set(found)) == len(ids) and set(found) == set(ids), 'prediction cohort/duplicates')
    indexed = {r['StudyInstanceUID']: r for r in rows}
    values = np.asarray([[float(indexed[uid][label]) for label in labels] for uid in ids])
    b.old.need(values.shape == (len(ids), 12) and np.isfinite(values).all()
        and ((values >= 0) & (values <= 1)).all(), 'prediction probability domain')
    return values


def carry_r6(files, refs):
    import numpy as np
    paths = sorted(p for p in DATA.iterdir() if p.is_dir())[:2]
    b.old.need(len(paths) == 2, 'two real studies required')
    pins = {str(p.relative_to(ROOT)): t.sha(p) for study in paths for p in sorted(study.rglob('*.dcm'))}
    b.old.need(len(pins) == 419, 'full real DICOM inventory')
    binding = dict(studies=[p.name for p in paths], dicom_sha256=pins,
        series_sha256=t.sha(DATA.parent / 'test_series.csv'), public_notebook_sha256=t.sha(b.r6.PUBLIC_NOTEBOOK),
        kernel_sha256=b.validate_r6(), tool_sha256=t.sha(Path(rp.__file__)))
    b.old.need(binding['kernel_sha256'] == b.R6_KERNEL, 'R6 source kernel identity')
    for name in (b.r6.PORT, b.r6.CHAIN, 'exp189_s6_port.py', 'exp169_r7_worker.py', *[k for k in files if k.startswith('exp169_r7_src/')]):
        if name in files:
            b.old.need(files[name] == refs['r6']['files'][name], 'R6 carry-forward source drift')
    port = b.r6.port_module()
    manifest, identity = port.verify_bundle(b.r6.BUNDLE, source_dir=b.r6.WORK / 'runtime/exp169_r7_src')
    b.old.need(port.worker_args(manifest, b.r6.BUNDLE, rp.d.ANATOMY) == rp.public_args(manifest, b.r6.BUNDLE, rp.d.ANATOMY), 'R6 public worker arguments')
    evidence = dict(binding['dicom_sha256'])
    source_result = b.r6.WORK / 'cpu-parity/result.json'
    result = b.old.read(source_result)
    b.old.need(result['state'] == 'PASS_REAL_FULL_STUDY_CPU_PARITY' and result['binding'] == binding
        and result['max_abs_delta'] == 0 and result['folds'] == 4, 'completed R6 result binding')
    evidence[str(source_result.relative_to(ROOT))] = t.sha(source_result)
    evidence[str((DATA.parent / 'test_series.csv').relative_to(ROOT))] = binding['series_sha256']
    for name, digest in identity['file_sha256'].items():
        evidence[str((b.r6.BUNDLE / name).relative_to(ROOT))] = digest
    arrays = []
    for arm in ('serving', 'public'):
        out = b.r6.WORK / 'cpu-parity' / arm
        rec = b.old.read(out / 'parity-receipt.json')
        b.old.need(rec['state'] == 'COMPLETE_CPU' and rec['binding'] == binding and rec['rows'] == 2
            and rec['folds'] == 4 and rec['flip'] is True and rec['full_studies'] is True, 'R6 incomplete/stale replay')
        b.old.need(rec['prediction_sha256'] == t.sha(out / 'predictions.csv') == R6_PREDICTION_SHA, 'R6 prediction pin')
        b.old.need(b.old.read(out / 'controller/submission_receipt.json') == rec['controller_receipt'], 'R6 controller receipt drift')
        for name in ('parity-receipt.json', 'predictions.csv', 'controller/submission_receipt.json', 'controller/worker0/receipt.json'):
            evidence[str((out / name).relative_to(ROOT))] = t.sha(out / name)
        arrays.append(predictions(out / 'predictions.csv', binding['studies'], list(port.LABELS)))
    b.old.need(np.array_equal(*arrays), 'R6 reference differs')
    return binding['studies'], list(port.LABELS), arrays, evidence


def rank(values):
    import pandas as pd
    return pd.DataFrame(values).rank(method='average', pct=True).to_numpy()


def composed(candidate, reference, r6, r6_reference, mirror, mirror_reference, ids):
    """Changed terms are real; unchanged R384/O fixtures are explicitly synthetic."""
    import numpy as np
    real = .175 * rank(mirror) + .5 * rank(r6)
    expected = .175 * rank(mirror_reference) + .5 * rank(r6_reference)
    b.old.need(np.array_equal(real, expected), 'actual changed-arm weighted contribution')
    rng = np.random.default_rng(179)
    cases = []
    for index in range(8):
        r, e143, e155, q = (rng.uniform(.01, .99, (len(ids), 12)) for _ in range(4))
        actual = candidate['blend'](ids, r, mirror, r6, e143, e155, q)
        ref = reference['blend'](ids, r, mirror_reference, r6_reference, e143, e155, q)
        independent = .85 * rank(.5 * (.65 * rank(r) + .35 * rank(mirror_reference)) + .5 * rank(r6_reference)) + .15 * rank((17 * rank(e143) + 17 * rank(e155) + 12 * rank(q)) / 46)
        b.old.need(np.array_equal(actual, ref) and np.array_equal(actual, independent), 'whole composition formula')
        cases.append(dict(index=index, exact=True, synthetic_unchanged_arms=True))
    return real, dict(changed_arm_max_abs_delta=float(np.max(np.abs(real - expected))),
                     full_blend_fixture_cases=cases, full_six_arm_neural_replay=False)


def run():
    b.old.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA_VISIBLE_DEVICES must be present and empty')
    valid = b.validate()
    refs = b.snapshots()
    files = b.payload(refs)[2]
    ids, labels, r6, evidence = carry_r6(files, refs)
    candidate, reference = namespace(files[b.SOURCE]), namespace(refs['mirror']['files'][b.SOURCE])
    np, torch = candidate['np'], candidate['torch']
    b.old.need(not torch.cuda.is_initialized(), 'CPU only')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    b.old.need(candidate['LAB'] == reference['LAB'] == ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'], 'R224 target order')
    b.old.need(labels == list(rp.d.gold.LABELS), 'R6 target order')
    b.old.need(t.sha(t.CHECKPOINT) == t.CHECKPOINT_SHA, 'R224 checkpoint drift')
    evidence[str(t.CHECKPOINT.relative_to(ROOT))] = t.CHECKPOINT_SHA
    with (DATA.parent / 'test_series.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    grouped = t.group_series(rows)
    arm = candidate['ARMS'][0]
    model, res = candidate['load_model'](str(t.CHECKPOINT), arm['arch'], arm['res'], 'cpu')
    outputs, oracle_outputs, summaries, coverage = [], [], [], set()
    started = time.perf_counter()
    for index, uid in enumerate(ids):
        io.atomic_json(b.WORK / 'cpu-status.json', dict(state='RUNNING_CPU', study_index=index, total=2, phase='decode'))
        vol, mask = candidate['build_study'](uid, grouped, str(DATA), candidate['_make_reader']())
        ref_vol, ref_mask = reference['build_study'](uid, grouped, str(DATA), reference['_make_reader']())
        t.validate_volume(vol, mask)
        b.old.need(np.array_equal(vol, ref_vol) and np.array_equal(mask, ref_mask), 'reference real decode')
        cs = np.array([max(1, min(c, 94)) for c in candidate['_eval_centers'](mask, 96, 94)])
        tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs-1, cs, cs+1], 1)]))
        slots = np.searchsorted(np.cumsum([0, 26, 22, 18, 12, 18]), cs, side='right') - 1
        coverage.update(int(x) for x in slots)
        oracle = torch.stack([x.flip(0) if slot in (0, 1) else x.flip(-1) for x, slot in zip(tri, slots)])
        b.old.need(torch.equal(candidate['_mirror_tri'](tri, cs, 96), oracle), 'anatomical uint8 axes')
        plain = candidate['gpu_windows'](vol, mask, 94, res, 'cpu')
        win = candidate['gpu_windows'](vol, mask, 94, res, 'cpu', mirror=True)
        rpwin = reference['gpu_windows'](ref_vol, ref_mask, 94, res, 'cpu')
        rmwin = reference['gpu_windows'](ref_vol, ref_mask, 94, res, 'cpu', mirror=True)
        expected = torch.nn.functional.interpolate((oracle.float()/255.0)[:, :, 38:345, 38:345], size=(224, 224), mode='bilinear', align_corners=False)
        b.old.need(torch.equal(win, (expected-candidate['_MEAN'])/candidate['_STD'])
            and torch.equal(win, rmwin) and torch.equal(plain, rpwin), 'window/reference/crop order')
        candidate['_WLUT'].clear()
        b.old.need(torch.equal(candidate['gpu_windows'](vol, mask, 94, res, 'cpu', norm='none', mirror=True), expected), 'none normalization')
        candidate['_WLUT'].clear()
        io.atomic_json(b.WORK / 'cpu-status.json', dict(state='RUNNING_CPU', study_index=index, total=2, phase='candidate_and_reference_forwards'))
        pair = candidate['infer_pair_probs'](model, plain, win, 'cpu')
        p = reference['infer_probs'](model, rpwin, 'cpu')
        q = reference['infer_probs'](model, rmwin, 'cpu')
        oracle_pair = .5 * (p + q)
        b.old.need(pair.shape == (12,) and np.isfinite(pair).all() and ((pair >= 0) & (pair <= 1)).all()
            and np.array_equal(pair, oracle_pair), 'real mirror pair probability mismatch')
        outputs.append(pair); oracle_outputs.append(oracle_pair)
        summaries.append(dict(study=uid, volume_sha256=hashlib.sha256(vol.tobytes()).hexdigest(),
            mirror_windows_sha256=hashlib.sha256(win.numpy().tobytes()).hexdigest(),
            uint8_axes_exact=True, window_reference_exact=True, none_norm_exact=True,
            pair_forward_exact=True, mirror_max_abs_delta=float(np.max(np.abs(pair-oracle_pair)))))
        print(json.dumps(dict(study_index=index, status='PASS_REAL_CPU_MIRROR', seconds=time.perf_counter()-started)), flush=True)
    b.old.need(coverage == {0, 1, 2, 3, 4} and not torch.cuda.is_initialized(), 'coverage/CPU boundary')
    cmath = b.r6.parent_builder.math_functions(files)
    rmath = b.r6.parent_builder.math_functions(refs['parent']['files'])
    changed, composition = composed(cmath, rmath, *r6, np.asarray(outputs), np.asarray(oracle_outputs), ids)
    numerical = dict(studies=ids, targets=labels, r6_serving=r6[0].tolist(), r6_reference=r6[1].tolist(),
        mirror_candidate=np.asarray(outputs).tolist(), mirror_reference=np.asarray(oracle_outputs).tolist(),
        changed_weighted_contribution=changed.tolist())
    io.atomic_json(b.WORK / 'real-neural-outputs.json', numerical)
    tracked = [Path(__file__), Path(b.__file__), Path(rp.__file__), Path(t.__file__), Path(io.__file__),
        Path(b.r6.__file__), Path(b.mirror.__file__), Path(rp.d.__file__), rp.d.ANATOMY,
        b.WORK / 'real-neural-outputs.json', b.WORK / 'SPEC.md', b.WORK / 'PRE-REVIEW.md',
        b.OUT / 'kernel/release.ipynb', b.OUT / 'kernel/kernel-metadata.json', b.r6.PUBLIC_NOTEBOOK]
    for path in tracked:
        evidence[str(path.relative_to(ROOT))] = t.sha(path)
    # Recheck all evidence after forwards to reject input mutation during replay.
    for name, digest in evidence.items():
        b.old.need(t.sha(ROOT / name) == digest, 'evidence changed during CPU replay: ' + name)
    result = dict(status='PASS_REAL_TWO_STUDY_COMPOSITION', version='1.0.0', kernel_sha256=valid['kernel_sha256'],
        studies=summaries, targets=12, centre_slots=sorted(coverage), r6_all_four_folds_reused=True,
        r6_serving_public_max_abs_delta=0.0, composition=composition, evidence_sha256=evidence,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], cuda_initialized=False,
        exposure='EXPOSED', clean_oof=False, seconds=time.perf_counter()-started,
        limitation='Actual R6 and R224 mirror neural outputs on the same two complete studies. R6 is authenticated carry-forward; unchanged-arm blend fixtures are synthetic, not new R384/O inference. No full six-arm, CUDA/Linux, timing or gain qualification.')
    io.atomic_json(b.WORK / 'cpu-replay.json', result)
    io.atomic_json(b.WORK / 'cpu-status.json', dict(state='COMPLETE_CPU', status=result['status'], seconds=result['seconds']))
    return {k: result[k] for k in ('status', 'kernel_sha256', 'seconds', 'cuda_visible_devices', 'composition', 'limitation')}
