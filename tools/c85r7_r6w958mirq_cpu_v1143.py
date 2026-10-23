"""H-B-195 CPU replay 1.0.0 (claude-opus-b6): real R224 mirror neural parity + cached blend parity.

Changelog 1.0.0: (a) real R224 plain/mirror forward on the two visible studies
through the child's embedded raptor224_run.py functions vs the admitted H-B-179
embedded functions, the saved H-B-179 real-neural-outputs and an independent
oracle (public nartaa-cell functions + explicit mirror/crop/normalisation);
(b) actual extracted archives (parent H-B-189, candidate, H-B-179) imported in
isolated -I subprocesses on authenticated cached Gold58 members, R224 member =
the authenticated measured HB170_GOLD58_V107 mirror pair; (c) descriptive,
EXPOSED Gold58 delta. CPU only, no notebook execution, no GPU/Kaggle. Writes
only fixed outputs under the builder scratch directory.
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r6w958mirq_v1143 as b

old = b.old
SCRATCH = b.SCRATCH
H179_WORK = ROOT / '.local/scratch/astra-b1/hb179'
H179_REPLAY = H179_WORK / 'cpu-replay.json'
H179_REPLAY_SHA = 'f3d67b14eda026e2a68189cddc4135c06a419a7258a93678389babb9e0fc4ec5'
H179_OUTPUTS = H179_WORK / 'real-neural-outputs.json'
H179_OUTPUTS_SHA = 'cfa05f2c2f70d19ebfa99d774d0e103a1851f8c50e437b9366c57dbcfe5f66c5'
PARENT_REPLAY = ROOT / '.local/scratch/hb189-build/hb189/replay.json'
PARENT_REPLAY_SHA = '5f176aa8ff2fe2f6909eef11dbc84517f41be4fd1d7ed1bc901dc5f03a4fc7fb'
PARENT_HEALTHY58_SHA = '97b35fd7f307ae53fd99e8c71bf2cc69fb7dbb46278a54bbda339d1d5019c805'
ABORT = 'Raptor-224 failed on more than 3% of studies'


def tool_sha():
    return old.digest(Path(__file__).read_bytes())


def cpu_only():
    import torch
    old.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'present-empty CUDA_VISIBLE_DEVICES and uninitialized CUDA required')


# ------------------------------------------------------------------ (a) neural parity
def neural():
    import numpy as np
    import torch
    from tools import c85r7_r224r6mir_cpu as cpu179
    t = cpu179.t
    cpu_only()
    valid = b.validate()
    child = b.payload()[2][b.SOURCE]
    ref = b.reference('h179')[1][b.SOURCE]
    old.need(child == ref and old.digest(child) == b.MIRROR_SHA, 'child raptor224 member != admitted H179 member')
    old.pin(H179_REPLAY, H179_REPLAY_SHA); old.pin(H179_OUTPUTS, H179_OUTPUTS_SHA)
    rec179, saved = old.read(H179_REPLAY), old.read(H179_OUTPUTS)
    old.need(rec179['status'] == 'PASS_REAL_TWO_STUDY_COMPOSITION'
             and rec179['kernel_sha256'] == b.REFERENCES['h179']['kernel'], 'H179 replay identity')
    evidence = rec179['evidence_sha256']
    key = [k for k in evidence if k.replace('\\', '/').endswith('astra-b1/hb179/real-neural-outputs.json')]
    old.need(len(key) == 1 and evidence[key[0]] == H179_OUTPUTS_SHA, 'H179 saved outputs not bound by its replay')
    data = cpu179.DATA
    paths = sorted(p for p in data.iterdir() if p.is_dir())[:2]
    studies = [p.name for p in paths]
    old.need(studies == saved['studies'], 'visible study pair differs from H179 replay')
    dicoms = {str(p.relative_to(ROOT)): t.sha(p) for study in paths for p in sorted(study.rglob('*.dcm'))}
    old.need(len(dicoms) == 419 and all(evidence.get(k) == v for k, v in dicoms.items()), 'real DICOM drift vs H179 replay')
    series = data.parent / 'test_series.csv'
    old.need(evidence[str(series.relative_to(ROOT))] == t.sha(series), 'series metadata drift')
    old.need(t.sha(t.CHECKPOINT) == t.CHECKPOINT_SHA, 'R224 checkpoint drift')
    cand, refns, pub = cpu179.namespace(child), cpu179.namespace(ref), t.public_namespace()
    torch.set_num_threads(4)          # same thread settings as the saved H-B-179 replay
    torch.set_num_interop_threads(1)
    old.need(cand['LAB'] == refns['LAB'] == pub['LAB'] == saved['targets'], 'R224 target order')
    with series.open(newline='', encoding='utf-8') as stream:
        grouped = t.group_series(list(csv.DictReader(stream)))
    arm = cand['ARMS'][0]
    model, res = cand['load_model'](str(t.CHECKPOINT), arm['arch'], arm['res'], 'cpu')
    old.need(res == 224, 'R224 resolution')
    saved_mirror = np.asarray(saved['mirror_candidate'], dtype=np.float32)
    old.need(saved_mirror.shape == (2, 12) and saved_mirror.astype(np.float64).tolist() == saved['mirror_candidate'],
             'saved H179 outputs are not exact float32 values')
    rows, coverage, started = [], set(), time.perf_counter()
    for index, uid in enumerate(studies):
        vol, mask = cand['build_study'](uid, grouped, str(data), cand['_make_reader']())
        rvol, rmask = refns['build_study'](uid, grouped, str(data), refns['_make_reader']())
        pvol, pmask = pub['build_study'](uid, grouped, str(data), pub['_make_reader']())
        t.validate_volume(vol, mask)
        old.need(np.array_equal(vol, rvol) and np.array_equal(mask, rmask) and np.array_equal(vol, pvol)
                 and np.array_equal(mask, pmask), 'real decode differs between child/H179/public functions')
        cs = np.array([max(1, min(c, 94)) for c in cand['_eval_centers'](mask, 96, 94)])
        tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)]))
        slots = np.searchsorted(np.cumsum([0, 26, 22, 18, 12, 18]), cs, side='right') - 1
        coverage.update(int(x) for x in slots)
        flipped = torch.stack([x.flip(0) if slot in (0, 1) else x.flip(-1) for x, slot in zip(tri, slots)])
        old.need(torch.equal(cand['_mirror_tri'](tri, cs, 96), flipped), 'anatomical uint8 mirror axes')
        plain = cand['gpu_windows'](vol, mask, 94, res, 'cpu')
        mirrored = cand['gpu_windows'](vol, mask, 94, res, 'cpu', mirror=True)
        rplain = refns['gpu_windows'](rvol, rmask, 94, res, 'cpu')
        rmirrored = refns['gpu_windows'](rvol, rmask, 94, res, 'cpu', mirror=True)
        pplain = pub['gpu_windows'](pvol, pmask, 94, res, 'cpu')
        crop = torch.nn.functional.interpolate((flipped.float() / 255.0)[:, :, 38:345, 38:345], size=(224, 224),
                                               mode='bilinear', align_corners=False)
        omirrored = (crop - cand['_MEAN']) / cand['_STD']
        old.need(list(plain.shape) == list(mirrored.shape) == [94, 3, 224, 224], 'pair shape')
        old.need(torch.equal(mirrored, omirrored) and torch.equal(mirrored, rmirrored)
                 and torch.equal(plain, rplain) and torch.equal(plain, pplain), 'window/crop/normalisation parity')
        cand['_WLUT'].clear()
        old.need(torch.equal(cand['gpu_windows'](vol, mask, 94, res, 'cpu', norm='none', mirror=True), crop),
                 'none-normalisation mirror order')
        cand['_WLUT'].clear()
        pair = cand['infer_pair_probs'](model, plain, mirrored, 'cpu')
        rpair = refns['infer_pair_probs'](model, rplain, rmirrored, 'cpu')
        p, q = pub['infer_probs'](model, pplain, 'cpu'), pub['infer_probs'](model, omirrored, 'cpu')
        opair = .5 * (p + q)
        old.need(pair.shape == (12,) and np.isfinite(pair).all() and ((pair >= 0) & (pair <= 1)).all(), 'pair domain')
        old.need(np.array_equal(pair, rpair), 'child pair != admitted H179 pair')
        old.need(np.array_equal(pair, opair), 'child pair != independent public-function oracle')
        old.need(np.array_equal(pair, saved_mirror[index]), 'child pair != saved H179 mirror_candidate')
        rows.append(dict(study=uid, volume_sha256=old.digest(vol.tobytes()),
            mirror_windows_sha256=old.digest(mirrored.numpy().tobytes()), pair=pair.astype(float).tolist(),
            child_eq_h179_eq_oracle_eq_saved=True, max_abs_plain_minus_pair=float(np.max(np.abs(p - pair)))))
        print(json.dumps(dict(study_index=index, status='PASS', seconds=round(time.perf_counter() - started, 1))), flush=True)
    old.need(coverage == {0, 1, 2, 3, 4}, 'centre-slot coverage')
    cpu_only()
    report = dict(status='PASS_REAL_TWO_STUDY_MIRROR_NEURAL_PARITY', kernel_sha256=valid['kernel_sha256'],
        replay_tool_sha256=tool_sha(), studies=rows, centre_slots=sorted(coverage), dicoms=len(dicoms),
        dicom_inventory_sha256=old.digest(old.encoded(dicoms)), checkpoint_sha256=t.CHECKPOINT_SHA,
        h179_replay_sha256=H179_REPLAY_SHA, h179_outputs_sha256=H179_OUTPUTS_SHA, mirror_member_sha256=b.MIRROR_SHA,
        torch_threads=dict(intra=4, interop=1), cuda_initialized=False,
        limitation='Real CPU fp32 forward of the two visible studies only; CUDA fp16/T4/Linux forward and hidden '
                   'runtime unmeasured. Byte identity of raptor224_run.py carries the admitted H-B-170/H-B-179 '
                   'serving qualification.')
    old.write(SCRATCH/'neural-parity.json', old.encoded(report))
    return report


# ------------------------------------------------------------------ (b) cached blend replay
def scenarios():
    """All 64 single-row failure-dict combinations, 32 R224-sentinel (mirror FALLBACK) combinations, aborts."""
    result = []
    for n in (2, 58):
        for case, bits in enumerate(itertools.product((False, True), repeat=6)):
            result.append(dict(key=f'n{n}-dict{case}', n=n, dict_rows=[[0] if bit else [] for bit in bits], sentinel=[]))
        for case, bits in enumerate(itertools.product((False, True), repeat=5)):
            rows = [[0] if bit else [] for bit in bits]
            result.append(dict(key=f'n{n}-sentinel{case}', n=n, dict_rows=rows[:1] + [[]] + rows[1:], sentinel=[0]))
    for key, rows, sentinel in [('n58-sentinel-two-rows', [], [0, 1]), ('n58-dict-two-rows', [0, 1], []),
                                ('n58-sentinel-plus-dict', [1], [0])]:
        result.append(dict(key=key, n=58, dict_rows=[[], rows, [], [], [], []], sentinel=sentinel))
    for item in result:
        failed = set(item['dict_rows'][1]) | set(item['sentinel'])
        item['expect_abort'] = len(failed) > .03 * item['n']
    return result


REPLAY_SCRIPT = r'''
import os,sys,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
runtime,source,plan,dest=map(Path,sys.argv[1:5])
sys.path.insert(0,str(runtime))
import exp169_release_c85r7codec as c
import exp169_r224 as h
assert c.r224 is h and Path(h.__file__).resolve().parent==runtime.resolve()
assert Path(c.__file__).resolve().parent==runtime.resolve()
assert all(not name.startswith('tools.') for name in sys.modules)
data=np.load(source,allow_pickle=False)
ids=list(data['ids']); arrays=[data['a'+str(i)] for i in range(6)]
names=('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')
HALF=(0,1,2,3,4,7,10,11); W=(5,6,8,9)
def csv_hash(raw,cols):
    v=b'\n'.join(b','.join(row.split(b',')[j] for j in (0,*[i+1 for i in cols])) for row in raw.splitlines())
    return hashlib.sha256(v).hexdigest()
outputs={}; checks=[]
for item in json.loads(plan.read_text()):
    n=item['n']; aa=[a[:n].copy() for a in arrays]
    for row in item['sentinel']: aa[1][row]=.5
    failed={name:{ids[r]:'injected cached member failure' for r in rows} for name,rows in zip(names,item['dict_rows'])}
    try:
        result=h.blend(ids[:n],*aa,failed)
    except ValueError as exc:
        assert item['expect_abort'] and 'Raptor-224 failed on more than 3% of studies' in str(exc),str(exc)
        checks.append(dict(key=item['key'],status='EXPECTED_R224_ABORT')); continue
    assert not item['expect_abort'],'R224 threshold bypass: '+item['key']
    assert np.isfinite(result).all() and np.all((result>=0)&(result<=1))
    if not item['sentinel'] and not any(item['dict_rows']): assert np.array_equal(result,h.reference(*aa))
    outputs[item['key']]=result
    raw=c.p.csv_bytes(ids[:n],result)
    checks.append(dict(key=item['key'],status='PASS',all_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        csv_sha256=hashlib.sha256(raw).hexdigest(),half_csv_sha256=csv_hash(raw,HALF),w_csv_sha256=csv_hash(raw,W)))
np.savez(dest.with_suffix('.npz'),**outputs)
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
dest.with_suffix('.json').write_text(json.dumps(dict(version=c.VERSION,kernel_id=c.KERNEL_ID,checks=checks,cuda_initialized=False,embedded_imports=True),indent=2))
print(json.dumps(dict(status='PASS',version=c.VERSION,cases=len(checks),successes=len(outputs),expected_aborts=len(checks)-len(outputs))))
'''


def blend_replay():
    import numpy as np
    import pandas as pd
    import torch
    from tools import c85r7_gold58_candidates as cache
    cpu_only()
    valid = b.validate()
    reg, common, failed, arms = cache.context()
    r6_reg, compact, r6_auth = b.reuse.authenticated_r6()
    ids = reg['binding']['ids']
    old.need(r6_reg['binding']['ids'] == ids and len(ids) == 58, 'R6/H174 Gold58 ordered cohort drift')
    plain = [np.asarray(common[k], float) for k in ('R', 'R224')] + [compact] + [np.asarray(common[k], float) for k in b.MEMBERS[3:]]
    masks = [np.asarray(failed[k], bool) if k in failed else np.zeros(58, bool) for k in b.MEMBERS]
    old.need(not any(m.any() for m in masks), 'Gold58 observed failures require review')
    old.need(not any(np.all(a == .5, axis=1).any() for a in plain[:2]), 'unexpected real R sentinel')
    old.pin(PARENT_REPLAY, PARENT_REPLAY_SHA)
    old.need(old.read(PARENT_REPLAY)['inputs_sha256'] == [old.digest(a.tobytes()) for a in plain], 'H189 replay input drift')
    for name, digest in r6_auth['admitted_H174_exact_contracts'].items():
        old.need(old.digest(b.parent()[1][name]) == digest and old.digest(b.payload()[2][name]) == digest,
                 'R6 measured contract differs: ' + name)
    # Measured mirror R224 pair on the same 58 studies, produced by the admitted H-B-170 raptor functions.
    mreg = cache.current()
    old.need(mreg['binding']['ids'] == ids, 'mirror measurement cohort/order differs')
    old.need(mreg['binding']['sources']['embedded']['mirror'][b.SOURCE] == b.MIRROR_SHA,
             'measured mirror functions are not the child raptor224 member bytes')
    mirror = np.asarray(cache.measured_mirror(mreg), float)
    old.need(mirror.shape == (58, 12) and not np.all(mirror == .5, axis=1).any(), 'measured mirror sentinel/shape')
    old.need(not np.array_equal(mirror, plain[1]), 'mirror equals plain R224')
    mirrored = list(plain); mirrored[1] = mirror
    plan = scenarios()
    old.write(SCRATCH/'replay-plan.json', old.encoded(plan))
    for label, values in [('plain', plain), ('mirror', mirrored)]:
        np.savez(SCRATCH/(label + '-real-input.npz'), ids=np.asarray(ids), **{'a' + str(i): a for i, a in enumerate(values)})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH', None)
    runs_plan = {'parent_plain': ('parent', 'plain'), 'candidate_plain': ('candidate', 'plain'),
                 'parent_mirror': ('parent', 'mirror'), 'candidate_mirror': ('candidate', 'mirror'),
                 'h179_mirror': ('h179', 'mirror')}
    runs, outputs = {}, {}
    for label, (runtime, inp) in runs_plan.items():
        output = SCRATCH/('real-' + label)
        proc = subprocess.run([sys.executable, '-B', '-I', '-c', REPLAY_SCRIPT, str(SCRATCH/(runtime + '-runtime')),
                               str(SCRATCH/(inp + '-real-input.npz')), str(SCRATCH/'replay-plan.json'), str(output)],
                              cwd=SCRATCH, env=env, capture_output=True, text=True)
        old.need(proc.returncode == 0, 'isolated replay failed (' + label + '): ' + proc.stderr[-6000:])
        runs[label] = old.read(output.with_suffix('.json'))
        with np.load(output.with_suffix('.npz'), allow_pickle=False) as data: outputs[label] = {k: data[k] for k in data.files}
    versions = {k: (v['version'], v['kernel_id']) for k, v in runs.items()}
    old.need(versions == {'parent_plain': (b.PARENT_VERSION, b.PARENT_SLUG), 'candidate_plain': (b.VERSION, b.SLUG),
                          'parent_mirror': (b.PARENT_VERSION, b.PARENT_SLUG), 'candidate_mirror': (b.VERSION, b.SLUG),
                          'h179_mirror': (b.REFERENCES['h179']['version'], b.REFERENCES['h179']['slug'])}, 'runtime versions')
    checks = {label: {row['key']: row for row in run['checks']} for label, run in runs.items()}
    col = lambda x, cols: np.ascontiguousarray(x[:, list(cols)]).tobytes()
    cases = []
    for item in plan:
        key, n = item['key'], item['n']
        row = checks['candidate_mirror'][key]
        old.need(all(checks[label][key]['status'] == row['status'] for label in checks), 'failure gate differs: ' + key)
        if row['status'] == 'EXPECTED_R224_ABORT':
            old.need(item['expect_abort'], 'unexpected abort: ' + key)
            cases.append(dict(key=key, status=row['status'])); continue
        current = [np.zeros(n, bool) for _ in range(6)]
        for mask, rows in zip(current, item['dict_rows']): mask[rows] = True
        current[1][item['sentinel']] = True
        for label, values in (('mirror', mirrored), ('plain', plain)):
            aa = [a[:n].copy() for a in values]
            aa[1][item['sentinel']] = .5
            actual = outputs['candidate_' + label][key]
            old.need(actual.tobytes() == b.oracle(aa, current).tobytes(), 'independent pandas byte parity: ' + label + ' ' + key)
            old.need(actual.tobytes() == outputs['parent_' + label][key].tobytes(), 'all-12 float64 vs H189: ' + label + ' ' + key)
            old.need(checks['candidate_' + label][key]['csv_sha256'] == checks['parent_' + label][key]['csv_sha256'],
                     'all-12 CSV vs H189: ' + label + ' ' + key)
        actual = outputs['candidate_mirror'][key]
        old.need(col(actual, b.HALF_COLUMNS) == col(outputs['h179_mirror'][key], b.HALF_COLUMNS),
                 'w=.5 columns float64 vs H179: ' + key)
        old.need(row['half_csv_sha256'] == checks['h179_mirror'][key]['half_csv_sha256'], 'w=.5 columns CSV vs H179: ' + key)
        cases.append(dict(key=key, status='PASS', sentinel_rows=item['sentinel'],
            failed=[m for m, rows in zip(b.MEMBERS, item['dict_rows']) if rows],
            independent_pandas_bytes_exact=True, all12_vs_h189_float64_csv_exact=True,
            half_columns_vs_h179_float64_csv_exact=True, candidate_output_sha256=row['all_sha256']))
    healthy = 'n58-dict0'
    old.need(old.digest(outputs['parent_plain'][healthy].tobytes()) == PARENT_HEALTHY58_SHA ==
             old.digest(outputs['candidate_plain'][healthy].tobytes()), 'plain-input output != admitted H189 replay output')
    child, prior = outputs['candidate_mirror'][healthy], outputs['parent_plain'][healthy]
    old.need(col(child, b.W_COLUMNS) != col(outputs['h179_mirror'][healthy], b.W_COLUMNS), 'H179 comparison has no power')
    # Real p_branch truth table with the measured mirror member (healthy 58).
    pb = b.math_functions(b.payload()[2])['p_branch'](*mirrored[:3], *masks[:3])
    rank = lambda a: pd.DataFrame(a).rank(method='average', pct=True).to_numpy()
    rr = .65*rank(mirrored[0]) + .35*rank(mirrored[1]); w = np.asarray(b.WEIGHTS)
    old.need(pb.tobytes() == (w*rr + (1-w)*rank(mirrored[2])).tobytes(), 'real p_branch weight truth table')
    # Descriptive EXPOSED Gold58 comparison (not a gate); labels never printed.
    spec = importlib.util.spec_from_file_location('hb195_fable', b.FABLE)
    fable = importlib.util.module_from_spec(spec); spec.loader.exec_module(fable)
    label_ids, labels = fable.d.gold.gold(); labels = np.asarray(labels, float)
    old.need(list(label_ids) == ids and labels.shape == (58, 12), 'Gold58 labels/roster alignment')
    metrics = {}
    for name, value in (('parent_h189', prior), ('child_h195', child)):
        aucs = np.asarray([fable.auc(labels[:, j], value[:, j]) for j in range(12)])
        metrics[name] = dict(macro=float(aucs.mean()), weak_four=float(aucs[fable.WEAK].mean()),
                             per_target={fable.LAB[j]: float(aucs[j]) for j in range(12)})
    metrics['delta'] = dict(macro=metrics['child_h195']['macro'] - metrics['parent_h189']['macro'],
                            weak_four=metrics['child_h195']['weak_four'] - metrics['parent_h189']['weak_four'],
                            per_target={k: metrics['child_h195']['per_target'][k] - metrics['parent_h189']['per_target'][k]
                                        for k in metrics['child_h195']['per_target']})
    cpu_only()
    successes = [c for c in cases if c['status'] == 'PASS']
    report = dict(status='PASS', kernel_sha256=valid['kernel_sha256'], replay_tool_sha256=tool_sha(), studies=[2, 58],
        targets=12, scenarios=len(plan), successes=len(successes), expected_r224_aborts=len(cases) - len(successes),
        sentinel_successes=sum(bool(c['sentinel_rows']) for c in successes),
        independent_pandas_bytes_exact=True, all12_vs_h189_float64_csv_exact=True,
        half_columns=list(b.HALF_COLUMNS), half_columns_vs_h179_float64_csv_exact=True,
        plain_input_healthy58_equals_admitted_h189_replay=True, real_p_branch_weight_truth_table=True,
        mirror_vs_plain_healthy58_changed_cells=int(np.sum(child != prior)),
        cases=cases, embedded_runs=runs, cohort_sha256=old.digest(old.encoded(ids)),
        historical_binding_sha256=reg['binding_sha256'], mirror_binding_sha256=mreg['binding_sha256'],
        mirror_inputs_sha256=old.digest(mirror.tobytes()), r6_authentication=r6_auth, preserved_pins_sha256=cache.PINS_SHA,
        inputs_sha256={'plain': [old.digest(a.tobytes()) for a in plain], 'mirror': [old.digest(a.tobytes()) for a in mirrored]},
        gold58=metrics, exposure='EXPOSED', clean_oof=False, gold58_gate=False,
        limitation='Authenticated real cached-serving arithmetic and actual embedded imports; mirror R224 member = '
                   'measured 4070 fp16 pair of the admitted H-B-170 functions (= child member bytes); single/double-row '
                   'injected failures; no CUDA/T4/Linux forward or runtime estimate. Gold58 labels read internally for '
                   'descriptive AUC only, never printed.')
    old.write(SCRATCH/'replay.json', old.encoded(report))
    return report


def run():
    first = neural()
    second = blend_replay()
    return dict(status='PASS', kernel_sha256=second['kernel_sha256'], replay_tool_sha256=tool_sha(),
                neural=dict(status=first['status'], studies=len(first['studies']), dicoms=first['dicoms'],
                            max_abs_plain_minus_pair=[r['max_abs_plain_minus_pair'] for r in first['studies']]),
                blend={k: second[k] for k in ('studies', 'scenarios', 'successes', 'expected_r224_aborts', 'sentinel_successes',
                                              'all12_vs_h189_float64_csv_exact', 'half_columns_vs_h179_float64_csv_exact',
                                              'plain_input_healthy58_equals_admitted_h189_replay',
                                              'mirror_vs_plain_healthy58_changed_cells')},
                gold58_delta={k: second['gold58']['delta'][k] for k in ('macro', 'weak_four')})


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
