"""H200 v1.0.0: fixed, exposed K3/H189 Gold58 CPU readout.

Reuses H197/H198 parent replay contracts, the deployed K3 overlay and Gold58
paired statistics. prepare never reads the live cache; run never waits.
No fit, forward, checkpoint load, download, GPU or submission interface.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = ROOT / '.local/scratch/slot-b/research-resume-1018'
OUT = HERE / 'h200-analysis'
LIVE = HERE / 'run1'
TAG = 'K1V2_H189_GOLD58_1018_V1'
VERSION = '1.0.0'
INVENTORY_SHA = 'f7e5d0d456d8e573edfe2256e01e021b9cd9542446152e1c36a2f10f46e8b3b0'
K3_NOTEBOOK = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.45-r6w958rco/kernel/release.ipynb'
K3_SHA = '641d675360d2b1ee935e1d9e6b74722285d3e03ab6c8234af0a9d4310e68b5e8'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def dependencies():
    # Existing CPU tripwires forbid CUDA and checkpoint loads. No data loader runs.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from tools import hb197_inf3_bridge_v1 as bridge
    from tools import c85r7_rco_overlay_v1 as overlay
    from tools import hb145_output_io as output_io
    from tools import job_runtime as jr
    from tools import l1_rco_k2_local_v103  # freeze the reused memory helper's source closure
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return bridge, overlay, output_io, jr


def save(name, obj):
    _, _, output_io, _ = dependencies()
    path = OUT / name
    need(path.parent == OUT and not path.exists(), 'fresh aggregate output required')
    output_io.atomic_json(path, obj)


def k3_source(overlay, parent_files):
    need(sha(K3_NOTEBOOK) == K3_SHA, 'K3 notebook identity')
    nb = read(K3_NOTEBOOK)
    tree = ast.parse(''.join(nb['cells'][1]['source']))
    archives = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
            if not isinstance(value, str) or len(value) < 1000:
                continue
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(value))) as z:
                names = z.namelist()
                need(len(names) == len(set(names)), 'duplicate K3 embedded name')
                archives.append({n: z.read(n) for n in names})
        except (ValueError, SyntaxError, zipfile.BadZipFile):
            continue
    need(len(archives) == 1 and set(archives[0]) == {'c85r7_rco_overlay_v1.py', 'parent-run.py'}, 'K3 archive schema')
    need(archives[0]['c85r7_rco_overlay_v1.py'] == Path(overlay.__file__).read_bytes(), 'deployed K3 overlay differs')
    nested = ast.parse(archives[0]['parent-run.py'])
    blobs = [ast.literal_eval(n.value) for n in ast.walk(nested) if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == 'BLOB' for t in n.targets)]
    need(len(blobs) == 1, 'K3 exact parent archive')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(blobs[0]))) as z:
        names = z.namelist()
        need(len(names) == len(set(names)), 'K3 parent duplicate member')
        need({n: z.read(n) for n in names} == parent_files, 'K3 embedded parent differs from exact H189')


def verify_pins(pins):
    for name, pin in pins.items():
        path = (ROOT / name).resolve()
        need(path.is_relative_to(ROOT) and not path.is_relative_to(LIVE), 'static pin outside scope or inside live output')
        need(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'changed input/source: ' + name)


def prepare():
    """Freeze metadata/source hashes only; no live-cache reads or result metrics."""
    bridge, overlay, _, _ = dependencies()
    need(sha(HERE / 'input-inventory.json') == INVENTORY_SHA, 'fixed launch inventory')
    inventory = read(HERE / 'input-inventory.json')
    # Completed-cache identity is carried by the pinned manifest/worker receipt;
    # CPU readout never needs to re-read the two checkpoint blobs beside training.
    pins = {name: pin for name, pin in inventory['pins'].items()
            if not name.startswith('.local/scratch/slot-b/k2-release/fix1/assets-v1/weights/')}
    verify_pins(pins)
    files = bridge.unpack_parent()
    bridge.parent.math_functions(files)  # load the existing pure arithmetic namespace only
    k3_source(overlay, files)
    routing = read(bridge.ROUTING)
    label = ROOT / 'storage/competition/train.csv'
    need(sha(label) == routing[rel(label)], 'official label pin')
    extra = [HERE / 'input-inventory.json', HERE / 'GPU-NEXT.md', HERE / 'launch-k1-gold58.py',
             HERE / 'gpu-queue.DRAFT.json', OUT / 'SPEC.md', Path(__file__),
             ROOT / 'tools/test_hb200_k3_gold58_v1.py', bridge.COMMON_REG, bridge.ROUTING,
             bridge.STAGE / 'kernel/release.ipynb', bridge.STAGE / 'new-members.json', K3_NOTEBOOK, label]
    for module in list(sys.modules.values()):
        source = getattr(module, '__file__', None)
        if source:
            path = Path(source).resolve()
            if path.is_relative_to(ROOT / 'tools') and path.suffix == '.py':
                extra.append(path)
    for path in extra:
        pins[rel(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    frozen = dict(version=VERSION, status='PREPARED_NO_RESULT', inventory_sha256=INVENTORY_SHA,
                  tag=TAG, formula=overlay.FORMULA, rows=58, targets=overlay.LABELS,
                  draws=300, seed=20261003, clean_oof=False, exposure='EXPOSED_GOLD58',
                  pins=dict(sorted(pins.items())))
    save('PREPARATION.json', frozen)
    return dict(status=frozen['status'], pinned_files=len(pins), result_metrics_computed=False)


def frozen_inputs():
    frozen = read(OUT / 'PREPARATION.json')
    need(frozen['version'] == VERSION and frozen['status'] == 'PREPARED_NO_RESULT' and
         frozen['tag'] == TAG and frozen['rows'] == 58 and frozen['draws'] == 300 and
         frozen['seed'] == 20261003 and frozen['clean_oof'] is False, 'frozen specification')
    need(frozen['inventory_sha256'] == INVENTORY_SHA, 'frozen inventory identity')
    verify_pins(frozen['pins'])
    return frozen, read(HERE / 'input-inventory.json')


def completed_barrier(jr):
    # This runs before any LIVE read/existence/listing operation. No polling.
    status = jr.job_status(ROOT / 'artifacts/logs', TAG, tail=False)
    need(status.get('state') == 'SUCCESS' and status.get('receipt_kind') == 'bound',
         'cache not completed: no run-output reads permitted')
    return status


def parent_arrays(common, leaves, reg, entries, inventory, bridge, overlay):
    import numpy as np
    ids = common['ids'].tolist()
    need(len(ids) == len(set(ids)) == 58 and ids == reg['ids'] == leaves['ids'].tolist(), 'exact parent 58 roster')
    need(common['targets'].tolist() == leaves['targets'].tolist() == overlay.LABELS, 'parent target order')
    need(hashlib.sha256('\n'.join(ids).encode()).hexdigest() == inventory['cohort']['ordered_uid_sha256'], 'launch/parent roster identity')
    bridge.validate_entries(entries)
    need([bridge.member_tuple(e) for e in reg['q_entries']] == [bridge.member_tuple(e) for e in entries], 'parent Q member identity/order')
    need(set(leaves) == {'ids', 'targets', *bridge.FAMILIES}, 'parent Q family set')
    values = {k: leaves[k] for k in bridge.FAMILIES}
    for family, array in values.items():
        need(array.dtype == np.float32 and array.shape == (10 if family == 'l2' else 5, 58, 12), 'FP32 Q leaf layout')
        for value in array:
            bridge.probabilities(value, 58)
    for name in ('R', 'R224', 'EXP143v2', 'EXP155', 'Q_A5q'):
        bridge.probabilities(common[name], 58)
        mask = common['failed_' + name]
        need(mask.dtype == bool and mask.shape == (58,) and not mask.any(), 'parent failure masks')
    return ids, values


def parent_replay(bridge, overlay, inventory):
    import numpy as np
    files = bridge.unpack_parent()
    receipt = read(bridge.COMMON / 'receipt.json')
    worker = read(bridge.COMMON / 'worker.json')
    reg = read(bridge.COMMON_REG)
    need(receipt['state'] == 'COMPLETE' and worker['state'] == 'INFERENCE_COMPLETE' and
         receipt['studies'] == worker['studies'] == 58, 'complete parent cache')
    need(receipt['worker_sha256'] == sha(bridge.COMMON / 'worker.json') and
         receipt['registration_sha256'] == worker['registration_sha256'] == sha(bridge.COMMON_REG), 'parent receipt identity')
    arrays = []
    for name, key in (('predictions.npz', 'predictions_sha256'), ('q-leaves.npz', 'q_leaves_sha256')):
        path = bridge.COMMON / name
        need(sha(path) == worker[key], 'parent output receipt binding')
        with np.load(path, allow_pickle=False) as z:
            arrays.append({k: z[k].copy() for k in z.files})
    common, leaves = arrays
    entries = json.loads(files['q-a5q-manifest.json'])['entries']
    ids, leaves = parent_arrays(common, leaves, reg, entries, inventory, bridge, overlay)
    r6 = read(bridge.R6 / 'controller/submission_receipt.json')
    need(r6['status'] == 'PASS' and r6['rows'] == r6['scored'] == 58 and r6['fallback_studies'] == 0 and
         r6['fallback_reasons'] == {} and r6['worker_exit_codes'] and all(x == 0 for x in r6['worker_exit_codes'].values()), 'R6 complete success receipt')
    # Parent component arithmetic uses the original float64 CSV parser, not K3's FP32 final-input parser.
    frame = bridge.pd.read_csv(bridge.R6 / 'predictions.csv', dtype={overlay.UID: str}, float_precision='round_trip')
    need(frame.columns.tolist() == [overlay.UID, *overlay.LABELS], 'R6 target order')
    common['S_R7'] = bridge.align(frame[overlay.UID].tolist(), frame[overlay.LABELS].to_numpy(float), ids)
    q0, _ = bridge.qparent.aggregate(leaves)
    need(np.array_equal(q0, common['Q_A5q']), 'parent Q exact replay')
    ns = bridge.parent.math_functions(files)
    args = [common[name] for name in bridge.NAMES]
    f0 = ns['blend'](ids, *args)
    need(np.array_equal(f0, ns['reference'](*args)), 'parent healthy arithmetic oracle')
    replacement = args.copy(); replacement[-1] = q0
    noop = ns['blend'](ids, *replacement)
    payload = ns['p'].csv_bytes(ids, f0)
    need(f0.tobytes() == noop.tobytes() and payload == ns['p'].csv_bytes(ids, noop), 'parent no-op float64/CSV exact')
    saved_ids, values = overlay.table(payload)
    need(saved_ids == ids and np.array_equal(values.astype(np.float32), f0.astype(np.float32)), 'parent final float32 CSV roundtrip')
    return ids, payload, values


def validate_completed(records, hashes, payload, ids, inventory, manifest, overlay, jr):
    pre, worker, process, cache, job = [records[n] for n in ('prelaunch', 'worker', 'process', 'cache', 'job')]
    need(pre['tag'] == cache['tag'] == job['tag'] == TAG and pre['job_identity'] == job['child_identity'], 'new job identity')
    need(pre['inventory_sha256'] == INVENTORY_SHA and pre['launcher_sha256'] == sha(HERE / 'launch-k1-gold58.py') and
         pre['supervisor_sha256'] == inventory['pins']['tools/l1_rco_k2_local_v103.py']['sha256'] and
         pre['worker_command'] == inventory['worker_command'], 'prelaunch source/command identity')
    need(process['schema'] == 'k2-local-process/v1' and process['version'] == '1.0.3' and
         process['exit_code'] == 0 and process['reason'] == 'completed' and process['survivors'] == [], 'process not cleanly complete')
    need(jr.liveness(process['child']) == 'dead' and all(jr.liveness(x) == 'dead' for x in process['owned_descendants']), 'worker/descendant remains alive or unknown')
    need(worker['environment']['pid'] == process['child']['pid'], 'worker process identity')
    need(cache['status'] == 'COMPLETE_PENDING_INDEPENDENT_RESULT_REVIEW' and cache['studies'] == 58 and
         cache['exposure'] == 'EXPOSED_GOLD58' and cache['clean_oof'] is False and cache['qualified_for_submission'] is False,
         'cache completion/exposure flags')
    need(cache['worker_receipt_sha256'] == hashes['worker'] and cache['output_sha256'] == hashlib.sha256(payload).hexdigest(), 'cache/output receipt binding')
    overlay.check_rco(worker, payload, ids, inventory['cohort']['template_sha256'], manifest)
    need(overlay.table(payload)[0] == ids, 'worker CSV order differs')
    need(worker['roster_sha256'] == hashlib.sha256((json.dumps(ids, sort_keys=True, indent=2, ensure_ascii=False)+'\n').encode()).hexdigest(), 'worker roster digest')
    need(worker['environment']['device'] == 'cuda:0' and worker['environment']['tf32'] is False and
         worker['environment']['autocast'] is False and worker['anatomy']['dtype'] == 'torch.float32', 'FP32 serving environment')
    need(worker['codecs']['files'] > 0 and worker['codecs']['syntaxes'] and
         all(x['decoded'] is True for x in worker['codecs']['syntaxes'].values()), 'active decoded codecs')
    peak = worker['peak_reserved_bytes']; before = pre['memory']; memory = worker['memory']
    from tools.l1_rco_k2_local_v103 import memory_ok
    need(memory_ok(peak, max(before['desktop_bytes'], memory['used_before_bytes']),
                   min(before['total_bytes'], memory['total_bytes'])), 'measured memory reserve')


def completed_rco(ids, inventory, overlay, jr):
    completed_barrier(jr)
    names = dict(prelaunch='prelaunch.json', worker='worker-receipt.json', process='process-receipt.json', cache='cache-complete.json')
    records = {name: read(LIVE / filename) for name, filename in names.items()}
    hashes = {name: sha(LIVE / filename) for name, filename in names.items()}
    log = ROOT / 'artifacts/logs' / TAG
    records['job'] = read(log / 'job.json')
    hashes.update(job=sha(log / 'job.json'), job_receipt=sha(log / 'receipt.json'))
    payload = (LIVE / 'predictions.csv').read_bytes()
    manifest = read(ROOT / '.local/scratch/slot-b/k2-release/fix1/assets-v1/asset-manifest.json')
    validate_completed(records, hashes, payload, ids, inventory, manifest, overlay, jr)
    return payload, hashes


def spearman_draws(parent, rco, weights):
    """True Spearman per resampled study cohort, including duplicated draw rows."""
    import numpy as np
    from scipy.stats import spearmanr
    need(parent.shape == rco.shape and parent.ndim == 2 and parent.shape[1] == 12 and
         np.isfinite(parent).all() and np.isfinite(rco).all(), 'Spearman input shape/finite')
    need(weights.ndim == 2 and weights.shape[1] == len(parent) and np.isfinite(weights).all() and
         (weights >= 0).all() and np.equal(weights, weights.astype(int)).all(), 'paired integer draw weights')
    out = np.full((len(weights), 12), np.nan)
    for i, counts in enumerate(weights):
        index = np.repeat(np.arange(len(parent)), counts.astype(int))
        for j in range(12):
            a, b = parent[index, j], rco[index, j]
            if len(a) > 1 and np.ptp(a) > 0 and np.ptp(b) > 0:
                out[i, j] = float(spearmanr(a, b).statistic)
    return out


def statistics(ids, y, parent, candidate, rco, gold):
    import numpy as np
    need(y.shape == parent.shape == candidate.shape == rco.shape == (58, 12) and
         np.isfinite(y).all() and np.isin(y, [0, 1]).all(), 'fixed complete binary Gold58 labels')
    weights = gold.bootstrap_weights(ids)
    need(weights.shape == (301, 58), 'fixed 300 paired draws')
    scores = gold.auc_table(dict(H189=parent, K3=candidate), y, weights)
    corr = spearman_draws(parent, rco, weights)
    return dict(parent=gold.block(scores['H189']), candidate=gold.block(scores['K3']),
                delta_k3_minus_h189=gold.block(scores['K3']-scores['H189']),
                rco_vs_h189_spearman=gold.block(corr))


def run():
    bridge, overlay, _, jr = dependencies()
    completed_barrier(jr)  # fail before any cache output reads or numerical replay
    frozen, inventory = frozen_inputs()
    need(frozen['formula'] == overlay.FORMULA and frozen['targets'] == overlay.LABELS, 'fixed formula/target identity')
    k3_source(overlay, bridge.unpack_parent())
    template = inventory['worker_command'][inventory['worker_command'].index('--template')+1]
    ids, _ = overlay.table(Path(template).read_bytes())
    need(len(ids) == 58, 'fixed template 58')
    rco_payload, receipts = completed_rco(ids, inventory, overlay, jr)
    parent_ids, parent_payload, parent = parent_replay(bridge, overlay, inventory)
    need(parent_ids == ids, 'parent/worker exact ordered roster')
    candidate_payload = overlay.blend(parent_payload, rco_payload, ids)
    _, candidate = overlay.table(candidate_payload, ids)
    _, rco = overlay.table(rco_payload, ids)
    import numpy as np
    expected = overlay.ranks(.90*overlay.ranks(parent)+.10*overlay.ranks(rco)).astype(np.float32)
    need(np.array_equal(candidate.astype(np.float32), expected), 'fixed K3 final FP32 roundtrip')
    label_ids, labels = bridge.gold.gold()  # only after all completion/source checks
    labels = bridge.align(label_ids, labels, ids)
    metrics = statistics(ids, labels, parent, candidate, rco, bridge.gold)
    result = dict(version=VERSION, status='COMPLETE_EXPOSED_DESCRIPTIVE_PENDING_RESULT_REVIEW',
        candidate='H-B-182-K3-RCO10-V1', formula=overlay.FORMULA, studies=58, targets=overlay.LABELS,
        exposure='EXPOSED_GOLD58', clean_oof=False, qualified_for_submission=False,
        preparation_sha256=sha(OUT/'PREPARATION.json'), source_and_input_pins=frozen['pins'],
        receipt_hashes=receipts, ordered_uid_sha256=inventory['cohort']['ordered_uid_sha256'],
        csv_sha256={n:hashlib.sha256(p).hexdigest() for n,p in (('H189',parent_payload),('K3',candidate_payload),('RCO',rco_payload))},
        parent_noop_exact=True, final_float32_roundtrip=True, metrics=metrics,
        bootstrap=dict(draws=300, seed=20261003, unit='study singleton', paired=True,
                       auc_ranks='fixed full58 before resampling', spearman='rerank each duplicated study sample', conditional=True),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=bridge.pd.__version__),
        limitations=['Repeatedly exposed tiny panel; not clean976 or an independent image lens.',
                     'Conditional paired intervals omit training and prior selection uncertainty.',
                     'No efficacy threshold, calibrated leaderboard forecast or release decision.'])
    save('RESULT.json', result)
    report = '# H200 exposed fixed K3 versus H189\n\n' + json.dumps(metrics, indent=2, allow_nan=False) + '\n\nEXPOSED Gold58; descriptive only; independent result review pending.\n'
    path = OUT/'REPORT.md'; need(not path.exists(), 'fresh report required')
    _, _, output_io, _ = dependencies()
    tmp = path.with_suffix('.md.tmp')
    output_io.retry(lambda: tmp.write_text(report, encoding='utf-8'))
    output_io.replace(tmp, path)
    return dict(status=result['status'], studies=58, no_release_decision=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == 'prepare' else run()))


if __name__ == '__main__':
    main()
