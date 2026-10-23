"""Exact frozen H202/H204/H207 final-F readout; completed caches, CPU only.

1.0.1: compare the complete used outer arithmetic closure, allowing registered
asset-overlay/loading changes outside it. Preserve v1.0.0 premetric failure.
Reuse H200 and Gold58; never execute notebook bodies or load model tensors.
"""
from pathlib import Path
import ast
import hashlib
import io
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import hb200_k3_gold58_v1 as old
VERSION = '1.0.1'
OUT = old.HERE / 'h207-final-f'
SMALL = old.HERE / 'h202-small58/run-fix1'
C4 = old.HERE / 'h204-prep/cache-v104/cache-run'
JOINT = old.HERE / 'h207-build/JOINT-REGISTRATION.json'
STAGES = {
    'F0': ('v1.1.39-r6w958q', 'fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'),
    'F2': ('v1.1.48-r6w958inf2', '205b57215b8bc0736410a236f8b305dc2c7aeca2e94cca86dda0af3631689d2d'),
    'F4': ('v1.1.47-r6w958c4', 'd86ca0e93e11773ad6119bb9c84ea632130723d899a5f4ef5d5faa4029fc28d9'),
    'F24': ('v1.1.49-r6w958c4inf2', '4089a26aaf74a7536fa8563da970587ee2bf5fc5b87e737c7baf78b004bca61c'),
}
SMALL_SHA = '8386011c95326e1cc91afe578c77b620156372c8d3d95270ef4d45cb9f98d3eb'
C4_SHA = 'a9450d18b43893b36550680ca87874361f9effbe8c2173ed913b254f91e91016'
CONTRASTS = [('F2','F0'),('F4','F0'),('F24','F0'),('F24','F4'),('F24','F2')]
need = old.need


def functions(source, names):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == set(names), 'missing pure function')
    return ast.Module(body=nodes, type_ignores=[])


def q_operator(source):
    namespace = {'need': need}
    exec(compile(functions(source, ('rank','aggregate')), '<pinned-Q-arithmetic>', 'exec'), namespace)
    return namespace['aggregate']


def verify_outer(files):
    for name in ('exp169_r224.py','exp189_s6_port.py'):
        need(all(f[name] == files['F0'][name] for f in files.values()), 'outer source drift: '+name)
    closure = {'exp169_release_c85.py':('rank','scored_rank'),
               'exp169_release_p956.py':('csv_bytes',),
               'exp169_release_v5q.py':('require','validate_ids','_probabilities')}
    for name,names in closure.items():
        baseline = ast.dump(functions(files['F0'][name],names))
        need(all(ast.dump(functions(f[name],names))==baseline for f in files.values()), 'outer arithmetic drift: '+name)


def cache_values(z, ids, targets, family):
    import numpy as np
    small = family == 'small'
    expected = {'ids','targets','member_ids','probabilities'} if small else {'ids','targets','folds','values'}
    need(set(z) == expected and len(ids) == len(set(ids)) == 58 and
         z['ids'].tolist() == ids and z['targets'].tolist() == targets, 'cache roster/columns')
    order = [f'S-s{s}-f{f}' for s in (42,43) for f in range(5)] if small else list(range(5))
    need(z['member_ids' if small else 'folds'].tolist() == order, 'cache member order')
    values = z['probabilities' if small else 'values']
    need(values.dtype == np.float32 and values.shape == (10 if small else 5,58,12) and
         np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'cache FP32/shape/range')
    return values


def run():
    need(not (OUT/'RESULT.json').exists(), 'completed final-F comparison must not repeat')
    bridge, overlay, output_io, _ = old.dependencies()
    from tools import c85r7_r6w958c4_v1147 as builder
    import numpy as np
    pins = {}
    def bound(path, expected=None):
        digest = old.sha(path)
        need(expected is None or digest == expected, 'input pin: '+old.rel(path))
        pins[old.rel(path)] = digest
        return path
    def obj(path, expected=None): return old.read(bound(path, expected))
    def archive(path, expected=None):
        bound(path, expected)
        with np.load(path, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

    _, inventory = old.frozen_inputs()
    ids, control_csv, _ = old.parent_replay(bridge, overlay, inventory)
    control_sha = obj(old.OUT/'RESULT.json')['csv_sha256']['H189']
    need(hashlib.sha256(control_csv).hexdigest() == control_sha, 'historical H189 control')
    joint = obj(JOINT, 'f216f73cebe6a211ba24cd73071e5650c3c500b4c83c66b42647229c38704516')
    files = {}
    for arm, (version, pin) in STAGES.items():
        nb = obj(ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7'/version/'kernel/release.ipynb', pin)
        files[arm], _, _ = builder.unpack(nb)
    need(files['F0'] == bridge.unpack_parent(), 'exact original parent archive')
    # Authentication/asset overlays differ by design; all used outer arithmetic
    # and its rank/validation/serialization closure must be identical.
    verify_outer(files)
    for name, digest in joint['operators'].items():
        need(hashlib.sha256(files['F24'][name]).hexdigest() == digest, 'joint operator drift')
    qname = 'exp169_partner_bundle_release.py'
    for a,b in [('F0','F4'),('F2','F24')]:
        need(ast.dump(functions(files[a][qname],('rank','aggregate'))) ==
             ast.dump(functions(files[b][qname],('rank','aggregate'))), 'paired Q operator parity')
    manifests = {a:json.loads(f['q-a5q-manifest.json']) for a,f in files.items()}
    need(manifests['F2']['inf2_entries'] == manifests['F24']['inf2_entries'], 'Small ancestry parity')
    c4_entries = [e for e in manifests['F4']['entries'] if e['family'] == 'convnext']
    need(c4_entries == [e for e in manifests['F24']['entries'] if e['family'] == 'convnext'], 'C4 ancestry parity')
    for arm in files:
        for family in ('l2','dino'):
            need([e for e in manifests[arm]['entries'] if e['family']==family] ==
                 [e for e in manifests['F0']['entries'] if e['family']==family], 'unchanged family ancestry')
    sr = obj(SMALL/'result.json')
    sw = obj(SMALL/'worker.json', sr['hashes']['worker.json'])
    sp = obj(SMALL/'process.json', sr['hashes']['process.json'])
    need(sr['failures'] == sr['fallback_count'] == sw['failures'] == sw['fallback_count'] == 0 and
         sw['state']=='COMPLETE' and sr['predictions']==sw['predictions']==580 and
         sp['child_exit_code']==0 and sp['owned_cleanup_complete'] and not sp['survivors'] and sp['failure'] is None,
         'Small completed serving success')
    need(sw['output_sha256']==sr['hashes']['probabilities.npz']==SMALL_SHA, 'Small receipt binding')
    for row, entry in zip(sw['model_measurements'],manifests['F2']['inf2_entries'],strict=True):
        need(row['model_sha256']==entry['sha256'] and row['member']==entry['id'] and
             row['studies']==58 and row['reference_max_abs']==0, 'Small model/order/serving reference')
    small = cache_values(archive(SMALL/'probabilities.npz',SMALL_SHA), ids, overlay.LABELS, 'small')
    cc = obj(C4.parent/'B-RESULT-CHECK.json')
    cw = obj(C4/'worker.json',cc['worker_sha256'])
    cm = obj(C4/'monitor.json',cc['monitor_sha256'])
    need(cw['state']=='INFERENCE_COMPLETE' and cw['studies']==58 and cw['models']==5 and
         cw['workload']==290 and cw['failures']==0 and cm['status']=='PASS' and
         cm['child_exit_code']==0 and cm['owned_cleanup_complete'], 'C4 completed serving success')
    need(cw['npz_sha256']==C4_SHA and cw['kernel_sha256']==STAGES['F4'][1] and
         cw['model_sha256']==[e['sha256'] for e in c4_entries], 'C4 model order/source binding')
    c4 = cache_values(archive(C4/'c4-members.npz',C4_SHA), ids, overlay.LABELS, 'convnext')
    common = archive(bridge.COMMON/'predictions.npz')
    leaves = archive(bridge.COMMON/'q-leaves.npz')
    _, leaves = old.parent_arrays(common,leaves,obj(bridge.COMMON_REG),manifests['F0']['entries'],inventory,bridge,overlay)
    frame = bridge.pd.read_csv(bound(bridge.R6/'predictions.csv'),dtype={overlay.UID:str},float_precision='round_trip')
    common['S_R7'] = bridge.align(frame[overlay.UID].tolist(),frame[overlay.LABELS].to_numpy(float),ids)
    ns = bridge.parent.math_functions(files['F0'])
    arrays, payloads, qhashes = {}, {}, {}
    for arm in STAGES:
        scores = {k:v.copy() for k,v in leaves.items()}
        if arm in ('F4','F24'): scores['convnext'] = c4
        if arm in ('F2','F24'): scores['small'] = small
        q, _ = q_operator(files[arm][qname])(scores)
        if arm=='F0': need(np.array_equal(q,common['Q_A5q']), 'Q0 exact cache replay')
        args = [common[name] for name in bridge.NAMES]; args[-1] = q
        final = ns['blend'](ids,*args)
        need(np.array_equal(final,ns['reference'](*args)), 'outer reference: '+arm)
        payload = ns['p'].csv_bytes(ids,final)
        parsed = bridge.pd.read_csv(io.BytesIO(payload),dtype={overlay.UID:str},float_precision='round_trip')
        need(parsed.columns.tolist()==[overlay.UID,*overlay.LABELS] and parsed[overlay.UID].tolist()==ids, 'CSV boundary order')
        values = parsed[overlay.LABELS].to_numpy(float)
        bridge.probabilities(values,58)
        need(np.array_equal(values.astype(np.float32),final.astype(np.float32)), 'CSV FP32 boundary')
        arrays[arm], payloads[arm] = values, payload
        qhashes[arm] = hashlib.sha256(q.tobytes()).hexdigest()
    need(payloads['F0']==control_csv, 'exact H189 full output bytes')
    # Freeze verified inputs before any new label-dependent numerical comparison.
    for p in [Path(__file__), ROOT/'tools/test_hb207_final_f_v1.py', OUT/'SPEC.md', OUT/'B-PRECHECK.md', old.OUT/'PREPARATION.json']:
        bound(p)
    label = ROOT/'storage/competition/train.csv'
    bound(label, old.read(bridge.ROUTING)[old.rel(label)])
    for module in list(sys.modules.values()):
        source = getattr(module,'__file__',None)
        if source:
            path = Path(source).resolve()
            if path.is_relative_to(ROOT/'tools') and path.suffix=='.py': bound(path)
    proof = dict(version=VERSION,source_inputs=pins,control_csv_sha256=control_sha,
                 q_sha256=qhashes,rows=58,targets=overlay.LABELS,clean_oof=False,
                 exposure='EXPOSED_GOLD58',independent_review=False)
    need(not (OUT/'INPUTS.json').exists(), 'input proof already exists; inspect before continuing')
    output_io.atomic_json(OUT/'INPUTS.json',proof)
    for arm,payload in payloads.items(): (OUT/(arm+'.csv')).write_bytes(payload)
    label_ids, labels = bridge.gold.gold()
    labels = bridge.align(label_ids,labels,ids)
    weights = bridge.gold.bootstrap_weights(ids)
    need(weights.shape==(301,58), 'registered paired bootstrap')
    auc = bridge.gold.auc_table(arrays,labels,weights)
    np.savez_compressed(OUT/'auc-draws.npz',**auc)
    metrics = {a+'-'+b:bridge.gold.block(auc[a]-auc[b]) for a,b in CONTRASTS}
    result = dict(version=VERSION,status='COMPLETE_EXPOSED_B_DIRECT_CHECK_REQUIRED',
        inputs_sha256=old.sha(OUT/'INPUTS.json'),csv_sha256={a:hashlib.sha256(p).hexdigest() for a,p in payloads.items()},
        auc_sha256=old.sha(OUT/'auc-draws.npz'),metrics=metrics,rows=58,targets=overlay.LABELS,
        bootstrap=dict(draws=300,seed=20261003,unit='study singleton',conditional=True),
        clean_oof=False,exposure='EXPOSED_GOLD58',qualified_for_submission=False,independent_review=False,
        limitations=['Repeatedly exposed small panel; not clean efficacy','Intervals omit training/selection uncertainty',
                     'H207 runtime HOLD unchanged','No hidden or leaderboard prediction'])
    output_io.atomic_json(OUT/'RESULT.json',result)
    print(json.dumps({k:{'macro12':v['macro12'],'weak_four':v['weak_four']} for k,v in metrics.items()}))


if __name__ == '__main__': run()
