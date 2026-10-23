"""CPU-only flat P evidence v1.0.0. Prediction inputs only; aggregate outputs.

No labels, inference, GPU, network, external notebook execution or admission writes.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tools import k1n_stage_v1 as b
from tools import k1n_flat_p_v1 as flat
from tools import k1n_cpu_replay_v1 as cpu

OUT = b.ROOT/'.local/scratch/astra-b5/k1n-flat-p'
CONFIG = b.ROOT/'configs/k1n/x3-r6s2-flatp-v9.json'


def rank(x):
    return rankdata(x, method='average', axis=0)/len(x)


def math_scope(files):
    def probabilities(name, values, n):
        assert values.shape == (n, 12) and np.isfinite(values).all()
        assert np.min(values) >= 0 and np.max(values) <= 1
        return values
    ns = dict(np=np, p=NS(need=b.util.need),
        v5q=NS(validate_ids=lambda ids: ids, _probabilities=probabilities))
    ns['c85'] = NS(scored_rank=cpu.bind(files, 'exp169_release_c85.py',
                                      'scored_rank', dict(np=np, rank=rank)))
    for name in ('failure_gate', 'r_half', 'p_branch', 'reference', 'blend'):
        cpu.bind(files, flat.HELPER, name, ns)
    return ns


def oracle(arrays, masks, weights):
    def scored(a, failed):
        out = np.zeros(a.shape, dtype=np.float64)
        if (~failed).any():
            out[~failed] = rank(a[~failed])
        return out
    r, r224, s = arrays
    rf, f224, sf = masks
    assert f224.sum() <= .03*len(r)
    rr, small, ss = scored(r, rf), scored(r224, f224), scored(s, sf)
    mix = .65*rr+.35*small
    mix[rf], mix[f224], mix[rf & f224] = small[rf], rr[f224], .5
    fail = rf & f224
    value = np.asarray(weights)*mix+(1-np.asarray(weights))*ss
    value[fail], value[sf], value[fail & sf] = ss[fail], mix[sf], .5
    return value


def configuration():
    return json.loads(CONFIG.read_bytes())


def fixture_c5():
    """Synthetic pins for arithmetic tests ONLY; never prepare/qualify a stage."""
    c = json.loads((b.ROOT/'configs/k1n/c5-dropo-p25-flatp.template.json').read_bytes())
    c.update(release_version='9.9.99-cpu-flatp-fixture', kernel_id='test-owner/cpu-flatp-fixture')
    for model in c['models']:
        for key in ('checkpoint_sha256', 'manifest_sha256'):
            if not b.schema.valid_hash(model[key]):
                model[key] = hashlib.sha256((model['id']+key).encode()).hexdigest()
    return c


def runtime(files):
    r = ModuleType('flat_p_cpu_runtime')
    exec(compile(files[flat.RUNTIME], '<trusted baked runtime>', 'exec'), r.__dict__)
    return r


def compare(arrays, masks, public_files, flat_files):
    old, new = math_scope(public_files), math_scope(flat_files)
    before = old['p_branch'](*arrays, *masks)
    after = new['p_branch'](*arrays, *masks)
    np.testing.assert_array_equal(before, oracle(arrays, masks, flat.PUBLIC))
    np.testing.assert_array_equal(after, oracle(arrays, masks, flat.FLAT))
    changed = np.flatnonzero(np.any(before != after, axis=0)).tolist()
    assert set(changed) <= set(flat.CHANGED)
    same = [j for j in range(12) if j not in flat.CHANGED]
    np.testing.assert_array_equal(before[:, same], after[:, same])
    np.testing.assert_array_equal(before.astype('float32')[:, same], after.astype('float32')[:, same])
    return before, after, dict(changed_columns=changed, unchanged_columns=same,
        independent_oracle_exact=True, max_abs_change_by_target=np.max(abs(after-before), axis=0).tolist())


def real_replay():
    from tools import c85r7_r6w958inf3rco_v1150 as h208
    inv = json.loads(h208.INVENTORY.read_bytes())
    assert b.util.sha(h208.INVENTORY) == h208.INVENTORY_SHA
    inputs = {}
    def pinned(rel):
        path = b.ROOT/rel
        pin = b.util.sha(path)
        assert pin == inv['files'][rel]['sha256']
        inputs[rel] = pin
        return path
    prefix = '.local/scratch/hb167-common/hb167/gpu/'
    with np.load(pinned(prefix+'predictions.npz'), allow_pickle=False) as z:
        ids = z['ids'].tolist()
        a = {k:z[k].astype(float) for k in ('R', 'R224', 'EXP143v2', 'EXP155')}
    def read(rel):
        df = pd.read_csv(pinned(rel), dtype={b.util.UID:str})
        assert not df[b.util.UID].duplicated().any() and set(df[b.util.UID]) == set(ids)
        return df.set_index(b.util.UID).loc[ids, b.util.LABELS].to_numpy(float)
    a['S_R7'] = read('.local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output/predictions.csv')
    k1 = read('.local/scratch/slot-b/research-resume-1018/run1/predictions.csv')
    masks = [np.zeros(len(ids), dtype=bool) for _ in range(3)]
    records = []
    for c in (configuration(), fixture_c5()):
        _, _, old, new, _, _ = b.package(c)
        before, after, result = compare([a[k] for k in ('R','R224','S_R7')], masks, old, new)
        if not c['drop_o']:
            with np.load(pinned(prefix+'q-leaves.npz'), allow_pickle=False) as z:
                leaves = {k:z[k].copy() for k in ('l2','dino','convnext')}
            tree = ast.parse(new['exp169_release_c85.py'])
            nodes = [n for n in tree.body if
                isinstance(n, ast.FunctionDef) and n.name == 'inf3_aggregate' or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id == 'INF3_COSTS' for t in n.targets)]
            ns = dict(need=b.util.need, rank=rank)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<INF3 pure math>', 'exec'), ns)
            q = ns['inf3_aggregate'](leaves, np)
            args = [a[k] for k in ('R','R224','S_R7','EXP143v2','EXP155')]+[q]
            before, after = [math_scope(f)['blend'](ids, *args) for f in (old,new)]
            for files, value in ((old,before),(new,after)):
                np.testing.assert_allclose(value, math_scope(files)['reference'](*args), atol=1e-15, rtol=0)
            same = result['unchanged_columns']
            np.testing.assert_array_equal(before[:,same], after[:,same])
        # Execute actual baked K1 blend. Repeat one saved member as an arithmetic
        # fixture; this is not a measured S1/C5 ensemble or model inference.
        r = runtime(new)
        rows = [dict(uid=uid, ok=True, config_sha256=r.CONFIG_SHA,
            k1_models=list(r.MODEL_IDS), finished_elapsed_seconds=10.,
            predictions={m:k1[i].tolist() for m in r.MODEL_IDS}) for i,uid in enumerate(ids)]
        outputs = []
        for parent in (before, after):
            payload = cpu.csv_bytes(ids, parent)
            actual = r.util.table(r.blend(payload, rows))[1]
            expected = rank((1-c['alpha'])*rank(parent.astype('float32'))+
                            c['alpha']*rank(rank(k1.astype('float32')))).astype('float32')
            np.testing.assert_array_equal(actual, expected)
            outputs.append(actual)
        same = result['unchanged_columns']
        np.testing.assert_array_equal(outputs[0][:,same], outputs[1][:,same])
        finishes = []
        for diagnostic in (False, True):
            # Fixed scratch only. Receipts are visibly stamped as arithmetic
            # fixtures and must never be used as visible inference evidence.
            count = 32 if diagnostic else len(ids)
            dest = OUT/'CPU_ONLY_NOT_QUALIFICATION'/('drop-o' if c['drop_o'] else 'with-o')/(
                'n32' if diagnostic else 'full')
            dest.mkdir(parents=True,exist_ok=True)
            payload = cpu.csv_bytes(ids[:count],after[:count])
            (dest/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(payload)
            worker = dest/'run/pr7/pr7-run/s_r7/controller/worker0'
            worker.mkdir(parents=True,exist_ok=True)
            (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows[:count])),encoding='utf-8')
            cfg = dict(b.config(c,diagnostic),cpu_math_replay=True,qualified_for_submission=False)
            receipt = r.finish(dest,dest/'run',cfg,10.)
            assert receipt['p_weight_mode'] == 'flat' and receipt['p_raptor_weights'] == [.5]*12
            assert receipt['config_sha256'] == cfg['config_sha256'] and receipt['formula'] == cfg['formula']
            assert receipt['status'] == 'COMPLETE_SHARED_K1N' and not receipt['qualified_for_submission']
            output = dest/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')
            assert output.read_bytes() == r.blend(payload,rows[:count])
            if diagnostic:
                assert receipt['never_submit'] and not (dest/'submission.csv').exists()
            finishes.append(dict(diagnostic=diagnostic,n=count,receipt_binding_exact=True,
                                 qualified_for_submission=False))
        result.update(drop_o=c['drop_o'], n=len(ids), actual_baked_k1_blend=True,
            fixture_members_repeated=True, inference=False, qualified_for_submission=False,
            actual_baked_finish=finishes)
        records.append(result)
    return dict(status='PASS_CPU_SAVED_PREDICTIONS', inputs=inputs, cases=records,
        exposure='EXPOSED_VISIBLE_58; cached P/R6flip; no labels or quality claim')


def compatibility():
    cases = json.loads((OUT/'PUBLIC-CASES.json').read_bytes())
    before = json.loads((OUT/'STAGES-BEFORE.json').read_bytes())
    results = []
    for case in cases:
        raw = json.loads((b.ROOT/case['config']).read_bytes())
        output, _ = b.artifacts(raw, case['diagnostic'])
        assert {n:b.digest(v) for n,v in output.items()} == case['generated']
        assert b.artifacts(dict(raw,p_weight_mode='public'),case['diagnostic'])[0] == output
        dest = OUT/'regenerated'/Path(case['stage']).name
        dest.mkdir(parents=True, exist_ok=True)
        for name, data in output.items():
            path = dest/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        current = {p.relative_to(b.ROOT/case['stage']).as_posix():b.util.sha(p)
                   for p in (b.ROOT/case['stage']).rglob('*') if p.is_file()}
        assert current == before[case['stage']], 'existing stage changed'
        results.append(dict(config=case['config'],diagnostic=case['diagnostic'],
            pre_edit_generator_identical=True, explicit_public_identical=True,
            existing_stage_untouched=True, preexisting_stage_match=case['existing_match']))
    return dict(status='PASS', cases=results)


def admission():
    from tools import k1_admit_v1 as admit
    c = configuration()
    result = []
    for diagnostic in (False, True):
        cfg = b.config(c, diagnostic)
        try:
            admit.validate(b.stage(c, diagnostic), cfg['kernel_id'], 1, OUT/'no-visible-qualification')
        except (ValueError, FileNotFoundError) as exc:
            message = str(exc)
            assert ('N32 candidate prohibited' in message if diagnostic else 'k1sr-receipt.json' in message)
            result.append(dict(diagnostic=diagnostic,status='EXPECTED_HOLD',reason=message))
        else:
            raise AssertionError('unqualified stage admitted')
    return dict(status='PASS_OFFLINE_NEGATIVE_DRY_RUN', cases=result, qualified_for_submission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('replay','compatibility','register','admit'))
    args = parser.parse_args()
    if args.command == 'register':
        result = dict(status='REGISTERED_PREVIEW_CPU', registrations=[b.artifacts(configuration(),d)[1]
                                                                    for d in (False,True)])
    else:
        result = dict(replay=real_replay, compatibility=compatibility, admit=admission)[args.command]()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/(args.command.upper()+'.json')).write_bytes(b.util.encoded(result))
    print(json.dumps(dict(command=args.command,status=result['status']),sort_keys=True))


if __name__ == '__main__':
    main()
