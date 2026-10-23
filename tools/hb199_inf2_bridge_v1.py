"""H-B-199 INF2 bounded cached-OOF bridge v1.0.0.

Changelog 1.0.0: authenticate one fixed H189/Small roster; replay original
QPOOL gate and exact parent-Q arithmetic; no release, model loads or forwards.
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
import time
import types
import uuid
import zipfile

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import numpy as np
from tools import hb105_aligned_oof_v1 as aligned  # CPU/no-checkpoint tripwires
from tools import multi_lens_readout as ml
from tools import ensemble_ablation_v1 as ea
from tools import hb105_serving_v101 as serving

VERSION = '1.0.0'
CANDIDATE = 'R6W958-INF2-V1'
OUT = ROOT / '.local/scratch/slot-b/batch5-1018/inf2-bridge'
PARENT = 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q'
PARENT_SHA = 'fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'
Q_SHA = '39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc'
BASE_IDENTITIES = 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0/input-identities.json'
QPOOL = '.local/scratch/slot-b/cpu-small/QPOOL'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
KEEP = (8, 9, 10, 11)
RAW_KEYS = {'R42', 'R43', 'DINO', 'CNX', 'S42', 'S43'}
GOLD = ['.local/scratch/hb167-common/hb167/gpu/' + n for n in
        ('worker.json', 'predictions.npz', 'q-leaves.npz')] + [
    '.local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output/predictions.csv',
    '.local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output/controller/submission_receipt.json']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    return serving.pin(path)


def rel(path):
    p = Path(path).resolve()
    need(p.is_relative_to(ROOT), 'input outside project')
    return p.relative_to(ROOT).as_posix()


def retry(call):
    for attempt in range(40):
        try:
            return call()
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32)) or attempt == 39:
                raise
            time.sleep(.05)


def atomic_new(path, payload):
    """Same-directory create-only publication, serialized per destination."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.publish-lock')
    lock.mkdir()  # An existing writer/stale lock fails closed.
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        need(not path.exists(), 'immutable output exists: ' + str(path))
        with temporary.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        retry(lambda: os.replace(temporary, path))
    finally:
        # Preserve temporary evidence after persistent publication failure.
        retry(lock.rmdir)


def write(path, value):
    atomic_new(path, (json.dumps(ml.clean(value), indent=2, sort_keys=True,
                                allow_nan=False) + '\n').encode())


def bind(bindings, path, expected=None):
    p = Path(path)
    if not p.is_absolute(): p = ROOT / p
    key = rel(p); value = pin(p)
    expected_pin = {k: expected[k] for k in ('bytes', 'sha256')} if isinstance(expected, dict) else expected
    need(expected_pin is None or (value == expected_pin if isinstance(expected_pin, dict)
         else value['sha256'] == expected_pin), 'input pin mismatch: ' + key)
    need(key not in bindings or bindings[key] == value, 'conflicting input pin: ' + key)
    bindings[key] = value
    return p


def verify_pins(bindings):
    for path, expected in bindings.items():
        bind({}, path, expected)


def parent_files():
    path = ROOT / PARENT / 'kernel/release.ipynb'
    need(pin(path)['sha256'] == PARENT_SHA, 'H189 notebook identity')
    nb = read(path)
    source = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
    assignments = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('BLOB', 'PINS'):
                need(name not in assignments, 'duplicate notebook payload assignment')
                assignments[name] = ast.literal_eval(node.value)
    blob = base64.b64decode(assignments['BLOB'], validate=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) == 31, 'embedded-file roster')
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names), 'unsafe archive path')
        files = {n: archive.read(n) for n in names}
    need({n: hashlib.sha256(v).hexdigest() for n, v in files.items()} == assignments['PINS'], 'embedded pins')
    need(hashlib.sha256(files['q-a5q-manifest.json']).hexdigest() == Q_SHA, 'parent Q manifest')
    return files


def check_roster(entries, small=False):
    keys = [(e['family'], e['seed'], e['fold']) for e in entries]
    expected = ({('S', s, f) for s in (42, 43) for f in range(5)} if small else
                {('l2', s, f) for s in (42, 43) for f in range(5)} |
                {(family, 42, f) for family in ('dino', 'convnext') for f in range(5)})
    need(len(keys) == len(set(keys)) == len(expected) and set(keys) == expected, 'source seed/fold/family identity')
    if small:
        for e in entries:
            folder = (f'.local/hb105-collected-v1/s{e["seed"]}/outputs' if e['fold'] == 0 else
                      f'.local/hb105m-collected-v1/s{e["seed"]}-f{e["fold"]}/outputs')
            need(e['path'] == folder + '/model_best.pt' and
                 e['id'] == f'S-s{e["seed"]}-f{e["fold"]}' and
                 e['arch'] == 'convnext_small.fb_in22k_ft_in1k', 'Small source/model variant')


def load_inputs(rehash_weights=False):
    raw_all, rosters, frozen, frozen_pin = aligned.load_frozen()
    need(list(ml.TARGETS) == TARGETS and tuple(ml.WEAK) == (3, 5, 6, 8), 'target/weak order changed')
    bindings = dict(frozen['bindings']); bindings.update(frozen['sources'])
    bind(bindings, aligned.EXP + '/input-source-manifest.json', frozen_pin)
    auth = read(bind(bindings, aligned.AUTH, aligned.AUTH_SHA))
    small = [e for e in auth['members'] if e['family'] == 'S']
    check_roster(small, True)
    files = parent_files(); q = json.loads(files['q-a5q-manifest.json'])
    parent = read(bind(bindings, BASE_IDENTITIES))['q_checkpoints']
    check_roster(parent)
    need(q['entries'] == parent, 'H189 source/deploy roster differs from historical parent')
    raw = {f: {k: raw_all[f][k] for k in ('R42', 'R43', 'S42', 'S43')} for f in range(5)}
    records = []
    for e in parent:
        f = e['fold']; directory = Path(e['source'].replace('\\', '/')).parent
        if e['family'] == 'dino':
            rc = read(bind(bindings, directory / 'receipt.json'))
            need(rc['status'] == 'PASS' and rc['phase'] == 'full', 'DINO receipt not full PASS')
            inventory = rc['outputs']
        else:
            inventory = read(bind(bindings, directory / 'output-inventory.json'))
        need(inventory['model_best.pt']['sha256'] == e['source_sha256'], 'parent source inventory')
        prov = read(bind(bindings, directory / 'provenance.json', inventory['provenance.json']))
        summary = read(bind(bindings, directory / 'summary.json', inventory['summary.json']))
        r = rosters[f]
        info = dict(train_sha256=serving.uid_hash(r['train']),
                    sorted_train_sha256=serving.uid_hash(sorted(r['train'])), val_sha256=serving.uid_hash(r['val']))
        ea.validate_summary({**summary, 'provenance': prov}, info, len(r['val']), f)
        if e['family'] == 'l2':
            entry = next(m for m in frozen['members'] if (m['member']['family'], m['member']['seed'],
                         m['member']['fold']) == ('R50', e['seed'], f))
            need(entry['member']['sha256'] == e['source_sha256'], 'R50 source/flip mismatch')
            prediction = entry['prediction']; key = 'R' + str(e['seed'])
        else:
            key = 'DINO' if e['family'] == 'dino' else 'CNX'
            filename = 'oof_best.csv' if key == 'DINO' else 'oof_k36_flip_tta.csv'
            prediction = (directory / filename).as_posix()
            p = bind(bindings, prediction, inventory[filename])
            raw[f][key] = aligned.load_csv(p, r['val'])
        source = (directory / 'model_best.pt').as_posix()
        source_pin = pin(ROOT / source) if rehash_weights else dict(
            bytes=(ROOT / source).stat().st_size, sha256=e['source_sha256'])
        need(source_pin['sha256'] == e['source_sha256'] and
             source_pin['bytes'] == inventory['model_best.pt']['bytes'], 'parent checkpoint source changed')
        bindings[source] = source_pin
        records.append(dict(key=key, fold=f, identity=e, prediction=prediction,
                            prediction_pin=bindings[prediction], source_pin=source_pin,
                            deployment_binding='exact embedded parent source-to-deploy tuple; no conversion'))
    for e in small:
        entry = next(m for m in frozen['members'] if m['member']['id'] == e['id'])
        need(entry['member'] == e, 'Small frozen tuple differs')
        value = pin(ROOT / e['path']) if rehash_weights else dict(bytes=(ROOT/e['path']).stat().st_size, sha256=e['sha256'])
        need(value == {k: e[k] for k in ('bytes', 'sha256')}, 'Small checkpoint changed')
        bindings[e['path']] = value
        records.append(dict(key='S' + str(e['seed']), fold=e['fold'], identity=e,
                            prediction=entry['prediction'], prediction_pin=entry['prediction_pin'],
                            source_pin=value, deployment_binding='HB105 loads this exact source checkpoint; no converted Small asset exists'))
    ids = sum((r['val'] for r in rosters), [])
    need(len(ids) == len(set(ids)) == 4343, 'heldout union')
    # No other Gold/OAI search: the named immutable HB105 metadata is the authority.
    panel_keys = [p for p in auth['bindings'] if any(k in p.lower() for k in ('gold', 'prediction', 'panel', 'oai'))]
    availability = dict(status='MISSING_AUTHENTICATED_SMALL_GOLD58', metadata=aligned.AUTH,
        metadata_sha256=aligned.AUTH_SHA, cohort_count=len(auth['cohort']), matching_panel_bindings=panel_keys,
        reason='HB105 binds ten metadata-selected timing studies; no same58 Small prediction/receipt binding.',
        global_filesystem_absence_claim=False, source_deployment='HB105 exact source weights; final packaging not built')
    need(not panel_keys and len(auth['cohort']) == 10, 'unexpected panel metadata; explicit review required')
    return raw, rosters, frozen, bindings, records, availability


def q_pair(raw, rank=ml.rank):
    need(set(raw) == set(range(5)), 'five folds required')
    families = {k: [] for k in ('R', 'D', 'C', 'S')}
    for f in range(5):
        need(set(raw[f]) == RAW_KEYS, 'missing/unexpected member')
        arrays = list(raw[f].values())
        shape = arrays[0].shape
        need(len(shape) == 2 and shape[0] > 0 and shape[1] == 12, 'member shape')
        need(all(a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all() and
                 ((a >= 0) & (a <= 1)).all() for a in arrays), 'invalid member probability')
        for name, keys in dict(R=('R42', 'R43'), D=('DINO',), C=('CNX',), S=('S42', 'S43')).items():
            families[name].append(np.stack([raw[f][key] for key in keys]).mean(0, dtype=np.float32))
    ranked = {k: rank(np.concatenate(v)) for k, v in families.items()}
    parent = rank((ranked['R'] + ranked['D'] + ranked['C']) / 3)
    hybrid = ranked['S'].copy(); hybrid[:, KEEP] = ranked['D'][:, KEEP]
    candidate = rank((ranked['R'] + hybrid + ranked['C']) / 3)
    need(np.array_equal(parent[:, KEEP], candidate[:, KEEP]), 'retained columns changed')
    return parent, candidate


def selection(delta):
    d = np.asarray(delta, np.float64)
    need(d.shape == (4, 12), 'gate shape')
    return bool(np.isfinite(d).all() and (d.mean(1) >= .0005 - 1e-12).sum() >= 3
                and np.all(d >= -.003 - 1e-12))


def describe(ctx, candidate, parent):
    delta = ctx.delta(candidate, parent)
    a, b = ctx.score(candidate), ctx.score(parent)
    need(np.isfinite(delta[0]).all(), 'undefined point AUC')
    return dict(gate_pass=selection(delta[0]), qualifying_rulers=int((delta[0].mean(1) >= .0005-1e-12).sum()),
        min_target_delta=float(delta[0].min()), rulers={r: dict(
            control_macro=float(b[0, i].mean()), arm_macro=float(a[0, i].mean()),
            macro_delta=ml.stat(delta[:, i].mean(1)), weak_four=ml.stat(delta[:, i, ml.WEAK].mean(1)),
            targets={t: dict(control_auc=float(b[0, i, j]), arm_auc=float(a[0, i, j]),
                             delta=ml.stat(delta[:, i, j])) for j, t in enumerate(TARGETS)})
            for i, r in enumerate(ml.RULERS)})


def support(ctx):
    # Only aggregate masks leave the registered context. No numeric label access.
    return {r: dict(target_valid_counts=np.asarray(ctx.refs[r][1], bool).sum(0).tolist(),
                    mask_sha256=hashlib.sha256(np.asarray(ctx.refs[r][1], bool).tobytes()).hexdigest(),
                    minimum_class=ctx.refs[r][2]) for r in ml.RULERS}


def extracted_functions(source, names, env):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == set(names) and len(nodes) == len(names), 'arithmetic extraction drift')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<pinned-parent-arithmetic-only>', 'exec'), env)
    return env


def parent_math(files=None):
    """Extract only named pure CPU functions; never execute notebook/module code."""
    files = files or parent_files()
    venv = extracted_functions(files['exp169_release_v5q.py'].decode(),
        ('validate_ids', '_probabilities'), dict(np=np, require=need))
    cenv = extracted_functions(files['exp169_release_c85.py'].decode(),
        ('scored_rank',), dict(np=np, rank=ml.rank))
    env = dict(np=np, p=types.SimpleNamespace(need=need), c85=types.SimpleNamespace(**cenv),
               v5q=types.SimpleNamespace(**venv))
    extracted_functions(files['exp169_r224.py'].decode(),
        ('failure_gate', 'r_half', 'p_branch', 'blend', 'reference'), env)
    serializer = extracted_functions(files['exp169_release_p956.py'].decode(),
        ('csv_bytes',), dict(v5q=types.SimpleNamespace(**venv), LABELS=TARGETS, UID='StudyInstanceUID'))
    env['csv_bytes'] = serializer['csv_bytes']
    return env


def registration():
    need(not (OUT/'registration.json').exists(), 'registration already exists')
    initial = read(OUT/'initial-pins.json'); verify_pins(initial['bindings'])
    raw, rosters, frozen, bindings, members, availability = load_inputs(True)
    # No calls to aggregate-label contexts or scoring occur in registration.
    q_pair(raw)
    bindings.update(initial['bindings'])
    for name in ('SPEC.md', 'SPEC-ERRATA.md', 'initial-pins.json'):
        bind(bindings, OUT/name)
    bind(bindings, __file__)
    bind(bindings, ROOT/'tools/test_hb199_inf2_bridge_v1.py')
    for path, expected in aligned.source_pins().items(): bind(bindings, path, expected)
    files = parent_files()
    embedded = {}
    for name, data in files.items():
        destination = OUT/'parent-embedded'/name
        atomic_new(destination, data)
        embedded[name] = pin(destination)
        bind(bindings, destination, embedded[name])
    document = dict(schema='hb199-inf2-bridge-registration-v1', version=VERSION, candidate=CANDIDATE,
        bindings=bindings, embedded=embedded, members=members, targets=TARGETS, keep_columns=KEEP,
        roster=dict(rows=4343, ordered_uid_sha256=serving.uid_hash(sum((r['val'] for r in rosters), [])),
                    folds=[dict(fold=f, rows=len(r['val']), ordered_uid_sha256=serving.uid_hash(r['val'])) for f,r in enumerate(rosters)]),
        ruler_bindings=frozen['ruler_bindings'], availability=availability,
        pre_execution=dict(path='.local/scratch/slot-b/batch5-1018/SLATE-REVIEW.md',
                           scope='card3 fixed cached OOF bridge only, no launch/release'),
        numeric_gate='macro >= .0005 in >=3/4 rulers AND all48 deltas >= -.003, tolerance1e-12',
        bootstrap=dict(draws=300,seed=20261003,unit='heldout connected component',conditional=True),
        rank_conventions=dict(historical_oof='average_rank/N', exact_parent_q='(average_rank-1)/max(1,N-1)',
                              outer_h189='average_rank/N; success-only under original masks'),
        exposure='EXPOSED', clean_oof=False, release='HOLD', model_forwards=0,
        checkpoint_policy='30 source checkpoints freshly hashed; parent deploy tuples authenticated, not repackaged')
    write(OUT/'registration.json', document)
    atomic_new(OUT/'registration.sha256', (pin(OUT/'registration.json')['sha256']+'\n').encode())
    return dict(status='REGISTERED_BEFORE_NEW_METRICS', members=len(members), pins=len(bindings))


def validate():
    reg = read(OUT/'registration.json')
    need(pin(OUT/'registration.json')['sha256'] == (OUT/'registration.sha256').read_text().strip(), 'registration seal')
    need(reg['version'] == VERSION and reg['candidate'] == CANDIDATE and reg['targets'] == TARGETS
         and reg['keep_columns'] == list(KEEP), 'registration identity')
    verify_pins(reg['bindings'])
    check_roster([m['identity'] for m in reg['members'] if m['key'].startswith('S')], True)
    check_roster([m['identity'] for m in reg['members'] if not m['key'].startswith('S')])
    return reg


def assert_historical(result, historical):
    maximum = 0.
    for r in ml.RULERS:
        for key in ('control_macro', 'arm_macro'):
            maximum = max(maximum, abs(result['rulers'][r][key] - historical['rulers'][r][key]))
        for t in TARGETS:
            maximum = max(maximum, abs(result['rulers'][r]['targets'][t]['delta']['value'] -
                                      historical['rulers'][r]['targets'][t]['delta']['value']))
    need(maximum < 1e-12 and result['gate_pass'] == historical['gate_pass'], 'historical INF2 replay differs')
    return maximum


def run():
    need(not (OUT/'results.json').exists(), 'scientific result already exists')
    started = time.perf_counter(); reg = validate()
    raw, rosters, frozen, bindings, members, availability = load_inputs(False)
    need(members == reg['members'] and availability == reg['availability'], 'input identities changed')
    for path, value in bindings.items(): need(reg['bindings'].get(path) == value, 'unregistered input: ' + path)
    ids = sum((r['val'] for r in rosters), [])
    parent, candidate = q_pair(raw)
    exact_parent, exact_candidate = q_pair(raw, serving.rank)
    ctx = ml.make_context(ids, ROOT)
    need(ctx.bindings == reg['ruler_bindings'], 'ruler binding changed')
    primary = describe(ctx, candidate, parent)
    exact = describe(ctx, exact_candidate, exact_parent)
    historical = read(ROOT/QPOOL/'results.json')
    historical_error = assert_historical(primary, historical['primary']['INF2'])
    folds = {}; offset = 0
    for f, roster in enumerate(rosters):
        ix = slice(offset, offset+len(roster['val'])); offset += len(roster['val'])
        fc = ml.make_context(roster['val'], ROOT)
        need(fc.bindings == reg['ruler_bindings'], 'fold ruler binding changed')
        item = describe(fc, candidate[ix], parent[ix])
        assert_historical(item, historical['fold_diagnostics'][str(f)]['INF2'])
        folds[str(f)] = dict(rows=len(roster['val']), scope='fixed pooled ranks sliced; diagnostic only',
                             readout=item, support=support(fc))
    validate()  # Rehash every registered input/source after the bounded comparison.
    import torch
    need(not torch.cuda._initialized, 'CUDA initialized')
    result = dict(status='COMPLETE_FIXED_INF2_EXPOSED_OOF', candidate=CANDIDATE, version=VERSION,
        registration_sha256=pin(OUT/'registration.json')['sha256'], rows=len(ids), targets=TARGETS,
        primary=primary, exact_parent_q_arithmetic=exact, folds=folds, support=support(ctx),
        historical_max_point_error=historical_error,
        exact_q_vs_historical=dict(parent_max_abs=float(np.abs(exact_parent-parent).max()),
            candidate_max_abs=float(np.abs(exact_candidate-candidate).max()),
            max_auc_delta_difference=max(abs(exact['rulers'][r]['targets'][t]['delta']['value']-
                                            primary['rulers'][r]['targets'][t]['delta']['value'])
                                        for r in ml.RULERS for t in TARGETS)),
        release_verdict='HOLD' if primary['gate_pass'] else 'NO_GO', clean_oof=False, exposure='EXPOSED',
        missing=['Exact H189 F(Q2) common Gold58 Small serving cache',
                 'Independent paired-seed result', 'Exact image-labelled candidate panel',
                 'Real Small-serving CPU/GPU/Linux parity, full-path visible/N32 and hidden projection'],
        availability=availability, checkpoint_loads=0, model_forwards=0, cuda_initialized=False,
        elapsed_seconds=time.perf_counter()-started)
    write(OUT/'results.json', result)
    atomic_new(OUT/'results.sha256', (pin(OUT/'results.json')['sha256']+'\n').encode())
    return dict(status=result['status'], release=result['release_verdict'], primary_gate=primary['gate_pass'],
                output=rel(OUT/'results.json'))


def check():
    reg = validate(); result = read(OUT/'results.json')
    need(pin(OUT/'results.json')['sha256'] == (OUT/'results.sha256').read_text().strip(), 'result seal')
    need(result['registration_sha256'] == pin(OUT/'registration.json')['sha256'] and
         result['candidate'] == CANDIDATE and result['rows'] == 4343 and result['targets'] == TARGETS,
         'result identity')
    expected = 'HOLD' if result['primary']['gate_pass'] else 'NO_GO'
    need(result['release_verdict'] == expected and result['clean_oof'] is False and
         result['checkpoint_loads'] == result['model_forwards'] == 0 and result['cuda_initialized'] is False,
         'result qualification/exposure')
    for item in [result['primary'],result['exact_parent_q_arithmetic'],*[v['readout'] for v in result['folds'].values()]]:
        delta = np.asarray([[item['rulers'][r]['targets'][t]['delta']['value'] for t in TARGETS] for r in ml.RULERS])
        need(selection(delta) == item['gate_pass'] and float(delta.min()) == item['min_target_delta'], 'result gate mismatch')
    need(reg['availability']['status'] == 'MISSING_AUTHENTICATED_SMALL_GOLD58', 'serving evidence drift')
    return dict(status='PASS_CPU_BRIDGE_CHECK_ONLY', result_sha256=pin(OUT/'results.json')['sha256'],
                release=expected, independent_result_review='REQUIRED')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'validate', 'run', 'check'))
    args = parser.parse_args(argv)
    result = registration() if args.command == 'register' else run() if args.command == 'run' else \
             check() if args.command == 'check' else dict(status='VALID', pins=len(validate()['bindings']))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
