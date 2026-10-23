"""CPU boundary replay of the generated P-only parent, v1.0.0.

Executes the actual preflight, run_release, collect and R224/P math. External
model processes, asset checks and codec installation are instrumented stubs.
Accepts predictions only; never reads labels or runs a GPU/public notebook.
"""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

from tools import c85r7_rco_overlay_v1 as util
from tools import k1n_parent_v1 as edit


def bind(files, member, name, scope):
    source = edit.function(files[member].decode(), name)
    exec(compile(source, '<CPU replay '+name+'>', 'exec'), scope)
    return scope[name]


def csv_bytes(ids, values):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow([util.UID]+util.LABELS)
    writer.writerows([uid, *map(float, row)] for uid, row in zip(ids, values.astype(np.float32)))
    return stream.getvalue().encode()


def replay(files, output, ids, arrays, *, failures=None, diagnostic=False):
    """No O implementation exists in the namespace: any O access fails closed."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    r, r224_values, s = arrays
    failures = failures or {'R': {}, 'R224': {}, 'S_R7': {}}
    calls, pins = [], []
    codec_member = 'exp169_release_c85r7codec.py'
    base_member = 'exp169_release_c85r7.py'
    encoded = lambda value: json.dumps(value, sort_keys=True).encode()
    def telemetry(out, name, fn):
        out[name] = fn()
    def phase(path, name, action):
        calls.append(('phase', path.name, name))
        action()
    def process(argv, **kwargs):
        assert argv[3] == 'public-worker', 'unexpected O process'
        calls.append(('process', 'public-worker'))
    def mount(slug, name):
        assert slug in ('public-weights', 'public-code'), 'O asset lookup'
        calls.append(('mount', slug))
        return output/'assets'/slug
    def publish(staged, dest, payload):
        dest.write_bytes(payload)
    def failed(*args):
        calls.append(('failed',))
    comp = output/'competition'
    comp.mkdir()
    (comp/'sample_submission.csv').write_bytes(csv_bytes(ids, r))
    with (comp/'test.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([util.UID]); writer.writerows([[uid] for uid in ids])
    v5q = NS(_read_csv=lambda path: util.table(path.read_bytes()),
        validate_ids=lambda values, *rest: values, _mount=mount, CODE_SLUG='public-code',
        run_process_group=process, _probabilities=lambda *args: None)
    p = NS(need=util.need, UID=util.UID, DATASETS=['public-weights'], R_NAME='weights.pt',
        R_SHA='pin', check_pin=lambda path, pin: pins.append(str(path)), csv_bytes=csv_bytes)
    c85 = NS(phase=phase, publish=publish)
    ranks = dict(np=np, rank=util.ranks)
    c85.scored_rank = bind(files, 'exp169_release_c85.py', 'scored_rank', ranks)
    r224scope = dict(np=np, p=p, c85=c85)
    for name in ('failure_gate', 'r_half', 'p_branch'):
        bind(files, 'exp169_r224.py', name, r224scope)
    r224 = NS(SOURCE='raptor224_run.py', p_branch=r224scope['p_branch'],
        read=lambda path, roster: (r224_values, failures['R224'], {'status':'CPU_BOUNDARY'}))
    def auth(path):
        roster = json.loads((path/'ids.json').read_bytes())
        assert roster == ids
        return ids, r, s, dict(failures={k:dict(v) for k,v in failures.items()},
            r_prediction_fallbacks=len(failures['R']), s_r7_prediction_fallbacks=len(failures['S_R7']),
            s_r7_anatomy_anchors={}, r7_identity={})
    scope = dict(np=np, Path=Path, sys=sys, json=json, hashlib=hashlib, need=util.need,
        RUN='run', RECEIPT='parent-receipt.json', STAGE='staged', DIAGNOSTIC='diagnostic-c85r7.csv',
        P_BRANCH='p-branch.csv', VERSION='cpu-replay', RELEASE='P-only', FORMULA='P-only', MEMBERS=['R','S_R7'],
        write_once=util.write_once, json_bytes=encoded, v5q=v5q, p=p, c85=c85,
        r224=r224, authenticated_pr7=auth, failed_execution=failed, telemetry=telemetry,
        codec_summary=lambda out: {}, competition_root=lambda:comp,
        runtime_pins=lambda runtime: None, child_base_env=lambda runtime:{},
        install_codecs=lambda *args, **kwargs:{'status':'CPU_BOUNDARY'},
        codec_dirs=lambda path:(path/'codecs',path/'telemetry'),
        codecs=NS(child_env=lambda env, *args, **kwargs:env),
        r7=NS(verify_source=lambda path:None, resolve_bundle=lambda:output/'bundle',
            verify_bundle=lambda *args, **kwargs:(None,dict(manifest_sha256='cpu',features={}))),
        s6=NS(prepare_code=lambda *args, **kwargs:(output/'code',output/'anatomy.pt')))
    bind(files, base_member, 'preflight', scope)
    bind(files, codec_member, 'candidate_receipt', scope)
    collect = bind(files, codec_member, 'collect', scope)
    run = bind(files, codec_member, 'run_release', scope)
    receipt = run(output/'runtime', output)
    payload = (output/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).read_bytes()
    rf = np.asarray([uid in failures['R'] for uid in ids]) | np.all(r == .5,axis=1)
    f224 = np.asarray([uid in failures['R224'] for uid in ids])
    sf = np.asarray([uid in failures['S_R7'] for uid in ids])
    expected = r224.p_branch(r, r224_values, s, rf, f224, sf).astype(np.float32)
    np.testing.assert_array_equal(util.table(payload)[1], expected)
    assert not (output/'run/a172').exists()
    assert ('process','public-worker') in calls
    assert [c for c in calls if c[0] == 'phase'] == [('phase','pr7','child')]
    assert receipt['a172'] == dict(status='SKIPPED_REGISTERED_DROP_O', executed=False)
    assert receipt['executed_members'] == ['R','S_R7','R224']
    if diagnostic:
        assert receipt['never_submit'] and not (output/'submission.csv').exists()
    return dict(status='PASS', receipt=receipt, calls=calls, asset_checks=len(pins),
        p_branch_exact=True, o_executed=False, scope='instrumented process/asset boundaries; actual P math'), payload
