"""HB127 vNext-R7 1.2.0: frozen HB116 native serving, R2 + R7, shared HB126 hook.

Offline builder only. Every admission remains HOLD for B and target evidence.
"""
from __future__ import annotations
import argparse
import ast
import base64
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb116_vnext_release as h
from tools import exp169_release_c85r7 as prior
from tools import exp169_r7_port as r7
from tools import hb126_codecs as codecs
from tools import exp169_release_c85r7codec as hook_integration

VERSION = '1.2.0'
OUT = ROOT/'artifacts/experiments/HB116_VNEXT/v1.2.0/release'
SCRATCH = ROOT/'.local/scratch/sol-1/hb127'
BUNDLE = ROOT/'storage/external/heliosli-r7cnxfsss-v1'
MANIFEST_SHA = 'c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6'
PUBLIC = ROOT/'.local/external/heliosli-r7-20261013/output/submission.csv'
BASE = ROOT/'artifacts/experiments/HB116_VNEXT/v1.1.2/release'
DATASETS = [h.NEW_DATASET, 'taeyangg4/knee-exp155-own0949-oai-v1-2-assets',
    'taeyangg4/knee-exp169-c4-v4-assets', r7.DATASET_SLUG, h.c.v5q.CODE_SLUG,
    'taeyangg4/knee-exp169-s43-v4-assets', 'taeyangg4/knee-hb113-rplus-weights-v2', codecs.DATASET_SLUG]
need, read, dump, sha = h.need, h.read, h.dump, h.sha


def config(candidate='A', stress=False):
    need(candidate in h.CONSTANTS and (not stress or candidate == 'A'), 'candidate/stress')
    return dict(version=VERSION, candidate=candidate, constants=h.CONSTANTS[candidate],
                rx='R2', s43=True, s='S_R7', stress=stress, manifest_sha256=MANIFEST_SHA, codecs=True)


def roster(reg):
    return [k if k != 'S6' else 'S_R7' for k in h.roster(reg, 'R2', True)]


def blend(reg, raw, candidate='A'):
    need('S6' not in raw, 'ambiguous S6 input in R7 release')
    # Only rename the S slot: all native arithmetic and recovery stays frozen.
    mapped = {('S6' if k == 'S_R7' else k): v for k, v in raw.items()}
    return h.blend(reg, mapped, candidate, 'R2', True)


def best_effort(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return dict(status='TELEMETRY_ERROR', error=f'{type(exc).__name__}: {exc}'[:300])


@contextmanager
def child_environment(receipt, branch, work):
    """All frozen nested launches inherit the hook, including spawn loaders."""
    before = dict(os.environ)
    try:
        try:
            env = codecs.child_env(before, receipt, branch=branch,
                telemetry_dir=work/'c85r7-run/codecs/telemetry')
        except Exception:
            env = before
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def install_codecs(runtime, work, **kwargs):
    return best_effort(hook_integration.install_codecs, runtime, work, **kwargs)


def r7_worker(job_path):
    job = read(job_path)
    if job.get('device', 'cuda') != 'cpu':
        h.c.s6.install_wheels(sorted(Path(job['wheels']).glob('*.whl')), Path(job['code']))
    # Reuse the independently reviewed integration; its failure is diagnostic.
    try:
        hook_integration.gate_s_r7(Path(job['work'])/'c85r7-run', job['codec'], Path(job['code']))
    except Exception:
        # Even a diagnostic print failure cannot veto inference.
        try:
            clean = codecs.strip_env(os.environ, job['codec'])
            os.environ.clear()
            os.environ.update(clean)
        except Exception:
            pass
    return r7.infer_studies([Path(job['competition'])/'test_series'/u for u in job['ids']],
        bundle_dir=Path(job['bundle']), code_dir=Path(job['code']), anatomy_weights=Path(job['anatomy']),
        output_dir=Path(job['output']), workers=0 if job.get('device') == 'cpu' else 2,
        amp='auto', device=job.get('device', 'cuda'), max_fallback_fraction=1.,
        source_dir=Path(job['source']), expected_manifest_sha256=MANIFEST_SHA)


def run_r7(job, work, phases, *, command=None):
    """Operational failure recovers authenticated rows; identity failures remain fatal."""
    import numpy as np
    public = work/'public'
    dump(public/'ids.json', job['ids'])
    dump(work/'r7-job.json', job)
    started = time.monotonic()
    command = command or [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                          'r7-worker', '--job', str(work/'r7-job.json')]
    with child_environment(job['codec'], 'S_R7', work), (work/'r7-child.log').open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=h.CHILD_TIMEOUT_SECONDS)
            record = dict(exit_code=result.returncode)
            if result.returncode:
                record['error'] = f'ChildExit: {result.returncode}'
        except (OSError, subprocess.SubprocessError) as exc:
            record = dict(exit_code=1, error=f'{type(exc).__name__}: {exc}')
    best_effort(dump, work/'r7-exit.json', record)
    phases['S_R7'] = dict(seconds=time.monotonic()-started, exit_codes=[record['exit_code']],
                         studies=len(job['ids']), failures=[record] if record['exit_code'] else [])
    best_effort(dump, work/'phases.json', phases)
    # Bind recovery to OUR preflight, not just to a mutually consistent pair of files.
    expected = read(public/'pr7-run/r7-identity.json')
    need(expected['manifest_sha256'] == MANIFEST_SHA and expected['manifest_pinned'], 'R7 pinned preflight')
    ids, _, values, info = prior.authenticated_pr7(public)
    need(ids == job['ids'], 'R7 recovery cohort/order')
    values = np.asarray(values, np.float32)
    for i, uid in enumerate(ids):
        if uid in info['failures']['S_R7']:
            values[i] = np.nan
    return values, info


def timing(work, phases, n):
    """Reuse native timing; R7 uses the identical uncapped per-study schema."""
    import numpy as np
    from scipy.stats import t
    result = h.timings(work, {k:v for k,v in phases.items() if k != 'S_R7'}, n)
    rec = read(work/'public/pr7-run/s_r7/controller/submission_receipt.json')
    groups = [[sum(float(x) for x in row[2:6]) for row in worker['per_study']]
              for worker in h.c.s6._worker_receipts(rec)]
    samples = [x for group in groups for x in group]
    phase = phases['S_R7']
    result['families']['S6'] = dict(wall_seconds=phase['seconds'], study_seconds=samples,
        setup_upper_seconds=max(0., phase['seconds']-max(map(sum, groups), default=0.)),
        service_seconds_per_study=float(np.mean(samples)) if samples else None,
        upper95_study_seconds=float(np.mean(samples)+t.ppf(.95,len(samples)-1)*np.std(samples,ddof=1)/np.sqrt(len(samples))) if len(samples)>1 else None)
    result['members']['S_R7'] = dict(seconds=sum(samples), seconds_per_study=sum(samples)/n, count=len(samples))
    result['branches']['P'] = (phases.get('Rx', {}).get('seconds', 0.)+phase['seconds'])/n
    result['s_slot'] = 'S_R7; S6 is only the frozen projection schema key'
    result['units'] += '; P includes R7 wall time; R2 remains in fused NATIVE_O_PLUS_RX'
    return result


def finish(runtime, output, reg, ids, raw, phases, info, setup, started, *, candidate='A', stress=False, device='2xT4'):
    import numpy as np
    cfg = read(runtime/'build-config.json')
    need(cfg == config(candidate, stress), 'build config differs')
    values, scores = blend(reg, raw, candidate)
    output.mkdir(parents=True, exist_ok=True)
    h.save_raw(output/'hb127-member-raw.npz', ids, raw)
    name = 'diagnostic-hb127.csv' if stress else 'submission.csv'
    staged = output/(name+'.tmp')
    staged.write_bytes(h.c.p.csv_bytes(ids, values))
    os.replace(staged, output/name)
    missing = {k:int((~np.isfinite(a).all(axis=1)).sum()) for k,a in raw.items()}
    complete = set(raw) == set(roster(reg)) and not any(missing.values())
    healthy = complete and all(all(x == 0 for x in p['exit_codes']) for p in phases.values()) and all(x == 0 for x in info.get('worker_exit_codes', {}).values())
    rec = dict(version=VERSION, native_version=h.VERSION, status='COMPLETE_HB127', config=cfg,
        candidate=candidate, diagnostic=stress, rx='R2', s='S_R7', studies=len(ids), ids=ids, device=device,
        fixed_seconds=setup, total_seconds=time.monotonic()-started, phases=phases, missing=missing,
        normal_visible_qualified=bool(healthy and not stress), qualified_for_submission=False,
        r7_telemetry=info, manifest_sha256=MANIFEST_SHA, config_sha256=sha(runtime/'build-config.json'),
        runtime_sha256=sha(Path(__file__)), native_runtime_sha256=sha(ROOT/'tools/hb116_native_runtime.py'),
        draft_sha256=h.DRAFT_SHA, raw_sha256=sha(output/'hb127-member-raw.npz'), prediction_sha256=sha(output/name))
    # Publication precedes optional telemetry, exactly as in 1.1.2.
    pvalues = scores['P'].copy()
    pvalues[~np.isfinite(pvalues).all(axis=1)] = .5
    rec['p_branch_write'] = best_effort(lambda: (output/'p-branch-r7.csv').write_bytes(
        h.c.p.csv_bytes(ids, pvalues.astype(np.float32))))
    rec['codecs'] = best_effort(hook_integration.codec_summary, output/'hb127-run')
    try:
        rec['timing'] = timing(output/'hb127-run', phases, len(ids))
        rec['projection'] = h.projection(rec)
        rec['projection']['kaggle_limit_seconds'] = 32400
    except Exception as exc:
        rec['timing_error'] = f'{type(exc).__name__}: {exc}'
    dump(output/'hb127-receipt.json', rec)
    return rec


def serve(runtime, output, candidate='A', stress=False):
    h.configure()
    started = time.monotonic()
    need(read(runtime/'build-config.json') == config(candidate, stress), 'build config differs')
    h.pin(runtime/'sealed-registration.json', {'sha256':h.DRAFT_SHA})
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    need(len(names) == 2 and all('T4' in x for x in names), 'exact two T4 required')
    need(not (output/'hb127-run').exists() and not (output/'submission.csv').exists(), 'stale output')
    work = output/'hb127-run'
    work.mkdir(parents=True)
    reg = read(runtime/'sealed-registration.json')
    h.verify_roster(reg)
    comp = h.c.v5q._mount(h.c.p.COMPETITION, 'test.csv')
    if stress:
        from tools import exp169_c85_stress
        comp = output/'hb127-stress-data'
        exp169_c85_stress.prepare(h.c.v5q._mount(h.c.p.COMPETITION, 'train_series.csv'), comp, n=32)
    entries, paths = h.remote_assets(reg, runtime, work, 'R2', True)
    bundle = h.c.v5q._mount(r7.DATASET_SLUG, 'manifest.json')
    _, identity = r7.verify_bundle(bundle, MANIFEST_SHA, runtime/'tools/exp169_r7_src')
    code, anatomy = h.c.s6.prepare_code(h.c.v5q._mount(h.c.v5q.CODE_SLUG, 'anatomy.pt'), work/'r7-code', install_wheels=False)
    r7.code_record(code, anatomy, runtime/'tools/exp169_r7_src')
    dump(work/'public/pr7-run/r7-identity.json', identity)
    codec = install_codecs(runtime, work)
    setup = time.monotonic()-started
    with child_environment(codec, 'NATIVE_O_R2', work):
        ids, raw, phases = h.run_phases(reg, entries, paths, comp, work, device='cuda')
    job = dict(ids=ids, competition=str(comp), bundle=str(bundle), code=str(code), anatomy=str(anatomy),
        output=str(work/'public/pr7-run/s_r7'), source=str(runtime/'tools/exp169_r7_src'),
        wheels=str(work/'r7-code/wheels'), codec=codec, work=str(work))
    raw['S_R7'], info = run_r7(job, work, phases)
    return finish(runtime, output, reg, ids, raw, phases, info, setup, started, candidate=candidate, stress=stress)


def source_payloads():
    files = h.source_payloads()
    for name, data in prior.source_payloads().items():
        path = 'tools/'+name
        need(path not in files or files[path] == data, 'frozen source conflict: '+name)
        files[path] = data
    for module in (codecs, hook_integration):
        files['tools/'+Path(module.__file__).name] = Path(module.__file__).read_bytes()
    files['tools/hb127_vnextr7_release.py'] = Path(__file__).read_bytes()
    h.pin(Path(codecs.__file__), {'sha256':hook_integration.CODECS_SHA})
    # Check the native code against the already-pushed base snapshot.
    pins = read(BASE/'kernels/r2/a/source-pins.json')
    for name in ('tools/hb116_vnext_release.py', 'tools/hb116_native_runtime.py'):
        need(hashlib.sha256(files[name]).hexdigest() == pins[name], 'native 1.1.2 source drift')
    return files


def register(out=OUT):
    previous = h.SCRATCH
    try:
        h.SCRATCH = SCRATCH/'native-assets'
        old = h.register(out)
    finally:
        h.SCRATCH = previous
    _, identity = r7.verify_bundle(BUNDLE, MANIFEST_SHA)
    files = source_payloads()
    record = dict(old, version=VERSION, native_version=h.VERSION, r7=identity, rx='R2',
        codecs_sha256=sha(Path(codecs.__file__)), formula='0.70 r(P) + 0.30 r(O); P=0.5 r(R2)+0.5 r(R7)',
        source_pins={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}, qualified_for_submission=False)
    dump(out/'registration.json', record)
    return dict(status='REGISTERED_CPU_ONLY', version=VERSION, manifest_sha256=MANIFEST_SHA)


def prepare(out=OUT):
    files = source_payloads()
    kernels = []
    for candidate, stress in [('A',False), ('B',False), ('A',True)]:
        cfg = config(candidate, stress)
        key = 'stress-n32' if stress else candidate.lower()
        folder = out/'kernels'/key
        blob, pins = h.package(dict(files, **{'build-config.json':(json.dumps(cfg,sort_keys=True)+'\n').encode()}))
        nb = h.notebook(blob, pins, dict(cfg, rx='R2', s43=True))
        src = ''.join(nb['cells'][1]['source']).replace("hb116-runtime", "hb127-runtime").replace(
            'from tools import hb116_vnext_release as release', 'from tools import hb127_vnextr7_release as release')
        src = src.replace(",rx='R2',s43=True", '')
        compile(src, '<HB127>', 'exec')
        nb['cells'][1]['source'] = src.splitlines(True)
        nb['cells'][0]['source'] = [f'# HB127 vNext-R7 {VERSION}\nPrivate/offline; R2 rule b; heliosli R7; shared HB126 codecs.\n'+('DIAGNOSTIC ONLY: NEVER SUBMIT.' if stress else 'B selects after review; not qualified by this build.')]
        slug = 'taeyangg4/knee-hb127-vnextr7-'+key
        dump(folder/'release.ipynb', nb)
        dump(folder/'build-config.json', cfg)
        dump(folder/'source-pins.json', pins)
        dump(folder/'kernel-metadata.json', dict(h.c.metadata(), id=slug, title=slug.split('/')[1], dataset_sources=DATASETS))
        kernels.append('kernels/'+key)
    dump(out/'build.json', dict(version=VERSION, kernels=kernels, qualified_for_submission=False))
    return dict(status='PREPARED', kernels=3, new_weight_uploads=0)


def validate(out=OUT):
    registration = read(out/'registration.json')
    need(registration['r7']['manifest_sha256'] == MANIFEST_SHA, 'registration pin')
    need(read(out/'build.json')['kernels'] == ['kernels/a','kernels/b','kernels/stress-n32'], 'kernel roster')
    files = source_payloads()
    need(registration['source_pins'] == {k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
         'registration source drift')
    for candidate, stress in [('A',False), ('B',False), ('A',True)]:
        key = 'stress-n32' if stress else candidate.lower()
        folder = out/'kernels'/key
        cfg = config(candidate, stress)
        need(read(folder/'build-config.json') == cfg, 'kernel config')
        meta = read(folder/'kernel-metadata.json')
        need(meta == dict(h.c.metadata(), id='taeyangg4/knee-hb127-vnextr7-'+key,
                         title='knee-hb127-vnextr7-'+key, dataset_sources=DATASETS), 'metadata differs')
        src = ''.join(read(folder/'release.ipynb')['cells'][1]['source'])
        tree = ast.parse(src)
        node = next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='blob' for t in n.targets))
        blob = base64.b64decode(ast.literal_eval(node.value.args[0]))
        pins = read(folder/'source-pins.json')
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            expected = dict(files, **{'build-config.json':(json.dumps(cfg,sort_keys=True)+'\n').encode()})
            need(len(z.namelist()) == len(expected) and set(z.namelist()) == set(expected), 'package inventory')
            need(all(z.read(k) == v and hashlib.sha256(v).hexdigest() == pins[k] for k,v in expected.items()), 'package/source pin')
        need({k.arg:ast.literal_eval(k.value) for k in tree.body[-1].value.keywords} ==
             dict(candidate=candidate,stress=stress), 'serve invocation')
    return dict(status='VALIDATED', kernels=3, source_files=len(files))


def admit(out=OUT):
    validate(out)
    replay = read(out/'validation/replay.json')
    need(replay['status'] == 'PASS' and replay['standin_exact'] and replay['independent_exact'], 'CPU replay')
    return dict(status='HOLD_NOT_QUALIFIED', cpu_preparation='PASS', qualified_for_submission=False,
        remaining=['B A/B selection and gain/duplicate review', 'independent review',
            'fresh visible R2+R7 P equality to heliosli output (R differs; not assumed)',
            'matched visible/N32 2xT4 member parity, codec coverage and full runtime',
            'original 28800 s gate retained; planning estimate 29510 s exceeds it'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('register','prepare','validate','admit'):
        p = subs.add_parser(name)
        p.add_argument('--out', type=Path, default=OUT)
        if name == 'admit': p.add_argument('--dry-run', action='store_true', required=True)
    p = subs.add_parser('r7-worker')
    p.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    if args.command != 'r7-worker':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        h.configure()
    result = r7_worker(args.job) if args.command == 'r7-worker' else globals()[args.command](args.out)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
