"""INF3 + shared R6 K1 release builder v2.0.0 (CPU packaging only).

Changelog: two-checkpoint rank mean inside the unchanged outer 80/20 blend.
Evidence: D-B-245 exact 976 holdout, all four rulers +.0015..+.0029, P>0 >= .97.
Original v1 builder/stages untouched. Only k1x2sr20 gets an N32 stage.
The user's explicit two-variant request authorizes this bounded builder batch.
No existing stages are edited. Prepare is immutable; validate is read-only.
"""
from __future__ import annotations
import argparse
import base64
import copy
import difflib
import io
import json
from pathlib import Path
import zipfile

from tools import c85r7_r6w958inf3rco_v1150 as parent
from tools import c85r7_r6w958n224_v1154 as n224
from tools import c85r7_rco_overlay_v1 as util
from tools import k1sr_runtime_v2 as rt

VERSION = '2.0.0'
ROOT = parent.ROOT
VARIANTS = {
    'k1x2sr20': dict(version='1.1.58-r6w958k1x2sr20', slug='taeyangg4/knee-lb-r6w958k1x2sr20-v1', r224_zero=False),
    'n224k1x2sr20': dict(version='1.1.59-r6w958n224k1x2sr20', slug='taeyangg4/knee-lb-r6w958n224k1x2sr20-v1', r224_zero=True),
}
ADDED = ['tools/__init__.py', 'tools/k1sr_runtime_v2.py', 'tools/k1_shared_render_v1.py',
         'tools/l1_rco_k2_worker_v102.py', 'tools/c85r7_rco_overlay_v1.py', 'tools/k1sr_n32_v1.py']
exact, digest = parent.exact, parent.digest


def config(name, diagnostic=False):
    util.need(not diagnostic or name == 'k1x2sr20', 'N32 registered only for k1x2sr20')
    v = VARIANTS[name]
    return dict(variant=name, release_version=v['version'] + ('-n32' if diagnostic else ''),
        kernel_id=v['slug'] + ('-n32' if diagnostic else ''), diagnostic=diagnostic,
        never_submit=diagnostic, parent_weight=.80, k1_weight=.20,
        r224_weight=0.0 if v['r224_zero'] else .35,
        formula='rank(0.80*rank(INF3%s_float32_csv)+0.20*rank(mean(rank(K1_v2_FP16),rank(K1_v3G_seed1_FP16)))); average ties/full roster; final float32'
                % ('_R224_ZERO' if v['r224_zero'] else ''),
        k1_models=list(rt.MODEL_IDS),
        model_sha256={rt.MODEL_IDS[0]: util.MODEL_SHA, rt.MODEL_IDS[1]: rt.SEED1_MODEL_SHA},
        asset_manifest_sha256={rt.MODEL_IDS[0]: util.MANIFEST_SHA, rt.MODEL_IDS[1]: rt.SEED1_MANIFEST_SHA},
        evidence='D-B-245: exact976; four rulers +.0015..+.0029; P>0 >= .97',
        contract=dict(rt.w.CONTRACT, precision='float16-autocast', autocast=True,
                      workers='inherited R6 loader workers; classifier in owner',
                      formula='sigmoid(float32(FP16-autocast logits))'))


def stage(name, diagnostic=False):
    return parent.INF3.parent / ('v' + config(name, diagnostic)['release_version'])


def packed(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def package(name, diagnostic=False):
    cfg = config(name, diagnostic)
    util.need(util.sha(parent.INF3 / 'kernel/release.ipynb') == parent.INF3_SHA, 'parent notebook pin')
    nb = json.loads((parent.INF3 / 'kernel/release.ipynb').read_bytes())
    code = ''.join(nb['cells'][2]['source'])
    old_blob, old_pins, old = parent.archive(code)
    new = dict(old)
    port = old['exp169_r7_port.py'].decode()
    port = exact(port, '        runtime_source = diagnostic_source(source_dir, out)\n',
        '        runtime_source = diagnostic_source(source_dir, out)\n'
        '        from tools.k1sr_runtime_v2 import install\n        _k1_worker = install(runtime_source)\n')
    port = exact(port, '"--worker", "compact_infer.py"', '"--worker", _k1_worker')
    new['exp169_r7_port.py'] = port.encode()
    new['exp169_release_c85r7.py'] = exact(old['exp169_release_c85r7.py'].decode(),
        digest(old['exp169_r7_port.py']), digest(new['exp169_r7_port.py'])).encode()
    if VARIANTS[name]['r224_zero']:
        new['exp169_r224.py'] = exact(old['exp169_r224.py'].decode(), n224.R224_OLD, n224.R224_NEW).encode()
    if diagnostic:
        new['raptor224_run.py'] = exact(old['raptor224_run.py'].decode(), 'def find_test_root():\n',
            'def find_test_root():\n'
            '    # Diagnostic N32 only: match the other parent branches.\n'
            '    if os.environ.get("C85_COMPETITION"):\n'
            '        return os.environ["C85_COMPETITION"]\n').encode()
    codec = old['exp169_release_c85r7codec.py'].decode()
    for member in ('exp169_release_c85r7.py', 'exp169_r224.py', 'raptor224_run.py'):
        if old[member] != new[member]:
            codec = exact(codec, digest(old[member]), digest(new[member]))
    codec = exact(codec, 'VERSION = "1.1.46-r6w958inf3"', 'VERSION = "%s"' % cfg['release_version'])
    codec = exact(codec, 'v1.1.46-r6w958inf3"', 'v%s"' % cfg['release_version'])
    codec = exact(codec, 'KERNEL_ID = "taeyangg4/knee-lb-r6w958inf3-v1"', 'KERNEL_ID = "%s"' % cfg['kernel_id'])
    new['exp169_release_c85r7codec.py'] = codec.encode()
    for member in ADDED:
        new[member] = b'' if member.endswith('__init__.py') else (ROOT / member).read_bytes()
    blob = packed(new)
    code = exact(code, repr(base64.b64encode(old_blob).decode()), repr(base64.b64encode(blob).decode()))
    code = exact(code, 'PINS = ' + repr(old_pins), 'PINS = ' + repr({n: digest(v) for n, v in new.items()}))
    code = exact(code, repr(digest(old_blob)), repr(digest(blob)))
    setup = ''
    if diagnostic:
        setup = """    from tools.k1sr_n32_v1 import prepare as _prepare_n32
    _competition = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
        Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')] if (p/'train_series.csv').is_file())
    _n32 = Path('/kaggle/working/k1sr-n32-data')
    print(_prepare_n32(_competition, _n32), flush=True)
    os.environ['C85_COMPETITION'] = str(_n32)
"""
    invocation = "    release.run_release(runtime, Path('/kaggle/working'))\n"
    replacement = ("    from tools import k1sr_runtime_v2 as _k1sr\n" + setup +
        "    _parent_start = time.perf_counter()\n" + invocation +
        "    _result = _k1sr.finish(Path('/kaggle/working'), Path('/kaggle/working') / release.RUN,\n"
        "        %r, time.perf_counter() - _parent_start)\n" % cfg +
        "    print('K1SR_RESULT', {k: _result[k] for k in ('status', 'k1_executed', 'k1_studies', 'k1_seconds', 'parent_seconds')}, flush=True)\n")
    code = exact(code, invocation, replacement)
    nb['cells'][0]['source'] = ['import os, time\n', "os.environ['K1SR_STARTED'] = str(time.time())\n", 'FAST = False\n']
    nb['cells'][1]['source'] = ['# Shared R6 K1 ' + cfg['release_version'] + '\n', cfg['formula'] + '\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else 'Ordinary visible qualification required before submission.\n']
    nb['cells'][2]['source'] = code.splitlines(True)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<k1sr notebook>', 'exec')
    for member, data in new.items():
        if member.endswith('.py'):
            compile(data, member, 'exec')
    meta = json.loads((parent.INF3 / 'kernel/kernel-metadata.json').read_bytes())
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    meta['dataset_sources'].extend(['taeyangg4/knee-l1-rcof0-v1-assets', 'taeyangg4/knee-k1-v3gs1-v1-assets'])
    util.need(meta['is_private'] and not meta['enable_internet'], 'private offline stage')
    return nb, meta, old, new


def artifacts(name, diagnostic=False):
    nb, meta, old, new = package(name, diagnostic)
    allowed = {'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'}
    if VARIANTS[name]['r224_zero']: allowed.add('exp169_r224.py')
    if diagnostic: allowed.add('raptor224_run.py')
    changed = {m for m in old if old[m] != new[m]}
    util.need(changed == allowed and set(new)-set(old) == set(ADDED), 'payload scope drift')
    diff = []
    for member in sorted(new):
        if old.get(member) != new[member]:
            diff += difflib.unified_diff(old.get(member, b'').decode().splitlines(True),
                new[member].decode().splitlines(True), fromfile='INF3/'+member, tofile='K1SR/'+member)
    outputs = {'kernel/release.ipynb': util.encoded(nb), 'kernel/kernel-metadata.json': util.encoded(meta),
        'full-payload.diff': ''.join(diff).encode(),
        'payload-scope.json': util.encoded(dict(changed=sorted(changed), added=ADDED,
            untouched={m:digest(old[m]) for m in old if m not in changed})),
        'CHANGELOG.md': ('# Shared R6 K1 release v2.0.0\n\n'+config(name, diagnostic)['formula']+'\n').encode()}
    reg = dict(schema='k1x2sr-release-registration/v1', **config(name, diagnostic),
        builder_version=VERSION, builder_sha256=util.sha(__file__), parent_notebook_sha256=parent.INF3_SHA,
        kernel_sha256=digest(outputs['kernel/release.ipynb']), metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        stage=stage(name, diagnostic).relative_to(ROOT).as_posix(), guard_seconds=rt.GUARD_SECONDS,
        call_timeout_seconds=rt.CALL_TIMEOUT_SECONDS, execution_verified=False,
        independent_review='REQUIRED', clean_oof=False, exposure='EXPOSED_EXPLORATORY',
        sources={n: digest(data) for n,data in new.items()},
        authorization='Taeyang explicit INF3 in-pass K1 build; no execution/push by builder')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg


def run(command, name, diagnostic=False):
    outputs, reg = artifacts(name, diagnostic)
    dest = stage(name, diagnostic)
    if command == 'prepare':
        util.need(not dest.exists(), 'immutable stage exists; use validate')
        for member, data in outputs.items(): util.write_once(dest / member, data)
    for member, data in outputs.items():
        util.need((dest/member).read_bytes() == data, 'immutable artifact drift: '+member)
    return dict(status='PREPARED_LOCAL_ONLY', kernel_id=reg['kernel_id'], stage=reg['stage'],
                kernel_sha256=reg['kernel_sha256'], never_submit=diagnostic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','validate'])
    parser.add_argument('variant', choices=sorted(VARIANTS))
    parser.add_argument('--n32', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.command, args.variant, args.n32), indent=2))
