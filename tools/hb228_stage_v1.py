"""H-B-228 v1.0.0: immutable R6 stride2/K1 x3 stride2 and paired R6 diagnostic.

Only the embedded copies change. Frozen sources, manifests and receipts stay intact.
"""
from __future__ import annotations

import argparse
import base64
import copy
import difflib
import json
from pathlib import Path

from tools import k1n_stage_v1 as parent

VERSION = '1.0.0'
ROOT, util = parent.ROOT, parent.util
exact, digest = parent.exact, parent.digest
CONFIG = ROOT/'configs/k1n/hb228-r6s2-k1x3s2.json'
STRIDE_ANCHOR = '"--eval-stride", str(manifest.get("eval_stride", 3))'
ADAPTER_ANCHOR = "dataset.eval_stride) == (384, 'v4', 3, 3.5, 1)"
PAIR_VERSION = '1.1.68-hb228-r6-pair'
PAIR_ID = 'taeyangg4/knee-hb228-r6-stride-pair-v1'


def adapter(data):
    text = exact(data.decode(), ADAPTER_ANCHOR,
                 "dataset.eval_stride) == (384, 'v4', 3, 3.5, 2)")
    return exact(text, "VERSION = '2.0.0'", "VERSION = '2.0.1-hb228-r6s2'").encode()


def configuration():
    c = parent.schema.validate(json.loads(CONFIG.read_bytes()))
    util.need(c['k1_eval_stride'] == 2 and not c['drop_o'] and c['alpha'] == .2
              and 'alpha_vector' not in c, 'HB228 fixed scientific recipe')
    util.need(c['models'] == parent.schema.current()['models'], 'HB228 x3 checkpoint identity')
    return c


def stage(kind):
    if kind == 'pair':
        return parent.before.parent.INF3.parent/('v'+PAIR_VERSION)
    return parent.stage(configuration(), kind == 'n32')


def package(kind):
    util.need(kind in ('candidate', 'n32', 'pair'), 'unknown stage')
    c = configuration()
    nb, meta, _, old, _, _ = parent.package(c, kind != 'candidate')
    old_nb, old_meta = copy.deepcopy(nb), copy.deepcopy(meta)
    files = dict(old)
    port, release, codec = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'
    text = old[port].decode()
    if kind == 'pair':
        text = exact(text, STRIDE_ANCHOR, '"--eval-stride", str(HB228_EVAL_STRIDE)')
        text = exact(text, 'VERSION = "1.0.1-r6"',
                     'VERSION = "1.0.2-hb228-pair"\nHB228_EVAL_STRIDE = 1')
        # Diagnostic only: use the inherited R6 worker, without K1 classifiers.
        text = exact(text, '        from tools.k1sr_runtime_v7 import install\n        _k1_worker = install(runtime_source)',
                     '        _k1_worker = "compact_infer.py"')
        files['tools/hb228_pair_v1.py'] = (ROOT/'tools/hb228_pair_v1.py').read_bytes()
        files['tools/hb228_compare_v1.py'] = (ROOT/'tools/hb228_compare_v1.py').read_bytes()
    else:
        text = exact(text, STRIDE_ANCHOR, '"--eval-stride", "2"')
        text = exact(text, 'VERSION = "1.0.1-r6"', 'VERSION = "1.0.2-hb228-r6s2"')
        files['tools/k1_shared_render_v2.py'] = adapter(old['tools/k1_shared_render_v2.py'])
    files[port] = text.encode()
    files[release] = exact(old[release].decode(), digest(old[port]), digest(files[port])).encode()
    files[codec] = exact(old[codec].decode(), digest(old[release]), digest(files[release])).encode()
    code = ''.join(nb['cells'][2]['source'])
    if kind == 'pair':
        start = code.index('import exp169_release_c85r7codec as release')
        code = code[:start] + (
            'from tools.hb228_pair_v1 import run\n'
            "    run(runtime, Path('/kaggle/working'))\n")
        meta.update(id=PAIR_ID, title=PAIR_ID.split('/')[1])
        nb['cells'][1]['source'] = [
            '# H-B-228 R6 stride 1 versus 2: N32, NEVER SUBMIT\n',
            'One fixed label-free roster; sequential fresh R6 controllers; no K1/O/R/R224 execution.\n']
    else:
        old_cfg = parent.config(c, kind == 'n32')
        cfg = dict(old_cfg, r6_eval_stride=2, r6_manifest_eval_stride=1,
                   shared_adapter_version='2.0.1-hb228-r6s2')
        code = exact(code, repr(old_cfg), repr(cfg))
        nb['cells'][1]['source'] += ['H-B-228: R6 stride 2; K1 x3 stride 2; O retained; alpha 0.20.\n']
    blob, pins, _ = parent.before.parent.archive(code)
    new_blob = parent.before.packed(files)
    code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(new_blob).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in files.items()}))
    code = exact(code, repr(digest(blob)), repr(digest(new_blob)))
    nb['cells'][2]['source'] = code.splitlines(True)
    changed = {n for n in old.keys() & files.keys() if old[n] != files[n]}
    expected = {port, release, codec} | (set() if kind == 'pair' else {'tools/k1_shared_render_v2.py'})
    added = {'tools/hb228_pair_v1.py', 'tools/hb228_compare_v1.py'} if kind == 'pair' else set()
    util.need(changed == expected and files.keys()-old.keys() == added and not old.keys()-files.keys(),
              'HB228 payload scope drift')
    util.need(parent.before.parent.archive(code)[2] == files, 'archive roundtrip')
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<HB228 notebook>', 'exec')
    return nb, meta, old, files, old_nb, old_meta


def artifacts(kind):
    nb, meta, old, new, old_nb, old_meta = package(kind)
    diffs = []
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) != new.get(name):
            diffs.extend(difflib.unified_diff(old.get(name,b'').decode().splitlines(True),
                new.get(name,b'').decode().splitlines(True), fromfile='v7-stride1/'+name, tofile='hb228/'+name))
    for name, a, b in [('notebook',old_nb,nb), ('metadata',old_meta,meta)]:
        diffs.extend(difflib.unified_diff(util.encoded(a).decode().splitlines(True),
            util.encoded(b).decode().splitlines(True), fromfile='v7/'+name, tofile='hb228/'+name))
    # Also retain the complete v6 -> v7 ancestor diff; do not hide inherited changes.
    ancestor = parent.artifacts(configuration(), kind != 'candidate')[0]
    scope = dict(status='PASS', changed=sorted(n for n in old if old[n] != new[n]),
        added=sorted(new.keys()-old.keys()), removed=[],
        untouched={n:digest(v) for n,v in old.items() if new[n] == v},
        candidate_launch_import_pool_thread_timing_unchanged=kind != 'pair')
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(configuration()), 'payload-scope.json':util.encoded(scope),
        'full-payload-notebook.diff':''.join(diffs).encode(),
        'ancestor-v6-to-v7.diff':ancestor['full-payload-notebook.diff'],
        'CHANGELOG.md':b'# H-B-228 v1.0.0\n\nR6 stride2; K1 x3 stride2, v7; paired R6 N32 diagnostic.\n'
            b'No GPU qualification or independent review yet. Never submit diagnostic stages.\n'}
    reg = dict(schema='hb228-stage/v1', builder_version=VERSION, kind=kind,
        stage=str(stage(kind).relative_to(ROOT)), kernel_id=meta['id'],
        parent_stage=str(parent.before.stage('k1x3sr20', kind != 'candidate').relative_to(ROOT)),
        hidden_parent=str(parent.before.parent.INF3.relative_to(ROOT)),
        config=configuration(), r6_eval_stride=[1,2] if kind == 'pair' else 2,
        manifest_eval_stride=1, flip_tta=True, sources={n:digest(v) for n,v in new.items()},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),
        metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        never_submit=kind != 'candidate', execution_verified=False, independent_review='REQUIRED',
        qualified_for_submission=False, clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg


def run(command):
    items = [(kind, stage(kind), *artifacts(kind)) for kind in ('candidate','n32','pair')]
    if command == 'prepare':
        util.need(all(not dest.exists() for _,dest,_,_ in items), 'immutable stage exists; use validate')
        for _,dest,outputs,_ in items:
            dest.mkdir()
            for name,data in outputs.items():
                util.write_once(dest/name, data)
    else:
        util.need(command == 'validate', 'unknown command')
    for _,dest,outputs,_ in items:
        util.need({p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(outputs),
                  'artifact inventory drift')
        for name,data in outputs.items():
            util.need((dest/name).read_bytes() == data, 'immutable artifact drift: '+name)
    return dict(status='PASS_LOCAL_ONLY', qualified_for_submission=False,
                stages=[dict(kind=k, stage=r['stage'], kernel_id=r['kernel_id'],
                             never_submit=r['never_submit']) for k,_,_,r in items])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare','validate'))
    print(json.dumps(run(p.parse_args().command), indent=2))


if __name__ == '__main__':
    main()
