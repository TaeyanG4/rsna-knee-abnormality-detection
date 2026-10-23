"""H218p/H219p v1.0.0 (B on Claude Code, 2026-10-19): parent-only INF3 variants, no K1 branch.

D-B-242: a sequential K1 branch cannot finish on hidden within 9 h, so H217/H218 would publish
their parent anyway. These notebooks run only the INF3 parent with registered edits:
  H218p (n224)    : exp169_r224.r_half .65/.35 -> 1.0/0.0 (R224 weight 0; fallbacks unchanged)
  H219p (n224o25) : H218p + exp169_r224.blend .85 r(P) + .15 r(O) -> .75 r(P) + .25 r(O)
plus identity (VERSION/OUT/KERNEL_ID) and the single exp169_r224.py runtime pin in
exp169_release_c85r7codec.py. CPU packaging only; no model loads, GPU or submission.
Changelog: v1.0.0 first version.
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path

from tools import c85r7_r6w958inf3rco_v1150 as h208
from tools import c85r7_rco_overlay_v1 as overlay

ROOT = h208.ROOT
INF3 = h208.INF3
INF3_SHA = h208.INF3_SHA
need, sha, encoded, digest, exact = overlay.need, overlay.sha, overlay.encoded, h208.digest, h208.exact
VARIANTS = {
    'n224': dict(version='1.1.54-r6w958n224', slug='taeyangg4/knee-lb-r6w958n224-v1', key='R6W958-N224-V1', o_share=None,
                 formula='INF3 with R224 weight 0: P_t=w_t*r(R384)_t+(1-w_t)*r(R6flip)_t; 0.85*r(P)+0.15*r(O); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_INF3))/46'),
    'n224o25': dict(version='1.1.55-r6w958n224o25', slug='taeyangg4/knee-lb-r6w958n224o25-v1', key='R6W958-N224O25-V1', o_share=.25,
                    formula='INF3 with R224 weight 0 and own share .25: P_t=w_t*r(R384)_t+(1-w_t)*r(R6flip)_t; 0.75*r(P)+0.25*r(O); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_INF3))/46'),
}
R224_OLD, R224_NEW = 'rr = .65 * a + .35 * b', 'rr = 1.0 * a + 0.0 * b  # H218p: R224 weight 0; fallbacks unchanged'
O_OLD, O_NEW = 'result = .85 * pr + .15 * oo', 'result = .75 * pr + .25 * oo  # H219p: own share .25'


def stage(name):
    return INF3.parent / ('v' + VARIANTS[name]['version'])


def package(name):
    v = VARIANTS[name]
    need(sha(INF3 / 'kernel/release.ipynb') == INF3_SHA, 'INF3 notebook pin')
    base = json.loads((INF3 / 'kernel/release.ipynb').read_bytes())
    code = ''.join(base['cells'][2]['source'])
    blob, pins, files = h208.archive(code)
    old224 = files['exp169_r224.py']
    s224 = exact(old224.decode(), R224_OLD, R224_NEW)
    if v['o_share'] is not None:
        s224 = exact(s224, O_OLD, O_NEW)
    new224 = s224.encode()
    codec = files['exp169_release_c85r7codec.py'].decode()
    codec = exact(codec, repr(digest(old224)), repr(digest(new224)))
    codec = exact(codec, 'VERSION = "1.1.46-r6w958inf3"', 'VERSION = "%s"' % v['version'])
    codec = exact(codec, 'v1.1.46-r6w958inf3"', 'v%s"' % v['version'])
    codec = exact(codec, 'KERNEL_ID = "taeyangg4/knee-lb-r6w958inf3-v1"', 'KERNEL_ID = "%s"' % v['slug'])
    new_files = dict(files, **{'exp169_r224.py': new224, 'exp169_release_c85r7codec.py': codec.encode()})
    new_blob = h208.packed(new_files, blob)
    new_pins = {n: digest(x) for n, x in new_files.items()}
    new_code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(new_blob).decode()))
    new_code = exact(new_code, 'PINS = ' + repr(pins), 'PINS = ' + repr(new_pins))
    new_code = exact(new_code, repr(digest(blob)), repr(digest(new_blob)))
    compile(new_code, '<%s notebook>' % name, 'exec')
    nb = copy.deepcopy(base)
    nb['cells'][2]['source'] = new_code.splitlines(True)
    nb['cells'][1]['source'] = ['# %s %s (parent only, no K1)\n' % (v['key'], v['version']), v['formula'] + '\n',
                                'Built from exact INF3 v1.1.46 by tools/c85r7_r6w958n224_v1154.py.\n']
    meta = json.loads((INF3 / 'kernel/kernel-metadata.json').read_bytes())
    meta.update(id=v['slug'], title=v['slug'].split('/')[1])
    need(meta['is_private'] is True and meta['enable_internet'] is False, 'private offline required')
    return nb, meta, files, new_files


def materialize(name):
    v = VARIANTS[name]
    nb, meta, old, new = package(name)
    st = stage(name)
    need(not st.exists(), 'immutable stage already exists')
    outputs = {'kernel/release.ipynb': encoded(nb), 'kernel/kernel-metadata.json': encoded(meta)}
    import difflib
    diff = []
    for m in sorted(old):
        if old[m] != new[m]:
            diff += difflib.unified_diff(old[m].decode().splitlines(True), new[m].decode().splitlines(True),
                                         fromfile='INF3/' + m, tofile=name + '/' + m)
    outputs['full-payload.diff'] = ''.join(diff).encode()
    outputs['CHANGELOG.md'] = ('# %s %s\n\n%s\nParent-only (no K1 branch), built from exact INF3 v1.1.46.\n'
                               % (v['key'], v['version'], v['formula'])).encode()
    for n, data in outputs.items():
        overlay.write_once(st / n, data)
    reg = dict(schema='ordinary-visible-submission-registration/v1', kernel_id=v['slug'], version=v['version'],
               stage=str(st.relative_to(ROOT)), kernel_sha256=sha(st / 'kernel/release.ipynb'),
               metadata_sha256=sha(st / 'kernel/kernel-metadata.json'), formula=v['formula'],
               builder_sha256=sha(__file__), authorization='Taeyang 2026-10-19 22:5x: submit H218/H219 (parent-only per D-B-242)',
               execution_verified=False)
    overlay.write_once(st / 'registration.json', encoded(reg))
    return dict(status='PREPARED_LOCAL_ONLY', **{k: reg[k] for k in ('kernel_id', 'version', 'kernel_sha256')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant', choices=sorted(VARIANTS))
    print(json.dumps(materialize(p.parse_args().variant), sort_keys=True))
