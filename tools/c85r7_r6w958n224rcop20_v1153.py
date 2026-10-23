"""H218 v1.0.0 (B on Claude Code, 2026-10-19): H217 with R224 weight 0 and RCO weight .20.

Built on the H217 package (repaired RCO worker v102 + RCO deadline). Two registered changes only:
  1. parent exp169_r224.r_half: .65*rank(R)+.35*rank(R224) -> 1.0*rank(R)+0.0*rank(R224); the
     existing fallback (R failed -> R224, R224 failed -> R) is unchanged; R224 still executes.
     The single runtime pin of exp169_r224.py in exp169_release_c85r7codec.py is updated.
  2. overlay blend .90/.10 -> .80/.20.
Gold58 (EXPOSED, D-B-241): macro +.0028 (P>0 .90), weak4 +.0100 vs INF3; selected after viewing ~10
variants, so the true effect is smaller. CPU packaging only; no model loads, GPU or submission.
Changelog: v1.0.0 first version.
"""
from __future__ import annotations

import argparse
import base64
import copy
import difflib
import json
from pathlib import Path

from tools import c85r7_r6w958inf3rco_v1150 as h208
from tools import c85r7_r6w958inf3rcop_v1152 as h217
from tools import c85r7_rco_overlay_v1 as overlay

ROOT = h208.ROOT
VERSION = '1.1.53-r6w958n224rcop20'
SLUG = 'taeyangg4/knee-lb-r6w958n224rcop20-v1'
KEY = 'R6W958-N224RCOP20-V1'
FORMULA = ('rank(0.80*rank(F_INF3noR224_float32_csv)+0.20*rank(RCO_v2_float32)); INF3 P-branch R224 weight 0; '
           'average ties/full roster; final float32')
STAGE = h217.STAGE.parent / ('v' + VERSION)
need, sha, encoded, digest, exact = overlay.need, overlay.sha, overlay.encoded, h208.digest, h208.exact
R224_OLD = 'rr = .65 * a + .35 * b'
R224_NEW = 'rr = 1.0 * a + 0.0 * b  # H218: R224 weight 0; fallbacks unchanged'


def parent_source(parent_run):
    code = parent_run.decode()
    blob, pins, files = h208.archive(code)
    old224 = files['exp169_r224.py']
    new224 = exact(old224.decode(), R224_OLD, R224_NEW).encode()
    codec = files['exp169_release_c85r7codec.py'].decode()
    codec = exact(codec, repr(digest(old224)), repr(digest(new224))).encode()
    files = dict(files, **{'exp169_r224.py': new224, 'exp169_release_c85r7codec.py': codec})
    new_blob = h208.packed(files, blob)
    new_pins = {n: digest(v) for n, v in files.items()}
    out = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(new_blob).decode()))
    out = exact(out, 'PINS = ' + repr(pins), 'PINS = ' + repr(new_pins))
    out = exact(out, repr(digest(blob)), repr(digest(new_blob)))
    compile(out, '<H218 parent>', 'exec')
    return out.encode()


def overlay_source(src):
    s = src.decode()
    for before, after in [("VERSION = " + repr(h217.VERSION), "VERSION = " + repr(VERSION)),
                          ("KERNEL = " + repr(h217.SLUG), "KERNEL = " + repr(SLUG)),
                          ("CANDIDATE = " + repr(h217.KEY), "CANDIDATE = " + repr(KEY)),
                          ("FORMULA = " + repr(h208.FORMULA), "FORMULA = " + repr(FORMULA)),
                          ("values = ranks(.90 * ranks(parent) + .10 * ranks(rco)).astype(np.float32)",
                           "values = ranks(.80 * ranks(parent) + .20 * ranks(rco)).astype(np.float32)")]:
        s = exact(s, before, after)
    compile(s, '<H218 overlay>', 'exec')
    return s.encode()


def package():
    nb17, meta17, proof17, _, files17 = h217.package()
    code = ''.join(nb17['cells'][1]['source'])
    blob17, pins17, _ = h208.archive(code)
    files = dict(files17, **{'parent-run.py': parent_source(files17['parent-run.py']),
                             'c85r7_rco_overlay_v1.py': overlay_source(files17['c85r7_rco_overlay_v1.py'])})
    blob = h208.packed(files, blob17)
    pins = {n: digest(v) for n, v in files.items()}
    new_code = exact(code, repr(base64.b64encode(blob17).decode()), repr(base64.b64encode(blob).decode()))
    new_code = exact(new_code, 'PINS = ' + repr(pins17), 'PINS = ' + repr(pins))
    new_code = exact(new_code, repr(digest(blob17)), repr(digest(blob)))
    compile(new_code, '<H218 notebook>', 'exec')
    nb = copy.deepcopy(nb17)
    nb['cells'][1]['source'] = new_code.splitlines(True)
    md = ''.join(nb17['cells'][0]['source'])
    md = exact(md, md.splitlines(True)[0], '# H218 = H217 with R224 weight 0 and RCO .20\n')
    nb['cells'][0]['source'] = md.splitlines(True)
    meta = dict(meta17, id=SLUG, title=SLUG.split('/')[1])
    proof = dict(proof17, parent='H217 ' + h217.VERSION, outer_members=pins, formula=FORMULA,
                 changes=['parent r_half R224 weight 0 (+ codec runtime pin)', 'overlay RCO weight .20'])
    return nb, meta, proof, files17, files


def materialize(stage=STAGE):
    nb, meta, proof, old, new = package()
    stage = Path(stage)
    need(not stage.exists(), 'immutable stage already exists')
    registration = dict(status='PREPARED_LOCAL_ONLY', candidate_id=KEY, kernel_id=SLUG, version=VERSION,
        formula=FORMULA, notebook_sha256=digest(encoded(nb)), metadata_sha256=digest(encoded(meta)),
        qualified_for_submission=False, source_sha256=sha(__file__),
        runtime='hidden UNMEASURED; RCO deadline guard publishes the noR224 parent if exceeded')
    diffs = list(difflib.unified_diff(old['c85r7_rco_overlay_v1.py'].decode().splitlines(True),
        new['c85r7_rco_overlay_v1.py'].decode().splitlines(True), fromfile='H217/overlay', tofile='H218/overlay'))
    _, _, p_old = h208.archive(old['parent-run.py'].decode())
    _, _, p_new = h208.archive(new['parent-run.py'].decode())
    for m in sorted(p_old):
        if p_old[m] != p_new[m]:
            diffs += difflib.unified_diff(p_old[m].decode().splitlines(True), p_new[m].decode().splitlines(True),
                                          fromfile='H217/parent/' + m, tofile='H218/parent/' + m)
    outputs = {'kernel/release.ipynb': encoded(nb), 'kernel/kernel-metadata.json': encoded(meta),
        'embedded-file-manifest.json': encoded(proof), 'registration.json': encoded(registration),
        'full-payload.diff': ''.join(diffs).encode(),
        'CHANGELOG.md': ('# H218 ' + VERSION + '\n\nH217 with parent R224 weight 0 (fallbacks unchanged) and RCO '
                         'weight .20. Everything else byte-identical to H217.\n').encode()}
    for name, data in outputs.items():
        overlay.write_once(stage / name, data)
    return registration


def validate(stage=STAGE):
    nb, meta, proof, _, files = package()
    stage = Path(stage)
    for name, value in [('kernel/release.ipynb', nb), ('kernel/kernel-metadata.json', meta),
                        ('embedded-file-manifest.json', proof)]:
        need((stage / name).read_bytes() == encoded(value), 'delivered drift: ' + name)
    _, _, actual = h208.archive(''.join(json.loads((stage / 'kernel/release.ipynb').read_bytes())['cells'][1]['source']))
    need(actual == files, 'actual outer archive differs')
    return dict(status='PASS_STATIC_B_DIRECT', members=sorted(actual), qualified_for_submission=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'validate'])
    action = p.parse_args().action
    print(json.dumps({'prepare': materialize, 'validate': validate}[action](), sort_keys=True))
