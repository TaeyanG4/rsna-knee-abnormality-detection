"""Focused CPU tests for H218p/H219p (tools/c85r7_r6w958n224_v1154.py). No GPU/model/network."""
from __future__ import annotations

import difflib

import numpy as np
import pytest

from tools import c85r7_r6w958inf3rco_v1150 as h208
from tools import c85r7_r6w958n224_v1154 as b
from tools import c85r7_r6w958q_v1139 as parent
from tools import c85r7_rco_overlay_v1 as ov

ORDER = ('R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q')


def changed(a, c):
    return [l for l in difflib.unified_diff(a.decode().splitlines(), c.decode().splitlines(), lineterm='', n=0)
            if l[:1] in '+-' and not l.startswith(('+++', '---'))]


@pytest.mark.parametrize('name', sorted(b.VARIANTS))
def test_scope_and_identity(name):
    nb, meta, old, new = b.package(name)
    v = b.VARIANTS[name]
    assert {m for m in old if old[m] != new[m]} == {'exp169_r224.py', 'exp169_release_c85r7codec.py'}
    c224 = changed(old['exp169_r224.py'], new['exp169_r224.py'])
    assert len(c224) == (4 if v['o_share'] else 2)
    ccodec = changed(old['exp169_release_c85r7codec.py'], new['exp169_release_c85r7codec.py'])
    assert len(ccodec) == 8  # VERSION, OUT, KERNEL_ID, r224 pin
    src = new['exp169_release_c85r7codec.py'].decode()
    assert 'VERSION = "%s"' % v['version'] in src and 'KERNEL_ID = "%s"' % v['slug'] in src
    assert meta['id'] == v['slug'] and meta['enable_internet'] is False and meta['is_private'] is True
    _, _, actual = h208.archive(''.join(nb['cells'][2]['source']))
    assert actual == new


@pytest.mark.parametrize('name', sorted(b.VARIANTS))
def test_parent_math(name):
    *_, new = b.package(name)
    ns = parent.math_functions(new)
    rng = np.random.default_rng(7)
    ids = ['1.2.%d' % i for i in range(40)]
    m = {k: rng.random((40, 12)) for k in ORDER}
    got = ov.table(ns['p'].csv_bytes(ids, ns['blend'](ids, *[m[k] for k in ORDER])), ids)[1]
    R = ov.ranks
    wt = np.asarray((.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5))
    P = wt * R(m['R']) + (1 - wt) * R(m['S_R7'])
    O = (17 * R(m['EXP143v2']) + 17 * R(m['EXP155']) + 12 * R(m['Q'])) / 46
    s = b.VARIANTS[name]['o_share'] or .15
    assert np.abs(got - ((1 - s) * R(P) + s * R(O))).max() < 1e-6
    m2 = dict(m, R224=rng.random((40, 12)))
    assert np.array_equal(got, ov.table(ns['p'].csv_bytes(ids, ns['blend'](ids, *[m2[k] for k in ORDER])), ids)[1])
