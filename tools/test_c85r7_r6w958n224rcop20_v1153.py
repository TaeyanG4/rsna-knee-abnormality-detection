"""Focused CPU tests for H218 (tools/c85r7_r6w958n224rcop20_v1153.py). No GPU/model/network."""
from __future__ import annotations

import difflib
import json

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_r6w958inf3rco_v1150 as h208
from tools import c85r7_r6w958n224rcop20_v1153 as h218
from tools import c85r7_r6w958q_v1139 as parent
from tools import c85r7_rco_overlay_v1 as ov
from tools import test_c85r7_r6w958inf3rcop_v1152 as t217


@pytest.fixture(scope='module')
def built():
    return h218.package()


def changed_lines(a, b):
    return [l for l in difflib.unified_diff(a.decode().splitlines(), b.decode().splitlines(), lineterm='', n=0)
            if l[:1] in '+-' and not l.startswith(('+++', '---'))]


def test_scope(built):
    nb, meta, proof, old, new = built
    assert set(new) == set(old) and new[h218.h217.NEW_WORKER_NAME] == old[h218.h217.NEW_WORKER_NAME]
    ov_changes = changed_lines(old['c85r7_rco_overlay_v1.py'], new['c85r7_rco_overlay_v1.py'])
    assert len(ov_changes) == 10 and any('.80 * ranks(parent) + .20 * ranks(rco)' in l for l in ov_changes)
    _, _, p_old = h208.archive(old['parent-run.py'].decode())
    _, _, p_new = h208.archive(new['parent-run.py'].decode())
    diff = {m for m in p_old if p_old[m] != p_new[m]}
    assert diff == {'exp169_r224.py', 'exp169_release_c85r7codec.py'}
    assert changed_lines(p_old['exp169_r224.py'], p_new['exp169_r224.py']) == \
        ['-    ' + h218.R224_OLD, '+    ' + h218.R224_NEW]
    assert len(changed_lines(p_old['exp169_release_c85r7codec.py'], p_new['exp169_release_c85r7codec.py'])) == 2
    old_code, new_code = old['parent-run.py'].decode(), new['parent-run.py'].decode()
    assert len(changed_lines(old['parent-run.py'], new['parent-run.py'])) <= 6
    assert meta['id'] == h218.SLUG and meta['enable_internet'] is False


def test_parent_math_equals_noR224(built):
    """The modified embedded parent blend equals INF3 with R224 weight 0 on Gold58 inputs."""
    *_, new = built
    _, _, src = h208.archive(new['parent-run.py'].decode())
    ns = parent.math_functions(src)
    rng = np.random.default_rng(1)
    ids = ['1.2.%d' % i for i in range(40)]
    m = {k: rng.random((40, 12)) for k in ('R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q')}
    got = ov.table(ns['p'].csv_bytes(ids, ns['blend'](ids, *[m[k] for k in ('R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q')])), ids)[1]
    R = ov.ranks
    wt = np.asarray((.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5))
    P = wt * R(m['R']) + (1 - wt) * R(m['S_R7'])
    O = (17 * R(m['EXP143v2']) + 17 * R(m['EXP155']) + 12 * R(m['Q'])) / 46
    want = .85 * R(P) + .15 * R(O)
    assert np.abs(got - want).max() < 1e-6
    # R224 values no longer matter while R is valid
    m2 = dict(m, R224=rng.random((40, 12)))
    got2 = ov.table(ns['p'].csv_bytes(ids, ns['blend'](ids, *[m2[k] for k in ('R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q')])), ids)[1]
    assert np.array_equal(got, got2)


def test_simulated_overlay_run(tmp_path, built):
    *_, files = built
    mod = t217.load_overlay(tmp_path, files)
    calls = []
    runtime, output, data, runner, par, rco = t217.setup_run(tmp_path, mod, files, 'ok', calls)
    record = mod.run(runtime, output, runner=runner, assets=t217.ASSETS, data=data)
    assert record['status'] == 'COMPLETE_K3'
    _, p = ov.table(par, t217.IDS); _, r = ov.table(rco, t217.IDS)
    want = ov.ranks(.80 * ov.ranks(p) + .20 * ov.ranks(r)).astype(np.float32)
    assert np.array_equal(ov.table((output / 'submission.csv').read_bytes(), t217.IDS)[1].astype(np.float32), want)
